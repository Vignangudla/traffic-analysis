from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from common.config import ConfigLoader
from common.types import FrameResult, TrackState
from detectors.yolo_detector import YoloDetector
from logic.association import assign_riders, rider_head_crops
from logic.plates import PlateRecognizer
from logic.speed import SpeedEstimator
from logic.triple_riding import TripleRidingDetector
from heads.helmet import HelmetClassifier, HelmetTemporalVoting, crop_head_region
from track.tracker import TrackerWrapper
from logic.plates_cache import PlateSnapshotStore


@dataclass
class ModelPaths:
    detector: Path
    helmet: Path
    plate: Path


class ViolationPipeline:
    def __init__(
        self,
        config_loader: ConfigLoader,
        model_paths: ModelPaths,
        camera_id: str = "default",
        device: Optional[str] = None,
        camera_fps: Optional[float] = None,
        enable_speed: bool = True,
        target_classes: Optional[List[str]] = None,
        snapshot_dir: Optional[Path] = None,
    ) -> None:
        self.cfg_loader = config_loader
        self.thresholds = config_loader.thresholds
        self.camera_id = camera_id
        self.device = device
        self.target_classes = target_classes or ["motorcycle"]
        self.target_class_set = {cls.lower() for cls in self.target_classes}
        self.calibration = self.cfg_loader.get_calibration(camera_id)
        self.detector = YoloDetector(
            weights=model_paths.detector,
            conf=self.thresholds.detector.conf,
            iou=self.thresholds.detector.iou,
            max_det=self.thresholds.detector.max_det,
            agnostic_nms=self.thresholds.detector.agnostic_nms,
            classes=None,
            device=device,
        )
        self.tracker = TrackerWrapper(self.thresholds.tracker, self.detector.class_map)
        self.helmet = HelmetClassifier(model_paths.helmet, device=device)
        self.helmet_voter = HelmetTemporalVoting(self.thresholds.rules.vote_window)
        try:
            self.plate_recognizer = PlateRecognizer(
                weights=model_paths.plate,
                params=self.thresholds.plate,
                device=device,
            )
        except FileNotFoundError as exc:
            logger.warning("%s; plate OCR disabled", exc)
            self.plate_recognizer = None
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.snapshot_store = (
            PlateSnapshotStore(self.snapshot_dir) if self.snapshot_dir and self.plate_recognizer else None
        )
        self.plate_cache: Dict[int, Tuple[str, float, int]] = {}
        self.plate_interval = 10
        fps = (
            camera_fps
            or (self.calibration.fps if self.calibration else None)
            or self.thresholds.tracker.frame_rate
        )
        self.triple = TripleRidingDetector(
            min_persons=self.thresholds.rules.triple_riding_min_persons,
            window_seconds=self.thresholds.rules.temporal_window_s,
            fps=fps,
        )
        self.speed_estimator = SpeedEstimator(self.calibration)
        self.enable_speed = enable_speed and self.calibration is not None
        self.frame_idx = 0
        self.video_mode = True

    def set_video_mode(self, is_video: bool) -> None:
        self.video_mode = is_video
        if not is_video:
            self.enable_speed = False

    def update_fps(self, fps: float) -> None:
        self.triple = TripleRidingDetector(
            min_persons=self.thresholds.rules.triple_riding_min_persons,
            window_seconds=self.thresholds.rules.temporal_window_s,
            fps=fps,
        )

    def _split_detections(self, detections):
        persons: List = []
        motorcycles: List = []
        for det in detections:
            name = det.label.lower()
            if name == "person":
                persons.append(det)
            elif name in self.target_class_set:
                motorcycles.append(det)
        return persons, motorcycles

    def _update_plate(self, frame: np.ndarray, track: TrackState, frame_idx: int) -> None:
        if not self.plate_recognizer:
            return
        cached = self.plate_cache.get(track.track_id)
        if cached and frame_idx - cached[2] < self.plate_interval:
            track.plate_text, track.plate_conf = cached[0], cached[1]
            return
        result = self.plate_recognizer.recognize(frame, track.motorcycle_box)
        if result.text:
            track.plate_text = result.text
            track.plate_conf = result.confidence
            self.plate_cache[track.track_id] = (result.text, result.confidence or 0.0, frame_idx)
        if self.snapshot_store and result.box and result.score is not None:
            self.snapshot_store.add(
                track.track_id,
                frame_idx,
                float(result.score),
                frame,
                result.box,
            )

    def process_frame(self, frame: np.ndarray, ts: float) -> FrameResult:
        classes = list({*self.target_classes, "person"})
        detections = self.detector.detect(frame, force_classes=classes)
        persons, motorcycles = self._split_detections(detections)
        track_outputs = self.tracker.update(motorcycles).detections
        assignments = assign_riders(persons, track_outputs, self.thresholds.association)

        tracks: List[TrackState] = []
        for track_det in track_outputs:
            if track_det.tracker_id is None:
                continue
            riders = assignments.get(track_det.tracker_id, [])
            for idx, rider in enumerate(riders):
                rider.rider_id = idx
                rider.head_box = rider_head_crops(rider, self.thresholds.association.head_crop_ratio_top)
                head_crop = crop_head_region(frame, rider.head_box)
                has_helmet, prob = self.helmet.predict(head_crop)
                voted_has, voted_prob = self.helmet_voter.update(track_det.tracker_id, idx, prob)
                rider.has_helmet = voted_has if voted_has is not None else has_helmet
                rider.helmet_prob = voted_prob if voted_prob is not None else prob
            if not self.video_mode:
                speed_kmh, speed_reason = (None, "single frame")
            elif not self.enable_speed:
                speed_kmh, speed_reason = (None, "no calib")
            elif self.speed_estimator is not None:
                speed_kmh, speed_reason = self.speed_estimator.update(track_det.tracker_id, track_det.box, ts)
            else:
                speed_kmh, speed_reason = (None, "disabled")
            triple = self.triple.update(track_det.tracker_id, len(riders)) if self.video_mode else False
            track_state = TrackState(
                track_id=track_det.tracker_id,
                motorcycle_box=track_det.box,
                ts=ts,
                riders=riders,
                speed_kmh=speed_kmh,
                speed_reason=speed_reason,
                triple_violation=triple,
            )
            self._update_plate(frame, track_state, self.frame_idx)
            tracks.append(track_state)
        self.frame_idx += 1
        return FrameResult(ts=ts, frame_idx=self.frame_idx, tracks=tracks)


__all__ = ["ViolationPipeline", "ModelPaths"]
