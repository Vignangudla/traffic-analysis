from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml


@dataclass
class DetectorParams:
    conf: float
    iou: float
    max_det: int
    agnostic_nms: bool


@dataclass
class TrackerParams:
    type: str
    frame_rate: float
    track_thresh: float
    match_thresh: float
    track_buffer: int


@dataclass
class AssociationParams:
    iou_person_bike: float
    center_dist_px: float
    head_crop_ratio_top: float


@dataclass
class RuleParams:
    triple_riding_min_persons: int
    temporal_window_s: float
    vote_window: int


@dataclass
class PlateParams:
    min_crop_w: int
    min_crop_h: int
    ocr: str


@dataclass
class ThresholdConfig:
    detector: DetectorParams
    tracker: TrackerParams
    association: AssociationParams
    rules: RuleParams
    plate: PlateParams


@dataclass
class CameraCalibration:
    camera_id: str
    fps: float
    homography: Optional[np.ndarray]
    meters_per_pixel: Optional[float]
    description: Optional[str] = None

    def transform_point(self, pt):
        if self.homography is None:
            return None
        vec = np.array([pt[0], pt[1], 1.0])
        mapped = self.homography @ vec
        if mapped[2] == 0:
            return None
        return mapped[:2] / mapped[2]


class ConfigLoader:
    def __init__(self, thresholds_path: Path, calib_path: Optional[Path] = None):
        self.thresholds_path = Path(thresholds_path)
        self.calib_path = Path(calib_path) if calib_path else None
        self.thresholds = self._load_thresholds()
        self.calibrations = self._load_calibrations()

    def _load_yaml(self, path: Path) -> Dict:
        if not path.exists():
            raise FileNotFoundError(f"Missing config file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _load_thresholds(self) -> ThresholdConfig:
        raw = self._load_yaml(self.thresholds_path)
        det = raw.get("detector", {})
        tracker = raw.get("tracker", {})
        assoc = raw.get("association", {})
        rules = raw.get("rules", {})
        plate = raw.get("plate", {})
        return ThresholdConfig(
            detector=DetectorParams(
                conf=float(det.get("conf", 0.25)),
                iou=float(det.get("iou", 0.5)),
                max_det=int(det.get("max_det", 300)),
                agnostic_nms=bool(det.get("agnostic_nms", False)),
            ),
            tracker=TrackerParams(
                type=str(tracker.get("type", "bytetrack")),
                frame_rate=float(tracker.get("frame_rate", 30.0)),
                track_thresh=float(tracker.get("track_thresh", 0.5)),
                match_thresh=float(tracker.get("match_thresh", 0.8)),
                track_buffer=int(tracker.get("track_buffer", 30)),
            ),
            association=AssociationParams(
                iou_person_bike=float(assoc.get("iou_person_bike", 0.1)),
                center_dist_px=float(assoc.get("center_dist_px", 150.0)),
                head_crop_ratio_top=float(assoc.get("head_crop_ratio_top", 0.4)),
            ),
            rules=RuleParams(
                triple_riding_min_persons=int(rules.get("triple_riding_min_persons", 3)),
                temporal_window_s=float(rules.get("temporal_window_s", 0.5)),
                vote_window=int(rules.get("vote_window", 5)),
            ),
            plate=PlateParams(
                min_crop_w=int(plate.get("min_crop_w", 40)),
                min_crop_h=int(plate.get("min_crop_h", 12)),
                ocr=str(plate.get("ocr", "paddleocr")),
            ),
        )

    def _load_calibrations(self) -> Dict[str, CameraCalibration]:
        if not self.calib_path or not self.calib_path.exists():
            return {}
        raw = self._load_yaml(self.calib_path)
        cams = {}
        for cam_id, payload in (raw.get("cameras", {}) or {}).items():
            homography = payload.get("homography")
            homography_mat = None
            if homography:
                homography_mat = np.asarray(homography, dtype=float)
                if homography_mat.shape != (3, 3):
                    raise ValueError(f"Camera {cam_id} homography must be 3x3")
            cams[cam_id] = CameraCalibration(
                camera_id=cam_id,
                fps=float(payload.get("fps", 30.0)),
                homography=homography_mat,
                meters_per_pixel=payload.get("meters_per_pixel"),
                description=payload.get("description"),
            )
        return cams

    def get_calibration(self, camera_id: str) -> Optional[CameraCalibration]:
        if not self.calibrations:
            return None
        if camera_id in self.calibrations:
            return self.calibrations[camera_id]
        return self.calibrations.get("default")


__all__ = [
    "ConfigLoader",
    "ThresholdConfig",
    "CameraCalibration",
]
