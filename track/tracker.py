from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

from common.types import BBox, Detection
from common.config import TrackerParams
from third_party import activate_repo

activate_repo("speed")


@dataclass
class TrackerOutput:
    detections: List[Detection]


class TrackerWrapper:
    def __init__(self, params: TrackerParams, class_map: Dict[int, str]):
        self.params = params
        self.class_map = class_map
        tracker_type = params.type.lower()
        if tracker_type == "bytetrack":
            from trackers.bytetrack.byte_tracker import BYTETracker

            self.tracker = BYTETracker(
                track_thresh=params.track_thresh,
                match_thresh=params.match_thresh,
                track_buffer=params.track_buffer,
                frame_rate=params.frame_rate,
            )
        elif tracker_type == "ocsort":
            from trackers.ocsort.ocsort import OCSort

            self.tracker = OCSort(
                det_thresh=params.track_thresh,
                max_age=params.track_buffer,
                use_byte=True,
            )
        else:
            raise ValueError(f"Unsupported tracker type: {params.type}")

    def _detections_to_tensor(self, detections: Sequence[Detection]) -> torch.Tensor:
        if not detections:
            return torch.zeros((0, 6), dtype=torch.float32)
        rows = []
        for det in detections:
            x1, y1, x2, y2 = det.box.as_xyxy()
            rows.append([x1, y1, x2, y2, det.score, det.cls])
        return torch.tensor(rows, dtype=torch.float32)

    def update(self, detections: Sequence[Detection]) -> TrackerOutput:
        det_tensor = self._detections_to_tensor(detections)
        outputs = self.tracker.update(det_tensor, None)
        tracked: List[Detection] = []
        if outputs is None or len(outputs) == 0:
            return TrackerOutput(detections=tracked)
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().numpy()
        for row in outputs:
            if len(row) < 7:
                continue
            x1, y1, x2, y2, track_id, score, cls_id = row[:7]
            tracked.append(
                Detection(
                    box=BBox(float(x1), float(y1), float(x2), float(y2)),
                    score=float(score),
                    cls=int(cls_id),
                    label=self.class_map.get(int(cls_id), str(int(cls_id))),
                    tracker_id=int(track_id),
                )
            )
        return TrackerOutput(detections=tracked)
