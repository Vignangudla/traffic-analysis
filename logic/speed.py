from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from common.config import CameraCalibration
from common.types import BBox


class SpeedEstimator:
    def __init__(self, calibration: Optional[CameraCalibration], window_seconds: float = 1.0) -> None:
        self.calibration = calibration
        self.window_seconds = window_seconds
        self.history: Dict[int, Deque[Tuple[float, np.ndarray]]] = defaultdict(lambda: deque())
        self.homography = None
        if calibration and calibration.homography is not None:
            self.homography = calibration.homography.astype(np.float32)

    def reset(self, track_id: int) -> None:
        if track_id in self.history:
            del self.history[track_id]

    def _project_point(self, box: BBox) -> Optional[np.ndarray]:
        cx, _ = box.center()
        bottom_center = np.array([[cx, box.y2]], dtype=np.float32)
        if self.homography is not None:
            pts = cv2.perspectiveTransform(bottom_center.reshape((-1, 1, 2)), self.homography)
            return pts.reshape(-1, 2)[0]
        if self.calibration and self.calibration.meters_per_pixel:
            return bottom_center[0] * float(self.calibration.meters_per_pixel)
        return None

    def update(self, track_id: int, box: BBox, ts: float) -> Tuple[Optional[float], str]:
        if not self.calibration:
            return None, "no calib"
        point = self._project_point(box)
        if point is None:
            return None, "no calib"
        history = self.history[track_id]
        history.append((ts, point))
        while history and ts - history[0][0] > self.window_seconds:
            history.popleft()
        if len(history) < 2:
            return None, "single frame"
        t0, p0 = history[0]
        t1, p1 = history[-1]
        dt = t1 - t0
        if dt <= 1e-3:
            return None, "single frame"
        dist = float(np.linalg.norm(p1 - p0))
        speed_mps = dist / dt
        speed_kmh = speed_mps * 3.6
        return speed_kmh, "ok"


__all__ = ["SpeedEstimator"]
