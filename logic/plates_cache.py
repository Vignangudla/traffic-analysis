from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from common.types import BBox


@dataclass
class CropEntry:
    frame_idx: int
    score: float
    crop_path: Path
    box: BBox


class PlateSnapshotStore:
    """Keeps top-N plate crops per track for downstream OCR."""

    def __init__(self, root: Path, per_track: int = 3) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.per_track = per_track
        self.entries: Dict[int, List[CropEntry]] = {}

    def add(self, track_id: int, frame_idx: int, score: float, frame: np.ndarray, box: BBox) -> None:
        if frame.size == 0:
            return
        crops_dir = self.root / f"track_{track_id:04d}"
        crops_dir.mkdir(parents=True, exist_ok=True)
        x1, y1, x2, y2 = map(int, box.as_xyxy())
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        filename = crops_dir / f"f{frame_idx:05d}_s{int(score * 1000):04d}.jpg"
        cv2.imwrite(str(filename), crop)
        entry = CropEntry(frame_idx=frame_idx, score=score, crop_path=filename, box=box)
        bucket = self.entries.setdefault(track_id, [])
        bucket.append(entry)
        bucket.sort(key=lambda e: e.score, reverse=True)
        while len(bucket) > self.per_track:
            old = bucket.pop()
            try:
                old.crop_path.unlink(missing_ok=True)
            except Exception:
                pass

    def best_crop(self, track_id: int) -> Optional[CropEntry]:
        bucket = self.entries.get(track_id)
        if not bucket:
            return None
        return bucket[0]


__all__ = ["PlateSnapshotStore"]
