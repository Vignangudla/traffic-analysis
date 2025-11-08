from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from common.types import FrameResult, TrackState


def track_to_schema(track: TrackState) -> dict:
    riders_missing = [idx for idx, rider in enumerate(track.riders) if rider.has_helmet is False]
    return {
        "ts": track.ts,
        "track_id": track.track_id,
        "motorcycle_box": list(track.motorcycle_box.as_xywh()),
        "riders_count": len(track.riders),
        "helmet_missing_ids": riders_missing,
        "plate_text": track.plate_text,
        "plate_conf": track.plate_conf,
        "speed_kmh": track.speed_kmh,
        "speed_reason": track.speed_reason,
    }


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.handle = self.path.open("w", encoding="utf-8")

    def write(self, frame_result: FrameResult) -> None:
        for track in frame_result.tracks:
            payload = track_to_schema(track)
            payload["ts"] = frame_result.ts
            self.handle.write(json.dumps(payload) + "\n")

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["JsonlWriter", "track_to_schema"]
