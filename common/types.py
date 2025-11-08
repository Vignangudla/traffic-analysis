from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coords."""

    x1: float
    y1: float
    x2: float
    y2: float

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def as_xywh(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1

    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def clip(self, w: int, h: int) -> "BBox":
        return BBox(
            x1=max(0.0, min(self.x1, w - 1)),
            y1=max(0.0, min(self.y1, h - 1)),
            x2=max(0.0, min(self.x2, w - 1)),
            y2=max(0.0, min(self.y2, h - 1)),
        )


@dataclass
class Detection:
    box: BBox
    score: float
    cls: int
    label: str
    tracker_id: Optional[int] = None


@dataclass
class RiderInfo:
    person_box: BBox
    helmet_prob: Optional[float]
    has_helmet: Optional[bool]
    rider_id: Optional[int] = None
    head_box: Optional[BBox] = None


@dataclass
class TrackState:
    track_id: int
    motorcycle_box: BBox
    ts: float
    riders: List[RiderInfo] = field(default_factory=list)
    speed_kmh: Optional[float] = None
    speed_reason: str = "ok"
    plate_text: Optional[str] = None
    plate_conf: Optional[float] = None
    triple_violation: bool = False


@dataclass
class FrameResult:
    ts: float
    frame_idx: int
    tracks: List[TrackState]
