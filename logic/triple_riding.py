from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict


class TripleRidingDetector:
    def __init__(self, min_persons: int, window_seconds: float, fps: float) -> None:
        self.min_persons = min_persons
        self.frames_required = max(1, int(window_seconds * fps))
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.frames_required))

    def update(self, track_id: int, rider_count: int) -> bool:
        history = self.history[track_id]
        history.append(rider_count)
        if len(history) < self.frames_required:
            return False
        return min(history) >= self.min_persons

    def reset(self, track_id: int) -> None:
        if track_id in self.history:
            del self.history[track_id]
