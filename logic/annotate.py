from __future__ import annotations

from typing import List

import cv2

from common.types import TrackState


COLORS = {
    "ok": (0, 255, 0),
    "violation": (0, 0, 255),
    "warning": (0, 165, 255),
}


def _draw_label(frame, text: str, x: int, y: int, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x, y - text_size[1] - 4), (x + text_size[0] + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def annotate_frame(frame, tracks: List[TrackState]) -> None:
    for track in tracks:
        color = COLORS["violation"] if track.triple_violation else COLORS["ok"]
        x1, y1, x2, y2 = map(int, track.motorcycle_box.as_xyxy())
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_lines = [f"ID {track.track_id}"]
        label_lines.append(f"riders {len(track.riders)}")
        if track.plate_text:
            label_lines.append(track.plate_text)
        for idx, line in enumerate(label_lines):
            _draw_label(frame, line, x1, y1 - idx * 18, color)

        for rider_idx, rider in enumerate(track.riders):
            rx1, ry1, rx2, ry2 = map(int, rider.person_box.as_xyxy())
            rider_color = COLORS["ok"] if rider.has_helmet else COLORS["violation"]
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), rider_color, 1)
            status = "helmet" if rider.has_helmet else "no helmet"
            _draw_label(frame, f"R{rider_idx}:{status}", rx1, ry1, rider_color)
            if rider.head_box:
                hx1, hy1, hx2, hy2 = map(int, rider.head_box.as_xyxy())
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 1)


__all__ = ["annotate_frame"]
