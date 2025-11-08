from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from common.config import AssociationParams
from common.types import BBox, Detection, RiderInfo


@dataclass
class RiderAssignment:
    track_id: int
    riders: List[RiderInfo]


def _bbox_iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a.as_xyxy()
    bx1, by1, bx2, by2 = box_b.as_xyxy()
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = box_a.area()
    area_b = box_b.area()
    return inter_area / max(1e-6, area_a + area_b - inter_area)


def _center_distance(box_a: BBox, box_b: BBox) -> float:
    ax, ay = box_a.center()
    bx, by = box_b.center()
    return float(np.hypot(ax - bx, ay - by))


def _head_crop(box: BBox, ratio_top: float) -> BBox:
    height = max(1.0, box.y2 - box.y1)
    return BBox(box.x1, box.y1, box.x2, box.y1 + height * ratio_top)


def assign_riders(
    persons: Sequence[Detection],
    motorcycle_tracks: Sequence[Detection],
    params: AssociationParams,
) -> Dict[int, List[RiderInfo]]:
    assignments: Dict[int, List[RiderInfo]] = {
        det.tracker_id: [] for det in motorcycle_tracks if det.tracker_id is not None
    }
    if not persons or not motorcycle_tracks:
        return assignments

    for person in persons:
        best_track = None
        best_score = -1.0
        for moto in motorcycle_tracks:
            if moto.tracker_id is None:
                continue
            iou = _bbox_iou(person.box, moto.box)
            dist = _center_distance(person.box, moto.box)
            if iou < params.iou_person_bike and dist > params.center_dist_px:
                continue
            score = iou - 0.001 * dist
            if score > best_score:
                best_score = score
                best_track = moto
        if best_track is None:
            continue
        rider = RiderInfo(
            person_box=person.box,
            helmet_prob=None,
            has_helmet=None,
            rider_id=None,
        )
        assignments.setdefault(best_track.tracker_id, []).append(rider)

    # sort riders top->bottom for stable IDs
    for riders in assignments.values():
        riders.sort(key=lambda r: r.person_box.y1)

    return assignments


def rider_head_crops(rider: RiderInfo, ratio_top: float) -> BBox:
    return _head_crop(rider.person_box, ratio_top)


__all__ = ["assign_riders", "rider_head_crops"]
