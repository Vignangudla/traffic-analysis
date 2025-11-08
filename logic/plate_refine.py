from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple

import cv2

from logic.plates import PlateRecognizer


def _score_from_name(path: Path) -> float:
    match = re.search(r"_s(\d+)", path.stem)
    if not match:
        return 0.0
    return int(match.group(1)) / 1000.0


def _best_snapshot(track_dir: Path) -> Path | None:
    candidates = list(track_dir.glob("*.jpg"))
    if not candidates:
        return None
    return max(candidates, key=_score_from_name)


def refine_plate_events(
    jsonl_path: Path,
    snapshots_root: Path,
    recognizer: PlateRecognizer,
    min_existing_conf: float = 0.5,
) -> None:
    if not snapshots_root or not snapshots_root.exists():
        return
    track_texts: Dict[int, Tuple[str, float]] = {}
    for track_dir in snapshots_root.glob("track_*"):
        try:
            track_id = int(track_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        best_file = _best_snapshot(track_dir)
        if not best_file:
            continue
        crop = cv2.imread(str(best_file))
        if crop is None:
            continue
        text, conf = recognizer.ocr_crop(crop)
        if text and conf:
            track_texts[track_id] = (text, conf)
    if not track_texts:
        return
    lines = []
    with jsonl_path.open("r", encoding="utf-8") as reader:
        for line in reader:
            data = json.loads(line)
            lines.append(data)
    changed = False
    for entry in lines:
        track_id = entry.get("track_id")
        if track_id in track_texts:
            text, conf = track_texts[track_id]
            current_conf = entry.get("plate_conf") or 0.0
            if current_conf < conf or current_conf < min_existing_conf:
                entry["plate_text"] = text
                entry["plate_conf"] = conf
                changed = True
    if not changed:
        return
    with jsonl_path.open("w", encoding="utf-8") as writer:
        for entry in lines:
            writer.write(json.dumps(entry) + "\n")


__all__ = ["refine_plate_events"]
