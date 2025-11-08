from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import inspect

import numpy as np
import torch
from ultralytics import YOLO

from common.types import BBox, Detection

# Torch 2.6+ safe loading guard
try:
    from torch.serialization import add_safe_globals
    from ultralytics.nn.tasks import DetectionModel

    add_safe_globals([DetectionModel])
except Exception:
    pass

# Force torch.load to allow checkpoint class definitions when loading trusted weights
if "weights_only" in inspect.signature(torch.load).parameters:  # type: ignore[attr-defined]
    _orig_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load  # type: ignore[assignment]


class YoloDetector:
    """Wrapper around the YOLOv8 model from the speed-estimation repo."""

    def __init__(
        self,
        weights: Path,
        conf: float = 0.25,
        iou: float = 0.5,
        max_det: int = 300,
        agnostic_nms: bool = False,
        classes: Sequence[int] | None = None,
        device: str | None = None,
    ) -> None:
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.classes = list(classes) if classes is not None else None
        if device:
            self.model.to(device)

    @property
    def class_map(self) -> Dict[int, str]:
        return self.model.model.names

    def _filter_classes(self, class_names: Iterable[str]) -> List[int]:
        name_to_id = {name: idx for idx, name in self.class_map.items()}
        result = []
        for name in class_names:
            if name not in name_to_id:
                raise KeyError(f"Class '{name}' is not present in YOLO weights")
            result.append(name_to_id[name])
        return result

    def detect(self, image: np.ndarray, force_classes: Sequence[str] | None = None) -> List[Detection]:
        if force_classes:
            class_ids = self._filter_classes(force_classes)
        elif self.classes is not None:
            class_ids = self.classes
        else:
            class_ids = None
        result = self.model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            agnostic_nms=self.agnostic_nms,
            classes=class_ids,
            max_det=self.max_det,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        if not hasattr(result, "boxes"):
            return detections
        boxes = result.boxes
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for box, score, cls_id in zip(xyxy, scores, cls):
            label = self.class_map.get(int(cls_id), str(cls_id))
            detections.append(
                Detection(
                    box=BBox(*box.tolist()),
                    score=float(score),
                    cls=int(cls_id),
                    label=label,
                )
            )
        return detections
