from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from torchvision import models, transforms
from ultralytics import YOLO

from common.types import BBox, RiderInfo
from third_party import activate_repo

activate_repo("helmet_vgg")

import inspect

try:  # torch>=2.6 safe-loading guard
    from torch.serialization import add_safe_globals
    from ultralytics.nn.tasks import DetectionModel

    add_safe_globals([DetectionModel])
except Exception:  # pragma: no cover
    pass

if "weights_only" in inspect.signature(torch.load).parameters:  # pragma: no cover
    _orig_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load  # type: ignore[assignment]


class HelmetClassifier:
    def __init__(self, weights_path: Path, device: Optional[str] = None) -> None:
        self.weights_path = Path(weights_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.mode = "vgg"
        self.yolo_model: Optional[YOLO] = None
        self.helmet_cls_ids: Tuple[int, ...] = tuple()
        self.no_helmet_cls_ids: Tuple[int, ...] = tuple()

        self.model = self._build_model().to(self.device)
        self.enabled = False
        if self.weights_path.exists():
            try:
                self.enabled = self._load_weights()
                self.model.eval()
            except Exception as exc:  # fallback to YOLO detector
                logger.warning(
                    "Helmet classifier weights failed to load (%s); switching to YOLO detector mode", exc
                )
                self._init_yolo()
        else:
            logger.warning("Helmet weights not found at %s; classifier will run with ImageNet head", self.weights_path)
            self.model.eval()

        if not self.enabled and self.mode != "yolo":
            logger.warning("Helmet classifier running with ImageNet weights; helmet precision may be low.")

    def _build_model(self) -> torch.nn.Module:
        try:
            base = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        except AttributeError:  # older torchvision
            base = models.vgg16(pretrained=True)
        base.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=2)
        return base

    def _load_weights(self) -> bool:
        if not self.weights_path.exists():
            return False
        state = torch.load(self.weights_path, map_location=self.device)
        if "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        logger.info("Loaded helmet classifier weights from %s", self.weights_path)
        return True

    def _init_yolo(self) -> None:
        try:
            self.yolo_model = YOLO(str(self.weights_path))
        except Exception as exc:
            logger.error("Failed to load YOLO helmet detector: %s", exc)
            self.mode = "disabled"
            self.yolo_model = None
            return
        names = self.yolo_model.model.names
        if isinstance(names, dict):
            items = names.items()
        else:
            items = enumerate(names)
        helmet_ids = []
        no_helmet_ids = []
        for idx, name in items:
            name_str = str(name).lower()
            if "helmet" in name_str and "no" not in name_str:
                helmet_ids.append(int(idx))
            elif "helmet" in name_str and "no" in name_str:
                no_helmet_ids.append(int(idx))
        self.helmet_cls_ids = tuple(helmet_ids)
        self.no_helmet_cls_ids = tuple(no_helmet_ids)
        if not self.helmet_cls_ids and not self.no_helmet_cls_ids:
            logger.warning(
                "YOLO helmet detector loaded but class mapping missing; detections may be ignored. Classes=%s",
                names,
            )
        self.mode = "yolo"

    def predict(self, crop: np.ndarray) -> Tuple[Optional[bool], Optional[float]]:
        if crop.size == 0:
            return None, None

        if self.mode == "yolo" and self.yolo_model is not None:
            return self._predict_yolo(crop)

        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        helmet_prob = float(probs[1])  # assume index 1 == helmet
        has_helmet = helmet_prob >= 0.5
        return has_helmet, helmet_prob

    def _predict_yolo(self, crop: np.ndarray) -> Tuple[Optional[bool], Optional[float]]:
        if crop.size == 0 or self.yolo_model is None:
            return None, None
        results = self.yolo_model.predict(crop, imgsz=160, conf=0.25, verbose=False)
        if not results:
            return None, None
        boxes = results[0].boxes
        if boxes is None or boxes.cls is None or boxes.conf is None:
            return None, None
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        helmet_conf = max(
            (float(conf) for conf, cls_id in zip(confs, cls_ids) if cls_id in self.helmet_cls_ids),
            default=0.0,
        )
        no_helmet_conf = max(
            (float(conf) for conf, cls_id in zip(confs, cls_ids) if cls_id in self.no_helmet_cls_ids),
            default=0.0,
        )
        if helmet_conf == 0.0 and no_helmet_conf == 0.0:
            return None, None
        if helmet_conf >= no_helmet_conf:
            return True, helmet_conf
        return False, no_helmet_conf


class HelmetTemporalVoting:
    def __init__(self, window: int) -> None:
        self.window = window
        self.cache: Dict[Tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, track_id: int, rider_idx: int, prob: Optional[float]) -> Tuple[Optional[bool], Optional[float]]:
        if prob is None:
            return None, None
        key = (track_id, rider_idx)
        self.cache[key].append(prob)
        avg = sum(self.cache[key]) / len(self.cache[key])
        return avg >= 0.5, avg

    def reset_track(self, track_id: int) -> None:
        for key in list(self.cache.keys()):
            if key[0] == track_id:
                del self.cache[key]


def crop_head_region(frame: np.ndarray, box: BBox) -> np.ndarray:
    h, w = frame.shape[:2]
    b = box.clip(w, h)
    x1, y1, x2, y2 = map(int, b.as_xyxy())
    return frame[y1:y2, x1:x2]


__all__ = ["HelmetClassifier", "HelmetTemporalVoting", "crop_head_region"]
