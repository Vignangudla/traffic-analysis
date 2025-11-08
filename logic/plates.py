from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger
from paddleocr import PaddleOCR
from ultralytics import YOLO
import torch
import inspect

try:
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None

from common.config import PlateParams
from common.types import BBox

try:
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


@dataclass
class PlateResult:
    text: Optional[str]
    confidence: Optional[float]
    box: Optional[BBox]
    score: Optional[float] = None


class PlateRecognizer:
    def __init__(
        self,
        weights: Path,
        params: PlateParams,
        device: Optional[str] = None,
        ocr_lang: str = "en",
    ) -> None:
        weights = Path(weights)
        self.detector = None
        if not weights.exists():
            logger.warning("Plate detector weights not found at %s; LP OCR disabled", weights)
        else:
            try:
                model = YOLO(str(weights))
                if device:
                    model.to(device)
                self.detector = model
            except Exception as exc:
                logger.warning("Plate detector load failed: %s; LP OCR disabled", exc)
        self.params = params
        self.expand_ratio = 0.4
        plate_ids: List[int] = []
        if self.detector is not None:
            names = self.detector.model.names
            if isinstance(names, dict):
                for idx, name in names.items():
                    if "plate" in str(name).lower():
                        plate_ids.append(int(idx))
                all_ids = list(names.keys())
            else:
                plate_ids = [idx for idx, name in enumerate(names) if "plate" in str(name).lower()]
                all_ids = list(range(len(names)))
        else:
            all_ids = []
        self.plate_classes = plate_ids or all_ids
        self.ocr_backend = params.ocr.lower()
        self.paddle_ocr = None
        self.easy_reader = None
        if self.ocr_backend in {"paddleocr", "auto"}:
            try:
                self.paddle_ocr = PaddleOCR(lang=ocr_lang, show_log=False)
            except (TypeError, ValueError):
                self.paddle_ocr = PaddleOCR(lang=ocr_lang)
        if (self.ocr_backend in {"easyocr", "auto"} or self.paddle_ocr is None) and easyocr:
            self.easy_reader = easyocr.Reader([ocr_lang])
        if self.paddle_ocr is None and self.easy_reader is None:
            raise RuntimeError("No OCR backend available (install PaddleOCR or EasyOCR)")

    def _enlarge_box(self, box: BBox, frame_shape: Tuple[int, int, int]) -> BBox:
        h, w = frame_shape[:2]
        width = box.x2 - box.x1
        height = box.y2 - box.y1
        dx = width * self.expand_ratio
        dy = height * self.expand_ratio
        return BBox(
            max(0, box.x1 - dx),
            max(0, box.y1 - dy),
            min(w - 1, box.x2 + dx),
            min(h - 1, box.y2 + dy),
        )

    def _run_detector(self, image: np.ndarray) -> Tuple[Optional[BBox], Optional[float]]:
        if self.detector is None:
            return None, None
        results = self.detector.predict(
            image,
            conf=0.3,
            iou=0.5,
            agnostic_nms=False,
            max_det=5,
            classes=self.plate_classes,
            verbose=False,
        )
        if not results:
            return None, None
        boxes = results[0].boxes
        if boxes is None or boxes.cls.numel() == 0:
            return None, None
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        mask = np.isin(cls_ids, self.plate_classes) if self.plate_classes else np.ones_like(cls_ids, dtype=bool)
        if not mask.any():
            return None, None
        confs = boxes.conf.cpu().numpy()[mask]
        xyxy = boxes.xyxy.cpu().numpy()[mask]
        idx = int(np.argmax(confs))
        xyxy = xyxy[idx].tolist()
        return BBox(*xyxy), float(confs[idx])

    def _run_detector_full(self, image: np.ndarray, track_box: BBox) -> Tuple[Optional[BBox], Optional[float]]:
        if self.detector is None:
            return None, None
        results = self.detector.predict(
            image,
            conf=0.25,
            iou=0.5,
            agnostic_nms=False,
            max_det=10,
            classes=self.plate_classes,
            verbose=False,
        )
        if not results:
            return None, None
        boxes = results[0].boxes
        if boxes is None or boxes.cls is None or boxes.cls.numel() == 0:
            return None, None
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        mask = np.isin(cls_ids, self.plate_classes) if self.plate_classes else np.ones_like(cls_ids, dtype=bool)
        if not mask.any():
            return None, None
        confs = boxes.conf.cpu().numpy()[mask]
        xyxy = boxes.xyxy.cpu().numpy()[mask]
        best = None
        best_score = -1.0
        best_conf = None
        for box, conf in zip(xyxy, confs):
            candidate = BBox(*box.tolist())
            score = self._bbox_iou(candidate, track_box) * 0.7 + float(conf) * 0.3
            if score > best_score:
                best_score = score
                best = candidate
                best_conf = float(conf)
        return best, best_conf

    @staticmethod
    def _bbox_iou(a: BBox, b: BBox) -> float:
        ax1, ay1, ax2, ay2 = a.as_xyxy()
        bx1, by1, bx2, by2 = b.as_xyxy()
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = a.area()
        area_b = b.area()
        return inter_area / max(1e-6, area_a + area_b - inter_area)

    def _run_ocr(self, crop: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        if crop.size == 0:
            return None, None
        candidates: List[Tuple[str, float]] = []
        for variant in self._generate_variants(crop):
            txt, conf = self._ocr_paddle(variant)
            if txt:
                candidates.append((txt, conf or 0.0))
            txt2, conf2 = self._ocr_easy(variant)
            if txt2:
                candidates.append((txt2, conf2 or 0.0))
        if not candidates:
            return None, None
        best = max(candidates, key=lambda item: item[1])
        cleaned = self._clean_plate(best[0])
        if not cleaned:
            return None, None
        return cleaned, best[1]

    def _ocr_paddle(self, crop: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        if self.paddle_ocr is None:
            return None, None
        try:
            result = self.paddle_ocr.ocr(crop, cls=False)
        except Exception:
            return None, None
        if not result:
            return None, None
        lines = result[0]
        if not lines:
            return None, None
        best = max(lines, key=lambda x: x[1][1])
        text, conf = best[1]
        return text, float(conf)

    def _ocr_easy(self, crop: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        if self.easy_reader is None:
            return None, None
        try:
            detections = self.easy_reader.readtext(crop)
        except Exception:
            return None, None
        if not detections:
            return None, None
        best = max(detections, key=lambda x: x[2])
        return best[1], float(best[2])

    def _generate_variants(self, crop: np.ndarray) -> List[np.ndarray]:
        variants = [crop]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        variants.append(cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR))
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))
        return variants

    @staticmethod
    def _clean_plate(text: str) -> Optional[str]:
        allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        cleaned = "".join(ch for ch in text.upper() if ch in allowed)
        return cleaned if cleaned else None

    def recognize(self, frame: np.ndarray, track_box: BBox) -> PlateResult:
        roi_box = self._enlarge_box(track_box, frame.shape)
        x1, y1, x2, y2 = map(int, roi_box.as_xyxy())
        roi = frame[y1:y2, x1:x2]
        if roi.size < self.params.min_crop_w * self.params.min_crop_h:
            return PlateResult(None, None, None)
        plate_box, score = self._run_detector(roi)
        if plate_box:
            px1, py1, px2, py2 = plate_box.as_xyxy()
            px1 += x1
            py1 += y1
            px2 += x2
            py2 += y2
        else:
            global_box, score = self._run_detector_full(frame, track_box)
            if not global_box:
                return PlateResult(None, None, None)
            px1, py1, px2, py2 = global_box.as_xyxy()
        crop = frame[int(py1):int(py2), int(px1):int(px2)]
        text, conf = self._run_ocr(crop)
        return PlateResult(text, conf, BBox(px1, py1, px2, py2), score)

    def ocr_crop(self, crop: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        return self._run_ocr(crop)


__all__ = ["PlateRecognizer", "PlateResult"]
