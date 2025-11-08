from __future__ import annotations

from pathlib import Path
from typing import Iterable

from loguru import logger
from ultralytics import YOLO


def export_yolo(weights: Path, output_dir: Path, formats: Iterable[str] | None = None, int8: bool = True) -> None:
    weights = Path(weights)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = list(formats or ["onnx", "openvino", "engine"])
    model = YOLO(str(weights))
    for fmt in formats:
        logger.info("Exporting %s -> %s", weights.name, fmt)
        quant = int8 if fmt in {"openvino", "engine"} else False
        model.export(format=fmt, imgsz=640, half=not quant, int8=quant, device="cpu", project=str(output_dir), name=fmt)


def export_all():
    export_yolo(
        weights=Path("third_party/yolo8-tracking-counting-speed_estimation/weights/motorcycle.pt"),
        output_dir=Path("deploy/artifacts/detector"),
    )
    export_yolo(
        weights=Path("third_party/Automatic_Number_Plate_Recognition_YOLO_OCR/model/best.pt"),
        output_dir=Path("deploy/artifacts/plates"),
    )


if __name__ == "__main__":
    export_all()
