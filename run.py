from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from loguru import logger

from common.config import ConfigLoader
from logic.annotate import annotate_frame
from logic.events import JsonlWriter
from logic.plate_refine import refine_plate_events
from pipeline.engine import ModelPaths, ViolationPipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-wheeler violation pipeline")
    parser.add_argument("--source", required=True, help="Image, video file, or RTSP URL")
    parser.add_argument("--calib", default="configs/calib.yaml", help="Calibration YAML path")
    parser.add_argument("--thresholds", default="configs/thresholds.yaml", help="Thresholds YAML path")
    parser.add_argument(
        "--detector-weights",
        default="third_party/yolo8-tracking-counting-speed_estimation/weights/motorcycle.pt",
        help="YOLO weights for person/motorcycle detection",
    )
    parser.add_argument(
        "--helmet-weights",
        default="third_party/Helmet-Violation-Detection-Using-YOLO-and-VGG16/weights/helmet_vgg16.pth",
        help="Helmet classifier weights",
    )
    parser.add_argument(
        "--plate-weights",
        default="third_party/Automatic_Number_Plate_Recognition_YOLO_OCR/model/best.pt",
        help="License plate detector weights",
    )
    parser.add_argument("--camera-id", default="default", help="Camera/calibration identifier")
    parser.add_argument("--device", default=None, help="Torch device override, e.g., cuda:0")
    parser.add_argument("--output-dir", default="outputs", help="Directory for artifacts")
    parser.add_argument("--jsonl-out", default=None, help="Path to JSONL events file")
    parser.add_argument("--annotated-out", default=None, help="Annotated media path")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to process")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["motorcycle"],
        help="Classes to track (default: motorcycle)",
    )
    parser.add_argument(
        "--plate-snapshots",
        default="outputs/plates",
        help="Directory to store plate snapshots for offline OCR refinement",
    )
    return parser


def detect_source_type(path: str) -> str:
    if path.startswith("rtsp://") or path.startswith("http://") or path.startswith("https://"):
        return "stream"
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTS:
        return "image"
    if suffix in VIDEO_EXTS:
        return "video"
    return "stream"


def prepare_outputs(args, source_type: str) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(args.jsonl_out) if args.jsonl_out else output_dir / "events.jsonl"
    annotated_default = "annotated.jpg" if source_type == "image" else "annotated.mp4"
    annotated_path = Path(args.annotated_out) if args.annotated_out else output_dir / annotated_default
    return jsonl_path, annotated_path


def load_frame(source: str):
    frame = cv2.imread(source)
    if frame is None:
        raise RuntimeError(f"Failed to load image: {source}")
    return frame


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    source_type = detect_source_type(args.source)
    jsonl_path, annotated_path = prepare_outputs(args, source_type)
    snapshot_dir = Path(args.plate_snapshots) if source_type != "image" else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    cfg_loader = ConfigLoader(Path(args.thresholds), Path(args.calib))
    model_paths = ModelPaths(
        detector=Path(args.detector_weights),
        helmet=Path(args.helmet_weights),
        plate=Path(args.plate_weights),
    )

    pipeline = ViolationPipeline(
        config_loader=cfg_loader,
        model_paths=model_paths,
        camera_id=args.camera_id,
        device=args.device,
        enable_speed=source_type != "image",
        target_classes=args.classes,
        snapshot_dir=snapshot_dir,
    )
    pipeline.set_video_mode(source_type != "image")

    logger.info("Processing %s as %s", args.source, source_type)

    if source_type == "image":
        frame = load_frame(args.source)
        result = pipeline.process_frame(frame, ts=0.0)
        annotate_frame(frame, result.tracks)
        cv2.imwrite(str(annotated_path), frame)
        with JsonlWriter(jsonl_path) as writer:
            writer.write(result)
        logger.success("Results saved: %s, %s", annotated_path, jsonl_path)
        return

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg_loader.thresholds.tracker.frame_rate
    if fps <= 1e-3:
        fps = cfg_loader.thresholds.tracker.frame_rate
    pipeline.update_fps(fps)
    writer = None
    frame_idx = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    with JsonlWriter(jsonl_path) as jsonl_writer:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))
            ts = frame_idx / fps
            result = pipeline.process_frame(frame, ts)
            annotate_frame(frame, result.tracks)
            writer.write(frame)
            jsonl_writer.write(result)
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    if writer is not None:
        writer.release()
    cap.release()
    if snapshot_dir and getattr(pipeline, "snapshot_store", None) and pipeline.plate_recognizer:
        refine_plate_events(
            Path(jsonl_path),
            pipeline.snapshot_store.root,
            pipeline.plate_recognizer,
        )
    logger.success("Processed %d frames -> %s", frame_idx, annotated_path)
    logger.success("JSONL saved to %s", jsonl_path)


if __name__ == "__main__":
    main()
