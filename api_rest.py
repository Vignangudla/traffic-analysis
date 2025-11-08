from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from common.config import ConfigLoader
from logic.annotate import annotate_frame
from logic.events import track_to_schema
from pipeline.engine import ModelPaths, ViolationPipeline
from run import detect_source_type

app = FastAPI(title="Two-Wheeler Violation API", version="0.1.0")


class PipelineManager:
    def __init__(self) -> None:
        self.cfg_loader = ConfigLoader(Path("configs/thresholds.yaml"), Path("configs/calib.yaml"))
        self.model_paths = ModelPaths(
            detector=Path("third_party/yolo8-tracking-counting-speed_estimation/weights/motorcycle.pt"),
            helmet=Path("third_party/Helmet-Violation-Detection-Using-YOLO-and-VGG16/weights/helmet_vgg16.pth"),
            plate=Path("third_party/Automatic_Number_Plate_Recognition_YOLO_OCR/model/best.pt"),
        )

    def build_pipeline(self, camera_id: str, classes: Tuple[str, ...]) -> ViolationPipeline:
        return ViolationPipeline(
            config_loader=self.cfg_loader,
            model_paths=self.model_paths,
            camera_id=camera_id,
            enable_speed=True,
            target_classes=list(classes),
        )


pipeline_manager = PipelineManager()


def frame_to_base64(frame) -> str:
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return ""
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def frame_result_to_dict(result) -> Dict:
    return {
        "frame_idx": result.frame_idx,
        "ts": result.ts,
        "tracks": [track_to_schema(track) for track in result.tracks],
    }


def process_image_path(pipeline: ViolationPipeline, path: Path):
    frame = cv2.imread(str(path))
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not read uploaded image")
    pipeline.set_video_mode(False)
    result = pipeline.process_frame(frame, ts=0.0)
    annotate_frame(frame, result.tracks)
    return [frame_result_to_dict(result)], frame_to_base64(frame)


def process_video_source(
    pipeline: ViolationPipeline,
    source: str,
    max_frames: int,
) -> Tuple[List[Dict], Optional[str]]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to open video source")
    fps = cap.get(cv2.CAP_PROP_FPS) or pipeline.cfg_loader.thresholds.tracker.frame_rate
    if fps <= 1e-3:
        fps = pipeline.cfg_loader.thresholds.tracker.frame_rate
    pipeline.set_video_mode(True)
    pipeline.update_fps(fps)
    frames = []
    annotated_preview = None
    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps
        result = pipeline.process_frame(frame, ts)
        annotate_frame(frame, result.tracks)
        frames.append(frame_result_to_dict(result))
        annotated_preview = frame
        frame_idx += 1
    cap.release()
    preview_b64 = frame_to_base64(annotated_preview) if annotated_preview is not None else None
    return frames, preview_b64


@app.post("/infer")
async def infer(
    file: UploadFile | None = File(default=None),
    source_url: str | None = Form(default=None),
    camera_id: str = Form(default="default"),
    classes: str = Form(default="motorcycle"),
    max_frames: int = Form(default=120),
):
    if not file and not source_url:
        raise HTTPException(status_code=400, detail="Provide either a file or source_url")
    class_tuple = tuple({cls.strip() for cls in classes.split(",") if cls.strip()}) or ("motorcycle",)
    pipeline = pipeline_manager.build_pipeline(camera_id, class_tuple)
    logger.info("REST infer camera=%s classes=%s file=%s source=%s", camera_id, class_tuple, bool(file), source_url)

    if file:
        suffix = Path(file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            data = await file.read()
            tmp.write(data)
            tmp_path = Path(tmp.name)
        source_type = detect_source_type(str(tmp_path))
        if source_type == "image":
            frames, media = process_image_path(pipeline, tmp_path)
        else:
            frames, media = process_video_source(pipeline, str(tmp_path), max_frames)
        tmp_path.unlink(missing_ok=True)
        return JSONResponse({"frames": frames, "annotated_preview_b64": media})

    frames, media = process_video_source(pipeline, source_url, max_frames)
    return JSONResponse({"frames": frames, "annotated_preview_b64": media})
