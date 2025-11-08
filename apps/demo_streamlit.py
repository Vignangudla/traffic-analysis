from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from common.config import ConfigLoader
from logic.annotate import annotate_frame
from logic.events import track_to_schema
from pipeline.engine import ModelPaths, ViolationPipeline

st.set_page_config(page_title="Two-Wheeler Violations", layout="wide")
st.title("Two-Wheeler Enforcement Demo")

cfg_loader = ConfigLoader(Path("configs/thresholds.yaml"), Path("configs/calib.yaml"))
model_paths = ModelPaths(
    detector=Path("third_party/yolo8-tracking-counting-speed_estimation/weights/motorcycle.pt"),
    helmet=Path("third_party/Helmet-Violation-Detection-Using-YOLO-and-VGG16/weights/helmet_vgg16.pth"),
    plate=Path("third_party/Automatic_Number_Plate_Recognition_YOLO_OCR/model/best.pt"),
)

classes = st.sidebar.multiselect("Classes", ["motorcycle", "scooter"], default=["motorcycle"])
max_frames = st.sidebar.slider("Max frames", 1, 300, 90)
input_choice = st.sidebar.selectbox("Input type", ["Image", "Video", "RTSP"], index=0)
rtsp_url = st.sidebar.text_input("RTSP URL", value="")
file_uploader = st.file_uploader("Upload media", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"]) if input_choice != "RTSP" else None
run_button = st.sidebar.button("Run inference")


def to_frame(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def render_results(results, annotated_frame):
    st.subheader("Events")
    rows = []
    for frame_result in results:
        for track in frame_result["tracks"]:
            row = {
                "frame": frame_result["frame_idx"],
                "ts": frame_result["ts"],
                **track,
            }
            rows.append(row)
    if rows:
        st.dataframe(rows)
    if annotated_frame is not None:
        st.subheader("Preview")
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")


def run_pipeline_on_source(source_path: str, source_type: str):
    pipeline = ViolationPipeline(
        config_loader=cfg_loader,
        model_paths=model_paths,
        target_classes=classes or ["motorcycle"],
        camera_id="default",
        enable_speed=source_type != "image",
    )
    pipeline.set_video_mode(source_type != "image")
    outputs = []
    annotated_preview = None
    if source_type == "image":
        frame = cv2.imread(source_path)
        result = pipeline.process_frame(frame, 0.0)
        annotate_frame(frame, result.tracks)
        outputs.append({
            "frame_idx": result.frame_idx,
            "ts": result.ts,
            "tracks": [track_to_schema(t) for t in result.tracks],
        })
        annotated_preview = frame
    else:
        cap = cv2.VideoCapture(source_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or cfg_loader.thresholds.tracker.frame_rate
        pipeline.update_fps(fps if fps > 1e-3 else cfg_loader.thresholds.tracker.frame_rate)
        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            ts = frame_idx / fps if fps > 1e-3 else frame_idx / cfg_loader.thresholds.tracker.frame_rate
            result = pipeline.process_frame(frame, ts)
            annotate_frame(frame, result.tracks)
            outputs.append({
                "frame_idx": result.frame_idx,
                "ts": result.ts,
                "tracks": [track_to_schema(t) for t in result.tracks],
            })
            annotated_preview = frame
            frame_idx += 1
        cap.release()
    render_results(outputs, annotated_preview)


if run_button:
    if input_choice == "RTSP":
        if not rtsp_url:
            st.error("Provide an RTSP URL")
        else:
            run_pipeline_on_source(rtsp_url, "stream")
    else:
        if not file_uploader:
            st.error("Upload a file")
        else:
            suffix = Path(file_uploader.name).suffix.lower()
            if input_choice == "Image" or suffix in {".jpg", ".jpeg", ".png"}:
                frame = to_frame(file_uploader.read())
                if frame is None:
                    st.error("Invalid image")
                else:
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp_path = Path(tmp.name)
                    cv2.imwrite(tmp_path.as_posix(), frame)
                    run_pipeline_on_source(tmp_path.as_posix(), "image")
                    tmp_path.unlink(missing_ok=True)
            else:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                with tmp_path.open("wb") as handle:
                    handle.write(file_uploader.read())
                run_pipeline_on_source(tmp_path.as_posix(), "video")
                tmp_path.unlink(missing_ok=True)
