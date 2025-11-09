# Two-Wheeler Violation Pipeline

End-to-end reference implementation that merges high-accuracy upstream repositories into a single, configurable inference stack for two-wheeler enforcement. The pipeline ingests still images, offline video, or live RTSP feeds and outputs both annotated media and JSONL events capturing speed, helmet compliance, triple-riding, and license-plate OCR per motorcycle track.



## Repository Layout
```
configs/                # data/threshold/calibration YAMLs
common/, detectors/, track/, logic/, heads/  # core runtime modules
pipeline/engine.py      # orchestration of detections -> violations
run.py                  # CLI entry
api_rest.py             # FastAPI service (/infer)
apps/demo_streamlit.py  # Streamlit dashboard
third_party/            # vendored upstream repos (unchanged)
deploy/exporters.py     # ONNX/TensorRT/OpenVINO exporters
outputs/                # default artifacts
```

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

>
> | Purpose | Expected path | Source repo |
> | --- | --- | --- |
> | Motorcycle/person YOLO | `third_party/yolo8-tracking-counting-speed_estimation/weights/motorcycle.pt` | speed repo checkpoints |
> | Helmet VGG head | `third_party/Helmet-Violation-Detection-Using-YOLO-and-VGG16/weights/helmet_vgg16.pth` | ThanhSan97 repo (`Source/VGG_Training.ipynb`) |
> | Plate detector | `third_party/Automatic_Number_Plate_Recognition_YOLO_OCR/model/best.pt` | ANPR repo (`model/best.pt`) |

## Calibration
`configs/calib.yaml` stores per-camera homographies or meters-per-pixel factors:
```yaml
cameras:
  default:
    fps: 30
    meters_per_pixel: 0.05
    homography:
      - [1.0, 0.0, 0.0]
      - [0.0, 1.0, 0.0]
      - [0.0, 0.0, 1.0]
```
1. Collect 4+ correspondences between image pixels and real-world coordinates.
2. Use the optional [`third_party/EVOCamCal-vehicleSpeedEstimation`](third_party/EVOCamCal-vehicleSpeedEstimation) notebooks or OpenCV `findHomography` to solve for `H`.
3. Update `fps`, `meters_per_pixel`, and `homography` per camera ID.

Speed defaults to `"speed_reason": "no calib"` unless a camera entry exists. Single images automatically set `speed_kmh = null`, `speed_reason = "single frame"`.

## CLI Usage
```
python run.py --source data/clip.mp4 --calib configs/calib.yaml \
    --detector-weights third_party/yolo8-tracking-counting-speed_estimation/weights/motorcycle.pt \
    --helmet-weights third_party/Helmet-Violation-Detection-Using-YOLO-and-VGG16/weights/helmet_vgg16.pth \
    --plate-weights third_party/Automatic_Number_Plate_Recognition_YOLO_OCR/model/best.pt \
    --camera-id default --classes motorcycle
```
- Accepts local image/video paths or RTSP/HTTP URLs.
- Outputs annotated media (`outputs/annotated.mp4` or `.jpg`) + JSONL (`outputs/events.jsonl`).
- Use `--max-frames N` for quick smoke tests.

## REST API
```
uvicorn api_rest:app --reload --host 0.0.0.0 --port 8000
```
`POST /infer`
- multipart `file` *(image/video)* **or** `source_url` *(RTSP/HTTP)*
- optional `camera_id`, `classes`, `max_frames`
- Response payload:
```json
{
  "frames": [
    {
      "frame_idx": 12,
      "ts": 0.4,
      "tracks": [
        {
          "ts": 0.4,
          "track_id": 7,
          "motorcycle_box": [x, y, w, h],
          "riders_count": 3,
          "helmet_missing_ids": [0],
          "plate_text": "MH12AB1234",
          "plate_conf": 0.91,
          "speed_kmh": 47.2,
          "speed_reason": "ok"
        }
      ]
    }
  ],
  "annotated_preview_b64": "..."
}
```

## Streamlit Demo
```
streamlit run apps/demo_streamlit.py
```
- Upload an image/video or paste an RTSP URL.
- Adjust classes + max frames via the sidebar.
- Displays annotated preview and results table interactively.

## JSONL Schema
Each line in `outputs/events.jsonl` conforms to:
```json
{
  "ts": 0.53,
  "track_id": 17,
  "motorcycle_box": [x, y, w, h],
  "riders_count": 2,
  "helmet_missing_ids": [0],
  "plate_text": "KA01HZ1234",
  "plate_conf": 0.88,
  "speed_kmh": 41.6,
  "speed_reason": "ok"
}
```
`helmet_missing_ids` enumerates rider indices (sorted by top-to-bottom position). `speed_reason` ∈ {`ok`, `single frame`, `no calib`}.

## Fine-Tuning Hooks
| Component | Script / data | Notes |
| --- | --- | --- |
| Motorcycle/person detector | `third_party/yolo8-tracking-counting-speed_estimation/predictor` + `configs/data.yaml` | Reuse YOLOv8 training scripts; place new weights under `third_party/.../weights/`.
| Triple-rider data | `third_party/triple-rider-detection/yolov8 (1).ipynb` | Use dataset to finetune multi-person association thresholds in `configs/thresholds.yaml`.
| Helmet head | `third_party/Helmet-Violation-Detection-Using-YOLO-and-VGG16/Source/VGG_Training.ipynb` | Export `.pth` and drop into `third_party/.../weights/helmet_vgg16.pth`.
| Plate detector / OCR | `third_party/Automatic_Number_Plate_Recognition_YOLO_OCR` notebooks | Optionally switch OCR backend via `configs/thresholds.yaml -> plate.ocr` (`paddleocr` or `easyocr`).

To re-train YOLOv8 on merged datasets:
```bash
ultralytics train model=yolov8m.pt data=configs/data.yaml epochs=100 imgsz=1280
cp runs/detect/train/weights/best.pt third_party/yolo8-tracking-counting-speed_estimation/weights/motorcycle.pt
```

## Exporting (ONNX / TensorRT / OpenVINO)
```bash
python deploy/exporters.py
```
Artifacts land in `deploy/artifacts/*`:
- `onnx` for CPU/GPU runtimes.
- `openvino` exported with INT8 quantization (set `meters_per_pixel` + calibration dataset for best accuracy).
- `engine` (TensorRT INT8) for Jetson/edge GPUs (make sure TensorRT is installed in the environment).

## Performance Tips
- Target input resolution: 960–1280 px short side → ≥20 FPS on RTX 3060 / Jetson Orin when using TensorRT INT8 exports.
- Use `--classes motorcycle scooter` when operating in regions with mixed bike taxonomies.
- Tweak tracker frame-rate buffers via `configs/thresholds.yaml -> tracker.frame_rate/track_buffer` for low-FPS RTSP feeds.
- Increase `plate_interval` in `pipeline/engine.py` for high traffic scenes to amortize OCR cost.

## Troubleshooting
- Missing weights → follow table above; run `python -m ultralytics.utils.checks` to validate environment.
- PaddleOCR download stuck → set `PADDLEOCR_HOME` to a writable path with enough disk space.
- Speed = `no calib` → ensure `--camera-id` exists in `configs/calib.yaml` and RTSP feed metadata exposes FPS; otherwise set `--max-frames` high enough for smoothing.

## Roadmap
- Batch async inference for multiple RTSP feeds.
- Plug-in ViT helmet classifier (drop-in replacement in `heads/helmet.py`).
- gRPC streaming + Kafka sink for city-scale deployments.
