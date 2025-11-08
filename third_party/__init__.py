"""
Helpers to treat vendored third-party repositories as importable modules.

Each upstream repository lives under this directory. We keep their original
layout untouched but make sure their paths are added to ``sys.path`` so the
original Python packages (``speed_estimator``, ``trackers``, OCR helpers, etc.)
can be imported without copying code.
"""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent

REPO_PATHS: Dict[str, Path] = {
    "speed": ROOT / "yolo8-tracking-counting-speed_estimation",
    "helmet_vgg": ROOT / "Helmet-Violation-Detection-Using-YOLO-and-VGG16",
    "helmet_yolo": ROOT / "Helmet_Detection",
    "triple": ROOT / "triple-rider-detection",
    "anpr": ROOT / "Automatic_Number_Plate_Recognition_YOLO_OCR",
    "tracking_alt": ROOT / "YOLOv8-DeepSORT-Object-Tracking",
    "legacy_motorcycle": ROOT / "public-motorcycle-violations",
    "calibration": ROOT / "EVOCamCal-vehicleSpeedEstimation",
}


def activate_repo(name: str) -> Path:
    """
    Ensure the repository ``name`` is importable by appending it to ``sys.path``.
    Returns the absolute path so callers can locate data/weights.
    """

    path = REPO_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"third_party repo '{name}' missing at {path}")

    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


# eagerly activate the repos we rely on for runtime imports
for key in ("speed", "helmet_vgg", "anpr", "triple"):
    try:
        activate_repo(key)
    except FileNotFoundError:
        # Allow partial checkouts; README documents how to pull the repos.
        pass

# Backwards compat: some vendored trackers expect ultralytics.yolo.*
try:
    import ultralytics.utils.ops as _ops  # type: ignore

    if "ultralytics.yolo.utils.ops" not in sys.modules:
        yolo_pkg = sys.modules.setdefault("ultralytics.yolo", types.ModuleType("ultralytics.yolo"))
        utils_pkg = sys.modules.setdefault("ultralytics.yolo.utils", types.ModuleType("ultralytics.yolo.utils"))
        setattr(utils_pkg, "ops", _ops)
        setattr(yolo_pkg, "utils", utils_pkg)
        sys.modules["ultralytics.yolo.utils.ops"] = _ops
except Exception:
    pass

try:
    from ultralytics.nn.tasks import DetectionModel as _DetectionModel

    if "verbose" not in inspect.signature(_DetectionModel.fuse).parameters:
        _orig_fuse = _DetectionModel.fuse

        def _patched_fuse(self, verbose: bool = True):
            return _orig_fuse(self)

        _DetectionModel.fuse = _patched_fuse  # type: ignore[assignment]
except Exception:  # pragma: no cover
    pass


__all__ = ["activate_repo", "REPO_PATHS"]
