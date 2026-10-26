"""Monocular depth with Depth Anything V2 Small (Apache-2.0) run through OpenCV's DNN module."""
import logging
from pathlib import Path

import cv2 as cv
import numpy as np

from .netpool import NetPool, pool_size

log = logging.getLogger(__name__)

MODEL = Path(__file__).resolve().parent / "models" / "depth_anything_v2_small.onnx"
LONG_SIDE = 686  # 49 * 14; the ViT patch size is 14, so both sides must be multiples of 14
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _load():
    if not MODEL.exists():
        raise FileNotFoundError(f"{MODEL} missing; run `python -m vision.models.fetch`")
    return cv.dnn.readNet(str(MODEL))


_pool = NetPool(_load, pool_size("DEPTH_NETS", 2))


def _input_size(h: int, w: int) -> tuple[int, int]:
    s = LONG_SIDE / max(h, w)
    return max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14)


def inverse_depth(img: np.ndarray) -> np.ndarray:
    """Relative inverse depth (bigger = closer) at the image's resolution, edge-aware refined."""
    h, w = img.shape[:2]
    ih, iw = _input_size(h, w)
    rgb = cv.cvtColor(cv.resize(img, (iw, ih), interpolation=cv.INTER_CUBIC), cv.COLOR_BGR2RGB)
    blob = np.transpose((rgb.astype(np.float32) / 255.0 - MEAN) / STD, (2, 0, 1))[np.newaxis]
    with _pool.net() as net:
        net.setInput(np.ascontiguousarray(blob))
        inv = np.squeeze(net.forward()).astype(np.float32)
    inv = cv.resize(inv, (w, h), interpolation=cv.INTER_CUBIC)
    # Snap depth edges to image edges so cut-outs follow object outlines, not the network's 14 px patches.
    guide = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    radius = max(4, min(h, w) // 120)
    return cv.ximgproc.guidedFilter(guide, inv, radius, 1e-3 * float(inv.max() - inv.min()) ** 2)


def normalized(inv: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(inv, [lo_pct, hi_pct])
    return np.clip((inv - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def to_metric(inv_n: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    """Map normalized inverse depth in [0, 1] to meters, linear in disparity between far (0) and near (1)."""
    return 1.0 / (inv_n * (1.0 / near_m - 1.0 / far_m) + 1.0 / far_m)
