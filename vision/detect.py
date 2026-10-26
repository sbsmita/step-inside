"""Object labels for change regions via OpenCV DNN + YOLOX (OpenCV Model Zoo, Apache-2.0).

Pre/post-processing follows the Zoo's reference implementation: RGB, 640x640 letterbox padded with 114,
grid decoding over strides 8/16/32, then class-aware NMS."""
import logging
from functools import lru_cache
from pathlib import Path

import cv2 as cv
import numpy as np

from .netpool import NetPool, pool_size

log = logging.getLogger(__name__)

MODEL = Path(__file__).resolve().parent / "models" / "object_detection_yolox_2022nov.onnx"
INPUT = 640
STRIDES = (8, 16, 32)
CONF = 0.35
NMS = 0.5

CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)
SPOKEN = {"cell phone": "phone", "potted plant": "plant", "dining table": "table", "tv": "TV",
          "sports ball": "ball", "teddy bear": "teddy bear", "hair drier": "hair dryer"}
TRANSIENT = {"person", "cat", "dog", "bird"}


def spoken(name: str) -> str:
    return SPOKEN.get(name, name)


def _anchors():
    grids, strides = [], []
    for s in STRIDES:
        n = INPUT // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


@lru_cache(maxsize=1)
def _setup():
    if not MODEL.exists():
        log.warning("YOLOX model missing (%s); run `python -m vision.models.fetch`. Objects stay unlabeled.", MODEL)
        return None
    return _anchors()


_pool = NetPool(lambda: cv.dnn.readNet(str(MODEL)), pool_size("DETECT_NETS", 2))


def available() -> bool:
    return _setup() is not None


def detect(img: np.ndarray) -> list[dict]:
    """Returns [{"name", "score", "box": (x0, y0, x1, y1)}] in image pixels."""
    anchors = _setup()
    if anchors is None:
        return []
    grids, strides = anchors
    ratio = min(INPUT / img.shape[0], INPUT / img.shape[1])
    resized = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB),
                        (int(img.shape[1] * ratio), int(img.shape[0] * ratio)), interpolation=cv.INTER_LINEAR)
    padded = np.full((INPUT, INPUT, 3), 114.0, np.float32)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    with _pool.net() as net:
        net.setInput(np.transpose(padded, (2, 0, 1))[np.newaxis])
        dets = net.forward(net.getUnconnectedOutLayersNames())[0][0].copy()

    dets[:, :2] = (dets[:, :2] + grids) * strides
    dets[:, 2:4] = np.exp(dets[:, 2:4]) * strides
    xyxy = np.stack([dets[:, 0] - dets[:, 2] / 2, dets[:, 1] - dets[:, 3] / 2,
                     dets[:, 0] + dets[:, 2] / 2, dets[:, 1] + dets[:, 3] / 2], 1)
    scores = dets[:, 4:5] * dets[:, 5:]
    best, cls = scores.max(1), scores.argmax(1)
    keep = cv.dnn.NMSBoxesBatched(xyxy.tolist(), best.tolist(), cls.tolist(), CONF, NMS)
    return [{"name": CLASSES[int(cls[i])], "score": round(float(best[i]), 3),
             "box": tuple(float(v) / ratio for v in xyxy[i])} for i in np.array(keep).ravel()]


def label_for(box: tuple, detections: list[dict], min_overlap: float = 0.5) -> dict | None:
    """The detection that best explains a change region: most of the detection lies inside the region, or IoU is high."""
    x0, y0, x1, y1 = box
    best, best_score = None, 0.0
    for d in detections:
        dx0, dy0, dx1, dy1 = d["box"]
        iw, ih = max(0.0, min(x1, dx1) - max(x0, dx0)), max(0.0, min(y1, dy1) - max(y0, dy0))
        inter = iw * ih
        det_area = max((dx1 - dx0) * (dy1 - dy0), 1.0)
        union = (x1 - x0) * (y1 - y0) + det_area - inter
        overlap = max(inter / det_area, inter / max(union, 1.0))
        if overlap >= min_overlap and overlap * d["score"] > best_score:
            best, best_score = d, overlap * d["score"]
    return best
