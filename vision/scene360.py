"""Full 360 scenes from an equirectangular (2:1) panorama: a photo sphere, viewed by rotation only,
like any standard 360-photo viewer. No depth/layers — a single flat photo has no information about
what's behind the camera, so that honesty only applies to real panoramas with true 2:1 coverage."""
import cv2 as cv
import numpy as np

from .detect import detect, spoken

ASPECT_TOL = 0.25
MAX_LONG_SIDE = 4096


def validate(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    ratio = w / h
    if abs(ratio - 2.0) > ASPECT_TOL:
        raise ValueError(
            f"This image is {ratio:.2f}:1, not a 2:1 equirectangular panorama. A regular 'Panorama' "
            f"mode photo only sweeps sideways and isn't full-sphere. Use your phone's Photo Sphere "
            f"capture mode, or an export from a 360 camera.")


def build(img: np.ndarray) -> tuple[np.ndarray, dict]:
    validate(img)
    h, w = img.shape[:2]
    if w > MAX_LONG_SIDE:
        s = MAX_LONG_SIDE / w
        img = cv.resize(img, (MAX_LONG_SIDE, round(h * s)), interpolation=cv.INTER_AREA)
    dets = detect(img)
    meta = {
        "type": "360",
        "image": {"w": img.shape[1], "h": img.shape[0]},
        "detections": [{"label": spoken(d["name"]), "score": d["score"]} for d in dets],
    }
    return img, meta
