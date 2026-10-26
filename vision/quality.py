import cv2 as cv
import numpy as np

QUALITY_WIDTH = 640
BLUR_MIN = 40.0
DARK_MAX = 45.0
GLARE_MAX = 0.08
MOTION_MAX = 0.02


def _small_gray(img: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = QUALITY_WIDTH / gray.shape[1]
    return cv.resize(gray, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)


def assess(img: np.ndarray) -> dict:
    """Scout checks: is this frame usable, and if not, what should the user do?"""
    gray = _small_gray(img)
    blur = float(cv.Laplacian(gray, cv.CV_64F).var())
    hist = cv.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    brightness = float((hist * np.arange(256)).sum() / max(hist.sum(), 1))

    hsv = cv.cvtColor(cv.resize(img, (gray.shape[1], gray.shape[0])), cv.COLOR_BGR2HSV)
    glare = float(((hsv[..., 2] > 245) & (hsv[..., 1] < 40)).mean())

    issues = []
    if blur < BLUR_MIN:
        issues.append("blurry")
    if brightness < DARK_MAX:
        issues.append("too_dark")
    if glare > GLARE_MAX:
        issues.append("glare")
    return {
        "ok": not issues,
        "issues": issues,
        "blur": round(blur, 1),
        "brightness": round(brightness, 1),
        "glare_ratio": round(glare, 4),
    }


def motion(prev: np.ndarray, cur: np.ndarray) -> float:
    """Mean optical-flow magnitude as a fraction of frame width; high means the phone is moving too fast."""
    a, b = _small_gray(prev), _small_gray(cur)
    if a.shape != b.shape:
        b = cv.resize(b, (a.shape[1], a.shape[0]))
    flow = cv.calcOpticalFlowFarneback(a, b, None, 0.5, 5, 21, 3, 5, 1.2, 0)
    # Textureless wall/floor pixels report ~0 flow, so a high percentile tracks the actual camera pan.
    mag = np.linalg.norm(flow, axis=2)
    return float(np.percentile(mag, 90) / a.shape[1])
