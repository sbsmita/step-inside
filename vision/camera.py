"""Single-image camera estimation: field of view from vanishing points, metric scale from known-size objects."""
import math

import cv2 as cv
import numpy as np

DEFAULT_HFOV = 60.0
KNOWN_HEIGHT_M = {"person": 1.7, "chair": 0.9, "couch": 0.85, "dining table": 0.75, "refrigerator": 1.75,
                  "car": 1.5, "bus": 3.0, "truck": 3.0, "bicycle": 1.0, "potted plant": 0.6, "tv": 0.6,
                  "bed": 0.6, "toilet": 0.75, "horse": 1.6, "cow": 1.4, "dog": 0.55, "cat": 0.3}


def _segments(img: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(cv.GaussianBlur(gray, (5, 5), 0), 60, 160)
    min_len = max(img.shape[:2]) // 25
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=min_len, maxLineGap=6)
    return np.empty((0, 4), np.float32) if lines is None else lines.reshape(-1, 4).astype(np.float32)


def _ransac_vp(segs: np.ndarray, rng: np.random.Generator, iters: int = 300, tol_deg: float = 2.0):
    """Vanishing point (homogeneous) supported by the most segments, and the inlier mask."""
    if len(segs) < 4:
        return None, np.zeros(len(segs), bool)
    p1 = np.c_[segs[:, :2], np.ones(len(segs))]
    p2 = np.c_[segs[:, 2:], np.ones(len(segs))]
    lines = np.cross(p1, p2)
    mids = (segs[:, :2] + segs[:, 2:]) / 2
    dirs = segs[:, 2:] - segs[:, :2]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
    lengths = np.linalg.norm(segs[:, 2:] - segs[:, :2], axis=1)
    best, best_score, best_mask = None, 0.0, None
    for _ in range(iters):
        i, j = rng.choice(len(segs), 2, replace=False)
        vp = np.cross(lines[i], lines[j])
        if abs(vp[2]) > 1e-9:
            to_vp = vp[:2] / vp[2] - mids
        else:
            to_vp = np.tile(vp[:2], (len(segs), 1))
        to_vp /= np.linalg.norm(to_vp, axis=1, keepdims=True) + 1e-9
        ang = np.degrees(np.arccos(np.clip(np.abs((to_vp * dirs).sum(axis=1)), 0, 1)))
        mask = ang < tol_deg
        score = lengths[mask].sum()
        if score > best_score:
            best, best_score, best_mask = vp, score, mask
    return best, best_mask


def estimate_hfov(img: np.ndarray, seed: int = 0) -> dict:
    """Two orthogonal finite vanishing points v1, v2 give f^2 = -(v1-c).(v2-c) (principal point at center)."""
    h, w = img.shape[:2]
    segs = _segments(img)
    rng = np.random.default_rng(seed)
    vps, remaining = [], segs
    for _ in range(3):
        vp, mask = _ransac_vp(remaining, rng)
        if vp is None or mask.sum() < 6:
            break
        vps.append(vp)
        remaining = remaining[~mask]
    c = np.array([w / 2, h / 2])
    finite = [v[:2] / v[2] for v in vps if abs(v[2]) > 1e-6 and np.linalg.norm(v[:2] / v[2] - c) < 6 * w]
    best = None
    for i in range(len(finite)):
        for j in range(i + 1, len(finite)):
            f2 = -np.dot(finite[i] - c, finite[j] - c)
            if f2 > 0:
                hfov = math.degrees(2 * math.atan(w / (2 * math.sqrt(f2))))
                if 35 <= hfov <= 100 and (best is None or abs(hfov - DEFAULT_HFOV) < abs(best - DEFAULT_HFOV)):
                    best = hfov
    if best is None:
        return {"hfov_deg": DEFAULT_HFOV, "source": "default", "segments": int(len(segs)), "vanishing_points": len(vps)}
    return {"hfov_deg": round(best, 1), "source": "vanishing_points", "segments": int(len(segs)),
            "vanishing_points": len(vps)}


def focal_px(width: int, hfov_deg: float) -> float:
    return (width / 2) / math.tan(math.radians(hfov_deg / 2))


def estimate_scale(inv_n: np.ndarray, detections: list[dict], f: float, outdoor: bool) -> dict:
    """Choose near/far planes so known-size objects come out at their pinhole-predicted distances."""
    ratio = 30.0 if outdoor else 6.0
    nears, used = [], []
    for d in detections:
        height = KNOWN_HEIGHT_M.get(d["name"])
        x0, y0, x1, y1 = (int(v) for v in d["box"])
        x0, y0 = max(0, x0), max(0, y0)
        px_h = y1 - y0
        if height is None or px_h < 20 or y1 >= inv_n.shape[0] - 2:  # cut off at the bottom: height unreliable
            continue
        z = f * height / px_h
        region = inv_n[y0 + (y1 - y0) // 4:y1, x0:x1]
        if region.size == 0:
            continue
        v = float(np.median(region))
        nears.append(z * (v * (1 - 1 / ratio) + 1 / ratio))
        used.append({"name": d["name"], "distance_m": round(z, 2)})
    near = float(np.median(nears)) if nears else 1.2
    near = float(np.clip(near, 0.3, 15.0))
    return {"near_m": round(near, 3), "far_m": round(min(near * ratio, 80.0), 2),
            "source": "known_objects" if used else "default", "objects": used}


def looks_outdoor(img: np.ndarray, inv_n: np.ndarray) -> bool:
    """Lots of far, bright, smooth pixels in the top third is a sky."""
    top = slice(0, inv_n.shape[0] // 3)
    far = inv_n[top] < 0.08
    bright = cv.cvtColor(img[top], cv.COLOR_BGR2HSV)[..., 2] > 140
    return float((far & bright).mean()) > 0.3
