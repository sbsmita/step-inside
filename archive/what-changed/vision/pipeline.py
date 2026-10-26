import time

import cv2 as cv
import numpy as np

from .align import Alignment, align, features, warp
from .change import detect as detect_changes
from .detect import TRANSIENT, detect, label_for, spoken
from .locate import DEFAULT_HFOV, bearing_deg, clock, direction_phrase, height_phrase
from .quality import assess

WORK_MAX = 960


def decode(data: bytes) -> np.ndarray:
    img = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a decodable image")
    return img


def prepare(img: np.ndarray) -> np.ndarray:
    s = WORK_MAX / max(img.shape[:2])
    return cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else img


def _project(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    return cv.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)


def _norm(pts: np.ndarray, shape: tuple) -> list:
    h, w = shape[:2]
    return [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in pts]


def compare(base: np.ndarray, cur: np.ndarray, hfov: float = DEFAULT_HFOV) -> dict:
    """Find what changed in `cur` relative to the `base` image, located in the current camera view."""
    t0 = time.perf_counter()
    b, c = prepare(base), prepare(cur)
    al = align(b, c, cur_feats=features(c))
    return _report(b, c, al, hfov, t0, base_dets=detect(b))


def scan(keyframes: list, cur: np.ndarray, hfov: float = DEFAULT_HFOV) -> dict:
    """Relocalize against the home's keyframes (prepared images with cached features), then compare."""
    t0 = time.perf_counter()
    c = prepare(cur)
    fc = features(c)
    best, best_al = None, None
    for kf in keyframes:
        al = align(kf.image, c, base_feats=kf.feats, cur_feats=fc, refine=False)
        if al.ok and (best_al is None or al.inliers > best_al.inliers):
            best, best_al = kf, al
    if best is None:
        return _report(None, c, Alignment(ok=False), hfov, t0)
    al = align(best.image, c, base_feats=best.feats, cur_feats=fc, refine=True)
    result = _report(best.image, c, al, hfov, t0, base_dets=best.detections)
    result["keyframe"] = best.name
    return result


def _px_box(corners: np.ndarray) -> tuple:
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def _attach_labels(changes: list[dict], raw: list[tuple], base_dets: list, cur_dets: list) -> None:
    """Name each change from YOLOX detections: missing things from the baseline, new/moved things from now."""
    for change, (base_corners, cur_corners, from_corners) in zip(changes, raw):
        if change["kind"] == "disappeared":
            d = label_for(_px_box(base_corners), base_dets)
        else:
            d = label_for(_px_box(cur_corners), cur_dets)
            if d is None and from_corners is not None:
                d = label_for(_px_box(from_corners), base_dets)
        if d is not None:
            change["label"] = spoken(d["name"])
            change["label_score"] = d["score"]
            change["transient"] = d["name"] in TRANSIENT


def _report(b: np.ndarray | None, c: np.ndarray, al, hfov: float, t0: float, base_dets: list | None = None) -> dict:
    result = {
        "quality": assess(c),
        "alignment": al.summary(),
        "frame": {"w": c.shape[1], "h": c.shape[0]},
        "changes": [],
    }
    if not al.ok:
        result["status"] = "lost"
        result["timing_ms"] = round((time.perf_counter() - t0) * 1000)
        return result

    warped, valid = warp(c, al.H, b.shape)
    regions, _ = detect_changes(b, warped, valid)
    Hinv = np.linalg.inv(al.H)
    raw = []

    for i, r in enumerate(regions):
        x, y, w, h = r["bbox"]
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float64)
        cur_corners = _project(corners, Hinv)
        cx, cy = _project(np.array([r["centroid"]]), Hinv)[0]
        bearing = bearing_deg(cx, c.shape[1], hfov)
        change = {
            "id": i,
            "kind": r["kind"],
            "score": r["score"],
            "area_frac": r["area_frac"],
            "bearing_deg": round(bearing, 1),
            "direction": direction_phrase(bearing),
            "clock": clock(bearing),
            "height": height_phrase(cy, c.shape[0]),
            "box_current": _norm(cur_corners, c.shape),
            "box_baseline": _norm(corners, b.shape),
            "polygon_baseline": _norm(np.array(r["polygon"]), b.shape),
        }
        from_corners = None
        if "from_bbox" in r:
            fx, fy, fw, fh = r["from_bbox"]
            from_corners = np.array([[fx, fy], [fx + fw, fy], [fx + fw, fy + fh], [fx, fy + fh]], np.float64)
            from_cur = _project(from_corners, Hinv)
            fcx, fcy = from_cur.mean(axis=0)
            from_bearing = bearing_deg(fcx, c.shape[1], hfov)
            change["from_box_baseline"] = _norm(from_corners, b.shape)
            change["from_box_current"] = _norm(from_cur, c.shape)
            change["from_direction"] = direction_phrase(from_bearing)
            change["from_height"] = height_phrase(fcy, c.shape[0])
            change["match"] = r["match"]
        result["changes"].append(change)
        raw.append((corners, cur_corners, from_corners))

    if result["changes"]:
        _attach_labels(result["changes"], raw, base_dets or [], detect(c))

    result["status"] = "ok"
    result["coverage"] = round(float(valid.mean()), 3)
    result["timing_ms"] = round((time.perf_counter() - t0) * 1000)
    return result
