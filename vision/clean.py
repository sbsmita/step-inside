"""Input clean-up: flatten photographed prints/pages and crop screenshot UI bars before 3D processing."""
import cv2 as cv
import numpy as np

MIN_QUAD_FRAC = 0.2
MAX_QUAD_FRAC = 0.97
BAND_STD = 4.0
MAX_BAND_FRAC = 0.18


def _order(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def find_print_quad(img: np.ndarray) -> np.ndarray | None:
    """Largest convex 4-corner outline (a photo print, book page or screen) that isn't the whole frame."""
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    small = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    gray = cv.GaussianBlur(cv.cvtColor(small, cv.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv.dilate(cv.Canny(gray, 50, 150), np.ones((3, 3), np.uint8))
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    area = small.shape[0] * small.shape[1]
    for c in sorted(contours, key=cv.contourArea, reverse=True)[:5]:
        frac = cv.contourArea(c) / area
        if not MIN_QUAD_FRAC < frac < MAX_QUAD_FRAC:
            continue
        approx = cv.approxPolyDP(c, 0.02 * cv.arcLength(c, True), True)
        if len(approx) == 4 and cv.isContourConvex(approx):
            return _order(approx.reshape(4, 2).astype(np.float32) / scale)
    return None


def rectify(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    return cv.warpPerspective(img, cv.getPerspectiveTransform(quad, dst), (w, h), flags=cv.INTER_CUBIC)


def _uniform_run(gray: np.ndarray, from_start: bool) -> int:
    """Rows from one edge that are flat single-color UI bars (status bar, nav bar)."""
    rows = gray if from_start else gray[::-1]
    limit = int(len(rows) * MAX_BAND_FRAC)
    std = rows[:limit].std(axis=1)
    n = 0
    while n < limit and std[n] < BAND_STD:
        n += 1
    return n


def crop_ui_bars(img: np.ndarray) -> tuple[np.ndarray, dict]:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32)
    top, bottom = _uniform_run(gray, True), _uniform_run(gray, False)
    left, right = _uniform_run(gray.T, True), _uniform_run(gray.T, False)
    h, w = gray.shape
    out = img[top:h - bottom, left:w - right]
    return out, {"top": top, "bottom": bottom, "left": left, "right": right}


def clean(img: np.ndarray, kind: str = "photo") -> tuple[np.ndarray, list[dict]]:
    """kind: "photo" (as is), "print" (photographed print/page/screen: flatten), "screenshot" (crop UI bars)."""
    steps = []
    if kind == "screenshot":
        img, bars = crop_ui_bars(img)
        steps.append({"step": "crop_ui_bars", **bars})
    if kind in ("print", "page"):
        quad = find_print_quad(img)
        if quad is not None:
            img = rectify(img, quad)
            steps.append({"step": "rectify", "corners": quad.round(1).tolist()})
        else:
            steps.append({"step": "rectify", "skipped": "no clear 4-corner outline found"})
    return img, steps
