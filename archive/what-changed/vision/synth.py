"""Synthetic rooms with known geometry: both camera views are perspective crops of one larger canvas."""
import cv2 as cv
import numpy as np

VIEW_W, VIEW_H = 960, 720
MARGIN_X, MARGIN_Y = 160, 120


def make_canvas(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w, h = VIEW_W + 2 * MARGIN_X, VIEW_H + 2 * MARGIN_Y
    horizon = int(h * 0.62)
    img = np.zeros((h, w, 3), np.uint8)
    img[:horizon] = (205, 210, 214)
    img[horizon:] = (95, 125, 160)
    for x in range(-200, w, 45):
        cv.line(img, (x, horizon), (x + 140, h), (75, 105, 140), 3)
    for _ in range(70):
        color = rng.integers(0, 256, 3).tolist()
        x, y = int(rng.integers(0, w)), int(rng.integers(0, horizon - 20))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv.rectangle(img, (x, y), (x + int(rng.integers(20, 100)), y + int(rng.integers(20, 100))), color, -1)
        elif kind == 1:
            cv.circle(img, (x, y), int(rng.integers(8, 45)), color, -1)
        else:
            text = "".join(rng.choice(list("ABCDEFGHKMNPRSTUVWXYZ"), 3))
            cv.putText(img, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
    noise = rng.normal(0, 3, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def place_object(canvas: np.ndarray, x: int, y: int, size: int = 90, seed: int = 1) -> tuple[np.ndarray, tuple]:
    """Draw a mug-like textured object; returns the new canvas and its bbox in canvas coords."""
    rng = np.random.default_rng(seed)
    out = canvas.copy()
    w, h = size, int(size * 1.1)
    cv.rectangle(out, (x, y), (x + w, y + h), (40, 40, 200), -1)
    cv.rectangle(out, (x, y + h // 3), (x + w, y + h // 2), (240, 240, 240), -1)
    cv.circle(out, (x + w + size // 6, y + h // 2), size // 4, (40, 40, 200), size // 10)
    for _ in range(25):
        px, py = int(rng.integers(x, x + w)), int(rng.integers(y, y + h))
        cv.circle(out, (px, py), 3, rng.integers(0, 256, 3).tolist(), -1)
    return out, (x, y, w + size // 6 + size // 4, h)


def place_shoe(canvas: np.ndarray, x: int, y: int, size: int = 120) -> tuple[np.ndarray, tuple]:
    out = canvas.copy()
    w, h = size, size // 2
    cv.ellipse(out, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, (30, 30, 35), -1)
    cv.rectangle(out, (x + w // 2, y), (x + w, y + h // 2), (30, 30, 35), -1)
    for i in range(4):
        cv.line(out, (x + w // 2 + 8 + i * 9, y + 4), (x + w // 2 + 14 + i * 9, y + h // 2 - 4), (230, 230, 230), 2)
    cv.ellipse(out, (x + w // 2, y + h - 4), (w // 2, 5), 0, 0, 360, (220, 220, 220), -1)
    return out, (x, y, w, h)


def add_spill(canvas: np.ndarray, x: int, y: int, size: int = 110, seed: int = 2) -> tuple[np.ndarray, tuple]:
    rng = np.random.default_rng(seed)
    out = canvas.copy()
    angles = np.sort(rng.uniform(0, 2 * np.pi, 14))
    radii = rng.uniform(0.55, 1.0, 14) * size / 2
    pts = np.stack([x + size / 2 + radii * np.cos(angles), y + size / 3 + 0.6 * radii * np.sin(angles)], 1)
    overlay = out.copy()
    cv.fillPoly(overlay, [pts.astype(np.int32)], (40, 90, 150))
    out = cv.addWeighted(overlay, 0.65, out, 0.35, 0)
    cv.circle(out, (x + size // 3, y + size // 4), 6, (235, 240, 245), -1)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return out, (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def view(canvas: np.ndarray, jitter: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Render a camera view; returns the image and M, the canvas->view homography."""
    rng = np.random.default_rng(seed)
    src = np.float32([[MARGIN_X, MARGIN_Y], [MARGIN_X + VIEW_W, MARGIN_Y],
                      [MARGIN_X + VIEW_W, MARGIN_Y + VIEW_H], [MARGIN_X, MARGIN_Y + VIEW_H]])
    src += rng.uniform(-jitter, jitter, src.shape).astype(np.float32) * np.float32([VIEW_W, VIEW_H])
    dst = np.float32([[0, 0], [VIEW_W, 0], [VIEW_W, VIEW_H], [0, VIEW_H]])
    M = cv.getPerspectiveTransform(src, dst)
    return cv.warpPerspective(canvas, M, (VIEW_W, VIEW_H), flags=cv.INTER_LINEAR), M


def relight(img: np.ndarray, gain: float = 1.2, bias: float = -15, tint=(1.0, 0.95, 1.06)) -> np.ndarray:
    """Warmer, brighter lamp light: global gain/bias plus a color tint (with some channel clipping)."""
    f = img.astype(np.float32) * gain + bias
    f *= np.float32(tint)
    return np.clip(f, 0, 255).astype(np.uint8)


def bbox_to_view(bbox: tuple, M: np.ndarray) -> tuple:
    x, y, w, h = bbox
    pts = np.float64([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
    p = cv.perspectiveTransform(pts, M).reshape(-1, 2)
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)
