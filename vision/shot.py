"""OpenCV on every picture a live story paints.

- Lettering check: image models like to letter signs and titles into a scene; PP-OCRv3 text detection (cv.dnn) finds
  them, and the moment is painted again.
- Depth: Depth Anything V2 (cv.dnn) gives each moment its relief, so the viewer's curved screen bends toward them
  where things are near.
- World: one wide painting of the place becomes a seamless 360 equirectangular world."""
import cv2 as cv
import numpy as np

from . import depth
from .detect import detect
from .page import detect_lines
from .world import FIGURES

TEXT_LIMIT = 0.004  # share of the picture covered by word-like text boxes; clean paintings measure 0.0000


def erase_figures(img: np.ndarray, score: float = 0.35) -> int:
    """People or animals the painter put into an empty world anyway (often a few small figures on the horizon):
    YOLOX (cv.dnn) finds them on half-overlapping square tiles, and xphoto's shift-map inpainting rebuilds the
    scenery behind them from patches of the painting itself (Telea or FSR would leave smears), working on a crop
    around the figures to stay fast. In place; returns how many were erased."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    found = 0
    for x in sorted({min(x, max(0, w - h)) for x in range(0, w, max(1, h // 2))}):
        for d in detect(img[:, x:x + h]):
            if d["name"] in FIGURES and d["score"] >= score:
                x0, y0, x1, y1 = (int(v) for v in d["box"])
                pad = max(4, (y1 - y0) // 6)
                cv.rectangle(mask, (x + x0 - pad, y0 - pad), (x + x1 + pad, y1 + pad), 255, cv.FILLED)
                found += 1
    if not found:
        return 0
    n, _, stats, _ = cv.connectedComponentsWithStats(mask)
    for bx, by, bw, bh, _ in stats[1:]:  # one small crop per figure (or group), at half size: fast enough to wait for
        m = max(40, bh // 2)  # surrounding scenery to borrow patches from
        x0, y0, x1, y1 = max(0, bx - m), max(0, by - m), min(w, bx + bw + m), min(h, by + bh + m)
        crop = cv.resize(img[y0:y1, x0:x1], None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        hole = cv.resize(mask[y0:y1, x0:x1], (crop.shape[1], crop.shape[0]), interpolation=cv.INTER_NEAREST)
        out = np.zeros_like(crop)
        cv.xphoto.inpaint(crop, 255 - hole, out, cv.xphoto.INPAINT_SHIFTMAP)
        full = cv.resize(out, (x1 - x0, y1 - y0), interpolation=cv.INTER_CUBIC)
        sel = mask[y0:y1, x0:x1] > 0
        img[y0:y1, x0:x1][sel] = full[sel]
    return found


def text_cover(img: np.ndarray) -> float:
    """Share of the picture covered by detected word-like text lines (wider than tall, at least 8 px high)."""
    s = 960 / max(img.shape[:2])
    small = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else img
    words = [q for q in detect_lines(small) if np.ptp(q[:, 0]) > 2 * np.ptp(q[:, 1]) and np.ptp(q[:, 1]) >= 8]
    return sum(cv.contourArea(q.astype(np.float32)) for q in words) / (small.shape[0] * small.shape[1])


def depth_png(img: np.ndarray, width: int = 512) -> bytes:
    """Relative depth as an 8-bit PNG, 255 = nearest, at a size the browser can sample per vertex."""
    return shot_maps(img, width)[0]


def _keep(mask: np.ndarray, min_frac: float, touch_top: bool = False, min_width: float = 0.0) -> np.ndarray:
    """Connected regions of `mask` big enough to animate (optionally touching the top edge, or wide enough)."""
    h, w = mask.shape
    n, labels, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros((h, w), bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area >= min_frac * h * w and bw >= min_width * w and (not touch_top or y <= 2):
            out |= labels == i
    return out


def motion_regions(img: np.ndarray, near: np.ndarray, water: bool = False) -> dict[str, np.ndarray]:
    """Where a painting can come alive, as soft 0..1 masks: sky (far by depth, smooth, not foliage, reaching the
    top edge), water (blue, low in the picture, smooth and wide) and foliage (green and textured). All from colour,
    local texture (standard deviation) and the Depth Anything map: no extra model. Water is only looked for when the
    story says there is some (`water`): by colour alone a blue coat or night hills would ripple."""
    h, w = near.shape
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0].astype(np.int16), hsv[..., 1]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv.blur(gray, (9, 9))
    texture = np.sqrt(np.maximum(cv.blur(gray * gray, (9, 9)) - mu * mu, 0))
    rows = np.arange(h)[:, None] / h
    close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))

    green = (hue >= 30) & (hue <= 90) & (sat > 50)
    sky = (near < 0.25) & (texture < 10) & (rows < 0.7) & ~green
    sky = cv.morphologyEx(sky.astype(np.uint8), cv.MORPH_CLOSE, close).astype(bool)  # stars don't break the sky
    sky = _keep(sky, 0.03, touch_top=True)
    water = (hue >= 85) & (hue <= 135) & (sat > 40) & (rows > 0.45) & (texture < 22) & ~sky if water else sky & False
    water = _keep(cv.morphologyEx(water.astype(np.uint8), cv.MORPH_CLOSE, close).astype(bool), 0.015, min_width=0.15)
    leaves = green & (texture > 10) & ~sky & ~water
    leaves = _keep(cv.morphologyEx(leaves.astype(np.uint8), cv.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool), 0.01)
    soft = lambda m: cv.GaussianBlur(m.astype(np.float32), (0, 0), 4)
    return {"sky": soft(sky), "water": soft(water), "leaves": soft(leaves)}


def shot_maps(img: np.ndarray, width: int = 512, water: bool = False) -> tuple[bytes, bytes, dict]:
    """The two maps the browser animates a moment with, as PNGs at `width`: depth (255 = nearest) for the relief,
    and motion (red = sky drifting, green = water rippling, blue = leaves swaying). Plus each region's share."""
    return maps_and_regions(img, width, water)[:3]


def maps_and_regions(img: np.ndarray, width: int = 512, water: bool = False) -> tuple[bytes, bytes, dict, dict]:
    """`shot_maps` plus the soft region masks themselves (for the world-life analysis)."""
    small = cv.resize(img, (width, max(1, round(img.shape[0] * width / img.shape[1]))), interpolation=cv.INTER_AREA)
    near = np.nan_to_num(depth.normalized(depth.inverse_depth(small)), nan=0.0)  # a flat picture has no depth range
    regions = motion_regions(small, near, water)
    motion = np.dstack([regions["leaves"], regions["water"], regions["sky"]])  # BGR, so sky lands in red
    return (cv.imencode(".png", (near * 255).astype(np.uint8))[1].tobytes(),
            cv.imencode(".png", (motion * 255).astype(np.uint8))[1].tobytes(),
            {k: round(float((v > 0.5).mean()), 3) for k, v in regions.items()}, regions)


def world_from_wide(img: np.ndarray, out_w: int = 3072, band: float = 0.12) -> np.ndarray:
    """A wide painting -> a seamless 2:1 equirectangular world: wrapped around the full circle at its own
    proportions, its two ends cross-faded behind the viewer (the painting's centre stays straight ahead), and the
    leftover sky and ground calmed toward the poles so nothing pinches at the zenith or nadir."""
    out_h = out_w // 2
    b = int(out_w * band)
    h, w = img.shape[:2]
    strip_h = min(out_h, round(h * (out_w + b) / w))
    strip = cv.resize(img, (out_w + b, strip_h), interpolation=cv.INTER_CUBIC).astype(np.float32)
    loop = strip[:, :out_w].copy()
    a = np.linspace(0, 1, b, dtype=np.float32)[None, :, None]
    loop[:, :b] = strip[:, :b] * a + strip[:, out_w:] * (1 - a)  # column 0 continues where column out_w-1 ends
    loop = np.roll(loop, -(b // 2), axis=1)  # the painting's centre column faces forward; the fade lands behind
    out = np.empty((out_h, out_w, 3), np.float32)
    top = (out_h - strip_h) // 2
    out[top:top + strip_h] = loop
    sky, ground = loop[: max(2, strip_h // 12)].reshape(-1, 3).mean(0), loop[-max(2, strip_h // 12):].reshape(-1, 3).mean(0)
    out[:top], out[top + strip_h:] = sky, ground
    t = np.linspace(0, 1, out_h, dtype=np.float32)[:, None, None]
    edge = strip_h / out_h / 2  # how far the painting reaches from the horizon, as a fraction of the height
    pole = np.clip((np.abs(t - 0.5) - (edge - 0.08)) / 0.08, 0, 1)
    out = out * (1 - pole) + np.where(t < 0.5, sky, ground) * pole
    return np.clip(out, 0, 255).astype(np.uint8)
