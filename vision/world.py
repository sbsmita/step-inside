"""A full 360 story world around a page's illustration.

The illustration sits at the centre of a 2:1 equirectangular panorama, covering exactly the angles it covers in
the 3D view, so the depth-layered scene and the surround line up. Around it, either OpenCV builds a soft wrap-around
from the illustration's own colours (no AI), or Stability AI paints the rest (AI-filled, labelled as such)."""
import logging
import math

import cv2 as cv
import numpy as np

log = logging.getLogger(__name__)

PANO_W, PANO_H = 2048, 1024
FEATHER = 36


AI_W, AI_H = 4096, 2048


def place(illus: np.ndarray, hfov_deg: float, pano_w: int = PANO_W, pano_h: int = PANO_H,
          matte: np.ndarray | None = None, vfov_deg: float | None = None) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Put the illustration on an empty panorama at its true angular size. Returns (canvas, keep_mask, box).
    With a paper matte, only the art (not the blank paper around it) counts as kept. `vfov_deg` overrides the
    pinhole vertical angle (a picture shown on a curved screen spans less than a flat camera image would)."""
    h, w = illus.shape[:2]
    vfov = vfov_deg or 2 * math.degrees(math.atan(math.tan(math.radians(hfov_deg / 2)) * h / w))
    iw = int(round(pano_w * hfov_deg / 360))
    ih = min(int(round(pano_h * vfov / 180)), int(pano_h * 0.9))
    x0, y0 = (pano_w - iw) // 2, (pano_h - ih) // 2
    canvas = np.zeros((pano_h, pano_w, 3), np.uint8)
    canvas[y0:y0 + ih, x0:x0 + iw] = cv.resize(illus, (iw, ih), interpolation=cv.INTER_AREA)
    keep = np.zeros((pano_h, pano_w), np.uint8)
    keep[y0:y0 + ih, x0:x0 + iw] = 255 if matte is None else cv.resize(matte, (iw, ih), interpolation=cv.INTER_NEAREST)
    return canvas, keep, (x0, y0, x0 + iw, y0 + ih)


def keep_region(matte: np.ndarray) -> np.ndarray:
    """The part of the art the AI must never touch: the matte closed generously, so pale details that touch the
    paper (a white apron, a pale sky gap) are sealed in rather than repainted."""
    h, w = matte.shape
    k = max(9, min(h, w) // 10) | 1
    closed = cv.morphologyEx(matte, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(matte)
    cv.drawContours(out, contours, -1, 255, thickness=cv.FILLED)
    return out


AI_FEATHER = 64


def protect_mask(matte: np.ndarray, layer_masks: list) -> np.ndarray:
    """What the AI must not repaint: the art itself (fine matte, so pale paper gaps between trees are repainted)
    plus every character's filled outline, so pale details inside a figure (a white apron) stay real. The outline,
    not its convex hull: a hull would also lock in the blank paper between ears or beside an arm."""
    h, w = matte.shape
    k = max(5, min(h, w) // 40) | 1
    keep = cv.morphologyEx(matte, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k)))
    for m in layer_masks:
        m8 = cv.resize(m.astype(np.uint8) * 255, (w, h), interpolation=cv.INTER_NEAREST)
        contours, _ = cv.findContours(m8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(keep, contours, -1, 255, thickness=cv.FILLED)
    return keep


def style_terms(illus: np.ndarray, matte: np.ndarray) -> str:
    """Describe the art's palette from measurements, so the painted world matches it (pale stays pale)."""
    hsv = cv.cvtColor(illus, cv.COLOR_BGR2HSV)
    art = matte > 0
    light, sat = float(hsv[..., 2][art].mean()), float(hsv[..., 1][art].mean())
    words = ["pale airy light watercolour washes" if light > 175 else
             "mid-tone, gentle light" if light > 120 else "deep, low-key tones",
             "soft muted low-saturation colours" if sat < 70 else "warm natural colours" if sat < 130 else "rich vivid colours",
             "richly detailed, dense layered scenery from foreground to far distance"]
    return ", ".join(words)


FIGURES = {"person", "dog", "cat", "horse", "sheep", "cow", "teddy bear"}  # birds are fine as scenery
FIGURE_SCORE = 0.45
KEEP_OVERLAP = 0.6  # a figure box this much inside the kept art is the book's own character (those are ~1.0)
MAX_REPAINTS = 4


STRAY_TILES = (512, 1024)


def stray_figures(pano: np.ndarray, keep: np.ndarray, detect_fn) -> list[tuple]:
    """People or animals the painter added despite the prompt. The world must be scenery only: characters come from
    the book. Scans the horizon band in square tiles at two sizes (512 px keeps small distant figures visible to the
    detector; 1024 px sees big foreground figures whole instead of in pieces) and ignores the real art.
    Boxes are in panorama pixels; one that crosses the 0/360 join runs past the right edge (x1 > width)."""
    h, w = pano.shape[:2]
    band0, band1 = int(h * 0.2), int(h * 0.9)
    # Tiles run on past the right edge into a copy of the left edge, so a figure on the join is seen whole.
    reach = max(STRAY_TILES)
    ext, ext_keep = (np.concatenate([a, a[:, :reach]], axis=1) for a in (pano, keep))
    found = []
    for tile in STRAY_TILES:
        step = tile // 2  # half-overlapping, so a figure on a tile edge is seen whole in the next tile
        for ty in range(band0, max(band0 + 1, band1 - step), step):
            for tx in range(0, w, step):
                for d in detect_fn(ext[ty:ty + tile, tx:tx + tile]):
                    if d["name"] not in FIGURES or d["score"] < FIGURE_SCORE:
                        continue
                    bx0, by0, bx1, by1 = (int(v) for v in d["box"])
                    box = (max(0, tx + bx0), max(0, ty + by0), min(w + reach, tx + bx1), min(h, ty + by1))
                    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                        continue
                    if ext_keep[box[1]:box[3], box[0]:box[2]].mean() / 255 > KEEP_OVERLAP:  # the book's own
                        continue
                    if box[0] >= w:  # wholly in the copied strip: report it where it really is
                        box = (box[0] - w, box[1], box[2] - w, box[3])
                    found.append(box)
    return _merge_boxes(found)


def _merge_boxes(boxes: list[tuple]) -> list[tuple]:
    """Overlapping tiles report the same figure more than once; merge overlapping boxes so each is repainted once."""
    merged: list[list[int]] = []
    for b in sorted(boxes):
        for m in merged:
            if b[0] <= m[2] and b[2] >= m[0] and b[1] <= m[3] and b[3] >= m[1]:
                m[:] = [min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])]
                break
        else:
            merged.append(list(b))
    return [tuple(m) for m in merged]


def repaint_figures(pano: np.ndarray, boxes: list[tuple], paint_fn, crop: int = 768, pad: int = 40) -> int:
    """Repaint each stray figure as scenery inside its own local crop (a whole-panorama call works at low detail
    and can invent new figures). Returns how many were repainted."""
    h, w = pano.shape[:2]
    done = 0
    for x0, y0, x1, y1 in boxes[:MAX_REPAINTS]:
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        size = int(min(h, max(crop, (x1 - x0) * 2 + 2 * pad, (y1 - y0) * 2 + 2 * pad)))
        top = int(np.clip(cy - size // 2, 0, h - size))
        shift = w // 2 - cx
        rolled = np.roll(pano, shift, axis=1)
        left = w // 2 - size // 2
        region = rolled[top:top + size, left:left + size]
        mask = np.zeros(region.shape[:2], np.uint8)
        bx0, by0 = x0 + shift - left, y0 - top
        cv.rectangle(mask, (bx0 - pad, by0 - pad), (bx0 + (x1 - x0) + pad, by0 + (y1 - y0) + pad), 255, cv.FILLED)
        mask = cv.GaussianBlur(mask, (0, 0), pad / 2)
        out = paint_fn(region.copy(), mask)
        if out is None:
            break
        alpha = (mask.astype(np.float32) / 255)[..., None]
        rolled[top:top + size, left:left + size] = (out * alpha + region * (1 - alpha)).astype(np.uint8)
        pano[:] = np.roll(rolled, -shift, axis=1)
        done += 1
    return done


def colour_transfer(img: np.ndarray, ref: np.ndarray, region: np.ndarray, ref_mask: np.ndarray,
                    strength: float = 0.3) -> np.ndarray:
    """Reinhard colour transfer in Lab: give the generated region the illustration's colour statistics."""
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB).astype(np.float32)
    rlab = cv.cvtColor(ref, cv.COLOR_BGR2LAB).astype(np.float32)
    src_px, ref_px = lab[region > 0], rlab[ref_mask > 0]
    if len(src_px) < 100 or len(ref_px) < 100:
        return img
    s_mu, s_sd = src_px.mean(0), src_px.std(0) + 1e-6
    r_mu, r_sd = ref_px.mean(0), ref_px.std(0) + 1e-6
    moved = (lab - s_mu) / s_sd * r_sd + r_mu
    lab = np.where(region[..., None] > 0, lab * (1 - strength) + moved * strength, lab)
    return cv.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv.COLOR_LAB2BGR)


QUILT = 64


def scenery_quilt(illus: np.ndarray, matte: np.ndarray, layer_masks: list, box: tuple,
                  size: tuple = (AI_W, AI_H)) -> np.ndarray | None:
    """A colour guide with the art's own palette and textures but none of its structure: random blocks of the
    art's scenery (not its paper, not its characters) laid like a quilt, each row of the panorama drawing on the
    same height of the art, so sky colours stay up and ground colours down. The painter takes colour hints from
    the guide but also copies whatever structure it shows: a stretched, blurred mirror of the art reads as pale
    empty paper next to the art and comes back as an empty field. None if the art has too little scenery."""
    h, w = illus.shape[:2]
    scenery = matte > 0
    for m in layer_masks:
        scenery &= ~cv.resize(m.astype(np.uint8), (w, h), interpolation=cv.INTER_NEAREST).astype(bool)
    q = max(16, min(QUILT, min(h, w) // 6))
    blocks, rows = [], []
    for y in range(0, h - q + 1, q // 2):
        for x in range(0, w - q + 1, q // 2):
            if scenery[y:y + q, x:x + q].mean() > 0.9:
                blocks.append(cv.resize(illus[y:y + q, x:x + q], (QUILT, QUILT), interpolation=cv.INTER_LINEAR))
                rows.append((y + q / 2) / h)
    if len(blocks) < 8:
        return None
    rows = np.array(rows)
    out_w, out_h = size
    _, y0, _, y1 = box
    rng = np.random.default_rng(0)
    out = np.zeros((out_h, out_w, 3), np.uint8)
    for ty in range(0, out_h, QUILT):
        u = np.clip((ty + QUILT / 2 - y0) / max(1, y1 - y0), 0, 1)
        near = np.argsort(np.abs(rows - u))[: max(8, len(blocks) // 5)]
        for tx in range(0, out_w, QUILT):
            b = blocks[rng.choice(near)]
            if rng.random() < 0.5:
                b = b[:, ::-1]
            out[ty:ty + QUILT, tx:tx + QUILT] = b[: out_h - ty, : out_w - tx]
    return cv.GaussianBlur(out, (0, 0), QUILT / 10)


def ai_canvas(illus: np.ndarray, matte: np.ndarray, hfov_deg: float, layer_masks: list = (),
              vfov_deg: float | None = None) -> dict:
    """Canvas for generative painting: real art at the centre (unrolled, so the scenery grows continuously out of
    it), a colour guide in the art's own palette elsewhere, and the generate mask (255 = paint)."""
    layer_masks = list(layer_masks)
    protect = protect_mask(matte, layer_masks)
    clean = cv.inpaint(illus, 255 - keep_region(matte), 5, cv.INPAINT_TELEA)  # no white paper in the colour guide
    art, keep, box = place(illus, hfov_deg, AI_W, AI_H, protect, vfov_deg)
    guide = scenery_quilt(illus, matte, layer_masks, box)
    if guide is None:  # (nearly) all characters: fall back to the blurred wrap-around of the art itself
        guide = cv.resize(opencv_surround(clean, hfov_deg, blur=4, vfov_deg=vfov_deg)[0], (AI_W, AI_H),
                          interpolation=cv.INTER_CUBIC)
    # Grain on top: a smooth guide gets copied back as flat colour fields.
    noise = np.random.default_rng(0).normal(0, 10, guide.shape)
    guide = np.clip(guide + noise, 0, 255).astype(np.uint8)
    canvas = np.where(keep[..., None] > 0, art, guide)
    # Let the model paint a little way into the art so it blends into real pixels instead of drawing an outline
    # around the hole; the real art is pasted back over that overlap afterwards.
    generate = 255 - cv.erode(keep, cv.getStructuringElement(cv.MORPH_ELLIPSE, (31, 31)))
    # If painting stops early, the unpainted part gets the clean no-AI surround, never the noisy guide.
    fallback = cv.resize(opencv_surround(clean, hfov_deg, vfov_deg=vfov_deg)[0], (AI_W, AI_H),
                         interpolation=cv.INTER_CUBIC)
    return {"canvas": canvas, "generate": generate, "art": art, "keep": keep, "box": box, "fallback": fallback}


TILE_W, TILE_STEP = 1536, 768


def tile_order(w: int = AI_W) -> list[int]:
    """Tile centres for painting the circle: start at the illustration, alternate right and left, and finish with the
    tile opposite it, which sees both painted sides and closes the loop (no wrap-around seam)."""
    c = w // 2
    right = list(range(c + TILE_STEP, c + w // 2 + 1, TILE_STEP))
    left = list(range(c - TILE_STEP, c - w // 2 - 1, -TILE_STEP))
    order = [c]
    for i in range(max(len(right), len(left))):
        order += [x % w for x in (right[i:i + 1] + left[i:i + 1])]
    seen, out = set(), []
    for x in order:  # the last right tile and last left tile can land on the same spot; keep the first
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


TILE_EDGE = 224  # models draw borders/gutters at a crop's edge; that band is left for the next tile to repaint
BLEND = 48       # outward feather of each tile, in pixels
FLAT_BLUR, FLAT_STD, FLAT_WINDOW = 3.0, 3.0, 31
FLAT_MAX = 0.12  # a tile with one empty flat area bigger than this share of its new scenery is painted again
SKY_BAND = 0.4   # top fraction of the panorama where flat colour (a pale sky) is legitimate
ATTEMPTS = 3     # painter calls per tile at most (failed calls and empty tiles are tried again)


def flat_fraction(img: np.ndarray, region: np.ndarray) -> float:
    """The largest connected area of flat colour in `region` (bool), below the sky band, as a share of that region.
    The painter's failure mode is to hand back the colour guide as one empty field instead of scenery. Grain is
    blurred away first, so an empty field with paper texture still counts as empty, while good scenery keeps
    structure at that scale and shows only small, scattered calm spots (measured on the painter's raw tiles: at
    most 0.011 for good ones, 0.29 for a failed one; finished panoramas score higher after colour transfer)."""
    g = cv.GaussianBlur(cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32), (0, 0), FLAT_BLUR)
    mu = cv.blur(g, (FLAT_WINDOW, FLAT_WINDOW))
    sd = np.sqrt(np.maximum(cv.blur(g * g, (FLAT_WINDOW, FLAT_WINDOW)) - mu * mu, 0))
    scenery = region.copy()
    scenery[: int(SKY_BAND * img.shape[0])] = False
    if not scenery.any():
        return 0.0
    n, _, stats, _ = cv.connectedComponentsWithStats(((sd < FLAT_STD) & scenery).astype(np.uint8), connectivity=8)
    return float(stats[1:, cv.CC_STAT_AREA].max() / scenery.sum()) if n > 1 else 0.0


def _paint_tile(pano: np.ndarray, todo: np.ndarray, cx: int, paint_fn, keep_edges: bool, attempts: int,
                judge: bool, stats: dict) -> bool:
    w = pano.shape[1]
    shift = w // 2 - cx
    rp, rt = np.roll(pano, shift, axis=1), np.roll(todo, shift, axis=1)
    x0 = w // 2 - TILE_W // 2
    mask = rt[:, x0:x0 + TILE_W].copy()
    if mask.max() == 0:
        return True
    accept = np.ones(TILE_W, bool)
    if not keep_edges:  # only accept the tile's middle where unknown content lay beyond the crop edge
        for side in (slice(0, TILE_EDGE), slice(TILE_W - TILE_EDGE, TILE_W)):
            edge_col = 0 if side.start == 0 else TILE_W - 1
            if mask[:, edge_col].max() > 0:
                accept[side] = False
    own = (mask > 0) & accept[None, :]  # the pixels this tile finishes
    best = None
    for attempt in range(attempts):
        out = paint_fn(rp[:, x0:x0 + TILE_W].copy(), mask)
        if out is None:
            stats["failed_calls"] += 1
            continue
        stats["calls"] += 1
        flat = flat_fraction(out, own) if judge else 0.0
        if best is None or flat < best[0]:
            best = (flat, out)
        if flat <= FLAT_MAX:
            break
        stats["flat_tiles"] += 1
        log.info("tile at %d: %.0f%% of the new scenery is flat colour; asking again", cx, 100 * flat)
    if best is None:
        return False
    out = best[1]
    # Every pixel the tile finishes takes its paint at full strength, and the feather runs outward only, over pixels
    # that are already final (where the model kept them as they were) or still to do. A feather inside `own` would
    # leave a band of the colour guide showing through at every tile boundary.
    grown = cv.dilate(own.astype(np.uint8), cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * BLEND + 1, 2 * BLEND + 1)))
    alpha = np.maximum(cv.GaussianBlur(grown.astype(np.float32), (0, 0), BLEND / 3), own)[..., None]
    region = rp[:, x0:x0 + TILE_W].astype(np.float32)
    rp[:, x0:x0 + TILE_W] = (out * alpha + region * (1 - alpha)).astype(np.uint8)
    rt[:, x0:x0 + TILE_W][own] = 0
    pano[:] = np.roll(rp, -shift, axis=1)
    todo[:] = np.roll(rt, -shift, axis=1)
    return True


def paint_tiled(canvas: np.ndarray, generate: np.ndarray, paint_fn, max_extra: int = 4, attempts: int = ATTEMPTS,
                judge: bool = True) -> tuple[np.ndarray, dict, np.ndarray]:
    """Paint the panorama tile by tile so each ~135 degree view is generated at proper detail and continues what is
    already there. `paint_fn(crop, mask) -> painted crop or None`. A failed call, or (with `judge`) a tile that came
    back as flat empty colour, is tried again with the next call, up to `attempts` per tile; of several flat
    attempts the least flat is kept. Returns (panorama, stats, mask still unpainted: all zero unless painting
    stopped early)."""
    pano, todo = canvas.copy(), generate.copy()
    w = pano.shape[1]
    stats = {"calls": 0, "failed_calls": 0, "flat_tiles": 0}
    for cx in tile_order(w):
        if not _paint_tile(pano, todo, cx, paint_fn, False, attempts, judge, stats):
            return pano, stats, todo
    # Fill whatever edge bands are left, centred on them, keeping everything this time so the loop always ends.
    for _ in range(max_extra):
        cols = np.nonzero(todo.max(axis=0))[0]
        if len(cols) == 0:
            break
        # Centre on the remaining run (circular mean, since runs can straddle the 0/360 join).
        ang = cols / w * 2 * np.pi
        cx = int(round((np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) / (2 * np.pi) * w)) % w
        if not _paint_tile(pano, todo, cx, paint_fn, True, attempts, judge, stats):
            break
    return pano, stats, todo


def fill_unpainted(pano: np.ndarray, todo: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Where painting stopped early (a call failed twice), show OpenCV's clean wrap-around, not the noisy guide."""
    if not todo.any():
        return pano
    hard = (todo > 0).astype(np.float32)
    alpha = np.maximum(_wrap_blur(hard, BLEND / 3), hard)[..., None]
    return (fallback * alpha + pano * (1 - alpha)).astype(np.uint8)


def ai_finish(painted: np.ndarray, prep: dict) -> np.ndarray:
    """Match the painted scenery to the illustration's palette, paste the real art back pixel-for-pixel (feathered),
    and calm the poles, which a flat image would otherwise pinch into a point at the zenith and nadir."""
    generated = 255 - prep["keep"]
    painted = colour_transfer(painted, prep["art"], generated, prep["keep"])
    inside = cv.distanceTransform((prep["keep"] > 0).astype(np.uint8), cv.DIST_L2, 5)
    alpha = np.clip(inside / AI_FEATHER, 0, 1)[..., None]
    out = (prep["art"] * alpha + painted * (1 - alpha)).astype(np.float32)
    h = out.shape[0]
    t = np.linspace(0, 1, h)[:, None, None]
    pole = np.clip((np.abs(t - 0.5) - 0.38) / 0.12, 0, 1)
    sky = out[: h // 10].reshape(-1, 3).mean(0)
    ground = out[-h // 10:].reshape(-1, 3).mean(0)
    out = out * (1 - pole) + np.where(t < 0.5, sky, ground) * pole
    return out.astype(np.uint8)


def _wrap_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that wraps around horizontally (the panorama's left and right edges meet)."""
    pad = int(3 * sigma)
    tiled = np.concatenate([img[:, -pad:], img, img[:, :pad]], axis=1)
    return cv.GaussianBlur(tiled, (0, 0), sigma)[:, pad:-pad]


def opencv_surround(illus: np.ndarray, hfov_deg: float, blur: float = 12,
                    vfov_deg: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """No-AI surround: the illustration mirrored into a seamless loop, heavily blurred, fading to its sky and
    ground colours toward the poles, with the sharp illustration feathered in at the centre."""
    canvas, keep, _ = place(illus, hfov_deg, vfov_deg=vfov_deg)
    loop = np.hstack([illus, cv.flip(illus, 1)])  # mirrored: the loop's two ends match, so no seam
    base = cv.resize(loop, (PANO_W, PANO_H), interpolation=cv.INTER_LINEAR)
    base = np.roll(base, PANO_W // 4, axis=1)  # the unflipped half's centre now faces forward
    # Enough blur that mirrored characters don't read as giant duplicates, little enough to keep the scenery.
    base = _wrap_blur(base, blur).astype(np.float32)

    sky = illus[: max(2, illus.shape[0] // 12)].reshape(-1, 3).mean(0)
    ground = illus[-max(2, illus.shape[0] // 12):].reshape(-1, 3).mean(0)
    t = np.linspace(0, 1, PANO_H)[:, None, None]
    pole = np.clip((np.abs(t - 0.5) - 0.18) / 0.32, 0, 1)  # 0 near the horizon, 1 at the poles
    target = np.where(t < 0.5, sky, ground)
    base = base * (1 - pole) + target * pole
    return composite(base.astype(np.uint8), canvas, keep), keep


def composite(background: np.ndarray, canvas: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Feather the real illustration into the surround so there's no hard rectangle edge."""
    inside = cv.distanceTransform((keep > 0).astype(np.uint8), cv.DIST_L2, 5)
    alpha = np.clip(inside / FEATHER, 0, 1)[..., None]
    return (canvas * alpha + background * (1 - alpha)).astype(np.uint8)
