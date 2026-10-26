"""One photo -> depth-layered 3D diorama (2.5D): what's visible is real; hidden areas are filled and flagged."""
import time
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from . import camera, depth
from .clean import clean
from .detect import detect, spoken
from .mesh import grid_mesh, textured, to_glb
from .world import protect_mask

WORK_LONG_SIDE = 1280
GRID_STEP = 4
MAX_LAYERS = 6
MIN_LAYER_FRAC = 0.004
MAX_LAYER_FRAC = 0.6
POP_OUT = 0.12  # normalized inverse depth above the local background that counts as "in front"
FG_CUT_RATIO = 1.12


@dataclass
class Scene25D:
    glb: bytes
    meta: dict
    previews: dict = field(default_factory=dict)  # name -> BGR image, for the UI's "how it was built" panel
    layer_masks: list = field(default_factory=list)  # boolean masks (work resolution) of the cut-out layers


def _resize(img: np.ndarray) -> np.ndarray:
    s = WORK_LONG_SIDE / max(img.shape[:2])
    return cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else img


def _depth_mask(inv_n: np.ndarray, box: tuple) -> np.ndarray:
    """Object pixels inside a detector box: closer than the background on the *same image row*.

    Comparing per row keeps the floor out: ground near the camera is close too, but no closer than the
    rest of its own row, while a person or a ball stands out from it."""
    h, w = inv_n.shape
    x0, y0, x1, y1 = box
    bw = x1 - x0
    sx0, sx1 = max(0, x0 - bw // 2), min(w, x1 + bw // 2)
    side = np.concatenate([inv_n[y0:y1, sx0:x0], inv_n[y0:y1, x1:sx1]], axis=1)
    if side.shape[1] < 4:
        side = inv_n[y0:y1, :]
    row_bg = np.median(side, axis=1, keepdims=True)
    lift = inv_n[y0:y1, x0:x1] - row_bg
    margin = max(0.04, 0.35 * float(np.percentile(lift, 90)))
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = lift > margin
    return m


def _refine_edges(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """GrabCut only in a thin band around the depth mask: color snaps the outline, depth keeps the parts."""
    if mask.sum() < 50:
        return mask
    band = np.ones((7, 7), np.uint8)
    sure_fg = cv.erode(mask.astype(np.uint8), band).astype(bool)
    maybe = cv.dilate(mask.astype(np.uint8), band).astype(bool)
    gc = np.full(mask.shape, cv.GC_BGD, np.uint8)
    gc[maybe] = cv.GC_PR_BGD
    gc[mask] = cv.GC_PR_FGD
    gc[sure_fg] = cv.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv.grabCut(img, gc, None, bgd, fgd, 2, cv.GC_INIT_WITH_MASK)
    except cv.error:
        return mask
    return (gc == cv.GC_FGD) | (gc == cv.GC_PR_FGD)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    m = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    m = cv.morphologyEx(m, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9)))
    n, labels, stats, _ = cv.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m.astype(bool)
    keep = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    return labels == keep


def foreground_layers(img: np.ndarray, inv_n: np.ndarray, dets: list[dict]) -> list[dict]:
    """Things standing in front of the background: detected objects plus anything that pops out in depth."""
    h, w = inv_n.shape
    area = h * w
    layers = []
    for d in dets:
        x0, y0, x1, y1 = (int(round(v)) for v in d["box"])
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if (x1 - x0) * (y1 - y0) < MIN_LAYER_FRAC * area or (x1 - x0) * (y1 - y0) > MAX_LAYER_FRAC * area:
            continue
        m = _clean_mask(_refine_edges(img, _clean_mask(_depth_mask(inv_n, (x0, y0, x1, y1)))))
        if m.sum() >= MIN_LAYER_FRAC * area:
            layers.append({"mask": m, "label": spoken(d["name"]), "score": d["score"], "source": "detector"})

    # Undetected things in front (e.g. drawn characters a detector won't recognise): well above the large-scale
    # background depth. The window must be wider than a character, or its own depth becomes the "background".
    k = max(31, (min(h, w) // 3) | 1)
    background = cv.medianBlur((inv_n * 255).astype(np.uint8), min(k, 255)).astype(np.float32) / 255
    pop = (inv_n - background) > POP_OUT
    taken = np.zeros_like(pop)
    for layer in layers:
        taken |= layer["mask"]
    pop &= ~cv.dilate(taken.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    n, labels, stats, _ = cv.connectedComponentsWithStats(pop.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if MIN_LAYER_FRAC * area * 3 <= stats[i, cv.CC_STAT_AREA] <= MAX_LAYER_FRAC * area:
            m = _clean_mask(labels == i)  # thin structures (branches, a fence) can vanish in the clean-up
            if m.sum() >= MIN_LAYER_FRAC * area:
                layers.append({"mask": m, "label": "something", "score": None, "source": "depth"})

    for layer in layers:
        layer["closeness"] = float(np.median(inv_n[layer["mask"]]))
    layers.sort(key=lambda l: -l["closeness"])
    return layers[:MAX_LAYERS]


def fill_behind(img: np.ndarray, depth_m: np.ndarray, hole: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inpaint color and depth behind the cut-out layers (at half resolution for speed, then pasted back)."""
    hole_u8 = cv.dilate(hole.astype(np.uint8) * 255, np.ones((11, 11), np.uint8))
    h, w = hole.shape
    small = lambda a, interp: cv.resize(a, (w // 2, h // 2), interpolation=interp)
    small_hole = (small(hole_u8, cv.INTER_NEAREST) > 0).astype(np.uint8) * 255
    color = cv.inpaint(small(img, cv.INTER_AREA), small_hole, 7, cv.INPAINT_TELEA)
    # Behind an object the surface continues the surroundings, so inpaint depth the same way.
    dep = cv.inpaint(small(depth_m, cv.INTER_AREA).astype(np.float32), small_hole, 7, cv.INPAINT_TELEA)
    color_full = img.copy()
    depth_full = depth_m.copy()
    sel = hole_u8 > 0
    color_full[sel] = cv.resize(color, (w, h), interpolation=cv.INTER_LINEAR)[sel]
    depth_full[sel] = cv.resize(dep, (w, h), interpolation=cv.INTER_LINEAR)[sel]
    return color_full, depth_full


def build(img: np.ndarray, kind: str = "photo", matte: np.ndarray | None = None) -> Scene25D:
    """`matte` (art = 255) makes the blank paper around a book illustration transparent, so the world shows through."""
    timings = {}
    t = time.perf_counter()

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        timings[name] = round((now - t) * 1000)
        t = now

    cleaned, steps = clean(img, kind)
    work = _resize(cleaned)
    h, w = work.shape[:2]
    lap("clean")

    inv_n = depth.normalized(depth.inverse_depth(work))
    lap("depth")
    dets = detect(work)
    lap("detect")

    cam = camera.estimate_hfov(work)
    f = camera.focal_px(w, cam["hfov_deg"])
    outdoor = camera.looks_outdoor(work, inv_n)
    scale = camera.estimate_scale(inv_n, dets, f, outdoor)
    depth_m = depth.to_metric(inv_n, scale["near_m"], scale["far_m"]).astype(np.float32)
    lap("camera")

    layers = foreground_layers(work, inv_n, dets)
    hole = np.zeros((h, w), bool)
    for layer in layers:
        hole |= layer["mask"]
    bg_color, bg_depth = fill_behind(work, depth_m, hole)
    lap("layers")

    cx, cy = w / 2, h / 2
    rgb = cv.cvtColor(work, cv.COLOR_BGR2RGB)
    bg_rgb = cv.cvtColor(bg_color, cv.COLOR_BGR2RGB)
    meshes = {}
    v, fcs, uv = grid_mesh(bg_depth, f, cx, cy, (0, 0, w, h), GRID_STEP)
    if matte is not None:
        # Same outline the world painter keeps: the art plus each character's silhouette, so no blank paper is left
        # standing in front of the painted world.
        art = protect_mask(cv.resize(matte, (w, h), interpolation=cv.INTER_NEAREST), [l["mask"] for l in layers])
        bg_rgb = np.dstack([bg_rgb, art])
    meshes["background"] = textured(v, fcs, uv, bg_rgb, "background")

    layer_meta = []
    for i, layer in enumerate(layers):
        ys, xs = np.nonzero(layer["mask"])
        pad = GRID_STEP * 2
        box = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(w, xs.max() + pad + 1), min(h, ys.max() + pad + 1))
        x0, y0, x1, y1 = box
        step = max(2, GRID_STEP // 2)
        grown = cv.dilate(layer["mask"].astype(np.uint8), np.ones((2 * step + 1, 2 * step + 1), np.uint8)).astype(bool)
        # Edge vertices just outside the object take the object's own depth (not the wall behind it),
        # so the alpha-cut outline doesn't stretch backwards.
        band = (grown & ~layer["mask"]).astype(np.uint8) * 255
        layer_depth = cv.inpaint(depth_m, band, 3, cv.INPAINT_TELEA)
        v, fcs, uv = grid_mesh(layer_depth, f, cx, cy, box, step, mask=layer["mask"], cut_ratio=FG_CUT_RATIO,
                               any_corner=True)
        if len(fcs) == 0:
            continue
        crop = rgb[y0:y1, x0:x1]
        alpha = (layer["mask"][y0:y1, x0:x1] * 255).astype(np.uint8)
        name = f"layer{i}_{layer['label'].replace(' ', '_')}"
        meshes[name] = textured(v, fcs, uv, np.dstack([crop, alpha]), name)
        layer_meta.append({"name": name, "label": layer["label"], "source": layer["source"],
                           "distance_m": round(float(np.median(depth_m[layer["mask"]])), 2),
                           "area_frac": round(float(layer["mask"].mean()), 4)})
    glb = to_glb(meshes)
    lap("mesh")

    near = scale["near_m"]
    meta = {
        "type": "2.5d",
        "image": {"w": w, "h": h},
        "camera": {**cam, "focal_px": round(f, 1), "position": [0, 0, 0], "look": [0, 0, -1]},
        "scale": scale,
        "outdoor": outdoor,
        "bounds": {"move_radius_m": round(0.35 * near, 3), "forward_m": round(0.5 * near, 3)},
        "layers": layer_meta,
        "detections": [{"label": spoken(d["name"]), "score": d["score"]} for d in dets],
        "filled": {"method": "opencv_inpaint_telea", "area_frac": round(float(hole.mean()), 4)},
        "clean_steps": steps,
        "timings_ms": timings,
    }
    overlay = work.copy()
    palette = [(66, 135, 245), (245, 166, 35), (80, 200, 120), (230, 80, 80), (180, 100, 220), (40, 200, 200)]
    for i, layer in enumerate(layers):
        overlay[layer["mask"]] = (0.45 * overlay[layer["mask"]] + 0.55 * np.array(palette[i % 6])).astype(np.uint8)
    previews = {
        "depth": cv.applyColorMap((inv_n * 255).astype(np.uint8), cv.COLORMAP_INFERNO),
        "layers": overlay,
        "filled": bg_color,
    }
    return Scene25D(glb=glb, meta=meta, previews=previews, layer_masks=[l["mask"] for l in layers])
