"""The actor agent: the character a moment is about, cut out of its painting so the theatre can make it move.

All OpenCV, a second or so per moment:
- find: YOLOX (cv.dnn) on the whole painting and on half-overlapping square tiles, plus Depth Anything V2 (cv.dnn)
  pop-outs for drawn characters a detector won't name (`scene25d.foreground_layers` refines both by depth and edges);
- judge: a cut-out that is too small, too big, ragged, mostly on the border or really the ground is refused, so
  the moment stays a living painting rather than half a tree walking off;
- cut: GrabCut sharpens the outline into a soft alpha matte, and xphoto's shift-map inpainting paints the scenery
  behind the character into a "plate", so when the cut-out moves away nothing is left behind."""
import re
from pathlib import Path

import cv2 as cv
import numpy as np

from . import depth
from .detect import detect
from .scene25d import foreground_layers
from .world import FIGURES

WORK_W = 1152            # half of a 2304 px painting: plenty for finding and outlining
LIVING = FIGURES | {"bird", "elephant", "bear", "zebra", "giraffe"}
FURNITURE = {"bench", "chair", "couch", "bed", "dining table", "potted plant", "tv", "toilet", "sink", "refrigerator",
             "oven", "microwave", "traffic light", "fire hydrant", "stop sign", "parking meter", "book", "vase"}
MIN_AREA, MAX_AREA = 0.004, 0.30   # share of the painting
MIN_SOLIDITY = 0.35                # area / convex hull area: a character, not a scatter of bits
MAX_BORDER = 0.25                  # share of the outline lying on the painting's edge
GROUND_WIDTH = 0.60                # wider than this and touching the bottom: it's the ground, not a character
MIN_HEIGHT = 0.15                  # a depth pop-out must be at least this tall to count as a character
# Words in the story's description of who moves -> the detector label that would find it. A subject none of these
# describe (an egg, a rabbit, a teapot) is looked for as the shape that stands out in depth instead.
LABEL_WORDS = {
    "person": ("person", "man", "woman", "boy", "girl", "child", "kid", "king", "queen", "prince", "princess", "lady",
               "farmer", "soldier", "knight", "baby", "giant", "witch", "wizard", "people", "mother", "father",
               "granny", "grandma", "grandpa", "jack", "jill", "maid", "sailor", "fisherman", "traveller", "traveler"),
    "dog": ("dog", "puppy", "hound"), "cat": ("cat", "kitten", "pussy"), "horse": ("horse", "pony", "foal"),
    "sheep": ("sheep", "lamb"), "cow": ("cow", "calf", "bull", "ox"), "elephant": ("elephant",), "bear": ("bear",),
    "bird": ("bird", "owl", "duck", "hen", "goose", "crow", "robin", "chick", "swan", "sparrow", "parrot", "eagle"),
    "zebra": ("zebra",), "giraffe": ("giraffe",), "teddy bear": ("teddy",),
}


def _tiled_detections(img: np.ndarray) -> list[dict]:
    """YOLOX on the whole picture and on square tiles (characters in a wide painting are small), merged by NMS."""
    h, w = img.shape[:2]
    dets = list(detect(img))
    for x in sorted({min(x, max(0, w - h)) for x in range(0, w, max(1, h // 2))}):
        for d in detect(img[:, x:x + h]):
            x0, y0, x1, y1 = d["box"]
            dets.append({**d, "box": (x0 + x, y0, x1 + x, y1)})
    if not dets:
        return []
    boxes = [[d["box"][0], d["box"][1], d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]] for d in dets]
    keep = cv.dnn.NMSBoxes(boxes, [d["score"] for d in dets], 0.3, 0.45)
    return [dets[i] for i in np.array(keep).ravel()]


def judge(mask: np.ndarray, label: str | None = None) -> dict:
    """Whether a cut-out mask is a usable character. Returns the measurements and the verdict with a reason."""
    h, w = mask.shape
    m8 = mask.astype(np.uint8)
    area = float(m8.mean())
    contours, _ = cv.findContours(m8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    hull = cv.convexHull(np.vstack(contours)) if contours else None
    solidity = float(m8.sum() / max(1.0, cv.contourArea(hull))) if hull is not None else 0.0
    pts = np.vstack(contours).reshape(-1, 2) if contours else np.zeros((0, 2), int)
    border = float(((pts[:, 0] <= 1) | (pts[:, 0] >= w - 2) | (pts[:, 1] <= 1) | (pts[:, 1] >= h - 2)).mean()) \
        if len(pts) else 1.0
    ys, xs = np.nonzero(m8)
    width = (xs.max() - xs.min() + 1) / w if len(xs) else 0.0
    ground = bool(len(ys) and ys.max() >= h - 2 and width > GROUND_WIDTH)
    reason = ("too small" if area < MIN_AREA else "too big" if area > MAX_AREA
              else "ragged outline" if solidity < MIN_SOLIDITY else "mostly on the edge" if border > MAX_BORDER
              else "it is the ground" if ground else None)
    return {"area": round(area, 4), "solidity": round(solidity, 3), "border": round(border, 3),
            "label": label, "ok": reason is None, "reason": reason}


def _filled(mask: np.ndarray) -> np.ndarray:
    """The outline filled in: a character taller than the depth window pops out only as a ring, and pale details
    inside a figure (a white apron) are part of it."""
    m8 = mask.astype(np.uint8)
    contours, _ = cv.findContours(m8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m8)
    cv.drawContours(out, contours, -1, 1, thickness=cv.FILLED)
    return out.astype(bool)


def wanted_labels(hint: str) -> set[str]:
    """Detector labels the story's words point to ("an old king" -> person; "a large egg" -> none)."""
    words = set(re.findall(r"[a-z]+", (hint or "").lower()))
    return {label for label, ws in LABEL_WORDS.items() if words & set(ws)}


def find_actor(work: np.ndarray, near: np.ndarray, dets: list[dict], hint: str = "") -> tuple[np.ndarray | None, dict]:
    """The one character to animate, chosen with the story's own description of who moves (`hint`): a detected
    person or animal of the kind it names; if it names none (an egg, a rabbit), the largest shape standing out in
    depth; with no description, a detected living thing, then a depth pop-out. Returns (mask or None, check)."""
    h = work.shape[0]
    layers = foreground_layers(work, near, dets)
    wanted = wanted_labels(hint)
    by_size = lambda ls: sorted(ls, key=lambda l: -l["mask"].sum())
    named = by_size(l for l in layers if l["source"] == "detector" and l["label"] in wanted)
    living = by_size(l for l in layers if l["source"] == "detector" and l["label"] in LIVING and l not in named)
    popouts = by_size(l for l in layers if l["source"] == "depth"
                      and np.ptp(np.nonzero(l["mask"])[0]) + 1 >= MIN_HEIGHT * h)
    # A subject the detector has no word for (an egg, a teapot) is often still found as SOME object ("clock"): the
    # largest standalone object or depth shape, people last.
    things = by_size(l for l in layers if (l["source"] == "depth" and l in popouts)
                     or (l["source"] == "detector" and l["label"] not in LIVING and l["label"] not in FURNITURE))
    order = named + living + popouts if wanted or not hint else things + living
    tried = []
    for layer in order:
        mask = _filled(layer["mask"])
        check = judge(mask, layer["label"] if layer["source"] == "detector" else "drawn figure")
        check["why"] = ("named by the story" if layer in named else "a detected figure" if layer["label"] in LIVING
                        else "stands out from the scene")
        check["source"] = layer["source"]
        if check["ok"]:
            if layer["source"] == "detector":  # the detector's box seeds GrabCut: the whole figure, feet included
                check["box"] = _box_for(mask, dets)
            return mask, check
        tried.append(check)
    reason = tried[0]["reason"] if tried else "nothing stands out"
    return None, {"ok": False, "reason": reason, "tried": len(tried)}


def _box_for(mask: np.ndarray, dets: list[dict]) -> list[float] | None:
    best, score = None, 0.0
    for d in dets:
        x0, y0, x1, y1 = (int(round(v)) for v in d["box"])
        inside = mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)].sum()
        if inside / max(1, mask.sum()) > score:
            best, score = list(d["box"]), inside / max(1, mask.sum())
    return best if score > 0.5 else None


def cut(img: np.ndarray, mask: np.ndarray, det_box: list | None = None, paint_fn=None,
        near: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(actor RGBA crop, plate, box [x0, y0, x1, y1]) at the painting's own resolution. `mask` may be smaller;
    `det_box` (same scale as `mask`) widens GrabCut's guess to the whole detected figure. `paint_fn(img, hole)`
    (an AI inpainter) paints the plate when given; else OpenCV's shift-map inpainting does."""
    H, W = img.shape[:2]
    s = W / mask.shape[1]
    full = cv.resize(mask.astype(np.uint8), (W, H), interpolation=cv.INTER_NEAREST)
    if det_box is not None:
        bx0, by0, bx1, by1 = (int(round(v * s)) for v in det_box)
        bw, bh = bx1 - bx0, by1 - by0  # a little room above and beside it: hats, crowns, ears, tails
        bx0, bx1, by0 = bx0 - bw // 20, bx1 + bw // 20, by0 - bh // 6
        guess = np.zeros_like(full)
        guess[max(0, by0):min(H, by1), max(0, bx0):min(W, bx1)] = 1
    else:
        guess = full
    ys, xs = np.nonzero(full | guess)
    pad = int(0.1 * max(xs.max() - xs.min(), ys.max() - ys.min())) + 4
    x0, y0, x1, y1 = max(0, xs.min() - pad), max(0, ys.min() - pad), min(W, xs.max() + pad + 1), min(H, ys.max() + pad + 1)
    crop, m, g = img[y0:y1, x0:x1], full[y0:y1, x0:x1], guess[y0:y1, x0:x1]

    # GrabCut, seeded by the mask: its core is surely the character, a ring around it probably background.
    k = max(3, (min(crop.shape[:2]) // 40) | 1)
    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
    gc = np.full(m.shape, cv.GC_BGD, np.uint8)
    gc[cv.dilate(m | g, ker, iterations=3) > 0] = cv.GC_PR_BGD
    gc[(m | g) > 0] = cv.GC_PR_FGD
    gc[cv.erode(m, ker, iterations=2) > 0] = cv.GC_FGD
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv.grabCut(crop, gc, None, bgd, fgd, 3, cv.GC_INIT_WITH_MASK)
        fg = ((gc == cv.GC_FGD) | (gc == cv.GC_PR_FGD)).astype(np.uint8)
    except cv.error:
        fg = m
    if fg.sum() < 0.3 * m.sum():  # GrabCut lost the character (flat colours): keep the depth outline
        fg = m
    if near is not None and (fg & ~m).any():
        # Beyond the first outline, keep only what is about as near as the character (its crown, not the archway
        # behind it), and only what still touches it.
        nc = cv.resize(near.astype(np.float32), (W, H), interpolation=cv.INTER_LINEAR)[y0:y1, x0:x1]
        core = float(np.median(nc[m > 0])) if m.any() else 0.0
        fg = (fg & (m | (nc >= core - 0.12))).astype(np.uint8)
        n, labels = cv.connectedComponents(fg)
        if n > 2:
            keep = np.unique(labels[(m > 0) & (fg > 0)])
            fg = np.isin(labels, keep[keep > 0]).astype(np.uint8)
    alpha = cv.GaussianBlur(fg.astype(np.float32) * 255, (0, 0), 1.5)
    actor = np.dstack([crop, alpha.clip(0, 255).astype(np.uint8)])

    hole = cv.dilate(fg * 255, ker, iterations=2)
    if paint_fn is not None:  # the plate painted by AI: the scenery that was behind the character
        whole = np.zeros((H, W), np.uint8)
        whole[y0:y1, x0:x1] = hole
        painted = paint_fn(img, cv.dilate(whole, ker, iterations=2))
        if painted is not None:
            a = cv.GaussianBlur(cv.dilate(whole, ker).astype(np.float32) / 255, (0, 0), 3)[..., None]
            return actor, (painted * a + img * (1 - a)).astype(np.uint8), [int(x0), int(y0), int(x1), int(y1)]
    # The plate: the scenery behind the character, rebuilt from patches of the painting (at half size, in a crop).
    small = cv.resize(crop, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    small_hole = cv.resize(hole, (small.shape[1], small.shape[0]), interpolation=cv.INTER_NEAREST)
    filled = np.zeros_like(small)
    try:
        cv.xphoto.inpaint(small, 255 - small_hole, filled, cv.xphoto.INPAINT_SHIFTMAP)
    except cv.error:
        filled = cv.inpaint(small, small_hole, 5, cv.INPAINT_TELEA)
    filled = cv.resize(filled, (crop.shape[1], crop.shape[0]), interpolation=cv.INTER_CUBIC)
    a = cv.GaussianBlur(hole.astype(np.float32) / 255, (0, 0), 2)[..., None]
    plate = img.copy()
    plate[y0:y1, x0:x1] = (filled * a + crop * (1 - a)).astype(np.uint8)
    return actor, plate, [int(x0), int(y0), int(x1), int(y1)]


FACE_MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"


def facing_from_landmarks(face: np.ndarray, turn: float = 0.18) -> str:
    """Which way a face is turned, from YuNet's landmarks (row: x, y, w, h, right eye x/y, left eye x/y, nose x/y,
    mouth corners, score): the nose sits off the eyes' midpoint toward the side the face is turned to."""
    eye_mid = (face[4] + face[6]) / 2
    eyes = abs(face[6] - face[4])
    off = (face[8] - eye_mid) / max(eyes, 1.0)
    return "right" if off > turn else "left" if off < -turn else "front"


def facing(actor: np.ndarray) -> str | None:
    """Which way the cut-out character looks on the screen ("left", "right", "front"), or None when OpenCV's YuNet
    face detector finds no face (animals and objects often have none). Used to mirror a character that walks the
    other way, so it doesn't walk backwards."""
    if not FACE_MODEL.exists():
        return None
    img = np.ascontiguousarray(actor[..., :3])
    s = 320 / max(img.shape[:2])
    if s > 1:
        img = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_CUBIC)
    h, w = img.shape[:2]
    try:
        det = cv.FaceDetectorYN.create(str(FACE_MODEL), "", (w, h), 0.6, 0.3, 5)
        _, faces = det.detect(img)
    except cv.error:
        return None
    if faces is None or not len(faces):
        return None
    return facing_from_landmarks(faces[int(np.argmax(faces[:, -1]))])


def make_actor(img: np.ndarray, hint: str = "", paint_fn=None) -> dict:
    """The whole agent for one painted moment. `hint`: the story's words for who moves (subject + their look).
    Returns {"check", and when ok: "actor" RGBA, "plate", "box"}."""
    s = WORK_W / img.shape[1]
    work = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else img
    near = depth.normalized(depth.inverse_depth(work))
    mask, check = find_actor(work, near, _tiled_detections(work), hint)
    if mask is None:
        return {"check": check}
    actor, plate, box = cut(img, mask, check.pop("box", None), paint_fn, near)
    check["plate"] = "ai" if paint_fn is not None else "opencv"
    check["facing"] = facing(actor)
    return {"check": check, "actor": actor, "plate": plate, "box": box}
