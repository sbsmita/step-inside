import cv2 as cv
import numpy as np

from .normalize import gray_world, norm_gray

THRESH = 0.22
MIN_AREA_FRAC = 0.002
KIND_MARGIN = 0.08
MOVE_MATCH_MIN = 0.55
MOVE_MIN_INLIERS = 8

_region_sift = cv.SIFT_create(nfeatures=600)
_region_matcher = cv.BFMatcher(cv.NORM_L2)


SCORE_SCALE = 0.5
CHROMA_SCALE = 0.3  # chromaticity distance that maps to score 1.0; tone curves shift dark saturated colors ~0.05
MIN_TEMPLATE_STD = 12.0
BRIGHT = 185


def contrast_structure_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SSIM without its luminance term: local brightness shifts (lamps, sunlight) don't count as change."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    c2 = (0.05 * 255) ** 2  # noise floor: near-flat walls/floors stay ~1 even if sensor noise differs

    def blur(x):
        return cv.GaussianBlur(x, (13, 13), 2.0)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    return (2 * cov + c2) / (var_a + var_b + c2)


def _chromaticity(img: np.ndarray) -> np.ndarray:
    """Normalized (b, g) chromaticity: unchanged when a lamp gets brighter or dimmer, unlike Lab a*/b*."""
    f = img.astype(np.float32) + 1.0
    return (f / f.sum(axis=2, keepdims=True))[..., :2]


def change_score(base: np.ndarray, warped: np.ndarray) -> np.ndarray:
    """Per-pixel change in [0, 1]: structural or chroma difference, after lighting normalization.

    Computed at reduced scale so sensor noise and resampling blur from the warp don't read as change."""
    h, w = base.shape[:2]

    def small(img):
        return cv.resize(img, None, fx=SCORE_SCALE, fy=SCORE_SCALE, interpolation=cv.INTER_AREA)

    rb, rw = small(base), small(warped)
    # The contrast-structure term is already gain/bias invariant, so it runs on white-balanced pixels only;
    # CLAHE here would amplify noise differently in flat tiles whenever brightness changes.
    nb, nw = gray_world(rb), gray_world(rw)
    structure = 1.0 - contrast_structure_map(cv.cvtColor(nb, cv.COLOR_BGR2GRAY), cv.cvtColor(nw, cv.COLOR_BGR2GRAY))
    diff = _chromaticity(nb) - _chromaticity(nw)
    diff -= np.median(diff.reshape(-1, 2), axis=0)  # residual global tint after white balance
    chroma = np.linalg.norm(diff, axis=2) / CHROMA_SCALE
    # Clipped or near-black pixels have meaningless hue; judged on raw pixels, before white balance rescales them.
    mx_b, mx_w = rb.max(axis=2), rw.max(axis=2)
    reliable = (mx_b < 250) & (mx_w < 250) & (mx_b > 25) & (mx_w > 25)
    chroma = cv.GaussianBlur(np.where(reliable, chroma, 0).astype(np.float32), (7, 7), 0)

    # Exposure clipping erases texture: a pixel clipped in one view that was already bright in the other is a
    # lighting artifact. A new white object on a dark surface fails the "already bright" test and still counts.
    clip_b, clip_w = mx_b >= 250, mx_w >= 250
    exposure = (clip_w & ~clip_b & (mx_b >= BRIGHT)) | (clip_b & ~clip_w & (mx_w >= BRIGHT))
    exposure = cv.dilate(exposure.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    structure = np.where(exposure, 0, structure)

    score = np.clip(np.maximum(structure, chroma), 0.0, 1.0)
    return cv.resize(score, (w, h), interpolation=cv.INTER_LINEAR)


def _distinctness(img: np.ndarray, lap: np.ndarray, bbox: tuple) -> float:
    """How much a region stands out from the ring around it (chroma histogram + texture). Objects stand out."""
    x, y, w, h = bbox
    X, Y, W, H = _pad(bbox, max(10, int(0.35 * max(w, h))), img.shape)
    inner = np.zeros(img.shape[:2], np.uint8)
    inner[y:y + h, x:x + w] = 255
    ring = np.zeros_like(inner)
    ring[Y:Y + H, X:X + W] = 255
    ring[inner > 0] = 0
    if cv.countNonZero(ring) < 50:
        return 0.0

    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    hists = []
    for m in (inner, ring):
        hist = cv.calcHist([lab], [1, 2], m, [16, 16], [0, 256, 0, 256])
        hists.append(cv.normalize(hist, hist, 1.0, 0.0, cv.NORM_L1))
    color = cv.compareHist(hists[0], hists[1], cv.HISTCMP_BHATTACHARYYA)
    texture = abs(np.log((lap[inner > 0].std() + 1.0) / (lap[ring > 0].std() + 1.0)))
    return float(color + 0.25 * texture)


def _kind(nb: np.ndarray, nw: np.ndarray, lap_b: np.ndarray, lap_w: np.ndarray, bbox: tuple) -> str:
    db, dw = _distinctness(nb, lap_b, bbox), _distinctness(nw, lap_w, bbox)
    if dw - db > KIND_MARGIN:
        return "appeared"
    if db - dw > KIND_MARGIN:
        return "disappeared"
    return "changed"


def _pad(bbox: tuple, pad: int, shape: tuple) -> tuple:
    x, y, w, h = bbox
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(shape[1], x + w + pad), min(shape[0], y + h + pad)
    return x0, y0, x1 - x0, y1 - y0


def _match(template: np.ndarray, window: np.ndarray) -> float:
    th, tw = template.shape[:2]
    wh, ww = window.shape[:2]
    if th > wh or tw > ww:
        s = min(wh / th, ww / tw)
        template = cv.resize(template, (max(1, int(tw * s)), max(1, int(th * s))))
    return float(cv.matchTemplate(window, template, cv.TM_CCOEFF_NORMED).max())


def _region_features(gray: np.ndarray, bbox: tuple):
    mask = np.zeros(gray.shape, np.uint8)
    x, y, w, h = _pad(bbox, 8, gray.shape)
    mask[y:y + h, x:x + w] = 255
    return _region_sift.detectAndCompute(gray, mask)


def _feature_inliers(fa, fb) -> int:
    """SIFT matches between two regions that agree on one similarity transform (ignores differing backgrounds)."""
    (ka, da), (kb, db) = fa, fb
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return 0
    good = [p[0] for p in _region_matcher.knnMatch(da, db, k=2)
            if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < 4:
        return 0
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    _, inliers = cv.estimateAffinePartial2D(src, dst, method=cv.RANSAC, ransacReprojThreshold=4.0)
    return int(inliers.sum()) if inliers is not None else 0


def _pair_moves(gb: np.ndarray, gw: np.ndarray, regions: list[dict]) -> list[dict]:
    """If what the baseline had at region A now shows up at region B, that's one moved object, not two changes.

    Matching is directional (baseline region A vs current region B), so the reverse pairing,
    background-vs-background, scores low and the move direction comes out right."""
    base_feats = [_region_features(gb, r["bbox"]) for r in regions]
    cur_feats = [_region_features(gw, r["bbox"]) for r in regions]
    candidates = []
    for i, src in enumerate(regions):
        if src["kind"] == "appeared":
            continue
        sx, sy, sw, sh = src["bbox"]
        template = gb[sy:sy + sh, sx:sx + sw]
        for j, dst in enumerate(regions):
            if i == j or dst["kind"] == "disappeared" or not 0.33 < dst["area"] / src["area"] < 3.0:
                continue
            inliers = _feature_inliers(base_feats[i], cur_feats[j])
            corr = 0.0
            if template.std() >= MIN_TEMPLATE_STD:
                x, y, w, h = _pad(dst["bbox"], 12, gw.shape)
                corr = _match(template, gw[y:y + h, x:x + w])
            if inliers >= MOVE_MIN_INLIERS or corr > MOVE_MATCH_MIN:
                candidates.append((min(inliers / 20.0, 1.0) + max(corr, 0.0), i, j))

    used = set()
    for score, i, j in sorted(candidates, reverse=True):
        if i in used or j in used:
            continue
        used.update((i, j))
        regions[j].update(kind="moved", from_bbox=regions[i]["bbox"], match=round(score, 3))
        regions[i]["kind"] = "_merged"
    return [r for r in regions if r["kind"] != "_merged"]


def detect(base: np.ndarray, warped: np.ndarray, valid: np.ndarray,
           thresh: float = THRESH, min_area_frac: float = MIN_AREA_FRAC) -> tuple[list[dict], np.ndarray]:
    score = change_score(base, warped)
    mask = ((score > thresh) & valid).astype(np.uint8) * 255
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (21, 21)))

    n, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    min_area = min_area_frac * h * w
    gb, gw = norm_gray(base), norm_gray(warped)
    nb, nw = gray_world(base), gray_world(warped)
    lap_b, lap_w = cv.Laplacian(gb, cv.CV_32F), cv.Laplacian(gw, cv.CV_32F)

    regions = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        region = labels == i
        contours, _ = cv.findContours(region.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        poly = cv.approxPolyDP(max(contours, key=cv.contourArea), 3.0, True).reshape(-1, 2)
        regions.append({
            "bbox": (x, y, bw, bh),
            "area": area,
            "area_frac": round(area / (h * w), 4),
            "centroid": (float(centroids[i][0]), float(centroids[i][1])),
            "score": round(float(score[region].mean()), 3),
            "kind": _kind(nb, nw, lap_b, lap_w, (x, y, bw, bh)),
            "polygon": poly.tolist(),
        })

    regions = _pair_moves(gb, gw, regions)
    regions.sort(key=lambda r: r["score"] * r["area"], reverse=True)
    return regions, score
