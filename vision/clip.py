"""The clip critic: OpenCV watches every generated video clip before a viewer sees it.

A clip is kept only if it is still the moment's painting, letters and invents nothing, holds the camera still, and
the subject moved the way the story said. Every check is plain OpenCV:
- fidelity: ORB features + RANSAC homography between the painting and the clip's first frame, and SSIM (cv.quality);
- lettering: PP-OCRv3 text detection (cv.dnn), the same check the paintings get;
- new figures: YOLOX (cv.dnn) people/animal counts against the painting;
- motion: Farneback optical flow (cv.video), following the subject's box frame to frame: how far and which way it
  went, how much the background moved (a moving camera), and how often it reversed (a sway);
- cuts: frame-to-frame SSIM and colour histogram distance (Bhattacharyya);
- loop point: the late frame that best matches the first, so a swaying clip can loop without a jump.
It cannot tell WHO moved, only that the subject region moved as described.

    python -m vision.clip still.jpg clip.mp4 --path fall    prints every score (for calibrating the thresholds)"""
import json
import os
import sys
import tempfile
from dataclasses import dataclass

import cv2 as cv
import numpy as np

SAMPLE_HZ = 6.0
FLOW_W = 480
SSIM_W = 320
T = dict(fidelity_inliers=40, fidelity_share=0.2, fidelity_ssim=0.8, text_limit=0.004, new_figures=3, figure_score=0.45,
         min_subject_speed=0.02, frozen_speed=0.01, net_move=0.08, wrong_way=0.05, jump_lift=0.05, sway_drift=0.05, sway_reversals=2,
         camera_speed=0.03, camera_ratio=0.6, cut_ssim=0.45, cut_drop=0.3, cut_hist=0.35,
         loop_diff=10.0, loop_ssim=0.75, loop_min_s=2.0, continuity_inliers=40)
FIGURE_NAMES = {"person", "dog", "cat", "horse", "sheep", "cow", "teddy bear", "bird", "elephant", "bear"}


@dataclass
class Frames:
    small: list            # grey frames at FLOW_W, sampled at SAMPLE_HZ
    colour: list           # the same frames in colour
    times: list            # seconds
    full: list             # a few full-resolution frames (lettering, figures)
    fps: float
    count: int
    size: tuple            # (w, h)
    duration: float


def read_clip(data: bytes, hz: float = SAMPLE_HZ, full_at: int = 5) -> Frames:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        cap = cv.VideoCapture(path)
        fps = cap.get(cv.CAP_PROP_FPS) or 24.0
        count = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
        every = max(1, round(fps / hz))
        want_full = set(np.linspace(0, max(0, count - 1), full_at).astype(int)) if count else set()
        small, colour, times, full = [], [], [], []
        i, size = 0, (0, 0)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            size = (f.shape[1], f.shape[0])
            if i % every == 0:
                s = FLOW_W / f.shape[1]
                c = cv.resize(f, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
                colour.append(c)
                small.append(cv.cvtColor(c, cv.COLOR_BGR2GRAY))
                times.append(i / fps)
            if i in want_full:
                full.append(f)
            i += 1
        cap.release()
    finally:
        os.unlink(path)
    if not full and colour:
        full = [colour[0], colour[-1]]
    return Frames(small, colour, times, full, fps, i, size, i / fps if fps else 0.0)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    w = SSIM_W
    a = cv.resize(a, (w, round(a.shape[0] * w / a.shape[1])), interpolation=cv.INTER_AREA)
    b = cv.resize(b, (a.shape[1], a.shape[0]), interpolation=cv.INTER_AREA)
    if a.ndim == 3:
        a, b = cv.cvtColor(a, cv.COLOR_BGR2GRAY), cv.cvtColor(b, cv.COLOR_BGR2GRAY)
    try:
        return float(cv.quality.QualitySSIM_compute(a, b)[0][0])
    except (AttributeError, cv.error):  # without contrib: the textbook formula with a Gaussian window
        a, b = a.astype(np.float64), b.astype(np.float64)
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        blur = lambda x: cv.GaussianBlur(x, (11, 11), 1.5)
        ma, mb = blur(a), blur(b)
        va, vb, cov = blur(a * a) - ma ** 2, blur(b * b) - mb ** 2, blur(a * b) - ma * mb
        return float((((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())


def _inliers(a: np.ndarray, b: np.ndarray) -> int:
    orb = cv.ORB_create(1500)
    ka, da = orb.detectAndCompute(cv.cvtColor(a, cv.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv.cvtColor(b, cv.COLOR_BGR2GRAY), None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    pairs = cv.BFMatcher(cv.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return 0
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    _, mask = cv.findHomography(src, dst, cv.RANSAC, 6.0)
    return int(mask.sum()) if mask is not None else 0


def _pair(still: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The painting and a clip frame at the same working size."""
    frame = cv.resize(frame, (FLOW_W * 2, round(frame.shape[0] * FLOW_W * 2 / frame.shape[1])))
    return cv.resize(still, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_AREA), frame


def fidelity(still: np.ndarray, first: np.ndarray) -> dict:
    """Is the clip still the painting? ORB + RANSAC inliers between the painting and the first frame, measured
    against how many the painting finds in a re-encoded copy of itself (paintings differ in how many corners they
    have); SSIM only decides for a painting too smooth for features (SSIM alone calls flat pictures alike)."""
    first = cv.resize(first, (FLOW_W * 2, round(first.shape[0] * FLOW_W * 2 / first.shape[1])))
    ref = cv.resize(still, (first.shape[1], first.shape[0]), interpolation=cv.INTER_AREA)
    copy = cv.imdecode(cv.imencode(".jpg", cv.GaussianBlur(ref, (0, 0), 0.8), [cv.IMWRITE_JPEG_QUALITY, 70])[1], 1)
    inliers, own = _inliers(ref, first), _inliers(ref, copy)
    s = round(ssim(ref, first), 3)
    if own >= T["fidelity_inliers"]:
        ok = inliers >= max(T["fidelity_inliers"], T["fidelity_share"] * own)
    else:
        ok = s >= T["fidelity_ssim"]
    return {"inliers": inliers, "own": own, "ssim": s, "ok": bool(ok)}


def lettering(full: list) -> float:
    from .shot import text_cover
    return max((text_cover(f) for f in full), default=0.0)


def figures(img: np.ndarray) -> int:
    from .detect import detect
    return sum(1 for d in detect(img) if d["name"] in FIGURE_NAMES and d["score"] >= T["figure_score"])


def _flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)


def flows(fr: Frames) -> list:
    return [_flow(a, b) for a, b in zip(fr.small, fr.small[1:])]


def subject_region(fr: Frames, fl: list, actor_box: list | None, still_size: tuple | None) -> np.ndarray:
    """Where the subject can be during the clip, as a mask: around the actor's box (scaled to the clip and grown by
    its own size each way, since a falling or walking character leaves its box), else around the largest region
    that moves during the clip, else the centre."""
    h, w = fr.small[0].shape
    region = np.zeros((h, w), bool)
    if actor_box and still_size:
        sx, sy = w / still_size[0], h / still_size[1]
        x0, y0, x1, y1 = actor_box[0] * sx, actor_box[1] * sy, actor_box[2] * sx, actor_box[3] * sy
        bw, bh = x1 - x0, y1 - y0
        region[int(max(0, y0 - bh)):int(min(h, y1 + bh)), int(max(0, x0 - bw)):int(min(w, x1 + bw))] = True
        return region
    mag = sum(np.hypot(f[..., 0], f[..., 1]) for f in fl)
    moving = (mag > 3 * np.median(mag) + 0.5 * len(fl) ** 0.5).astype(np.uint8)
    moving = cv.morphologyEx(moving, cv.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv.connectedComponentsWithStats(moving)
    if n > 1:
        i = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
        if stats[i, cv.CC_STAT_AREA] >= 0.002 * w * h:
            return cv.dilate((labels == i).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    region[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
    return region


def track(fr: Frames, fl: list, region: np.ndarray) -> dict:
    """How the subject moved: per step, the mean optical flow of the pixels that move inside the subject's region
    (a character crossing a still background), and the background's own flow outside it (a moving camera)."""
    h, w = fr.small[0].shape
    dxs, dys, subj, back = [], [], [], []
    for f in fl:
        mag = np.hypot(f[..., 0], f[..., 1])
        outside = mag[~region]
        speed_out = float(np.median(outside)) if outside.size else 0.0
        moving = region & (mag > max(0.4, 3 * speed_out))
        if moving.sum() >= 20:
            mx, my = f[moving].mean(0)
            speed_in = float(np.percentile(mag[moving], 75))
        else:
            mx = my = speed_in = 0.0
        dxs.append(mx / w)
        dys.append(my / h)
        subj.append(speed_in / h)
        back.append(speed_out / h)
    step = 1.0 / SAMPLE_HZ
    cum_y = np.cumsum(dys) if dys else np.zeros(1)
    signs = np.sign([d for d in dxs if abs(d) > 0.002]) if dxs else []
    reversals = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0
    return {"dx": round(float(np.sum(dxs)), 4), "dy": round(float(np.sum(dys)), 4),
            "subject_speed": round(float(np.mean(subj) / step) if subj else 0.0, 4),
            "background_speed": round(float(np.median(back) / step) if back else 0.0, 4),
            "peak_lift": round(float(max(0.0, -cum_y.min())), 4), "reversals": reversals}


def cuts(fr: Frames) -> int:
    scores, hists = [], []
    for a, b in zip(fr.colour, fr.colour[1:]):
        scores.append(ssim(a, b))
        ha, hb = (cv.calcHist([cv.cvtColor(x, cv.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256]) for x in (a, b))
        cv.normalize(ha, ha)
        cv.normalize(hb, hb)
        hists.append(cv.compareHist(ha, hb, cv.HISTCMP_BHATTACHARYYA))
    if not scores:
        return 0
    med = float(np.median(scores))
    return int(sum(1 for s, d in zip(scores, hists) if (s < T["cut_ssim"] and s < med - T["cut_drop"]) or d > T["cut_hist"]))


def loop_point(fr: Frames) -> dict:
    first = fr.small[0].astype(np.float32)
    best = {"at": None, "diff": None, "ssim": None}
    start = max(T["loop_min_s"], 0.6 * fr.duration)
    for f, t in zip(fr.small, fr.times):
        if t < start:
            continue
        d = float(np.abs(f.astype(np.float32) - first).mean())
        if best["diff"] is None or d < best["diff"]:
            best = {"at": round(t, 2), "diff": round(d, 2), "ssim": round(ssim(fr.small[0], f), 3)}
    return best


def classify(path: str, t: dict) -> tuple[str, bool, list[str]]:
    """(what was seen, worth another try, reasons to reject)."""
    reasons, observed = [], "unclear"
    if t["subject_speed"] < T["frozen_speed"]:  # a clip in which nothing moves is not worth showing over the painting
        return "nothing", True, ["nothing happens"]
    if path in (None, "", "still"):
        return "still", False, []
    if t["subject_speed"] < T["min_subject_speed"]:
        return "nothing", True, ["nothing happens"]
    if t["background_speed"] >= T["camera_speed"] and t["background_speed"] >= T["camera_ratio"] * t["subject_speed"]:
        reasons.append("the camera moved")
    dx, dy = t["dx"], t["dy"]
    if path == "fall":
        observed = "fall" if dy >= T["net_move"] else "rise" if dy <= -T["wrong_way"] else "unclear"
        if observed == "rise":
            reasons.append("it went up instead of falling")
    elif path == "rise":
        observed = "rise" if dy <= -T["net_move"] else "fall" if dy >= T["wrong_way"] else "unclear"
        if observed == "fall":
            reasons.append("it went down instead of rising")
    elif path in ("walk-left", "walk-right"):
        # The story agent's left/right is a guess from the words: a character walking the other way still acts the
        # line. Report the direction seen; never reject for it.
        observed = ("walk-right" if dx >= T["net_move"] else "walk-left" if dx <= -T["net_move"] else "unclear")
    elif path == "jump":
        observed = "jump" if t["peak_lift"] >= T["jump_lift"] and abs(dy) < 0.04 else "unclear"
    elif path == "sway":
        ok = abs(dx) < T["sway_drift"] and abs(dy) < T["sway_drift"] and t["reversals"] >= T["sway_reversals"]
        observed = "sway" if ok else "unclear"
    return observed, bool(reasons) or observed == "unclear", reasons


def verify(still: np.ndarray, mp4: bytes, action: dict, actor_box: list | None = None,
           from_still: bool = True) -> dict:
    """The verdict on one clip; `ok` means show it, `retry` means another try is worth paying for."""
    import time
    t0 = time.monotonic()
    fr = read_clip(mp4)
    if len(fr.small) < 3:
        return {"ok": False, "retry": True, "reasons": ["the clip could not be read"], "observed": "unclear",
                "scores": {}, "loop_at": None, "hold": True, "frames": fr.count, "fps": fr.fps,
                "duration": fr.duration, "size": list(fr.size), "seconds": round(time.monotonic() - t0, 2)}
    path = action.get("path", "still")
    reasons = []
    fid = fidelity(still, fr.colour[0]) if from_still else {"inliers": None, "own": None, "ssim": None, "ok": True}
    if not fid["ok"]:
        reasons.append("it does not start from the painting")
    # Frame 0 is the painting itself (the video model is handed it), so it always matches. A clip that morphs into
    # another scene, or that something fills, has lost the painting by its middle or end: real clips that did that
    # kept 4-14 matches; the weakest clip that still looked right kept 58.
    later = [_inliers(*_pair(still, fr.colour[i])) for i in (len(fr.colour) // 2, -1)] if from_still else []
    if later and min(later) < T["continuity_inliers"]:
        reasons.append("it turns into a different scene")
    text = lettering(fr.full)
    if text > T["text_limit"]:
        reasons.append("letters appeared in the picture")
    fig_still = figures(still)
    fig_max = max((figures(f) for f in fr.full), default=0)
    fl = flows(fr)
    region = subject_region(fr, fl, actor_box, (still.shape[1], still.shape[0]) if actor_box else None)
    t = track(fr, fl, region)
    observed, retry, motion_reasons = classify(path, t)
    reasons += motion_reasons
    n_cuts = cuts(fr)
    if n_cuts:
        reasons.append("it jumps to a different shot")
    # A zoom enlarges painted figures and YOLOX then counts more of them, and cartoon crowds count unreliably: extra
    # figures reject a clip only when the camera held still (so they were really invented). Otherwise just recorded.
    camera_moved = n_cuts > 0 or "the camera moved" in motion_reasons
    if fig_max - fig_still >= T["new_figures"] and not camera_moved:
        reasons.append("new figures appeared")
    lp = loop_point(fr)
    loops = lp["at"] is not None and ((lp["diff"] is not None and lp["diff"] <= T["loop_diff"])
                                      or (lp["ssim"] or 0) >= T["loop_ssim"])
    if not action.get("loop", path in ("sway", "still")):
        loops = False  # a fall ends somewhere else: hold its last frame rather than replay it
    ok = not reasons
    return {"ok": ok, "retry": (not ok) or retry, "reasons": reasons, "observed": observed,
            "scores": {"fidelity_inliers": fid["inliers"], "continuity_inliers": min(later) if later else None, "fidelity_own": fid["own"], "fidelity_ssim": fid["ssim"],
                       "text_cover": round(text, 4),
                       "figures_still": fig_still, "figures_max": fig_max, **t, "cuts": n_cuts,
                       "loop_diff": lp["diff"], "loop_ssim": lp["ssim"]},
            "loop_at": lp["at"] if loops else None, "hold": not loops, "frames": fr.count, "fps": round(fr.fps, 2),
            "duration": round(fr.duration, 2), "size": list(fr.size), "seconds": round(time.monotonic() - t0, 2)}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python -m vision.clip still.jpg clip.mp4 [--path fall|rise|walk-left|walk-right|jump|sway]")
    p = sys.argv[sys.argv.index("--path") + 1] if "--path" in sys.argv else "still"
    with open(sys.argv[2], "rb") as fh:
        print(json.dumps(verify(cv.imread(sys.argv[1]), fh.read(), {"path": p}), indent=1))
