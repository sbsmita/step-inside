"""Capture Coach: live active perception while recording. Decides which frames become keyframes and tells the
user how to move so the reconstruction will succeed (enough parallax, sharp frames, steady tracking)."""
import time
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from ..quality import assess

WORK_WIDTH = 640
KEYFRAME_FLOW = 0.035   # fraction of width tracked features must move before a new keyframe
MIN_TRACKED = 0.35      # below this fraction of features surviving, tracking is lost
ROTATION_RATIO = 0.92   # homography explaining this share of motion = turning on the spot, no parallax
MIN_PARALLAX_PX = 1.2   # median motion a homography can't explain (true 3D parallax)
TARGET_KEYFRAMES = 18


@dataclass
class CoachSession:
    ref_gray: np.ndarray | None = None
    ref_pts: np.ndarray | None = None
    keyframes: int = 0
    rotation_streak: int = 0
    directions: list = field(default_factory=list)
    started: float = field(default_factory=time.time)


def _gray(img: np.ndarray) -> np.ndarray:
    s = WORK_WIDTH / img.shape[1]
    return cv.cvtColor(cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA), cv.COLOR_BGR2GRAY)


def _set_ref(s: CoachSession, gray: np.ndarray) -> None:
    s.ref_gray = gray
    s.ref_pts = cv.goodFeaturesToTrack(gray, 500, 0.01, 8)


def step(s: CoachSession, frame: np.ndarray) -> dict:
    q = assess(frame)
    if "too_dark" in q["issues"]:
        return {"status": "warn", "say": "It's too dark. Turn on a light.", "keyframe": False, "keyframes": s.keyframes}
    if "blurry" in q["issues"]:
        return {"status": "warn", "say": "Hold steadier, the picture is blurry.", "keyframe": False, "keyframes": s.keyframes}

    gray = _gray(frame)
    if s.ref_gray is None or s.ref_pts is None or len(s.ref_pts) < 40:
        _set_ref(s, gray)
        if s.ref_pts is None or len(s.ref_pts) < 40:
            return {"status": "warn", "say": "Point at something with more detail.", "keyframe": False, "keyframes": s.keyframes}
        s.keyframes += 1
        return {"status": "ok", "say": "Good. Now slide the phone slowly to one side.", "keyframe": True, "keyframes": s.keyframes}

    nxt, st, _ = cv.calcOpticalFlowPyrLK(s.ref_gray, gray, s.ref_pts, None, winSize=(21, 21), maxLevel=4)
    ok = st.ravel() == 1
    if ok.mean() < MIN_TRACKED or ok.sum() < 30:
        _set_ref(s, gray)
        return {"status": "warn", "say": "Too fast, I lost track. Slow down.", "keyframe": False, "keyframes": s.keyframes}

    p0 = s.ref_pts.reshape(-1, 2)[ok]
    p1 = nxt.reshape(-1, 2)[ok]
    flow = p1 - p0
    moved = float(np.median(np.linalg.norm(flow, axis=1)))
    if moved < KEYFRAME_FLOW * gray.shape[1]:
        return {"status": "ok", "say": "", "keyframe": False, "keyframes": s.keyframes, "motion_px": round(moved, 1)}

    # Enough motion for a new view. Is it real parallax, or did the user just turn?
    H, hmask = cv.findHomography(p0, p1, cv.RANSAC, 2.0)
    h_ratio = float(hmask.mean()) if hmask is not None else 0.0
    parallax = float(np.median(np.linalg.norm(cv.perspectiveTransform(p0.reshape(-1, 1, 2), H).reshape(-1, 2) - p1, axis=1))) if H is not None else 99.0
    if h_ratio > ROTATION_RATIO and parallax < MIN_PARALLAX_PX:
        s.rotation_streak += 1
        _set_ref(s, gray)
        return {"status": "warn", "say": "You're turning, not moving. Slide the phone sideways instead.",
                "keyframe": False, "keyframes": s.keyframes, "rotation_only": True, "parallax_px": round(parallax, 2)}

    s.rotation_streak = 0
    s.keyframes += 1
    s.directions.append(float(np.sign(np.median(flow[:, 0]))))
    _set_ref(s, gray)
    if s.keyframes >= TARGET_KEYFRAMES:
        return {"status": "done", "say": "That's enough views. You can stop.", "keyframe": True, "keyframes": s.keyframes,
                "parallax_px": round(parallax, 2)}
    say = ""
    if s.keyframes == TARGET_KEYFRAMES // 2:
        say = "Halfway there. Keep going, or come back the other way to see behind things."
    return {"status": "ok", "say": say, "keyframe": True, "keyframes": s.keyframes, "parallax_px": round(parallax, 2)}
