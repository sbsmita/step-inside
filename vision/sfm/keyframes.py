"""Pick sharp keyframes with enough camera motion between them from a video (or a list of photos)."""
import cv2 as cv
import numpy as np

LONG_SIDE = 960
SAMPLE_HZ = 6.0
MIN_FLOW_FRAC = 0.03
MIN_KEYFRAMES = 12
MAX_KEYFRAMES = 60


def _resize(img: np.ndarray) -> np.ndarray:
    s = LONG_SIDE / max(img.shape[:2])
    return cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else img


def sample_video(path: str) -> list[np.ndarray]:
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("could not open video")
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    every = max(1, round(fps / SAMPLE_HZ))
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            frames.append(_resize(frame))
        i += 1
    cap.release()
    return frames


def _sharpness(img: np.ndarray) -> float:
    return float(cv.Laplacian(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.CV_64F).var())


def select(frames: list[np.ndarray]) -> tuple[list[np.ndarray], dict]:
    """Keep a frame once tracked features have moved enough since the last keyframe; skip blurry ones."""
    if not frames:
        return [], {"sampled": 0, "keyframes": 0}
    sharp = np.array([_sharpness(f) for f in frames])
    blur_cut = 0.5 * np.median(sharp)
    keys = [0]
    last_gray = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)
    last_pts = cv.goodFeaturesToTrack(last_gray, 600, 0.01, 8)
    width = frames[0].shape[1]
    for i in range(1, len(frames)):
        gray = cv.cvtColor(frames[i], cv.COLOR_BGR2GRAY)
        moved = True
        if last_pts is not None and len(last_pts) >= 20:
            nxt, st, _ = cv.calcOpticalFlowPyrLK(last_gray, gray, last_pts, None, winSize=(21, 21), maxLevel=4)
            ok = st.ravel() == 1
            if ok.sum() >= 20:
                flow = np.median(np.linalg.norm((nxt - last_pts).reshape(-1, 2)[ok], axis=1))
                moved = flow > MIN_FLOW_FRAC * width
        if moved and sharp[i] >= blur_cut:
            keys.append(i)
            last_gray = gray
            last_pts = cv.goodFeaturesToTrack(gray, 600, 0.01, 8)
    if len(keys) < MIN_KEYFRAMES <= len(frames):
        # Slow, steady sweeps rarely cross the motion threshold; dense coverage still needs enough views.
        usable = [i for i in range(len(frames)) if sharp[i] >= blur_cut]
        keys = sorted({usable[int(round(k))] for k in np.linspace(0, len(usable) - 1, min(len(usable), 2 * MIN_KEYFRAMES))})
    if len(keys) > MAX_KEYFRAMES:
        keys = [keys[int(round(k))] for k in np.linspace(0, len(keys) - 1, MAX_KEYFRAMES)]
    return [frames[k] for k in keys], {"sampled": len(frames), "keyframes": len(keys)}
