"""Looking at a page through a camera, frame by frame (OpenCV only, ~0.1 s a frame).

- check(): is there a page worth reading in view? Text lines (PP-OCRv3 DB, cv.dnn), their size, sharpness (variance
  of the Laplacian over the text), glare on the text, darkness, and text cut off by the frame edge. Returns what
  to fix in plain words, so the user is coached to a good photo before any AI call is spent.
- PageIndex: recognizes a page it has seen before (ORB features, ratio test, RANSAC homography), so pointing the
  camera at a page that already has a story starts that story at once."""
import threading
import time

import cv2 as cv
import numpy as np

from .page import detect_lines

WORK = 960            # frames are judged at this long side
MIN_LINES = 2
MIN_TEXT_PX = 10      # median text-line height (at WORK size) the reader can still read
MIN_SHARP = 120.0     # variance of the Laplacian over the text lines
MAX_GLARE = 0.05      # share of text pixels blown out brighter than the paper (clean photos measure 0.0)
MIN_LIGHT = 55        # mean grey level of the frame
EDGE = 4              # px: a text line this close to the frame edge is cut off


def _small(img: np.ndarray, side: int = WORK) -> np.ndarray:
    s = side / max(img.shape[:2])
    return cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else img


def check(frame: np.ndarray) -> dict:
    """{"ok", "say" (what to do, or None), "issues", "lines", "text_px", "sharpness", "glare", "light", "box"}.
    "box" is the text area as [x0, y0, x1, y1] in 0..1 frame coordinates, for drawing a guide."""
    img = _small(frame)
    h, w = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    light = float(gray.mean())
    quads = detect_lines(img)
    out = {"ok": False, "say": None, "issues": [], "lines": len(quads), "text_px": 0.0, "sharpness": 0.0,
           "glare": 0.0, "light": round(light, 1), "box": None}
    if light < MIN_LIGHT:
        out["issues"].append("dark")
    if len(quads) < MIN_LINES:
        out["issues"].append("no_text")
    else:
        heights = [float(np.ptp(q[:, 1])) for q in quads]
        mask = np.zeros((h, w), np.uint8)
        for q in quads:
            cv.fillPoly(mask, [q.astype(np.int32)], 255)
        text = mask > 0
        pts = np.concatenate(quads)
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        out["box"] = [round(float(x0) / w, 3), round(float(y0) / h, 3), round(float(x1) / w, 3), round(float(y1) / h, 3)]
        out["text_px"] = round(float(np.median(heights)), 1)
        out["sharpness"] = round(float(cv.Laplacian(gray, cv.CV_64F)[text].var()), 1)
        # Glare: clipped white that is clearly brighter than the page around the text. White paper alone isn't
        # glare (and on a screenshot the paper itself is 255, with nothing brighter).
        paper = float(np.median(gray[text]))
        blown = (gray >= 250) & (gray > paper + 15)
        out["glare"] = round(float(blown[text].mean()), 3)
        if out["text_px"] < MIN_TEXT_PX:
            out["issues"].append("small")
        if out["sharpness"] < MIN_SHARP:
            out["issues"].append("blurry")
        if out["glare"] > MAX_GLARE:
            out["issues"].append("glare")
        at_edge = sum(q[:, 0].min() < EDGE or q[:, 1].min() < EDGE or q[:, 0].max() > w - EDGE
                      or q[:, 1].max() > h - EDGE for q in quads)
        if at_edge >= max(1, 0.2 * len(quads)):  # one logo at the edge isn't a page cut off
            out["issues"].append("cut_off")
    say = {"dark": "It's too dark. Find more light.",
           "no_text": "Point the camera at a page with some text.",
           "small": "Move closer: the words are too small to read.",
           "blurry": "Hold still: the picture is blurry.",
           "glare": "There's a reflection on the page. Tilt it away from the light.",
           "cut_off": "Part of the text is cut off. Move back a little."}
    out["ok"] = not out["issues"]
    out["say"] = None if out["ok"] else say[out["issues"][0]]
    return out


class PageIndex:
    """Pages that already have a story, recognized by ORB features + RANSAC homography."""

    def __init__(self, min_inliers: int = 30):
        self.min_inliers = min_inliers
        self.orb = cv.ORB_create(1500)
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING)
        self.pages: dict[str, tuple] = {}  # id -> (title, keypoints xy, descriptors)
        self.lock = threading.Lock()
        self.checked = 0.0

    def _features(self, img: np.ndarray):
        gray = cv.cvtColor(_small(img, 800), cv.COLOR_BGR2GRAY)
        kps, des = self.orb.detectAndCompute(gray, None)
        return np.float32([k.pt for k in kps]) if kps else np.zeros((0, 2), np.float32), des

    def add(self, page_id: str, title: str, img: np.ndarray) -> None:
        xy, des = self._features(img)
        if des is not None and len(des) >= self.min_inliers:
            with self.lock:
                self.pages[page_id] = (title, xy, des)

    def refresh(self, store, every_s: float = 10.0) -> None:
        """Index finished stories that aren't indexed yet (at most every `every_s` seconds)."""
        if time.monotonic() - self.checked < every_s:
            return
        self.checked = time.monotonic()
        import json
        for sid in store.list_dirs("live/")[:200]:
            if sid in self.pages:
                continue
            raw = store.get(f"live/{sid}/state.json")
            rec = json.loads(raw) if raw else None
            page = store.get(f"live/{sid}/page.jpg") if rec and rec.get("status") == "ready" else None
            if page:
                self.add(sid, rec.get("title") or "", cv.imdecode(np.frombuffer(page, np.uint8), cv.IMREAD_COLOR))

    def match(self, frame: np.ndarray) -> dict | None:
        """The story whose page is in view: {"id", "title", "inliers"}, or None. Also None when two pages fit almost
        equally well (features sit mostly on the text, so pages with the same words can't be told apart): better
        to make a new story than to open the wrong one."""
        xy, des = self._features(frame)
        if des is None or len(des) < self.min_inliers:
            return None
        with self.lock:
            pages = list(self.pages.items())
        best, runner_up = None, 0
        for pid, (title, pxy, pdes) in pages:
            pairs = self.matcher.knnMatch(des, pdes, k=2)
            good = [m for m, *n in pairs if n and m.distance < 0.75 * n[0].distance]
            if len(good) < self.min_inliers:
                continue
            src = xy[[m.queryIdx for m in good]]
            dst = pxy[[m.trainIdx for m in good]]
            _, inl = cv.findHomography(src, dst, cv.RANSAC, 6.0)
            n = int(inl.sum()) if inl is not None else 0
            if n < self.min_inliers:
                continue
            if best is None or n > best["inliers"]:
                runner_up = best["inliers"] if best else runner_up
                best = {"id": pid, "title": title, "inliers": n}
            else:
                runner_up = max(runner_up, n)
        return best if best and runner_up < 0.8 * best["inliers"] else None
