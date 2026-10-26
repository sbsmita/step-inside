"""Read a photographed book page with OpenCV: flatten it, find text lines, split lines into words, recognize
words (OpenCV Zoo PP-OCRv3 detector + CRNN recognizer, Apache-2.0), and locate the illustration."""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np

from .clean import find_print_quad, rectify
from .netpool import NetPool, pool_size

MODELS = Path(__file__).resolve().parent / "models"
DET_MODEL = MODELS / "text_detection_en_ppocrv3_2023may.onnx"
REC_MODEL = MODELS / "text_recognition_CRNN_CH_2021sep.onnx"
# Character set of the CRNN "CH" model, per OpenCV Zoo: digits, a-z, A-Z, then ASCII punctuation.
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" + "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
WORK_LONG_SIDE = 1280
REC_W, REC_H = 100, 32


@dataclass
class Page:
    page: np.ndarray                      # flattened page (BGR)
    text: str
    lines: list = field(default_factory=list)   # [{"box": [[x,y]*4], "text": str}]
    illustration: np.ndarray | None = None
    illustration_box: tuple | None = None       # (x0, y0, x1, y1) in page pixels
    steps: list = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)


def _load_detector():
    model = cv.dnn_TextDetectionModel_DB(cv.dnn.readNet(str(DET_MODEL)))
    model.setBinaryThreshold(0.3)
    model.setPolygonThreshold(0.5)
    model.setUnclipRatio(2.0)
    model.setMaxCandidates(400)
    model.setInputMean((123.675, 116.28, 103.53))
    model.setInputScale(1.0 / 255.0 / np.array([0.229, 0.224, 0.225]))
    return model


_detectors = NetPool(_load_detector, pool_size("TEXT_DET_NETS", 2))
# The recognizer reads one 100x32 word per call (~50 ms), so a page of 160 words is read by parallel copies.
REC_NETS = pool_size("REC_NETS", min(8, os.cpu_count() or 2))
_recognizers = NetPool(lambda: cv.dnn.readNet(str(REC_MODEL)), REC_NETS)
_rec_threads = ThreadPoolExecutor(max_workers=REC_NETS)


def _resize(img: np.ndarray) -> np.ndarray:
    """Work at a fixed size: big photos are shrunk, and small images (a phone screenshot, a web poster) are enlarged,
    since the recognizer needs letters about 20 px tall and a 700 px poster has 12 px text."""
    s = WORK_LONG_SIDE / max(img.shape[:2])
    if s < 1:
        return cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
    return cv.resize(img, None, fx=min(s, 3.0), fy=min(s, 3.0), interpolation=cv.INTER_CUBIC) if s > 1.1 else img


def _light_text(crop: np.ndarray) -> bool:
    """Is the text lighter than its background (white letters on a blue poster)? Text is the minority of a line
    crop, so after an Otsu split the smaller class is the ink."""
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    _, bright = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return (bright > 0).mean() < 0.5


def detect_lines(img: np.ndarray) -> list[np.ndarray]:
    """Text-line quadrilaterals (4x2 float32), sorted top-to-bottom then left-to-right."""
    h, w = img.shape[:2]
    # DB needs input sides that are multiples of 32.
    iw, ih = max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)
    with _detectors.net() as det:
        det.setInputSize((iw, ih))
        boxes, _ = det.detect(cv.resize(img, (iw, ih)))
    sx, sy = w / iw, h / ih
    quads = [np.float32(b).reshape(4, 2) * [sx, sy] for b in boxes]
    if not quads:
        return []
    # Reading order: group boxes into rows (centres closer than half a line height), rows top-down, left-right.
    quads.sort(key=lambda q: q[:, 1].mean())
    line_h = float(np.median([np.ptp(q[:, 1]) for q in quads]))
    rows, row = [], [quads[0]]
    for q in quads[1:]:
        if q[:, 1].mean() - np.mean([r[:, 1].mean() for r in row]) < 0.5 * line_h:
            row.append(q)
        else:
            rows.append(row)
            row = [q]
    rows.append(row)
    return [q for r in rows for q in sorted(r, key=lambda q: q[:, 0].min())]


def _upright(quad: np.ndarray) -> np.ndarray:
    """Order corners TL, TR, BR, BL for a roughly horizontal text line."""
    s, d = quad.sum(axis=1), np.diff(quad, axis=1).ravel()
    return np.float32([quad[s.argmin()], quad[d.argmin()], quad[s.argmax()], quad[d.argmax()]])


def _line_crop(img: np.ndarray, quad: np.ndarray, height: int = 48) -> np.ndarray:
    tl, tr, br, bl = _upright(quad)
    lw = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    lh = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr), 1.0)
    out_w = max(8, int(round(lw * height / lh)))
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, height - 1], [0, height - 1]])
    return cv.warpPerspective(img, cv.getPerspectiveTransform(np.float32([tl, tr, br, bl]), dst), (out_w, height),
                              flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)


def split_words(line: np.ndarray) -> list[tuple[int, int]]:
    """Column spans of words in a horizontal line crop: ink columns separated by gaps wider than a letter gap."""
    gray = cv.cvtColor(line, cv.COLOR_BGR2GRAY)
    _, ink = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    ink = cv.morphologyEx(ink, cv.MORPH_OPEN, np.ones((2, 1), np.uint8))
    cols = (ink > 0).sum(axis=0) > 0
    h = line.shape[0]
    min_gap = max(3, int(0.17 * h))  # word spaces are ~0.2-0.3 of line height; letter gaps under ~0.1
    spans, start, gap = [], None, 0
    for x, on in enumerate(cols):
        if on:
            if start is None:
                start = x
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                spans.append((start, x - gap + 1))
                start, gap = None, 0
    if start is not None:
        spans.append((start, len(cols)))
    return [(max(0, a - 2), min(len(cols), b + 2)) for a, b in spans if b - a >= 3]


def recognize_word(crop: np.ndarray) -> str:
    """CRNN on one word (stretched to 100x32 like the Zoo reference), CTC greedy decoding."""
    blob = cv.dnn.blobFromImage(cv.resize(crop, (REC_W, REC_H), interpolation=cv.INTER_CUBIC),
                                scalefactor=1 / 127.5, size=(REC_W, REC_H), mean=127.5)
    with _recognizers.net() as net:
        net.setInput(blob)
        idx = net.forward()[:, 0, :].argmax(axis=1)
    chars, prev = [], 0
    for c in idx:
        if c != 0 and c != prev and c - 1 < len(CHARSET):
            chars.append(CHARSET[c - 1])
        prev = c
    return "".join(chars)


def _contains_text(quad: np.ndarray, lines: list[np.ndarray], min_frac: float = 0.8) -> bool:
    if not lines:
        return True
    contour = quad.reshape(-1, 1, 2).astype(np.float32)
    inside = [cv.pointPolygonTest(contour, tuple(float(v) for v in q.mean(axis=0)), False) >= 0 for q in lines]
    return sum(inside) / len(inside) >= min_frac


def find_illustration(img: np.ndarray, text_mask: np.ndarray) -> tuple | None:
    """The biggest coloured / inked region that isn't text: the page's picture."""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    paper = cv.GaussianBlur(gray, (0, 0), 25)
    content = ((hsv[..., 1] > 45) | (gray < paper.astype(np.int16) - 35)).astype(np.uint8) * 255
    content[cv.dilate(text_mask, np.ones((15, 15), np.uint8)) > 0] = 0
    k = max(15, min(img.shape[:2]) // 25)
    content = cv.morphologyEx(content, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k)))
    n, _, stats, _ = cv.connectedComponentsWithStats(content, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    x, y, w, h, area = (int(v) for v in stats[i])
    if area < 0.04 * img.shape[0] * img.shape[1]:
        return None
    return x, y, x + w, y + h


def flatten_page(img: np.ndarray, flatten: bool = True) -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
    """The page at working size, flattened if it was photographed at an angle, and its text lines.
    Returns (page, line quads, steps)."""
    steps = []
    page = _resize(img)
    quads = detect_lines(page)
    if flatten:
        quad = find_print_quad(page)
        # A 4-corner outline is the page only if the text is inside it; otherwise it's probably the
        # illustration's frame (e.g. when the page already fills the photo).
        if quad is not None and _contains_text(quad, quads):
            page = rectify(page, quad)
            steps.append({"step": "flatten_page", "corners": quad.round(1).tolist()})
            quads = detect_lines(page)
    return page, quads, steps


def recognize_lines(page: np.ndarray, quads: list[np.ndarray]) -> list[dict]:
    """Every detected line's words, recognized in parallel. Returns [{"box", "text"}] in reading order."""
    crops = []
    for q in quads:
        crop = _line_crop(page, q)
        if _light_text(crop):  # word splitting and the recognizer both expect dark ink on light paper
            crop = 255 - crop
        crops.append(crop)
    spans = [split_words(c) for c in crops]
    words = list(_rec_threads.map(recognize_word, [c[:, a:b] for c, s in zip(crops, spans) for a, b in s]))
    lines, k = [], 0
    for q, s in zip(quads, spans):
        text = " ".join(w for w in words[k:k + len(s)] if w)
        k += len(s)
        if text:
            lines.append({"box": q.round(1).tolist(), "text": text})
    return lines


def read_page(img: np.ndarray, flatten: bool = True) -> Page:
    timings, t = {}, time.perf_counter()

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        timings[name] = round((now - t) * 1000)
        t = now

    page, quads, steps = flatten_page(img, flatten)
    lap("flatten_and_detect_text")
    lines = recognize_lines(page, quads)
    lap("recognize_text")
    text_mask = np.zeros(page.shape[:2], np.uint8)
    for q in quads:
        cv.fillPoly(text_mask, [q.astype(np.int32)], 255)
    box = find_illustration(page, text_mask)
    lap("find_illustration")
    illus = page[box[1]:box[3], box[0]:box[2]].copy() if box else None
    return Page(page=page, text=_join(lines), lines=lines, illustration=illus, illustration_box=box,
                steps=steps, timings_ms=timings)


def _join(lines: list[dict]) -> str:
    """Re-join hyphenated line breaks and lines into running text."""
    out = ""
    for ln in lines:
        s = ln["text"].strip()
        if out.endswith("-"):
            out = out[:-1] + s
        else:
            out = (out + " " + s).strip()
    return out


def annotate(p: Page) -> np.ndarray:
    """Visual evidence for the UI: text lines in blue, illustration in orange."""
    vis = p.page.copy()
    for ln in p.lines:
        cv.polylines(vis, [np.int32(ln["box"])], True, (255, 140, 30), 2, cv.LINE_AA)
    if p.illustration_box:
        x0, y0, x1, y1 = p.illustration_box
        cv.rectangle(vis, (x0, y0), (x1, y1), (30, 140, 255), 3, cv.LINE_AA)
    return vis
