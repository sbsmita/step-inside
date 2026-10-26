"""Paper matte: separate a book illustration from the blank paper around it.

Picture-book art is often a vignette that fades into white paper; treating that paper as part of the scene is what
made worlds look like "a white card in fog". The paper colour is estimated from the page border, and anything that
differs from it (or carries pen lines, which keep pale washes like a white apron) is art."""
import cv2 as cv
import numpy as np

MIN_BORDER_PAPER = 0.03


def _border(h: int, w: int, band: int) -> np.ndarray:
    sel = np.zeros((h, w), bool)
    sel[:band] = sel[-band:] = sel[:, :band] = sel[:, -band:] = True
    return sel


def paper_colour(img: np.ndarray, band: int = 6) -> np.ndarray:
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB).astype(np.float32)
    return np.median(lab[_border(*img.shape[:2], band)], axis=0)


def paper_field(img: np.ndarray, band: int = 6, threshold: float = 10.0) -> np.ndarray:
    """The paper colour at every pixel (h, w, 3 Lab). A photographed page is lit unevenly, so one global paper
    colour turns the shaded side of the paper into "art"; instead a smooth quadratic surface is fitted through the
    border, robustly (border pixels that are art, not paper, are left out of the fit)."""
    h, w = img.shape[:2]
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB).astype(np.float32)
    ys, xs = np.mgrid[:h, :w].astype(np.float32)

    def design(x, y):
        x, y = x / w - 0.5, y / h - 0.5
        return np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], axis=-1)

    border = _border(h, w, band)
    a, v = design(xs[border], ys[border]), lab[border]
    sel = np.linalg.norm(v - np.median(v, axis=0), axis=1) < 2 * threshold  # start from pixels near the median
    coef = None
    for _ in range(3):
        if sel.sum() < 60:
            break
        coef = np.linalg.lstsq(a[sel], v[sel], rcond=None)[0]
        sel = np.linalg.norm(a @ coef - v, axis=1) < 1.5 * threshold
    if coef is None:
        return np.broadcast_to(np.median(v, axis=0), lab.shape).copy()
    field = design(xs, ys) @ coef
    lo, hi = v[sel].min(axis=0), v[sel].max(axis=0)  # the fit may only interpolate the border, never run wild
    return np.clip(field, lo, hi)


def paper_matte(img: np.ndarray, threshold: float = 10.0) -> tuple[np.ndarray, bool]:
    """Returns (matte uint8 0/255, full_bleed). Full-bleed images (no paper border) get an all-art matte."""
    h, w = img.shape[:2]
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB).astype(np.float32)
    paper = paper_colour(img)
    dist = cv.GaussianBlur(np.linalg.norm(lab - paper_field(img, threshold=threshold), axis=2), (0, 0), 1.5)

    border = np.concatenate([dist[:4].ravel(), dist[-4:].ravel(), dist[:, :4].ravel(), dist[:, -4:].ravel()])
    if (border < threshold).mean() < MIN_BORDER_PAPER or paper[0] < 170:  # no light paper surround
        return np.full((h, w), 255, np.uint8), True

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ink = cv.dilate(cv.Canny(gray, 40, 120), np.ones((3, 3), np.uint8))
    art = ((dist > threshold) | (ink > 0)).astype(np.uint8) * 255
    k = max(5, min(h, w) // 30)
    art = cv.morphologyEx(art, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k)))
    art = cv.morphologyEx(art, cv.MORPH_OPEN, np.ones((7, 7), np.uint8))
    contours, _ = cv.findContours(art, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    matte = np.zeros((h, w), np.uint8)
    keep = [c for c in contours if cv.contourArea(c) > 0.01 * h * w]
    cv.drawContours(matte, keep, -1, 255, thickness=cv.FILLED)  # filled: holes inside the art are art
    if matte.mean() < 0.05 * 255:
        return np.full((h, w), 255, np.uint8), True
    return matte, False
