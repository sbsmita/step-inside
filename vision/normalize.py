import cv2 as cv
import numpy as np

_clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def gray_world(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    f *= means.mean() / np.maximum(means, 1e-3)
    return np.clip(f, 0, 255).astype(np.uint8)


def normalize(img: np.ndarray) -> np.ndarray:
    """White-balance then equalize lightness locally, so lamp/sunlight changes don't read as scene changes."""
    lab = cv.cvtColor(gray_world(img), cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    return cv.cvtColor(cv.merge([_clahe.apply(l), a, b]), cv.COLOR_LAB2BGR)


def norm_gray(img: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return _clahe.apply(gray)
