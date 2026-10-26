from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .normalize import norm_gray

MIN_MATCHES = 12
MIN_INLIERS = 25
RATIO = 0.75

_sift = cv.SIFT_create(nfeatures=4000)
_matcher = cv.BFMatcher(cv.NORM_L2)


@dataclass
class Features:
    keypoints: tuple
    descriptors: np.ndarray | None


@dataclass
class Alignment:
    ok: bool
    H: np.ndarray | None = None  # maps current-frame pixels to baseline-frame pixels
    matches: int = 0
    inliers: int = 0
    refined: bool = False

    def summary(self) -> dict:
        return {"ok": self.ok, "matches": self.matches, "inliers": self.inliers, "ecc_refined": self.refined}


def features(img: np.ndarray) -> Features:
    kp, des = _sift.detectAndCompute(norm_gray(img), None)
    return Features(kp, des)


def _sane(H: np.ndarray) -> bool:
    det = np.linalg.det(H[:2, :2])
    return 0.25 < det < 4.0 and abs(H[2, 0]) < 2e-3 and abs(H[2, 1]) < 2e-3


def _ecc_refine(base: np.ndarray, cur: np.ndarray, H: np.ndarray) -> np.ndarray | None:
    # ECC's warp maps template (baseline) coords into input (current) coords, i.e. the inverse of H.
    scale = 0.5
    S = np.diag([scale, scale, 1.0])
    b = cv.resize(norm_gray(base), None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    c = cv.resize(norm_gray(cur), None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    W = (S @ np.linalg.inv(H) @ np.linalg.inv(S)).astype(np.float32)
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 60, 1e-5)
    try:
        _, W = cv.findTransformECC(b, c, W, cv.MOTION_HOMOGRAPHY, criteria, None, 5)
    except cv.error:
        return None
    H2 = np.linalg.inv(np.linalg.inv(S) @ W.astype(np.float64) @ S)
    H2 /= H2[2, 2]
    return H2 if _sane(H2) else None


def align(base: np.ndarray, cur: np.ndarray, base_feats: Features | None = None,
          cur_feats: Features | None = None, refine: bool = True) -> Alignment:
    fb = base_feats or features(base)
    fc = cur_feats or features(cur)
    if fb.descriptors is None or fc.descriptors is None or len(fb.keypoints) < 8 or len(fc.keypoints) < 8:
        return Alignment(ok=False)

    pairs = _matcher.knnMatch(fc.descriptors, fb.descriptors, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < MIN_MATCHES:
        return Alignment(ok=False, matches=len(good))

    src = np.float32([fc.keypoints[m.queryIdx].pt for m in good])
    dst = np.float32([fb.keypoints[m.trainIdx].pt for m in good])
    H, mask = cv.findHomography(src, dst, cv.USAC_MAGSAC, 3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < MIN_INLIERS or not _sane(H):
        return Alignment(ok=False, matches=len(good), inliers=inliers)

    refined = False
    if refine:
        H2 = _ecc_refine(base, cur, H)
        if H2 is not None:
            H, refined = H2, True
    return Alignment(ok=True, H=H, matches=len(good), inliers=inliers, refined=refined)


def warp(cur: np.ndarray, H: np.ndarray, shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Warp the current frame into the baseline frame; also return the mask of pixels it actually covers."""
    h, w = shape[:2]
    warped = cv.warpPerspective(cur, H, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    valid = cv.warpPerspective(np.full(cur.shape[:2], 255, np.uint8), H, (w, h), flags=cv.INTER_NEAREST)
    valid = cv.erode(valid, np.ones((15, 15), np.uint8))
    return warped, valid > 0
