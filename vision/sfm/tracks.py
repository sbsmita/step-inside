"""SIFT features, geometrically verified pairwise matches, and multi-view feature tracks."""
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from ..align import features

RATIO = 0.8
MIN_MATCHES = 30
MIN_INLIERS = 25

_matcher = cv.BFMatcher(cv.NORM_L2)


@dataclass
class FrameFeatures:
    pts: np.ndarray        # (N, 2) float32 pixel coordinates
    des: np.ndarray | None  # (N, 128) float32 SIFT descriptors
    colors: np.ndarray     # (N, 3) uint8 RGB at each keypoint


def extract(img: np.ndarray) -> FrameFeatures:
    f = features(img)
    pts = np.float32([k.pt for k in f.keypoints]).reshape(-1, 2)
    xy = np.clip(pts.round().astype(int), 0, [img.shape[1] - 1, img.shape[0] - 1])
    colors = img[xy[:, 1], xy[:, 0], ::-1] if len(pts) else np.zeros((0, 3), np.uint8)
    return FrameFeatures(pts, f.descriptors, colors)


def match_pair(a: FrameFeatures, b: FrameFeatures, K: np.ndarray) -> np.ndarray | None:
    """Ratio-test matches that agree on one essential matrix (rigid camera motion). Returns (M, 2) index pairs."""
    if a.des is None or b.des is None or len(a.pts) < MIN_MATCHES or len(b.pts) < MIN_MATCHES:
        return None
    pairs = _matcher.knnMatch(a.des, b.des, k=2)
    good = np.array([(p[0].queryIdx, p[0].trainIdx) for p in pairs
                     if len(p) == 2 and p[0].distance < RATIO * p[1].distance], dtype=np.int32).reshape(-1, 2)
    if len(good) < MIN_MATCHES:
        return None
    _, mask = cv.findEssentialMat(a.pts[good[:, 0]], b.pts[good[:, 1]], K, cv.USAC_MAGSAC, 0.999, 1.0)
    if mask is None:
        return None
    inl = good[mask.ravel().astype(bool)]
    return inl if len(inl) >= MIN_INLIERS else None


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_tracks(matches: dict[tuple[int, int], np.ndarray], min_len: int = 2) -> list[dict[int, int]]:
    """Chain pairwise matches into tracks {frame: keypoint}. Tracks that hit one frame twice are inconsistent; drop them."""
    uf = _UnionFind()
    for (i, j), m in matches.items():
        for a, b in m:
            uf.union((i, int(a)), (j, int(b)))
    groups: dict = {}
    for node in list(uf.parent):
        groups.setdefault(uf.find(node), []).append(node)
    tracks = []
    for nodes in groups.values():
        frames = [f for f, _ in nodes]
        if len(nodes) >= min_len and len(set(frames)) == len(frames):
            tracks.append({f: k for f, k in nodes})
    return tracks


def match_sequence(feats: list[FrameFeatures], K: np.ndarray, window: int = 6) -> dict[tuple[int, int], np.ndarray]:
    """Match each frame with its next `window` frames (video order), plus sparse long-range pairs for loop closure."""
    out = {}
    n = len(feats)
    pairs = {(i, j) for i in range(n) for j in range(i + 1, min(n, i + window + 1))}
    pairs |= {(i, j) for i in range(0, n, 4) for j in range(i + window + 1, n, 4)}
    for i, j in sorted(pairs):
        m = match_pair(feats[i], feats[j], K)
        if m is not None:
            out[(i, j)] = m
    return out
