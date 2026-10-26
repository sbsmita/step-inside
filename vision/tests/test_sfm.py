"""SfM geometry against synthetic ground truth: known 3D points seen by a known camera path."""
import cv2 as cv
import numpy as np
import pytest

from vision.sfm.bundle import project, rodrigues
from vision.sfm.incremental import Incremental

W, H, F = 960, 720, 760.0


def look_at(center: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV camera (x right, y down, z forward) at `center` looking at `target`; returns (rvec, tvec)."""
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, -1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return cv.Rodrigues(R)[0].ravel(), -R @ center


def synthetic(n_frames=24, n_points=1500, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    # A room-ish cloud: floor, back wall, and boxes in between (all in front of the camera path).
    floor = np.c_[rng.uniform(-2, 2, 600), np.full(600, 1.4), rng.uniform(2, 6, 600)]
    wall = np.c_[rng.uniform(-2.5, 2.5, 500), rng.uniform(-1.5, 1.4, 500), np.full(500, 6.0)]
    boxes = np.c_[rng.uniform(-1, 1, 400), rng.uniform(0.4, 1.4, 400), rng.uniform(3, 4.5, 400)]
    X = np.vstack([floor, wall, boxes])[:n_points]
    cams = []
    for i in range(n_frames):
        s = i / (n_frames - 1)
        center = np.array([-0.8 + 1.6 * s, 0.05 * np.sin(6 * s), 0.3 * s])  # a 1.6 m sideways sweep
        cams.append(look_at(center, np.array([0.0, 0.6, 4.5])))
    pts, tracks_by_point = [], {}
    c = np.array([W / 2, H / 2])
    for fr, (rv, tv) in enumerate(cams):
        uv = project(X, np.tile(rv, (len(X), 1)), np.tile(tv, (len(X), 1)), F, c)
        depth = (rodrigues(rv[None])[0] @ X.T).T[:, 2] + tv[2]
        vis = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        uv = uv + rng.normal(0, noise, uv.shape)
        frame_pts = []
        for p in np.nonzero(vis)[0]:
            tracks_by_point.setdefault(p, {})[fr] = len(frame_pts)
            frame_pts.append(uv[p])
        pts.append(np.float32(frame_pts).reshape(-1, 2))
    tracks = [t for t in tracks_by_point.values() if len(t) >= 2]
    return X, cams, pts, tracks


def _align_umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity transform (s, R, t) mapping src onto dst (SfM is only defined up to scale and pose)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / a.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def _path_error(cams, r) -> float:
    """Mean camera-center error after similarity alignment, as a fraction of the path length."""
    frames = sorted(r.rvecs)
    est = np.array([r.center(fr) for fr in frames])
    true = np.array([-cv.Rodrigues(cams[fr][0])[0].T @ cams[fr][1] for fr in frames])
    s, R, t = _align_umeyama(est, true)
    aligned = (s * (R @ est.T)).T + t
    path_len = np.linalg.norm(np.diff(true, axis=0), axis=1).sum()
    return float(np.linalg.norm(aligned - true, axis=1).mean() / path_len)


@pytest.fixture(scope="module")
def recon():
    X, cams, pts, tracks = synthetic()
    return X, cams, Incremental(pts, tracks, (W, H), f0=F).run()


def test_registers_every_frame_with_subpixel_error(recon):
    _, cams, r = recon
    assert r is not None
    assert r.stats["registered"] == len(cams)
    assert r.stats["median_reproj_px"] < 1.0


def test_camera_path_matches_ground_truth(recon):
    _, cams, r = recon
    assert _path_error(cams, r) < 0.01


def test_tolerates_focal_prior_error():
    """A phone's field of view is only roughly known; an 8% focal error must still give a usable path."""
    _, cams, pts, tracks = synthetic(seed=1)
    r = Incremental(pts, tracks, (W, H), f0=F * 1.08).run()
    assert r is not None and r.stats["registered"] == len(cams)
    assert _path_error(cams, r) < 0.05
