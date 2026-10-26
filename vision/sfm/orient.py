"""Make the reconstruction comfortable for VR: floor level, real-world scale, start where the first camera stood."""
import cv2 as cv
import numpy as np

from .incremental import Reconstruction

HANDHELD_HEIGHT_M = 1.4
DEFAULT_DEPTH_M = 2.5


def _camera_up(recon: Reconstruction) -> np.ndarray:
    """Phones are held upright, so the average camera 'up' (-y in OpenCV camera axes) approximates gravity."""
    ups = [cv.Rodrigues(recon.rvecs[f])[0].T @ np.array([0.0, -1.0, 0.0]) for f in recon.rvecs]
    u = np.mean(ups, axis=0)
    return u / np.linalg.norm(u)


def _floor_plane(pts: np.ndarray, up: np.ndarray, centers: np.ndarray, rng: np.random.Generator):
    """RANSAC for the largest roughly-horizontal plane below the cameras."""
    if len(pts) < 500:
        return None
    sample = pts[rng.choice(len(pts), min(len(pts), 60_000), replace=False)]
    heights = sample @ up
    below = sample[heights < np.median(centers @ up)]
    if len(below) < 300:
        return None
    extent = np.ptp(sample, axis=0).max()
    tol = 0.01 * extent
    best, best_n = None, 0
    for _ in range(400):
        a, b, c = below[rng.choice(len(below), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if np.dot(n, up) < 0:
            n = -n
        if np.dot(n, up) < np.cos(np.radians(25)):
            continue
        d = -np.dot(n, a)
        count = int((np.abs(below @ n + d) < tol).sum())
        if count > best_n:
            best, best_n = (n, d), count
    if best is None or best_n < 0.08 * len(below):
        return None
    return best[0], best[1], best_n


def orient(recon: Reconstruction, pts: np.ndarray, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Returns a 4x4 similarity transform into viewer space (meters, +Y up, first camera at x=z=0 looking -Z)."""
    rng = np.random.default_rng(seed)
    frames = sorted(recon.rvecs)
    centers = np.array([recon.center(f) for f in frames])
    up = _camera_up(recon)
    plane = _floor_plane(pts, up, centers, rng)
    if plane is not None:
        n, d, _ = plane
        up = n
        cam_h = float(np.median(centers @ n + d))
        scale = HANDHELD_HEIGHT_M / cam_h if cam_h > 1e-6 else 1.0
        scale_source, floor = "floor_and_handheld_height", True
    else:
        R0 = cv.Rodrigues(recon.rvecs[frames[0]])[0]
        z = (R0 @ pts.T).T[:, 2] + recon.tvecs[frames[0]][2]
        scale = DEFAULT_DEPTH_M / float(np.median(z[z > 0])) if (z > 0).any() else 1.0
        cam_h = HANDHELD_HEIGHT_M / scale
        scale_source, floor = "assumed_room_depth", False

    R0 = cv.Rodrigues(recon.rvecs[frames[0]])[0]
    forward = R0.T @ np.array([0.0, 0.0, 1.0])
    forward -= np.dot(forward, up) * up
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    # Rows map world axes to viewer axes: x=right, y=up, z=-forward.
    Rv = np.stack([right, up, -forward])
    C0 = centers[0]
    T = np.eye(4)
    T[:3, :3] = scale * Rv
    T[:3, 3] = -scale * Rv @ C0
    return T, {"scale_source": scale_source, "floor_found": floor, "camera_height_m": round(cam_h * scale, 3)}


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ pts.T).T + T[:3, 3]
