"""Bundle adjustment: jointly refine camera poses, shared focal length and 3D points by minimizing reprojection error.

OpenCV's `sfm` module isn't in the pip wheels, so this uses scipy's sparse least-squares with OpenCV's camera model
(pinhole, principal point at the image center)."""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def rodrigues(rvecs: np.ndarray) -> np.ndarray:
    """Vectorized rotation vectors (N, 3) -> rotation matrices (N, 3, 3), same convention as cv.Rodrigues."""
    theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
    k = rvecs / np.maximum(theta, 1e-12)
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zero = np.zeros_like(kx)
    Kx = np.stack([np.stack([zero, -kz, ky], 1), np.stack([kz, zero, -kx], 1), np.stack([-ky, kx, zero], 1)], 1)
    s, c = np.sin(theta)[:, :, None], np.cos(theta)[:, :, None]
    eye = np.eye(3)[None]
    return eye + s * Kx + (1 - c) * (Kx @ Kx)


def project(points: np.ndarray, rvecs: np.ndarray, tvecs: np.ndarray, f: float, c: np.ndarray) -> np.ndarray:
    """Per-observation projection: points (M,3), rvecs/tvecs (M,3) -> pixels (M,2)."""
    cam = np.einsum("mij,mj->mi", rodrigues(rvecs), points) + tvecs
    return f * cam[:, :2] / cam[:, 2:3] + c


def bundle_adjust(f: float, c: np.ndarray, rvecs: np.ndarray, tvecs: np.ndarray, points: np.ndarray,
                  cam_idx: np.ndarray, pt_idx: np.ndarray, uv: np.ndarray, fixed_cam: int = 0,
                  refine_focal: bool = True, max_nfev: int = 40):
    """Returns refined (f, rvecs, tvecs, points) and the final per-observation reprojection errors (px).

    Camera `fixed_cam` is held still to pin the coordinate frame."""
    n_cams, n_pts = len(rvecs), len(points)
    free = np.array([i for i in range(n_cams) if i != fixed_cam], dtype=int)
    slot = -np.ones(n_cams, dtype=int)
    slot[free] = np.arange(len(free))
    off_cam = 1
    off_pts = off_cam + 6 * len(free)

    def unpack(x):
        fx = x[0] if refine_focal else f
        r, t = rvecs.copy(), tvecs.copy()
        cam = x[off_cam:off_pts].reshape(-1, 6)
        r[free], t[free] = cam[:, :3], cam[:, 3:]
        return fx, r, t, x[off_pts:].reshape(-1, 3)

    def residuals(x):
        fx, r, t, p = unpack(x)
        return (project(p[pt_idx], r[cam_idx], t[cam_idx], fx, c) - uv).ravel()

    x0 = np.concatenate([[f], np.hstack([rvecs[free], tvecs[free]]).ravel(), points.ravel()])
    m = 2 * len(uv)
    A = lil_matrix((m, len(x0)), dtype=np.uint8)
    rows = np.arange(len(uv))
    if refine_focal:
        A[:, 0] = 1
    for k in range(6):
        sel = slot[cam_idx] >= 0
        A[2 * rows[sel], off_cam + 6 * slot[cam_idx[sel]] + k] = 1
        A[2 * rows[sel] + 1, off_cam + 6 * slot[cam_idx[sel]] + k] = 1
    for k in range(3):
        A[2 * rows, off_pts + 3 * pt_idx + k] = 1
        A[2 * rows + 1, off_pts + 3 * pt_idx + k] = 1

    res = least_squares(residuals, x0, jac_sparsity=A, x_scale="jac", loss="huber", f_scale=2.0,
                        method="trf", max_nfev=max_nfev)
    fx, r, t, p = unpack(res.x)
    err = np.linalg.norm(res.fun.reshape(-1, 2), axis=1)
    return fx, r, t, p, err
