"""Dense geometry: per-view DNN depth, put into the reconstruction's units using its triangulated points,
then kept only where neighboring views agree."""
import cv2 as cv
import numpy as np

from .. import depth
from .incremental import Reconstruction

STEP = 3
MAX_VIEWS = 24
AGREE = 0.06  # relative depth disagreement allowed between neighboring views


def _fit_inverse_depth(inv_pred: np.ndarray, z_true: np.ndarray) -> tuple[float, float] | None:
    """Robust fit 1/z = a * inv_pred + b (the network's output is affine in inverse depth)."""
    keep = np.ones(len(z_true), bool)
    a = b = None
    for _ in range(4):
        if keep.sum() < 15:
            return None
        A = np.c_[inv_pred[keep], np.ones(keep.sum())]
        a, b = np.linalg.lstsq(A, 1.0 / z_true[keep], rcond=None)[0]
        resid = np.abs(a * inv_pred + b - 1.0 / z_true)
        keep = resid < 2.5 * np.median(resid[keep]) + 1e-9
    return (float(a), float(b)) if a is not None and a > 0 else None


def view_depths(images: dict[int, np.ndarray], recon: Reconstruction) -> dict[int, dict]:
    frames = sorted(recon.rvecs)
    if len(frames) > MAX_VIEWS:
        frames = [frames[int(round(i))] for i in np.linspace(0, len(frames) - 1, MAX_VIEWS)]
    K = recon.K()
    out = {}
    for fr in frames:
        img = images[fr]
        inv = depth.inverse_depth(img)
        R = cv.Rodrigues(recon.rvecs[fr])[0]
        cam = (R @ recon.points.T).T + recon.tvecs[fr]
        front = cam[:, 2] > 1e-6
        uv = (K @ cam[front].T).T
        uv = uv[:, :2] / uv[:, 2:3]
        h, w = inv.shape
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if inside.sum() < 15:
            continue
        px = uv[inside].astype(int)
        fit = _fit_inverse_depth(inv[px[:, 1], px[:, 0]], cam[front][inside, 2])
        if fit is None:
            continue
        a, b = fit
        disp = a * inv + b
        z = np.where(disp > 1e-6, 1.0 / np.maximum(disp, 1e-6), 0.0).astype(np.float32)
        out[fr] = {"z": z, "R": R, "t": recon.tvecs[fr], "image": img, "fit_points": int(inside.sum())}
    return out


def fuse(images: dict[int, np.ndarray], recon: Reconstruction) -> tuple[np.ndarray, np.ndarray, dict]:
    views = view_depths(images, recon)
    frames = sorted(views)
    K = recon.K()
    Kinv = np.linalg.inv(K)
    all_pts, all_cols = [], []
    kept_total, cand_total = 0, 0
    for idx, fr in enumerate(frames):
        v = views[fr]
        h, w = v["z"].shape
        uu, vv = np.meshgrid(np.arange(0, w, STEP), np.arange(0, h, STEP))
        z = v["z"][vv, uu].ravel()
        valid = z > 0
        rays = Kinv @ np.vstack([uu.ravel(), vv.ravel(), np.ones(uu.size)])
        cam = (rays * z).T
        world = (v["R"].T @ (cam - v["t"]).T).T
        agree = np.zeros(len(z), bool)
        for nb in frames[max(0, idx - 2):idx] + frames[idx + 1:idx + 3]:
            n = views[nb]
            c2 = (n["R"] @ world.T).T + n["t"]
            p = (K @ c2.T).T
            with np.errstate(divide="ignore", invalid="ignore"):
                px = p[:, :2] / p[:, 2:3]
            ok = (c2[:, 2] > 0) & (px[:, 0] >= 0) & (px[:, 0] < w - 1) & (px[:, 1] >= 0) & (px[:, 1] < h - 1)
            zn = np.zeros(len(z), np.float32)
            zn[ok] = n["z"][px[ok, 1].astype(int), px[ok, 0].astype(int)]
            agree |= ok & (zn > 0) & (np.abs(c2[:, 2] - zn) < AGREE * zn)
        keep = valid & agree
        cand_total += int(valid.sum())
        kept_total += int(keep.sum())
        cols = v["image"][vv.ravel()[keep], uu.ravel()[keep], ::-1]
        all_pts.append(world[keep])
        all_cols.append(cols)
    if not all_pts:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8), {"views": 0}
    pts, cols = np.vstack(all_pts), np.vstack(all_cols)
    pts, cols = voxel_downsample(pts, cols)
    return pts, cols, {"views": len(frames), "consistent_frac": round(kept_total / max(cand_total, 1), 3),
                       "points": int(len(pts))}


def voxel_downsample(pts: np.ndarray, cols: np.ndarray, target: int = 600_000) -> tuple[np.ndarray, np.ndarray]:
    if len(pts) == 0:
        return pts, cols
    extent = np.percentile(pts, 98, axis=0) - np.percentile(pts, 2, axis=0)
    voxel = max(float(extent.max()) / 400.0, 1e-6)
    while True:
        keys = np.floor(pts / voxel).astype(np.int64)
        _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        if len(first) <= target:
            break
        voxel *= 1.3
    counts = np.bincount(inverse.ravel())
    out_p = np.zeros((len(first), 3))
    out_c = np.zeros((len(first), 3))
    np.add.at(out_p, inverse.ravel(), pts)
    np.add.at(out_c, inverse.ravel(), cols.astype(np.float64))
    return out_p / counts[:, None], (out_c / counts[:, None]).astype(np.uint8)
