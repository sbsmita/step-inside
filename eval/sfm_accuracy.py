"""Reconstruction accuracy on the rendered synthetic room, against its exact ground-truth camera path.

Usage: python -m eval.sfm_accuracy FRAMES_DIR  (render first with: python -m eval.synth_room.render FRAMES_DIR)"""
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

from vision.sfm.keyframes import select
from vision.sfm.pipeline import reconstruct


def umeyama(src: np.ndarray, dst: np.ndarray):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / a.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def main(frames_dir: Path) -> dict:
    gt = json.loads((frames_dir / "ground_truth.json").read_text())
    paths = sorted(frames_dir.glob("frame_*.png"))
    images = [cv.imread(str(p)) for p in paths]
    keys, kinfo = select(images)
    key_idx = [next(i for i, im in enumerate(images) if im is k) for k in keys]
    res = reconstruct(keys, hfov_deg=gt["meta"]["hfov"])
    recon = res["recon"]
    frames = sorted(recon.rvecs)
    est = np.array([recon.center(f) for f in frames])
    true = np.array([gt["poses"][key_idx[f]]["position"] for f in frames])
    s, R, t = umeyama(est, true)
    aligned = (s * (R @ est.T)).T + t
    err = np.linalg.norm(aligned - true, axis=1)
    path_len = float(np.linalg.norm(np.diff(true, axis=0), axis=1).sum())
    # Metric scale check: our scale comes from the detected floor + assumed 1.4 m handheld height.
    est_v = res["cameras"]
    scale_ratio = float(np.linalg.norm(np.diff(est_v, axis=0), axis=1).sum() / path_len)
    report = {
        "keyframes": kinfo, "sfm": recon.stats, "dense": res["dense"], "orient": res["orient"],
        "camera_path_error_cm_mean": round(100 * float(err.mean()), 2),
        "camera_path_error_cm_max": round(100 * float(err.max()), 2),
        "camera_path_error_pct_of_path": round(100 * float(err.mean()) / path_len, 2),
        "metric_scale_ratio": round(scale_ratio, 3),
        "timings_ms": res["timings_ms"],
    }
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main(Path(sys.argv[1]))
