"""Video or photo set -> real 3D point-cloud scene (OpenCV SfM + DNN depth fusion)."""
import io
import math
import time

import numpy as np
import trimesh

from . import dense, keyframes, orient
from .incremental import Incremental, Reconstruction
from .tracks import build_tracks, extract, match_sequence

DEFAULT_HFOV_LONG_SIDE = 65.0  # typical phone main camera across the long side


class ReconstructionFailed(Exception):
    pass


def reconstruct(images: list, hfov_deg: float = DEFAULT_HFOV_LONG_SIDE, on_stage=None) -> dict:
    """Run the full pipeline; `on_stage(name, info)` is called as each stage finishes (for progress + agents)."""
    timings, t = {}, time.perf_counter()

    def done(name, info=None):
        nonlocal t
        now = time.perf_counter()
        timings[name] = round((now - t) * 1000)
        t = now
        if on_stage:
            on_stage(name, info or {})

    h, w = images[0].shape[:2]
    f0 = (max(w, h) / 2) / math.tan(math.radians(hfov_deg / 2))
    feats = [extract(img) for img in images]
    done("features", {"mean_keypoints": int(np.mean([len(f.pts) for f in feats]))})
    K = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1]])
    matches = match_sequence(feats, K)
    tracks = build_tracks(matches)
    done("matching", {"verified_pairs": len(matches), "tracks": len(tracks)})
    recon = Incremental([f.pts for f in feats], tracks, (w, h), f0, [f.colors for f in feats]).run()
    if recon is None:
        raise ReconstructionFailed("not enough camera movement or overlap to start a reconstruction")
    done("sfm", recon.stats)
    pts, cols, dense_stats = dense.fuse({i: images[i] for i in recon.rvecs}, recon)
    if len(pts) < 1000:
        pts, cols = recon.points, recon.colors
        dense_stats["fallback"] = "sparse"
    done("dense", dense_stats)
    T, orient_info = orient.orient(recon, pts)
    pts_v = orient.apply(T, pts)
    cams_v = orient.apply(T, np.array([recon.center(f) for f in sorted(recon.rvecs)]))
    done("orient", orient_info)
    return {"recon": recon, "points": pts_v, "colors": cols, "cameras": cams_v, "orient": orient_info,
            "dense": dense_stats, "timings_ms": timings, "frames": len(images)}


def bounds(points: np.ndarray, cameras: np.ndarray) -> dict:
    """Free-roam box: the inner part of the cloud, always including the capture path."""
    lo = np.minimum(np.percentile(points, 5, axis=0), cameras.min(axis=0))
    hi = np.maximum(np.percentile(points, 95, axis=0), cameras.max(axis=0))
    return {"min": lo.round(3).tolist(), "max": hi.round(3).tolist()}


def to_glb(points: np.ndarray, colors: np.ndarray) -> bytes:
    cloud = trimesh.PointCloud(vertices=points.astype(np.float32),
                               colors=np.c_[colors, np.full(len(colors), 255, np.uint8)])
    buf = io.BytesIO()
    trimesh.Scene(cloud).export(buf, file_type="glb")
    return buf.getvalue()


def scene_meta(result: dict) -> dict:
    recon: Reconstruction = result["recon"]
    return {
        "type": "3d",
        "camera": {"hfov_deg": round(math.degrees(2 * math.atan(recon.size[0] / (2 * recon.f))), 1),
                   "focal_px": round(recon.f, 1), "source": "phone_prior"},
        "sfm": recon.stats,
        "dense": result["dense"],
        "orient": result["orient"],
        "bounds": bounds(result["points"], result["cameras"]),
        "camera_path": result["cameras"].round(3).tolist(),
        "frames_used": result["frames"],
        "timings_ms": result["timings_ms"],
    }
