"""Incremental structure-from-motion built from OpenCV's calib3d: findEssentialMat, recoverPose, triangulatePoints,
solvePnPRansac and solvePnPRefineLM, with periodic bundle adjustment."""
import logging
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from .bundle import bundle_adjust, rodrigues

log = logging.getLogger(__name__)

MIN_SHARED_INIT = 60
MIN_INIT_ANGLE_DEG = 2.0
MIN_TRI_ANGLE_DEG = 1.5
MAX_REPROJ_PX = 4.0
MIN_PNP_POINTS = 20
MIN_PNP_INLIERS = 18


@dataclass
class Reconstruction:
    f: float
    c: np.ndarray
    size: tuple[int, int]
    rvecs: dict[int, np.ndarray]
    tvecs: dict[int, np.ndarray]
    points: np.ndarray
    colors: np.ndarray
    point_track: np.ndarray
    stats: dict = field(default_factory=dict)

    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.c[0]], [0, self.f, self.c[1]], [0, 0, 1]])

    def center(self, frame: int) -> np.ndarray:
        R = cv.Rodrigues(self.rvecs[frame])[0]
        return -R.T @ self.tvecs[frame]


class Incremental:
    def __init__(self, pts: list[np.ndarray], tracks: list[dict[int, int]], size: tuple[int, int],
                 f0: float, colors: list[np.ndarray] | None = None):
        self.pts, self.tracks, self.size = pts, tracks, size
        self.colors = colors
        self.f = float(f0)
        self.c = np.array([size[0] / 2, size[1] / 2], np.float64)
        self.rvecs: dict[int, np.ndarray] = {}
        self.tvecs: dict[int, np.ndarray] = {}
        self.track_point = -np.ones(len(tracks), dtype=int)
        self.points: list[np.ndarray] = []
        self.point_track: list[int] = []
        self.frame_tracks: dict[int, list[int]] = {}
        for t, tr in enumerate(tracks):
            for fr in tr:
                self.frame_tracks.setdefault(fr, []).append(t)
        self.failed: set[int] = set()
        self.log: list[dict] = []

    # ---------- geometry helpers ----------
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.c[0]], [0, self.f, self.c[1]], [0, 0, 1]])

    def _P(self, fr: int) -> np.ndarray:
        R = cv.Rodrigues(self.rvecs[fr])[0]
        return self.K() @ np.hstack([R, self.tvecs[fr].reshape(3, 1)])

    def _center(self, fr: int) -> np.ndarray:
        R = cv.Rodrigues(self.rvecs[fr])[0]
        return -R.T @ self.tvecs[fr]

    def _reproj(self, X: np.ndarray, fr: int, kp: int) -> tuple[float, float]:
        """(pixel error, depth) of point X in frame fr."""
        R = cv.Rodrigues(self.rvecs[fr])[0]
        cam = R @ X + self.tvecs[fr]
        if cam[2] <= 1e-6:
            return np.inf, cam[2]
        uv = self.f * cam[:2] / cam[2] + self.c
        return float(np.linalg.norm(uv - self.pts[fr][kp])), cam[2]

    @staticmethod
    def _angle(X, C1, C2) -> float:
        a, b = X - C1, X - C2
        cosv = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        return float(np.degrees(np.arccos(np.clip(cosv, -1, 1))))

    def _triangulate_track(self, t: int) -> bool:
        views = [fr for fr in self.tracks[t] if fr in self.rvecs]
        if len(views) < 2:
            return False
        centers = {fr: self._center(fr) for fr in views}
        best = None
        for i in range(min(len(views), 6)):
            for j in range(i + 1, len(views)):
                a, b = views[i], views[j]
                x1 = self.pts[a][self.tracks[t][a]].reshape(2, 1).astype(np.float64)
                x2 = self.pts[b][self.tracks[t][b]].reshape(2, 1).astype(np.float64)
                Xh = cv.triangulatePoints(self._P(a), self._P(b), x1, x2).ravel()
                if abs(Xh[3]) < 1e-12:
                    continue
                X = Xh[:3] / Xh[3]
                ang = self._angle(X, centers[a], centers[b])
                if best is None or ang > best[1]:
                    best = (X, ang)
        if best is None or best[1] < MIN_TRI_ANGLE_DEG:
            return False
        X = best[0]
        good = 0
        for fr in views:
            err, depth = self._reproj(X, fr, self.tracks[t][fr])
            if depth <= 0:
                return False
            good += err < MAX_REPROJ_PX
        if good < 2:
            return False
        self.track_point[t] = len(self.points)
        self.points.append(X)
        self.point_track.append(t)
        return True

    # ---------- initialization ----------
    def _shared(self) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for tr in self.tracks:
            frs = sorted(tr)
            for i in range(len(frs)):
                for j in range(i + 1, len(frs)):
                    counts[(frs[i], frs[j])] = counts.get((frs[i], frs[j]), 0) + 1
        return counts

    def _try_init(self, a: int, b: int) -> dict | None:
        ts = [t for t in self.frame_tracks.get(a, []) if b in self.tracks[t]]
        p1 = np.float64([self.pts[a][self.tracks[t][a]] for t in ts])
        p2 = np.float64([self.pts[b][self.tracks[t][b]] for t in ts])
        E, emask = cv.findEssentialMat(p1, p2, self.K(), cv.USAC_MAGSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3):
            return None
        _, hmask = cv.findHomography(p1, p2, cv.RANSAC, 4.0)
        e_inl = int(emask.sum())
        h_ratio = (int(hmask.sum()) / max(e_inl, 1)) if hmask is not None else 0.0
        n, R, tvec, pmask = cv.recoverPose(E, p1, p2, self.K(), mask=emask.copy())
        if n < MIN_SHARED_INIT // 2:
            return None
        P1 = self.K() @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K() @ np.hstack([R, tvec])
        inl = pmask.ravel() > 0
        Xh = cv.triangulatePoints(P1, P2, p1[inl].T, p2[inl].T)
        X = (Xh[:3] / Xh[3]).T
        C2 = (-R.T @ tvec).ravel()
        angles = [self._angle(x, np.zeros(3), C2) for x in X]
        return {"a": a, "b": b, "R": R, "t": tvec.ravel(), "inliers": int(inl.sum()),
                "angle": float(np.median(angles)) if angles else 0.0, "h_ratio": h_ratio}

    def initialize(self) -> bool:
        shared = sorted(((n, p) for p, n in self._shared().items() if n >= MIN_SHARED_INIT), reverse=True)[:40]
        best, best_score = None, -1.0
        for n, (a, b) in shared:
            cand = self._try_init(a, b)
            if cand is None or cand["angle"] < MIN_INIT_ANGLE_DEG:
                continue
            # Prefer many inliers with real baseline; a homography explaining everything means no parallax.
            score = cand["inliers"] * min(cand["angle"], 8.0) / 8.0 * (1.0 if cand["h_ratio"] < 0.8 else 0.3)
            if score > best_score:
                best, best_score = cand, score
        if best is None:
            return False
        a, b = best["a"], best["b"]
        self.rvecs[a], self.tvecs[a] = np.zeros(3), np.zeros(3)
        self.rvecs[b], self.tvecs[b] = cv.Rodrigues(best["R"])[0].ravel(), best["t"]
        self.init_frame = a
        made = sum(self._triangulate_track(t) for t in self.frame_tracks[a] if b in self.tracks[t])
        self.log.append({"event": "init", "pair": [a, b], "angle": round(best["angle"], 2),
                         "inliers": best["inliers"], "points": made})
        return made >= MIN_SHARED_INIT // 2

    # ---------- registration ----------
    def _correspondences(self, fr: int):
        ts = [t for t in self.frame_tracks.get(fr, []) if self.track_point[t] >= 0]
        obj = np.float64([self.points[self.track_point[t]] for t in ts]).reshape(-1, 3)
        img = np.float64([self.pts[fr][self.tracks[t][fr]] for t in ts]).reshape(-1, 2)
        return ts, obj, img

    def register_next(self) -> bool:
        cands = []
        for fr in self.frame_tracks:
            if fr in self.rvecs or fr in self.failed:
                continue
            n = sum(self.track_point[t] >= 0 for t in self.frame_tracks[fr])
            if n >= MIN_PNP_POINTS:
                cands.append((n, fr))
        for n, fr in sorted(cands, reverse=True):
            _, obj, img = self._correspondences(fr)
            ok, rvec, tvec, inl = cv.solvePnPRansac(obj, img, self.K(), None, iterationsCount=300,
                                                    reprojectionError=MAX_REPROJ_PX, confidence=0.999)
            if not ok or inl is None or len(inl) < MIN_PNP_INLIERS:
                self.failed.add(fr)
                continue
            idx = inl.ravel()
            rvec, tvec = cv.solvePnPRefineLM(obj[idx], img[idx], self.K(), None, rvec, tvec)
            self.rvecs[fr], self.tvecs[fr] = rvec.ravel(), tvec.ravel()
            made = sum(self._triangulate_track(t) for t in self.frame_tracks[fr] if self.track_point[t] < 0)
            self.log.append({"event": "register", "frame": fr, "pnp_inliers": int(len(idx)), "new_points": made})
            return True
        return False

    # ---------- bundle adjustment ----------
    def bundle(self, max_nfev: int = 30) -> float:
        frames = sorted(self.rvecs)
        cam_of = {fr: i for i, fr in enumerate(frames)}
        cam_idx, pt_idx, uv = [], [], []
        for pid, t in enumerate(self.point_track):
            if self.track_point[t] != pid:
                continue
            for fr, kp in self.tracks[t].items():
                if fr in cam_of:
                    cam_idx.append(cam_of[fr])
                    pt_idx.append(pid)
                    uv.append(self.pts[fr][kp])
        if not uv:
            return 0.0
        cam_idx, pt_idx, uv = np.array(cam_idx), np.array(pt_idx), np.float64(uv)
        rv = np.array([self.rvecs[fr] for fr in frames])
        tv = np.array([self.tvecs[fr] for fr in frames])
        P = np.array(self.points)
        # Focal length stays fixed at the phone's known field of view: a mostly-sideways sweep can't
        # pin it down (a wrong focal fits equally well with rescaled depth), so refining it only adds drift.
        _, rv, tv, P, err = bundle_adjust(self.f, self.c, rv, tv, P, cam_idx, pt_idx, uv,
                                          fixed_cam=cam_of[self.init_frame], refine_focal=False,
                                          max_nfev=max_nfev)
        for i, fr in enumerate(frames):
            self.rvecs[fr], self.tvecs[fr] = rv[i], tv[i]
        self.points = list(P)
        # Drop points that still don't fit: their worst observation is far off.
        worst = np.zeros(len(P))
        np.maximum.at(worst, pt_idx, err)
        for pid in np.nonzero(worst > 2 * MAX_REPROJ_PX)[0]:
            self.track_point[self.point_track[pid]] = -1
        return float(np.median(err))

    # ---------- driver ----------
    def run(self) -> Reconstruction | None:
        if not self.initialize():
            return None
        self.bundle()
        next_ba = 4
        while self.register_next():
            if len(self.rvecs) >= next_ba:
                self.bundle()
                next_ba = int(next_ba * 1.4) + 1
        median_err = self.bundle(max_nfev=80)
        return self._result(median_err)

    def _result(self, median_err: float) -> Reconstruction:
        live = [(pid, t) for pid, t in enumerate(self.point_track) if self.track_point[t] == pid]
        pts = np.array([self.points[pid] for pid, _ in live]).reshape(-1, 3)
        tracks = np.array([t for _, t in live], dtype=int)
        colors = np.zeros((len(live), 3), np.uint8)
        if self.colors is not None:
            for i, (_, t) in enumerate(live):
                fr, kp = next(iter(self.tracks[t].items()))
                colors[i] = self.colors[fr][kp]
        n_frames = len(self.frame_tracks)
        stats = {"frames": n_frames, "registered": len(self.rvecs),
                 "registered_frac": round(len(self.rvecs) / max(n_frames, 1), 3),
                 "points": int(len(pts)), "median_reproj_px": round(median_err, 3), "focal_px": round(self.f, 1),
                 "mean_track_len": round(float(np.mean([len(self.tracks[t]) for t in tracks])) if len(tracks) else 0, 2)}
        return Reconstruction(self.f, self.c, self.size, dict(self.rvecs), dict(self.tvecs), pts, colors, tracks, stats)
