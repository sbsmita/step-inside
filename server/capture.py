"""Video / photo-set capture -> 3D scene. Runs locally on a background thread, or on AWS as a Step Functions
execution (frames go to S3, a Fargate worker reconstructs, a second step narrates)."""
import json
import logging
import os
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import boto3
import cv2 as cv
import numpy as np

from agents.director import direct
from agents.llm import label as llm_label
from agents.verifier import verify
from agents.voice import synthesize
from vision.scene25d import build as build_25d
from vision.sfm import keyframes
from vision.sfm.pipeline import ReconstructionFailed, reconstruct, scene_meta, to_glb

log = logging.getLogger(__name__)
STAGES = ["keyframes", "features", "matching", "sfm", "dense", "orient", "export"]
_pool = ThreadPoolExecutor(max_workers=1)
_side = ThreadPoolExecutor(max_workers=2)
_lock = threading.Lock()


def _jpg(img: np.ndarray) -> bytes:
    return cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def top_down(points: np.ndarray, colors: np.ndarray, cams: np.ndarray, size: int = 640) -> np.ndarray:
    """Bird's-eye view of the reconstruction with the capture path, drawn with OpenCV."""
    img = np.full((size, size, 3), 18, np.uint8)
    if len(points) == 0:
        return img
    xz = np.c_[points[:, 0], points[:, 2]]
    lo, hi = np.percentile(xz, 1, axis=0), np.percentile(xz, 99, axis=0)
    lo, hi = np.minimum(lo, cams[:, [0, 2]].min(0)), np.maximum(hi, cams[:, [0, 2]].max(0))
    s = (size - 40) / max(float((hi - lo).max()), 1e-6)

    def to_px(p):
        return ((p - lo) * s + 20).astype(int)

    order = np.argsort(points[:, 1])  # draw low points first so tall things stay visible
    px = to_px(xz[order])
    ok = (px >= 0).all(1) & (px < size).all(1)
    img[px[ok, 1], px[ok, 0]] = colors[order][ok][:, ::-1]
    path = to_px(cams[:, [0, 2]])
    cv.polylines(img, [path.reshape(-1, 1, 2)], False, (255, 160, 60), 2, cv.LINE_AA)
    for p in path:
        cv.circle(img, tuple(int(v) for v in p), 4, (255, 200, 120), -1, cv.LINE_AA)
    cv.putText(img, "capture path", (12, size - 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 120), 1, cv.LINE_AA)
    return img


def frames_from_upload(data: bytes, filename: str) -> list[np.ndarray]:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        return keyframes.sample_video(tmp.name)


class Jobs:
    def __init__(self, store, scenes):
        self.store, self.scenes = store, scenes
        self.sfn_arn = os.environ.get("STATE_MACHINE_ARN")
        self.sfn = boto3.client("stepfunctions") if self.sfn_arn else None

    # ---------- job records ----------
    def save(self, job: dict) -> None:
        job["updated"] = time.time()
        self.store.put(f"jobs/{job['id']}.json", json.dumps(job).encode(), "application/json")

    def get(self, job_id: str) -> dict | None:
        raw = self.store.get(f"jobs/{job_id}.json")
        return json.loads(raw) if raw else None

    def load_frames(self, job_id: str) -> list[np.ndarray]:
        frames, i = [], 0
        while (raw := self.store.get(f"jobs/{job_id}/frames/{i:03d}.jpg")) is not None:
            frames.append(cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_COLOR))
            i += 1
        return frames

    # ---------- submission ----------
    def submit(self, frames: list[np.ndarray], title: str) -> dict:
        job = {"id": uuid.uuid4().hex[:12], "status": "running", "stage": "keyframes", "title": title,
               "stages": {}, "scene_id": None, "error": None}
        if self.sfn_arn:
            for i, f in enumerate(frames):
                self.store.put(f"jobs/{job['id']}/frames/{i:03d}.jpg", _jpg(f), "image/jpeg")
            self.save(job)
            exe = self.sfn.start_execution(stateMachineArn=self.sfn_arn, name=job["id"],
                                           input=json.dumps({"job_id": job["id"]}))
            job["execution_arn"] = exe["executionArn"]
            self.save(job)
        else:
            self.save(job)
            _pool.submit(self._local, job, frames)
        return job

    def _local(self, job: dict, frames: list[np.ndarray]) -> None:
        self.run(job, frames)
        if job.get("scene_id"):
            self.enrich(job["scene_id"])

    def after_scene(self, scene_id: str) -> None:
        _side.submit(self.enrich, scene_id)

    # ---------- reconstruction ----------
    def _save_25d(self, job: dict, keys: list[np.ndarray], reason: str) -> None:
        """Verifier decided 3D isn't recoverable: build a single-view scene from the sharpest frame instead."""
        sharpest = max(keys, key=lambda k: cv.Laplacian(cv.cvtColor(k, cv.COLOR_BGR2GRAY), cv.CV_64F).var())
        scene = build_25d(sharpest)
        scene.meta["fallback_reason"] = reason
        files = {"scene.glb": scene.glb, "input.jpg": _jpg(sharpest)}
        files.update({f"{n}.jpg": _jpg(p) for n, p in scene.previews.items()})
        job["scene_id"] = self.scenes.save(scene.meta, files, job["title"], "photo")
        job["verifier"] = {"verdict": "failed", "next_action": "fallback_2_5d", "reasons": [reason],
                           "advice": ["Record again moving sideways about a metre, slowly, with the same area in view."]}
        job["status"] = "done"

    def run(self, job: dict, frames: list[np.ndarray]) -> None:
        def stage(name, info):
            with _lock:
                job["stages"][name] = info
                nxt = STAGES.index(name) + 1
                job["stage"] = STAGES[nxt] if nxt < len(STAGES) else "done"
                self.save(job)

        keys: list[np.ndarray] = []
        try:
            keys, kinfo = keyframes.select(frames)
            stage("keyframes", kinfo)
            if len(keys) < 3:
                raise ReconstructionFailed("need at least 3 sharp views; record a slower, longer sweep")
            result = reconstruct(keys, on_stage=stage)
            meta = scene_meta(result)
            meta["keyframe_count"] = len(keys)
            meta["verifier"] = verdict = verify(meta)
            job["verifier"] = verdict
            if verdict["next_action"] == "fallback_2_5d":
                self._save_25d(job, keys, "; ".join(verdict["reasons"]))
            else:
                files = {
                    "scene.glb": to_glb(result["points"], result["colors"]),
                    "input.jpg": _jpg(keys[0]),
                    "layers.jpg": _jpg(top_down(result["points"], result["colors"], result["cameras"])),
                }
                scene_id = self.scenes.save(meta, files, job["title"], "capture")
                stage("export", {"scene_id": scene_id})
                job.update(status="done", scene_id=scene_id)
        except ReconstructionFailed as e:
            if keys:
                self._save_25d(job, keys, str(e))
            else:
                job.update(status="failed", error=str(e))
        except Exception as e:  # surface unexpected failures to the client instead of a stuck "running" job
            traceback.print_exc()
            job.update(status="failed", error=f"internal error: {type(e).__name__}")
        self.save(job)

    # ---------- narration ----------
    def enrich(self, scene_id: str) -> None:
        """Scene Director writes the welcome from OpenCV's measurements; Polly voices it."""
        meta = self.scenes.get(scene_id)
        if meta is None:
            return
        try:
            out, source = direct(meta)
            audio = synthesize(out["narration"])
            if audio:
                self.store.put(f"scenes/{scene_id}/narration.mp3", audio, "audio/mpeg")
            meta["narration"] = {"text": out["narration"], "suggested_title": out["title"], "source": source,
                                 "model": llm_label() if source == "bedrock" else None, "audio": bool(audio)}
            self.scenes.put_meta(meta)
        except Exception:
            log.exception("narration failed for %s", scene_id)
