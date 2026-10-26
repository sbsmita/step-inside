"""The animator agent: every painted moment becomes a short video clip in which its characters act the lines.

Backends (`VIDEO_MODEL`):
- `luma.ray-v2:0` (default): Luma AI Ray2 on Bedrock, us-west-2, image-to-video, 5 s, 21:9. The account allows ONE
  job at a time and a clip takes 2-5 minutes (~$0.75 per second at 540p, ~$3.75 a clip). Luma is sold through AWS
  Marketplace: the account owner must accept its agreement once before the first clip.
- `amazon.nova-reel-v1:0` / `v1:1`: Amazon Nova Reel, us-east-1, 6 s at 1280x720 (exact), up to 10/3 jobs at once,
  ~90 s and ~$0.48 a clip. **Both versions answer "reached the end of its life" since 2026** (checked Oct 26,
  although `get-foundation-model-availability` still says AUTHORIZED); kept for accounts or regions where a Nova
  Reel version is still served.

Clips go through one priority queue (oldest story first, then its moments in order) served by up to
`VIDEO_CONCURRENCY` workers. Bedrock writes each clip to `VIDEO_BUCKET` (in the model's region). A daily cap applies
(`VIDEO_DAILY_LIMIT`). A retired model or missing permission switches the animator off for the process, once.

    VIDEO_BUCKET=<bucket> python -m agents.animator still.jpg "the egg wobbles and falls off the wall" out.mp4
runs one clip by hand (it prints how long it took and where Bedrock put it)."""
import base64
import itertools
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from functools import lru_cache

import boto3
import cv2 as cv
import numpy as np
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from . import budget

log = logging.getLogger(__name__)
PREFIX = "clips/"
QUOTA_WAIT_S = 15.0
NOVA_SIZE = (1280, 720)
NOVA_PROMPT_MAX = 512
END_STATE = {"fall": "By the end {who} lies on the ground below, where it fell.",
             "walk-left": "By the end {who} has walked out of the left edge.",
             "walk-right": "By the end {who} has walked out of the right edge.",
             "rise": "By the end {who} has risen high, near the top of the picture.",
             "jump": "{who} jumps up and lands where it started.",
             "sway": "{who} keeps swaying gently in place."}


# ---------- configuration (read at call time, so tests and the CLI can set the environment) ----------
def model() -> str:
    # Nova Reel (v1:0, v1:1) answers "reached the end of its life" since 2026; Luma Ray2 is the active model.
    return os.environ.get("VIDEO_MODEL", "luma.ray-v2:0")


def is_nova() -> bool:
    return model().startswith("amazon.nova-reel")


def region() -> str:
    return os.environ.get("VIDEO_REGION") or ("us-east-1" if is_nova() else "us-west-2")


def concurrency() -> int:
    default = {"amazon.nova-reel-v1:0": 8, "amazon.nova-reel-v1:1": 3}.get(model(), 1)
    try:
        return max(1, int(os.environ.get("VIDEO_CONCURRENCY", default)))
    except ValueError:
        return default


def label() -> str:
    return ("Amazon Nova Reel on Bedrock (image-to-video, 6 s)" if is_nova()
            else "Luma AI Ray2 on Bedrock (image-to-video, 5 s)")


def bucket() -> str | None:
    """Where Bedrock writes clips. `VIDEO_BUCKET` if set (an empty value turns video off); otherwise found by itself:
    the bucket named after this AWS account and the model's region (`step-inside-clips-<account>-<region>`, the name
    the deploy creates), if it exists. Looked up once per process; no bucket means no video."""
    if "VIDEO_BUCKET" in os.environ:
        return os.environ["VIDEO_BUCKET"] or None
    reg = region()
    if reg not in _found:
        _found[reg] = _discover(reg)
    return _found[reg]


_found: dict = {}


def _discover(reg: str) -> str | None:
    quick = Config(connect_timeout=3, read_timeout=5, retries={"total_max_attempts": 1, "mode": "standard"})
    try:
        account = boto3.client("sts", config=quick).get_caller_identity()["Account"]
        name = f"step-inside-clips-{account}-{reg}"
        boto3.client("s3", region_name=reg, config=quick).head_bucket(Bucket=name)
        log.info("video clips go to s3://%s", name)
        return name
    except Exception as e:  # noqa: BLE001 - no login, no bucket, no permission: video stays off
        log.warning("no video bucket found (%s); moments stay living paintings", type(e).__name__)
        return None


def resolution() -> str:
    if is_nova():
        return "720p"
    r = os.environ.get("VIDEO_RESOLUTION", "540p")
    return r if r in ("540p", "720p") else "540p"


def poll_s() -> float:
    return 5.0 if is_nova() else 10.0


def give_up_s() -> float:
    return float(os.environ.get("VIDEO_GIVE_UP_S", "480" if is_nova() else "600"))


def enabled() -> bool:
    return (os.environ.get("STEP_INSIDE_LLM", "bedrock") == "bedrock"
            and os.environ.get("STEP_INSIDE_VIDEO", "on") != "off" and bool(bucket()) and not _state["disabled"])


# Kept as module attributes for callers that read them (the conductor's state and drawer).
MODEL = model()
LABEL = label()


class AnimatorError(Exception):
    def __init__(self, reason: str, retry: bool = False):
        super().__init__(reason)
        self.reason, self.retry = reason, retry


@dataclass
class Clip:
    mp4: bytes
    invocation: str
    seconds: float
    resolution: str
    prompt: str
    from_still: bool


# ---------- the start frame ----------
def start_frame(img: np.ndarray, size: tuple = NOVA_SIZE) -> np.ndarray:
    """The picture as Nova Reel needs it: exactly 1280x720. A 16:9 picture is resized; any other shape is fitted
    inside and the margins filled by mirroring the picture's own edges (never stretched, never black bars)."""
    w, h = size
    ih, iw = img.shape[:2]
    if abs(iw / ih - w / h) < 0.02:
        return cv.resize(img, (w, h), interpolation=cv.INTER_AREA)
    s = min(w / iw, h / ih)
    fit = cv.resize(img, (max(1, round(iw * s)), max(1, round(ih * s))), interpolation=cv.INTER_AREA)
    top, left = (h - fit.shape[0]) // 2, (w - fit.shape[1]) // 2
    out = cv.copyMakeBorder(fit, top, h - fit.shape[0] - top, left, w - fit.shape[1] - left, cv.BORDER_REFLECT)
    return out[:h, :w]


def start_frame_jpg(jpg: bytes) -> bytes:
    img = cv.imdecode(np.frombuffer(jpg, np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise AnimatorError("the start frame could not be read")
    return cv.imencode(".jpg", start_frame(img), [cv.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


# ---------- the prompt ----------
def clip_prompt(action: dict, look: str | None, style: str, see: str, attempt: int = 1) -> str:
    """What the clip should show. The characters act the moment out; the picture stays as painted. Nova Reel reads
    at most 512 characters, Luma at most 300 tokens."""
    who = action.get("subject") or "the characters"
    scene = " ".join((see or "").split(".")[0].split()[:18])
    style_bit = " ".join((style or "").split()[:8])
    if not (action.get("motion") or "").strip():  # nothing described: gentle living motion
        parts = [f"{scene}.", f"{who}: {look}." if look else "",
                 f"Gentle living motion: {who} breathe, look around and gesture while speaking; leaves, water and "
                 f"clouds move softly. Locked-off camera: no pan, no zoom, no cut; same painted style ({style_bit}); "
                 f"no text."]
    else:
        parts = [f"{who}: {look}." if look else "", f"{action.get('motion', '')}.", f"{scene}.",
                 f"Storybook film: {who} acts this out with natural movement while speaking. One continuous shot in "
                 f"this same place; locked-off camera: no pan, no zoom, no cut; same painted style ({style_bit}); no "
                 f"new characters, no text."]
    if attempt > 1:
        parts.append(END_STATE.get(action.get("path"), "").format(who=who))
        parts.append("The movement is large and clear, all within the same view: never cut to another place.")
    text = " ".join(p for p in parts if p.strip(". "))
    text = " ".join(text.split())
    limit = NOVA_PROMPT_MAX if is_nova() else 1200
    if len(text) > limit:  # keep whole words; the action comes first, so it survives the cut
        text = text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "."
    return text


def anonymize(prompt: str, cast: list[dict], subject: str = "") -> str:
    """The prompt with character names replaced by what they look like: video models refuse prompts naming
    well-known characters ("ip prompt moderation failed")."""
    cast_names = {c["name"] for c in cast if c.get("name")}
    if subject and subject not in cast_names:  # "Cinderella's fairy godmother" -> "a kind figure"
        prompt = re.sub(rf"(\b[\w']+'s\s+)?\b{re.escape(subject)}\b", "a kind figure", prompt, flags=re.I)
    names = sorted(cast_names, key=len, reverse=True)
    for n in names:
        look = next((c.get("look") for c in cast if c.get("name") == n and c.get("look")), "")
        words = re.sub(r"^(a|an|the)\s+", "", " ".join(look.split()[:8]).rstrip(",.;"), flags=re.I)
        desc = f"the {words}" if words else "the character"
        prompt = re.sub(rf"\b{re.escape(n)}('s)?\b", lambda m: desc + (m.group(1) or ""), prompt, flags=re.I)
    return prompt


def moderation_refusal(reason: str) -> bool:
    return "moderation" in (reason or "").lower()


# ---------- Bedrock ----------
@lru_cache(maxsize=4)
def _runtime(reg: str):
    return boto3.client("bedrock-runtime", region_name=reg,
                        config=Config(read_timeout=60, connect_timeout=10,
                                      retries={"total_max_attempts": 2, "mode": "standard"}))


@lru_cache(maxsize=4)
def _s3(reg: str):
    return boto3.client("s3", region_name=reg)


def request_body(still_jpg: bytes | None, prompt: str, loop: bool, seed: int = 0) -> dict:
    """The modelInput for the configured model."""
    if is_nova():
        params = {"text": prompt[:NOVA_PROMPT_MAX]}
        if still_jpg:
            params["images"] = [{"format": "jpeg", "source": {"bytes": base64.b64encode(start_frame_jpg(still_jpg)).decode()}}]
        return {"taskType": "TEXT_VIDEO", "textToVideoParams": params,
                "videoGenerationConfig": {"durationSeconds": 6, "fps": 24, "dimension": "1280x720",
                                          "seed": int(seed) % 2147483646}}
    body = {"prompt": prompt, "aspect_ratio": luma_aspect(still_jpg), "loop": bool(loop), "duration": "5s",
            "resolution": resolution()}
    if still_jpg:
        body["keyframes"] = {"frame0": {"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(luma_frame(still_jpg)).decode()}}}
    return body


LUMA_MAX_W = 1536  # Luma refuses a start frame wider than 1552 px


def luma_frame(still_jpg: bytes) -> bytes:
    """The start frame no wider than Luma allows (same shape)."""
    img = cv.imdecode(np.frombuffer(still_jpg, np.uint8), cv.IMREAD_COLOR)
    if img is None or img.shape[1] <= LUMA_MAX_W:
        return still_jpg
    s = LUMA_MAX_W / img.shape[1]
    small = cv.resize(img, (LUMA_MAX_W, max(1, round(img.shape[0] * s))), interpolation=cv.INTER_AREA)
    return cv.imencode(".jpg", small, [cv.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


LUMA_ASPECTS = {"1:1": 1.0, "16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4, "21:9": 21 / 9, "9:21": 9 / 21}


def luma_aspect(still_jpg: bytes | None) -> str:
    """Luma's aspect ratio closest to the painting's, so the clip lies exactly over it on the curved screen."""
    img = cv.imdecode(np.frombuffer(still_jpg, np.uint8), cv.IMREAD_REDUCED_GRAYSCALE_4) if still_jpg else None
    if img is None:
        return "16:9"
    r = img.shape[1] / img.shape[0]
    return min(LUMA_ASPECTS, key=lambda k: abs(np.log(LUMA_ASPECTS[k] / r)))


def _start(still_jpg: bytes | None, prompt: str, loop: bool, seed: int = 0) -> str:
    resp = _runtime(region()).start_async_invoke(
        modelId=model(), clientRequestToken=uuid.uuid4().hex, modelInput=request_body(still_jpg, prompt, loop, seed),
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{bucket()}/{PREFIX.rstrip('/')}"}})
    return resp["invocationArn"]


def _wait(arn: str, deadline: float) -> dict:
    while True:
        try:
            rec = _runtime(region()).get_async_invoke(invocationArn=arn)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                time.sleep(10)
                continue
            raise
        if rec["status"] != "InProgress":
            return rec
        if time.monotonic() > deadline:
            raise AnimatorError("the video model took too long", retry=True)
        time.sleep(poll_s())


def _folder(arn: str, rec: dict) -> tuple[str, str]:
    """(bucket, key prefix) of the folder Bedrock wrote this invocation into."""
    inv = arn.rsplit("/", 1)[-1]
    uri = rec.get("outputDataConfig", {}).get("s3OutputDataConfig", {}).get("s3Uri") or f"s3://{bucket()}/{PREFIX}"
    b, _, prefix = uri.removeprefix("s3://").partition("/")
    prefix = prefix.strip("/")
    if prefix.endswith(inv):  # Nova reports the invocation's own folder
        return b, prefix + "/"
    return b, f"{prefix}/{inv}/" if prefix else f"{inv}/"


def _failure(arn: str, rec: dict) -> str:
    """Why a job failed: Nova writes video-generation-status.json with the reason (e.g. the content filter)."""
    reason = rec.get("failureMessage") or rec.get("status", "failed")
    try:
        b, folder = _folder(arn, rec)
        status = json.loads(_s3(region()).get_object(Bucket=b, Key=folder + "video-generation-status.json")["Body"].read())
        full = status.get("fullVideo") or {}
        shots = status.get("shots") or [{}]
        reason = full.get("failureMessage") or shots[0].get("failureMessage") or full.get("failureType") or reason
    except Exception:  # noqa: BLE001 - the status file is a nicety
        pass
    return str(reason)


def _download(arn: str, rec: dict) -> bytes:
    """The clip: Nova Reel writes <prefix>/<invocation id>/output.mp4; Luma's key isn't documented, so list."""
    b, folder = _folder(arn, rec)
    s3 = _s3(region())
    if is_nova():
        try:
            return s3.get_object(Bucket=b, Key=folder + "output.mp4")["Body"].read()
        except ClientError:
            pass
    inv = arn.rsplit("/", 1)[-1]
    for p in (folder, folder.split(inv)[0]):
        listed = s3.list_objects_v2(Bucket=b, Prefix=p).get("Contents", [])
        mp4s = sorted((o for o in listed if o["Key"].endswith(".mp4") and inv in o["Key"]
                       and "shot_" not in o["Key"].rsplit("/", 1)[-1]),
                      key=lambda o: o["LastModified"], reverse=True)
        if mp4s:
            log.info("clip at s3://%s/%s", b, mp4s[0]["Key"])
            return s3.get_object(Bucket=b, Key=mp4s[0]["Key"])["Body"].read()
    raise AnimatorError("the clip was not found in S3")


def _generate(still_jpg: bytes, prompt: str, loop: bool, deadline: float, on_status=None, seed: int = 0) -> Clip:
    t0 = time.monotonic()
    from_still = True
    while True:
        try:
            arn = _start(still_jpg if from_still else None, prompt, loop, seed)
            break
        except ClientError as e:
            code, msg = e.response["Error"]["Code"], e.response["Error"].get("Message", "")
            if code in ("ThrottlingException", "ServiceQuotaExceededException", "TooManyRequestsException"):
                if time.monotonic() > deadline:  # every slot stayed busy (other stories, another process)
                    raise AnimatorError("the video model stayed busy", retry=False) from e
                time.sleep(QUOTA_WAIT_S)
                continue
            if code == "ValidationException" and from_still and ("keyframe" in msg.lower() or "image" in msg.lower()):
                log.warning("image-to-video refused (%s); making the clip from the words alone", msg)
                from_still = False
                continue
            if code == "AccessDeniedException":
                _state["disabled"] = f"video unavailable: {msg[:120]}"
                raise AnimatorError("video unavailable: permissions") from e
            if code == "ResourceNotFoundException" or "end of its life" in msg:
                # A retired model (Nova Reel reached end of life in 2026 although the account still lists it as
                # authorised): stop asking for every moment and say so once.
                _state["disabled"] = f"video model {model()} is retired"
                raise AnimatorError(_state["disabled"]) from e
            raise AnimatorError(f"{code}: {msg[:160]}") from e
        except BotoCoreError as e:
            raise AnimatorError(f"video model unreachable: {e}", retry=True) from e
    if on_status:
        on_status("generating")
    rec = _wait(arn, min(deadline, time.monotonic() + give_up_s()))
    if rec["status"] != "Completed":
        reason = _failure(arn, rec)
        raise AnimatorError(reason, retry="RATE_LIMIT" in reason.upper() or "INTERNAL" in reason.upper())
    return Clip(_download(arn, rec), arn, round(time.monotonic() - t0, 1), resolution(), prompt, from_still)


# ---------- the shared queue: up to VIDEO_CONCURRENCY clips at once ----------
_state = {"disabled": None}
_jobs: queue.PriorityQueue = queue.PriorityQueue()
_seq = itertools.count()
_worker_lock = threading.Lock()
_workers: list = []
_running = [0]  # jobs holding a slot
_slot = threading.Condition()


def _work() -> None:
    while True:
        item = _jobs.get()
        with _slot:  # never more jobs at once than the account allows (VIDEO_CONCURRENCY), whatever threads exist
            if _running[0] >= concurrency():
                _jobs.put(item)  # back in the queue, in priority order, for whichever worker gets the next slot
                _slot.wait_for(lambda: _running[0] < concurrency())
                continue
            _running[0] += 1
        _, _, job = item
        fut: Future = job["future"]
        if not fut.set_running_or_notify_cancel():
            with _slot:
                _running[0] -= 1
                _slot.notify_all()
            continue
        try:
            if time.monotonic() > job["expires_at"]:
                raise AnimatorError("out of time")
            if _state["disabled"]:
                raise AnimatorError(_state["disabled"])
            if not budget.take("video"):
                raise AnimatorError("daily video limit")
            fut.set_result(_generate(job["still"], job["prompt"], job["loop"], job["expires_at"], job["on_status"],
                                     job["seed"]))
        except Exception as e:  # noqa: BLE001 - every failure goes back to the story that asked
            fut.set_exception(e if isinstance(e, AnimatorError) else AnimatorError(f"animator error: {e}"))
        finally:
            with _slot:
                _running[0] -= 1
                _slot.notify_all()


def submit(priority: tuple, still_jpg: bytes, prompt: str, loop: bool, expires_at: float, on_status=None,
           seed: int = 0) -> Future:
    """Queue one clip. `expires_at` is a time.monotonic() deadline; the Future resolves to a Clip or AnimatorError."""
    fut: Future = Future()
    _jobs.put((priority, next(_seq), {"future": fut, "still": still_jpg, "prompt": prompt, "loop": loop,
                                      "expires_at": expires_at, "on_status": on_status, "seed": seed}))
    with _worker_lock:
        while len(_workers) < concurrency():
            t = threading.Thread(target=_work, daemon=True, name=f"animator-{len(_workers)}")
            t.start()
            _workers.append(t)
    return fut


def pending() -> int:
    return _jobs.qsize()


def running() -> int:
    return _running[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 4 or not bucket():
        sys.exit("usage: VIDEO_BUCKET=<bucket in the model's region> python -m agents.animator still.jpg PROMPT out.mp4 [--loop]")
    with open(sys.argv[1], "rb") as f:
        still = f.read()
    clip = _generate(still, sys.argv[2], "--loop" in sys.argv, time.monotonic() + 900)
    with open(sys.argv[3], "wb") as f:
        f.write(clip.mp4)
    print(f"{model()}: {clip.seconds} s, {len(clip.mp4) / 1e6:.1f} MB, {clip.invocation}, from the still: {clip.from_still}")
