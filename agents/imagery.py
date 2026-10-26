"""Fast pictures for live stories on Amazon Bedrock: Stability AI Stable Image Core paints a wide 2304x960 moment in
about 3-4 s, and parallel calls take no longer. The account allows only a few calls per minute per model, so calls go
through a token bucket per model (never tripping the service's throttling), and SD3.5 Large takes the overflow."""
import base64
import json
import logging
import os
import threading
import time
from functools import lru_cache

import boto3
import cv2 as cv
import numpy as np
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from . import budget

log = logging.getLogger(__name__)
REGION = os.environ.get("IMAGE_REGION", "us-west-2")  # where Stable Image Core and SD3.5 Large run on demand
# (name, model id, requests per minute the account allows), in order of preference.
MODELS = [
    ("stable-image-core", os.environ.get("CORE_MODEL", "stability.stable-image-core-v1:1"),
     float(os.environ.get("CORE_RPM", "9"))),
    ("sd3.5-large", os.environ.get("SD35_MODEL", "stability.sd3-5-large-v1:0"), float(os.environ.get("SD35_RPM", "2"))),
]
LABELS = {"stable-image-core": "Stability AI Stable Image Core", "sd3.5-large": "Stability AI SD3.5 Large"}
RETRY_CODES = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException", "ModelTimeoutException",
               "InternalServerException"}


class Bucket:
    """A per-minute request quota as a token bucket: full at the start, refilling steadily."""

    def __init__(self, per_minute: float):
        self.rate, self.capacity = per_minute / 60.0, max(1.0, per_minute)
        self.tokens, self.t = self.capacity, time.monotonic()
        self.lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
        self.t = now

    def take(self) -> float:
        """0 if a request may go now (and counts it), else the seconds until one may."""
        with self.lock:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return 0.0
            return (1 - self.tokens) / self.rate

    def drain(self) -> None:
        """The service throttled us anyway (another process shares the quota): count the bucket as empty."""
        with self.lock:
            self._refill()
            self.tokens = min(self.tokens, 0.0)


_buckets = {name: Bucket(rpm) for name, _, rpm in MODELS}


def enabled() -> bool:
    return (os.environ.get("STEP_INSIDE_LLM", "bedrock") == "bedrock"
            and os.environ.get("STEP_INSIDE_IMAGES", "stability") == "stability")


@lru_cache(maxsize=1)
def _client():
    return boto3.client("bedrock-runtime", region_name=REGION,
                        config=Config(read_timeout=60, connect_timeout=10, max_pool_connections=16,
                                      retries={"total_max_attempts": 1, "mode": "standard"}))


def _call(name: str, model: str, prompt: str, negative: str, aspect: str, seed: int):
    """(image or None, worth trying again?)"""
    body = {"prompt": prompt[:9000], "negative_prompt": negative[:9000], "aspect_ratio": aspect,
            "output_format": "jpeg", "seed": int(seed) % 4294967294}
    if name == "sd3.5-large":
        body["mode"] = "text-to-image"
    try:
        resp = json.loads(_client().invoke_model(modelId=model, body=json.dumps(body))["body"].read())
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in RETRY_CODES:
            _buckets[name].drain()
            log.info("%s busy (%s); trying elsewhere", name, code)
            return None, True
        log.warning("%s failed: %s", name, e)
        return None, False
    except BotoCoreError as e:  # timeouts, dropped connections
        log.warning("%s unreachable: %s", name, e)
        return None, True
    reason = (resp.get("finish_reasons") or [None])[0]
    if reason or not resp.get("images"):
        log.warning("%s returned no image: %s", name, reason)  # e.g. its content filter: another model won't differ
        return None, False
    img = cv.imdecode(np.frombuffer(base64.b64decode(resp["images"][0]), np.uint8), cv.IMREAD_COLOR)
    return img, False


def generate(prompt: str, negative: str, aspect: str = "21:9", seed: int = 0,
             wait_s: float = 30.0) -> tuple[np.ndarray | None, str | None]:
    """A picture for `prompt` from the first model with quota to spare, waiting up to `wait_s` for quota.
    Returns (BGR image, model name), or (None, None) if none could be made (no quota in time, daily budget spent,
    the service refused)."""
    if not enabled():
        return None, None
    end = time.monotonic() + wait_s
    while True:
        waits = []
        for name, model, _ in MODELS:
            wait = _buckets[name].take()
            if wait:
                waits.append(wait)
                continue
            if not budget.take():
                log.warning("daily image limit (%d) reached", budget.DAILY_LIMIT)
                return None, None
            img, retry = _call(name, model, prompt, negative, aspect, seed)
            if img is not None:
                return img, name
            if not retry:
                return None, None
            waits.append(2.0)
        left = end - time.monotonic()
        if left <= 0:
            return None, None
        time.sleep(min(min(waits), left))
