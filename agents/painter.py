"""Stability AI Inpaint on Amazon Bedrock: paints the story world around the real illustration (AI-filled, labelled).

Inpaint (not Outpaint) because it takes a negative prompt and an explicit keep/generate mask. The panorama is painted
in overlapping tiles (vision/world.py); each call here paints one tile."""
import base64
import json
import logging
import os
from functools import lru_cache

import boto3
import cv2 as cv
import numpy as np
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from . import budget

log = logging.getLogger(__name__)
MODEL = os.environ.get("STABILITY_INPAINT_MODEL", "us.stability.stable-image-inpaint-v1:0")


def enabled() -> bool:
    return os.environ.get("STEP_INSIDE_LLM", "bedrock") == "bedrock" and os.environ.get("STEP_INSIDE_WORLD_AI", "stability") == "stability"


@lru_cache(maxsize=1)
def _client():
    # One retry here (throttling, timeouts); the tile painter retries a failed tile once more with a new seed.
    return boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"),
                        config=Config(read_timeout=120, retries={"total_max_attempts": 2, "mode": "standard"}))


def _b64(img: np.ndarray, ext: str, params=()) -> str:
    return base64.b64encode(cv.imencode(ext, img, list(params))[1].tobytes()).decode()


def paint(canvas: np.ndarray, generate: np.ndarray, prompt: str, negative: str, seed: int = 7) -> np.ndarray | None:
    """`generate` is 255 where new content is painted. Returns the painted canvas, or None if unavailable."""
    if not enabled():
        return None
    if not budget.take():
        log.warning("daily Stability call limit (%d) reached; the world will be built by OpenCV", budget.DAILY_LIMIT)
        return None
    body = {
        "image": _b64(canvas, ".jpg", (cv.IMWRITE_JPEG_QUALITY, 92)),
        "mask": _b64(generate, ".png"),  # white = full inpainting strength, black = untouched
        "prompt": prompt[:9000],
        "negative_prompt": negative[:9000],
        "grow_mask": 4,
        "seed": seed,
        "output_format": "jpeg",
    }
    try:
        resp = json.loads(_client().invoke_model(modelId=MODEL, body=json.dumps(body))["body"].read())
    except (BotoCoreError, ClientError) as e:
        log.warning("stability inpaint unavailable: %s", e)
        return None
    reason = (resp.get("finish_reasons") or [None])[0]
    if reason or not resp.get("images"):
        log.warning("stability inpaint returned no image: %s", reason)
        return None
    out = cv.imdecode(np.frombuffer(base64.b64decode(resp["images"][0]), np.uint8), cv.IMREAD_COLOR)
    if out.shape[:2] != canvas.shape[:2]:
        out = cv.resize(out, (canvas.shape[1], canvas.shape[0]), interpolation=cv.INTER_CUBIC)
    return out
