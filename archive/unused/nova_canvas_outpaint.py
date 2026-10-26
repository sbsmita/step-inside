"""Amazon Nova Canvas outpainting: paints the story world around the real illustration (AI-filled)."""
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

log = logging.getLogger(__name__)
MODEL = os.environ.get("CANVAS_MODEL", "amazon.nova-canvas-v1:0")
# Outpainting masks are the opposite of inpainting masks: black marks the region to keep.
KEEP_IS_BLACK = True
MAX_KEEP_DRIFT = 28.0  # mean pixel change allowed inside the kept region before we suspect inverted mask polarity


@lru_cache(maxsize=1)
def _client():
    region = os.environ.get("AWS_REGION", "us-east-1")
    return boto3.client("bedrock-runtime", region_name=region, config=Config(read_timeout=300, retries={"max_attempts": 2}))


def _b64png(img: np.ndarray) -> str:
    return base64.b64encode(cv.imencode(".png", img)[1].tobytes()).decode()


def _call(image: np.ndarray, keep: np.ndarray, prompt: str, negative: str, keep_is_black: bool, seed: int) -> np.ndarray:
    mask = np.where(keep > 0, 0, 255) if keep_is_black else np.where(keep > 0, 255, 0)
    mask = cv.cvtColor(mask.astype(np.uint8), cv.COLOR_GRAY2BGR)
    body = {
        "taskType": "OUTPAINTING",
        "outPaintingParams": {
            "image": _b64png(image), "maskImage": _b64png(mask), "outPaintingMode": "DEFAULT",
            "text": prompt[:1024], "negativeText": negative[:1024],
        },
        "imageGenerationConfig": {"numberOfImages": 1, "quality": "standard", "cfgScale": 6.5, "seed": seed},
    }
    resp = json.loads(_client().invoke_model(modelId=MODEL, body=json.dumps(body), accept="application/json",
                                             contentType="application/json")["body"].read())
    if not resp.get("images"):
        raise RuntimeError(resp.get("error") or "no image returned")
    raw = base64.b64decode(resp["images"][0])
    out = cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_COLOR)
    return cv.resize(out, (image.shape[1], image.shape[0])) if out.shape[:2] != image.shape[:2] else out


def outpaint(image: np.ndarray, keep: np.ndarray, prompt: str, negative: str, seed: int = 7) -> np.ndarray | None:
    """Returns the painted panorama, or None if Nova Canvas is unavailable (caller keeps the OpenCV surround)."""
    if os.environ.get("STEP_INSIDE_LLM") == "off":
        return None
    try:
        out = _call(image, keep, prompt, negative, KEEP_IS_BLACK, seed)
        sel = keep > 0
        drift = float(np.abs(out[sel].astype(np.float32) - image[sel].astype(np.float32)).mean())
        if drift > MAX_KEEP_DRIFT:  # the illustration got repainted: the mask was read the other way round
            log.warning("nova canvas changed the kept region (drift %.1f); retrying with inverted mask", drift)
            out = _call(image, keep, prompt, negative, not KEEP_IS_BLACK, seed)
        return out
    except (BotoCoreError, ClientError, RuntimeError, ValueError) as e:
        log.warning("nova canvas unavailable: %s", e)
        return None
