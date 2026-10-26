"""Ceilings on paid calls per day (UTC), shared by every agent, so a public demo can't run up an unbounded bill.
Pictures past the image limit come from OpenCV instead (and are labelled that way); past the video limit, moments
keep their OpenCV-animated characters instead of a generated clip."""
import os
import threading
import time

LIMITS = {"image": int(os.environ.get("STABILITY_DAILY_LIMIT", "400")),
          "video": int(os.environ.get("VIDEO_DAILY_LIMIT", "24"))}
DAILY_LIMIT = LIMITS["image"]  # kept for the agents' log lines
_state = {"day": "", "used": {k: 0 for k in LIMITS}}
_lock = threading.Lock()


def _today() -> None:
    day = time.strftime("%Y-%m-%d", time.gmtime())
    if _state["day"] != day:
        _state.update(day=day, used={k: 0 for k in LIMITS})


def take(kind: str = "image") -> bool:
    """Count one paid call of this kind; False once today's limit is reached."""
    with _lock:
        _today()
        if _state["used"][kind] >= LIMITS[kind]:
            return False
        _state["used"][kind] += 1
        return True


def used(kind: str = "image") -> int:
    with _lock:
        _today()
        return _state["used"][kind]


def left(kind: str = "image") -> int:
    return max(0, LIMITS[kind] - used(kind))
