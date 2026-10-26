"""Fix-it loop: Safety gate -> Planner -> (Coach <-> Verifier) until OpenCV sees the scene restored to baseline."""
import uuid
from dataclasses import dataclass, field

import numpy as np

from vision.align import align, features, warp
from vision.change import change_score
from vision.locate import DEFAULT_HFOV, bearing_deg, direction_phrase, height_phrase
from vision.pipeline import prepare

from . import llm
from .scout import ADVICE

RESTORED = 0.12
MIN_VISIBLE = 0.6
TURN_DEG = 12
STUCK_FRAMES = 120

PLANNER_SYSTEM = """You coach a blind person through restoring one thing in their home. Write one short, calm \
spoken instruction per state (max 12 words each). Do not mention directions or distances; the app adds those \
from measured geometry. States: "pickup" (pick the item up / take it away), "place" (put it back where it \
belongs), "clean" (wipe the area clean), "done" (praise, confirm it's restored)."""

PLANNER_SCHEMA = {
    "type": "object",
    "properties": {s: {"type": "string"} for s in ("pickup", "place", "clean", "done")},
    "required": ["pickup", "place", "clean", "done"],
    "additionalProperties": False,
}


def _template(label: str) -> dict:
    return {
        "pickup": f"Reach for {label} and pick it up.",
        "place": f"Now put {label} back where it belongs.",
        "clean": "Wipe this area until it's clean.",
        "done": "Perfect. That's back to how it should be.",
    }


@dataclass
class FixSession:
    home: str
    keyframe: str
    change: dict
    verdict: dict
    lines: dict
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: str = ""
    frames: int = 0
    frames_in_state: int = 0
    last_say: str = ""
    trace: list = field(default_factory=list)

    def __post_init__(self):
        self.state = "clean" if self.verdict["fix"] == "spill" else "pickup"


def safety_gate(verdict: dict) -> dict:
    if not verdict["safe_for_user"]:
        return {"allowed": False, "say": "This one isn't safe to handle alone. I'll let your helper know."}
    if verdict["fix"] not in ("rearrange", "spill"):
        return {"allowed": False, "say": "I can't guide a fix for this one yet."}
    return {"allowed": True}


def plan(change: dict, verdict: dict) -> tuple[dict, str]:
    label = verdict["label"]
    try:
        return llm.structured(PLANNER_SYSTEM, {"item": label, "kind": change["kind"], "fix": verdict["fix"]},
                              PLANNER_SCHEMA, max_tokens=1500), "bedrock"
    except llm.LLMUnavailable:
        return _template(label), "rules"


def _region_px(norm_box: list, shape: tuple) -> tuple:
    h, w = shape[:2]
    pts = np.array(norm_box) * [w, h]
    x0, y0 = np.clip(pts.min(axis=0), 0, [w, h]).astype(int)
    x1, y1 = np.clip(pts.max(axis=0), 0, [w, h]).astype(int)
    return x0, y0, x1, y1


def _region_state(score: np.ndarray, valid: np.ndarray, norm_box: list) -> dict:
    x0, y0, x1, y1 = _region_px(norm_box, score.shape)
    v = valid[y0:y1, x0:x1]
    visible = float(v.mean()) if v.size else 0.0
    mean = float(score[y0:y1, x0:x1][v].mean()) if v.any() else 1.0
    return {"visible": round(visible, 2), "score": round(mean, 3),
            "restored": visible >= MIN_VISIBLE and mean < RESTORED}


def verify(base: np.ndarray, cur: np.ndarray, H: np.ndarray, change: dict) -> dict:
    """Verifier: do the changed regions match the baseline again?"""
    warped, valid = warp(cur, H, base.shape)
    score = change_score(base, warped)
    out = {"here": _region_state(score, valid, change["box_baseline"])}
    if "from_box_baseline" in change:
        out["home"] = _region_state(score, valid, change["from_box_baseline"])
    return out


def _guide_to(norm_box: list, base_shape: tuple, cur_shape: tuple, H: np.ndarray, hfov: float) -> tuple[float, str]:
    h, w = base_shape[:2]
    center = (np.array(norm_box) * [w, h]).mean(axis=0)
    p = np.linalg.inv(H) @ np.array([center[0], center[1], 1.0])
    cx, cy = p[0] / p[2], p[1] / p[2]
    bearing = bearing_deg(cx, cur_shape[1], hfov)
    return bearing, f"{direction_phrase(bearing)}, {height_phrase(cy, cur_shape[0])}"


def step(session: FixSession, keyframe, frame: np.ndarray, hfov: float = DEFAULT_HFOV) -> dict:
    """Coach: one frame of guidance. OpenCV verification decides the next state, not a timer."""
    session.frames += 1
    session.frames_in_state += 1
    c = prepare(frame)
    al = align(keyframe.image, c, base_feats=keyframe.feats, cur_feats=features(c))
    if not al.ok:
        return _emit(session, ADVICE["lost"], {"aligned": False})

    check = verify(keyframe.image, c, al.H, session.change)
    here, home = check["here"], check.get("home")
    prev = session.state

    if session.state in ("pickup", "clean") and here["restored"]:
        session.state = "place" if home is not None and not home["restored"] else "done"
    if session.state == "place" and home is not None and home["restored"]:
        session.state = "done"
    if session.state != prev:
        session.frames_in_state = 0
        session.trace.append({"agent": "verifier", "frame": session.frames, "from": prev, "to": session.state,
                              "evidence": check})

    if session.state == "done":
        return _emit(session, session.lines["done"], check, done=True)

    target = session.change["box_baseline"] if session.state in ("pickup", "clean") else session.change["from_box_baseline"]
    bearing, where = _guide_to(target, keyframe.image.shape, c.shape, al.H, hfov)
    if abs(bearing) > TURN_DEG:
        say = f"Turn {'right' if bearing > 0 else 'left'} a little."
    else:
        say = f"{session.lines[session.state]} It's {where}."
    if session.frames_in_state > STUCK_FRAMES:
        say = "This is taking a while. Say 'help' and I'll ask your helper."
    return _emit(session, say, check)


def _emit(session: FixSession, say: str, evidence: dict, done: bool = False) -> dict:
    new = say != session.last_say
    session.last_say = say
    return {"session": session.id, "state": session.state, "say": say, "speak": new, "done": done,
            "evidence": evidence, "frames": session.frames}
