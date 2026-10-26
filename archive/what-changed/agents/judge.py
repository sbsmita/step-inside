"""Judge: decides which OpenCV-detected changes matter to a blind user, and whether they're safe to fix alone."""
from . import llm

SYSTEM = """You help a blind person living independently. A computer-vision system compared their home \
against a baseline scan and found the changes listed in the input JSON. Each change has geometry measured by \
OpenCV (kind, size, height in view, direction) and sometimes an object label from a detector, plus household notes.

For each change decide:
- verdict: "hazard" (trip/burn/safety risk), "useful" (worth telling them), or "ignore" (noise, e.g. tiny \
or likely a lighting/shadow artifact).
- label: a bare noun phrase for the thing, e.g. "the mug", "a shoe", "something". The app builds the full \
sentence around it, so do not add verbs, locations or directions. Only name an object if the input label \
supports it; otherwise use "something". Changes marked transient are people or pets passing by: ignore them.
- fix: "rearrange" (move it back / put it away), "spill" (clean up), "water_plant", or "none".
- safe_for_user: false if fixing could hurt a blind person (broken glass, hot surfaces, liquid near outlets).
- escalate: true only when a sighted helper should be told now.
Prefer fewer, more important reports. Keep every reason under 15 words."""

SCHEMA = {
    "type": "object",
    "properties": {
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["hazard", "useful", "ignore"]},
                    "label": {"type": "string"},
                    "reason": {"type": "string"},
                    "fix": {"type": "string", "enum": ["rearrange", "spill", "water_plant", "none"]},
                    "safe_for_user": {"type": "boolean"},
                    "escalate": {"type": "boolean"},
                },
                "required": ["id", "verdict", "label", "reason", "fix", "safe_for_user", "escalate"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["changes"],
    "additionalProperties": False,
}

_FIELDS = ("kind", "area_frac", "height", "direction", "score", "label", "transient", "match",
           "from_direction", "from_height")


def _rules(change: dict) -> dict:
    kind, low, name = change["kind"], change["height"].startswith("low"), change.get("label")
    the, a = (f"the {name}", f"a {name}") if name else ("something", "something")
    if change.get("transient"):
        v = ("ignore", "someone" if name == "person" else "a pet", "Passing by, not part of the room.", "none")
    elif kind == "moved":
        v = ("useful", the, "It is not where it usually is.", "rearrange")
    elif kind == "appeared" and low:
        v = ("hazard", a, "Possible trip hazard.", "rearrange")
    elif kind == "appeared":
        v = ("useful", a, "It wasn't here before.", "rearrange")
    elif kind == "disappeared":
        v = ("useful", the, "Usually here, now gone.", "none")
    elif change["area_frac"] >= 0.01:
        v = ("useful", the, "This area changed.", "none")
    else:
        v = ("ignore", the, "Too small to matter.", "none")
    verdict, label, reason, fix = v
    return {"id": change["id"], "verdict": verdict, "label": label, "reason": reason, "fix": fix,
            "safe_for_user": True, "escalate": False}


def judge(scan: dict, household_notes: list[str] | None = None) -> tuple[list[dict], str]:
    """Returns (verdicts, source) where source is "bedrock" or "rules"."""
    changes = scan["changes"]
    if not changes:
        return [], "none"
    payload = {
        "room": scan.get("keyframe"),
        "household_notes": household_notes or [],
        "changes": [{"id": c["id"], **{k: c[k] for k in _FIELDS if k in c}} for c in changes],
    }
    try:
        out = llm.structured(SYSTEM, payload, SCHEMA)["changes"]
        known = {c["id"] for c in changes}
        verdicts = [v for v in out if v["id"] in known]
        missing = known - {v["id"] for v in verdicts}
        verdicts += [_rules(c) for c in changes if c["id"] in missing]
        return verdicts, "bedrock"
    except llm.LLMUnavailable:
        return [_rules(c) for c in changes], "rules"
