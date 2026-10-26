"""Scene Director: turns OpenCV's measurements of a scene into a short spoken introduction.

The model only sees measured facts (objects, distances, scale, indoor/outdoor); it never sees pixels and never
invents geometry, so the narration can't contradict the 3D scene."""
from . import llm

SYSTEM = """You write the short spoken welcome for a 3D scene someone is about to step into in VR or on screen. \
The scene was rebuilt from a photo or a short video; the input JSON lists what computer vision measured. \
Write in second person, present tense, warm and concrete, at most 55 words. Mention only objects listed in the \
input, with their measured distances rounded naturally ("about two metres ahead"). Never invent objects, people's \
identities, places or events. If the input has a user title, you may use it. Also suggest a short title (max 6 words)."""

SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "narration": {"type": "string"}},
    "required": ["title", "narration"],
    "additionalProperties": False,
}


def _facts(meta: dict) -> dict:
    if meta["type"] == "360":
        return {"type": "360 panorama (surrounds the viewer)", "user_title": meta.get("title"),
                "detected": sorted({d["label"] for d in meta["detections"]})}
    if meta["type"] == "3d":
        return {"type": "3d reconstruction", "user_title": meta.get("title"),
                "views": meta["sfm"]["registered"], "points": meta["dense"].get("points"),
                "floor_found": meta["orient"]["floor_found"],
                "room_extent_m": [round(b - a, 1) for a, b in zip(meta["bounds"]["min"], meta["bounds"]["max"])]}
    return {"type": "single photo", "user_title": meta.get("title"), "outdoor": meta["outdoor"],
            "objects_in_front": [{"label": l["label"], "distance_m": l["distance_m"]} for l in meta["layers"]],
            "detected": sorted({d["label"] for d in meta["detections"]}),
            "nearest_m": meta["scale"]["near_m"], "farthest_m": meta["scale"]["far_m"]}


def _template(meta: dict) -> dict:
    if meta["type"] == "360":
        seen = sorted({d["label"] for d in meta["detections"]})
        things = f" Around you: {', '.join(seen[:4])}." if seen else ""
        return {"title": meta.get("title") or "All around you",
                "narration": f"This scene surrounds you completely.{things} Turn your head, or drag, "
                             f"to look in every direction."}
    if meta["type"] == "3d":
        ext = [round(b - a, 1) for a, b in zip(meta["bounds"]["min"], meta["bounds"]["max"])]
        return {"title": meta.get("title") or "Rebuilt space",
                "narration": f"You're standing where the camera started. This space was rebuilt from "
                             f"{meta['sfm']['registered']} views and is about {ext[0]} by {ext[2]} metres. "
                             f"Look around, then move with the thumbstick or teleport to explore."}
    layers = meta["layers"]
    if layers:
        near = layers[0]
        intro = f"The {near['label']} is about {near['distance_m']:.1f} metres in front of you."
    else:
        intro = "You're standing right where the photo was taken."
    # Sky detection can say "outdoors" with confidence; its absence doesn't prove "indoors".
    where = "You're outdoors, and the" if meta["outdoor"] else "The"
    return {"title": meta.get("title") or "Step inside",
            "narration": f"{intro} {where} far side is about {meta['scale']['far_m']:.0f} metres away. "
                         f"Lean or step a little to see around things."}


def direct(meta: dict) -> tuple[dict, str]:
    try:
        return llm.structured(SYSTEM, _facts(meta), SCHEMA, max_tokens=1200), "bedrock"
    except llm.LLMUnavailable:
        return _template(meta), "rules"
