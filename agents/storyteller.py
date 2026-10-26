"""Storyteller: turns a page's text (read by OpenCV) into spoken beats with character voices, plus a title and setting.
Faithfulness is enforced in code, not just asked for: beats that aren't on the page are dropped."""
import re
from difflib import SequenceMatcher

from . import llm

SYSTEM = """You turn one page of a picture book into a short read-aloud script for a reader standing inside the
page's illustration. The input has the page text (read by OCR: fix obvious slips such as a missing first letter or
missing quote marks) and what computer vision saw in the illustration.
Rules:
- Use only the page's own words. Never add sentences, descriptions or plot that aren't on the page.
- Split the text into 1 to 5 beats, in order, covering the whole page.
- Quoted speech is its own beat, spoken by that character (speaker = the character's name, e.g. "Mrs. Rabbit").
  Put the speech-tag words ("said old Mrs. Rabbit one morning") in a Narrator beat.
- Everything else is spoken by "Narrator".
- characters: everyone except the Narrator who speaks, with a voice of "woman", "man" or "child".
- title: at most 6 words. setting: one short sentence naming the place, taken from the page or the illustration.
- world_prompt: the reader will stand inside this illustration with a full 360 degree view at eye level. Look at the
  attached illustration and describe, in at most 70 words, the place the reader is standing in, as seen at eye level:
  what continues to the left, right and behind, the sky (or ceiling) above, the ground below, distant scenery. Describe
  where the illustration's scene takes place (if it shows a lane or a wood, describe that, not somewhere only mentioned
  in the text), consistent with this story, in the SAME art style,
  medium, palette and line work as the illustration (name them: e.g. "delicate pen-and-ink watercolour, soft greens and
  browns, white paper glow"). Scenery only: no people, animals or characters, no text.
- negative_prompt: comma-separated things to keep out of the painted world: always include text, letters, words,
  signature, watermark, frame, border, blank white paper, plus characters or animals, and anything off-style."""

# Every object closes additionalProperties: Claude's structured outputs reject a schema that leaves it open.
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "setting": {"type": "string"},
        "beats": {"type": "array", "items": {
            "type": "object",
            "properties": {"speaker": {"type": "string"}, "text": {"type": "string"}},
            "required": ["speaker", "text"], "additionalProperties": False}},
        "characters": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "voice": {"type": "string", "enum": ["woman", "man", "child"]}},
            "required": ["name", "voice"], "additionalProperties": False}},
        "world_prompt": {"type": "string"},
        "negative_prompt": {"type": "string"},
    },
    "required": ["title", "setting", "beats", "characters", "world_prompt", "negative_prompt"],
    "additionalProperties": False,
}
VOICES = ("woman", "man", "child")
MIN_OVERLAP = 0.7
DEFAULT_WORLD = ("the scenery continuing all around the illustration, gentle landscape with trees and open ground, "
                 "soft sky, in the same delicate watercolour picture-book style, soft natural palette")
DEFAULT_NEGATIVE = ("text, letters, words, signature, watermark, frame, border, blank white paper, "
                    "(people:1.5), (person:1.5), (man:1.4), (woman:1.4), (child:1.5), (boy:1.5), (girl:1.5), "
                    "hiker, figure, silhouette, animals, characters, "
                    "(photograph:1.5), (photorealistic:1.5), (3d render:1.4), realistic")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _sentences(text: str, max_beats: int = 4) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    if len(parts) <= max_beats:
        return parts
    size = -(-len(parts) // max_beats)
    return [" ".join(parts[i:i + size]) for i in range(0, len(parts), size)]


def template(text: str) -> dict:
    words = text.split()
    return {
        "title": " ".join(words[:5]).strip(",.;:\"'") or "A page of the story",
        "setting": "The scene in the page's illustration.",
        "beats": [{"speaker": "Narrator", "text": s} for s in _sentences(text)] or [{"speaker": "Narrator", "text": text}],
        "characters": [],
        "world_prompt": DEFAULT_WORLD,
        "negative_prompt": DEFAULT_NEGATIVE,
    }


def _on_page(word: str, page: set[str]) -> bool:
    """Exact, or one OCR slip away (a fixed letter like "ow" -> "now" still counts as the page's word)."""
    return word in page or any(SequenceMatcher(None, word, p).ratio() >= 0.8 for p in page if abs(len(p) - len(word)) <= 1)


def faithful(beats: list[dict], page_text: str) -> list[dict]:
    """Keep only beats whose words are (almost all) on the page; models sometimes embellish."""
    page = set(_words(page_text))
    kept = []
    for b in beats:
        if not isinstance(b, dict):
            continue
        words = _words(b.get("text") or "")
        if words and sum(_on_page(w, page) for w in words) / len(words) >= MIN_OVERLAP:
            kept.append({"speaker": (b.get("speaker") or "").strip() or "Narrator", "text": b["text"].strip()})
    return kept


def adapt(text: str, detected: list[str], story_so_far: list[str], illustration_jpg: bytes | None = None) -> tuple[dict, str]:
    if not text.strip():
        text = "This page has a picture but no words. Look around."
    payload = {"page_text": text, "illustration_shows": detected, "story_so_far": story_so_far[-3:]}
    try:
        out = llm.structured(SYSTEM, payload, SCHEMA, max_tokens=2000,
                             images=[illustration_jpg] if illustration_jpg else None)
    except llm.LLMUnavailable:
        return template(text), "rules"
    # Always keep the defaults: models sometimes drop "people" or "photograph", and the world must stay scenery-only.
    out["negative_prompt"] = DEFAULT_NEGATIVE + ", " + out["negative_prompt"]
    beats = faithful(out["beats"] or [], text)[:5]
    if not beats:
        return template(text), "rules"
    speakers = {b["speaker"].lower() for b in beats}
    # Model output is checked field by field: a character without a usable name or voice is dropped, not fatal.
    out["characters"] = [{"name": c["name"], "voice": c["voice"]} for c in out["characters"] or []
                         if isinstance(c, dict) and str(c.get("name", "")).lower() in speakers
                         and c.get("voice") in VOICES]
    out["beats"] = beats
    return out, "bedrock"
