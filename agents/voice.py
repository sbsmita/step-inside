"""Amazon Polly narration and character voices."""
import json
import logging
import os
from functools import lru_cache

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

log = logging.getLogger(__name__)
VOICE = os.environ.get("POLLY_VOICE", "Joanna")
# Neural voices: a British storyteller narrator, and distinct voices for the characters.
CAST = {"narrator": "Amy", "woman": "Emma", "man": "Brian", "child": "Ivy"}


def _enabled() -> bool:
    return os.environ.get("STEP_INSIDE_LLM") != "off" and os.environ.get("STEP_INSIDE_TTS") != "off"


@lru_cache(maxsize=1)
def _polly():
    # Adaptive retries pace the client when Polly says "rate exceeded" (a page's lines are voiced all at once).
    return boto3.client("polly", region_name=os.environ.get("AWS_REGION", "us-east-1"),
                        config=Config(max_pool_connections=16, retries={"total_max_attempts": 8, "mode": "adaptive"}))


def synthesize(text: str, voice: str | None = None) -> bytes | None:
    """MP3 bytes, or None if Polly isn't reachable (the browser then reads the text itself)."""
    if not _enabled():
        return None
    try:
        resp = _polly().synthesize_speech(Text=text, OutputFormat="mp3", VoiceId=voice or VOICE, Engine="neural")
        return resp["AudioStream"].read()
    except (BotoCoreError, ClientError) as e:
        log.warning("polly unavailable: %s", e)
        return None


def speak_line(text: str, voice: str) -> tuple[bytes | None, list[dict] | None]:
    """One line's MP3 plus its word timings, so captions can light up each word as it is spoken:
    [{"t": ms from the start, "s": first char, "e": end char, "w": word}]. (None, None) if Polly isn't reachable."""
    if not _enabled():
        return None, None
    try:
        audio = _polly().synthesize_speech(Text=text, OutputFormat="mp3", VoiceId=voice, Engine="neural")
        raw = _polly().synthesize_speech(Text=text, OutputFormat="json", SpeechMarkTypes=["word"], VoiceId=voice,
                                         Engine="neural")
        audio, raw = audio["AudioStream"].read(), raw["AudioStream"].read()
    except (BotoCoreError, ClientError) as e:
        log.warning("polly unavailable: %s", e)
        return None, None
    data, marks = text.encode("utf-8"), []
    for row in raw.decode("utf-8").splitlines():
        if row.strip():
            m = json.loads(row)  # offsets are bytes into the UTF-8 text; captions need characters
            marks.append({"t": int(m["time"]), "s": len(data[:m["start"]].decode("utf-8", "ignore")),
                          "e": len(data[:m["end"]].decode("utf-8", "ignore")), "w": m["value"]})
    return audio, marks


def voice_for(speaker: str, characters: list[dict]) -> str:
    if speaker.lower() == "narrator":
        return CAST["narrator"]
    for c in characters:
        if c["name"].lower() == speaker.lower():
            return CAST.get(c["voice"], CAST["narrator"])
    return CAST["woman"]
