"""Story agent: turns one page into a short illustrated film, streamed, so painting starts while it is still writing.

It looks at the page (the image OpenCV flattened) and streams JSON Lines: a header (title, painting style, the
place, the cast with fixed looks), then one object per moment: the page's own words for that moment, who speaks
each line, and what the viewer should see. Each object is handed over the moment it closes."""
import json
import logging
import math
import os
import re
from functools import lru_cache

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from .llm import LLMUnavailable, enabled

log = logging.getLogger(__name__)
MODEL = os.environ.get("PLANNER_MODEL", "us.amazon.nova-lite-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")
LABEL = "Amazon Nova Lite on Bedrock"
MAX_SHOTS = 6  # one world + six moments stay inside the image quota (9 per minute)
EFFECTS = {"stars", "snow", "rain", "fireflies", "leaves", "sparkles"}
VOICES = ("woman", "man", "child")
TIMES = ("day", "night", "dusk", "dawn")
WEATHERS = ("clear", "clouds", "rain", "snow", "storm", "fog")
WINDS = ("still", "breeze", "gusty")
PATHS = ("still", "sway", "fall", "walk-left", "walk-right", "rise", "jump")
# Words a model uses for a movement, folded into the seven paths the theatre and the critic understand.
# Stems, matched against the start of the model's word ("tumbles", "flying", "danced").
PATH_WORDS = {"fall": ("fall", "fell", "tumbl", "drop", "toppl", "slip", "crash", "plung", "sink", "sank"),
              "walk-right": ("walk", "run", "ran", "march", "swim", "swam", "sail", "go", "went", "stroll", "trot",
                             "gallop", "ride", "rode", "move", "travel", "rush", "hurr", "dash", "flee", "fled", "escap",
                             "chase", "race", "rac", "leav", "left", "hop-along", "skat", "drive", "drove", "wander"),
              "rise": ("ris", "rose", "fly", "fli", "flew", "float", "climb", "lift", "soar", "up", "ascend"),
              "jump": ("jump", "hop", "leap", "bounc", "spring", "sprang", "skip"),
              "sway": ("sway", "danc", "wav", "spin", "rock", "wobbl", "shak", "shook", "nod", "bob", "swing",
                       "swung", "twirl", "twinkl", "flutter"),
              "still": ("still", "sit", "sat", "stand", "stood", "sleep", "slept", "look", "rest", "none")}
STILL = {"subject": "", "motion": "", "path": "still", "loop": True}
DEFAULT_STYLE = "soft watercolour picture-book illustration, delicate ink lines, warm gentle light, rich detail"

SYSTEM = f"""You turn ONE page of a story (a book page, a poem, a poster or a screenshot, attached as an image) into a
short illustrated film that a viewer watches inside VR, line by line. Output JSON Lines only: one JSON object per
line, no prose, no code fences, no blank lines.

Line 1, the header:
{{"title": at most 6 words, "style": how every picture is painted (if the page has artwork, describe its medium,
palette and line work; otherwise choose a style that suits the text, e.g. "{DEFAULT_STYLE}"), "place": the main
place where it happens, as scenery seen all around, at most 40 words, no people or animals, "cast": [{{"name": ...,
"look": fixed appearance in at most 20 words (species, age, clothes, colours), "voice": "woman" | "man" | "child"}}],
"time": "day" | "night" | "dusk" | "dawn", "weather": "clear" | "clouds" | "rain" | "snow" | "storm" | "fog",
"wind": "still" | "breeze" | "gusty"}}

Then one line per moment, in reading order (each moment is its OWN JSON object on its own line, never a list
inside the header), AT MOST {MAX_SHOTS} moments (fewer for a short page), together covering
ALL of the page's story text exactly once and in order; on a long page put several lines (a whole verse) in each:
{{"shot": 1, "text": ["exact words of the page", ...], "speaker": ["Narrator", ...], "see": ..., "effects": [...],
"action": {{"subject": ..., "motion": ..., "path": ..., "loop": ...}}}}
- text: the page's own words copied exactly, split into short spoken lines. A poem or song keeps its verse lines.
  Prose is split at sentences and clauses (at most about 20 words each), NOT at the page's printed line breaks.
  Quoted speech is its own line, spoken by that character, and the words around it ("said old Mrs. Rabbit one
  morning,") are a separate Narrator line. Fix obvious misreadings by looking at the image, but never add, drop or
  reword anything.
- speaker: one entry per text line: "Narrator", or the character's name from the cast for the words inside quotes.
- see: what THESE lines describe, shown as ONE wide picture, at most 45 words: a close or medium shot of the ONE
  character (two at most) doing what the lines say, filling the frame, with the place behind them; who is there
  (use cast names with their looks), the light and mood. Never a wide crowd, a stage full of people or a tiny figure
  in a big landscape: the picture will be filmed, and the character must be big enough to act. Literal and concrete: "when the blazing sun is gone"
  is a sunset over the hills; "the traveller in the dark" is a traveller on a night road. Follow the words, not the
  page's own picture, and make every moment look different from the others (new scene, view or action). No text,
  letters or words in the picture.
- effects: zero or more of "stars", "snow", "rain", "fireflies", "leaves", "sparkles" for ambient motion that fits.
- action: the ONE movement these lines describe, which the viewer will see happen: "subject" is who or what moves
  (a cast name or a thing, at most 6 words), "motion" what physically happens, present tense, one continuous
  movement a 5-second shot can show (at most 25 words), "path" its direction: "fall" | "walk-left" | "walk-right" |
  "rise" | "jump" | "sway" | "still", "loop": true if it can repeat seamlessly (swaying), false if it ends in a new
  state (a fall). Use "still" when nothing in the lines moves; the camera never moves.
Leave out anything that isn't the story: page numbers, headers, logos, website or brand names, menus, links, credits.
Inside JSON strings write quotation marks as curly quotes “ and ”, never a bare " (it would break the JSON)."""


class ObjectStream:
    """Splits streamed text into top-level JSON objects as each one closes, whether the model writes one object per
    line or pretty-prints them; anything outside an object (prose, code fences) is ignored."""

    def __init__(self):
        self.buf, self.depth, self.in_str, self.esc = "", 0, False, False

    def feed(self, text: str) -> list[dict]:
        out = []
        for ch in text:
            if self.depth == 0:
                if ch == "{":
                    self.buf, self.depth = "{", 1
                continue
            self.buf += ch
            if self.in_str:
                if self.esc:
                    self.esc = False
                elif ch == "\\":
                    self.esc = True
                elif ch == '"':
                    self.in_str = False
            elif ch == '"':
                self.in_str = True
            elif ch == "{":
                self.depth += 1
            elif ch == "}":
                self.depth -= 1
                if self.depth == 0:
                    try:
                        obj = json.loads(self.buf)
                        if isinstance(obj, dict):
                            out.append(obj)
                    except json.JSONDecodeError:
                        log.warning("planner wrote a malformed object; skipped")
        return out


def clean_header(h: dict) -> dict:
    cast = []
    for c in h.get("cast") or []:
        if isinstance(c, dict) and str(c.get("name", "")).strip():
            cast.append({"name": str(c["name"]).strip()[:40], "look": str(c.get("look", "")).strip()[:200],
                         "voice": c.get("voice") if c.get("voice") in VOICES else "woman"})
    pick = lambda key, allowed, default: h.get(key) if h.get(key) in allowed else default
    return {"title": str(h.get("title") or "").strip()[:80] or None,
            "style": str(h.get("style") or "").strip()[:300] or DEFAULT_STYLE,
            "place": str(h.get("place") or "").strip()[:400] or "a gentle storybook landscape",
            "cast": cast[:6], "time": pick("time", TIMES, "day"), "weather": pick("weather", WEATHERS, "clear"),
            "wind": pick("wind", WINDS, "breeze")}


def path_from_words(text: str) -> str:
    """The first movement verb in a description ("waves her wand" -> sway; "sits quietly" -> still). Stems of
    three letters or more, matched at the start of a word at most two letters longer, or an -ing form
    ("runs", "running"; but "slipper" is not "slip")."""
    for word in re.findall(r"[a-z]+", text.lower()):
        for p, stems in PATH_WORDS.items():
            if any(len(st) >= 3 and word.startswith(st)
                   and (len(word) <= len(st) + 2 or (word.endswith("ing") and len(word) <= len(st) + 4))
                   for st in stems):
                return p
    return "still"


def clean_action(a) -> dict:
    """The moment's movement, folded into one of the seven paths; anything unusable becomes "still"."""
    if not isinstance(a, dict):
        return dict(STILL)
    raw = re.sub(r"[\s_]+", "-", str(a.get("path") or "").strip().lower())
    first = raw.split("-")[0]
    if raw in PATHS:
        path = raw
    elif "left" in raw or "right" in raw:
        path = "walk-left" if "left" in raw else "walk-right"
    else:
        path = next((p for p, words in PATH_WORDS.items() if any(w and first.startswith(w) for w in words)), "still")
    motion = str(a.get("motion") or "").strip()[:200]
    if path != "still" and not motion:
        path = "still"
    subject = str(a.get("subject") or "").strip()
    if subject.lower() in ("narrator", "the narrator") or re.match(r"(?i)\s*(describes|narrates|tells|explains)\b", motion):
        motion, path = "", "still"  # narration, not something that happens in the picture
    if path == "still" and motion:
        path = path_from_words(motion)  # the model said "still" while describing a movement ("rushes out")
    loop = a.get("loop")
    loop = bool(loop) if isinstance(loop, bool) else path in ("sway", "still")
    if path in ("fall", "rise"):
        loop = False  # it ends somewhere else: the last frame is held
    return {"subject": str(a.get("subject") or "").strip()[:40], "motion": motion, "path": path, "loop": loop}


def join_broken(text: list[str], speakers: list[str]) -> tuple[list[str], list[str]]:
    """Prose copied with the page's printed line breaks ("They lived with their Mother in a" / "sand-bank, ...")
    would be read aloud in fragments. A line that stops mid-clause (no closing punctuation) joins the next one when
    that starts in lower case; verse lines start with capitals or end in punctuation, so they stay apart."""
    out_t, out_s = [], []
    for t, s in zip(text, speakers):
        if out_t and out_s[-1] == s and out_t[-1][-1:].isalnum() and t[:1].islower():
            out_t[-1] += " " + t
        else:
            out_t.append(t)
            out_s.append(s)
    return out_t, out_s


def clean_shot(s: dict) -> dict | None:
    text = s.get("text")
    if isinstance(text, str):
        text = [text]
    text = [str(t).strip() for t in text or [] if str(t).strip()]
    if not text:
        return None
    speakers = s.get("speaker")
    if isinstance(speakers, str):
        speakers = [speakers]
    speakers = [str(p).strip() or "Narrator" for p in (speakers or [])][:len(text)]
    speakers += ["Narrator"] * (len(text) - len(speakers))
    text, speakers = join_broken(text, speakers)
    see = str(s.get("see") or "").strip() or " ".join(text)
    effects = [e for e in (s.get("effects") or []) if e in EFFECTS] if isinstance(s.get("effects"), list) else []
    return {"text": text, "speaker": speakers, "see": see[:500], "effects": effects,
            "action": clean_action(s.get("action"))}


SPEECH_TAG = re.compile(r"\b(said|says|shouted|shouts|asked|asks|cried|replied|whispered|called|answered|exclaimed|"
                        r"yelled|laughed|added|begged|sang|told|muttered|screamed)\b", re.I)


class Dialogue:
    """Splits narration from quoted speech and gives each quote to the character the narration around it names.
    Done in code because the model often shifts speakers by a line ("said old Mrs. Rabbit" read by Mrs. Rabbit).
    Keeps state across moments, since a quotation can run on from one moment into the next."""

    def __init__(self, cast: list[dict]):
        self.names = [c["name"] for c in cast]
        self.open = False  # inside a quotation
        self.last = None   # who spoke the last quotation

    def _named(self, text: str) -> str | None:
        hits = [(text.lower().find(n.lower()), n) for n in self.names if n.lower() in text.lower()]
        return min(hits)[1] if hits else None

    def apply(self, shot: dict) -> dict:
        segs = []  # [text, is_speech, the model's speaker]
        for text, model in zip(shot["text"], shot["speaker"]):
            buf = ""
            for ch in text:
                opens = ch == "“" or (ch == '"' and not self.open)
                closes = ch == "”" or (ch == '"' and self.open)
                if opens and not self.open:
                    if buf.strip():
                        segs.append([buf, False, model])
                    buf, self.open = ch, True
                elif closes and self.open:
                    segs.append([buf + ch, True, model])
                    buf, self.open = "", False
                else:
                    buf += ch
            if buf.strip():
                segs.append([buf, self.open, model])
        merged = []
        for seg in segs:  # a lone comma or full stop joins the line before it
            seg[0] = seg[0].strip()
            if merged and not any(c.isalnum() for c in seg[0]):
                merged[-1][0] += seg[0]
            elif any(c.isalnum() for c in seg[0]):
                merged.append(seg)
        speakers = []
        for i, (text, speech, model) in enumerate(merged):
            if not speech:
                # The model often drops the quote marks; a line it gives to a character is still that character's,
                # unless it reads like the narration around speech ("Timmy shouted.", "she said with a smile").
                if model in self.names and not SPEECH_TAG.search(text):
                    self.last = model
                    speakers.append(model)
                else:
                    speakers.append("Narrator")
                continue
            near = [merged[j][0] for j in (i + 1, i - 1) if 0 <= j < len(merged) and not merged[j][1]]
            who = next((n for n in map(self._named, near) if n), None)
            if not who:  # "she said": nobody named. Not the one being spoken to ("You're free now, Timmy!")
                addressed = {n for n in self.names if re.search(rf"\b{re.escape(n)}\b", text, re.I)}
                options = [model if model in self.names else None, self.last, *self.names]
                who = next((n for n in options if n and n not in addressed), None)
            who = who or (model if model in self.names else None) or self.last or (model if model != "Narrator" else None)
            self.last = who or self.last
            speakers.append(who or "Narrator")
        if not merged:
            return shot
        return {**shot, "text": [m[0] for m in merged], "speaker": speakers}


@lru_cache(maxsize=1)
def _runtime():
    return boto3.client("bedrock-runtime", region_name=REGION,
                        config=Config(read_timeout=60, connect_timeout=10,
                                      retries={"total_max_attempts": 2, "mode": "standard"}))


def plan(page_jpg: bytes, on_header, on_shot, line_count: int = 0) -> None:
    """Stream the plan for a page: on_header(header) once, then on_shot(shot) per moment as it arrives; a moment
    past MAX_SHOTS is handed over as more lines for the last one (on_shot(shot, extend=True)), so no words are lost.
    `line_count` is how many text lines OpenCV found, so the model can size the moments.
    Raises LLMUnavailable if the model can't be reached or produced no usable moment."""
    if not enabled():
        raise LLMUnavailable("LLM disabled by STEP_INSIDE_LLM")
    hint = "Read the page from the image."
    if line_count > 2 * MAX_SHOTS:  # a long page: say how big the moments must be to fit
        per = math.ceil(line_count / MAX_SHOTS)
        hint += (f" It is long: OpenCV found {line_count} printed lines, so each moment must hold about {per} or more"
                 f" of them (a whole verse or paragraph).")
    sent = {"shots": 0, "pending": None}  # before the call: the error handler below reads it
    try:
        resp = _runtime().converse_stream(
            modelId=MODEL, system=[{"text": SYSTEM}],
            messages=[{"role": "user", "content": [{"image": {"format": "jpeg", "source": {"bytes": page_jpg}}},
                                                   {"text": hint}]}],
            inferenceConfig={"maxTokens": 3000, "temperature": 0.2})
        stream, header, dialogue = ObjectStream(), None, None

        def emit(shot):
            if sent["shots"] < MAX_SHOTS:
                sent["shots"] += 1
                on_shot(shot)
            else:
                on_shot(shot, extend=True)

        def take(shot):
            # Each moment waits for the next one, so a sentence the model split across two moments is joined
            # back into one spoken line ("...ran" / "straight away to...") before anything is voiced.
            held = sent["pending"]
            if (held and held["speaker"][-1] == shot["speaker"][0] and held["text"][-1][-1:].isalnum()
                    and shot["text"][0][:1].islower()):
                held["text"][-1] += " " + shot["text"][0]
                shot = {**shot, "text": shot["text"][1:], "speaker": shot["speaker"][1:]}
            if not shot["text"]:
                return
            if held:
                emit(held)
            sent["pending"] = shot

        def moment(obj):
            nonlocal header, dialogue
            shot = clean_shot(obj)
            if not shot:
                return
            if header is None:  # a model that skipped the header still gets painted
                header = clean_header({})
                on_header(header)
            dialogue = dialogue or Dialogue(header["cast"])
            shot = dialogue.apply(shot)
            text, speakers = join_broken(shot["text"], shot["speaker"])  # now that speakers are right
            take({**shot, "text": text, "speaker": speakers})

        stop, raw = None, []
        for event in resp["stream"]:
            stop = event.get("messageStop", {}).get("stopReason") or stop
            delta = event.get("contentBlockDelta", {}).get("delta", {}).get("text")
            if delta:
                raw.append(delta)
            for obj in stream.feed(delta) if delta else ():
                if "shot" in obj or "text" in obj:
                    moment(obj)
                elif header is None:
                    header = clean_header(obj)
                    on_header(header)
                    # Nova Lite often nests every moment inside the header ("scenes", "moments") instead of one
                    # object per line: read them from there (they arrive together, at the end of the header).
                    for obj2 in nested_moments(obj):
                        moment(obj2)
        filtered = stop == "content_filtered" or FILTER_NOTE in "".join(raw[-3:])
        if filtered:
            # Bedrock's filter cut the reply (it happens on published books: the agent copies the page). Keep the
            # moments that arrived complete, even inside an unfinished header; OpenCV's check restores the rest.
            log.warning("story agent reply cut by the content filter after %d moments", sent["shots"])
            if header is None:
                h = partial_header("".join(raw))
                if h:
                    header = clean_header(h)
                    on_header(header)
            if not sent["shots"] and not sent["pending"]:
                for obj in rescued_moments("".join(raw)):
                    moment(obj)
    except (BotoCoreError, ClientError) as e:
        log.warning("planner call failed: %s", e)
        if sent["pending"]:
            emit(sent["pending"])
        raise LLMUnavailable(str(e)) from e
    if sent["pending"]:
        emit(sent["pending"])
    if not sent["shots"]:
        if filtered:
            raise ContentFiltered("the story agent's reply was blocked by the content filter",
                                  header_sent=header is not None)
        raise LLMUnavailable("the planner produced no moments")


FILTER_NOTE = "blocked by our content filters"


class ContentFiltered(LLMUnavailable):
    """The model's own filter blocked its reply (typically copying a published page)."""

    def __init__(self, msg: str, header_sent: bool = False):
        super().__init__(msg)
        self.header_sent = header_sent


def _decode_at(text: str, i: int):
    try:
        return json.JSONDecoder().raw_decode(text, i)[0]
    except ValueError:
        return None


def rescued_moments(text: str) -> list[dict]:
    """Complete moment objects anywhere in a reply that was cut off (inside an unfinished header or list)."""
    out = []
    for m in re.finditer(r'\{\s*"shot"\s*:', text):
        obj = _decode_at(text, m.start())
        if isinstance(obj, dict) and ("text" in obj or "lines" in obj):
            out.append(obj)
    return out


def partial_header(text: str) -> dict | None:
    """The header fields of a reply cut off before its header closed."""
    h = {}
    for key in ("title", "style", "place", "time", "weather", "wind"):
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            h[key] = m.group(1)
    m = re.search(r'"cast"\s*:\s*\[', text)
    if m:
        cast = _decode_at(text, m.end() - 1)
        if isinstance(cast, list):
            h["cast"] = cast
    return h or None


NESTED_KEYS = ("moments", "scenes", "shots", "lines", "story")


def nested_moments(obj: dict) -> list[dict]:
    """Moments a model put inside the header object instead of after it."""
    for key in NESTED_KEYS:
        items = obj.get(key)
        if isinstance(items, list) and items and all(isinstance(x, dict) for x in items):
            return [x for x in items if "text" in x or "shot" in x]
    return []


SYSTEM_LINES = f"""You turn ONE page of a story (attached as an image) into a short illustrated film watched line by line
in VR. OpenCV has already read the page's printed lines; they are listed, numbered, after the image. Do NOT copy
the page's text: refer to its lines by number. Output JSON Lines only, one object per line, no prose, no fences.

Line 1, the header:
{{"title": at most 6 words, "style": how every picture is painted, "place": the main place as scenery seen all
around (at most 40 words, no people), "cast": [{{"name", "look": fixed appearance in at most 20 words, "voice":
"woman" | "man" | "child"}}], "time": "day" | "night" | "dusk" | "dawn", "weather": "clear" | "clouds" | "rain" |
"snow" | "storm" | "fog", "wind": "still" | "breeze" | "gusty", "skip": [numbers of lines that are not the story:
titles, author and illustrator credits, copyright notes, page numbers, headers], "fixes": {{"misread words as
listed": "as printed in the image"}} for words OpenCV misread or punctuation it lost, at most 3 words each}}

Then one object per moment, in reading order, AT MOST {MAX_SHOTS} moments, together covering every story line exactly
once (whole paragraphs or verses per moment on a long page):
{{"shot": 1, "lines": [line numbers], "speaker": ["Narrator" or a cast name, one per line], "see": what these
lines describe as ONE picture, at most 45 words, a close or medium shot of the one character (two at most) doing it,
"effects": [], "action": {{"subject": a character in the picture (never "Narrator"), "motion": what that character
physically does in the picture, e.g. "skips a stone across the lake" (at most 25 words; never "describes ..."),
"path": "fall" | "walk-left" | "walk-right" | "rise" | "jump" | "sway" | "still", "loop": true | false}}}}
A moment ends at the end of a sentence: never split a sentence between two moments. List EVERY misread word in
"fixes" (compare each listed line with the image: "ived" -> "lived", "iinding" -> "finding", "a a" -> "a")."""


SENTENCE_END = (".", "!", "?", "\"", "\u201d", "'")


COMMON = set("""a an the and or but if so as at by for from in into of on onto to up with without over under
he she it they we you i me him her them us his hers its their our your my mine this that these those there here
was were is are be been am do did does done had has have not no yes all any some one two who whom whose which what
when where why how than then too very can could will would shall should may might must said says say
out off down just only also not now new old big little""".split())


def _apply_fixes(text: str, fixes: dict) -> str:
    """Apply the model's corrections of OpenCV's misreadings. A fix whose wrong side is a single common word is
    refused: applied everywhere it would change every "and" on the page (seen: "and" -> "though")."""
    for wrong, right in fixes.items():
        if not isinstance(wrong, str) or not isinstance(right, str):
            continue
        if " " not in wrong.strip() and wrong.strip().lower() in COMMON:
            continue
        if wrong and right and len(wrong) <= 30 and len(right) <= 30:
            text = re.sub(rf"(?<![\w']){re.escape(wrong)}(?![\w'])", right, text)
    return text


def _sentences(text: list[str], speakers: list[str]) -> tuple[list[str], list[str]]:
    """Prose joined from printed lines, split again at sentence ends so each spoken line is a sentence."""
    out_t, out_s = [], []
    for t, sp in zip(text, speakers):
        for part in re.split(r"(?:(?<=[.!?])|(?<=[.!?][\"\u201d']))\s+(?=[A-Z\"\u201c])", t):
            if part.strip():
                out_t.append(part.strip())
                out_s.append(sp)
    return out_t, out_s


def plan_from_lines(page_jpg: bytes, ocr_lines: list[str], on_header, on_shot, send_header: bool = True) -> None:
    """The story agent for pages it may not copy (published books trip the model's content filter): it sees the image
    and OpenCV's numbered lines, and answers with line numbers, speakers, what to show and how things move, plus
    which lines to skip (credits, titles) and single-word fixes for OpenCV's misreadings. The words come from OpenCV."""
    if not enabled():
        raise LLMUnavailable("LLM disabled by STEP_INSIDE_LLM")
    listing = "\n".join(f"{i}: {ln}" for i, ln in enumerate(ocr_lines))
    try:
        resp = _runtime().converse(
            modelId=MODEL, system=[{"text": SYSTEM_LINES}],
            messages=[{"role": "user", "content": [{"image": {"format": "jpeg", "source": {"bytes": page_jpg}}},
                                                   {"text": "OpenCV's lines:\n" + listing}]}],
            inferenceConfig={"maxTokens": 2500, "temperature": 0.2})
    except (BotoCoreError, ClientError) as e:
        raise LLMUnavailable(str(e)) from e
    if resp.get("stopReason") == "content_filtered":
        raise ContentFiltered("blocked again in lines mode")
    text = "".join(c.get("text", "") for c in resp["output"]["message"]["content"])
    objs = ObjectStream().feed(text)
    head = next((o for o in objs if "shot" not in o and "lines" not in o), {})
    moments = [o for o in objs if "shot" in o and isinstance(o.get("lines"), list)] or \
        [o for o in nested_moments_lines(head)]
    if not moments:
        raise LLMUnavailable("lines mode produced no moments")
    fixes = head.get("fixes") if isinstance(head.get("fixes"), dict) else {}
    skip = {i for i in head.get("skip", []) if isinstance(i, int)}
    header = clean_header(head)
    if send_header:
        on_header(header)
    dialogue = Dialogue(header["cast"])
    used, groups = set(), []
    for m in moments:
        idx = [i for i in m["lines"] if isinstance(i, int) and 0 <= i < len(ocr_lines) and i not in skip and i not in used]
        if idx:
            used.update(idx)
            groups.append([m, idx])
    for g, nxt in zip(groups, groups[1:]):  # a sentence cut between two moments goes back together
        while g[1] and nxt[1] and not _apply_fixes(ocr_lines[g[1][-1]], fixes).rstrip().endswith(SENTENCE_END) \
                and _apply_fixes(ocr_lines[nxt[1][0]], fixes)[:1].islower():
            g[1].append(nxt[1].pop(0))
    sent = 0
    for m, idx in groups:
        if not idx:
            continue
        speakers = m.get("speaker") if isinstance(m.get("speaker"), list) else []
        speakers = [str(x) for x in speakers][:len(idx)] + ["Narrator"] * (len(idx) - len(speakers[:len(idx)]))
        shot = clean_shot({**m, "text": [_apply_fixes(ocr_lines[i], fixes) for i in idx], "speaker": speakers})
        if not shot:
            continue
        if not shot["action"]["motion"]:  # no real action given: take it from what the picture shows, if someone acts
            who = next((c["name"] for c in header["cast"] if c["name"].lower() in shot["see"].lower()), "")
            if who:
                motion = " ".join(shot["see"].split()[:25])
                shot["action"] = {"subject": who, "motion": motion, "path": path_from_words(motion),
                                  "loop": path_from_words(motion) in ("still", "sway")}
        shot = dialogue.apply(shot)
        t, sp = join_broken(shot["text"], shot["speaker"])
        t, sp = _sentences(t, sp)
        shot = {**shot, "text": t, "speaker": sp}
        if sent < MAX_SHOTS:
            on_shot(shot)
            sent += 1
        else:
            on_shot(shot, extend=True)
    if not sent:
        raise LLMUnavailable("lines mode produced no usable moments")


def nested_moments_lines(obj: dict) -> list[dict]:
    for key in NESTED_KEYS:
        items = obj.get(key)
        if isinstance(items, list) and items and all(isinstance(x, dict) for x in items):
            return [x for x in items if isinstance(x.get("lines"), list)]
    return []


CREDIT = re.compile(r"©|\(c\)|\bcopyright\b|\bwritten by\b|\billustrat|\bbased on\b|\bpublished\b|\bauthor\b|"
                    r"\btranslated\b|\ball rights\b|\bisbn\b|^\s*(page\s*)?\d+\s*$", re.I)


def _is_title(line: str) -> bool:
    """A heading, not a story line: short, no closing punctuation, its main words capitalized ("Hansel and Gretel")."""
    words = line.split()
    if not 1 <= len(words) <= 6 or line.rstrip()[-1:] in ".,;:!?\"'\u201d":
        return False
    main = [w for w in words if len(w) > 3]
    return bool(main) and sum(w[0].isupper() for w in main) / len(main) >= 0.75


def _looks_like_text(line: str) -> bool:
    """Drop OCR lines that are decoration or junk: too short, or mostly not letters."""
    letters = sum(ch.isalpha() for ch in line)
    return letters >= 3 and letters / max(1, len(line.replace(" ", ""))) > 0.6


def fallback_plan(ocr_lines: list[str], max_shots: int = MAX_SHOTS) -> tuple[dict, list[dict]]:
    """No model: the page's lines as OpenCV read them, grouped into evenly sized moments."""
    raw = [re.sub(r"\s+", " ", ln).strip() for ln in ocr_lines if _looks_like_text(ln)]
    title_line = raw[0] if raw and _is_title(raw[0]) else None
    lines = [ln for ln in raw if not CREDIT.search(ln)]  # credits and copyright are not the story
    if title_line and lines and lines[0] == title_line and len(lines) > 1:
        lines = lines[1:]  # a short heading above the story: the title, not a line to read
    if not lines:
        lines = ["This page has no words I could read."]
    per = max(1, math.ceil(len(lines) / max_shots))
    shots = [{"text": lines[i:i + per], "speaker": ["Narrator"] * len(lines[i:i + per]),
              "see": " ".join(lines[i:i + per]), "effects": [], "action": dict(STILL)}
             for i in range(0, len(lines), per)]
    title = " ".join((title_line or lines[0]).split()[:6]).strip(",.;:!?\"'")
    return {**clean_header({}), "title": title}, shots
