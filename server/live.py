"""Live stories: one page in, a story to witness line by line within seconds, built by agents working in parallel.

    reader (OpenCV): flatten the page, find its text lines ─┬─> story agent (Nova Lite, streamed) ─┬─ header ─> world painter
                                                            │                                      ├─ each moment ─> painter ─> critic + depth (OpenCV)
                                                            │                                      └─ each line ─> voice (Polly, word timings)
                                                            └─> OCR (OpenCV, parallel) ─> checks the story agent: lines it skipped are put
                                                                back, and every spoken line gets its box on the page for highlighting

Everything is published to the in-memory state the moment it exists and the browser polls it, so playback starts
when the world, the first moment and its lines are ready, while later moments are still being painted.

Then each moment keeps getting more alive while the story plays (the "upgrades"):
    each painted moment ─┬─> actor agent (OpenCV): the character that moves, cut out, judged, with a clean plate
                         ├─> world composer (OpenCV): the moment set into its own 360 world + wind, light, lamps, haze
                         ├─> world agent (Stability AI, one at a time): that 360 world repainted around the moment
                         └─> animator (Luma Ray2, one clip at a time, action moments only) ─> clip critic (OpenCV)"""
import json
import logging
import math
import os
import re
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from difflib import SequenceMatcher

import cv2 as cv
import numpy as np

from agents import animator, imagery, painter, planner
from agents.llm import LLMUnavailable
from agents.storyteller import DEFAULT_NEGATIVE
from agents.voice import speak_line, voice_for
from vision import clip as clip_critic
from vision.actors import make_actor
from vision.life import analyse as world_life
from vision.page import find_illustration, flatten_page, recognize_lines
from vision.shot import TEXT_LIMIT, erase_figures, maps_and_regions, text_cover, world_from_wide
from vision.world import composite, opencv_surround, place

from .stories import paint_world

log = logging.getLogger(__name__)
MAX_PENDING = 3       # live stories building at once (the image quota is shared)
DEADLINE_S = 300      # give up on whatever is still missing after this long
CLIP_DEADLINE_S = int(os.environ.get("CLIP_DEADLINE_S", "1500"))   # upgrades (worlds, clips) stop after this long
CLIP_SLACK_S = 30.0  # a clip already being made may finish this long after the deadline
ANIMATE_MAX = int(os.environ.get("ANIMATE_MAX", "0"))              # video clips per story; 0 = every painted moment
SHOT_ASPECT = os.environ.get("SHOT_ASPECT", "16:9")                 # moments are painted at the video's own shape
PAGE_PICTURE_MIN = 0.20  # a page picture covering at least this share of the page becomes the first moment
ANIMATE_RETRIES = int(os.environ.get("ANIMATE_RETRIES", "1"))
FULL_WORLDS = int(os.environ.get("FULL_WORLDS_PER_STORY", "6"))    # moments whose 360 world is repainted by AI
SCREEN_HFOV = 150.0   # what the theatre's curved screen spans
BIG_MOVES = ("fall", "jump", "rise", "walk-left", "walk-right")
PENDING_CLIP = ("queued", "generating", "checking")
FINAL_CLIP = ("ready", "rejected", "off")
FILE = re.compile(r"^(page\.jpg|world(_full)?\.jpg|world(_full)?_(depth|motion)\.png|shot_\d{2}\.jpg|"
                  r"shot_\d{2}_(depth|motion)\.png|line_\d{3}\.mp3|shot_\d{2}_actor\.png|shot_\d{2}_plate\.jpg|"
                  r"shot_\d{2}_world(_full)?\.jpg|shot_\d{2}_world(_full)?_(depth|motion)\.png|"
                  r"shot_\d{2}(_try\d)?\.mp4|shot_\d{2}_720\.jpg)$")
WATER = re.compile(r"\b(water|pond|river|lake|sea|ocean|stream|brook|waves?|puddles?|fountain|rain)\b", re.I)
NEGATIVE_SHOT = ("text, letters, words, caption, title, signature, watermark, logo, frame, border, panel, comic strip, "
                 "split screen, collage, blurry, photograph, 3d render, deformed")
NEGATIVE_WORLD = "people, person, child, figure, silhouette, animals, characters, " + NEGATIVE_SHOT
_voices = threading.Semaphore(3)  # lines voiced at once (two Polly calls each); more trips Polly's rate limit


def content_type(name: str) -> str | None:
    if not FILE.match(name):
        return None
    return {"jpg": "image/jpeg", "png": "image/png", "mp3": "audio/mpeg", "mp4": "video/mp4"}[name.rsplit(".", 1)[1]]


def _jpg(img: np.ndarray, q: int = 88) -> bytes:
    return cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _joined(lines: list[str]) -> tuple[str, list[int]]:
    """Normalized lines joined by spaces, and which line every character came from (-1 for the separators)."""
    text, owner = "", []
    for i, ln in enumerate(lines):
        n = _norm(ln)
        if text:
            text += " "
            owner.append(-1)
        text += n
        owner += [i] * len(n)
    return text, owner


def align(plan_lines: list[str], ocr_lines: list[str]) -> tuple[list[list[int]], list[int], float]:
    """Match the story agent's lines to the lines OpenCV read, character by character. Returns (for each planned
    line, the OCR lines it covers; OCR lines no planned line covers; share of OCR characters matched)."""
    p_text, p_owner = _joined(plan_lines)
    o_text, o_owner = _joined(ocr_lines)
    pairs: dict[tuple[int, int], int] = {}
    hits = [0] * len(ocr_lines)
    for a, b, size in SequenceMatcher(None, o_text, p_text, autojunk=False).get_matching_blocks():
        for k in range(size):
            oi, pi = o_owner[a + k], p_owner[b + k]
            if oi >= 0 and pi >= 0:
                pairs[(pi, oi)] = pairs.get((pi, oi), 0) + 1
                hits[oi] += 1
    lengths = [len(_norm(ln)) for ln in ocr_lines]
    cover = [sorted(oi for (pi, oi), n in pairs.items() if pi == p and n >= max(3, 0.3 * lengths[oi]))
             for p in range(len(plan_lines))]
    missed = [i for i, n in enumerate(lengths) if n and hits[i] < 0.3 * n]
    agreement = sum(hits) / max(1, sum(lengths))
    return cover, missed, round(agreement, 3)


def _story_like(line: str) -> bool:
    words = line.split()
    letters = sum(ch.isalpha() for ch in line)
    return (len(words) >= 3 and letters / max(1, len(line.replace(" ", ""))) > 0.75
            and not re.search(r"www|http|\.com|@|©", line, re.I))


_SHAPES = str.maketrans({"i": "l", "1": "l", "|": "l", "0": "o"})  # letters a recognizer confuses


def _fix_words(text: str, vocab: set[str]) -> str:
    """OCR slips ("ittle", "iight") snapped to the nearest word the story agent read elsewhere on the page. Words
    are compared by shape first (i, l and 1 look alike), so "iight" becomes "light", not "night"."""
    out = []
    for w in text.split():
        core = re.sub(r"[^A-Za-z']", "", w)
        low = core.lower()
        if core and low not in vocab and vocab:
            def score(v):
                return (SequenceMatcher(None, low.translate(_SHAPES), v.translate(_SHAPES)).ratio(),
                        SequenceMatcher(None, low, v).ratio(), v)
            best = max(sorted(vocab), key=score)
            if score(best)[0] >= 0.75:
                w = w.replace(core, best.capitalize() if core[0].isupper() else best)
        out.append(w)
    return " ".join(out)


def _trim_overlap(before: str, text: str) -> str:
    """A restored printed line can begin with words the line before it already ends with ("garden," then
    "garden, and squeezed under the gate!"): drop the repeat."""
    a, b = before.split(), text.split()
    for k in range(min(5, len(a), len(b) - 1), 0, -1):
        if [_norm(w) for w in a[-k:]] == [_norm(w) for w in b[:k]]:
            return " ".join(b[k:])
    return text


def _union_box(quads: list) -> list | None:
    if not quads:
        return None
    pts = np.array([p for q in quads for p in q], np.float32)
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    return [[round(float(x0), 1), round(float(y0), 1)], [round(float(x1), 1), round(float(y0), 1)],
            [round(float(x1), 1), round(float(y1), 1)], [round(float(x0), 1), round(float(y1), 1)]]


class Live:
    def __init__(self, store, on_ready=None):
        self.store = store
        self.on_ready = on_ready  # (story id, title, flattened page) once a story is finished
        self.runs: dict[str, dict] = {}  # stories being built, by id (the browser polls these)
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=32)
        self.world_pool = ThreadPoolExecutor(max_workers=1)  # full 360 worlds, one at a time (shared image quota)

    # ---------- public ----------
    def _building(self) -> int:
        return sum(1 for s in self.runs.values() if s.get("status", "running") == "running")

    def busy(self) -> bool:
        with self.lock:
            return self._building() >= MAX_PENDING

    def submit(self, img: np.ndarray) -> dict | None:
        live_id = uuid.uuid4().hex[:12]
        models = {"reader": "OpenCV 5: PP-OCRv3 text detection + CRNN recognition (cv.dnn)",
                  "planner": planner.LABEL, "painter": imagery.LABELS["stable-image-core"],
                  "voice": "Amazon Polly (neural)", "depth": "Depth Anything V2 Small (cv.dnn)",
                  "actors": "OpenCV 5: YOLOX + depth pop-out + GrabCut + shift-map inpainting"}
        filming = animator.enabled()
        if filming:
            models["animator"] = animator.label()
            models["critic"] = ("OpenCV 5: ORB + RANSAC fidelity, Farneback optical flow, PP-OCRv3 lettering, "
                                "YOLOX figures, SSIM cuts")
        st = {"id": live_id, "status": "running", "error": None, "created": time.time(), "stage": "Reading the page",
              "title": None, "style": None, "place": None, "cast": [], "time": "day", "weather": "clear",
              "wind": "breeze", "page": None, "lines": [], "shots": [], "upgrading": False,
              # When the story may start: "all" (default with video) = once it is completely made, every clip made
              # or given up; "clip" = once the first moment is filmed; "still" = once the first moment is painted.
              "play_when": (os.environ.get("PLAY_WHEN", "all") if os.environ.get("PLAY_WHEN", "all") in
                            ("all", "clip", "still") else "all") if filming else "still",
              "world": {"image": None, "ready": False, "source": None}, "check": None, "models": models,
              "timings": dict.fromkeys(("read", "header", "planned_first", "plan", "world", "first_shot", "playable",
                                        "shots", "voices", "world_full", "first_clip", "clips", "film", "all",
                                        "upgrades")),
              "_t0": time.monotonic(), "_plan_done": False, "_slow": []}
        with self.lock:
            if self._building() >= MAX_PENDING:
                return None
            self.runs[live_id] = st
        self._save(st)
        threading.Thread(target=self._run, args=(st, img), daemon=True).start()
        return self.get(live_id)

    def get(self, live_id: str) -> dict | None:
        with self.lock:
            st = self.runs.get(live_id)
            if st is not None:
                return self._snapshot(st)
        raw = self.store.get(f"live/{live_id}/state.json")
        if not raw:
            return None
        rec = json.loads(raw)
        if rec["status"] == "running":  # saved by a process that is gone (restart): don't let the browser poll forever
            rec.update(status="failed", error="The server restarted while building this story. Please try again.")
        if rec.get("upgrading"):  # the same for upgrades still pending when the process went away
            rec["upgrading"] = False
            for s in rec.get("shots", []):
                if (s.get("world") or {}).get("upgrading"):
                    s["world"]["upgrading"] = False
                if (s.get("clip") or {}).get("status") in PENDING_CLIP:
                    s["clip"].update(status="off", reason="the server restarted")
            rec.get("world", {})["upgrading"] = False
        return rec

    def recent(self, limit: int = 24) -> list[dict]:
        out = []
        for lid in self.store.list_dirs("live/")[:100]:
            raw = self.store.get(f"live/{lid}/state.json")
            rec = json.loads(raw) if raw else None
            if rec and rec["status"] == "ready" and rec["shots"]:
                out.append({"id": rec["id"], "title": rec["title"], "created": rec["created"],
                            "thumb": rec["shots"][0]["image"]})
        out.sort(key=lambda r: r["created"], reverse=True)
        return out[:limit]

    def file(self, live_id: str, name: str) -> bytes | None:
        return self.store.get(f"live/{live_id}/{name}") if content_type(name) else None

    # ---------- state ----------
    def _elapsed(self, st: dict) -> float:
        return round(time.monotonic() - st["_t0"], 2)

    def _mark(self, st: dict, key: str) -> None:
        if st["timings"][key] is None:
            st["timings"][key] = self._elapsed(st)

    def _snapshot(self, st: dict) -> dict:
        out = json.loads(json.dumps({k: v for k, v in st.items() if not k.startswith("_")}))
        t = out["timings"]
        shots_ready = sum(s["ready"] for s in out["shots"])
        busy = out["status"] == "running" or (st.get("_slow") and any(not f.done() for f in st["_slow"]))
        out["elapsed_s"] = self._elapsed(st) if busy else t["upgrades"] or t["all"]
        out["steps"] = [
            {"name": "read", "label": "Reading the page (OpenCV)", "done": t["read"] is not None, "seconds": t["read"]},
            {"name": "plan", "label": f"Writing the story ({planner.LABEL.replace(' on Bedrock', '')})",
             "done": t["plan"] is not None, "seconds": t["plan"]},
            {"name": "world", "label": "Painting the world", "done": out["world"]["ready"], "seconds": t["world"]},
            {"name": "shots", "label": "Painting the moments", "done": t["shots"] is not None, "seconds": t["shots"],
             "done_count": shots_ready, "total": len(out["shots"])},
            {"name": "voices", "label": "Giving it voices (Amazon Polly)", "done": t["voices"] is not None,
             "seconds": t["voices"]},
        ]
        if out["world"].get("upgrading") or t["world_full"] is not None:
            out["steps"].append({"name": "world_full", "label": "Painting the full 360° world (Stability AI)",
                                 "done": t["world_full"] is not None, "seconds": t["world_full"]})
        worlds = [s["world"] for s in out["shots"] if s.get("world")]
        if any(w.get("quality") == "full" or w.get("upgrading") for w in worlds):
            full = sum(w.get("quality") == "full" for w in worlds)
            out["steps"].append({"name": "worlds", "label": "Painting each moment's 360° world (Stability AI)",
                                 "done": not any(w.get("upgrading") for w in worlds), "done_count": full,
                                 "total": sum(w.get("quality") == "full" or bool(w.get("upgrading")) for w in worlds),
                                 "seconds": t["upgrades"]})
        clips = [s["clip"] for s in out["shots"] if s.get("clip") and s["clip"]["status"] != "off"]
        if clips:
            out["steps"].append({"name": "clips", "label": f"Filming the moments ({'Nova Reel' if animator.is_nova() else 'Luma Ray2'}) · checked by OpenCV",
                                 "done": not any(c["status"] in PENDING_CLIP for c in clips),
                                 "done_count": sum(c["status"] == "ready" for c in clips), "total": len(clips),
                                 "seconds": t["clips"]})
        out["upgrading"] = self._upgrading(out)
        return out

    @staticmethod
    def _upgrading(st: dict) -> bool:
        return bool(st["world"].get("upgrading")
                    or any((s.get("world") or {}).get("upgrading") for s in st["shots"])
                    or any((s.get("clip") or {}).get("status") in PENDING_CLIP for s in st["shots"]))

    def _save(self, st: dict) -> None:
        self.store.put(f"live/{st['id']}/state.json", json.dumps(self._snapshot(st)).encode(), "application/json")

    def _put(self, st: dict, name: str, data: bytes) -> None:
        self.store.put(f"live/{st['id']}/{name}", data, content_type(name))

    def _check_progress(self, st: dict) -> None:
        """Called under the lock whenever something becomes ready."""
        lines, shots = st["lines"], st["shots"]
        if shots and shots[0]["ready"] and st["world"]["ready"] and all(lines[i]["ready"] for i in shots[0]["lines"]):
            self._mark(st, "playable")
            first = (shots[0].get("clip") or {}).get("status")
            if st.get("play_when") == "clip" and first in FINAL_CLIP:
                self._mark(st, "film")  # the story can start, fully in motion
            settled = lambda s: ((s.get("clip") or {}).get("status") in FINAL_CLIP
                                 or (s.get("clip") is None and s.get("source") == "opencv"))
            if (st.get("play_when") == "all" and st["_plan_done"] and all(s["ready"] and settled(s) for s in shots)
                    and all(ln["ready"] for ln in lines)):
                self._mark(st, "film")  # completely made: every moment painted, voiced, and filmed or given up
        if st["_plan_done"]:
            if shots and all(s["ready"] for s in shots):
                self._mark(st, "shots")
            if lines and all(ln["ready"] for ln in lines):
                self._mark(st, "voices")
            ready = sum(s["ready"] for s in shots)
            st["stage"] = (f"Painting moment {ready + 1} of {len(shots)}" if ready < len(shots)
                           else "Finishing the voices" if st["timings"]["voices"] is None else "Ready")
            if ready == len(shots) and st["timings"]["voices"] is not None:
                doing = []
                clip = next(((k, s["clip"]) for k, s in enumerate(shots)
                             if (s.get("clip") or {}).get("status") in PENDING_CLIP), None)
                if clip:
                    k, c = clip
                    clips = [s.get("clip") or {} for s in shots]
                    total = sum(1 for x in clips if x.get("status") not in (None, "off"))
                    done = sum(1 for x in clips if x.get("status") in ("ready", "rejected"))
                    name = "Nova Reel, ~90 s" if animator.is_nova() else "Luma Ray2, 2–5 min"
                    doing.append(f"Filming your story: {done} of {total} scenes ({name})" if c["status"] != "checking"
                                 else f"OpenCV is checking the clip for moment {k + 1} ({done} of {total} filmed)")
                if st["world"].get("upgrading") or any((s.get("world") or {}).get("upgrading") for s in shots):
                    doing.append("painting the 360° worlds")
                if doing:
                    st["stage"] = " · ".join(doing)[:1].upper() + " · ".join(doing)[1:]

    # ---------- the agents ----------
    def _run(self, st: dict, img: np.ndarray) -> None:
        tasks: list = []
        page = None
        try:
            page, quads, _ = flatten_page(img)
            page_jpg = _jpg(page, 85)
            self._put(st, "page.jpg", page_jpg)
            mask = np.zeros(page.shape[:2], np.uint8)
            for q in quads:
                cv.fillPoly(mask, [q.astype(np.int32)], 255)
            box = find_illustration(page, mask)
            fallback = page[box[1]:box[3], box[0]:box[2]] if box else page
            if box and (box[2] - box[0]) * (box[3] - box[1]) >= PAGE_PICTURE_MIN * page.shape[0] * page.shape[1]:
                st["_page_picture"] = fallback  # the first moment starts from the page's own picture
            with self.lock:
                st["page"] = {"image": "page.jpg", "w": page.shape[1], "h": page.shape[0]}
                st["stage"] = "Writing the story"
            seed = int(st["id"][:6], 16)
            st["_tasks"] = tasks
            ocr_job = self.pool.submit(recognize_lines, page, quads)
            plan_job = self.pool.submit(self._plan, st, page_jpg, len(quads), fallback, seed, tasks)
            ocr = ocr_job.result()
            with self.lock:
                self._mark(st, "read")
            try:
                plan_job.result()
            except LLMUnavailable as e:
                with self.lock:
                    have = bool(st["shots"])
                if not have and (isinstance(e, planner.ContentFiltered) or planner.enabled()):
                    # Blocked (a published page the model may not copy), throttled or unparseable: ask again with
                    # OpenCV's numbered lines, so the model only points at lines and fixes misread words; the words
                    # themselves come from OpenCV. Only if that fails too does the raw reading play.
                    log.warning("story agent failed (%s); asking again in lines mode", e)
                    sent_header = getattr(e, "header_sent", False)
                    for attempt in (1, 2):  # the filter is not deterministic: one more try is often enough
                        try:
                            planner.plan_from_lines(
                                page_jpg, [ln["text"] for ln in ocr],
                                lambda h: self._on_header(st, h, fallback, seed, tasks) if not sent_header
                                else self._merge_cast(st, h),
                                lambda sh, extend=False: self._on_shot(st, sh, fallback, seed, tasks, extend),
                                send_header=True)
                            with self.lock:
                                st["models"]["planner"] = (f"{planner.LABEL} (lines mode: it pointed at OpenCV's "
                                                           f"lines, the page's words were not copied)")
                                have = bool(st["shots"])
                            break
                        except planner.ContentFiltered as e2:
                            log.warning("lines mode blocked too (attempt %d): %s", attempt, e2)
                        except LLMUnavailable as e2:  # throttled or a broken reply: one more try, after a breath
                            log.warning("lines mode failed (attempt %d): %s", attempt, e2)
                            time.sleep(2)
                if not have:
                    log.warning("story agent unavailable (%s); planning from OpenCV's reading", e)
                    header, shots = planner.fallback_plan([ln["text"] for ln in ocr])
                    with self.lock:
                        st["models"]["planner"] = "the page's lines as OpenCV read them (no model)"
                    self._on_header(st, header, fallback, seed, tasks)
                    for s in shots:
                        self._on_shot(st, s, fallback, seed, tasks)
            self._check_reading(st, ocr, tasks)
            with self.lock:
                st["_plan_done"] = True
                self._mark(st, "plan")
                self._check_progress(st)
            self._plan_clips(st)
            late = self._wait_all(st, tasks, DEADLINE_S)
            with self.lock:
                if late:
                    log.warning("live story %s: %d tasks still running at the deadline", st["id"], len(late))
                    for s in st["shots"]:
                        s["ready"] = True
                    for ln in st["lines"]:
                        ln["ready"] = True
                st["status"], st["stage"] = "ready", "Ready"
                self._mark(st, "all")
                self._check_progress(st)
                self._save(st)
            if self.on_ready and page is not None:
                self.on_ready(st["id"], st["title"] or "", page)
            # The story plays from here; its moments keep getting more alive (worlds, clips) while it does.
            late = self._wait_all(st, st["_slow"], CLIP_DEADLINE_S)
            if late:
                log.warning("live story %s: %d upgrades still running at the deadline", st["id"], len(late))
                for f in late:
                    f.cancel()
                with self.lock:
                    for s in st["shots"]:
                        if (s.get("clip") or {}).get("status") in PENDING_CLIP:
                            s["clip"].update(status="off", reason="out of time")
        except Exception as e:  # surface failures to the browser instead of a stuck job
            traceback.print_exc()
            with self.lock:
                if st["status"] == "running":
                    st.update(status="failed", error=f"{type(e).__name__}: {e}")
        finally:
            with self.lock:
                self._mark(st, "all")
                if st["_slow"]:
                    self._mark(st, "upgrades")
                st["world"]["upgrading"] = False
                for s in st["shots"]:
                    if (s.get("world") or {}).get("upgrading"):
                        s["world"]["upgrading"] = False
                self._check_progress(st)
                self._save(st)
                self.runs.pop(st["id"], None)

    def _wait_all(self, st: dict, tasks: list, deadline_s: float) -> list:
        """Wait for a task list that keeps growing (moments add their own tasks); returns what's left at the deadline."""
        while True:
            pending = [t for t in tasks if not t.done()]
            if not pending:
                return []
            _, late = wait(pending, timeout=max(1.0, deadline_s - self._elapsed(st)))
            if late and self._elapsed(st) >= deadline_s - 1:
                return list(late)

    def _plan(self, st: dict, page_jpg: bytes, line_count: int, fallback: np.ndarray, seed: int, tasks: list) -> None:
        planner.plan(page_jpg, lambda h: self._on_header(st, h, fallback, seed, tasks),
                     lambda s, extend=False: self._on_shot(st, s, fallback, seed, tasks, extend), line_count)

    def _on_header(self, st: dict, h: dict, fallback: np.ndarray, seed: int, tasks: list) -> None:
        with self.lock:
            st.update(title=st["title"] or h["title"], style=h["style"], place=h["place"], cast=h["cast"],
                      time=h.get("time", "day"), weather=h.get("weather", "clear"), wind=h.get("wind", "breeze"))
            self._mark(st, "header")
        tasks.append(self.pool.submit(self._world, st, h, fallback, seed))

    def _merge_cast(self, st: dict, h: dict) -> None:
        """A second header (lines mode after a cut-off first reply): keep the world already being painted, take the
        cast if the first header had none."""
        with self.lock:
            if not st["cast"] and h.get("cast"):
                st["cast"] = h["cast"]

    def _on_shot(self, st: dict, s: dict, fallback: np.ndarray, seed: int, tasks: list, extend: bool = False) -> None:
        with self.lock:
            created = not (extend and st["shots"])  # past the moment limit, the lines join the last moment
            if created:
                action = dict(s.get("action") or planner.STILL)
                names = {c["name"].lower(): c["name"] for c in st["cast"]}
                action["subject"] = names.get(action["subject"].lower(), action["subject"])
                shot = {"i": len(st["shots"]), "lines": [], "effects": s["effects"], "prompt": s["see"],
                        "caption": s["see"].split(".")[0][:120], "image": None, "depth": None, "aspect": None,
                        "ready": False, "source": None, "action": action, "actor": None, "plate": None,
                        "actor_box": None, "actor_check": None, "world": None, "clip": None}
                st["shots"].append(shot)
                self._mark(st, "planned_first")
            else:
                shot = st["shots"][-1]
            new = []
            for text, speaker in zip(s["text"], s["speaker"]):
                line = {"i": len(st["lines"]), "text": text, "speaker": speaker, "voice": None, "box": None,
                        "audio": None, "marks": None, "duration": None, "ready": False}
                st["lines"].append(line)
                shot["lines"].append(line["i"])
                new.append(line)
        tasks.extend(self.pool.submit(self._voice, st, line) for line in new)
        if created:
            tasks.append(self.pool.submit(self._shot, st, shot, fallback, seed))

    def _world(self, st: dict, h: dict, fallback: np.ndarray, seed: int) -> None:
        prompt = (f"An uninhabited ultra-wide panoramic landscape, nobody in it: {h['place']}. {h['style']}. Horizon "
                  f"across the middle, open sky above, ground below, scenery continuing on every side.")
        img, source = imagery.generate(prompt, NEGATIVE_WORLD, "21:9", seed)
        erased = erase_figures(img) if img is not None else 0  # the painter adds a few figures anyway
        if img is None and fallback is not None and fallback.shape[1] > fallback.shape[0]:
            img, source = fallback, "opencv"  # a wide picture from the page itself still makes a world
        world = {"image": None, "ready": True, "source": "none"}
        if img is not None:
            pano = world_from_wide(img)
            self._put(st, "world.jpg", _jpg(pano, 86))
            world = {"image": "world.jpg", "ready": True, "source": source, "figures_erased": erased,
                     "quality": "quick", **self._world_maps(st, pano, "world")}
            st["_world_pano"] = pano  # each moment's own world is composed onto this
        with self.lock:
            st["world"] = world
            self._mark(st, "world")
            self._check_progress(st)

    def _shot(self, st: dict, shot: dict, fallback: np.ndarray, seed: int) -> None:
        with self.lock:
            k, see, style = shot["i"], shot["prompt"], st["style"]
            speakers = {st["lines"][i]["speaker"] for i in shot["lines"]}
            subject = shot["action"].get("subject") or ""
            # Everyone the moment names (in what to show, among the speakers, or doing the action) is painted from
            # their fixed description; else "Tortoise and Hare decide to race" came out as three tortoises.
            looks = [f"{c['name']}: {c['look']}" for c in st["cast"]
                     if c["name"] in see or c["name"] in speakers or c["name"].lower() in subject.lower()]
        prompt = f"{see} {'Characters: ' + '; '.join(looks) + '. ' if looks else ''}{style}. Wide cinematic storybook scene."
        img, source = (self._page_picture(st, prompt, seed) if k == 0 else (None, None))
        if img is None:
            img, source = imagery.generate(prompt, NEGATIVE_SHOT, SHOT_ASPECT, seed + 101 * (k + 1))
        else:
            with self.lock:
                shot["from_page"] = True
        cover = text_cover(img) if img is not None else 0.0
        if cover > TEXT_LIMIT:  # the critic: lettering in the picture; paint it again if the quota allows right now
            again, source2 = imagery.generate(prompt + " A wordless scene, no signs.", NEGATIVE_SHOT, SHOT_ASPECT,
                                              seed + 101 * (k + 1) + 7, wait_s=0)
            if again is not None and text_cover(again) < cover:
                img, source, cover = again, source2, text_cover(again)
        if img is None:
            img, source = fallback, "opencv"
        name = f"shot_{k:02d}"
        # Depth for the relief, and where it can move: the story says whether there is water, OpenCV finds where.
        depth_map, motion_map, moving, _ = maps_and_regions(img, water=bool(WATER.search(see)))
        self._put(st, f"{name}.jpg", _jpg(img))
        self._put(st, f"{name}_depth.png", depth_map)
        self._put(st, f"{name}_motion.png", motion_map)
        painted = source != "opencv"
        film = painted and animator.enabled()
        if film:  # the video's start frame: exactly 1280x720 (Nova Reel's only size)
            self._put(st, f"{name}_720.jpg", _jpg(animator.start_frame(img), 92))
        with self.lock:
            shot.update(image=f"{name}.jpg", depth=f"{name}_depth.png", motion=f"{name}_motion.png", moving=moving,
                        aspect=round(img.shape[1] / img.shape[0], 3), ready=True, source=source,
                        text_cover=round(cover, 4))
            if film:
                filmed = sum(1 for s in st["shots"] if (s.get("clip") or {}).get("status") not in (None, "off"))
                if not (shot["action"].get("motion") or "").strip():
                    # Nothing described as happening: a clip would only drift. Keep the living painting (and save
                    # the wait and the cost).
                    shot["clip"] = {"status": "off", "reason": "nothing to film: a living painting"}
                elif ANIMATE_MAX and filmed >= ANIMATE_MAX:
                    shot["clip"] = {"status": "off", "reason": "page limit"}
                else:  # filmed now, in parallel with every other moment, while the rest is still being made
                    shot["clip"] = self._new_clip(f"{name}_720.jpg")
                    st["_slow"].append(self.pool.submit(self._clip_job, st, shot, k, seed))
            if k == 0:
                self._mark(st, "first_shot")
            self._check_progress(st)
        # The moment keeps getting more alive while the story plays.
        if painted and shot["action"]["path"] != "still":
            st["_tasks"].append(self.pool.submit(self._actor, st, shot, img))
        st["_tasks"].append(self.pool.submit(self._moment_world, st, shot, img, seed))

    def _page_picture(self, st: dict, prompt: str, seed: int) -> tuple[np.ndarray | None, str | None]:
        """The page's own picture as the first moment: fitted into the 16:9 frame and the margins painted by
        Stability AI Inpaint to continue it, so the first clip brings the page's own drawing to life."""
        pic = st.get("_page_picture")
        if pic is None or not painter.enabled():
            return None, None
        if text_cover(pic) > TEXT_LIMIT:  # the crop caught the page's printing: a film of it would show letters
            log.info("the page's picture has text in it; painting the first moment instead")
            st["_page_picture"] = None
            return None, None
        W, H = 1344, 756
        s = min(W / pic.shape[1], H / pic.shape[0])
        fit = cv.resize(pic, (max(1, round(pic.shape[1] * s)), max(1, round(pic.shape[0] * s))),
                        interpolation=cv.INTER_AREA if s < 1 else cv.INTER_CUBIC)
        y0, x0 = (H - fit.shape[0]) // 2, (W - fit.shape[1]) // 2
        canvas = cv.copyMakeBorder(fit, y0, H - fit.shape[0] - y0, x0, W - fit.shape[1] - x0, cv.BORDER_REFLECT)
        if fit.shape[1] >= W - 8 and fit.shape[0] >= H - 8:
            return canvas, "page"  # already the right shape
        grow = np.full((H, W), 255, np.uint8)
        grow[y0 + 6:y0 + fit.shape[0] - 6, x0 + 6:x0 + fit.shape[1] - 6] = 0
        out = painter.paint(canvas, grow, prompt + " Continue the picture seamlessly to the edges.", NEGATIVE_SHOT,
                            seed=seed + 3)
        return (out, "page + Stability AI Inpaint") if out is not None else (None, None)

    def _actor(self, st: dict, shot: dict, img: np.ndarray) -> None:
        """The actor agent (OpenCV): the character this moment is about, found with the story's own words for it, cut
        out so the theatre can move it; Stability AI Inpaint paints the scenery that was behind it (the plate)."""
        with self.lock:
            who = shot["action"]["subject"]
            look = next((c["look"] for c in st["cast"] if c["name"] == who), "")
            plate_prompt = (f"{st['place']}. {st['style']}. The same scene with nobody there: only the scenery "
                            f"continues, seamless.")
        paint_fn = ((lambda canvas, hole: painter.paint(canvas, hole, plate_prompt, NEGATIVE_WORLD, seed=shot["i"] + 77))
                    if painter.enabled() else None)
        try:
            out = make_actor(img, f"{who} {look}", paint_fn)
        except Exception:
            traceback.print_exc()
            out = {"check": {"ok": False, "reason": "actor agent error"}}
        stem = shot["image"].rsplit(".", 1)[0]
        if "actor" in out:
            self._put(st, f"{stem}_actor.png", cv.imencode(".png", out["actor"])[1].tobytes())
            self._put(st, f"{stem}_plate.jpg", _jpg(out["plate"]))
        with self.lock:
            shot["actor_check"] = out["check"]
            if "actor" in out:
                shot.update(actor=f"{stem}_actor.png", plate=f"{stem}_plate.jpg", actor_box=out["box"])

    def _moment_world(self, st: dict, shot: dict, img: np.ndarray, seed: int) -> None:
        """The world composer (OpenCV): the moment set into the story's world at its true angular size, so when the
        story moves on, the world around the viewer moves with it. Then (if painting is on) the world agent repaints
        that 360 world around the moment itself, in the background, one moment at a time."""
        stem = shot["image"].rsplit(".", 1)[0]
        aspect = img.shape[1] / img.shape[0]
        vfov = 2 * math.degrees(math.atan(1.309 / aspect))  # what the curved screen (150 deg, radius 7 m) spans
        for _ in range(40):  # the quick world is usually painted alongside the first moment
            if st.get("_world_pano") is not None or st["world"]["ready"]:
                break
            time.sleep(0.25)
        backdrop = st.get("_world_pano")
        if backdrop is None:
            backdrop = cv.resize(opencv_surround(img, SCREEN_HFOV, vfov_deg=vfov)[0], (3072, 1536))
        h, w = backdrop.shape[:2]
        canvas, keep, _ = place(img, SCREEN_HFOV, pano_w=w, pano_h=h, vfov_deg=vfov)
        pano = composite(backdrop, canvas, keep)
        self._put(st, f"{stem}_world.jpg", _jpg(pano, 86))
        maps = self._world_maps(st, pano, f"{stem}_world")
        with self.lock:
            k = shot["i"]
            full = (painter.enabled() and shot["source"] not in (None, "opencv") and k < FULL_WORLDS)
            shot["world"] = {"image": f"{stem}_world.jpg", "quality": "quick", "upgrading": full, **maps}
            self._check_progress(st)
        if full:
            with self.lock:
                if k == 0:
                    st["world"]["upgrading"] = True
                st["_slow"].append(self.world_pool.submit(self._world_full, st, img, seed, shot))

    def _world_maps(self, st: dict, pano: np.ndarray, name: str) -> dict:
        """Depth (so the world has relief as you glide through it), motion (sky, water, leaves) and life (wind,
        light, lamps, haze) for a world."""
        depth_map, motion_map, moving, regions = maps_and_regions(pano, width=1024,
                                                                  water=bool(WATER.search(st.get("place") or "")))
        self._put(st, f"{name}_depth.png", depth_map)
        self._put(st, f"{name}_motion.png", motion_map)
        try:
            life = world_life(pano, regions["sky"], {k: st.get(k) for k in ("time", "weather", "wind")})
        except Exception:
            traceback.print_exc()
            life = None
        return {"depth": f"{name}_depth.png", "motion": f"{name}_motion.png", "moving": moving, "life": life}

    def _world_full(self, st: dict, moment: np.ndarray, seed: int, shot: dict | None = None) -> None:
        """The world agent: a high-resolution 360 world painted outward from a moment itself (5 overlapping
        Stability Inpaint tiles, each continuing its neighbours, checked by OpenCV for empty fields and stray
        figures), so the scenery around you continues the picture in front of you. About 2 minutes per moment;
        the story plays meanwhile and the browser fades the new world in. Without `shot`, the story's first world."""
        k = shot["i"] if shot else 0
        with self.lock:
            story = {"world_prompt": f"{st['place']}. {st['style']}", "negative_prompt": DEFAULT_NEGATIVE}
            if k == 0:
                st["world"]["upgrading"] = True
        aspect = moment.shape[1] / moment.shape[0]
        vfov = 2 * math.degrees(math.atan(1.309 / aspect))  # what the curved screen (150 deg, radius 7 m) spans
        matte = np.full(moment.shape[:2], 255, np.uint8)
        try:
            pano, meta = paint_world(moment, matte, SCREEN_HFOV, story, seed=seed + 999 + k, vfov=vfov)
        except Exception:
            traceback.print_exc()
            meta = {"ai_filled": False}
        if not meta.get("ai_filled"):
            with self.lock:
                if k == 0:
                    st["world"]["upgrading"] = False
                if shot and shot.get("world"):
                    shot["world"]["upgrading"] = False
                self._check_progress(st)
            return
        stem = shot["image"].rsplit(".", 1)[0] + "_world_full" if shot else "world_full"
        self._put(st, f"{stem}.jpg", _jpg(pano, 88))
        maps = self._world_maps(st, pano, stem)
        extra = {"tiles": meta.get("tiles"), "stray_figures_removed": meta.get("stray_figures_removed")}
        with self.lock:
            if shot is not None:
                shot["world"] = {"image": f"{stem}.jpg", "quality": "full", "upgrading": False, **maps, **extra}
            if k == 0:
                st["world"].update(image=f"{stem}.jpg", quality="full", upgrading=False, **extra, **maps)
                self._mark(st, "world_full")
            self._check_progress(st)
            self._save(st)

    # ---------- the animator (Nova Reel; Luma as a fallback) and its critic (OpenCV) ----------
    @staticmethod
    def _new_clip(start: str) -> dict:
        return {"status": "queued", "reason": None, "video": None, "start": start, "attempts": 0, "tries": [],
                "model": animator.model(), "resolution": animator.resolution(), "verdict": None, "loop_at": None,
                "hold": True, "seconds": None, "from_still": True}

    def _plan_clips(self, st: dict) -> None:
        """Moments are filmed as soon as they are painted (`_shot`); this only says why the others aren't."""
        if not animator.enabled():
            return
        with self.lock:
            for s in st["shots"]:
                if s.get("clip") is None and s.get("ready"):
                    s["clip"] = {"status": "off", "reason": "not a painted moment" if s.get("source") == "opencv"
                                 else "page limit"}
            self._check_progress(st)

    def _clip_job(self, st: dict, shot: dict, rank: int, seed: int) -> None:
        try:
            self._clip(st, shot, rank, seed)
        except Exception as e:  # never a stuck story
            traceback.print_exc()
            with self.lock:
                shot["clip"].update(status="off", reason=f"animator error: {type(e).__name__}")
        with self.lock:
            clips = [s.get("clip") or {} for s in st["shots"]]
            if not any(c.get("status") in PENDING_CLIP for c in clips) and any(c.get("status") == "ready" for c in clips):
                self._mark(st, "clips")
            self._check_progress(st)
            self._save(st)

    def _clip(self, st: dict, shot: dict, rank: int, seed: int = 0) -> None:
        clip = shot["clip"]
        expires = st["_t0"] + CLIP_DEADLINE_S - 60
        stem = shot["image"].rsplit(".", 1)[0]
        start_jpg = self.store.get(f"live/{st['id']}/{clip.get('start') or shot['image']}")
        start = cv.imdecode(np.frombuffer(start_jpg, np.uint8), cv.IMREAD_COLOR)  # what the clip must begin from
        with self.lock:
            action = dict(shot["action"])
            look = next((c["look"] for c in st["cast"] if c["name"] == action["subject"]), None)
            style, see = st["style"], shot["prompt"]

        def status(s):
            with self.lock:
                clip["status"] = s
                self._check_progress(st)

        anonymous = False
        for attempt in range(1, 2 + ANIMATE_RETRIES):
            with self.lock:
                clip.update(status="queued", attempts=attempt)
                self._check_progress(st)
                cast = list(st["cast"])
            prompt = animator.clip_prompt(action, look, style, see, attempt)
            if anonymous:
                prompt = animator.anonymize(prompt, cast, action.get("subject", ""))
            fut = animator.submit((st["created"], rank, attempt), start_jpg, prompt,
                                  action["loop"] and action["path"] not in ("fall", "rise"), expires, status,
                                  seed=seed + 31 * rank + attempt)
            try:
                res = fut.result(timeout=max(1.0, expires - time.monotonic() + CLIP_SLACK_S))
            except animator.AnimatorError as e:
                with self.lock:
                    clip.update(status="off", reason=e.reason)
                if animator.moderation_refusal(e.reason) and not anonymous and attempt <= ANIMATE_RETRIES:
                    anonymous = True  # a well-known name was refused: try again describing the characters instead
                    continue
                if e.retry and attempt <= ANIMATE_RETRIES:
                    continue
                return
            except Exception:
                fut.cancel()
                with self.lock:
                    clip.update(status="off", reason="out of time")
                return
            status("checking")
            verdict = clip_critic.verify(start, res.mp4, action, self._box_in_start(st, shot, start), res.from_still)
            try_name = f"{stem}_try{attempt}.mp4"
            self._put(st, try_name, res.mp4)
            with self.lock:
                clip["tries"].append({"video": try_name, "verdict": verdict, "seconds": res.seconds,
                                      "prompt": prompt})
            if verdict["ok"]:
                self._put(st, f"{stem}.mp4", res.mp4)
                with self.lock:
                    clip.update(status="ready", video=f"{stem}.mp4", verdict=verdict, loop_at=verdict["loop_at"],
                                hold=verdict["hold"], size=verdict["size"], duration=verdict["duration"],
                                seconds=round(time.monotonic() - st["_t0"], 1), from_still=res.from_still,
                                prompt=prompt)
                    self._mark(st, "first_clip")
                return
            if not (verdict["retry"] and attempt <= ANIMATE_RETRIES):
                break
            if verdict.get("observed") == "nothing" and (shot.get("action") or {}).get("path", "still") == "still":
                break  # a frozen take of a quiet moment: a second take gets the same prompt, so keep the painting
        with self.lock:
            last = clip["tries"][-1]["verdict"] if clip["tries"] else None
            clip.update(status="rejected", verdict=last,
                        reason=(last["reasons"][0] if last and last["reasons"] else "the clip did not show the movement"))

    def _box_in_start(self, st: dict, shot: dict, start: np.ndarray) -> list | None:
        """The moving character's box (painting pixels) in the start frame's pixels, when the start frame is the
        painting resized (16:9); else unknown."""
        with self.lock:
            box, aspect = shot.get("actor_box"), shot.get("aspect") or 0
        if not box or abs(aspect - start.shape[1] / start.shape[0]) > 0.02:
            return None
        raw = self.store.get(f"live/{st['id']}/{shot['image']}")
        painting = cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_GRAYSCALE) if raw else None
        if painting is None:
            return None
        k = start.shape[1] / painting.shape[1]
        return [round(v * k) for v in box]

    def _voice(self, st: dict, line: dict) -> None:
        with self.lock:
            voice = voice_for(line["speaker"], st["cast"])
            text, i = line["text"], line["i"]
        with _voices:
            audio, marks = speak_line(text, voice)
        name = f"line_{i:03d}.mp3"
        if audio:
            self._put(st, name, audio)
        with self.lock:
            line.update(voice=voice, audio=name if audio else None, marks=marks,
                        duration=round(marks[-1]["t"] / 1000 + 0.6, 2) if marks else None, ready=True)
            self._check_progress(st)

    def _check_reading(self, st: dict, ocr: list[dict], tasks: list) -> None:
        """OpenCV checks the story agent: lines it skipped are put back where they belong (spelled with the words it
        did read), and every spoken line gets its box on the page so the reader can follow along."""
        with self.lock:
            plan_lines = [ln["text"] for ln in st["lines"]]
        ocr_text = [ln["text"] for ln in ocr]
        cover, missed, agreement = align(plan_lines, ocr_text)
        # Words the agent wrote twice that the page shows only once match nothing the second time: drop them.
        # Only when OpenCV read the page well (else everything matches poorly), and only repeats of earlier text,
        # so a chorus printed twice (which the page does support) stays.
        drop, before = set(), ""
        for p, ln in enumerate(plan_lines):
            n = _norm(ln)
            if agreement >= 0.7 and not cover[p] and n and n in before:
                drop.add(p)
            before += " " + n
        covered = {o for c in cover for o in c}
        vocab = {w for ln in plan_lines for w in _norm(ln).split()}
        restore = []
        for oi in missed:
            if not covered or oi < min(covered) or not _story_like(ocr_text[oi]):
                continue  # titles and headers above the story, decorations, links
            # Put it after the story line that covers the nearest line above it on the page.
            prev = max((p for p, c in enumerate(cover) if any(o < oi for o in c)), default=None)
            if prev is not None:
                restore.append((prev, oi))
        with self.lock:
            for p, c in enumerate(cover):
                if p < len(st["lines"]):
                    st["lines"][p]["box"] = _union_box([ocr[o]["box"] for o in c])
            added = []
            for p, oi in reversed(restore):  # from the bottom up, so earlier insert points stay valid
                owner = next((s for s in st["shots"] if p in s["lines"]), None)
                if owner is None:
                    continue
                text = _trim_overlap(st["lines"][p]["text"], _fix_words(ocr_text[oi], vocab))
                line = {"i": len(st["lines"]), "text": text, "speaker": st["lines"][p]["speaker"],
                        "voice": None, "box": ocr[oi]["box"], "audio": None, "marks": None, "duration": None,
                        "ready": False}
                st["lines"].append(line)
                owner["lines"].insert(owner["lines"].index(p) + 1, line["i"])
                added.append(line)
            if drop:
                for s in st["shots"]:
                    s["lines"] = [i for i in s["lines"] if i not in drop]
                st["shots"] = [s for s in st["shots"] if s["lines"]]
                for k, s in enumerate(st["shots"]):
                    s["i"] = k
                self._check_progress(st)
            st["check"] = {"ocr_lines": len(ocr), "story_lines": len(plan_lines), "agreement": agreement,
                           "restored": len(added), "dropped": len(drop)}
        tasks.extend(self.pool.submit(self._voice, st, line) for line in added)
