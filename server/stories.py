"""Book pages -> a story you can step into, one scene per page (a background job with stage-by-stage status)."""
import json
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2 as cv
import numpy as np

from agents import painter
from agents.llm import label as llm_label
from agents.storyteller import adapt
from agents.voice import synthesize, voice_for
from vision.detect import detect, spoken
from vision.page import annotate, read_page
from vision.scene25d import build as build_25d
from vision.matte import paper_matte
from vision.world import (MAX_REPAINTS, ai_canvas, ai_finish, fill_unpainted, keep_region, opencv_surround,
                          paint_tiled, repaint_figures, stray_figures, style_terms, tile_order)

log = logging.getLogger(__name__)
_pool = ThreadPoolExecutor(max_workers=1)
_lock = threading.Lock()
MAX_PENDING = 2  # stories queued or being built at once; every page makes several paid image calls


def _jpg(img: np.ndarray, q: int = 88) -> bytes:
    return cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def _small(img: np.ndarray, side: int = 768) -> np.ndarray:
    s = side / max(img.shape[:2])
    return cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else img


def paint_world(illus: np.ndarray, matte: np.ndarray, hfov: float, story: dict,
                layer_masks: list = (), progress=lambda text: None, seed: int = 7,
                vfov: float | None = None) -> tuple[np.ndarray, dict]:
    """Stability AI paints the 360 world around the real art (which is pasted back untouched); if it isn't
    available, OpenCV builds the surround from the illustration's own pixels. `seed` starts the painter's seed
    sequence: the painter is deterministic per seed, so creating a story again gives it a different world."""
    if painter.enabled():
        prep = ai_canvas(illus, matte, hfov, layer_masks, vfov)
        # A 360 view only works at eye level: horizon across the middle, sky above, ground below.
        prompt = (f"(hand-painted picture book illustration:1.5), (watercolour:1.4), (delicate pen and ink:1.2). "
                  f"Eye-level view from where the reader stands: {story['world_prompt']}. "
                  f"Horizon across the middle, open sky above, ground below, scenery receding into the distance. "
                  f"{style_terms(illus, matte)}. Seamless wide panoramic landscape continuing in every direction.")
        negative = (story["negative_prompt"] + ", top-down view, aerial view, overhead, map, flat lay, "
                    "saturated, dark, heavy black outlines, comic, harsh contrast, outline around objects, "
                    "(white margin:1.4), (white stripe:1.4), (panel:1.3), gutter, vertical line, split image, "
                    "collage, triptych, flat empty background")
        seeds = iter(range(seed, seed + 13 * 100, 13))
        parts, done = len(tile_order()), 0

        def paint_part(crop, mask):
            nonlocal done
            progress(f"painting the 360° world, part {min(done + 1, parts)} of {parts}")
            out = painter.paint(crop, mask, prompt, negative, seed=next(seeds))
            done += out is not None
            return out

        painted, stats, todo = paint_tiled(prep["canvas"], prep["generate"], paint_part)
        if stats["calls"]:
            painted = fill_unpainted(painted, todo, prep["fallback"])
            # People/animals the painter added despite the prompt: repaint those spots as scenery, then look again,
            # since a repaint can invent a new figure. At most MAX_REPAINTS repaints in all.
            scenery = "empty scenery, " + story["world_prompt"]
            found, removed, left = None, 0, 0
            for round_ in range(3):
                progress("checking the painted world for stray figures")
                strays = stray_figures(painted, prep["keep"], detect)
                found = len(strays) if found is None else found
                left = len(strays)
                boxes = strays[: MAX_REPAINTS - removed]
                if not boxes or round_ == 2:
                    break
                progress(f"repainting {len(boxes)} stray figure{'s' if len(boxes) > 1 else ''} as scenery")
                n = repaint_figures(painted, boxes, lambda c, m: painter.paint(c, m, scenery, negative, seed=next(seeds)))
                removed += n
                if n == 0:
                    break
            pano = ai_finish(painted, prep)
            unpainted = float((todo > 0).sum()) / max(1, int((prep["generate"] > 0).sum()))
            return pano, {
                "source": "stability-inpaint", "ai_filled": True, "model": painter.MODEL, "prompt": prompt,
                "tiles": stats["calls"], "flat_tiles_repainted": stats["flat_tiles"], "failed_calls": stats["failed_calls"],
                "complete": unpainted == 0, "painted_fraction": round(1 - unpainted, 3),
                "stray_figures_found": found, "stray_figures_removed": removed, "stray_figures_left": left,
                "real_fraction": round(float(prep["keep"].mean() / 255), 4)}
    clean = cv.inpaint(illus, 255 - keep_region(matte), 5, cv.INPAINT_TELEA)  # no white paper in the mirror loop
    pano, _ = opencv_surround(clean, hfov)
    return pano, {"source": "opencv", "ai_filled": False}


class Stories:
    def __init__(self, store, scenes):
        self.store, self.scenes = store, scenes
        self.active: set[str] = set()  # stories this process has queued or is building

    def save(self, rec: dict) -> None:
        rec["updated"] = time.time()
        self.store.put(f"stories/{rec['id']}/story.json", json.dumps(rec).encode(), "application/json")

    def _load(self, story_id: str) -> dict | None:
        raw = self.store.get(f"stories/{story_id}/story.json")
        return json.loads(raw) if raw else None

    def get(self, story_id: str) -> dict | None:
        rec = self._load(story_id)
        if rec and rec["status"] == "running":
            with _lock:
                if story_id not in self.active:
                    # Nothing here is building it: the server restarted mid-story. Say so rather than poll forever.
                    rec = self._load(story_id)  # re-read: it may have finished a moment ago
                    if rec and rec["status"] == "running":
                        rec.update(status="failed", error="The server restarted while building this story. "
                                                          "Please create it again.")
                        self.save(rec)
        return rec

    def recent(self, limit: int = 24) -> list[dict]:
        recs = [self._load(sid) for sid in self.store.list_dirs("stories/")[:100]]  # one GET each on S3
        done = [r for r in recs if r and r["status"] == "done" and r["pages"]]
        done.sort(key=lambda r: r.get("updated", 0), reverse=True)  # S3 lists folders by id, not by date
        return [{"id": r["id"], "title": r["title"], "pages": r["pages"]} for r in done[:limit]]

    def busy(self) -> bool:
        with _lock:
            return len(self.active) >= MAX_PENDING

    def submit(self, pages: list[np.ndarray], title: str) -> dict | None:
        """Queue a story. None if MAX_PENDING stories are already queued or being built."""
        rec = {"id": uuid.uuid4().hex[:12], "title": title.strip()[:80], "status": "running",
               "stage": "starting", "total": len(pages), "pages": [], "errors": [], "error": None}
        with _lock:
            if len(self.active) >= MAX_PENDING:
                return None
            if self.active:
                rec["stage"] = "Waiting for the story ahead to finish"
            self.active.add(rec["id"])
            self.save(rec)
        _pool.submit(self.run, rec, pages)
        return rec

    def _stage(self, rec: dict, text: str) -> None:
        with _lock:
            rec["stage"] = text
            self.save(rec)

    def run(self, rec: dict, pages: list[np.ndarray]) -> None:
        so_far: list[str] = []
        try:
            for i, img in enumerate(pages):
                try:
                    scene_id, title = self._page(rec, i, len(pages), img, so_far)
                except Exception as e:  # one bad page shouldn't throw away the pages already built
                    traceback.print_exc()
                    with _lock:
                        rec["errors"].append({"page": i + 1, "error": f"{type(e).__name__}: {e}"})
                        self.save(rec)
                    continue
                with _lock:
                    rec["pages"].append(scene_id)
                    if not rec["title"]:
                        rec["title"] = title
                    self.save(rec)
        finally:
            with _lock:
                try:
                    if rec["pages"]:
                        rec.update(status="done", stage="done")
                    else:
                        rec.update(status="failed", error=rec["errors"][0]["error"] if rec["errors"] else "no pages")
                    self.save(rec)
                finally:
                    self.active.discard(rec["id"])

    def _page(self, rec: dict, i: int, total: int, img: np.ndarray, so_far: list[str]) -> tuple[str, str]:
        """Build one page's scene. Returns (scene id, the page's title)."""
        def stage(text):
            self._stage(rec, f"Page {i + 1} of {total}: {text}")

        stage("reading the page")
        page = read_page(img)
        illus = page.illustration if page.illustration is not None else page.page
        labels = sorted({spoken(d["name"]) for d in detect(illus)})
        matte, full_bleed = paper_matte(illus)

        stage("writing the scene")
        story, source = adapt(page.text, labels, so_far, _jpg(_small(illus)))
        so_far.append(page.text)

        stage("building the 3D illustration")
        scene = build_25d(illus, matte=None if full_bleed else matte)
        hfov = scene.meta["camera"]["hfov_deg"]

        stage("painting the 360° world around it")
        masks = [cv.resize(m.astype(np.uint8), (illus.shape[1], illus.shape[0])) > 0 for m in scene.layer_masks]
        seed = 7 + (int(rec["id"], 16) + i * 101) % 4000  # per story and page, reproducible from the story id
        pano, world = paint_world(illus, matte, hfov, story, masks, progress=stage, seed=seed)
        world["paper_matte"] = {"full_bleed": full_bleed, "art_fraction": round(float(matte.mean() / 255), 3)}

        stage("giving the characters voices")
        files = {"scene.glb": scene.glb, "input.jpg": _jpg(illus), "page.jpg": _jpg(annotate(page)),
                 "panorama.jpg": _jpg(pano, 90)}
        files.update({f"{k}.jpg": _jpg(v) for k, v in scene.previews.items()})
        beats = []
        for j, b in enumerate(story["beats"]):
            voice = voice_for(b["speaker"], story["characters"])
            audio = synthesize(b["text"], voice)
            if audio:
                files[f"beat_{j:02d}.mp3"] = audio
            beats.append({**b, "voice": voice, "audio": bool(audio)})

        meta = {**scene.meta, "type": "story", "story": {
            "story_id": rec["id"], "page": i + 1, "pages": total, "text": page.text,
            "title": story["title"], "setting": story["setting"], "beats": beats,
            "characters": story["characters"], "source": source,
            "model": llm_label() if source == "bedrock" else None, "world": world,
            "ocr": {"lines": len(page.lines), "flattened": bool(page.steps), "timings_ms": page.timings_ms},
        }}
        return self.scenes.save(meta, files, story["title"], "story_page"), story["title"]
