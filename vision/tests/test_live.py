"""Live stories: the story agent's stream, OpenCV's check of it, the picture quota, the world, the reader."""
import time

import cv2 as cv
import numpy as np

from agents.imagery import Bucket
from agents.planner import STILL, ObjectStream, clean_action, clean_shot, fallback_plan
from server.live import _fix_words, _story_like, align, content_type
from vision.page import read_page
from vision.shot import text_cover, world_from_wide

POEM = ["Twinkle, twinkle, little star,", "How I wonder what you are.", "Up above the world so high,",
        "Like a diamond in the sky.", "When the blazing sun is gone,", "When he nothing shines upon,",
        "Then you show your little light,", "Twinkle, twinkle, all the night."]


def test_object_stream_hands_over_each_object_as_it_closes():
    raw = ('```json\n{"title": "Star", "cast": [{"name": "Kid", "voice": "child"}]}\n'
           '{"shot": 1, "text": ["Twinkle, {twinkle}"], "see": "a \\"bright\\" star"}],\n'
           '{"shot": 2, "text": ["Up above"], "effects": ["stars", "lasers"]}\n{"shot": 3, "text": [oops]}\n```')
    stream, got = ObjectStream(), []
    for i in range(0, len(raw), 5):  # tiny chunks, as a model streams them
        got += stream.feed(raw[i:i + 5])
    assert [list(o)[0] for o in got] == ["title", "shot", "shot"]  # the malformed third moment is skipped
    assert got[1]["text"] == ["Twinkle, {twinkle}"] and got[1]["see"] == 'a "bright" star'
    assert clean_shot(got[2]) == {"text": ["Up above"], "speaker": ["Narrator"], "see": "Up above", "effects": ["stars"],
                                  "action": STILL}


def test_opencv_check_finds_the_lines_the_story_agent_skipped():
    ocr = ["Bees", "Twinkle Twinkle"] + [ln.replace("little", "ittle") for ln in POEM]  # a logo, a title, slips
    plan = POEM[:4] + POEM[6:]  # the agent dropped two lines in the middle
    cover, missed, agreement = align(plan, ocr)
    assert {0, 6, 7} <= set(missed) and not {2, 3, 4, 5, 8, 9} & set(missed) and agreement > 0.6
    assert 2 in cover[0] and cover[4] == [8]  # every planned line knows its line on the page
    vocab = {w.strip(",.").lower() for ln in plan for w in ln.split()}
    assert _fix_words("Then you show your ittle iight,", vocab) == "Then you show your little light,"
    assert _story_like(ocr[6]) and not _story_like("Bees") and not _story_like("www.bees.com stories for kids")


def test_fallback_plan_uses_the_page_lines_in_at_most_six_moments():
    header, shots = fallback_plan(["12", "~~", *POEM, *POEM])
    assert len(shots) <= 6 and sum(len(s["text"]) for s in shots) == 16
    assert shots[0]["text"][0] == POEM[0] and header["title"] == "Twinkle, twinkle, little star"


def test_quota_bucket_spreads_calls_over_the_minute():
    b = Bucket(per_minute=2)
    assert b.take() == 0 and b.take() == 0
    wait = b.take()
    assert 25 < wait <= 30  # the third call waits for the bucket to refill (2 per minute)
    b.drain()
    assert b.take() > 25


def test_world_wraps_around_seamlessly_with_the_painting_centred():
    img = np.zeros((960, 2304, 3), np.uint8)
    img[:, :, 1] = np.linspace(0, 255, 2304, dtype=np.uint8)[None, :]  # a left-to-right ramp: worst case for a seam
    img[400:560, 1100:1204] = (0, 0, 255)  # a red mark at the centre
    world = world_from_wide(img)
    assert world.shape == (1536, 3072, 3)
    assert np.abs(world[:, 0].astype(int) - world[:, -1].astype(int)).mean() < 6  # no seam where the ends meet
    mid = world[world.shape[0] // 2, world.shape[1] // 2 - 20: world.shape[1] // 2 + 20]
    assert (mid[:, 2] > 200).mean() > 0.8  # the painting's centre faces straight ahead


def test_lettering_check_tells_text_from_scenery():
    scene = np.full((600, 1400, 3), (180, 140, 90), np.uint8)
    cv.circle(scene, (700, 300), 120, (60, 160, 220), -1)
    cv.GaussianBlur(scene, (0, 0), 3, dst=scene)
    sign = scene.copy()
    cv.putText(sign, "WELCOME TO TOWN", (300, 520), cv.FONT_HERSHEY_DUPLEX, 2.2, (20, 20, 20), 4, cv.LINE_AA)
    assert text_cover(scene) < 0.002 < 0.004 < text_cover(sign)


def test_reader_handles_light_text_on_a_dark_poster():
    page = np.full((380, 720, 3), (150, 60, 20), np.uint8)  # deep blue
    for i, line in enumerate(["Twinkle twinkle little star", "How I wonder what you are"]):
        cv.putText(page, line, (30, 120 + i * 110), cv.FONT_HERSHEY_DUPLEX, 1.25, (255, 255, 255), 2, cv.LINE_AA)
    text = read_page(cv.resize(page, (360, 190)), flatten=False).text.lower()  # small, like a web poster
    assert "twinkle" in text and "wonder" in text and "star" in text


def test_only_story_files_are_served():
    assert content_type("shot_03.jpg") == "image/jpeg" and content_type("line_012.mp3") == "audio/mpeg"
    assert content_type("shot_03_depth.png") == "image/png" and content_type("world.jpg") == "image/jpeg"
    assert content_type("shot_03.mp4") == "video/mp4" and content_type("shot_03_actor.png") == "image/png"
    assert content_type("shot_03_plate.jpg") == "image/jpeg" and content_type("shot_03_world_full_motion.png")
    assert content_type("shot_03_try2.mp4") == "video/mp4"
    for bad in ("state.json", "../page.jpg", "shot_3.jpg", "line_12.mp3", "x.html", "shot_3.mp4", "clip.mp4"):
        assert content_type(bad) is None, bad


def test_full_world_agent_paints_outward_from_the_first_moment(monkeypatch, tmp_path):
    """The world agent's second pass, with a fake painter: the 360 world is painted around the first moment, given
    depth and motion maps by OpenCV, and replaces the quick world in the story's state."""
    from agents import painter
    from server.live import Live
    from server.storage import LocalStore

    def fake_paint(canvas, generate, prompt, negative, seed=7):
        out = canvas.copy()
        blotches = np.random.default_rng(seed).integers(40, 200, (canvas.shape[0] // 16 + 1, canvas.shape[1] // 16 + 1, 3))
        out[generate > 0] = np.repeat(np.repeat(blotches, 16, 0), 16, 1)[:canvas.shape[0], :canvas.shape[1]][generate > 0]
        return out

    monkeypatch.setattr(painter, "enabled", lambda: True)
    monkeypatch.setattr(painter, "paint", fake_paint)
    live = Live(LocalStore(tmp_path))
    st = _state(place="a moonlit village", style="watercolour", world={"image": "world.jpg", "ready": True})
    moment = np.zeros((960, 2304, 3), np.uint8)
    moment[:480] = (120, 70, 30)
    moment[480:] = (40, 120, 60)
    live._world_full(st, moment, seed=3)
    w = st["world"]
    assert w["image"] == "world_full.jpg" and w["quality"] == "full" and not w["upgrading"]
    assert w["depth"] == "world_full_depth.png" and w["motion"] == "world_full_motion.png"
    pano = cv.imread(str(tmp_path / "live" / "abcdef123456" / "world_full.jpg"))
    assert pano.shape == (2048, 4096, 3)
    centre = pano[1024, 2048].astype(int)
    assert abs(centre - (40, 120, 60)).max() < 40 or abs(centre - (120, 70, 30)).max() < 40  # the moment, kept
    assert st["timings"]["world_full"] is not None


def test_story_agent_failing_to_connect_falls_back_cleanly(monkeypatch):
    """An expired AWS login must mean 'plan from OpenCV's reading', not a crashed story."""
    import pytest
    from botocore.exceptions import BotoCoreError
    from agents import planner
    from agents.llm import LLMUnavailable

    class Down:
        def converse_stream(self, **kw):
            raise BotoCoreError()

    monkeypatch.setenv("STEP_INSIDE_LLM", "bedrock")
    monkeypatch.setattr(planner, "_runtime", lambda: Down())
    with pytest.raises(LLMUnavailable):
        planner.plan(b"jpg", lambda h: None, lambda s, extend=False: None)


TIMINGS = ("read", "header", "planned_first", "plan", "world", "first_shot", "playable", "shots", "voices",
           "world_full", "first_clip", "clips", "film", "all", "upgrades")


def _state(**kw) -> dict:
    """A story's state as the conductor keeps it, for driving one agent at a time."""
    st = {"id": "abcdef123456", "status": "ready", "error": None, "created": time.time(), "stage": "", "title": "T",
          "style": "watercolour", "place": "a field", "cast": [], "time": "day", "weather": "clear", "wind": "breeze",
          "page": None, "lines": [], "shots": [], "upgrading": False,
          "world": {"image": None, "ready": True, "source": None}, "check": None, "models": {},
          "timings": dict.fromkeys(TIMINGS), "_t0": time.monotonic(), "_plan_done": True, "_slow": [], "_tasks": []}
    st.update(kw)
    return st


def test_story_agent_actions_fold_into_seven_paths():
    assert clean_action({"path": "tumbles", "motion": "the egg tumbles off the wall"})["path"] == "fall"
    assert clean_action({"path": "flying", "motion": "the owl flies up"})["path"] == "rise"
    assert clean_action({"path": "walking left", "motion": "Jack walks away"})["path"] == "walk-left"
    assert clean_action({"path": "dancing", "motion": "the cat dances"}) == {
        "subject": "", "motion": "the cat dances", "path": "sway", "loop": True}
    assert clean_action({"path": "fall"})["path"] == "still"  # a movement with no description is not trusted
    assert clean_action({"path": "fall", "motion": "falls", "loop": True})["loop"] is False  # a fall ends elsewhere
    assert clean_action("nonsense") == STILL and all(s["action"] == STILL for s in fallback_plan(POEM)[1])


def _falling_clip(tmp_path, still, name="clip.mp4", move=True):
    h, w = still.shape[:2]
    path = tmp_path / name
    out = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
    if not out.isOpened():
        import pytest
        pytest.skip("this OpenCV build cannot write mp4v")
    for i in range(96):
        f = still.copy()
        y = 40 + (int(120 * min(1, i / 70) ** 2) if move else 0)
        cv.rectangle(f, (300, y), (350, y + 60), (20, 20, 30), -1)
        out.write(f)
    out.release()
    return path.read_bytes()


def _textured(w=640, h=274):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = np.linspace(80, 200, w, dtype=np.uint8)[None, :, None]
    rng = np.random.default_rng(1)
    for _ in range(60):
        cv.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(3, 12)),
                  tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return img


def _clip_story(tmp_path, monkeypatch, clips):
    """A story with one falling moment and one still moment, and a fake animator handing out `clips` in turn."""
    from concurrent.futures import Future
    from agents import animator
    from server.live import Live
    from server.storage import LocalStore
    still = _textured()
    cv.rectangle(still, (300, 40), (350, 100), (20, 20, 30), -1)
    live = Live(LocalStore(tmp_path / "store"))
    calls = []

    def fake_submit(priority, still_jpg, prompt, loop, expires_at, on_status=None, seed=0):
        calls.append({"priority": priority, "prompt": prompt, "loop": loop})
        if on_status:
            on_status("generating")
        f = Future()
        f.set_result(animator.Clip(clips[min(len(calls), len(clips)) - 1], "arn:x/abc", 1.0, "540p", prompt, True))
        return f

    monkeypatch.setattr(animator, "enabled", lambda: True)
    monkeypatch.setattr(animator, "submit", fake_submit)
    shot = lambda i, path: {"i": i, "lines": [], "effects": [], "prompt": "An egg on a wall.", "caption": "",
                            "image": f"shot_{i:02d}.jpg", "ready": True, "source": "stable-image-core",
                            "action": {"subject": "Egg", "motion": "the egg falls off the wall", "path": path,
                                       "loop": False}, "actor_box": None, "clip": None, "world": None}
    st = _state(shots=[shot(0, "fall"), shot(1, "still")])
    live._put(st, "shot_00.jpg", cv.imencode(".jpg", still, [cv.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
    st["shots"][0]["clip"] = live._new_clip("shot_00.jpg")  # filmed (as `_shot` does once a moment is painted)
    live._clip_job(st, st["shots"][0], 0, 1)
    live._plan_clips(st)  # the unfilmed moment says why
    return live, st, calls


def test_conductor_animates_the_action_moment_and_keeps_the_verdict(tmp_path, monkeypatch):
    base = _textured()
    live, st, calls = _clip_story(tmp_path, monkeypatch, [_falling_clip(tmp_path, base)])
    c0, c1 = st["shots"][0]["clip"], st["shots"][1]["clip"]
    assert c0["status"] == "ready" and c0["verdict"]["observed"] == "fall" and c0["hold"], c0
    assert c1 == {"status": "off", "reason": "page limit"} and len(calls) == 1
    assert "locked-off camera" in calls[0]["prompt"] and calls[0]["loop"] is False
    assert live.file("abcdef123456", "shot_00.mp4") and live.file("abcdef123456", "shot_00_try1.mp4")
    assert st["timings"]["clips"] is not None and not live._snapshot(st)["upgrading"]
    assert [s["name"] for s in live._snapshot(st)["steps"]][-1] == "clips"


def test_a_clip_where_nothing_happens_is_retried_then_rejected(tmp_path, monkeypatch):
    base = _textured()
    still_clip = _falling_clip(tmp_path, base, "s.mp4", move=False)
    _, st, calls = _clip_story(tmp_path, monkeypatch, [still_clip, still_clip])
    c0 = st["shots"][0]["clip"]
    assert len(calls) == 2 and "large and clear" in calls[1]["prompt"] and "no cut" in calls[0]["prompt"]  # the second try asks for a bigger movement
    assert c0["status"] == "rejected" and c0["reason"] == "nothing happens" and len(c0["tries"]) == 2


def test_a_story_left_upgrading_by_a_restart_is_finalized(tmp_path):
    import json
    from server.live import Live
    from server.storage import LocalStore
    live = Live(LocalStore(tmp_path))
    rec = {"id": "abcdefabcdef", "status": "ready", "upgrading": True, "world": {"upgrading": True},
           "shots": [{"world": {"upgrading": True}, "clip": {"status": "generating"}}]}
    live.store.put("live/abcdefabcdef/state.json", json.dumps(rec).encode(), "application/json")
    got = live.get("abcdefabcdef")
    assert not got["upgrading"] and not got["shots"][0]["world"]["upgrading"]
    assert got["shots"][0]["clip"] == {"status": "off", "reason": "the server restarted"}


def test_video_budget_is_separate_from_images(monkeypatch):
    from agents import budget
    monkeypatch.setitem(budget.LIMITS, "video", 1)
    monkeypatch.setitem(budget._state, "day", "")
    assert budget.take("video") and not budget.take("video")
    assert budget.take("image") and budget.left("video") == 0


def test_animator_runs_one_clip_at_a_time_in_priority_order(monkeypatch):
    import threading
    from agents import animator
    running, seen, lock = [0], [], threading.Lock()
    peak = [0]

    def fake_generate(still, prompt, loop, deadline, on_status=None, seed=0):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.15)
        with lock:
            running[0] -= 1
            seen.append(prompt)
        return animator.Clip(b"", "arn", 0.15, "540p", prompt, True)

    monkeypatch.setattr(animator, "_generate", fake_generate)
    monkeypatch.setitem(animator._state, "disabled", None)
    monkeypatch.setenv("VIDEO_MODEL", "luma.ray-v2:0")  # Luma: the account allows one job at a time
    monkeypatch.delenv("VIDEO_CONCURRENCY", raising=False)
    far = time.monotonic() + 60
    first = animator.submit((0, 0, 1), b"", "first", False, far)
    time.sleep(0.05)  # the worker is busy with the first clip while the others queue
    later = [animator.submit((5, 0, 1), b"", "older story last", False, far),
             animator.submit((1, 0, 1), b"", "second", False, far)]
    for f in [first, *later]:
        f.result(timeout=10)
    assert peak[0] == 1 and seen == ["first", "second", "older story last"]


def test_each_moment_gets_its_own_world_with_life(tmp_path):
    from server.live import Live
    from server.storage import LocalStore
    live = Live(LocalStore(tmp_path))
    world = np.zeros((1536, 3072, 3), np.uint8)
    world[:700] = (200, 160, 110)
    world[700:] = (60, 130, 80)
    moment = np.zeros((960, 2304, 3), np.uint8)
    moment[:] = (30, 30, 200)
    shot = {"i": 0, "image": "shot_00.jpg", "source": "opencv", "action": STILL, "ready": True, "lines": []}
    st = _state(shots=[shot], _world_pano=world)
    live._moment_world(st, shot, moment, seed=1)
    w = shot["world"]
    assert w["image"] == "shot_00_world.jpg" and w["quality"] == "quick" and not w["upgrading"]
    assert set(w["life"]) == {"wind", "light", "lamps", "haze", "time", "weather"}
    pano = cv.imread(str(tmp_path / "live" / "abcdef123456" / "shot_00_world.jpg"))
    assert abs(pano[768, 1536].astype(int) - (30, 30, 200)).max() < 30  # the moment faces the viewer
    assert abs(pano[768, 100].astype(int) - (60, 130, 80)).max() < 40   # the story's world all around


def test_story_agent_moments_nested_inside_the_header_are_read(monkeypatch):
    """What Nova Lite really sent for the Timmy and Bella page in 3 of 4 calls (Oct 26): every moment inside the
    header under "scenes". It must not fall back to OpenCV's raw reading."""
    from agents import planner
    raw = ('{"title": "Timmy\'s Rescue", "style": "watercolour", "place": "a forest clearing", "cast": '
           '[{"name": "Timmy", "look": "a small caterpillar", "voice": "child"}, {"name": "Bella", "look": "a blue '
           'butterfly", "voice": "woman"}], "time": "day", "scenes": [{"shot": 1, "text": ["Help! I can\'t get out!", '
           '"Timmy shouted."], "speaker": ["Narrator", "Timmy"], "see": "Timmy stuck under a leaf", "effects": [], '
           '"action": {"subject": "Timmy", "motion": "struggles", "path": "still", "loop": false}}, {"shot": 2, '
           '"text": ["Bella flew as fast as she could to Timmy."], "speaker": ["Narrator"], "see": "Bella flying", '
           '"effects": [], "action": {"subject": "Bella", "motion": "flies to Timmy", "path": "fly", "loop": false}}]}')

    class Stream:
        def converse_stream(self, **kw):
            return {"stream": [{"contentBlockDelta": {"delta": {"text": raw[i:i + 40]}}} for i in range(0, len(raw), 40)]}

    monkeypatch.setenv("STEP_INSIDE_LLM", "bedrock")
    monkeypatch.setattr(planner, "_runtime", lambda: Stream())
    headers, shots = [], []
    planner.plan(b"jpg", headers.append, lambda s, extend=False: shots.append(s), 3)
    assert headers[0]["title"] == "Timmy's Rescue" and len(shots) == 2
    assert shots[0]["text"] == ["Help! I can't get out!", "Timmy shouted."]
    assert shots[1]["action"]["path"] == "rise" and shots[1]["action"]["subject"] == "Bella"


def test_speech_without_quote_marks_keeps_the_character_but_speech_tags_stay_narration():
    from agents.planner import Dialogue
    d = Dialogue([{"name": "Timmy"}, {"name": "Bella"}])
    shot = {"text": ["Help! I can't get out!", "Timmy shouted.", "You're free now, Timmy!", "she said with a big smile."],
            "speaker": ["Timmy", "Timmy", "Bella", "Bella"], "see": "", "effects": []}
    assert d.apply(shot)["speaker"] == ["Timmy", "Narrator", "Bella", "Narrator"]


def test_a_quote_naming_someone_is_not_given_to_them():
    from agents.planner import Dialogue
    d = Dialogue([{"name": "Timmy"}, {"name": "Bella"}])
    d.apply({"text": ["“Help!” Timmy shouted."], "speaker": ["Narrator"], "see": "", "effects": []})
    out = d.apply({"text": ["“You're free now, Timmy!”", "she said with a big smile."],
                   "speaker": ["Narrator", "Narrator"], "see": "", "effects": []})
    assert out["speaker"] == ["Bella", "Narrator"]


def test_a_clip_that_never_arrives_is_given_up_at_the_deadline(tmp_path, monkeypatch):
    from concurrent.futures import Future
    from agents import animator
    import server.live as live_mod
    from server.storage import LocalStore
    monkeypatch.setattr(live_mod, "CLIP_DEADLINE_S", 61)  # the deadline is 1 s away
    monkeypatch.setattr(live_mod, "CLIP_SLACK_S", 0.5)
    pending = []
    monkeypatch.setattr(animator, "submit", lambda *a, **k: pending.append(Future()) or pending[-1])
    live = live_mod.Live(LocalStore(tmp_path))
    shot = {"i": 0, "image": "shot_00.jpg", "prompt": "A scene.", "ready": True, "source": "stable-image-core",
            "lines": [], "aspect": 1.78, "actor_box": None,
            "action": {"subject": "Egg", "motion": "falls", "path": "fall", "loop": False}}
    st = _state(shots=[shot])
    live._put(st, "shot_00.jpg", cv.imencode(".jpg", np.zeros((756, 1344, 3), np.uint8))[1].tobytes())
    shot["clip"] = live._new_clip("shot_00.jpg")
    t0 = time.monotonic()
    live._clip(st, shot, 0)
    assert shot["clip"] == {**shot["clip"], "status": "off", "reason": "out of time"}
    assert pending[0].cancelled() and time.monotonic() - t0 < 10


def test_the_world_agent_repaints_a_later_moments_own_world(monkeypatch, tmp_path):
    from agents import painter
    from server.live import Live
    from server.storage import LocalStore

    def fake_paint(canvas, generate, prompt, negative, seed=7):
        out = canvas.copy()
        out[generate > 0] = (90, 160, 90)
        return out

    monkeypatch.setattr(painter, "enabled", lambda: True)
    monkeypatch.setattr(painter, "paint", fake_paint)
    live = Live(LocalStore(tmp_path))
    shot = {"i": 1, "image": "shot_01.jpg", "ready": True, "lines": [], "source": "stable-image-core",
            "world": {"image": "shot_01_world.jpg", "quality": "quick", "upgrading": True}}
    st = _state(place="a meadow", style="watercolour", world={"image": "world.jpg", "ready": True},
                shots=[{"i": 0, "ready": True, "lines": []}, shot])
    moment = np.zeros((756, 1344, 3), np.uint8)
    moment[:] = (40, 60, 200)
    live._world_full(st, moment, 3, shot)
    w = shot["world"]
    assert w["image"] == "shot_01_world_full.jpg" and w["quality"] == "full" and not w["upgrading"]
    assert w["depth"] == "shot_01_world_full_depth.png" and w["life"] is not None
    assert live.file(st["id"], "shot_01_world_full.jpg")
    assert st["world"]["image"] == "world.jpg" and st["timings"]["world_full"] is None  # the first world is untouched


def test_the_story_is_ready_to_enter_only_when_completely_made(tmp_path):
    """play_when "all": every moment painted and voiced, and every clip made or given up (a failed clip still lets
    the story start, on its painting)."""
    from server.live import Live
    from server.storage import LocalStore
    live = Live(LocalStore(tmp_path))
    line = lambda i: {"i": i, "ready": True}
    shot = lambda i, status: {"i": i, "ready": True, "lines": [i], "source": "stable-image-core",
                              "clip": {"status": status}}
    st = _state(play_when="all", lines=[line(0), line(1), line(2)],
                shots=[shot(0, "ready"), shot(1, "generating"), shot(2, "queued")])
    with live.lock:
        live._check_progress(st)
    assert st["timings"]["film"] is None
    st["shots"][1]["clip"]["status"] = "rejected"  # a failed clip: that moment will play as its painting
    st["shots"][2]["clip"]["status"] = "ready"
    with live.lock:
        live._check_progress(st)
    assert st["timings"]["film"] is not None


HANSEL_OCR = ["Hanse and G retel", "o Written by Tasha Guenther and lustrated by Leanne and Kaitlyn Guenther",
              "Hanse and Grete based on the German tairy tale first collected by the Brothers Grimm",
              "Once upon a time there ived a woodcutter and his wife Their",
              "early days were lived in bliss in a small wood cottage with their",
              "two children Hansel and Gretel."]


def _stream(chunks, stop="end_turn"):
    class S:
        def converse_stream(self, **kw):
            return {"stream": [{"contentBlockDelta": {"delta": {"text": c}}} for c in chunks]
                    + [{"messageStop": {"stopReason": stop}}]}
    return S()


def test_a_reply_cut_by_the_content_filter_is_detected(monkeypatch):
    """Hansel and Gretel (Oct 26): Nova Lite's filter cut its reply mid-header on a published page."""
    import pytest
    from agents import planner
    raw = ('{"title": "Hansel and Gretel", "style": "watercolour", "place": "a cottage", "cast": [{"name": "Hansel", '
           '"look": "a boy", "voice": "child"}], "scenes": [{"shot": 1, "text": ["Once upon a time there lived a '
           'woodcutter and his wife.')
    monkeypatch.setenv("STEP_INSIDE_LLM", "bedrock")
    monkeypatch.setattr(planner, "_runtime", lambda: _stream([raw, " The generated text has been blocked by our "
                                                                    "content filters."], "content_filtered"))
    headers = []
    with pytest.raises(planner.ContentFiltered) as err:
        planner.plan(b"jpg", headers.append, lambda s, extend=False: None, 6)
    assert err.value.header_sent and headers[0]["title"] == "Hansel and Gretel"  # the header was rescued


def test_moments_that_arrived_before_the_cut_are_kept(monkeypatch):
    from agents import planner
    raw = ('{"title": "Hansel and Gretel", "cast": [], "scenes": [{"shot": 1, "text": ["Once upon a time there '
           'lived a woodcutter."], "speaker": ["Narrator"], "see": "a cottage", "effects": [], "action": {"subject": '
           '"woodcutter", "motion": "chops wood", "path": "sway", "loop": true}}, {"shot": 2, "text": ["These child')
    monkeypatch.setenv("STEP_INSIDE_LLM", "bedrock")
    monkeypatch.setattr(planner, "_runtime", lambda: _stream([raw], "content_filtered"))
    shots = []
    planner.plan(b"jpg", lambda h: None, lambda s, extend=False: shots.append(s), 6)
    assert len(shots) == 1 and shots[0]["text"] == ["Once upon a time there lived a woodcutter."]


def test_lines_mode_never_copies_the_page(monkeypatch):
    from agents import planner
    reply = ('{"title": "Hansel and Gretel", "style": "watercolour", "place": "a wood cottage by a lake", "cast": '
             '[{"name": "Hansel", "look": "a boy in a brown tunic", "voice": "child"}], "skip": [0, 1, 2], '
             '"fixes": {"ived": "lived", "Hanse": "Hansel", "wife Their": "wife. Their"}}\n'
             '{"shot": 1, "lines": [3, 4, 5], "speaker": ["Narrator", "Narrator", "Narrator"], "see": "a woodcutter '
             'and his wife outside their cottage", "effects": [], "action": {"subject": "woodcutter", "motion": '
             '"waves to the children", "path": "sway", "loop": true}}')
    sent = {}

    class C:
        def converse(self, **kw):
            sent.update(kw)
            return {"stopReason": "end_turn", "output": {"message": {"content": [{"text": reply}]}}}

    monkeypatch.setenv("STEP_INSIDE_LLM", "bedrock")
    monkeypatch.setattr(planner, "_runtime", lambda: C())
    headers, shots = [], []
    planner.plan_from_lines(b"jpg", HANSEL_OCR, headers.append, lambda s, extend=False: shots.append(s))
    assert "0: Hanse and G retel" in sent["messages"][0]["content"][1]["text"]  # it is shown OpenCV's lines
    text = shots[0]["text"]
    assert text == ["Once upon a time there lived a woodcutter and his wife.",
                    "Their early days were lived in bliss in a small wood cottage with their two children Hansel and Gretel."]
    assert all("Written by" not in t for t in text) and headers[0]["cast"][0]["name"] == "Hansel"
    assert shots[0]["action"]["path"] == "sway"


def test_the_raw_fallback_leaves_out_titles_and_credits():
    header, shots = fallback_plan(HANSEL_OCR + ["Hansel and Gretel"])
    lines = [t for s in shots for t in s["text"]]
    assert not any("Written by" in t or "based on" in t for t in lines)


def test_nothing_described_is_not_filmed_and_a_frozen_clip_is_rejected(monkeypatch, tmp_path):
    from agents import animator, imagery, painter
    from server.live import Live
    from server.storage import LocalStore
    from vision import clip as critic
    painting = np.full((756, 1344, 3), 140, np.uint8)
    monkeypatch.setattr(imagery, "generate", lambda *a, **k: (painting.copy(), "stable-image-core"))
    monkeypatch.setattr(painter, "enabled", lambda: False)
    monkeypatch.setattr(animator, "enabled", lambda: True)
    submitted = []
    monkeypatch.setattr(Live, "_clip_job", lambda self, st, shot, k, seed: submitted.append(k))
    live = Live(LocalStore(tmp_path))
    shot = lambda i, motion: {"i": i, "lines": [], "effects": [], "prompt": "A scene.", "caption": "", "image": None,
                              "ready": False, "source": None, "actor_box": None, "clip": None, "world": None,
                              "action": {"subject": "", "motion": motion, "path": "still", "loop": True}}
    st = _state(shots=[shot(0, ""), shot(1, "looks surprised")], _world_pano=np.full((1536, 3072, 3), 120, np.uint8))
    for s in st["shots"]:
        live._shot(st, s, painting, seed=1)
    for f in list(st["_slow"]):
        f.result(timeout=30)
    assert st["shots"][0]["clip"] == {"status": "off", "reason": "nothing to film: a living painting"}
    assert st["shots"][1]["clip"]["status"] == "queued" and submitted == [1]  # "looks surprised" is worth a clip
    t = {"subject_speed": 0.0, "background_speed": 0.0, "dx": 0, "dy": 0, "peak_lift": 0, "reversals": 0}
    assert critic.classify("still", t) == ("nothing", True, ["nothing happens"])


def test_a_page_picture_with_printing_in_it_is_not_used(monkeypatch, tmp_path):
    from agents import painter
    from server.live import Live
    from server.storage import LocalStore
    pic = np.full((600, 800, 3), 240, np.uint8)
    for i, line in enumerate(["Hansel and Gretel spent most", "of their time finding stones", "Their collection"]):
        cv.putText(pic, line, (30, 120 + i * 80), cv.FONT_HERSHEY_DUPLEX, 1.4, (30, 30, 30), 3, cv.LINE_AA)
    monkeypatch.setattr(painter, "enabled", lambda: True)
    monkeypatch.setattr(painter, "paint", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not paint")))
    st = {"_page_picture": pic}
    assert Live(LocalStore(tmp_path))._page_picture(st, "a cottage", 1) == (None, None)
    assert st["_page_picture"] is None


def test_a_blocked_story_agent_hands_over_to_lines_mode(monkeypatch, tmp_path):
    from agents import planner
    from server.live import Live
    from server.storage import LocalStore
    monkeypatch.setenv("STEP_INSIDE_LLM", "off")  # no real model, painter or voice in this test
    calls = []

    def blocked(page_jpg, on_header, on_shot, line_count=0):
        raise planner.ContentFiltered("blocked", header_sent=False)

    def lines_mode(page_jpg, ocr_lines, on_header, on_shot, send_header=True):
        calls.append(ocr_lines)
        on_header({**planner.clean_header({}), "title": "The Rabbit"})
        on_shot({"text": ["The little rabbit sat under the big tree."], "speaker": ["Narrator"], "see": "a rabbit",
                 "effects": [], "action": {"subject": "rabbit", "motion": "nibbles grass", "path": "sway", "loop": True}})

    monkeypatch.setattr(planner, "plan", blocked)
    monkeypatch.setattr(planner, "plan_from_lines", lines_mode)
    page = np.full((900, 700, 3), 240, np.uint8)
    for i, line in enumerate(["The little rabbit sat", "under the big tree."]):
        cv.putText(page, line, (60, 400 + i * 90), cv.FONT_HERSHEY_DUPLEX, 1.6, (30, 30, 30), 3, cv.LINE_AA)
    live = Live(LocalStore(tmp_path))
    st = live.submit(page)
    for _ in range(120):
        st = live.get(st["id"])
        if st["status"] != "running":
            break
        time.sleep(0.25)
    assert calls and "rabbit" in " ".join(calls[0]).lower()  # lines mode got OpenCV's reading
    assert st["title"] == "The Rabbit" and "lines mode" in st["models"]["planner"]
    assert st["shots"][0]["action"]["path"] == "sway"
