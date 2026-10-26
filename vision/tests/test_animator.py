"""The animator on Amazon Nova Reel: the request it sends, the start frame, the parallel queue, and the conductor
filming every painted moment as soon as it is painted."""
import base64
import threading
import time
from concurrent.futures import Future

import cv2 as cv
import numpy as np
import pytest

from agents import animator


@pytest.fixture
def nova(monkeypatch):
    monkeypatch.setenv("VIDEO_MODEL", "amazon.nova-reel-v1:0")
    monkeypatch.setenv("VIDEO_BUCKET", "test-bucket")
    monkeypatch.delenv("VIDEO_REGION", raising=False)
    monkeypatch.delenv("VIDEO_CONCURRENCY", raising=False)
    monkeypatch.setitem(animator._state, "disabled", None)


def _jpg(img):
    return cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def test_nova_reel_request(nova, monkeypatch):
    captured = {}

    class Client:
        def start_async_invoke(self, **kw):
            captured.update(kw)
            return {"invocationArn": "arn:aws:bedrock:us-east-1:1:async-invoke/abc123"}

    monkeypatch.setattr(animator, "_runtime", lambda reg: captured.setdefault("region", reg) and Client())
    painting = np.full((768, 1365, 3), (90, 140, 200), np.uint8)  # ~16:9, as Stable Image Core paints it
    action = {"subject": "Humpty Dumpty", "motion": "wobbles on the wall and falls to the ground", "path": "fall",
              "loop": False}
    prompt = animator.clip_prompt(action, "a large egg in a purple coat", "soft watercolour picture-book",
                                  "Humpty Dumpty sits on a high stone wall above the meadow. Morning light.")
    arn = animator._start(_jpg(painting), prompt, False, seed=7)
    body = captured["modelInput"]
    assert arn.endswith("abc123") and captured["region"] == "us-east-1"
    assert captured["modelId"] == "amazon.nova-reel-v1:0" and captured["clientRequestToken"]
    assert captured["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"] == "s3://test-bucket/clips"
    assert body["taskType"] == "TEXT_VIDEO"
    cfg = body["videoGenerationConfig"]
    assert cfg == {"durationSeconds": 6, "fps": 24, "dimension": "1280x720", "seed": 7}
    image = body["textToVideoParams"]["images"][0]
    assert image["format"] == "jpeg"
    frame = cv.imdecode(np.frombuffer(base64.b64decode(image["source"]["bytes"]), np.uint8), cv.IMREAD_COLOR)
    assert frame.shape == (720, 1280, 3)
    text = body["textToVideoParams"]["text"]
    assert len(text) <= 512 and "Humpty Dumpty" in text and "falls to the ground" in text
    assert "locked-off camera: no pan, no zoom, no cut" in text


def test_prompts_stay_inside_nova_limit_and_still_moments_get_living_motion(nova):
    long = {"subject": "the old woman who lived in a shoe", "motion": "chases " + "her many children " * 30,
            "path": "walk-right", "loop": False}
    p = animator.clip_prompt(long, "a tall thin woman " * 20, "watercolour " * 20, "A huge shoe house. " * 20, 2)
    assert len(p) <= 512 and p.startswith("the old woman")
    still = animator.clip_prompt({"subject": "Little Bo Peep", "motion": "", "path": "still", "loop": True},
                                 None, "ink and wash", "Bo Peep stands in a green field with her crook.")
    assert "Gentle living motion" in still and "Little Bo Peep" in still and len(still) <= 512


def test_start_frame_is_exactly_1280_by_720_and_never_stretched():
    wide = np.zeros((960, 2304, 3), np.uint8)  # 21:9
    wide[:, 1100:1200] = (0, 0, 255)  # a red bar in the middle
    f = animator.start_frame(wide)
    assert f.shape == (720, 1280, 3)
    mid = f[360, 600:680]
    assert (mid[:, 2] > 200).any()  # the picture is fitted in, centred
    sixteen = np.full((768, 1366, 3), 50, np.uint8)
    assert animator.start_frame(sixteen).shape == (720, 1280, 3)


def test_nova_downloads_output_mp4_from_the_invocation_folder(nova, monkeypatch):
    got = {}

    class S3:
        def get_object(self, Bucket, Key):
            got["key"] = (Bucket, Key)
            return {"Body": type("B", (), {"read": lambda self: b"mp4"})()}

    monkeypatch.setattr(animator, "_s3", lambda reg: S3())
    arn = "arn:aws:bedrock:us-east-1:1:async-invoke/inv42"
    rec = {"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://test-bucket/clips/inv42"}}}
    assert animator._download(arn, rec) == b"mp4" and got["key"] == ("test-bucket", "clips/inv42/output.mp4")
    rec2 = {"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://test-bucket/clips"}}}
    animator._download(arn, rec2)
    assert got["key"] == ("test-bucket", "clips/inv42/output.mp4")


def test_parallel_workers_never_exceed_the_concurrency(nova, monkeypatch):
    monkeypatch.setenv("VIDEO_CONCURRENCY", "3")
    lock, running, peak = threading.Lock(), [0], [0]

    def fake_generate(still, prompt, loop, deadline, on_status=None, seed=0):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.2)
        with lock:
            running[0] -= 1
        return animator.Clip(b"", "arn", 0.2, "720p", prompt, True)

    monkeypatch.setattr(animator, "_generate", fake_generate)
    far = time.monotonic() + 60
    futs = [animator.submit((0, i, 1), b"", f"clip {i}", False, far) for i in range(7)]
    t0 = time.monotonic()
    for f in futs:
        f.result(timeout=20)
    assert peak[0] == 3  # in parallel, up to the limit and no further
    assert time.monotonic() - t0 < 7 * 0.2  # faster than one at a time


def _painting(w=1344, h=756):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = np.linspace(80, 200, w, dtype=np.uint8)[None, :, None]
    rng = np.random.default_rng(3)
    for _ in range(90):
        cv.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(4, 16)),
                  tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return img


def _moving_clip(path, start_jpg):
    frame = cv.imdecode(np.frombuffer(start_jpg, np.uint8), cv.IMREAD_COLOR)
    out = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), 24, (1280, 720))
    if not out.isOpened():
        pytest.skip("this OpenCV build cannot write mp4v")
    for i in range(144):  # 6 s
        f = frame.copy()
        y = 120 + int(260 * min(1, i / 100) ** 2)
        cv.rectangle(f, (600, y), (690, y + 110), (20, 20, 30), -1)
        out.write(f)
    out.release()
    return path.read_bytes()


def test_every_painted_moment_is_filmed_as_soon_as_it_is_painted(nova, monkeypatch, tmp_path):
    from agents import imagery, painter
    from server.live import Live
    from server.storage import LocalStore
    from vision.tests.test_live import _state

    painting = _painting()
    monkeypatch.setattr(imagery, "generate", lambda prompt, negative, aspect="16:9", seed=0, wait_s=30.0:
                        (painting.copy(), "stable-image-core"))
    monkeypatch.setattr(painter, "enabled", lambda: False)  # no full-world repaint in this test
    calls, clip_dir = [], tmp_path / "clips"
    clip_dir.mkdir()

    def fake_submit(priority, still_jpg, prompt, loop, expires_at, on_status=None, seed=0):
        calls.append({"priority": priority, "prompt": prompt, "t": time.monotonic(), "size":
                      cv.imdecode(np.frombuffer(still_jpg, np.uint8), cv.IMREAD_COLOR).shape})
        if on_status:
            on_status("generating")
        f = Future()
        f.set_result(animator.Clip(_moving_clip(clip_dir / f"{len(calls)}.mp4", still_jpg), "arn:x/i", 1.0, "720p",
                                   prompt, True))
        return f

    monkeypatch.setattr(animator, "submit", fake_submit)
    live = Live(LocalStore(tmp_path / "store"))
    paths = ["fall", "still", "walk-right", "still"]
    shot = lambda i, path: {"i": i, "lines": [], "effects": [], "prompt": f"Moment {i}. A scene.", "caption": "",
                            "image": None, "depth": None, "aspect": None, "ready": False, "source": None,
                            "action": {"subject": "Egg", "motion": f"the egg does {path}", "path": path,
                                       "loop": path == "still"},
                            "actor": None, "plate": None, "actor_box": None, "actor_check": None, "world": None,
                            "clip": None}
    st = _state(shots=[shot(i, p) for i, p in enumerate(paths)], status="running", _plan_done=False,
                play_when="clip", _world_pano=np.full((1536, 3072, 3), 120, np.uint8))
    for s in st["shots"]:
        live._shot(st, s, painting, seed=5)
    # Every moment is queued for filming the moment it is painted (no _check_reading has run).
    assert all(s["clip"] and s["clip"]["status"] not in ("off", None) for s in st["shots"])
    assert len(st["_slow"]) == 4
    for f in list(st["_slow"]):
        f.result(timeout=120)
    assert len(calls) == 4 and all(c["size"] == (720, 1280, 3) for c in calls)
    assert sorted(c["priority"][1] for c in calls) == [0, 1, 2, 3]
    for f in list(st["_tasks"]):
        f.result(timeout=120)
    clips = [s["clip"] for s in st["shots"]]
    assert [c["status"] for c in clips] == ["ready"] * 4, [(c["status"], c.get("reason")) for c in clips]
    for s in st["shots"]:
        assert live.file(st["id"], s["image"].replace(".jpg", ".mp4"))
        assert live.file(st["id"], s["image"].replace(".jpg", "_720.jpg"))
        assert s["clip"]["size"] == [1280, 720]
    snap = live._snapshot(st)
    assert not snap["upgrading"] and st["timings"]["clips"] is not None
    assert snap["play_when"] == "clip"
    assert next(x for x in snap["steps"] if x["name"] == "clips")["done_count"] == 4


def test_a_retired_video_model_switches_the_animator_off_once(nova, monkeypatch):
    """What Bedrock really answered on Oct 26 for Nova Reel: the story must not wait on a model that is gone."""
    from botocore.exceptions import ClientError

    class Gone:
        def start_async_invoke(self, **kw):
            raise ClientError({"Error": {"Code": "ResourceNotFoundException", "Message":
                               "This model version has reached the end of its life."}}, "StartAsyncInvoke")

    monkeypatch.setattr(animator, "_runtime", lambda reg: Gone())
    with pytest.raises(animator.AnimatorError) as err:
        animator._generate(_jpg(np.zeros((720, 1280, 3), np.uint8)), "a scene", False, time.monotonic() + 30)
    assert "retired" in err.value.reason and not err.value.retry
    assert not animator.enabled()  # no further clips are asked for


def test_the_first_moment_starts_from_the_page_own_picture(monkeypatch, tmp_path):
    from agents import painter
    from server.live import Live
    from server.storage import LocalStore
    pic = np.zeros((600, 450, 3), np.uint8)  # a portrait drawing on the page
    pic[:] = (30, 160, 60)
    cv.circle(pic, (225, 300), 120, (20, 20, 200), -1)
    seen = {}

    def fake_paint(canvas, generate, prompt, negative, seed=7):
        seen["mask"] = generate.copy()
        out = canvas.copy()
        out[generate > 0] = (200, 180, 150)
        return out

    monkeypatch.setattr(painter, "enabled", lambda: True)
    monkeypatch.setattr(painter, "paint", fake_paint)
    live = Live(LocalStore(tmp_path))
    img, source = live._page_picture({"_page_picture": pic}, "a frog by a pond", 1)
    assert img.shape == (756, 1344, 3) and source == "page + Stability AI Inpaint"
    assert abs(img[378, 672].astype(int) - (20, 20, 200)).max() < 30  # the page's own drawing, in the middle
    assert (seen["mask"][:, :200] == 255).all() and (seen["mask"][378, 600:740] == 0).all()  # only margins painted
    assert live._page_picture({}, "x", 1) == (None, None)


def test_luma_start_frame_is_never_wider_than_its_limit(monkeypatch):
    monkeypatch.setenv("VIDEO_MODEL", "luma.ray-v2:0")
    wide = _jpg(np.zeros((960, 2304, 3), np.uint8))
    body = animator.request_body(wide, "a scene", False)
    frame = cv.imdecode(np.frombuffer(base64.b64decode(body["keyframes"]["frame0"]["source"]["data"]), np.uint8), 1)
    assert frame.shape[1] == 1536 and abs(frame.shape[1] / frame.shape[0] - 2304 / 960) < 0.01
    assert body["aspect_ratio"] == "21:9" and body["duration"] == "5s"


def test_what_the_cinderella_story_exposed(monkeypatch):
    """Oct 26, a real Cinderella page: every moment came back "still" though the agent described movement, the
    video prompt dropped the described action, and Luma refused prompts naming famous characters."""
    from agents.planner import clean_action
    monkeypatch.setenv("VIDEO_MODEL", "luma.ray-v2:0")
    rush = clean_action({"subject": "Cinderella", "motion": "rushes out of the ball, leaving a glass slipper behind",
                         "path": "still"})
    assert rush["path"] == "walk-right"
    assert clean_action({"motion": "waves her wand", "path": "still"})["path"] == "sway"
    assert clean_action({"motion": "tries the glass slipper on various women", "path": "still"})["path"] == "still"
    prompt = animator.clip_prompt({"subject": "fairy godmother", "motion": "waves her wand", "path": "still"},
                                  None, "watercolour", "Cinderella's fairy godmother appears.")
    assert "waves her wand" in prompt and "Gentle living motion" not in prompt  # the described action is filmed
    cast = [{"name": "Cinderella", "look": "A young woman in a ragged grey dress"}]
    anon = animator.anonymize(prompt, cast, "fairy godmother")
    assert "Cinderella" not in anon and "godmother" not in anon and "waves her wand" in anon
    assert animator.moderation_refusal("ip prompt moderation failed")


def test_a_refused_famous_name_is_retried_with_descriptions(monkeypatch, tmp_path):
    from server.live import Live
    from server.storage import LocalStore
    from vision.tests.test_live import _state
    monkeypatch.setenv("VIDEO_MODEL", "luma.ray-v2:0")
    prompts = []

    def fake_submit(priority, still_jpg, prompt, loop, expires_at, on_status=None, seed=0):
        prompts.append(prompt)
        f = Future()
        f.set_exception(animator.AnimatorError("ip prompt moderation failed"))
        return f

    monkeypatch.setattr(animator, "submit", fake_submit)
    live = Live(LocalStore(tmp_path))
    shot = {"i": 0, "image": "shot_00.jpg", "prompt": "Cinderella sits by the fire.", "ready": True, "lines": [],
            "source": "stable-image-core", "aspect": 1.78, "actor_box": None,
            "action": {"subject": "Cinderella", "motion": "sweeps the floor", "path": "still", "loop": True}}
    st = _state(shots=[shot], cast=[{"name": "Cinderella", "look": "a young woman in a grey dress", "voice": "woman"}])
    live._put(st, "shot_00.jpg", _jpg(np.zeros((756, 1344, 3), np.uint8)))
    shot["clip"] = live._new_clip("shot_00.jpg")
    live._clip(st, shot, 0)
    assert len(prompts) == 2 and "Cinderella" in prompts[0] and "Cinderella" not in prompts[1]
    assert "the young woman in a grey dress" in prompts[1]
    assert shot["clip"]["status"] == "off" and "moderation" in shot["clip"]["reason"]


def test_every_character_the_moment_names_is_painted_from_its_description(monkeypatch, tmp_path):
    """The tortoise story (Oct 26): "Tortoise and Hare decide to race" named both only in the action, so the hare
    was never described to the painter and the picture showed three tortoises."""
    from agents import imagery, painter
    from server.live import Live
    from server.storage import LocalStore
    from vision.tests.test_live import _state
    prompts = []
    monkeypatch.setattr(imagery, "generate", lambda prompt, negative, aspect="16:9", seed=0, wait_s=30.0:
                        (prompts.append(prompt) or None, None))
    monkeypatch.setattr(painter, "enabled", lambda: False)
    monkeypatch.setattr(animator, "enabled", lambda: False)
    live = Live(LocalStore(tmp_path))
    shot = {"i": 0, "lines": [], "effects": [], "prompt": "A meadow with animals gathered, laughing.", "caption": "",
            "image": None, "ready": False, "source": None, "actor_box": None, "clip": None, "world": None,
            "action": {"subject": "Tortoise and Hare", "motion": "decide to race", "path": "walk-right", "loop": False}}
    cast = [{"name": "Hare", "look": "a brown hare with long ears", "voice": "man"},
            {"name": "Tortoise", "look": "a green tortoise with a brown shell", "voice": "child"}]
    st = _state(shots=[shot], cast=cast, _world_pano=np.full((1536, 3072, 3), 120, np.uint8))
    live._shot(st, shot, np.full((600, 800, 3), 200, np.uint8), seed=1)
    assert "Hare: a brown hare with long ears" in prompts[0] and "Tortoise: a green tortoise" in prompts[0]


def test_the_clips_bucket_is_found_without_any_setting(monkeypatch):
    from botocore.exceptions import ClientError
    monkeypatch.delenv("VIDEO_BUCKET", raising=False)
    monkeypatch.delenv("VIDEO_REGION", raising=False)
    monkeypatch.setenv("VIDEO_MODEL", "luma.ray-v2:0")
    exists = {"step-inside-clips-123456789012-us-west-2"}

    class Fake:
        def __init__(self, service, **kw):
            self.service = service

        def get_caller_identity(self):
            return {"Account": "123456789012"}

        def head_bucket(self, Bucket):
            if Bucket not in exists:
                raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket")

    monkeypatch.setattr(animator.boto3, "client", lambda service, **kw: Fake(service, **kw))
    monkeypatch.setattr(animator, "_found", {})
    assert animator.bucket() == "step-inside-clips-123456789012-us-west-2"
    exists.clear()
    monkeypatch.setattr(animator, "_found", {})
    assert animator.bucket() is None  # no bucket: video simply stays off
    monkeypatch.setenv("VIDEO_BUCKET", "")
    assert animator.bucket() is None  # an empty setting turns video off without asking AWS
