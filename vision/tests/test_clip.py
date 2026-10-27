"""The clip critic on small synthetic clips, and the actor agent on a synthetic painting."""
import cv2 as cv
import numpy as np
import pytest

from vision import clip as critic

W, H = 640, 274  # 21:9


def _still() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = np.linspace(80, 200, W, dtype=np.uint8)[None, :, None]
    img[:, :, 0] = np.linspace(200, 120, H, dtype=np.uint8)[:, None]
    rng = np.random.default_rng(1)
    for _ in range(60):  # texture, so features and flow have something to hold on to
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv.circle(img, (x, y), int(rng.integers(3, 12)), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return img


def _frame(base: np.ndarray, x: int, y: int) -> np.ndarray:
    f = base.copy()
    cv.rectangle(f, (x, y), (x + 50, y + 60), (20, 20, 30), -1)
    cv.circle(f, (x + 25, y + 20), 8, (240, 240, 240), -1)
    return f


def _write(path, frames, fps=24):
    out = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not out.isOpened():
        pytest.skip("this OpenCV build cannot write mp4v")
    for f in frames:
        out.write(f)
    out.release()
    return path.read_bytes()


@pytest.fixture(scope="module")
def base():
    return _still()


def test_critic_sees_a_fall(tmp_path, base):
    frames = [_frame(base, 300, 40 + int(120 * min(1, i / 70) ** 2)) for i in range(96)]
    v = critic.verify(frames[0], _write(tmp_path / "f.mp4", frames), {"path": "fall"})
    assert v["ok"], v
    assert v["observed"] == "fall" and v["scores"]["dy"] > 0.08 and v["hold"] and v["loop_at"] is None
    wrong = critic.verify(frames[0], _write(tmp_path / "f2.mp4", frames), {"path": "rise"})
    assert not wrong["ok"] and wrong["retry"]


def test_critic_rejects_a_clip_where_nothing_happens(tmp_path, base):
    frames = [_frame(base, 300, 100) for _ in range(72)]
    v = critic.verify(frames[0], _write(tmp_path / "s.mp4", frames), {"path": "fall"})
    assert not v["ok"] and v["retry"] and "nothing happens" in v["reasons"]


def test_critic_rejects_lettering(tmp_path, base):
    frames = []
    for i in range(72):
        f = _frame(base, 200 + 2 * i, 100)
        cv.putText(f, "WELCOME TO TOWN", (60, 230), cv.FONT_HERSHEY_DUPLEX, 1.4, (10, 10, 10), 3, cv.LINE_AA)
        frames.append(f)
    v = critic.verify(frames[0], _write(tmp_path / "t.mp4", frames), {"path": "walk-right"})
    assert "letters appeared in the picture" in v["reasons"]


def test_critic_rejects_a_clip_that_forgets_the_painting(tmp_path, base):
    other = np.random.default_rng(7).integers(0, 255, (H, W, 3), dtype=np.uint8)
    other = cv.GaussianBlur(other, (0, 0), 3)
    frames = [_frame(other, 300, 40 + i) for i in range(72)]
    v = critic.verify(_frame(base, 300, 40), _write(tmp_path / "o.mp4", frames), {"path": "fall"})
    assert "it does not start from the painting" in v["reasons"]


def test_critic_finds_a_loop_point_for_a_sway(tmp_path, base):
    frames = [_frame(base, 300 + int(26 * np.sin(2 * np.pi * i / 48)), 100) for i in range(120)]
    v = critic.verify(frames[0], _write(tmp_path / "w.mp4", frames), {"path": "sway", "loop": True})
    assert v["observed"] == "sway" and v["loop_at"] is not None and v["loop_at"] >= 2 and not v["hold"], v


def test_critic_flags_a_jump_cut(tmp_path, base):
    other = cv.flip(cv.cvtColor(cv.cvtColor(base, cv.COLOR_BGR2HSV)[..., ::-1].copy(), cv.COLOR_HSV2BGR), 1)
    frames = [_frame(base if i < 48 else other, 300, 40 + i) for i in range(96)]
    v = critic.verify(frames[0], _write(tmp_path / "c.mp4", frames), {"path": "fall"})
    assert v["scores"]["cuts"] >= 1 and "it jumps to a different shot" in v["reasons"]


def test_actor_is_cut_out_of_a_painting_and_the_plate_has_no_hole(monkeypatch):
    from vision import actors
    img = np.zeros((480, 1152, 3), np.uint8)
    img[:] = (170, 140, 90)
    img[300:] = (60, 130, 70)
    cv.ellipse(img, (560, 250), (60, 90), 0, 0, 360, (40, 60, 200), -1)  # the character
    near = np.full((480, 1152), 0.2, np.float32)
    near[300:] = np.linspace(0.3, 0.6, 180, dtype=np.float32)[:, None]
    cv.ellipse(near, (560, 250), (60, 90), 0, 0, 360, 0.9, -1)
    monkeypatch.setattr(actors.depth, "inverse_depth", lambda x: near)
    monkeypatch.setattr(actors.depth, "normalized", lambda x: x)
    monkeypatch.setattr(actors, "detect", lambda x: [])
    out = actors.make_actor(img)
    assert out["check"]["ok"], out["check"]
    x0, y0, x1, y1 = out["box"]
    assert x0 < 520 and x1 > 600 and y0 < 180 and y1 > 320
    alpha = out["actor"][..., 3]
    assert alpha.shape[:2] == (y1 - y0, x1 - x0) and (alpha > 128).mean() > 0.3
    plate = out["plate"][180:250, 540:580].reshape(-1, 3).astype(int)
    assert np.abs(plate - (170, 140, 90)).max(1).mean() < 40  # the sky shows where the character stood


def test_actor_judge_refuses_the_ground_and_scraps():
    from vision.actors import judge
    ground = np.zeros((480, 1152), bool)
    ground[380:] = True
    assert judge(ground)["reason"] in ("it is the ground", "too big", "mostly on the edge")
    tiny = np.zeros((480, 1152), bool)
    tiny[10:14, 10:14] = True
    assert judge(tiny)["reason"] == "too small"
    blob = np.zeros((480, 1152), bool)
    cv.circle(blob.view(np.uint8), (500, 240), 70, 1, -1)
    assert judge(blob)["ok"]


def test_extra_figures_reject_only_a_steady_shot(monkeypatch, tmp_path, base):
    """Real Luma clips (Oct 26): a zoom on a crowded stage made YOLOX count more people. That is reported as the
    camera moving, not as invented figures; figures invented while the camera holds still are still rejected."""
    frames = [_frame(base, 200 + 3 * i, 100) for i in range(96)]  # a walk with a steady camera
    data = _write(tmp_path / "w.mp4", frames)
    counts = iter([1] + [5] * 10)  # the painting has 1 figure; every clip frame "has" 5
    monkeypatch.setattr(critic, "figures", lambda img: next(counts))
    steady = critic.verify(frames[0], data, {"path": "walk-right"})
    assert "new figures appeared" in steady["reasons"]
    counts2 = iter([1] + [5] * 10)
    monkeypatch.setattr(critic, "figures", lambda img: next(counts2))
    monkeypatch.setattr(critic, "cuts", lambda fr: 1)  # the zoom reads as a change of shot
    zoom = critic.verify(frames[0], data, {"path": "walk-right"})
    assert "new figures appeared" not in zoom["reasons"] and "it jumps to a different shot" in zoom["reasons"]
    assert zoom["scores"]["figures_max"] == 5  # still recorded


def test_story_agent_frames_one_character_close():
    from agents.planner import SYSTEM
    assert "close or medium shot" in SYSTEM and "Never a wide crowd" in SYSTEM


def test_which_way_a_face_looks_from_its_landmarks():
    from vision.actors import facing, facing_from_landmarks
    row = lambda nose_x: np.array([0, 0, 100, 100, 30, 40, 70, 40, nose_x, 60, 35, 80, 65, 80, 0.9], np.float32)
    assert facing_from_landmarks(row(50)) == "front"
    assert facing_from_landmarks(row(64)) == "right" and facing_from_landmarks(row(36)) == "left"
    assert facing(np.zeros((200, 120, 4), np.uint8)) is None  # nothing like a face: no guess


def test_a_walk_in_either_direction_is_the_action():
    """The tortoise story (Oct 26): a take was rejected only because the tortoise walked left, not right."""
    t = {"subject_speed": 0.05, "background_speed": 0.0, "dx": -0.2, "dy": 0.0, "peak_lift": 0.0, "reversals": 0}
    observed, retry, reasons = critic.classify("walk-right", t)
    assert observed == "walk-left" and reasons == [] and not retry
    observed, retry, reasons = critic.classify("walk-left", {**t, "dx": 0.2})
    assert observed == "walk-right" and reasons == []
    assert critic.classify("walk-right", {**t, "dx": 0.01})[0] == "unclear"


def test_a_clip_that_turns_into_another_scene_is_rejected(tmp_path, base):
    """The minister story (Oct 26): a man at a desk became two men standing over one second; frame 0 matched the
    painting (815 features) and nothing caught it. The middle and last frames are now compared too."""
    other = cv.GaussianBlur(np.random.default_rng(3).integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 2)
    frames = [_frame(base, 300, 100) if i < 30 else _frame(other, 300, 100) for i in range(96)]
    for i in range(30, 42):  # a slow morph, not a cut
        a = (i - 29) / 13
        frames[i] = cv.addWeighted(_frame(base, 300, 100), 1 - a, _frame(other, 300, 100), a, 0)
    v = critic.verify(frames[0], _write(tmp_path / "m.mp4", frames), {"path": "still"})
    assert "it turns into a different scene" in v["reasons"] and v["scores"]["continuity_inliers"] < 40
    steady = [_frame(base, 300 + i, 100) for i in range(96)]
    ok = critic.verify(steady[0], _write(tmp_path / "s.mp4", steady), {"path": "walk-right"})
    assert "it turns into a different scene" not in ok["reasons"] and ok["scores"]["continuity_inliers"] >= 40
