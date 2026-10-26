"""Looking through a camera: the photo coach, recognizing a known page, and finding what can move in a painting."""
import cv2 as cv
import numpy as np

from vision.look import PageIndex, check
from vision.shot import motion_regions


STORY = ["Once upon a time there lived", "a small rabbit under a tree", "who loved to look at stars"]
OTHER = ["The fox ran down the lane", "past the old mill and the pond", "and home before the rain"]


def _page(seed: int = 0, text: list[str] = STORY) -> np.ndarray:
    page = np.full((900, 700, 3), (236, 240, 244), np.uint8)
    rng = np.random.default_rng(seed)
    cv.rectangle(page, (80, 60), (620, 420), tuple(int(v) for v in rng.integers(40, 200, 3)), -1)
    for _ in range(25):  # a "picture"
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv.circle(page, tuple(int(v) for v in rng.integers((100, 80), (600, 400))), int(rng.integers(8, 50)), c, -1)
    for i, line in enumerate(text):
        cv.putText(page, line, (70, 520 + i * 70), cv.FONT_HERSHEY_DUPLEX, 1.1, (30, 30, 30), 2, cv.LINE_AA)
    return page


def _held(page: np.ndarray) -> np.ndarray:
    """The page as a camera sees it: smaller, in perspective, on a table."""
    h, w = page.shape[:2]
    dst = np.float32([[300, 60], [860, 100], [900, 700], [260, 680]])
    M = cv.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
    return cv.warpPerspective(page, M, (1280, 760), borderValue=(70, 60, 50))


def test_coach_tells_what_to_fix():
    frame = _held(_page())
    assert check(frame)["ok"], check(frame)
    assert check(cv.GaussianBlur(frame, (0, 0), 3))["issues"][:1] == ["blurry"]
    assert "dark" in check((frame * 0.15).astype(np.uint8))["issues"]
    assert check(np.full((760, 1280, 3), 90, np.uint8))["issues"] == ["no_text"]
    cut = check(frame[:, 560:])
    assert "cut_off" in cut["issues"] and cut["say"]


def test_known_page_is_recognized_and_others_are_not():
    idx = PageIndex()
    idx.add("a", "Rabbit", _page(0))
    idx.add("b", "Fox", _page(7, OTHER))
    hit = idx.match(_held(_page(0)))
    assert hit and hit["id"] == "a" and hit["inliers"] >= 30
    assert idx.match(_held(_page(7, OTHER)))["id"] == "b"
    noise = np.random.default_rng(3).integers(0, 255, (760, 1280, 3), dtype=np.uint8)
    assert idx.match(noise) is None


def test_pages_that_cannot_be_told_apart_make_no_match():
    idx = PageIndex()
    idx.add("a", "", _page(0))
    idx.add("b", "", _page(7))  # same words, different (featureless) picture
    assert idx.match(_held(_page(0))) is None


def test_sky_and_leaves_move_and_water_only_when_the_story_has_some():
    h, w = 200, 480
    img = np.zeros((h, w, 3), np.uint8)
    img[:90] = (110, 60, 20)                      # smooth night sky
    img[90:] = (40, 140, 60)                      # green ground...
    noise = np.random.default_rng(0).integers(-40, 40, (h - 90, w, 3))
    img[90:] = np.clip(img[90:] + noise, 0, 255)  # ...with the texture of leaves
    img[150:, 100:380] = (160, 110, 30)           # a smooth blue band low down: a pond, or a blue coat?
    near = np.zeros((h, w), np.float32)
    near[90:] = np.linspace(0.3, 1.0, h - 90)[:, None]
    dry = motion_regions(img, near)
    assert dry["sky"][:60].mean() > 0.8 and dry["sky"][120:].mean() < 0.05
    assert dry["leaves"][100:140].mean() > 0.5 and dry["water"].max() == 0
    wet = motion_regions(img, near, water=True)
    assert wet["water"][165:, 150:330].mean() > 0.6


def test_look_endpoint(monkeypatch):
    import importlib
    import os
    from fastapi.testclient import TestClient
    monkeypatch.setenv("STEP_INSIDE_LLM", "off")
    monkeypatch.setenv("DATA_DIR", os.environ.get("PYTEST_TMP", "/tmp") + "/look-test")
    import server.app
    app_module = importlib.reload(server.app)
    with TestClient(app_module.app) as c:
        jpg = cv.imencode(".jpg", _held(_page()))[1].tobytes()
        body = c.post("/api/look", files={"image": ("f.jpg", jpg, "image/jpeg")}).json()
    assert body["quality"]["ok"] and body["quality"]["lines"] >= 2 and body["match"] is None
