import importlib
import time

import cv2 as cv
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vision.tests.test_coach import parallax_view


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("DATA_DIR", str(tmp_path_factory.mktemp("data")))
    mp.setenv("STEP_INSIDE_LLM", "off")
    mp.delenv("SCENE_BUCKET", raising=False)
    mp.delenv("STATE_MACHINE_ARN", raising=False)
    import server.app
    app_module = importlib.reload(server.app)
    with TestClient(app_module.app) as c:
        yield c
    mp.undo()


def _jpg(img: np.ndarray) -> bytes:
    return cv.imencode(".jpg", img)[1].tobytes()


def test_health(client):
    body = client.get("/api/health").json()
    assert body["ok"] and body["opencv"].startswith("5.") and body["pipeline"] == "local"


def test_photo_becomes_a_layered_scene_with_narration(client):
    r = client.post("/api/scenes", files={"image": ("p.jpg", _jpg(parallax_view(0)), "image/jpeg")},
                    data={"kind": "photo", "title": "Dots"})
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["type"] == "2.5d" and meta["title"] == "Dots"
    for name in ("scene.glb", "input.jpg", "depth.jpg", "layers.jpg", "filled.jpg"):
        f = client.get(f"/api/scenes/{meta['id']}/files/{name}")
        assert f.status_code == 200 and len(f.content) > 1000, name
    assert client.get(f"/api/scenes/{meta['id']}/files/scene.glb").content[:4] == b"glTF"
    for _ in range(20):
        meta = client.get(f"/api/scenes/{meta['id']}").json()
        if "narration" in meta:
            break
        time.sleep(0.25)
    assert meta["narration"]["source"] == "rules" and meta["narration"]["text"]
    assert any(s["id"] == meta["id"] for s in client.get("/api/scenes").json())


def test_360_panorama_is_accepted_and_needs_no_mesh(client):
    pano = cv.resize(parallax_view(0), (2048, 1024))
    r = client.post("/api/scenes", files={"image": ("p.jpg", _jpg(pano), "image/jpeg")}, data={"kind": "360"})
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["type"] == "360" and meta["image"] == {"w": 2048, "h": 1024}
    assert client.get(f"/api/scenes/{meta['id']}/files/input.jpg").status_code == 200


def test_360_rejects_a_regular_photo_with_a_clear_reason(client):
    r = client.post("/api/scenes", files={"image": ("p.jpg", _jpg(parallax_view(0)), "image/jpeg")},
                    data={"kind": "360"})
    assert r.status_code == 422
    assert "2:1" in r.json()["detail"]


def _book_page() -> np.ndarray:
    page = np.full((1300, 1000, 3), (238, 244, 248), np.uint8)
    page[80:700, 160:840] = parallax_view(0)[100:720, 200:880]  # the "illustration"
    for i, line in enumerate(["The little rabbit sat", "under the big tree."]):
        cv.putText(page, line, (150, 860 + i * 90), cv.FONT_HERSHEY_DUPLEX, 2.0, (40, 30, 30), 3, cv.LINE_AA)
    return page


def test_story_from_a_book_page(client):
    r = client.post("/api/stories", files=[("pages", ("p1.jpg", _jpg(_book_page()), "image/jpeg"))],
                    data={"title": "Test story"})
    assert r.status_code == 200, r.text
    rec = r.json()
    for _ in range(240):
        rec = client.get(f"/api/stories/{rec['id']}").json()
        if rec["status"] != "running":
            break
        time.sleep(0.5)
    assert rec["status"] == "done", rec
    assert rec["title"] == "Test story" and len(rec["pages"]) == 1
    meta = client.get(f"/api/scenes/{rec['pages'][0]}").json()
    assert meta["type"] == "story" and meta["kind"] == "story_page"
    assert "rabbit" in meta["story"]["text"].lower()
    assert meta["story"]["beats"] and meta["story"]["world"]["source"] == "opencv"
    for name in ("scene.glb", "panorama.jpg", "page.jpg"):
        assert client.get(f"/api/scenes/{meta['id']}/files/{name}").status_code == 200, name
    assert all(s["id"] != meta["id"] for s in client.get("/api/scenes").json())  # pages live under their story
    assert any(s["id"] == rec["id"] for s in client.get("/api/stories").json())


def test_live_story_from_one_page(client):
    r = client.post("/api/live", files={"page": ("p.jpg", _jpg(_book_page()), "image/jpeg")})
    assert r.status_code == 200, r.text
    st = r.json()
    for _ in range(240):
        st = client.get(f"/api/live/{st['id']}").json()
        if st["status"] != "running":
            break
        time.sleep(0.25)
    assert st["status"] == "ready", st
    # No model in tests: the lines are OpenCV's reading, the moments show the page's own picture.
    assert "rabbit" in " ".join(ln["text"] for ln in st["lines"]).lower()
    assert st["shots"] and all(s["ready"] and s["source"] == "opencv" for s in st["shots"])
    assert all(ln["ready"] and ln["box"] for ln in st["lines"]) and st["timings"]["playable"] is not None
    assert [s["name"] for s in st["steps"]] == ["read", "plan", "world", "shots", "voices"]
    for name in ("page.jpg", st["shots"][0]["image"], st["shots"][0]["depth"]):
        assert client.get(f"/api/live/{st['id']}/files/{name}").status_code == 200, name
    assert client.get(f"/api/live/{st['id']}/files/state.json").status_code == 404
    assert all(s["clip"] is None and s["action"]["path"] == "still" for s in st["shots"])  # no animator in tests
    assert st["shots"][0]["world"]["image"] == "shot_00_world.jpg" and not st["upgrading"]
    assert st["play_when"] == "still"  # no animator: the story starts on the paintings, it doesn't wait for a film
    assert any(s["id"] == st["id"] for s in client.get("/api/live").json())


def test_live_queue_is_capped(client):
    import server.app as app_module
    from server.live import MAX_PENDING
    fakes = {f"fake-{i}": {"status": "running"} for i in range(MAX_PENDING)}
    app_module.live.runs.update(fakes)
    page = np.full((900, 700, 3), 235, np.uint8)  # a page no story has yet
    cv.putText(page, "A brand new page of words", (40, 400), cv.FONT_HERSHEY_DUPLEX, 1.2, (20, 20, 20), 2)
    try:
        r = client.post("/api/live", files={"page": ("p.jpg", _jpg(page), "image/jpeg")})
        assert r.status_code == 429
    finally:
        for k in fakes:
            app_module.live.runs.pop(k, None)


def test_a_page_that_already_has_a_story_is_replayed_not_rebuilt(client):
    first = client.get("/api/live").json()
    assert first, "test_live_story_from_one_page runs first and makes a story"
    r = client.post("/api/live", files={"page": ("p.jpg", _jpg(_book_page()), "image/jpeg")})
    body = r.json()
    assert r.status_code == 200 and body.get("reused") and body["id"] == first[0]["id"]


def test_story_queue_is_capped(client):
    import server.app as app_module
    app_module.stories.active.update({"fake-1", "fake-2"})
    try:
        r = client.post("/api/stories", files=[("pages", ("p1.jpg", _jpg(_book_page()), "image/jpeg"))])
        assert r.status_code == 429
    finally:
        app_module.stories.active.difference_update({"fake-1", "fake-2"})


def test_story_left_running_by_a_restart_is_reported_as_failed(client):
    import json
    import server.app as app_module
    rec = {"id": "abcdefabcdef", "title": "Lost", "status": "running", "stage": "Page 1 of 2: painting",
           "total": 2, "pages": [], "errors": [], "error": None}
    app_module.store.put("stories/abcdefabcdef/story.json", json.dumps(rec).encode(), "application/json")
    body = client.get("/api/stories/abcdefabcdef").json()
    assert body["status"] == "failed" and "restarted" in body["error"]


def test_recent_stories_are_newest_first_whatever_order_the_store_lists():
    import json
    from server.stories import Stories

    class Store:  # lists folders alphabetically, like S3
        def __init__(self):
            self.data = {}

        def put(self, key, data, content_type):
            self.data[key] = data

        def get(self, key):
            return self.data.get(key)

        def list_dirs(self, prefix):
            return sorted({k[len(prefix):].split("/")[0] for k in self.data if k.startswith(prefix)})

    s = Stories(Store(), None)
    for sid, t in [("aaaaaaaaaaaa", 100), ("bbbbbbbbbbbb", 300), ("cccccccccccc", 200)]:
        rec = {"id": sid, "title": sid, "status": "done", "pages": ["x"], "updated": t}
        s.store.put(f"stories/{sid}/story.json", json.dumps(rec).encode(), "application/json")
    assert [r["id"] for r in s.recent()] == ["bbbbbbbbbbbb", "cccccccccccc", "aaaaaaaaaaaa"]


def test_capture_job_finishes_with_a_scene(client):
    files = [("photos", (f"{i}.jpg", _jpg(parallax_view(dx)), "image/jpeg")) for i, dx in enumerate(range(0, 120, 8))]
    job = client.post("/api/captures", files=files, data={"title": "Sweep"}).json()
    for _ in range(240):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.5)
    assert job["status"] == "done", job
    meta = client.get(f"/api/scenes/{job['scene_id']}").json()
    assert meta["type"] in ("3d", "2.5d")  # flat two-plane scene may legitimately fall back to 2.5D
    assert job["verifier"]["next_action"] in ("done", "record_more", "fallback_2_5d")


def test_coach_session(client):
    sid = client.post("/api/coach").json()["session"]
    r = client.post(f"/api/coach/{sid}/frame", files={"image": ("f.jpg", _jpg(parallax_view(0)), "image/jpeg")})
    assert r.json()["keyframe"] is True


def test_rejects_bad_ids_and_names(client):
    assert client.get("/api/scenes/..%2F..%2Fetc").status_code in (400, 404)
    assert client.get("/api/scenes/abcdefabcdef/files/..%2Fmeta.json").status_code == 404
    assert client.get("/api/jobs/not-a-job").status_code == 400


def test_rejects_non_images(client):
    r = client.post("/api/scenes", files={"image": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 415


def test_video_of_a_page_uses_its_clearest_frame(client, tmp_path):
    page = np.full((720, 1280, 3), 60, np.uint8)
    sheet = np.full((560, 820, 3), 238, np.uint8)
    for i, line in enumerate(["The little owl looked up", "at the big round moon", "and sang a quiet song"]):
        cv.putText(sheet, line, (40, 140 + i * 110), cv.FONT_HERSHEY_DUPLEX, 1.5, (25, 25, 25), 3, cv.LINE_AA)
    page[80:640, 230:1050] = sheet
    path = tmp_path / "page.mp4"
    out = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
    for i in range(30):  # blurry while the hand moves, then still and sharp
        out.write(cv.GaussianBlur(page, (0, 0), 6) if i < 18 else page)
    out.release()
    r = client.post("/api/live", files={"video": ("page.mp4", path.read_bytes(), "video/mp4")})
    assert r.status_code == 200, r.text
    picked = r.json()["picked_frame"]
    assert picked["ok"] and picked["frame"] >= 18 * picked["frames"] // 30 - 1  # one of the sharp frames


def test_story_files_answer_range_requests(client):
    st = client.get("/api/live").json()[0]
    url = f"/api/live/{st['id']}/files/page.jpg"
    whole = client.get(url)
    total = len(whole.content)
    assert whole.status_code == 200 and whole.headers["accept-ranges"] == "bytes"
    part = client.get(url, headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100 and part.headers["content-range"] == f"bytes 0-99/{total}"
    tail = client.get(url, headers={"Range": "bytes=-10"})
    assert tail.status_code == 206 and tail.content == whole.content[-10:]
    assert client.get(url, headers={"Range": f"bytes={total + 5}-"}).status_code == 416


def test_one_password_guards_making_stories_but_not_watching(client, monkeypatch):
    from server import auth
    monkeypatch.setenv("APP_PASSWORD", "judges-only-test")
    monkeypatch.setattr(auth, "_fails", {})
    client.cookies.clear()
    page = {"page": ("p.jpg", _jpg(_book_page()), "image/jpeg")}
    assert client.get("/api/auth").json() == {"required": True, "ok": False}
    assert client.post("/api/live", files=page).status_code == 401
    assert client.post("/api/look", files={"image": ("f.jpg", _jpg(_book_page()), "image/jpeg")}).status_code == 401
    assert client.get("/api/live").status_code == 200  # watching stays open
    assert client.post("/api/login", json={"password": "wrong"}).status_code == 401
    r = client.post("/api/login", json={"password": "judges-only-test"})
    assert r.status_code == 200 and "httponly" in r.headers["set-cookie"].lower()
    assert "samesite=strict" in r.headers["set-cookie"].lower()
    assert client.get("/api/auth").json()["ok"] is True
    assert client.post("/api/live", files=page).status_code in (200, 429)  # allowed now (or busy)
    client.post("/api/logout")
    client.cookies.clear()
    assert client.post("/api/live", files=page).status_code == 401


def test_wrong_passwords_are_slowed_down(client, monkeypatch):
    from server import auth
    monkeypatch.setenv("APP_PASSWORD", "judges-only-test")
    monkeypatch.setattr(auth, "_fails", {})
    codes = [client.post("/api/login", json={"password": f"guess{i}"}).status_code for i in range(7)]
    assert codes[:5] == [401] * 5 and codes[5:] == [429, 429]
    assert client.post("/api/login", json={"password": "judges-only-test"}).status_code == 429  # even the right one


def test_security_headers_and_no_public_api_docs(client):
    r = client.get("/")
    for h in ("content-security-policy", "x-content-type-options", "x-frame-options", "referrer-policy"):
        assert h in r.headers, h
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404


def test_session_tokens_expire_and_cannot_be_forged(monkeypatch):
    from server import auth
    monkeypatch.setenv("APP_PASSWORD", "judges-only-test")
    t = auth.make_token(now=1000)
    assert auth.valid(t, now=1000 + 60) and not auth.valid(t, now=1000 + auth.TTL_S + 1)
    exp, sig = t.split(".")
    assert not auth.valid(f"{int(exp) + 9999}.{sig}", now=1000)  # a stretched expiry breaks the signature
    monkeypatch.setenv("APP_PASSWORD", "a-new-password")
    assert not auth.valid(t, now=1000 + 60)  # changing the password logs everyone out


def test_requests_that_skip_cloudfront_are_refused(client, monkeypatch):
    monkeypatch.setenv("ORIGIN_VERIFY", "cloudfront-only-secret")
    assert client.get("/api/live").status_code == 404
    assert client.get("/api/health").status_code == 200  # the load balancer's health check
    assert client.get("/api/live", headers={"x-origin-verify": "cloudfront-only-secret"}).status_code == 200
