import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("DATA_DIR", str(tmp_path_factory.mktemp("data")))
    mp.setenv("WHATCHANGED_LLM", "off")
    import server.app
    app_module = importlib.reload(server.app)
    with TestClient(app_module.app) as c:
        yield c
    mp.undo()


def _scan(client, image: str) -> dict:
    with open(ROOT / "web" / image, "rb") as f:
        r = client.post("/api/homes/demo/scan", files={"image": ("f.jpg", f, "image/jpeg")})
    assert r.status_code == 200, r.text
    return r.json()


def _index() -> dict:
    return json.loads((ROOT / "web" / "demo" / "index.json").read_text())


def test_health(client):
    body = client.get("/api/health").json()
    assert body["ok"] and body["opencv"].startswith("5.")


def test_demo_scenarios_report_expected_changes(client):
    expected = {
        "lights_on": [],
        "shoe_on_floor": ["appeared"],
        "mug_moved": ["moved"],
        "mug_missing": ["disappeared"],
        "kitchen_spill": ["appeared"],
    }
    for s in _index()["scenarios"]:
        body = _scan(client, s["image"])
        assert [c["kind"] for c in body["scan"]["changes"]] == expected[s["name"]], s["name"]
        assert body["analysis"]["say"]


def test_fix_loop_walks_pickup_place_done(client):
    index = _index()
    body = _scan(client, "demo/mug_moved.jpg")
    change_id = body["analysis"]["items"][0]["id"]
    start = client.post("/api/homes/demo/fix", json={"change_id": change_id}).json()
    assert start["allowed"] and start["state"] == "pickup"

    states = []
    for frame in index["fix"]["frames"]:
        with open(ROOT / "web" / frame, "rb") as f:
            r = client.post(f"/api/fix/{start['session']}/frame", files={"image": ("f.jpg", f, "image/jpeg")})
        assert r.status_code == 200, r.text
        states.append(r.json()["state"])
    assert states == ["pickup", "place", "done"]


def test_rejects_featureless_keyframe(client):
    import cv2 as cv
    import numpy as np
    blank = np.full((480, 640, 3), 128, np.uint8)
    blank[::40] = 140
    ok, buf = cv.imencode(".jpg", blank)
    r = client.post("/api/homes/test/keyframes", files={"image": ("b.jpg", buf.tobytes(), "image/jpeg")},
                    data={"name": "wall"})
    assert r.status_code == 422


def test_rejects_path_traversal(client):
    assert client.get("/api/homes/..%2F..%2Fetc/keyframes").status_code in (400, 404)
    assert client.get("/api/homes/demo/keyframes/..%2Fx.jpg").status_code in (400, 404)
