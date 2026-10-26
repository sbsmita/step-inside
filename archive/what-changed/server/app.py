import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2 as cv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents import fixit, llm, orchestrator
from vision.align import features
from vision.pipeline import compare, decode, scan
from vision.quality import assess
from vision.store import HomeStore, safe_id

from .demo import ensure_demo_home

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
MAX_UPLOAD = 12 * 1024 * 1024
MIN_KEYFRAME_FEATURES = 150

store = HomeStore(DATA_DIR / "homes")
sessions: dict[str, fixit.FixSession] = {}
last_scans: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_demo_home(store)
    yield


app = FastAPI(title="What Changed?", lifespan=lifespan)


def _read(upload: UploadFile):
    data = upload.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image too large (max 12 MB)")
    try:
        return decode(data)
    except ValueError:
        raise HTTPException(415, "could not decode image")


def _home(home: str) -> str:
    try:
        return safe_id(home)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/health")
def health():
    return {"ok": True, "opencv": cv.__version__, "llm": llm.MODEL if llm.enabled() else "off"}


@app.post("/api/compare")
def compare_images(baseline: UploadFile = File(...), current: UploadFile = File(...), hfov: float = Form(66.0)):
    result = compare(_read(baseline), _read(current), hfov=hfov)
    return {"scan": result, "analysis": orchestrator.analyze("upload", result)}


@app.get("/api/homes/{home}/keyframes")
def list_keyframes(home: str):
    home = _home(home)
    return [{"name": kf.name, "url": f"/api/homes/{home}/keyframes/{kf.name}.jpg"} for kf in store.keyframes(home)]


@app.post("/api/homes/{home}/keyframes")
def add_keyframe(home: str, image: UploadFile = File(...), name: str = Form(...)):
    home = _home(home)
    try:
        safe_id(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    img = _read(image)
    quality = assess(img)
    if not quality["ok"]:
        raise HTTPException(422, {"reason": "quality", "quality": quality})
    n = len(features(img).keypoints)
    if n < MIN_KEYFRAME_FEATURES:
        raise HTTPException(422, {"reason": "featureless", "features": n,
                                  "say": "This view is too plain for me to recognize later. Include more of the room."})
    store.add(home, name, img)
    return {"name": name, "features": n, "quality": quality}


@app.get("/api/homes/{home}/keyframes/{name}.jpg")
def keyframe_image(home: str, name: str):
    try:
        path = store.image_path(_home(home), name)
    except ValueError:
        raise HTTPException(400, "bad name")
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


@app.delete("/api/homes/{home}/keyframes/{name}")
def delete_keyframe(home: str, name: str):
    try:
        store.remove(_home(home), name)
    except ValueError:
        raise HTTPException(400, "bad name")
    return {"deleted": name}


@app.post("/api/homes/{home}/scan")
def scan_home(home: str, image: UploadFile = File(...), hfov: float = Form(66.0), motion: float | None = Form(None)):
    home = _home(home)
    keyframes = store.keyframes(home)
    if not keyframes:
        raise HTTPException(409, "this home has no baseline yet")
    result = scan(keyframes, _read(image), hfov=hfov)
    analysis = orchestrator.analyze(home, result, motion=motion)
    last_scans[home] = {"scan": result, "analysis": analysis}
    return {"scan": result, "analysis": analysis}


class FixRequest(BaseModel):
    change_id: int


@app.post("/api/homes/{home}/fix")
def start_fix(home: str, req: FixRequest):
    home = _home(home)
    last = last_scans.get(home)
    item = next((i for i in (last or {}).get("analysis", {}).get("items", []) if i["id"] == req.change_id), None)
    if item is None:
        raise HTTPException(404, "no such change in the latest scan")
    verdict = item["verdict"]
    gate = fixit.safety_gate(verdict)
    if not gate["allowed"]:
        return {"allowed": False, "say": gate["say"]}
    lines, source = fixit.plan(item, verdict)
    session = fixit.FixSession(home=home, keyframe=last["scan"]["keyframe"], change=item, verdict=verdict, lines=lines)
    session.trace.append({"agent": "planner", "source": source, "lines": lines})
    sessions[session.id] = session
    return {"allowed": True, "session": session.id, "state": session.state, "lines": lines}


@app.post("/api/fix/{session_id}/frame")
def fix_frame(session_id: str, image: UploadFile = File(...), hfov: float = Form(66.0)):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "unknown fix session")
    keyframe = next((k for k in store.keyframes(session.home) if k.name == session.keyframe), None)
    if keyframe is None:
        raise HTTPException(410, "baseline keyframe was removed")
    result = fixit.step(session, keyframe, _read(image), hfov=hfov)
    if result["done"]:
        result["trace"] = session.trace
        sessions.pop(session_id, None)
    return result


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
