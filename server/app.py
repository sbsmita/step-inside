import hmac
import os
import re
import time
import uuid
from pathlib import Path

import cv2 as cv
import numpy as np
from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from vision import scene360
from vision.look import PageIndex
from vision.look import check as check_view
from vision.scene25d import build
from vision.sfm.coach import TARGET_KEYFRAMES, CoachSession
from vision.sfm.coach import step as coach_step

from . import auth
from . import live as live_files
from .capture import Jobs, frames_from_upload
from .live import Live
from .scenes import SceneIndex, content_type
from .stories import Stories
from .storage import make_store

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(os.environ.get("WEB_DIR", ROOT / "web"))  # the container serves a minified copy
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
MAX_UPLOAD = 15 * 1024 * 1024
MAX_VIDEO = 200 * 1024 * 1024
MAX_PHOTOS = 80
KINDS = {"photo", "print", "page", "screenshot", "360"}
SAFE_ID = re.compile(r"^[a-f0-9]{12}$")

app = FastAPI(title="Step Inside", docs_url=None, redoc_url=None, openapi_url=None)  # no public API browser


def _csp() -> str:
    """Content Security Policy: scripts only from this site, three.js's CDN and the page's own import map (by hash)."""
    import base64
    import hashlib
    html = (WEB / "index.html").read_text()
    m = re.search(r'<script type="importmap">(.*?)</script>', html, re.S)
    sha = "'sha256-" + base64.b64encode(hashlib.sha256(m.group(1).encode()).digest()).decode() + "'" if m else ""
    return "; ".join([
        "default-src 'self'", f"script-src 'self' https://cdn.jsdelivr.net {sha}".strip(),
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com", "font-src https://fonts.gstatic.com",
        "img-src 'self' data: blob:", "media-src 'self' data: blob:", "connect-src 'self' blob: https://cdn.jsdelivr.net",
        "frame-ancestors 'none'", "base-uri 'none'", "form-action 'self'", "object-src 'none'"])


CSP = _csp()


@app.middleware("http")
async def security_headers(request: Request, call_next):
    # Deployed, only CloudFront (and its firewall) may reach the app: it adds a secret header the app checks, so
    # requests sent straight to the load balancer are refused. The load balancer's health check is let through.
    origin_secret = os.environ.get("ORIGIN_VERIFY")
    if origin_secret and request.url.path != "/api/health" and \
            not hmac.compare_digest(request.headers.get("x-origin-verify", ""), origin_secret):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    response = await call_next(request)
    h = response.headers
    h.setdefault("Content-Security-Policy", CSP)
    h.setdefault("X-Content-Type-Options", "nosniff")
    h.setdefault("X-Frame-Options", "DENY")
    h.setdefault("Referrer-Policy", "same-origin")
    h.setdefault("Permissions-Policy", "camera=(self), microphone=(), geolocation=(), xr-spatial-tracking=(self)")
    if request.headers.get("x-forwarded-proto", request.url.scheme) == "https":
        h.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    h.setdefault("Server", "step-inside")
    return response
store = make_store(DATA_DIR)
scenes = SceneIndex(store)
jobs = Jobs(store, scenes)
stories = Stories(store, scenes)
pages = PageIndex()
live = Live(store, on_ready=pages.add)  # a finished story's page is recognized from then on
coach_sessions: dict[str, CoachSession] = {}
MAX_PAGES = 12
MAX_PAGE_SIDE = 2048  # pages are read at 1280 px; a bigger photo would only take memory while it waits in the queue
BUSY = "Busy building other stories. Please try again in a few minutes."


def _decode(upload: UploadFile, max_side: int | None = None) -> np.ndarray:
    data = upload.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image too large (max 15 MB)")
    img = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise HTTPException(415, "could not decode image")
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
    return img


def _jpg(img: np.ndarray) -> bytes:
    return cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def _id(value: str) -> str:
    if not SAFE_ID.match(value):
        raise HTTPException(400, "bad id")
    return value


@app.get("/api/auth")
def auth_state(request: Request):
    """Whether a password is needed to make stories, and whether this browser has logged in."""
    return {"required": auth.required(), "ok": auth.logged_in(request)}


@app.post("/api/login")
def login(request: Request, password: str = Body(..., embed=True)):
    if not auth.required():
        return {"ok": True}
    if not auth.check_password(request, password):
        raise HTTPException(401, "That password is not right.")
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    response = JSONResponse({"ok": True})
    response.set_cookie(auth.COOKIE, auth.make_token(), max_age=auth.TTL_S, httponly=True, secure=secure,
                        samesite="strict", path="/")
    return response


@app.post("/api/logout")
def logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie(auth.COOKIE, path="/")
    return response


@app.get("/api/health")
def health():
    return {"ok": True, "opencv": cv.__version__, "pipeline": "step_functions" if jobs.sfn_arn else "local"}


@app.post("/api/scenes", dependencies=[Depends(auth.require)])
def create_scene(image: UploadFile = File(...), kind: str = Form("photo"), title: str = Form("")):
    if kind not in KINDS:
        raise HTTPException(400, f"kind must be one of {sorted(KINDS)}")
    img = _decode(image)
    if kind == "360":
        try:
            pano, meta = scene360.build(img)
        except ValueError as e:
            raise HTTPException(422, str(e))
        # The viewer wraps input.jpg on a sphere directly; no mesh needed.
        scene_id = scenes.save(meta, {"input.jpg": _jpg(pano)}, title, kind)
        jobs.after_scene(scene_id)
        return scenes.get(scene_id)
    scene = build(img, kind=kind)
    files = {"scene.glb": scene.glb, "input.jpg": _jpg(img)}
    files.update({f"{name}.jpg": _jpg(preview) for name, preview in scene.previews.items()})
    scene_id = scenes.save(scene.meta, files, title, kind)
    jobs.after_scene(scene_id)
    return scenes.get(scene_id)


@app.post("/api/captures", dependencies=[Depends(auth.require)])
def create_capture(video: UploadFile | None = File(None), photos: list[UploadFile] | None = File(None),
                   title: str = Form("")):
    """A short sweep video, or several photos of one place, becomes a real 3D reconstruction (background job)."""
    if video is not None:
        data = video.file.read(MAX_VIDEO + 1)
        if len(data) > MAX_VIDEO:
            raise HTTPException(413, "video too large (max 200 MB)")
        frames = frames_from_upload(data, video.filename or "capture.mp4")
        if not frames:
            raise HTTPException(415, "could not read any frames from the video")
    elif photos:
        if len(photos) > MAX_PHOTOS:
            raise HTTPException(413, f"at most {MAX_PHOTOS} photos")
        frames = [_decode(p) for p in photos]
    else:
        raise HTTPException(400, "send a video or photos")
    return jobs.submit(frames, title)


@app.post("/api/stories", dependencies=[Depends(auth.require)])
def create_story(pages: list[UploadFile] = File(...), title: str = Form("")):
    """Photos of book pages, in reading order, become a story to step into: one scene per page."""
    if len(pages) > MAX_PAGES:
        raise HTTPException(413, f"at most {MAX_PAGES} pages per story")
    if stories.busy():  # checked before decoding, and again when queueing
        raise HTTPException(429, BUSY)
    rec = stories.submit([_decode(p, MAX_PAGE_SIDE) for p in pages], title)
    if rec is None:
        raise HTTPException(429, BUSY)
    return rec


def best_frame(frames: list[np.ndarray]) -> tuple[np.ndarray | None, dict]:
    """Active perception on a video: OpenCV judges up to 40 evenly spaced frames (text found, readable size,
    sharpness, glare) and keeps the one it can read best."""
    step = max(1, len(frames) // 40)
    best, best_score, best_q = None, -1.0, {}
    for i in range(0, len(frames), step):
        q = check_view(frames[i])
        if q["lines"] < 2:
            continue
        score = q["ok"] * 1e6 + q["lines"] * 1e3 + min(q["sharpness"], 999) * min(1.0, q["text_px"] / 14)
        if score > best_score:
            best, best_score, best_q = frames[i], score, {**q, "frame": i, "frames": len(frames)}
    return best, best_q


@app.post("/api/live", dependencies=[Depends(auth.require)])
def create_live(page: UploadFile | None = File(None), video: UploadFile | None = File(None)):
    """A page (photo or video of it) becomes a story to live in, built by parallel agents; poll GET /api/live/{id}.
    A page that already has a story is recognized (ORB + homography) and that story is returned at once."""
    picked = None
    if video is not None:
        data = video.file.read(MAX_VIDEO + 1)
        if len(data) > MAX_VIDEO:
            raise HTTPException(413, "video too large (max 200 MB)")
        frames = frames_from_upload(data, video.filename or "page.mp4")
        img, picked = best_frame(frames) if frames else (None, None)
        if img is None:
            raise HTTPException(422, "Couldn't find a readable page in that video. Hold the page still, close up.")
        s = MAX_PAGE_SIDE / max(img.shape[:2])
        img = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else img
    elif page is not None:
        img = _decode(page, MAX_PAGE_SIDE)
    else:
        raise HTTPException(400, "send a page photo or a video")
    pages.refresh(store, every_s=5)
    hit = pages.match(img)
    if hit and hit["inliers"] >= 60 and (rec := live.get(hit["id"])):
        return {**rec, "reused": True}
    if live.busy():
        raise HTTPException(429, BUSY)
    rec = live.submit(img)
    if rec is None:
        raise HTTPException(429, BUSY)
    return {**rec, "picked_frame": picked} if picked else rec


@app.post("/api/look", dependencies=[Depends(auth.require)])
def look(image: UploadFile = File(...)):
    """One camera frame: is a readable page in view (and what to fix if not), and is it a page that already has a
    story? OpenCV only, about 0.1 s, so the browser can call it a few times a second."""
    frame = _decode(image, 1280)
    pages.refresh(store, every_s=5)
    return {"quality": check_view(frame), "match": pages.match(frame)}


@app.get("/api/live")
def list_live(limit: int = 24):
    return live.recent(max(1, min(limit, 100)))


@app.get("/api/live/{live_id}")
def get_live(live_id: str):
    rec = live.get(_id(live_id))
    if rec is None:
        raise HTTPException(404, "no such story")
    return rec


def ranged(data: bytes, media: str, range_header: str | None) -> Response:
    """A whole file, or the byte range a video player asked for (206), so clips can be streamed and seeked."""
    headers = {"Cache-Control": "public, max-age=86400", "Accept-Ranges": "bytes"}
    total = len(data)
    m = re.fullmatch(r"bytes=(\d*)-(\d*)", (range_header or "").strip())
    if not m or not (m.group(1) or m.group(2)):
        return Response(data, media_type=media, headers=headers)
    if m.group(1):
        start, end = int(m.group(1)), min(total - 1, int(m.group(2)) if m.group(2) else total - 1)
    else:
        start, end = max(0, total - int(m.group(2))), total - 1
    if start >= total or start > end:
        return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{total}"})
    return Response(data[start:end + 1], status_code=206, media_type=media,
                    headers={**headers, "Content-Range": f"bytes {start}-{end}/{total}"})


@app.get("/api/live/{live_id}/files/{name}")
def live_file(live_id: str, name: str, request: Request):
    media = live_files.content_type(name)
    data = live.file(_id(live_id), name) if media else None
    if data is None:
        raise HTTPException(404)
    return ranged(data, media, request.headers.get("range"))


@app.get("/api/stories")
def list_stories(limit: int = 24):
    return stories.recent(max(1, min(limit, 100)))


@app.get("/api/stories/{story_id}")
def get_story(story_id: str):
    rec = stories.get(_id(story_id))
    if rec is None:
        raise HTTPException(404, "no such story")
    return rec


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(_id(job_id))
    if job is None:
        raise HTTPException(404, "no such job")
    return job


@app.post("/api/coach", dependencies=[Depends(auth.require)])
def start_coach():
    now = time.time()
    for sid in [k for k, v in coach_sessions.items() if now - v.started > 1800]:
        coach_sessions.pop(sid, None)
    sid = uuid.uuid4().hex[:12]
    coach_sessions[sid] = CoachSession()
    return {"session": sid, "target_keyframes": TARGET_KEYFRAMES}


@app.post("/api/coach/{sid}/frame", dependencies=[Depends(auth.require)])
def coach_frame(sid: str, image: UploadFile = File(...)):
    session = coach_sessions.get(_id(sid))
    if session is None:
        raise HTTPException(404, "no such coach session")
    return coach_step(session, _decode(image))


@app.get("/api/scenes")
def list_scenes(limit: int = 24):
    return scenes.list(max(1, min(limit, 100)))


@app.get("/api/scenes/{scene_id}")
def get_scene(scene_id: str):
    meta = scenes.get(_id(scene_id))
    if meta is None:
        raise HTTPException(404, "no such scene")
    return meta


@app.get("/api/scenes/{scene_id}/files/{name}")
def scene_file(scene_id: str, name: str):
    media = content_type(name)
    if media is None:
        raise HTTPException(404)
    data = store.get(f"scenes/{_id(scene_id)}/{name}")
    if data is None:
        raise HTTPException(404)
    return Response(data, media_type=media, headers={"Cache-Control": "public, max-age=86400"})


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
