"""One shared password guards everything that makes a story (and costs money): uploads, the camera, video pages.
Watching stories needs no login. There are no user accounts: whoever has the password (the judges) can create.

- `APP_PASSWORD` (environment; on AWS it comes from Secrets Manager): the password. Unset = no login required
  (local development).
- Logging in sets an HttpOnly, SameSite=Strict cookie holding an expiry and an HMAC signature: nothing to steal
  from JavaScript, nothing stored on the server.
- Wrong passwords are slowed down per address (5 tries a minute)."""
import base64
import hashlib
import hmac
import os
import secrets
import threading
import time

from fastapi import HTTPException, Request

COOKIE = "si_session"
TTL_S = 7 * 24 * 3600
TRIES_PER_MIN = 5
_fails: dict[str, list[float]] = {}
_lock = threading.Lock()
_random_key = secrets.token_bytes(32)


def password() -> str | None:
    return os.environ.get("APP_PASSWORD") or None


def required() -> bool:
    return password() is not None


def _key() -> bytes:
    # Sessions survive a restart when SESSION_KEY is set; else they last until the server restarts.
    k = os.environ.get("SESSION_KEY")
    base = k.encode() if k else _random_key
    return hashlib.sha256(base + (password() or "").encode()).digest()  # a new password logs everyone out


def make_token(now: float | None = None) -> str:
    exp = str(int((now or time.time()) + TTL_S))
    sig = hmac.new(_key(), exp.encode(), hashlib.sha256).digest()
    return exp + "." + base64.urlsafe_b64encode(sig).decode().rstrip("=")


def valid(token: str | None, now: float | None = None) -> bool:
    if not token or "." not in token:
        return False
    exp, sig = token.split(".", 1)
    if not exp.isdigit() or int(exp) < (now or time.time()):
        return False
    good = base64.urlsafe_b64encode(hmac.new(_key(), exp.encode(), hashlib.sha256).digest()).decode().rstrip("=")
    return hmac.compare_digest(good, sig)


def logged_in(request: Request) -> bool:
    return not required() or valid(request.cookies.get(COOKIE))


def require(request: Request) -> None:
    """FastAPI dependency for every endpoint that makes something."""
    if not logged_in(request):
        raise HTTPException(401, "Log in to bring your own page to life.")


def client(request: Request) -> str:
    # Behind CloudFront and the load balancer the first X-Forwarded-For entry is the visitor.
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "?")


def check_password(request: Request, given: str) -> bool:
    who, now = client(request), time.time()
    with _lock:
        recent = [t for t in _fails.get(who, []) if now - t < 60]
        if len(recent) >= TRIES_PER_MIN:
            raise HTTPException(429, "Too many tries. Wait a minute.")
        ok = required() and hmac.compare_digest((given or "").encode(), password().encode())
        if not ok:
            recent.append(now)
        _fails[who] = recent
    return bool(ok)
