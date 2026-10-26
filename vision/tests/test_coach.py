import cv2 as cv
import numpy as np

from vision.sfm.coach import CoachSession, step

W, H = 960, 720


def texture(seed: int, w: int = W + 400, h: int = H + 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 200, np.uint8)
    for _ in range(900):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv.circle(img, (x, y), int(rng.integers(3, 18)), rng.integers(0, 256, 3).tolist(), -1)
    return img


BG, FG = texture(1), texture(2, 420, 300)


def parallax_view(dx: int) -> np.ndarray:
    """Far wall shifts by dx, a near object by 3*dx: genuine depth parallax."""
    img = BG[100:100 + H, 200 + dx:200 + dx + W].copy()
    x = 500 - 3 * dx
    img[250:550, x:x + 420] = FG
    return img


def rotated_view(deg: float) -> np.ndarray:
    """Turning on the spot: the image change is exactly the homography K R K^-1."""
    f = 760.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    R = cv.Rodrigues(np.array([0.0, np.radians(deg), 0.0]))[0]
    base = BG[100:100 + H, 200:200 + W]
    return cv.warpPerspective(base, K @ R @ np.linalg.inv(K), (W, H), borderMode=cv.BORDER_REFLECT)


def test_sideways_sweep_produces_keyframes():
    s = CoachSession()
    results = [step(s, parallax_view(dx)) for dx in range(0, 150, 4)]
    assert results[0]["keyframe"]
    assert s.keyframes >= 4
    assert not any(r.get("rotation_only") for r in results)


def test_turning_on_the_spot_is_called_out():
    s = CoachSession()
    results = [step(s, rotated_view(a)) for a in np.arange(0, 12, 0.5)]
    assert any(r.get("rotation_only") for r in results)
    assert s.keyframes == 1  # only the first view; rotations add no 3D information


def test_blur_asks_to_hold_steady():
    s = CoachSession()
    r = step(s, cv.GaussianBlur(parallax_view(0), (0, 0), 7))
    assert r["status"] == "warn" and "blurry" in r["say"]
