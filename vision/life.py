"""World life: what OpenCV reads from a painted 360 world so the theatre can make it feel alive.

- wind: the direction of the streaks in the sky (structure tensor of Sobel gradients), how strongly they agree, and
  the story's own wind; the sky then drifts along its own brush strokes;
- light: the sun or moon, as the brightest compact blob in the sky; the theatre puts a glow (and rays) there;
- lamps: at night, small warm bright spots below the sky (windows, lanterns, fires), which then flicker;
- haze: the colour of the sky at the horizon, for a band of mist where land meets sky.
Coordinates are fractions of the equirectangular image (u across, v down)."""
import cv2 as cv
import numpy as np

WIND = {"still": 0.2, "breeze": 0.5, "gusty": 1.0}
HAZE = {"fog": 0.35, "rain": 0.2, "snow": 0.2, "storm": 0.25}
MAX_LAMPS = 24


def wind(gray: np.ndarray, sky: np.ndarray, story_wind: str = "breeze", weather: str = "clear") -> dict:
    """Streak angle in degrees (0 = horizontal) and a strength 0..1."""
    g = gray.astype(np.float32)
    g = g - cv.GaussianBlur(g, (0, 0), 8)  # brush strokes, not the sky's gradient from horizon to zenith
    gx, gy = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3), cv.Sobel(g, cv.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = (cv.GaussianBlur(a, (0, 0), 15) for a in (gx * gx, gy * gy, gx * gy))
    sel = sky > 0.5
    base = WIND.get(story_wind, 0.5)
    if sel.sum() < 0.01 * sky.size:
        angle, coherence = 0.0, 0.0
    else:
        # The gradient's dominant orientation is across the streaks; the streaks run 90 degrees from it.
        theta = 0.5 * np.arctan2(2 * jxy[sel], jxx[sel] - jyy[sel]) + np.pi / 2
        coh = np.sqrt((jxx[sel] - jyy[sel]) ** 2 + 4 * jxy[sel] ** 2) / (jxx[sel] + jyy[sel] + 1e-6)
        # Average doubled angles (an orientation, not a direction), weighted by how clear each streak is.
        c, s = (coh * np.cos(2 * theta)).sum(), (coh * np.sin(2 * theta)).sum()
        angle = float(np.degrees(0.5 * np.arctan2(s, c)))
        angle = (angle + 90) % 180 - 90  # -90..90
        if abs(angle) > 45:  # vertical streaks are rain or brushwork, not wind: drift sideways anyway
            angle = 0.0
        coherence = float(coh.mean())
    strength = base * (0.6 + 0.4 * min(1.0, coherence))
    if weather == "storm":
        strength = max(strength, 0.9)
    return {"strength": round(float(strength), 3), "angle_deg": round(angle, 1), "coherence": round(coherence, 3)}


def light(img: np.ndarray, sky: np.ndarray, time: str = "day") -> dict | None:
    """The sun or moon: the brightest compact blob inside the sky (or near it)."""
    h, w = sky.shape
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    v = cv.GaussianBlur(hsv[..., 2].astype(np.float32), (0, 0), max(2.0, w / 160))
    near_sky = cv.dilate((sky > 0.5).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    if near_sky.sum() < 0.01 * sky.size:
        return None
    v_sky = np.where(near_sky, v, 0)
    _, peak, _, (x, y) = cv.minMaxLoc(v_sky)
    floor = 0.5 * 255 if time == "night" else 0.8 * 255
    if peak < floor or peak - np.median(v[near_sky]) < 25:
        return None
    blob = (v_sky >= 0.9 * peak).astype(np.uint8)
    n, labels, stats, _ = cv.connectedComponentsWithStats(blob)
    area = stats[labels[y, x], cv.CC_STAT_AREA] if n > 1 else blob.sum()
    if area > 0.006 * h * w:  # a bright overcast patch, not a sun or moon disc
        return None
    hue, sat = int(hsv[y, x, 0]), int(hsv[y, x, 1])
    warm = (sat > 40 and (hue < 25 or hue > 160)) or time in ("dusk", "dawn")
    return {"u": round(x / w, 4), "v": round(y / h, 4), "size": round(float(np.sqrt(area) / w), 4), "warm": bool(warm)}


def lamps(img: np.ndarray, sky: np.ndarray, time: str = "day") -> list[list[float]]:
    """Warm bright spots below the sky at night: windows, lanterns, fires. [[u, v, size], ...]"""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, w = sky.shape
    if time != "night" and np.median(hsv[..., 2]) >= 70:
        return []
    hue, sat, val = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    warm = ((hue < 35) | (hue > 165)) & (sat > 60) & (val > 200) & (sky < 0.5)
    warm = cv.morphologyEx(warm.astype(np.uint8), cv.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, cents = cv.connectedComponentsWithStats(warm)
    found = []
    for i in range(1, n):
        a = stats[i, cv.CC_STAT_AREA]
        if 0.0001 * h * w <= a <= 0.003 * h * w:
            found.append((a, [round(cents[i][0] / w, 4), round(cents[i][1] / h, 4), round(float(np.sqrt(a) / w), 4)]))
    found.sort(key=lambda t: -t[0])
    return [f for _, f in found[:MAX_LAMPS]]


def haze(img: np.ndarray, sky: np.ndarray, weather: str = "clear") -> dict:
    """Mist where land meets sky, coloured like the sky just above the horizon."""
    h = sky.shape[0]
    rows = np.zeros_like(sky, bool)
    rows[int(0.38 * h):int(0.5 * h)] = True
    sel = rows & (sky > 0.5)
    pix = img[sel] if sel.sum() > 50 else img[int(0.4 * h):int(0.5 * h)].reshape(-1, 3)
    b, g, r = np.median(pix, axis=0)
    return {"amount": HAZE.get(weather, 0.12), "color": [int(r), int(g), int(b)]}


def analyse(pano: np.ndarray, sky: np.ndarray, header: dict | None = None) -> dict:
    """Everything above for one world. `sky` is the soft 0..1 sky mask from `shot.motion_regions` (any size)."""
    header = header or {}
    time, weather = header.get("time", "day"), header.get("weather", "clear")
    small = cv.resize(pano, (1024, 512), interpolation=cv.INTER_AREA)
    sky = cv.resize(sky.astype(np.float32), (1024, 512), interpolation=cv.INTER_LINEAR)
    gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
    return {"wind": wind(gray, sky, header.get("wind", "breeze"), weather), "light": light(small, sky, time),
            "lamps": lamps(small, sky, time), "haze": haze(small, sky, weather), "time": time, "weather": weather}
