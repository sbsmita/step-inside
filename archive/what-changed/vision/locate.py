import math

DEFAULT_HFOV = 66.0
STEP_M = 0.7


def bearing_deg(x: float, width: int, hfov: float = DEFAULT_HFOV) -> float:
    """Horizontal angle from the camera axis to pixel column x (positive = right)."""
    half = math.radians(hfov / 2)
    return math.degrees(math.atan(((x - width / 2) / (width / 2)) * math.tan(half)))


def direction_phrase(bearing: float) -> str:
    a = abs(bearing)
    if a < 7:
        return "straight ahead"
    side = "right" if bearing > 0 else "left"
    return f"slightly to your {side}" if a < 20 else f"to your {side}"


def clock(bearing: float) -> str:
    hour = (12 + round(bearing / 30)) % 12 or 12
    return f"{hour} o'clock"


def height_phrase(y: float, height: int) -> str:
    r = y / height
    if r < 0.3:
        return "high up"
    if r > 0.72:
        return "low, near the floor"
    return "at chest height"


def distance_m(px_height: float, real_height_m: float, img_height: int, vfov_deg: float) -> float:
    f = (img_height / 2) / math.tan(math.radians(vfov_deg / 2))
    return real_height_m * f / max(px_height, 1.0)


def steps(meters: float) -> int:
    return max(1, round(meters / STEP_M))
