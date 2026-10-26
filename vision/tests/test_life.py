"""World life: wind from the sky's streaks, the sun or moon, lamps at night, haze at the horizon."""
import cv2 as cv
import numpy as np

from vision.life import analyse

W, H = 2048, 1024


def _sky_mask():
    m = np.zeros((H, W), np.float32)
    m[: int(0.48 * H)] = 1
    return m


def test_wind_follows_horizontal_streaks():
    img = np.full((H, W, 3), (200, 160, 110), np.uint8)
    rng = np.random.default_rng(3)
    for _ in range(140):  # long thin clouds, drawn left to right
        x, y = int(rng.integers(0, W)), int(rng.integers(20, int(0.45 * H)))
        cv.line(img, (x, y), (x + int(rng.integers(150, 400)), y + int(rng.integers(-6, 6))), (245, 240, 235), 5)
    img[int(0.5 * H):] = (60, 130, 80)
    life = analyse(img, _sky_mask(), {"wind": "gusty"})
    assert abs(life["wind"]["angle_deg"]) < 12 and life["wind"]["strength"] > 0.6


def test_sun_is_found_where_it_is_painted():
    img = np.full((H, W, 3), (210, 170, 120), np.uint8)
    img[int(0.5 * H):] = (60, 130, 80)
    cv.circle(img, (1400, 230), 40, (220, 250, 255), -1)
    life = analyse(img, _sky_mask(), {"time": "day"})
    assert life["light"] is not None
    assert abs(life["light"]["u"] - 1400 / W) < 0.02 and abs(life["light"]["v"] - 230 / H) < 0.03


def test_overcast_sky_has_no_sun():
    img = np.full((H, W, 3), (235, 235, 235), np.uint8)
    img[int(0.5 * H):] = (60, 130, 80)
    assert analyse(img, _sky_mask(), {"time": "day"})["light"] is None


def test_lamps_at_night_but_not_a_blue_coat():
    img = np.full((H, W, 3), (60, 30, 20), np.uint8)  # night sky
    img[int(0.5 * H):] = (30, 40, 30)
    for x in (300, 700, 1100):
        cv.rectangle(img, (x, 650), (x + 18, 672), (60, 190, 255), -1)  # warm lit windows
    cv.rectangle(img, (1500, 600), (1560, 700), (230, 120, 40), -1)   # a bright blue coat
    lamps = analyse(img, _sky_mask(), {"time": "night"})["lamps"]
    xs = sorted(round(u * W) for u, v, s in lamps)
    assert len(lamps) == 3 and all(abs(a - b) < 20 for a, b in zip(xs, (309, 709, 1109)))


def test_haze_takes_the_horizon_colour():
    img = np.full((H, W, 3), (230, 190, 150), np.uint8)
    img[int(0.5 * H):] = (60, 130, 80)
    haze = analyse(img, _sky_mask(), {"weather": "fog"})["haze"]
    assert haze["amount"] == 0.35 and haze["color"] == [150, 190, 230]
