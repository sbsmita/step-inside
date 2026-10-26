"""Paper matte: art versus blank paper on a photographed page."""
import numpy as np

from vision.matte import paper_matte


def test_matte_ignores_uneven_light_on_the_paper():
    h, w = 400, 600
    shade = 1 - 0.3 * np.linspace(0, 1, w, dtype=np.float32)[None, :, None]  # a lamp on the left: paper darkens rightwards
    page = (np.array([235, 240, 245], np.float32) * shade * np.ones((h, 1, 1), np.float32)).astype(np.uint8)
    page[150:260, 230:370] = (60, 90, 200)  # the picture
    matte, full_bleed = paper_matte(page)
    assert not full_bleed
    assert (matte[150:260, 230:370] > 0).mean() > 0.95
    paper = np.ones((h, w), bool)
    paper[140:270, 220:380] = False
    assert (matte[paper] > 0).mean() < 0.02  # shaded paper is still paper
