import cv2 as cv
import numpy as np
import pytest

from vision.align import align
from vision.pipeline import compare
from vision.quality import assess, motion
from vision.synth import VIEW_H, VIEW_W, bbox_to_view, iou, make_canvas, place_object, relight, view

FLOOR_A = (420, 760)
FLOOR_B = (900, 700)


def _box_px(change: dict, key: str = "box_baseline") -> tuple:
    pts = np.array(change[key]) * [VIEW_W, VIEW_H]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return x0, y0, x1 - x0, y1 - y0


@pytest.fixture(scope="module")
def canvas():
    return make_canvas(seed=3)


def test_alignment_recovers_true_homography(canvas):
    base, Mb = view(canvas)
    cur, Mc = view(canvas, jitter=0.04, seed=7)
    al = align(base, cur)
    assert al.ok and al.inliers > 100
    true_H = Mb @ np.linalg.inv(Mc)
    pts = np.float64([[100, 100], [860, 100], [860, 620], [100, 620]]).reshape(-1, 1, 2)
    err = np.linalg.norm(cv.perspectiveTransform(pts, al.H) - cv.perspectiveTransform(pts, true_H), axis=2)
    assert err.max() < 2.0


def test_lighting_and_viewpoint_change_alone_is_not_a_change(canvas):
    base, _ = view(canvas)
    cur, _ = view(canvas, jitter=0.03, seed=11)
    result = compare(base, relight(cur))
    assert result["status"] == "ok"
    assert result["changes"] == []


def test_new_object_is_detected_where_it_is(canvas):
    with_mug, bbox = place_object(canvas, *FLOOR_A)
    base, Mb = view(canvas)
    cur, _ = view(with_mug, jitter=0.03, seed=5)
    result = compare(base, cur)
    assert result["status"] == "ok"
    assert len(result["changes"]) == 1
    change = result["changes"][0]
    assert change["kind"] in ("appeared", "changed")
    assert iou(_box_px(change), bbox_to_view(bbox, Mb)) > 0.4
    assert change["direction"] == "slightly to your left"
    assert change["height"] == "low, near the floor"


def test_white_object_on_floor_is_detected_despite_clipping(canvas):
    paper = canvas.copy()
    cv.rectangle(paper, (520, 700), (660, 780), (255, 255, 255), -1)
    base, _ = view(canvas)
    cur, _ = view(paper, jitter=0.03, seed=4)
    assert len(compare(base, cur)["changes"]) == 1


def test_removed_object_is_reported_missing(canvas):
    with_mug, bbox = place_object(canvas, *FLOOR_A)
    base, Mb = view(with_mug)
    cur, _ = view(canvas, jitter=0.03, seed=9)
    result = compare(base, cur)
    assert [c["kind"] for c in result["changes"]] == ["disappeared"]
    assert iou(_box_px(result["changes"][0]), bbox_to_view(bbox, Mb)) > 0.4


def test_moved_object_is_paired_into_one_change(canvas):
    at_a, bbox_a = place_object(canvas, *FLOOR_A)
    at_b, bbox_b = place_object(canvas, *FLOOR_B)
    base, Mb = view(at_a)
    cur, _ = view(at_b, jitter=0.03, seed=13)
    result = compare(base, cur)
    moved = [c for c in result["changes"] if c["kind"] == "moved"]
    assert len(moved) == 1
    assert iou(_box_px(moved[0]), bbox_to_view(bbox_b, Mb)) > 0.4
    assert iou(_box_px(moved[0], "from_box_baseline"), bbox_to_view(bbox_a, Mb)) > 0.4


def test_unrecognized_view_reports_lost(canvas):
    base, _ = view(canvas)
    noise = np.random.default_rng(0).integers(0, 256, (VIEW_H, VIEW_W, 3), dtype=np.uint8)
    assert compare(base, noise)["status"] == "lost"


def test_quality_flags_blur_dark_and_glare(canvas):
    img, _ = view(canvas)
    assert assess(img)["ok"]
    assert "blurry" in assess(cv.GaussianBlur(img, (0, 0), 6))["issues"]
    assert "too_dark" in assess((img * 0.15).astype(np.uint8))["issues"]
    glare = img.copy()
    glare[100:400, 200:600] = 255
    assert "glare" in assess(glare)["issues"]


def test_motion_is_high_for_large_shift(canvas):
    a, _ = view(canvas)
    assert motion(a, a) < 0.002
    assert motion(a, np.roll(a, 60, axis=1)) > 0.02
