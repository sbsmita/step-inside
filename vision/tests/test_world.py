"""The 360 world around a page: tile bookkeeping, painter failures, the stray-figure scan, the protect mask."""
import numpy as np

from agents import llm, storyteller
from vision.scene25d import foreground_layers
from vision.world import (AI_H, AI_W, fill_unpainted, flat_fraction, paint_tiled, protect_mask, stray_figures,
                          tile_order)

MAGENTA = (255, 0, 255)


def _canvas():
    """A random 'colour guide' with a kept rectangle (the real art) in the middle, at the real painting size."""
    canvas = np.random.default_rng(1).integers(0, 255, (AI_H, AI_W, 3), np.uint8)
    keep = np.zeros((AI_H, AI_W), np.uint8)
    keep[AI_H // 4: 3 * AI_H // 4, AI_W // 2 - 340: AI_W // 2 + 340] = 255
    return canvas, 255 - keep


def _textured(crop, mask, colour=(0, 0, 255), seed=3):
    """Paints scenery-like texture where the mask asks for it: structure at the scale of leaves and stones (16-px
    blotches), not just pixel grain, which is what separates scenery from an empty field with paper texture."""
    out = crop.copy()
    h, w = crop.shape[:2]
    blotches = np.random.default_rng(seed).integers(-60, 60, (h // 16 + 1, w // 16 + 1, 3), np.int16)
    blotches = np.repeat(np.repeat(blotches, 16, axis=0), 16, axis=1)[:h, :w]
    out[mask > 0] = np.clip(np.array(colour, np.int16) + blotches, 0, 255).astype(np.uint8)[mask > 0]
    return out


def _flat(crop, mask, colour=(0, 0, 255)):
    """What a failed painter does: hands back an empty flat field."""
    out = crop.copy()
    out[mask > 0] = colour
    return out


def test_every_generated_pixel_is_painted_at_full_strength():
    canvas, generate = _canvas()
    pano, stats, todo = paint_tiled(canvas, generate, _flat, judge=False)
    assert stats["calls"] == len(tile_order(AI_W)) == 5 and not todo.any()
    # No half-blended bands of the guide at tile boundaries, and the kept art is untouched (to rounding).
    assert np.abs(pano[generate > 0].astype(int) - (0, 0, 255)).max() <= 1
    assert np.abs(pano[generate == 0].astype(int) - canvas[generate == 0]).max() <= 1


def test_a_failed_call_is_retried():
    canvas, generate = _canvas()
    n = {"calls": 0}

    def flaky(crop, mask):
        n["calls"] += 1
        return None if n["calls"] == 1 else _textured(crop, mask)

    _, stats, todo = paint_tiled(canvas, generate, flaky)
    assert stats["calls"] == 5 and stats["failed_calls"] == 1 and n["calls"] == 6 and not todo.any()


def test_painting_that_stops_early_reports_the_gap_and_the_fallback_fills_it():
    canvas, generate = _canvas()
    n = {"calls": 0}

    def dies(crop, mask):
        n["calls"] += 1
        return _textured(crop, mask) if n["calls"] <= 2 else None

    pano, stats, todo = paint_tiled(canvas, generate, dies)
    assert stats["calls"] == 2 and n["calls"] == 5  # two tiles painted, then three failed attempts at the third
    assert todo.any() and not (todo & (generate == 0)).any()
    filled = fill_unpainted(pano, todo, np.full_like(canvas, 7))
    assert (filled[todo > 0] == 7).all()  # the clean surround, never the noisy guide


def test_a_flat_empty_tile_is_painted_again_and_the_textured_one_kept():
    canvas, generate = _canvas()
    n = {"calls": 0}

    def flat_once(crop, mask):
        n["calls"] += 1
        return _flat(crop, mask, (90, 120, 140)) if n["calls"] == 1 else _textured(crop, mask, (90, 120, 140))

    pano, stats, todo = paint_tiled(canvas, generate, flat_once)
    assert stats["flat_tiles"] == 1 and stats["calls"] == 6 and not todo.any()
    assert flat_fraction(pano, generate > 0) < 0.05  # the empty attempt was replaced, not kept
    # A painter that only ever returns flat colour still gets its world: the least flat attempt is kept.
    pano, stats, _ = paint_tiled(canvas, generate, _flat)
    assert stats["flat_tiles"] == 15 and stats["calls"] == 15  # three attempts per tile, all flat
    assert np.abs(pano[generate > 0].astype(int) - (0, 0, 255)).max() <= 1


def _fake_detect(crop):
    """Finds a magenta figure only when the crop shows it whole, like a detector that needs the full figure."""
    ys, xs = np.nonzero((crop == MAGENTA).all(axis=2))
    if len(xs) == 0:
        return []
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    if x0 == 0 or y0 == 0 or x1 == crop.shape[1] or y1 == crop.shape[0]:
        return []
    return [{"name": "person", "score": 0.9, "box": (x0, y0, x1, y1)}]


def _covers(box, rect):
    return box[0] <= rect[0] and box[1] <= rect[1] and box[2] >= rect[2] and box[3] >= rect[3]


def test_stray_scan_sees_big_figures_whole_and_across_the_join():
    pano = np.zeros((AI_H, AI_W, 3), np.uint8)
    keep = np.zeros((AI_H, AI_W), np.uint8)
    keep[800:1300, 1900:2200] = 255
    big, small, kept = (600, 1300, 750, 1750), (2600, 1000, 2660, 1120), (1950, 900, 2000, 1000)
    for x0, y0, x1, y1 in (big, small, kept):
        pano[y0:y1, x0:x1] = MAGENTA
    pano[1200:1400, AI_W - 60:] = MAGENTA  # a figure cut by the 0/360 join...
    pano[1200:1400, :60] = MAGENTA         # ...continues on the other edge
    boxes = stray_figures(pano, keep, _fake_detect)
    assert len(boxes) == 3  # the big one (too tall for a 512 tile), the small one, the one on the join
    assert any(_covers(b, big) for b in boxes) and any(_covers(b, small) for b in boxes)
    assert any(_covers(b, (AI_W - 60, 1200, AI_W + 60, 1400)) for b in boxes)
    assert not any(_covers(b, kept) for b in boxes)  # the book's own character is left alone


def test_colour_guide_uses_the_arts_palette_at_the_right_heights():
    from vision.world import ai_canvas
    illus = np.full((600, 600, 3), 242, np.uint8)  # paper
    illus[100:300, 100:500] = (200, 210, 170)  # pale sky in the top half of the art
    illus[300:500, 100:500] = (40, 120, 60)    # green ground in the bottom half
    matte = np.zeros((600, 600), np.uint8)
    matte[100:500, 100:500] = 255
    prep = ai_canvas(illus, matte, 60.0)
    _, y0, _, y1 = prep["box"]
    outside = prep["keep"] == 0
    top = prep["canvas"][: y0 - 64][outside[: y0 - 64]].mean(axis=0)
    bottom = prep["canvas"][y1 + 64:][outside[y1 + 64:]].mean(axis=0)
    assert np.abs(top - (200, 210, 170)).max() < 30 and np.abs(bottom - (40, 120, 60)).max() < 30
    assert np.abs(prep["canvas"][outside].mean(axis=0) - 242).max() > 40  # never the paper's colour


def test_protect_mask_keeps_the_outline_not_its_hull():
    matte = np.zeros((200, 200), np.uint8)
    layer = np.zeros((200, 200), bool)
    layer[40:160, 40:70] = layer[40:70, 40:160] = layer[130:160, 40:160] = True  # a "C"
    keep = protect_mask(matte, [layer])
    assert keep[40:160, 40:70].all() and keep[100, 120] == 0  # the C's mouth is blank paper, not kept


def test_depth_layers_that_vanish_in_the_cleanup_are_dropped():
    inv_n = np.zeros((480, 640), np.float32)
    inv_n[::6, :] = inv_n[:, ::6] = 1.0  # a 1-px lattice pops out in depth but can't survive the opening
    layers = foreground_layers(np.zeros((480, 640, 3), np.uint8), inv_n, [])
    assert all(l["mask"].any() and np.isfinite(l["closeness"]) for l in layers)


def test_storyteller_schema_closes_every_object():
    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for v in node.values():
                yield from objects(v)
    found = list(objects(storyteller.SCHEMA))
    assert len(found) == 3 and all(o.get("additionalProperties") is False for o in found)


def test_storyteller_tolerates_sloppy_model_output(monkeypatch):
    text = "Once upon a time there were four little Rabbits."
    monkeypatch.setattr(llm, "structured", lambda *a, **k: {
        "title": "Four Rabbits", "setting": "A sand-bank.",
        "beats": [{"text": text}, "not a beat", {"speaker": "Peter", "text": "Words that are not on the page."}],
        "characters": [{"name": "Peter"}, {"name": "Mrs. Rabbit", "voice": "woman"}, "junk"],
        "world_prompt": "a sandy bank under fir roots", "negative_prompt": "people"})
    out, source = storyteller.adapt(text, [], [])
    assert source == "bedrock"
    assert out["beats"] == [{"speaker": "Narrator", "text": text}]  # the invented beat is gone
    assert out["characters"] == []  # no voice, or not a speaker: dropped, not a crash
    assert out["negative_prompt"].startswith(storyteller.DEFAULT_NEGATIVE)
