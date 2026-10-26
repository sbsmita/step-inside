"""Built-in demo home so anyone can try the app without a camera."""
import json
from pathlib import Path

import cv2 as cv

from vision.store import HomeStore
from vision.synth import add_spill, make_canvas, place_object, place_shoe, relight, view

DEMO_HOME = "demo"
WEB_DEMO = Path(__file__).resolve().parent.parent / "web" / "demo"
MUG_HOME = (900, 700)
MUG_AWAY = (430, 690)


def _scenes():
    living = make_canvas(seed=3)
    living_base, _ = place_object(living, *MUG_HOME)
    kitchen = make_canvas(seed=8)

    mug_away, _ = place_object(living, *MUG_AWAY)
    shoe, _ = place_shoe(living_base, 380, 760)
    spill, _ = add_spill(kitchen, 560, 720)

    keyframes = {"living_room": view(living_base)[0], "kitchen": view(kitchen)[0]}
    scenarios = [
        ("lights_on", "Lamp switched on, nothing moved", relight(view(living_base, 0.03, 21)[0])),
        ("shoe_on_floor", "A shoe left in the walkway", view(shoe, 0.03, 22)[0]),
        ("mug_moved", "The mug was moved", view(mug_away, 0.03, 23)[0]),
        ("mug_missing", "The mug is gone", view(living, 0.03, 24)[0]),
        ("kitchen_spill", "Something spilled in the kitchen", view(spill, 0.03, 25)[0]),
    ]
    fix_frames = [
        ("fix_1_still_away", view(mug_away, 0.02, 31)[0]),
        ("fix_2_picked_up", view(living, 0.02, 32)[0]),
        ("fix_3_put_back", view(living_base, 0.02, 33)[0]),
    ]
    return keyframes, scenarios, fix_frames


def ensure_demo_home(store: HomeStore) -> None:
    if store.keyframes(DEMO_HOME) and (WEB_DEMO / "index.json").exists():
        return
    keyframes, scenarios, fix_frames = _scenes()
    for name, img in keyframes.items():
        store.add(DEMO_HOME, name, img)

    WEB_DEMO.mkdir(parents=True, exist_ok=True)
    for name, _, img in scenarios:
        cv.imwrite(str(WEB_DEMO / f"{name}.jpg"), img, [cv.IMWRITE_JPEG_QUALITY, 90])
    for name, img in fix_frames:
        cv.imwrite(str(WEB_DEMO / f"{name}.jpg"), img, [cv.IMWRITE_JPEG_QUALITY, 90])
    index = {
        "home": DEMO_HOME,
        "scenarios": [{"name": n, "title": t, "image": f"demo/{n}.jpg"} for n, t, _ in scenarios],
        "fix": {"scenario": "mug_moved", "frames": [f"demo/{n}.jpg" for n, _ in fix_frames]},
    }
    (WEB_DEMO / "index.json").write_text(json.dumps(index, indent=2))
