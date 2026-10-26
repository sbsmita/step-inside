"""Render the synthetic room along its camera path with headless Chrome; writes frames + ground-truth poses.

Usage: python -m eval.synth_room.render OUT_DIR [--frames 40]"""
import argparse
import base64
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(HERE)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/render.html?frames={args.frames}"
    poses = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True,
                                    args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 960, "height": 720})
        page.goto(url)
        page.wait_for_function("() => window.ready === true", timeout=60000)
        meta = page.evaluate("window.meta")
        for i in range(args.frames):
            r = page.evaluate(f"window.render({i})")
            (args.out / f"frame_{i:03d}.png").write_bytes(base64.b64decode(r["png"].split(",", 1)[1]))
            poses.append({"frame": i, "position": r["position"], "quaternion": r["quaternion"]})
        browser.close()
    server.shutdown()
    (args.out / "ground_truth.json").write_text(json.dumps({"meta": meta, "poses": poses}, indent=1))
    print(f"wrote {args.frames} frames to {args.out}")


if __name__ == "__main__":
    main()
