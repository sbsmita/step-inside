"""Compose book pages from The Tale of Peter Rabbit (Beatrix Potter, 1902; public domain) and a "photographed"
version of each (perspective, lighting, table background), for OCR evaluation.

Usage: python -m eval.make_pages ILLUSTRATIONS_DIR OUT_DIR
Illustrations: https://www.gutenberg.org/files/14838/14838-h/images/peter08.jpg, peter11.jpg, peter19.jpg"""
import json
import sys
import textwrap
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Supplemental/Georgia.ttf"
PAGES = [
    ("peter08.jpg", "They lived with their Mother in a sand-bank, underneath the root of a very big fir-tree."),
    ("peter11.jpg", "\"Now, my dears,\" said old Mrs. Rabbit one morning, \"you may go into the fields or down "
                    "the lane, but don't go into Mr. McGregor's garden.\""),
    ("peter19.jpg", "But Peter, who was very naughty, ran straight away to Mr. McGregor's garden, and squeezed "
                    "under the gate!"),
]


def compose(illus: np.ndarray, text: str, w: int = 1000, h: int = 1400) -> np.ndarray:
    page = Image.new("RGB", (w, h), (246, 240, 226))
    pic = Image.fromarray(cv.cvtColor(illus, cv.COLOR_BGR2RGB))
    s = (w * 0.72) / pic.width
    pic = pic.resize((int(pic.width * s), int(pic.height * s)), Image.LANCZOS)
    page.paste(pic, ((w - pic.width) // 2, 90))
    draw = ImageDraw.Draw(page)
    font = ImageFont.truetype(FONT, 40)
    y = 90 + pic.height + 70
    for line in textwrap.wrap(text, 38):
        tw = draw.textlength(line, font=font)
        draw.text(((w - tw) / 2, y), line, fill=(40, 34, 30), font=font)
        y += 58
    return cv.cvtColor(np.array(page), cv.COLOR_RGB2BGR)


def photograph(page: np.ndarray, seed: int) -> np.ndarray:
    """Place the page on a wooden table at an angle, with uneven light, as a phone photo would be."""
    rng = np.random.default_rng(seed)
    H, W = 1600, 1200
    table = np.zeros((H, W, 3), np.uint8)
    table[:] = (60, 90, 130)
    for x in range(0, W, 9):
        cv.line(table, (x, 0), (x + 40, H), tuple(int(v) for v in rng.integers(45, 80, 3) + [0, 25, 50]), 2)
    h, w = page.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    j = lambda: rng.uniform(-60, 60)
    dst = np.float32([[150 + j(), 120 + j()], [1050 + j(), 150 + j()], [1080 + j(), 1480 + j()], [120 + j(), 1450 + j()]])
    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(page, M, (W, H))
    mask = cv.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    out = np.where(mask[..., None] > 0, warped, table)
    yy, xx = np.mgrid[0:H, 0:W]
    light = 0.85 + 0.25 * np.exp(-(((xx - W * 0.7) / 700) ** 2 + ((yy - H * 0.3) / 900) ** 2))
    return np.clip(out * light[..., None], 0, 255).astype(np.uint8)


def main(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    truth = []
    for i, (name, text) in enumerate(PAGES, 1):
        page = compose(cv.imread(str(src / name)), text)
        cv.imwrite(str(out / f"page{i}.png"), page)
        cv.imwrite(str(out / f"page{i}_photo.jpg"), photograph(page, seed=i), [cv.IMWRITE_JPEG_QUALITY, 90])
        truth.append({"page": i, "text": text})
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=1))
    print(f"wrote {len(PAGES)} pages to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
