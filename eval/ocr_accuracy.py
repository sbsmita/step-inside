"""OCR accuracy of the OpenCV page reader on photographed pages (character and word error rates).

Usage: python -m eval.ocr_accuracy FOLDER
FOLDER holds pageN_photo.jpg files and ground_truth.json, as written by `python -m eval.make_pages`."""
import json
import sys
from pathlib import Path

import cv2 as cv

from vision.page import read_page


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main(folder: Path) -> dict:
    truth = json.loads((folder / "ground_truth.json").read_text())
    rows, ce, cn, we, wn = [], 0, 0, 0, 0
    for t in truth:
        page = read_page(cv.imread(str(folder / f"page{t['page']}_photo.jpg")))
        ref, hyp = t["text"], page.text
        c, w = edit_distance(list(ref), list(hyp)), edit_distance(ref.split(), hyp.split())
        ce, cn, we, wn = ce + c, cn + len(ref), we + w, wn + len(ref.split())
        rows.append({"page": t["page"], "cer": round(c / len(ref), 4), "wer": round(w / len(ref.split()), 4),
                     "ms": sum(page.timings_ms.values()), "flattened": bool(page.steps), "read": hyp})
    report = {"pages": rows, "cer": round(ce / cn, 4), "wer": round(we / wn, 4)}
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
