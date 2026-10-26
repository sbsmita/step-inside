import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np

from .align import Features, features
from .detect import detect
from .pipeline import prepare

_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def safe_id(value: str) -> str:
    if not _SAFE.match(value):
        raise ValueError("ids may only contain letters, digits, '-' and '_' (max 40)")
    return value


@dataclass
class Keyframe:
    name: str
    image: np.ndarray
    feats: Features = field(repr=False)
    detections: list = field(default_factory=list)

    @classmethod
    def build(cls, name: str, image: np.ndarray) -> "Keyframe":
        return cls(name, image, features(image), detect(image))


class HomeStore:
    """Baseline keyframes per home on local disk, with SIFT features cached in memory."""

    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, dict[str, Keyframe]] = {}

    def _dir(self, home: str) -> Path:
        return self.root / safe_id(home)

    def add(self, home: str, name: str, img: np.ndarray) -> Keyframe:
        safe_id(name)
        image = prepare(img)
        d = self._dir(home)
        d.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(d / f"{name}.jpg"), image, [cv.IMWRITE_JPEG_QUALITY, 92])
        kf = Keyframe.build(name, image)
        self.keyframes(home)
        self._cache[home][name] = kf
        return kf

    def keyframes(self, home: str) -> list[Keyframe]:
        if home not in self._cache:
            frames = {}
            d = self._dir(home)
            for path in sorted(d.glob("*.jpg")) if d.exists() else []:
                frames[path.stem] = Keyframe.build(path.stem, cv.imread(str(path)))
            self._cache[home] = frames
        return list(self._cache[home].values())

    def image_path(self, home: str, name: str) -> Path:
        return self._dir(home) / f"{safe_id(name)}.jpg"

    def remove(self, home: str, name: str) -> None:
        self.image_path(home, name).unlink(missing_ok=True)
        self._cache.get(home, {}).pop(name, None)
