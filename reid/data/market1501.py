"""Market-1501 dataset parsing utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from PIL import Image

from reid.data.common import ReIDSample, SplitName

_FILENAME_RE = re.compile(r"^(?P<pid>-?\d+)_c(?P<camid>\d+)s\d+_\d+_\d+(?:\.jpg)+$")
_SPLIT_DIRS: dict[SplitName, str] = {
    "train": "bounding_box_train",
    "query": "query",
    "gallery": "bounding_box_test",
}


Market1501Sample = ReIDSample


def parse_market1501_filename(filename: str | Path) -> tuple[int, int]:
    """Return `(pid, camid)` parsed from a Market-1501 image filename.

    Market-1501 camera IDs are encoded from 1, so the returned `camid` is
    converted to zero-based indexing for use in training and evaluation code.
    """

    name = Path(filename).name
    match = _FILENAME_RE.match(name)
    if match is None:
        raise ValueError(f"Invalid Market-1501 filename: {name!r}")

    pid = int(match.group("pid"))
    camid = int(match.group("camid")) - 1
    if camid < 0:
        raise ValueError(f"Invalid Market-1501 camera id in filename: {name!r}")

    return pid, camid


def list_market1501_split(root: str | Path, split: SplitName) -> list[Market1501Sample]:
    if split not in _SPLIT_DIRS:
        valid = ", ".join(sorted(_SPLIT_DIRS))
        raise ValueError(f"Unknown Market-1501 split {split!r}; expected one of: {valid}")

    split_dir = Path(root) / _SPLIT_DIRS[split]
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Market-1501 split directory not found: {split_dir}")

    samples: list[Market1501Sample] = []
    for path in sorted(split_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".jpg":
            continue

        pid, camid = parse_market1501_filename(path.name)
        samples.append(ReIDSample(path=path, pid=pid, camid=camid))

    return samples


class Market1501Dataset:
    def __init__(
        self,
        root: str | Path,
        split: SplitName,
        transform: Callable[[Image.Image], object] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = list_market1501_split(self.root, self.split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int, int, Path]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, sample.pid, sample.camid, sample.path
