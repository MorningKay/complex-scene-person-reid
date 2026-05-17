"""VC-Clothes dataset parsing utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from PIL import Image

from reid.data.common import ReIDSample, SplitName

_FILENAME_RE = re.compile(
    r"^(?P<pid>\d+)[-_](?P<camid>\d+)[-_](?P<clothes>\d+)[-_](?P<image>\d+)\.jpg$"
)
_SPLIT_DIRS: dict[SplitName, str] = {
    "train": "train",
    "query": "query",
    "gallery": "gallery",
}


def parse_vc_clothes_filename(filename: str | Path) -> tuple[int, int, int]:
    name = Path(filename).name
    match = _FILENAME_RE.match(name)
    if match is None:
        raise ValueError(f"Invalid VC-Clothes filename: {name!r}")

    pid = int(match.group("pid"))
    camid = int(match.group("camid")) - 1
    clothes_id = int(match.group("clothes"))
    if camid < 0:
        raise ValueError(f"Invalid VC-Clothes camera id in filename: {name!r}")
    return pid, camid, clothes_id


def list_vc_clothes_split(root: str | Path, split: SplitName) -> list[ReIDSample]:
    if split not in _SPLIT_DIRS:
        valid = ", ".join(sorted(_SPLIT_DIRS))
        raise ValueError(f"Unknown VC-Clothes split {split!r}; expected one of: {valid}")

    split_dir = Path(root) / _SPLIT_DIRS[split]
    if not split_dir.is_dir():
        raise FileNotFoundError(f"VC-Clothes split directory not found: {split_dir}")

    samples: list[ReIDSample] = []
    for path in sorted(split_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".jpg":
            continue

        pid, camid, clothes_id = parse_vc_clothes_filename(path)
        samples.append(
            ReIDSample(
                path=path,
                pid=pid,
                camid=camid,
                metadata={"clothes_id": clothes_id},
            )
        )

    return samples


class VCClothesDataset:
    def __init__(
        self,
        root: str | Path,
        split: SplitName,
        transform: Callable[[Image.Image], object] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = list_vc_clothes_split(self.root, self.split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int, int, Path]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, sample.pid, sample.camid, sample.path
