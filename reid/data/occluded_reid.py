"""Occluded_REID dataset parsing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image

from reid.data.common import ReIDSample, SplitName

_SPLIT_DIRS: dict[SplitName, str] = {
    "query": "occluded_body_images",
    "gallery": "whole_body_images",
}
_SPLIT_CAMIDS: dict[SplitName, int] = {
    "query": 0,
    "gallery": 1,
}
_IMAGE_SUFFIXES = {".tif", ".tiff"}


def parse_occluded_reid_filename(path: str | Path) -> int:
    image_path = Path(path)
    try:
        pid = int(image_path.parent.name)
    except ValueError as exc:
        raise ValueError(f"Invalid Occluded_REID identity directory: {image_path}") from exc
    return pid


def list_occluded_reid_split(root: str | Path, split: SplitName) -> list[ReIDSample]:
    if split == "train":
        raise ValueError("Occluded_REID does not provide a train split")
    if split not in _SPLIT_DIRS:
        valid = ", ".join(sorted(_SPLIT_DIRS))
        raise ValueError(f"Unknown Occluded_REID split {split!r}; expected one of: {valid}")

    split_dir = Path(root) / _SPLIT_DIRS[split]
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Occluded_REID split directory not found: {split_dir}")

    camid = _SPLIT_CAMIDS[split]
    samples: list[ReIDSample] = []
    for path in sorted(split_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue

        pid = parse_occluded_reid_filename(path)
        samples.append(
            ReIDSample(
                path=path,
                pid=pid,
                camid=camid,
                metadata={"split_role": split},
            )
        )

    return samples


class OccludedREIDDataset:
    def __init__(
        self,
        root: str | Path,
        split: SplitName,
        transform: Callable[[Image.Image], object] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = list_occluded_reid_split(self.root, self.split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int, int, Path]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, sample.pid, sample.camid, sample.path
