"""MSMT17_V1 dataset parsing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image

from reid.data.common import ReIDSample, SplitName

_SPLIT_LISTS: dict[SplitName, tuple[str, ...]] = {
    "train": ("list_train.txt", "list_val.txt"),
    "query": ("list_query.txt",),
    "gallery": ("list_gallery.txt",),
}
_SPLIT_IMAGE_DIRS: dict[SplitName, str] = {
    "train": "train",
    "query": "test",
    "gallery": "test",
}


def parse_msmt17_filename(filename: str | Path) -> int:
    name = Path(filename).name
    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid MSMT17 filename: {name!r}")

    try:
        camid = int(parts[2]) - 1
    except ValueError as exc:
        raise ValueError(f"Invalid MSMT17 camera id in filename: {name!r}") from exc
    if camid < 0:
        raise ValueError(f"Invalid MSMT17 camera id in filename: {name!r}")
    return camid


def list_msmt17_split(root: str | Path, split: SplitName) -> list[ReIDSample]:
    if split not in _SPLIT_LISTS:
        valid = ", ".join(sorted(_SPLIT_LISTS))
        raise ValueError(f"Unknown MSMT17 split {split!r}; expected one of: {valid}")

    root_path = Path(root)
    image_dir = root_path / _SPLIT_IMAGE_DIRS[split]
    if not image_dir.is_dir():
        raise FileNotFoundError(f"MSMT17 image directory not found: {image_dir}")

    samples: list[ReIDSample] = []
    for list_name in _SPLIT_LISTS[split]:
        list_path = root_path / list_name
        if not list_path.is_file():
            raise FileNotFoundError(f"MSMT17 list file not found: {list_path}")

        with list_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                parts = stripped.split()
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid MSMT17 list entry at {list_path}:{line_number}: {stripped!r}"
                    )
                relative_path, pid_text = parts
                try:
                    pid = int(pid_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid MSMT17 pid at {list_path}:{line_number}: {pid_text!r}"
                    ) from exc

                path = image_dir / relative_path
                camid = parse_msmt17_filename(relative_path)
                samples.append(
                    ReIDSample(
                        path=path,
                        pid=pid,
                        camid=camid,
                        metadata={"relative_path": relative_path, "source_list": list_name},
                    )
                )

    return samples


class MSMT17Dataset:
    def __init__(
        self,
        root: str | Path,
        split: SplitName,
        transform: Callable[[Image.Image], object] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = list_msmt17_split(self.root, self.split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int, int, Path]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, sample.pid, sample.camid, sample.path
