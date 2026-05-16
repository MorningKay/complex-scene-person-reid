"""DataLoader helpers for Market-1501."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from reid.data.market1501 import Market1501Dataset, SplitName
from reid.data.transforms import ImageSize, build_eval_transform, build_train_transform

Market1501BatchItem = tuple[torch.Tensor, int, int, str | Path]


def market1501_collate(
    batch: Sequence[Market1501BatchItem],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    if not batch:
        raise ValueError("Cannot collate an empty Market-1501 batch")

    images, pids, camids, paths = zip(*batch, strict=True)
    return (
        torch.stack(tuple(images), dim=0).float(),
        torch.as_tensor(pids, dtype=torch.long),
        torch.as_tensor(camids, dtype=torch.long),
        tuple(str(path) for path in paths),
    )


def build_market1501_dataloader(
    root: str | Path,
    split: SplitName,
    batch_size: int,
    image_size: ImageSize = (256, 128),
    random_erasing: bool = False,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    transform = (
        build_train_transform(image_size=image_size, random_erasing=random_erasing)
        if split == "train"
        else build_eval_transform(image_size=image_size)
    )
    dataset = Market1501Dataset(root=root, split=split, transform=transform)

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=market1501_collate,
    )
