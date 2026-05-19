"""DataLoader helpers for Re-ID datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import torch
from torch.utils.data import DataLoader

from reid.data.common import SplitName
from reid.data.market1501 import Market1501Dataset
from reid.data.msmt17 import MSMT17Dataset
from reid.data.occluded_reid import OccludedREIDDataset
from reid.data.samplers import PKBatchSampler
from reid.data.transforms import ImageSize, build_eval_transform, build_train_transform
from reid.data.vc_clothes import VCClothesDataset

DatasetBuilder = Callable[[str | Path, SplitName, Callable[[object], object] | None], object]
ReIDBatchItem = tuple[torch.Tensor, int, int, str | Path]

_DATASET_BUILDERS: dict[str, DatasetBuilder] = {
    "market1501": Market1501Dataset,
    "msmt17_v1": MSMT17Dataset,
    "occluded_reid": OccludedREIDDataset,
    "vc_clothes": VCClothesDataset,
}


def reid_collate(
    batch: Sequence[ReIDBatchItem],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    if not batch:
        raise ValueError("Cannot collate an empty Re-ID batch")

    images, pids, camids, paths = zip(*batch, strict=True)
    return (
        torch.stack(tuple(images), dim=0).float(),
        torch.as_tensor(pids, dtype=torch.long),
        torch.as_tensor(camids, dtype=torch.long),
        tuple(str(path) for path in paths),
    )


def build_reid_dataset(
    name: str,
    root: str | Path,
    split: SplitName,
    transform: Callable[[object], object] | None = None,
) -> object:
    normalized_name = normalize_dataset_name(name)
    try:
        builder = _DATASET_BUILDERS[normalized_name]
    except KeyError as exc:
        valid = ", ".join(sorted(_DATASET_BUILDERS))
        raise ValueError(f"Unknown Re-ID dataset {name!r}; expected one of: {valid}") from exc
    return builder(root, split, transform)


def build_reid_dataloader(
    name: str,
    root: str | Path,
    split: SplitName,
    batch_size: int,
    image_size: ImageSize = (256, 128),
    random_erasing: bool = False,
    random_erasing_prob: float = 0.5,
    padding: int = 0,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    sampler_name: str | None = None,
    sampler_num_pids: int | None = None,
    sampler_num_instances: int | None = None,
    sampler_batches_per_epoch: int | None = None,
) -> DataLoader:
    transform = (
        build_train_transform(
            image_size=image_size,
            random_erasing=random_erasing,
            erase_prob=random_erasing_prob,
            padding=padding,
        )
        if split == "train"
        else build_eval_transform(image_size=image_size)
    )
    dataset = build_reid_dataset(name=name, root=root, split=split, transform=transform)

    if sampler_name is not None:
        normalized_sampler_name = sampler_name.lower().replace("-", "_")
        if normalized_sampler_name != "pk":
            raise ValueError("sampler_name must be one of: pk")
        if split != "train":
            raise ValueError("PK sampler is only supported for the train split")
        if sampler_num_pids is None:
            raise ValueError("sampler_num_pids is required when sampler_name='pk'")
        if sampler_num_instances is None:
            raise ValueError("sampler_num_instances is required when sampler_name='pk'")
        expected_batch_size = sampler_num_pids * sampler_num_instances
        if batch_size != expected_batch_size:
            raise ValueError(
                "data.batch_size must equal sampler_num_pids * sampler_num_instances "
                f"for PK sampler, got {batch_size} and {expected_batch_size}"
            )
        return DataLoader(
            dataset,
            batch_sampler=PKBatchSampler(
                samples=dataset.samples,
                num_pids=sampler_num_pids,
                num_instances=sampler_num_instances,
                batches_per_epoch=sampler_batches_per_epoch,
            ),
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=reid_collate,
        )

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=reid_collate,
    )


def normalize_dataset_name(name: str | None) -> str:
    if name is None or name == "":
        return "market1501"
    normalized = name.lower().replace("-", "_")
    aliases = {
        "market_1501": "market1501",
        "market1501": "market1501",
        "msmt17": "msmt17_v1",
        "msmt17_v1": "msmt17_v1",
        "occluded_reid": "occluded_reid",
        "vc_clothes": "vc_clothes",
    }
    return aliases.get(normalized, normalized)


def market1501_collate(
    batch: Sequence[ReIDBatchItem],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    return reid_collate(batch)


def build_market1501_dataloader(
    root: str | Path,
    split: SplitName,
    batch_size: int,
    image_size: ImageSize = (256, 128),
    random_erasing: bool = False,
    random_erasing_prob: float = 0.5,
    padding: int = 0,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    sampler_name: str | None = None,
    sampler_num_pids: int | None = None,
    sampler_num_instances: int | None = None,
    sampler_batches_per_epoch: int | None = None,
) -> DataLoader:
    return build_reid_dataloader(
        name="market1501",
        root=root,
        split=split,
        batch_size=batch_size,
        image_size=image_size,
        random_erasing=random_erasing,
        random_erasing_prob=random_erasing_prob,
        padding=padding,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler_name=sampler_name,
        sampler_num_pids=sampler_num_pids,
        sampler_num_instances=sampler_num_instances,
        sampler_batches_per_epoch=sampler_batches_per_epoch,
    )
