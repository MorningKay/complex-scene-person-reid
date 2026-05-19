from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from reid.data import (
    build_eval_transform,
    build_market1501_dataloader,
    build_reid_dataloader,
    build_train_transform,
    market1501_collate,
)

DATA_ROOT = Path("data/Market-1501-v15.09.15")


def make_image() -> Image.Image:
    return Image.new("RGB", (64, 96), color=(20, 40, 60))


def test_train_transform_returns_normalized_tensor() -> None:
    transform = build_train_transform(image_size=(256, 128), random_erasing=False)

    image = transform(make_image())

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 256, 128)
    assert image.dtype == torch.float32


def test_train_transform_can_enable_random_erasing() -> None:
    transform = build_train_transform(random_erasing=True, erase_prob=0.25)

    assert isinstance(transform.transforms[-1], transforms.RandomErasing)
    assert transform.transforms[-1].p == 0.25


def test_train_transform_can_enable_padding_random_crop() -> None:
    transform = build_train_transform(image_size=(64, 32), padding=8)

    image = transform(make_image())
    transform_types = tuple(type(step) for step in transform.transforms)

    assert image.shape == (3, 64, 32)
    assert transforms.Pad in transform_types
    assert transforms.RandomCrop in transform_types


def test_eval_transform_returns_tensor_without_random_augmentation() -> None:
    transform = build_eval_transform(image_size=(256, 128))

    image = transform(make_image())
    transform_types = tuple(type(step) for step in transform.transforms)

    assert image.shape == (3, 256, 128)
    assert image.dtype == torch.float32
    assert transforms.RandomHorizontalFlip not in transform_types
    assert transforms.RandomErasing not in transform_types


def test_market1501_collate_stacks_images_and_metadata() -> None:
    batch = [
        (torch.zeros(3, 256, 128), 1, 0, Path("a.jpg")),
        (torch.ones(3, 256, 128), 2, 1, Path("b.jpg")),
    ]

    images, pids, camids, paths = market1501_collate(batch)

    assert images.shape == (2, 3, 256, 128)
    assert images.dtype == torch.float32
    assert pids.dtype == torch.long
    assert camids.dtype == torch.long
    assert pids.tolist() == [1, 2]
    assert camids.tolist() == [0, 1]
    assert paths == ("a.jpg", "b.jpg")


def test_market1501_collate_rejects_empty_batches() -> None:
    with pytest.raises(ValueError):
        market1501_collate([])


def test_build_market1501_dataloader_returns_tensor_batch() -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    dataloader = build_market1501_dataloader(
        root=DATA_ROOT,
        split="query",
        batch_size=4,
        num_workers=0,
    )

    images, pids, camids, paths = next(iter(dataloader))

    assert images.shape == (4, 3, 256, 128)
    assert images.dtype == torch.float32
    assert pids.shape == (4,)
    assert pids.dtype == torch.long
    assert camids.shape == (4,)
    assert camids.dtype == torch.long
    assert len(paths) == 4
    assert all(isinstance(path, str) for path in paths)


def test_train_dataloader_passes_random_erasing_probability() -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    dataloader = build_reid_dataloader(
        name="market1501",
        root=DATA_ROOT,
        split="train",
        batch_size=2,
        random_erasing=True,
        random_erasing_prob=0.25,
        num_workers=0,
    )

    transform = dataloader.dataset.transform
    assert isinstance(transform.transforms[-1], transforms.RandomErasing)
    assert transform.transforms[-1].p == 0.25


def test_train_dataloader_passes_padding_to_train_transform() -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    dataloader = build_reid_dataloader(
        name="market1501",
        root=DATA_ROOT,
        split="train",
        batch_size=2,
        padding=8,
        num_workers=0,
    )
    transform_types = tuple(type(step) for step in dataloader.dataset.transform.transforms)

    assert transforms.Pad in transform_types
    assert transforms.RandomCrop in transform_types


def test_eval_dataloader_ignores_random_erasing_controls() -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    dataloader = build_reid_dataloader(
        name="market1501",
        root=DATA_ROOT,
        split="query",
        batch_size=2,
        random_erasing=True,
        random_erasing_prob=0.25,
        padding=8,
        num_workers=0,
    )
    transform_types = tuple(type(step) for step in dataloader.dataset.transform.transforms)

    assert transforms.RandomErasing not in transform_types
    assert transforms.Pad not in transform_types
    assert transforms.RandomCrop not in transform_types
