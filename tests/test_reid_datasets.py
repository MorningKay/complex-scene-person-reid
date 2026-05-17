from pathlib import Path

import pytest
import torch

from reid.data import (
    build_reid_dataloader,
    build_reid_dataset,
    list_msmt17_split,
    list_occluded_reid_split,
    list_vc_clothes_split,
    parse_msmt17_filename,
    parse_occluded_reid_filename,
    parse_vc_clothes_filename,
    reid_collate,
)

MARKET_ROOT = Path("data/Market-1501-v15.09.15")
MSMT17_ROOT = Path("data/MSMT17_V1")
OCCLUDED_REID_ROOT = Path("data/Occluded_REID")
VC_CLOTHES_ROOT = Path("data/VC-Clothes")


def require_dataset(root: Path) -> None:
    if not root.is_dir():
        pytest.skip(f"Dataset not found at {root}")


def test_reid_collate_stacks_images_and_metadata() -> None:
    batch = [
        (torch.zeros(3, 256, 128), 1, 0, Path("a.jpg")),
        (torch.ones(3, 256, 128), 2, 1, Path("b.jpg")),
    ]

    images, pids, camids, paths = reid_collate(batch)

    assert images.shape == (2, 3, 256, 128)
    assert images.dtype == torch.float32
    assert pids.tolist() == [1, 2]
    assert camids.tolist() == [0, 1]
    assert paths == ("a.jpg", "b.jpg")


def test_reid_collate_rejects_empty_batches() -> None:
    with pytest.raises(ValueError):
        reid_collate([])


@pytest.mark.parametrize(
    ("name", "root", "split", "expected_count"),
    [
        ("market1501", MARKET_ROOT, "train", 12936),
        ("market1501", MARKET_ROOT, "query", 3368),
        ("market1501", MARKET_ROOT, "gallery", 19732),
        ("msmt17_v1", MSMT17_ROOT, "train", 32621),
        ("msmt17_v1", MSMT17_ROOT, "query", 11659),
        ("msmt17_v1", MSMT17_ROOT, "gallery", 82161),
        ("occluded_reid", OCCLUDED_REID_ROOT, "query", 1000),
        ("occluded_reid", OCCLUDED_REID_ROOT, "gallery", 1000),
        ("vc_clothes", VC_CLOTHES_ROOT, "train", 9449),
        ("vc_clothes", VC_CLOTHES_ROOT, "query", 1020),
        ("vc_clothes", VC_CLOTHES_ROOT, "gallery", 8591),
    ],
)
def test_build_reid_dataset_counts(
    name: str,
    root: Path,
    split: str,
    expected_count: int,
) -> None:
    require_dataset(root)

    dataset = build_reid_dataset(name=name, root=root, split=split)

    assert len(dataset) == expected_count
    assert len(dataset.samples) == expected_count


@pytest.mark.parametrize(
    ("name", "root", "split"),
    [
        ("market1501", MARKET_ROOT, "query"),
        ("msmt17_v1", MSMT17_ROOT, "query"),
        ("occluded_reid", OCCLUDED_REID_ROOT, "query"),
        ("vc_clothes", VC_CLOTHES_ROOT, "query"),
    ],
)
def test_build_reid_dataloader_returns_standard_batch(
    name: str,
    root: Path,
    split: str,
) -> None:
    require_dataset(root)

    dataloader = build_reid_dataloader(
        name=name,
        root=root,
        split=split,
        batch_size=2,
        image_size=(64, 32),
        num_workers=0,
    )
    images, pids, camids, paths = next(iter(dataloader))

    assert images.shape == (2, 3, 64, 32)
    assert pids.shape == (2,)
    assert camids.shape == (2,)
    assert all(isinstance(path, str) for path in paths)


def test_parse_msmt17_filename_reads_zero_based_camera_id() -> None:
    assert parse_msmt17_filename("0000/0000_008_01_0303morning_0019_2.jpg") == 0
    assert parse_msmt17_filename("0000/0000_045_12_0303morning_0006_2.jpg") == 11


def test_msmt17_train_split_combines_train_and_val_lists() -> None:
    require_dataset(MSMT17_ROOT)

    samples = list_msmt17_split(MSMT17_ROOT, "train")

    source_lists = {sample.metadata["source_list"] for sample in samples}
    assert source_lists == {"list_train.txt", "list_val.txt"}


def test_occluded_reid_uses_occluded_query_and_whole_gallery() -> None:
    require_dataset(OCCLUDED_REID_ROOT)

    query_samples = list_occluded_reid_split(OCCLUDED_REID_ROOT, "query")
    gallery_samples = list_occluded_reid_split(OCCLUDED_REID_ROOT, "gallery")

    assert query_samples[0].path.suffix.lower() == ".tif"
    assert query_samples[0].camid == 0
    assert gallery_samples[0].camid == 1
    assert parse_occluded_reid_filename(query_samples[0].path) == query_samples[0].pid


def test_occluded_reid_rejects_train_split() -> None:
    require_dataset(OCCLUDED_REID_ROOT)

    with pytest.raises(ValueError, match="does not provide a train split"):
        list_occluded_reid_split(OCCLUDED_REID_ROOT, "train")


def test_vc_clothes_parses_camera_and_clothes_metadata() -> None:
    pid, camid, clothes_id = parse_vc_clothes_filename("0001-02-03-04.jpg")

    assert pid == 1
    assert camid == 1
    assert clothes_id == 3


def test_vc_clothes_samples_keep_clothes_id_metadata() -> None:
    require_dataset(VC_CLOTHES_ROOT)

    samples = list_vc_clothes_split(VC_CLOTHES_ROOT, "query")

    assert isinstance(samples[0].metadata["clothes_id"], int)
