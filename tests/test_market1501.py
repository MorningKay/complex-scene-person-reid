from pathlib import Path

import pytest
from PIL import Image

from reid.data.market1501 import (
    Market1501Dataset,
    list_market1501_split,
    parse_market1501_filename,
)

DATA_ROOT = Path("data/Market-1501-v15.09.15")


def require_market1501() -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")


def test_parse_market1501_filename_uses_zero_based_camera_ids() -> None:
    pid, camid = parse_market1501_filename("0002_c1s1_000451_03.jpg")

    assert pid == 2
    assert camid == 0


def test_parse_market1501_filename_accepts_path_objects() -> None:
    pid, camid = parse_market1501_filename(Path("query/1345_c2s3_034007_00.jpg"))

    assert pid == 1345
    assert camid == 1


def test_parse_market1501_filename_accepts_duplicate_jpg_suffix() -> None:
    pid, camid = parse_market1501_filename("1488_c1s6_023021_00.jpg.jpg")

    assert pid == 1488
    assert camid == 0


def test_parse_market1501_filename_rejects_non_image_names() -> None:
    with pytest.raises(ValueError):
        parse_market1501_filename("Thumbs.db")


@pytest.mark.parametrize(
    ("split", "expected_count"),
    [
        ("train", 12936),
        ("query", 3368),
        ("gallery", 19732),
    ],
)
def test_list_market1501_split_counts_only_jpg_files(split: str, expected_count: int) -> None:
    require_market1501()

    samples = list_market1501_split(DATA_ROOT, split)

    assert len(samples) == expected_count
    assert all(sample.path.suffix.lower() == ".jpg" for sample in samples)


def test_market1501_dataset_getitem_returns_image_and_metadata() -> None:
    require_market1501()

    dataset = Market1501Dataset(DATA_ROOT, "query")

    image, pid, camid, path = dataset[0]

    assert len(dataset) == 3368
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert isinstance(pid, int)
    assert isinstance(camid, int)
    assert isinstance(path, Path)
    assert path.suffix.lower() == ".jpg"


def test_market1501_dataset_applies_transform() -> None:
    require_market1501()

    dataset = Market1501Dataset(DATA_ROOT, "query", transform=lambda image: image.size)

    image_size, pid, camid, path = dataset[0]

    assert isinstance(image_size, tuple)
    assert len(image_size) == 2
    assert isinstance(pid, int)
    assert isinstance(camid, int)
    assert path.is_file()
