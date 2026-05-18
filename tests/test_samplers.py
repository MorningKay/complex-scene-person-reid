from collections import Counter
from pathlib import Path

import pytest

from reid.data import PKBatchSampler, ReIDSample, build_reid_dataloader

DATA_ROOT = Path("data/Market-1501-v15.09.15")


def make_samples(num_pids: int = 4, instances_per_pid: int = 3) -> list[ReIDSample]:
    return [
        ReIDSample(path=Path(f"{pid}_{index}.jpg"), pid=pid, camid=0)
        for pid in range(num_pids)
        for index in range(instances_per_pid)
    ]


def test_pk_sampler_yields_p_by_k_batches() -> None:
    samples = make_samples(num_pids=5, instances_per_pid=4)
    sampler = PKBatchSampler(
        samples=samples,
        num_pids=3,
        num_instances=2,
        batches_per_epoch=1,
    )

    batch = next(iter(sampler))
    pids = [samples[index].pid for index in batch]
    counts = Counter(pids)

    assert len(batch) == 6
    assert len(counts) == 3
    assert set(counts.values()) == {2}


def test_pk_sampler_samples_with_replacement_when_identity_has_too_few_images() -> None:
    samples = make_samples(num_pids=2, instances_per_pid=1)
    sampler = PKBatchSampler(
        samples=samples,
        num_pids=2,
        num_instances=2,
        batches_per_epoch=1,
    )

    batch = next(iter(sampler))
    pids = [samples[index].pid for index in batch]

    assert len(batch) == 4
    assert Counter(pids) == {0: 2, 1: 2}
    assert len(set(batch)) == 2


def test_pk_sampler_default_epoch_length_matches_image_count() -> None:
    samples = make_samples(num_pids=5, instances_per_pid=3)
    sampler = PKBatchSampler(samples=samples, num_pids=2, num_instances=2)

    assert len(sampler) == 4


@pytest.mark.parametrize(
    ("num_pids", "num_instances", "message"),
    [
        (0, 2, "num_pids"),
        (2, 0, "num_instances"),
        (5, 2, "at least num_pids"),
    ],
)
def test_pk_sampler_rejects_invalid_batch_structure(
    num_pids: int,
    num_instances: int,
    message: str,
) -> None:
    samples = make_samples(num_pids=2, instances_per_pid=2)

    with pytest.raises(ValueError, match=message):
        PKBatchSampler(
            samples=samples,
            num_pids=num_pids,
            num_instances=num_instances,
            batches_per_epoch=1,
        )


def test_pk_sampler_rejects_invalid_epoch_length() -> None:
    samples = make_samples(num_pids=2, instances_per_pid=2)

    with pytest.raises(ValueError, match="batches_per_epoch"):
        PKBatchSampler(
            samples=samples,
            num_pids=2,
            num_instances=2,
            batches_per_epoch=0,
        )


def test_reid_dataloader_supports_pk_sampler_train_batches() -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    dataloader = build_reid_dataloader(
        name="market1501",
        root=DATA_ROOT,
        split="train",
        batch_size=4,
        image_size=(64, 32),
        num_workers=0,
        sampler_name="pk",
        sampler_num_pids=2,
        sampler_num_instances=2,
        sampler_batches_per_epoch=1,
    )

    images, pids, camids, paths = next(iter(dataloader))

    assert images.shape == (4, 3, 64, 32)
    assert len(paths) == 4
    assert camids.shape == (4,)
    counts = Counter(pids.tolist())
    assert len(counts) == 2
    assert set(counts.values()) == {2}


def test_reid_dataloader_rejects_pk_sampler_for_non_train_split() -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    with pytest.raises(ValueError, match="train split"):
        build_reid_dataloader(
            name="market1501",
            root=DATA_ROOT,
            split="query",
            batch_size=4,
            num_workers=0,
            sampler_name="pk",
            sampler_num_pids=2,
            sampler_num_instances=2,
        )


def test_reid_dataloader_rejects_pk_batch_size_mismatch() -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    with pytest.raises(ValueError, match="batch_size"):
        build_reid_dataloader(
            name="market1501",
            root=DATA_ROOT,
            split="train",
            batch_size=3,
            num_workers=0,
            sampler_name="pk",
            sampler_num_pids=2,
            sampler_num_instances=2,
        )
