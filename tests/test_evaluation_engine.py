from pathlib import Path

import pytest
import torch

from reid.engine import run_evaluation
from reid.engine.train import run_training

DATA_ROOT = Path("data/Market-1501-v15.09.15")


def make_smoke_config() -> dict:
    return {
        "run": {
            "name": "eval_pytest_smoke",
            "seed": 42,
            "smoke": True,
        },
        "data": {
            "root": str(DATA_ROOT),
            "image_size": [32, 16],
            "batch_size": 2,
            "num_workers": 0,
            "random_erasing": False,
            "pin_memory": False,
            "drop_last": False,
        },
        "model": {
            "num_classes": 751,
            "feature_dim": 2048,
            "last_stride": 1,
        },
        "loss": {
            "label_smoothing": 0.0,
        },
        "optimizer": {
            "lr": 0.0003,
            "weight_decay": 0.0005,
        },
        "train": {
            "epochs": 1,
            "max_batches": 1,
            "log_interval": 1,
        },
    }


def test_run_evaluation_writes_metrics_and_log(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    run_training(config=make_smoke_config(), output_dir=train_dir, device="cpu")

    metrics = run_evaluation(
        checkpoint_path=train_dir / "ckpt" / "best.pth",
        data_root=DATA_ROOT,
        output_dir=eval_dir,
        device="cpu",
        batch_size=4,
        num_workers=0,
        max_query=8,
        max_gallery=32,
    )

    assert (eval_dir / "eval_metrics.json").is_file()
    assert (eval_dir / "logs" / "eval.txt").is_file()
    assert metrics["num_query"] == 8
    assert metrics["num_gallery"] == 32
    assert metrics["num_valid_queries"] > 0
    for key in ("rank1", "rank5", "rank10", "mAP"):
        assert 0 <= metrics[key] <= 1


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"config": {"model": {}}},
        {"model": {}},
    ],
)
def test_run_evaluation_rejects_malformed_checkpoints(
    tmp_path: Path,
    checkpoint: dict,
) -> None:
    checkpoint_path = tmp_path / "bad.pth"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError):
        run_evaluation(
            checkpoint_path=checkpoint_path,
            data_root=DATA_ROOT,
            output_dir=tmp_path / "eval",
            device="cpu",
            max_query=1,
            max_gallery=1,
        )
