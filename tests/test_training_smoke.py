from pathlib import Path

import pytest
import torch

from reid.engine import run_training

DATA_ROOT = Path("data/Market-1501-v15.09.15")


def make_smoke_config() -> dict:
    return {
        "run": {
            "name": "pytest_smoke",
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


def test_run_training_writes_smoke_artifacts(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    output_dir = tmp_path / "run"

    metrics = run_training(
        config=make_smoke_config(),
        output_dir=output_dir,
        device="cpu",
    )

    assert metrics["epoch"] == 1
    assert metrics["num_batches"] == 1
    assert metrics["num_samples"] == 2
    assert metrics["avg_train_loss"] > 0

    assert (output_dir / "config.yaml").is_file()
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "logs" / "train.log").is_file()
    assert (output_dir / "ckpt" / "latest.pth").is_file()
    assert (output_dir / "ckpt" / "best.pth").is_file()
    assert (output_dir / "run_summary.md").is_file()

    checkpoint = torch.load(output_dir / "ckpt" / "latest.pth", map_location="cpu")
    assert {"model", "optimizer", "epoch", "metrics"}.issubset(checkpoint)
    assert checkpoint["epoch"] == 1
    assert checkpoint["metrics"]["num_batches"] == 1


def test_run_training_rejects_missing_required_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_training(config={"run": {"name": "bad"}}, output_dir=tmp_path, device="cpu")
