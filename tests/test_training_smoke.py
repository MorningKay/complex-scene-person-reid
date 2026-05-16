import json
from pathlib import Path

import pytest
import torch

from reid.engine import run_training
from reid.utils import validate_training_config

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
            "pretrained": False,
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
        "eval": {
            "enabled": True,
            "interval": 1,
            "batch_size": 4,
            "num_workers": 0,
            "distance": "cosine",
            "max_query": 8,
            "max_gallery": 32,
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
    assert (output_dir / "logs" / "train.txt").is_file()
    assert (output_dir / "ckpt" / "latest.pth").is_file()
    assert (output_dir / "ckpt" / "best.pth").is_file()
    assert (output_dir / "run_summary.md").is_file()

    checkpoint = torch.load(output_dir / "ckpt" / "latest.pth", map_location="cpu")
    assert {"model", "optimizer", "epoch", "metrics"}.issubset(checkpoint)
    assert checkpoint["epoch"] == 1
    assert checkpoint["metrics"]["num_batches"] == 1
    assert "eval" in checkpoint["metrics"]
    assert 0 <= checkpoint["metrics"]["eval"]["mAP"] <= 1

    best_checkpoint = torch.load(output_dir / "ckpt" / "best.pth", map_location="cpu")
    assert "eval" in best_checkpoint["metrics"]

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["best_metric_name"] == "mAP"
    assert 0 <= metrics["best_mAP"] <= 1
    assert 0 <= metrics["best_rank1"] <= 1
    assert "eval" in metrics["history"][0]

    train_log = (output_dir / "logs" / "train.txt").read_text(encoding="utf-8")
    assert "model_pretrained=False" in train_log
    assert "mAP=" in train_log
    assert "rank1=" in train_log

    run_summary = (output_dir / "run_summary.md").read_text(encoding="utf-8")
    assert "- model_pretrained: False" in run_summary
    assert "- best_metric_name: mAP" in run_summary
    assert "- best_mAP:" in run_summary
    assert "- best_rank1:" in run_summary


def test_run_training_rejects_missing_required_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_training(config={"run": {"name": "bad"}}, output_dir=tmp_path, device="cpu")


def test_validate_training_config_accepts_boolean_pretrained() -> None:
    config = make_smoke_config()

    validate_training_config(config)


def test_validate_training_config_rejects_non_boolean_pretrained() -> None:
    config = make_smoke_config()
    config["model"]["pretrained"] = "true"

    with pytest.raises(ValueError, match="model.pretrained"):
        validate_training_config(config)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("interval", 0, "eval.interval"),
        ("batch_size", 0, "eval.batch_size"),
        ("num_workers", -1, "eval.num_workers"),
        ("distance", "bad", "eval.distance"),
    ],
)
def test_validate_training_config_rejects_invalid_eval_config(
    key: str,
    value: object,
    message: str,
) -> None:
    config = make_smoke_config()
    config["eval"][key] = value

    with pytest.raises(ValueError, match=message):
        validate_training_config(config)
