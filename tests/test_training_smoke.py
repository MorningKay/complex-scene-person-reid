import json
from pathlib import Path

import pytest
import torch

from reid.engine import run_training
from reid.utils import validate_training_config

DATA_ROOT = Path("data/Market-1501-v15.09.15")
MSMT17_ROOT = Path("data/MSMT17_V1")


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
    assert metrics["dataset_name"] == "market1501"
    assert metrics["num_batches"] == 1
    assert metrics["num_samples"] == 2
    assert metrics["num_train_ids"] == 751
    assert metrics["avg_train_loss"] > 0
    assert metrics["avg_ce_loss"] == pytest.approx(metrics["avg_train_loss"])
    assert 0 <= metrics["train_id_acc"] <= 1
    assert metrics["lr"] == pytest.approx(0.0003)

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
    assert "avg_ce_loss" in checkpoint["metrics"]
    assert "train_id_acc" in checkpoint["metrics"]
    assert "lr" in checkpoint["metrics"]
    assert "eval" in checkpoint["metrics"]
    assert 0 <= checkpoint["metrics"]["eval"]["mAP"] <= 1

    best_checkpoint = torch.load(output_dir / "ckpt" / "best.pth", map_location="cpu")
    assert "eval" in best_checkpoint["metrics"]

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset_name"] == "market1501"
    assert metrics["num_train_ids"] == 751
    assert metrics["avg_ce_loss"] == pytest.approx(metrics["avg_train_loss"])
    assert 0 <= metrics["train_id_acc"] <= 1
    assert metrics["best_metric_name"] == "mAP"
    assert 0 <= metrics["best_mAP"] <= 1
    assert 0 <= metrics["best_rank1"] <= 1
    assert "eval" in metrics["history"][0]
    assert "avg_ce_loss" in metrics["history"][0]
    assert "train_id_acc" in metrics["history"][0]
    assert "lr" in metrics["history"][0]

    train_log = (output_dir / "logs" / "train.txt").read_text(encoding="utf-8")
    assert "dataset_name=market1501" in train_log
    assert "model_pretrained=False" in train_log
    assert "ce_loss=" in train_log
    assert "id_acc=" in train_log
    assert "lr=" in train_log
    assert "mAP=" in train_log
    assert "rank1=" in train_log

    run_summary = (output_dir / "run_summary.md").read_text(encoding="utf-8")
    assert "- dataset_name: market1501" in run_summary
    assert "- model_pretrained: False" in run_summary
    assert "- num_train_ids: 751" in run_summary
    assert "- final_avg_ce_loss:" in run_summary
    assert "- final_train_id_acc:" in run_summary
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


def test_validate_training_config_rejects_non_string_dataset_name() -> None:
    config = make_smoke_config()
    config["data"]["name"] = 123

    with pytest.raises(ValueError, match="data.name"):
        validate_training_config(config)


def test_run_training_accepts_msmt17_dataset_name(tmp_path: Path) -> None:
    if not MSMT17_ROOT.is_dir():
        pytest.skip(f"MSMT17_V1 dataset not found at {MSMT17_ROOT}")

    config = make_smoke_config()
    config["run"]["name"] = "pytest_msmt17_smoke"
    config["data"]["name"] = "msmt17_v1"
    config["data"]["root"] = str(MSMT17_ROOT)
    config["model"]["num_classes"] = 1041
    config["eval"] = {"enabled": False}
    output_dir = tmp_path / "msmt17_run"

    metrics = run_training(config=config, output_dir=output_dir, device="cpu")

    assert metrics["dataset_name"] == "msmt17_v1"
    assert metrics["num_train_ids"] == 1041
    assert metrics["num_batches"] == 1
    assert metrics["num_samples"] == 2
    assert metrics["avg_ce_loss"] == pytest.approx(metrics["avg_train_loss"])

    train_log = (output_dir / "logs" / "train.txt").read_text(encoding="utf-8")
    assert "dataset_name=msmt17_v1" in train_log
    assert "ce_loss=" in train_log
    assert "id_acc=" in train_log


def test_run_training_accepts_msmt17_training_time_eval(tmp_path: Path) -> None:
    if not MSMT17_ROOT.is_dir():
        pytest.skip(f"MSMT17_V1 dataset not found at {MSMT17_ROOT}")

    config = make_smoke_config()
    config["run"]["name"] = "pytest_msmt17_eval_smoke"
    config["data"]["name"] = "msmt17_v1"
    config["data"]["root"] = str(MSMT17_ROOT)
    config["model"]["num_classes"] = 1041
    config["eval"]["max_query"] = 8
    config["eval"]["max_gallery"] = 64
    config["eval"]["query_chunk_size"] = 2
    output_dir = tmp_path / "msmt17_eval_run"

    metrics = run_training(config=config, output_dir=output_dir, device="cpu")

    assert metrics["dataset_name"] == "msmt17_v1"
    assert metrics["best_metric_name"] == "mAP"
    assert metrics["best_mAP"] is not None
    assert metrics["history"][0]["eval"]["dataset_name"] == "msmt17_v1"
    assert metrics["history"][0]["eval"]["query_chunk_size"] == 2
    assert metrics["history"][0]["eval"]["num_query"] == 8
    assert metrics["history"][0]["eval"]["num_gallery"] == 64

    checkpoint = torch.load(output_dir / "ckpt" / "best.pth", map_location="cpu")
    assert checkpoint["metrics"]["eval"]["dataset_name"] == "msmt17_v1"
    assert checkpoint["metrics"]["eval"]["query_chunk_size"] == 2

    train_log = (output_dir / "logs" / "train.txt").read_text(encoding="utf-8")
    assert "dataset_name=msmt17_v1" in train_log
    assert "query_chunk_size=2" in train_log
    assert "mAP=" in train_log


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("interval", 0, "eval.interval"),
        ("batch_size", 0, "eval.batch_size"),
        ("num_workers", -1, "eval.num_workers"),
        ("distance", "bad", "eval.distance"),
        ("query_chunk_size", 0, "eval.query_chunk_size"),
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
