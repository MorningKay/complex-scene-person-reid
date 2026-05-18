import json
from pathlib import Path

import pytest
import torch

from reid.engine import run_training
from reid.utils import load_config, validate_training_config

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
    assert metrics["avg_triplet_loss"] == pytest.approx(0.0)
    assert 0 <= metrics["train_id_acc"] <= 1
    assert metrics["lr"] == pytest.approx(0.0003)
    assert metrics["scheduler_name"] == "constant"
    assert metrics["scheduler_state"]["name"] == "constant"
    assert metrics["random_erasing"] is False
    assert metrics["random_erasing_prob"] == pytest.approx(0.5)
    assert metrics["sampler_name"] == "shuffle"
    assert metrics["sampler_num_pids"] is None
    assert metrics["sampler_num_instances"] is None
    assert metrics["triplet_enabled"] is False
    assert metrics["triplet_margin"] is None
    assert metrics["triplet_weight"] == pytest.approx(0.0)
    assert metrics["triplet_normalize_features"] is False
    assert metrics["amp_enabled"] is False
    assert metrics["grad_clip_norm"] is None

    assert (output_dir / "config.yaml").is_file()
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "logs" / "train.txt").is_file()
    assert (output_dir / "ckpt" / "latest.pth").is_file()
    assert (output_dir / "ckpt" / "best.pth").is_file()
    assert (output_dir / "run_summary.md").is_file()

    checkpoint = torch.load(output_dir / "ckpt" / "latest.pth", map_location="cpu")
    assert {"model", "optimizer", "scheduler_state", "scaler", "epoch", "metrics"}.issubset(
        checkpoint
    )
    assert checkpoint["epoch"] == 1
    assert checkpoint["history"][0]["epoch"] == 1
    assert checkpoint["metrics"]["num_batches"] == 1
    assert "avg_ce_loss" in checkpoint["metrics"]
    assert "avg_triplet_loss" in checkpoint["metrics"]
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
    assert metrics["avg_triplet_loss"] == pytest.approx(0.0)
    assert 0 <= metrics["train_id_acc"] <= 1
    assert metrics["best_metric_name"] == "mAP"
    assert 0 <= metrics["best_mAP"] <= 1
    assert 0 <= metrics["best_rank1"] <= 1
    assert "eval" in metrics["history"][0]
    assert "avg_ce_loss" in metrics["history"][0]
    assert "train_id_acc" in metrics["history"][0]
    assert "lr" in metrics["history"][0]
    assert metrics["scheduler_name"] == "constant"
    assert metrics["random_erasing"] is False
    assert metrics["random_erasing_prob"] == pytest.approx(0.5)
    assert metrics["sampler_name"] == "shuffle"
    assert metrics["triplet_enabled"] is False
    assert metrics["history"][0]["avg_triplet_loss"] == pytest.approx(0.0)
    assert metrics["amp_enabled"] is False
    assert metrics["grad_clip_norm"] is None

    train_log = (output_dir / "logs" / "train.txt").read_text(encoding="utf-8")
    assert "dataset_name=market1501" in train_log
    assert "model_pretrained=False" in train_log
    assert "random_erasing=False" in train_log
    assert "random_erasing_prob=0.500000" in train_log
    assert "sampler_name=shuffle" in train_log
    assert "triplet_enabled=False" in train_log
    assert "triplet_loss=" in train_log
    assert "scheduler_name=constant" in train_log
    assert "amp_enabled=False" in train_log
    assert "grad_clip_norm=null" in train_log
    assert "ce_loss=" in train_log
    assert "id_acc=" in train_log
    assert "lr=" in train_log
    assert "mAP=" in train_log
    assert "rank1=" in train_log

    run_summary = (output_dir / "run_summary.md").read_text(encoding="utf-8")
    assert "- dataset_name: market1501" in run_summary
    assert "- model_pretrained: False" in run_summary
    assert "- random_erasing: False" in run_summary
    assert "- random_erasing_prob: 0.500000" in run_summary
    assert "- sampler_name: shuffle" in run_summary
    assert "- triplet_enabled: False" in run_summary
    assert "- scheduler_name: constant" in run_summary
    assert "- amp_enabled: False" in run_summary
    assert "- grad_clip_norm: null" in run_summary
    assert "- num_train_ids: 751" in run_summary
    assert "- final_avg_ce_loss:" in run_summary
    assert "- final_avg_triplet_loss:" in run_summary
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
    assert metrics["avg_triplet_loss"] == pytest.approx(0.0)

    train_log = (output_dir / "logs" / "train.txt").read_text(encoding="utf-8")
    assert "dataset_name=msmt17_v1" in train_log
    assert "ce_loss=" in train_log
    assert "id_acc=" in train_log


def test_run_training_accepts_triplet_pk_sampler(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    config = make_smoke_config()
    config["eval"] = {"enabled": False}
    config["data"]["batch_size"] = 4
    config["sampler"] = {
        "name": "pk",
        "num_pids": 2,
        "num_instances": 2,
    }
    config["loss"]["triplet"] = {
        "enabled": True,
        "margin": 0.3,
        "weight": 1.0,
        "normalize_features": True,
    }
    output_dir = tmp_path / "triplet_pk"

    metrics = run_training(config=config, output_dir=output_dir, device="cpu")

    assert metrics["num_batches"] == 1
    assert metrics["num_samples"] == 4
    assert metrics["sampler_name"] == "pk"
    assert metrics["sampler_num_pids"] == 2
    assert metrics["sampler_num_instances"] == 2
    assert metrics["triplet_enabled"] is True
    assert metrics["triplet_margin"] == pytest.approx(0.3)
    assert metrics["triplet_weight"] == pytest.approx(1.0)
    assert metrics["triplet_normalize_features"] is True
    assert metrics["avg_triplet_loss"] >= 0
    assert metrics["avg_train_loss"] == pytest.approx(
        metrics["avg_ce_loss"] + metrics["avg_triplet_loss"]
    )
    assert metrics["history"][0]["avg_triplet_loss"] == pytest.approx(
        metrics["avg_triplet_loss"]
    )

    train_log = (output_dir / "logs" / "train.txt").read_text(encoding="utf-8")
    assert "sampler_name=pk" in train_log
    assert "sampler_num_pids=2" in train_log
    assert "sampler_num_instances=2" in train_log
    assert "triplet_enabled=True" in train_log
    assert "triplet_loss=" in train_log

    run_summary = (output_dir / "run_summary.md").read_text(encoding="utf-8")
    assert "- sampler_name: pk" in run_summary
    assert "- sampler_num_pids: 2" in run_summary
    assert "- sampler_num_instances: 2" in run_summary
    assert "- triplet_enabled: True" in run_summary
    assert "- triplet_margin: 0.300000" in run_summary
    assert "- triplet_weight: 1.000000" in run_summary


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


def test_run_training_applies_cosine_scheduler(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    config = make_smoke_config()
    config["eval"] = {"enabled": False}
    config["train"]["epochs"] = 4
    config["scheduler"] = {
        "name": "cosine",
        "min_lr": 0.000001,
        "warmup_epochs": 2,
        "warmup_factor": 0.1,
    }

    metrics = run_training(config=config, output_dir=tmp_path / "scheduled", device="cpu")

    history = metrics["history"]
    assert [item["epoch"] for item in history] == [1, 2, 3, 4]
    assert history[0]["lr"] == pytest.approx(0.00003)
    assert history[1]["lr"] == pytest.approx(0.0003)
    assert history[-1]["lr"] == pytest.approx(0.000001)
    assert metrics["scheduler_name"] == "cosine"
    assert metrics["scheduler_state"]["last_epoch"] == 4


def test_run_training_records_amp_fallback_and_grad_clip(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    config = make_smoke_config()
    config["eval"] = {"enabled": False}
    config["train"]["amp"] = True
    config["train"]["grad_clip_norm"] = 1.0
    output_dir = tmp_path / "amp_clip"

    metrics = run_training(config=config, output_dir=output_dir, device="cpu")

    assert metrics["amp_enabled"] is False
    assert metrics["grad_clip_norm"] == pytest.approx(1.0)
    checkpoint = torch.load(output_dir / "ckpt" / "latest.pth", map_location="cpu")
    assert "scaler" in checkpoint

    train_log = (output_dir / "logs" / "train.txt").read_text(encoding="utf-8")
    assert "amp_enabled=False" in train_log
    assert "grad_clip_norm=1.000000" in train_log


def test_run_training_records_random_erasing_controls(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    config = make_smoke_config()
    config["eval"] = {"enabled": False}
    config["data"]["random_erasing"] = True
    config["data"]["random_erasing_prob"] = 0.5
    output_dir = tmp_path / "random_erasing"

    metrics = run_training(config=config, output_dir=output_dir, device="cpu")

    assert metrics["random_erasing"] is True
    assert metrics["random_erasing_prob"] == pytest.approx(0.5)

    metrics_json = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_json["random_erasing"] is True
    assert metrics_json["random_erasing_prob"] == pytest.approx(0.5)

    train_log = (output_dir / "logs" / "train.txt").read_text(encoding="utf-8")
    assert "random_erasing=True" in train_log
    assert "random_erasing_prob=0.500000" in train_log

    run_summary = (output_dir / "run_summary.md").read_text(encoding="utf-8")
    assert "- random_erasing: True" in run_summary
    assert "- random_erasing_prob: 0.500000" in run_summary


def test_run_training_resumes_from_checkpoint(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    output_dir = tmp_path / "resume_run"
    first_config = make_smoke_config()
    first_config["eval"] = {"enabled": False}
    run_training(config=first_config, output_dir=output_dir, device="cpu")

    resumed_config = make_smoke_config()
    resumed_config["eval"] = {"enabled": False}
    resumed_config["train"]["epochs"] = 2
    metrics = run_training(
        config=resumed_config,
        output_dir=output_dir,
        device="cpu",
        resume_checkpoint=output_dir / "ckpt" / "latest.pth",
    )

    assert metrics["epoch"] == 2
    assert len(metrics["history"]) == 2
    assert [item["epoch"] for item in metrics["history"]] == [1, 2]
    assert (output_dir / "ckpt" / "best.pth").is_file()

    checkpoint = torch.load(output_dir / "ckpt" / "latest.pth", map_location="cpu")
    assert checkpoint["epoch"] == 2
    assert len(checkpoint["history"]) == 2
    assert "scheduler_state" in checkpoint
    assert "scaler" in checkpoint

    train_log = (output_dir / "logs" / "train.txt").read_text(encoding="utf-8")
    assert "resume_checkpoint=" in train_log


def test_run_training_rejects_resume_config_mismatch(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    output_dir = tmp_path / "bad_resume"
    first_config = make_smoke_config()
    first_config["eval"] = {"enabled": False}
    run_training(config=first_config, output_dir=output_dir, device="cpu")

    resumed_config = make_smoke_config()
    resumed_config["eval"] = {"enabled": False}
    resumed_config["train"]["epochs"] = 2
    resumed_config["data"]["batch_size"] = 4

    with pytest.raises(ValueError, match="Current config must match"):
        run_training(
            config=resumed_config,
            output_dir=output_dir,
            device="cpu",
            resume_checkpoint=output_dir / "ckpt" / "latest.pth",
        )


@pytest.mark.parametrize(
    ("scheduler_config", "message"),
    [
        ({"name": "step", "min_lr": 0.0, "warmup_epochs": 0, "warmup_factor": 1.0}, "name"),
        ({"name": "cosine", "min_lr": -1.0, "warmup_epochs": 0, "warmup_factor": 1.0}, "min_lr"),
        ({"name": "cosine", "min_lr": 1.0, "warmup_epochs": 0, "warmup_factor": 1.0}, "min_lr"),
        ({"name": "cosine", "min_lr": 0.0, "warmup_epochs": -1, "warmup_factor": 1.0}, "warmup_epochs"),
        ({"name": "cosine", "min_lr": 0.0, "warmup_epochs": 0, "warmup_factor": 0.0}, "warmup_factor"),
        ({"name": "cosine", "min_lr": 0.0, "warmup_epochs": 0, "warmup_factor": 2.0}, "warmup_factor"),
    ],
)
def test_validate_training_config_rejects_invalid_scheduler_config(
    scheduler_config: dict,
    message: str,
) -> None:
    config = make_smoke_config()
    config["scheduler"] = scheduler_config

    with pytest.raises(ValueError, match=message):
        validate_training_config(config)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("amp", "true", "train.amp"),
        ("grad_clip_norm", 0, "train.grad_clip_norm"),
        ("grad_clip_norm", -1.0, "train.grad_clip_norm"),
    ],
)
def test_validate_training_config_rejects_invalid_train_controls(
    key: str,
    value: object,
    message: str,
) -> None:
    config = make_smoke_config()
    config["train"][key] = value

    with pytest.raises(ValueError, match=message):
        validate_training_config(config)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("random_erasing", "true", "data.random_erasing"),
        ("random_erasing_prob", 0, "data.random_erasing_prob"),
        ("random_erasing_prob", -0.1, "data.random_erasing_prob"),
        ("random_erasing_prob", 1.1, "data.random_erasing_prob"),
        ("random_erasing_prob", "0.5", "data.random_erasing_prob"),
        ("random_erasing_prob", True, "data.random_erasing_prob"),
    ],
)
def test_validate_training_config_rejects_invalid_random_erasing_controls(
    key: str,
    value: object,
    message: str,
) -> None:
    config = make_smoke_config()
    config["data"][key] = value

    with pytest.raises(ValueError, match=message):
        validate_training_config(config)


@pytest.mark.parametrize(
    ("sampler_config", "message"),
    [
        ("pk", "sampler"),
        ({"name": "random"}, "sampler.name"),
        ({"name": "pk", "num_instances": 2}, "sampler.num_pids"),
        ({"name": "pk", "num_pids": 0, "num_instances": 2}, "sampler.num_pids"),
        ({"name": "pk", "num_pids": 2, "num_instances": 0}, "sampler.num_instances"),
        (
            {"name": "pk", "num_pids": 2, "num_instances": 2, "batches_per_epoch": 0},
            "sampler.batches_per_epoch",
        ),
    ],
)
def test_validate_training_config_rejects_invalid_sampler_config(
    sampler_config: object,
    message: str,
) -> None:
    config = make_smoke_config()
    config["data"]["batch_size"] = 4
    config["sampler"] = sampler_config

    with pytest.raises(ValueError, match=message):
        validate_training_config(config)


def test_validate_training_config_rejects_pk_batch_size_mismatch() -> None:
    config = make_smoke_config()
    config["data"]["batch_size"] = 3
    config["sampler"] = {"name": "pk", "num_pids": 2, "num_instances": 2}

    with pytest.raises(ValueError, match="data.batch_size"):
        validate_training_config(config)


@pytest.mark.parametrize(
    ("triplet_config", "message"),
    [
        ("enabled", "loss.triplet"),
        ({}, "loss.triplet.enabled"),
        ({"enabled": "true"}, "loss.triplet.enabled"),
        (
            {"enabled": True, "weight": 1.0, "normalize_features": True},
            "loss.triplet.margin",
        ),
        (
            {"enabled": True, "margin": 0.0, "weight": 1.0, "normalize_features": True},
            "loss.triplet.margin",
        ),
        (
            {"enabled": True, "margin": 0.3, "weight": 0.0, "normalize_features": True},
            "loss.triplet.weight",
        ),
        (
            {"enabled": True, "margin": 0.3, "weight": 1.0, "normalize_features": "yes"},
            "loss.triplet.normalize_features",
        ),
    ],
)
def test_validate_training_config_rejects_invalid_triplet_config(
    triplet_config: object,
    message: str,
) -> None:
    config = make_smoke_config()
    config["loss"]["triplet"] = triplet_config

    with pytest.raises(ValueError, match=message):
        validate_training_config(config)


def test_msmt17_random_erasing_config_matches_ms1_recipe() -> None:
    config = load_config("configs/resnet50_ce_pretrained_msmt17_random_erasing.yaml")

    assert config["run"]["name"] == "resnet50_ce_pretrained_msmt17_random_erasing"
    assert config["data"]["name"] == "msmt17_v1"
    assert config["data"]["random_erasing"] is True
    assert config["data"]["random_erasing_prob"] == pytest.approx(0.5)
    assert config["model"]["pretrained"] is True
    assert config["loss"]["label_smoothing"] == pytest.approx(0.0)
    assert config["eval"]["enabled"] is True
    assert config["eval"]["query_chunk_size"] == 256


@pytest.mark.parametrize(
    ("path", "run_name", "random_erasing"),
    [
        (
            "configs/resnet50_ce_triplet_pretrained_msmt17.yaml",
            "resnet50_ce_triplet_pretrained_msmt17",
            False,
        ),
        (
            "configs/resnet50_ce_triplet_pretrained_msmt17_random_erasing.yaml",
            "resnet50_ce_triplet_pretrained_msmt17_random_erasing",
            True,
        ),
    ],
)
def test_msmt17_triplet_pk_configs_match_ms2_recipe(
    path: str,
    run_name: str,
    random_erasing: bool,
) -> None:
    config = load_config(path)

    assert config["run"]["name"] == run_name
    assert config["data"]["name"] == "msmt17_v1"
    assert config["data"]["batch_size"] == 16
    assert config["data"]["random_erasing"] is random_erasing
    if random_erasing:
        assert config["data"]["random_erasing_prob"] == pytest.approx(0.5)
    assert config["model"]["pretrained"] is True
    assert config["sampler"]["name"] == "pk"
    assert config["sampler"]["num_pids"] == 8
    assert config["sampler"]["num_instances"] == 2
    assert config["loss"]["label_smoothing"] == pytest.approx(0.0)
    assert config["loss"]["triplet"]["enabled"] is True
    assert config["loss"]["triplet"]["margin"] == pytest.approx(0.3)
    assert config["loss"]["triplet"]["weight"] == pytest.approx(1.0)
    assert config["loss"]["triplet"]["normalize_features"] is True
    assert config["eval"]["enabled"] is True
    assert config["eval"]["query_chunk_size"] == 256


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
