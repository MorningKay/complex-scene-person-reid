"""YAML configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]

_REQUIRED_TOP_LEVEL_KEYS = (
    "run",
    "data",
    "model",
    "loss",
    "optimizer",
    "train",
)
_REQUIRED_NESTED_KEYS = {
    "run": ("name", "seed"),
    "data": ("root", "batch_size", "num_workers"),
    "model": ("num_classes", "feature_dim", "last_stride"),
    "loss": ("label_smoothing",),
    "optimizer": ("lr", "weight_decay"),
    "train": ("epochs",),
}


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    validate_training_config(config)
    return config


def write_config(config: Config, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def validate_training_config(config: Config) -> None:
    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
        if not isinstance(config[key], dict):
            raise ValueError(f"Config section must be a mapping: {key}")

    for section, keys in _REQUIRED_NESTED_KEYS.items():
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing required config key: {section}.{key}")

    data_name = config["data"].get("name")
    if data_name is not None and not isinstance(data_name, str):
        raise ValueError("data.name must be a string when provided")
    _ensure_positive_int(config["data"]["batch_size"], "data.batch_size")
    _ensure_non_negative_int(config["data"]["num_workers"], "data.num_workers")
    _ensure_positive_int(config["model"]["num_classes"], "model.num_classes")
    _ensure_positive_int(config["model"]["feature_dim"], "model.feature_dim")
    _ensure_positive_int(config["model"]["last_stride"], "model.last_stride")
    if "pretrained" in config["model"] and not isinstance(config["model"]["pretrained"], bool):
        raise ValueError("model.pretrained must be a boolean when provided")
    _ensure_positive_int(config["train"]["epochs"], "train.epochs")
    _validate_eval_config(config)

    max_batches = config["train"].get("max_batches")
    if max_batches is not None:
        _ensure_positive_int(max_batches, "train.max_batches")

    _ensure_non_negative_float(config["loss"]["label_smoothing"], "loss.label_smoothing")
    _ensure_positive_float(config["optimizer"]["lr"], "optimizer.lr")
    _ensure_non_negative_float(config["optimizer"]["weight_decay"], "optimizer.weight_decay")


def _ensure_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _ensure_non_negative_int(value: object, name: str) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _ensure_positive_float(value: object, name: str) -> None:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")


def _ensure_non_negative_float(value: object, name: str) -> None:
    if not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")


def _validate_eval_config(config: Config) -> None:
    if "eval" not in config:
        return

    eval_config = config["eval"]
    if not isinstance(eval_config, dict):
        raise ValueError("Config section must be a mapping: eval")
    if "enabled" not in eval_config:
        raise ValueError("Missing required config key: eval.enabled")
    if not isinstance(eval_config["enabled"], bool):
        raise ValueError("eval.enabled must be a boolean")
    if not eval_config["enabled"]:
        return

    for key in ("interval", "batch_size", "num_workers", "distance"):
        if key not in eval_config:
            raise ValueError(f"Missing required config key: eval.{key}")

    _ensure_positive_int(eval_config["interval"], "eval.interval")
    _ensure_positive_int(eval_config["batch_size"], "eval.batch_size")
    _ensure_non_negative_int(eval_config["num_workers"], "eval.num_workers")
    if eval_config["interval"] > config["train"]["epochs"]:
        raise ValueError("eval.interval must be less than or equal to train.epochs")
    if eval_config["distance"] not in {"cosine", "euclidean"}:
        raise ValueError("eval.distance must be one of: cosine, euclidean")

    for key in ("max_query", "max_gallery"):
        value = eval_config.get(key)
        if value is not None:
            _ensure_positive_int(value, f"eval.{key}")
