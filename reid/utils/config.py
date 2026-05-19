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
    "model": ("num_classes", "feature_dim"),
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
    if "random_erasing" in config["data"] and not isinstance(
        config["data"]["random_erasing"], bool
    ):
        raise ValueError("data.random_erasing must be a boolean when provided")
    random_erasing_prob = config["data"].get("random_erasing_prob")
    if random_erasing_prob is not None:
        _ensure_probability(random_erasing_prob, "data.random_erasing_prob")
    padding = config["data"].get("padding")
    if padding is not None:
        _ensure_non_negative_int(padding, "data.padding")
    _ensure_positive_int(config["data"]["batch_size"], "data.batch_size")
    _ensure_non_negative_int(config["data"]["num_workers"], "data.num_workers")
    _validate_sampler_config(config)
    _validate_model_config(config)
    _ensure_positive_int(config["model"]["num_classes"], "model.num_classes")
    _ensure_positive_int(config["model"]["feature_dim"], "model.feature_dim")
    _ensure_positive_int(config["train"]["epochs"], "train.epochs")
    _validate_optimizer_config(config)
    _validate_scheduler_config(config)
    _validate_eval_config(config)

    max_batches = config["train"].get("max_batches")
    if max_batches is not None:
        _ensure_positive_int(max_batches, "train.max_batches")
    if "amp" in config["train"] and not isinstance(config["train"]["amp"], bool):
        raise ValueError("train.amp must be a boolean when provided")
    grad_clip_norm = config["train"].get("grad_clip_norm")
    if grad_clip_norm is not None:
        _ensure_positive_float(grad_clip_norm, "train.grad_clip_norm")

    _ensure_non_negative_float(config["loss"]["label_smoothing"], "loss.label_smoothing")
    part_weight = config["loss"].get("part_weight")
    if part_weight is not None:
        _ensure_positive_float(part_weight, "loss.part_weight")
    _validate_triplet_config(config)


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


def _ensure_probability(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number in the range (0, 1]")
    if value <= 0 or value > 1:
        raise ValueError(f"{name} must be in the range (0, 1]")


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

    for key in ("max_query", "max_gallery", "query_chunk_size"):
        value = eval_config.get(key)
        if value is not None:
            _ensure_positive_int(value, f"eval.{key}")


def _validate_optimizer_config(config: Config) -> None:
    optimizer_config = config["optimizer"]
    name = optimizer_config.get("name", "adam")
    if not isinstance(name, str):
        raise ValueError("optimizer.name must be a string when provided")
    if name.lower() not in {"adam", "adamw"}:
        raise ValueError("optimizer.name must be one of: adam, adamw")

    _ensure_positive_float(optimizer_config["lr"], "optimizer.lr")
    _ensure_non_negative_float(optimizer_config["weight_decay"], "optimizer.weight_decay")


def _validate_model_config(config: Config) -> None:
    model_config = config["model"]
    model_name = model_config.get("name", "resnet50")
    if not isinstance(model_name, str):
        raise ValueError("model.name must be a string when provided")
    normalized_model_name = model_name.lower().replace("-", "_")
    aliases = {
        "resnet": "resnet50",
        "resnet50": "resnet50",
        "resnet_50": "resnet50",
        "osnet": "osnet_x1_0",
        "osnet_x1_0": "osnet_x1_0",
        "vit": "vit_patch16_global_local",
        "deit": "vit_patch16_global_local",
        "vit_patch16_global_local": "vit_patch16_global_local",
    }
    normalized_model_name = aliases.get(normalized_model_name, normalized_model_name)
    if normalized_model_name not in {
        "resnet50",
        "osnet_x1_0",
        "vit_patch16_global_local",
    }:
        raise ValueError("model.name must be one of: resnet50, osnet_x1_0, vit_patch16_global_local")

    last_stride = model_config.get("last_stride")
    if last_stride is not None:
        _ensure_positive_int(last_stride, "model.last_stride")
    if "pretrained" in model_config and not isinstance(model_config["pretrained"], bool):
        raise ValueError("model.pretrained must be a boolean when provided")
    pretrained_path = model_config.get("pretrained_path")
    if pretrained_path is not None and not isinstance(pretrained_path, str):
        raise ValueError("model.pretrained_path must be a string when provided")

    if normalized_model_name == "vit_patch16_global_local":
        backbone_name = model_config.get("backbone_name")
        if not isinstance(backbone_name, str) or backbone_name == "":
            raise ValueError("model.backbone_name must be a non-empty string")
        patch_size = model_config.get("patch_size", 16)
        num_parts = model_config.get("num_parts", 4)
        _ensure_positive_int(patch_size, "model.patch_size")
        _ensure_positive_int(num_parts, "model.num_parts")
        sie_camera = model_config.get("sie_camera", False)
        if not isinstance(sie_camera, bool):
            raise ValueError("model.sie_camera must be a boolean when provided")
        if sie_camera:
            if "sie_num_cameras" not in model_config:
                raise ValueError("Missing required config key: model.sie_num_cameras")
            _ensure_positive_int(model_config["sie_num_cameras"], "model.sie_num_cameras")
        sie_coefficient = model_config.get("sie_coefficient")
        if sie_coefficient is not None:
            _ensure_positive_float(sie_coefficient, "model.sie_coefficient")
        part_classifiers = model_config.get("part_classifiers", False)
        if not isinstance(part_classifiers, bool):
            raise ValueError("model.part_classifiers must be a boolean when provided")


def _validate_sampler_config(config: Config) -> None:
    if "sampler" not in config:
        return

    sampler_config = config["sampler"]
    if not isinstance(sampler_config, dict):
        raise ValueError("Config section must be a mapping: sampler")

    name = sampler_config.get("name")
    if name != "pk":
        raise ValueError("sampler.name must be one of: pk")

    for key in ("num_pids", "num_instances"):
        if key not in sampler_config:
            raise ValueError(f"Missing required config key: sampler.{key}")
        _ensure_positive_int(sampler_config[key], f"sampler.{key}")

    batches_per_epoch = sampler_config.get("batches_per_epoch")
    if batches_per_epoch is not None:
        _ensure_positive_int(batches_per_epoch, "sampler.batches_per_epoch")

    expected_batch_size = sampler_config["num_pids"] * sampler_config["num_instances"]
    if config["data"]["batch_size"] != expected_batch_size:
        raise ValueError("data.batch_size must equal sampler.num_pids * sampler.num_instances")


def _validate_scheduler_config(config: Config) -> None:
    if "scheduler" not in config:
        return

    scheduler_config = config["scheduler"]
    if not isinstance(scheduler_config, dict):
        raise ValueError("Config section must be a mapping: scheduler")

    name = scheduler_config.get("name")
    if name != "cosine":
        raise ValueError("scheduler.name must be one of: cosine")

    for key in ("min_lr", "warmup_epochs", "warmup_factor"):
        if key not in scheduler_config:
            raise ValueError(f"Missing required config key: scheduler.{key}")

    min_lr = scheduler_config["min_lr"]
    _ensure_non_negative_float(min_lr, "scheduler.min_lr")
    if float(min_lr) > float(config["optimizer"]["lr"]):
        raise ValueError("scheduler.min_lr must be less than or equal to optimizer.lr")

    _ensure_non_negative_int(scheduler_config["warmup_epochs"], "scheduler.warmup_epochs")
    if scheduler_config["warmup_epochs"] > config["train"]["epochs"]:
        raise ValueError("scheduler.warmup_epochs must be less than or equal to train.epochs")

    warmup_factor = scheduler_config["warmup_factor"]
    _ensure_positive_float(warmup_factor, "scheduler.warmup_factor")
    if float(warmup_factor) > 1:
        raise ValueError("scheduler.warmup_factor must be less than or equal to 1")


def _validate_triplet_config(config: Config) -> None:
    triplet_config = config["loss"].get("triplet")
    if triplet_config is None:
        return
    if not isinstance(triplet_config, dict):
        raise ValueError("loss.triplet must be a mapping when provided")
    if "enabled" not in triplet_config:
        raise ValueError("Missing required config key: loss.triplet.enabled")
    if not isinstance(triplet_config["enabled"], bool):
        raise ValueError("loss.triplet.enabled must be a boolean")
    if not triplet_config["enabled"]:
        return

    for key in ("margin", "weight", "normalize_features"):
        if key not in triplet_config:
            raise ValueError(f"Missing required config key: loss.triplet.{key}")

    _ensure_positive_float(triplet_config["margin"], "loss.triplet.margin")
    _ensure_positive_float(triplet_config["weight"], "loss.triplet.weight")
    if not isinstance(triplet_config["normalize_features"], bool):
        raise ValueError("loss.triplet.normalize_features must be a boolean")
