"""Model registry for Re-ID architectures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from reid.models.osnet_reid import osnet_x1_0_reid
from reid.models.resnet_reid import resnet50_reid
from reid.models.vit_reid import vit_patch16_global_local_reid

Config = Mapping[str, Any]


def normalize_model_name(name: str | None) -> str:
    if name is None or name == "":
        return "resnet50"
    normalized = name.lower().replace("-", "_")
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
    return aliases.get(normalized, normalized)


def build_reid_model(config: Config, load_pretrained: bool = True) -> nn.Module:
    model_config = _model_section(config)
    model_name = normalize_model_name(model_config.get("name"))
    num_classes = int(model_config["num_classes"])
    feature_dim = int(model_config["feature_dim"])
    pretrained = bool(model_config.get("pretrained", False)) and load_pretrained

    if model_name == "resnet50":
        return resnet50_reid(
            num_classes=num_classes,
            feature_dim=feature_dim,
            last_stride=int(model_config.get("last_stride", 1)),
            pretrained=pretrained,
        )
    if model_name == "osnet_x1_0":
        return osnet_x1_0_reid(
            num_classes=num_classes,
            feature_dim=feature_dim,
            pretrained=pretrained,
            pretrained_path=model_config.get("pretrained_path"),
        )
    if model_name == "vit_patch16_global_local":
        image_size = tuple(config.get("data", {}).get("image_size", (256, 128)))
        return vit_patch16_global_local_reid(
            num_classes=num_classes,
            backbone_name=str(model_config["backbone_name"]),
            image_size=(int(image_size[0]), int(image_size[1])),
            patch_size=int(model_config.get("patch_size", 16)),
            num_parts=int(model_config.get("num_parts", 4)),
            feature_dim=feature_dim,
            pretrained=pretrained,
        )

    raise ValueError("model.name must be one of: resnet50, osnet_x1_0, vit_patch16_global_local")


def _model_section(config: Config) -> Config:
    model_config = config.get("model")
    if isinstance(model_config, Mapping):
        return model_config
    return config
