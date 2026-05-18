"""Self-written OSNet baseline for person re-identification."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        instance_norm: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        if instance_norm:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
            groups=out_channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.relu(self.bn(x))


class ChannelGate(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_gates: int | None = None,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        if reduction <= 0:
            raise ValueError("reduction must be positive")
        if num_gates is None:
            num_gates = in_channels

        hidden_channels = max(1, in_channels // reduction)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_channels, num_gates, 1)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.global_avgpool(x)
        gates = self.relu(self.fc1(gates))
        gates = self.gate_activation(self.fc2(gates))
        return x * gates


class OSBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        instance_norm: bool = False,
        bottleneck_reduction: int = 4,
    ) -> None:
        super().__init__()
        if bottleneck_reduction <= 0:
            raise ValueError("bottleneck_reduction must be positive")

        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.instance_norm = None
        if instance_norm:
            self.instance_norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = (
            self.gate(self.conv2a(x))
            + self.gate(self.conv2b(x))
            + self.gate(self.conv2c(x))
            + self.gate(self.conv2d(x))
        )
        x = self.conv3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x + identity
        if self.instance_norm is not None:
            out = self.instance_norm(out)
        return F.relu(out)


class OSNetReID(nn.Module):
    def __init__(
        self,
        num_classes: int,
        blocks: list[type[OSBlock]],
        layers: list[int],
        channels: list[int],
        feature_dim: int = 512,
        normalize_features: bool = True,
        instance_norm: bool = False,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if len(blocks) != 3 or len(layers) != 3 or len(channels) != 4:
            raise ValueError("OSNet x1.0 expects 3 block groups and 4 channel entries")

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.normalize_features = normalize_features

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, instance_norm=instance_norm)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            instance_norm=instance_norm,
        )
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True,
        )
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False,
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.bnneck = nn.BatchNorm1d(feature_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

        self._init_parameters()

    def _make_layer(
        self,
        block: type[OSBlock],
        num_layers: int,
        in_channels: int,
        out_channels: int,
        reduce_spatial_size: bool,
        instance_norm: bool = False,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [block(in_channels, out_channels, instance_norm=instance_norm)]
        for _ in range(1, num_layers):
            layers.append(block(out_channels, out_channels, instance_norm=instance_norm))
        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2),
                )
            )
        return nn.Sequential(*layers)

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                if module is self.classifier:
                    nn.init.normal_(module.weight, mean=0, std=0.001)
                else:
                    nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def featuremaps(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.featuremaps(x)
        x = self.global_avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def forward(
        self,
        x: torch.Tensor,
        return_feature: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        features = self.extract_features(x)
        bn_features = self.bnneck(features)

        if return_feature:
            if self.normalize_features:
                return F.normalize(bn_features, p=2, dim=1)
            return bn_features

        logits = self.classifier(bn_features)
        return logits, features


def osnet_x1_0_reid(
    num_classes: int,
    feature_dim: int = 512,
    pretrained: bool = False,
    pretrained_path: str | Path | None = None,
) -> OSNetReID:
    model = OSNetReID(
        num_classes=num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        feature_dim=feature_dim,
    )
    if pretrained:
        if pretrained_path is None:
            raise ValueError("OSNet pretrained loading requires model.pretrained_path")
        load_imagenet_osnet_weights(model, pretrained_path)
    return model


def load_imagenet_osnet_weights(
    model: OSNetReID,
    pretrained_path: str | Path,
) -> tuple[str, ...]:
    path = Path(pretrained_path)
    if not path.is_file():
        raise FileNotFoundError(f"OSNet pretrained checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    source_state_dict = _extract_state_dict(checkpoint)
    return load_osnet_backbone_state_dict(model, source_state_dict)


def load_osnet_backbone_state_dict(
    model: OSNetReID,
    source_state_dict: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    target_state_dict = model.state_dict()
    compatible_state_dict = {}
    loaded_keys: list[str] = []

    for raw_name, tensor in source_state_dict.items():
        name = raw_name[7:] if raw_name.startswith("module.") else raw_name
        if name.startswith(_OSNET_HEAD_PREFIXES):
            continue
        if name not in target_state_dict:
            continue
        if target_state_dict[name].shape != tensor.shape:
            continue
        compatible_state_dict[name] = tensor.detach().clone()
        loaded_keys.append(name)

    if not loaded_keys:
        raise ValueError("No compatible OSNet pretrained weights were found")

    model.load_state_dict(compatible_state_dict, strict=False)
    return tuple(loaded_keys)


def _extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("OSNet pretrained checkpoint must contain a state dict")


_OSNET_HEAD_PREFIXES = (
    "classifier.",
    "bnneck.",
)
