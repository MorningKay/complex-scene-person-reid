"""Self-written ResNet baseline for person re-identification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        out_channels = bottleneck_channels * self.expansion

        self.conv1 = conv1x1(in_channels, bottleneck_channels)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = conv3x3(bottleneck_channels, bottleneck_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = conv1x1(bottleneck_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if downsample is None and (stride != 1 or in_channels != out_channels):
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNetReID(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 2048,
        last_stride: int = 1,
        normalize_features: bool = True,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.normalize_features = normalize_features
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3,
            self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, blocks=3)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=last_stride)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        backbone_dim = 512 * Bottleneck.expansion
        if feature_dim == backbone_dim:
            self.embedding = nn.Identity()
        else:
            self.embedding = nn.Linear(backbone_dim, feature_dim, bias=False)

        self.bnneck = nn.BatchNorm1d(feature_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

        self._init_parameters()

    def _make_layer(
        self,
        bottleneck_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        layers = [
            Bottleneck(
                in_channels=self.in_channels,
                bottleneck_channels=bottleneck_channels,
                stride=stride,
            )
        ]
        self.in_channels = bottleneck_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    in_channels=self.in_channels,
                    bottleneck_channels=bottleneck_channels,
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
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                if module is self.classifier:
                    nn.init.normal_(module.weight, mean=0, std=0.001)
                else:
                    nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.embedding(x)

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


def resnet50_reid(
    num_classes: int,
    feature_dim: int = 2048,
    last_stride: int = 1,
) -> ResNetReID:
    return ResNetReID(
        num_classes=num_classes,
        feature_dim=feature_dim,
        last_stride=last_stride,
    )
