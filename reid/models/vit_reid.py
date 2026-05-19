"""Timm-based Vision Transformer Re-ID model with global-local pooling."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

try:
    import timm
except ImportError as exc:  # pragma: no cover - exercised only when env is incomplete.
    timm = None  # type: ignore[assignment]
    _TIMM_IMPORT_ERROR = exc
else:
    _TIMM_IMPORT_ERROR = None


class ViTPatch16GlobalLocalReID(nn.Module):
    """Transformer Re-ID head using CLS and vertical patch part features."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str,
        image_size: tuple[int, int] = (256, 128),
        patch_size: int = 16,
        num_parts: int = 4,
        feature_dim: int = 512,
        pretrained: bool = False,
        normalize_features: bool = True,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for ViT/DeiT Re-ID models") from _TIMM_IMPORT_ERROR
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if num_parts <= 0:
            raise ValueError("num_parts must be positive")
        _validate_image_size(image_size, patch_size, num_parts)

        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_parts = num_parts
        self.feature_dim = feature_dim
        self.normalize_features = normalize_features

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=image_size,
        )
        self.embed_dim = int(getattr(self.backbone, "num_features"))
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        self.grid_size = _patch_grid_from_backbone(self.backbone, image_size, patch_size)
        _validate_grid(self.grid_size, num_parts)

        fused_dim = self.embed_dim * (1 + num_parts)
        self.projection = nn.Sequential(
            nn.Linear(fused_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.bnneck = nn.BatchNorm1d(feature_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
        self._init_reid_head()

    def _init_reid_head(self) -> None:
        for module in (self.projection[0], self.classifier):
            if module is self.classifier:
                nn.init.normal_(module.weight, mean=0, std=0.001)
            else:
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        for module in (self.projection[1], self.bnneck):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-2:]) != self.image_size:
            raise ValueError(
                "Input image size must match the configured ViT image_size, "
                f"got {(x.shape[-2], x.shape[-1])} and expected {self.image_size}"
            )
        grid_h, grid_w = _runtime_patch_grid(x, self.patch_size, self.num_parts)

        tokens = self.backbone.forward_features(x)
        if not torch.is_tensor(tokens) or tokens.ndim != 3:
            raise ValueError("ViT backbone forward_features must return a 3D token tensor")
        if self.num_prefix_tokens < 1:
            raise ValueError("ViT global-local pooling requires a CLS prefix token")

        cls_feature = tokens[:, 0]
        patch_tokens = tokens[:, self.num_prefix_tokens :]
        expected_tokens = grid_h * grid_w
        if patch_tokens.shape[1] != expected_tokens:
            raise ValueError(
                "ViT patch token count does not match the configured image grid, "
                f"got {patch_tokens.shape[1]} and expected {expected_tokens}"
            )

        batch_size, _num_tokens, channels = patch_tokens.shape
        patch_tokens = patch_tokens.reshape(batch_size, grid_h, grid_w, channels)
        rows_per_part = grid_h // self.num_parts
        local_features = patch_tokens.reshape(
            batch_size,
            self.num_parts,
            rows_per_part,
            grid_w,
            channels,
        ).mean(dim=(2, 3))
        fused = torch.cat([cls_feature, local_features.flatten(1)], dim=1)
        return self.projection(fused)

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


def vit_patch16_global_local_reid(
    num_classes: int,
    backbone_name: str = "deit_small_patch16_224",
    image_size: tuple[int, int] = (256, 128),
    patch_size: int = 16,
    num_parts: int = 4,
    feature_dim: int = 512,
    pretrained: bool = False,
) -> ViTPatch16GlobalLocalReID:
    return ViTPatch16GlobalLocalReID(
        num_classes=num_classes,
        backbone_name=backbone_name,
        image_size=image_size,
        patch_size=patch_size,
        num_parts=num_parts,
        feature_dim=feature_dim,
        pretrained=pretrained,
    )


def _validate_image_size(
    image_size: tuple[int, int],
    patch_size: int,
    num_parts: int,
) -> None:
    if len(image_size) != 2:
        raise ValueError("image_size must contain height and width")
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image_size values must be positive")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image_size height and width must be divisible by patch_size")
    grid = (height // patch_size, width // patch_size)
    _validate_grid(grid, num_parts)


def _validate_grid(grid_size: tuple[int, int], num_parts: int) -> None:
    grid_h, grid_w = grid_size
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("ViT patch grid must be positive")
    if grid_h % num_parts != 0:
        raise ValueError("ViT patch grid height must be divisible by num_parts")


def _runtime_patch_grid(
    x: torch.Tensor,
    patch_size: int,
    num_parts: int,
) -> tuple[int, int]:
    if x.ndim != 4:
        raise ValueError("Input images must be a 4D tensor")
    height, width = int(x.shape[-2]), int(x.shape[-1])
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Input image height and width must be divisible by patch_size")
    grid = (height // patch_size, width // patch_size)
    _validate_grid(grid, num_parts)
    return grid


def _patch_grid_from_backbone(
    backbone: nn.Module,
    image_size: tuple[int, int],
    patch_size: int,
) -> tuple[int, int]:
    patch_embed = getattr(backbone, "patch_embed", None)
    grid_size = getattr(patch_embed, "grid_size", None)
    if isinstance(grid_size, Sequence) and len(grid_size) == 2:
        return int(grid_size[0]), int(grid_size[1])
    return image_size[0] // patch_size, image_size[1] // patch_size
