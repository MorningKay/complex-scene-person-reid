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
        sie_camera: bool = False,
        sie_num_cameras: int | None = None,
        sie_coefficient: float = 1.0,
        part_classifiers: bool = False,
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
        if sie_camera and (sie_num_cameras is None or sie_num_cameras <= 0):
            raise ValueError("sie_num_cameras must be positive when camera SIE is enabled")
        if sie_coefficient <= 0:
            raise ValueError("sie_coefficient must be positive")
        _validate_image_size(image_size, patch_size, num_parts)

        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_parts = num_parts
        self.feature_dim = feature_dim
        self.normalize_features = normalize_features
        self.sie_camera = sie_camera
        self.sie_num_cameras = sie_num_cameras if sie_camera else None
        self.sie_coefficient = sie_coefficient
        self.part_classifiers_enabled = part_classifiers

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
        if sie_camera:
            self.sie_camera_embeddings = nn.Embedding(int(sie_num_cameras), self.embed_dim)
        else:
            self.sie_camera_embeddings = None
        if part_classifiers:
            self.part_bnnecks = nn.ModuleList(
                nn.BatchNorm1d(self.embed_dim) for _ in range(num_parts)
            )
            for bnneck in self.part_bnnecks:
                bnneck.bias.requires_grad_(False)
            self.part_classifiers = nn.ModuleList(
                nn.Linear(self.embed_dim, num_classes, bias=False) for _ in range(num_parts)
            )
        else:
            self.part_bnnecks = nn.ModuleList()
            self.part_classifiers = nn.ModuleList()
        self._init_reid_head()

    def _init_reid_head(self) -> None:
        for module in (self.projection[0], self.classifier):
            if module is self.classifier:
                nn.init.normal_(module.weight, mean=0, std=0.001)
            else:
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        for module in self.part_classifiers:
            nn.init.normal_(module.weight, mean=0, std=0.001)
        for module in (self.projection[1], self.bnneck, *self.part_bnnecks):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        if self.sie_camera_embeddings is not None:
            nn.init.trunc_normal_(self.sie_camera_embeddings.weight, std=0.02)

    def extract_features(
        self,
        x: torch.Tensor,
        camids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features, _local_features = self._extract_features_and_parts(x, camids=camids)
        return features

    def _extract_features_and_parts(
        self,
        x: torch.Tensor,
        camids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(x.shape[-2:]) != self.image_size:
            raise ValueError(
                "Input image size must match the configured ViT image_size, "
                f"got {(x.shape[-2], x.shape[-1])} and expected {self.image_size}"
            )
        grid_h, grid_w = _runtime_patch_grid(x, self.patch_size, self.num_parts)

        tokens = self._forward_backbone_tokens(x, camids=camids)
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
        return self.projection(fused), local_features

    def _forward_backbone_tokens(
        self,
        x: torch.Tensor,
        camids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.sie_camera_embeddings is None:
            return self.backbone.forward_features(x)

        if camids is None:
            raise ValueError("Camera SIE requires camids in model forward")
        if camids.ndim != 1 or camids.shape[0] != x.shape[0]:
            raise ValueError("camids must be a 1D tensor with one value per image")
        camids = camids.to(device=x.device, dtype=torch.long)
        if camids.numel() > 0 and (
            int(camids.min().item()) < 0 or int(camids.max().item()) >= int(self.sie_num_cameras)
        ):
            raise ValueError("camids are outside the configured SIE camera range")

        patch_embed = getattr(self.backbone, "patch_embed", None)
        pos_embed = getattr(self.backbone, "_pos_embed", None)
        blocks = getattr(self.backbone, "blocks", None)
        norm = getattr(self.backbone, "norm", None)
        if patch_embed is None or pos_embed is None or blocks is None or norm is None:
            raise ValueError("Camera SIE requires a timm VisionTransformer-style backbone")

        tokens = patch_embed(x)
        tokens = pos_embed(tokens)
        patch_drop = getattr(self.backbone, "patch_drop", nn.Identity())
        norm_pre = getattr(self.backbone, "norm_pre", nn.Identity())
        tokens = patch_drop(tokens)
        tokens = norm_pre(tokens)
        tokens = tokens + self.sie_coefficient * self.sie_camera_embeddings(camids).unsqueeze(1)
        tokens = blocks(tokens)
        return norm(tokens)

    def forward(
        self,
        x: torch.Tensor,
        return_feature: bool = False,
        camids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]] | torch.Tensor:
        features, local_features = self._extract_features_and_parts(x, camids=camids)
        bn_features = self.bnneck(features)

        if return_feature:
            if self.normalize_features:
                return F.normalize(bn_features, p=2, dim=1)
            return bn_features

        logits = self.classifier(bn_features)
        if self.part_classifiers_enabled:
            part_logits = tuple(
                classifier(bnneck(local_features[:, index]))
                for index, (bnneck, classifier) in enumerate(
                    zip(self.part_bnnecks, self.part_classifiers, strict=True)
                )
            )
            return logits, features, part_logits
        return logits, features


def vit_patch16_global_local_reid(
    num_classes: int,
    backbone_name: str = "deit_small_patch16_224",
    image_size: tuple[int, int] = (256, 128),
    patch_size: int = 16,
    num_parts: int = 4,
    feature_dim: int = 512,
    pretrained: bool = False,
    sie_camera: bool = False,
    sie_num_cameras: int | None = None,
    sie_coefficient: float = 1.0,
    part_classifiers: bool = False,
) -> ViTPatch16GlobalLocalReID:
    return ViTPatch16GlobalLocalReID(
        num_classes=num_classes,
        backbone_name=backbone_name,
        image_size=image_size,
        patch_size=patch_size,
        num_parts=num_parts,
        feature_dim=feature_dim,
        pretrained=pretrained,
        sie_camera=sie_camera,
        sie_num_cameras=sie_num_cameras,
        sie_coefficient=sie_coefficient,
        part_classifiers=part_classifiers,
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
