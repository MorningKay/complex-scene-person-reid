"""Metric learning losses for Re-ID embeddings."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _validate_triplet_inputs(features: torch.Tensor, targets: torch.Tensor) -> None:
    if features.ndim != 2:
        raise ValueError(f"features must be a 2D tensor, got shape {tuple(features.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets must be a 1D tensor, got shape {tuple(targets.shape)}")
    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            "features and targets must have the same batch size, "
            f"got {features.shape[0]} and {targets.shape[0]}"
        )
    if targets.dtype != torch.long:
        raise ValueError(f"targets must have dtype torch.long, got {targets.dtype}")


def _pairwise_squared_distance(features: torch.Tensor) -> torch.Tensor:
    distmat = (
        features.pow(2).sum(dim=1, keepdim=True)
        + features.pow(2).sum(dim=1, keepdim=True).t()
        - 2 * features @ features.t()
    )
    return distmat.clamp_min(0)


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3, normalize_features: bool = False) -> None:
        super().__init__()
        if margin <= 0:
            raise ValueError("margin must be positive")

        self.margin = margin
        self.normalize_features = normalize_features

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_triplet_inputs(features, targets)
        if features.shape[0] == 0:
            return features.sum() * 0

        if self.normalize_features:
            features = F.normalize(features, p=2, dim=1)

        distmat = _pairwise_squared_distance(features)
        same_identity = targets.unsqueeze(0).eq(targets.unsqueeze(1))
        identity = torch.eye(
            targets.shape[0],
            dtype=torch.bool,
            device=targets.device,
        )
        positive_mask = same_identity & ~identity
        negative_mask = ~same_identity
        valid_anchor_mask = positive_mask.any(dim=1) & negative_mask.any(dim=1)

        if not valid_anchor_mask.any():
            return features.sum() * 0

        hardest_positive = distmat.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
        hardest_negative = distmat.masked_fill(~negative_mask, float("inf")).min(dim=1).values
        losses = F.relu(hardest_positive - hardest_negative + self.margin)
        return losses[valid_anchor_mask].mean()
