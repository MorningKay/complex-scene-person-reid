"""Distance helpers for retrieval features."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_feature_matrices(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
) -> None:
    if query_features.ndim != 2:
        raise ValueError(
            f"query_features must be a 2D tensor, got shape {tuple(query_features.shape)}"
        )
    if gallery_features.ndim != 2:
        raise ValueError(
            f"gallery_features must be a 2D tensor, got shape {tuple(gallery_features.shape)}"
        )
    if query_features.shape[1] != gallery_features.shape[1]:
        raise ValueError(
            "query_features and gallery_features must have the same feature dimension, "
            f"got {query_features.shape[1]} and {gallery_features.shape[1]}"
        )


def pairwise_distance(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
) -> torch.Tensor:
    """Return squared Euclidean distances between query and gallery features."""

    _validate_feature_matrices(query_features, gallery_features)
    distmat = (
        query_features.pow(2).sum(dim=1, keepdim=True)
        + gallery_features.pow(2).sum(dim=1, keepdim=True).t()
        - 2 * query_features @ gallery_features.t()
    )
    return distmat.clamp_min(0)


def cosine_distance(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return `1 - cosine_similarity` between query and gallery features."""

    _validate_feature_matrices(query_features, gallery_features)
    query_features = F.normalize(query_features, p=2, dim=1, eps=eps)
    gallery_features = F.normalize(gallery_features, p=2, dim=1, eps=eps)
    return 1 - query_features @ gallery_features.t()
