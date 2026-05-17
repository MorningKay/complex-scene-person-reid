"""Retrieval metrics for Market-1501 style evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F

DistanceName = Literal["cosine", "euclidean"]


@dataclass(frozen=True)
class RetrievalMetrics:
    cmc: torch.Tensor
    mAP: float
    num_valid_queries: int


def _as_1d_long_tensor(values: Sequence[int] | torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.long)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(tensor.shape)}")
    return tensor.cpu()


def _validate_market1501_inputs(
    distmat: torch.Tensor,
    query_pids: torch.Tensor,
    gallery_pids: torch.Tensor,
    query_camids: torch.Tensor,
    gallery_camids: torch.Tensor,
) -> None:
    if distmat.ndim != 2:
        raise ValueError(f"distmat must be 2D, got shape {tuple(distmat.shape)}")
    if distmat.shape[0] != query_pids.numel():
        raise ValueError(
            f"distmat has {distmat.shape[0]} query rows but query_pids has {query_pids.numel()} items"
        )
    if distmat.shape[1] != gallery_pids.numel():
        raise ValueError(
            "distmat has "
            f"{distmat.shape[1]} gallery columns but gallery_pids has {gallery_pids.numel()} items"
        )
    if query_camids.numel() != query_pids.numel():
        raise ValueError("query_camids must have the same length as query_pids")
    if gallery_camids.numel() != gallery_pids.numel():
        raise ValueError("gallery_camids must have the same length as gallery_pids")


def _validate_feature_inputs(
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


def _validate_query_gallery_lengths(
    num_query: int,
    num_gallery: int,
    query_pids: torch.Tensor,
    gallery_pids: torch.Tensor,
    query_camids: torch.Tensor,
    gallery_camids: torch.Tensor,
) -> None:
    if num_query != query_pids.numel():
        raise ValueError(
            f"query features contain {num_query} rows but query_pids has {query_pids.numel()} items"
        )
    if num_gallery != gallery_pids.numel():
        raise ValueError(
            "gallery features contain "
            f"{num_gallery} rows but gallery_pids has {gallery_pids.numel()} items"
        )
    if query_camids.numel() != query_pids.numel():
        raise ValueError("query_camids must have the same length as query_pids")
    if gallery_camids.numel() != gallery_pids.numel():
        raise ValueError("gallery_camids must have the same length as gallery_pids")


def evaluate_market1501(
    distmat: torch.Tensor,
    query_pids: Sequence[int] | torch.Tensor,
    gallery_pids: Sequence[int] | torch.Tensor,
    query_camids: Sequence[int] | torch.Tensor,
    gallery_camids: Sequence[int] | torch.Tensor,
    max_rank: int = 50,
) -> RetrievalMetrics:
    """Evaluate CMC and mAP using the Market-1501 retrieval protocol."""

    if max_rank <= 0:
        raise ValueError(f"max_rank must be positive, got {max_rank}")

    distmat = torch.as_tensor(distmat).detach().cpu()
    query_pids = _as_1d_long_tensor(query_pids, "query_pids")
    gallery_pids = _as_1d_long_tensor(gallery_pids, "gallery_pids")
    query_camids = _as_1d_long_tensor(query_camids, "query_camids")
    gallery_camids = _as_1d_long_tensor(gallery_camids, "gallery_camids")
    _validate_market1501_inputs(distmat, query_pids, gallery_pids, query_camids, gallery_camids)

    max_rank = min(max_rank, gallery_pids.numel())
    indices = torch.argsort(distmat, dim=1)
    return _evaluate_sorted_indices(
        indices=indices,
        query_pids=query_pids,
        gallery_pids=gallery_pids,
        query_camids=query_camids,
        gallery_camids=gallery_camids,
        max_rank=max_rank,
        empty_message="No valid Market-1501 query with at least one gallery match",
    )


def evaluate_market_style_retrieval(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_pids: Sequence[int] | torch.Tensor,
    gallery_pids: Sequence[int] | torch.Tensor,
    query_camids: Sequence[int] | torch.Tensor,
    gallery_camids: Sequence[int] | torch.Tensor,
    distance: DistanceName = "cosine",
    max_rank: int = 50,
    query_chunk_size: int = 256,
    compute_device: str | torch.device | None = None,
) -> RetrievalMetrics:
    """Evaluate CMC and mAP from features with Market-style filtering.

    Distance rows are computed and sorted in query chunks so large datasets can
    be evaluated without materializing a full query-by-gallery distance matrix.
    """

    if distance not in {"cosine", "euclidean"}:
        raise ValueError("distance must be one of: cosine, euclidean")
    if max_rank <= 0:
        raise ValueError(f"max_rank must be positive, got {max_rank}")
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")

    query_features = torch.as_tensor(query_features).detach()
    gallery_features = torch.as_tensor(gallery_features).detach()
    _validate_feature_inputs(query_features, gallery_features)
    query_pids = _as_1d_long_tensor(query_pids, "query_pids")
    gallery_pids = _as_1d_long_tensor(gallery_pids, "gallery_pids")
    query_camids = _as_1d_long_tensor(query_camids, "query_camids")
    gallery_camids = _as_1d_long_tensor(gallery_camids, "gallery_camids")
    _validate_query_gallery_lengths(
        num_query=query_features.shape[0],
        num_gallery=gallery_features.shape[0],
        query_pids=query_pids,
        gallery_pids=gallery_pids,
        query_camids=query_camids,
        gallery_camids=gallery_camids,
    )

    max_rank = min(max_rank, gallery_pids.numel())
    device = torch.device(compute_device) if compute_device is not None else query_features.device
    gallery_on_device = gallery_features.to(device=device, dtype=torch.float32)
    if distance == "cosine":
        gallery_on_device = F.normalize(gallery_on_device, p=2, dim=1, eps=1e-12)

    all_cmc: list[torch.Tensor] = []
    all_ap: list[float] = []
    for start in range(0, query_features.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, query_features.shape[0])
        query_chunk = query_features[start:end].to(device=device, dtype=torch.float32)
        distmat = _compute_distance_chunk(query_chunk, gallery_on_device, distance)
        indices = torch.argsort(distmat, dim=1).cpu()
        _append_query_metrics(
            indices=indices,
            query_offset=start,
            query_pids=query_pids,
            gallery_pids=gallery_pids,
            query_camids=query_camids,
            gallery_camids=gallery_camids,
            max_rank=max_rank,
            all_cmc=all_cmc,
            all_ap=all_ap,
        )

    return _finalize_metrics(
        all_cmc,
        all_ap,
        empty_message="No valid Market-style query with at least one gallery match",
    )


def _compute_distance_chunk(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    distance: DistanceName,
) -> torch.Tensor:
    if distance == "cosine":
        query_features = F.normalize(query_features, p=2, dim=1, eps=1e-12)
        return 1 - query_features @ gallery_features.t()
    return (
        query_features.pow(2).sum(dim=1, keepdim=True)
        + gallery_features.pow(2).sum(dim=1, keepdim=True).t()
        - 2 * query_features @ gallery_features.t()
    ).clamp_min(0)


def _evaluate_sorted_indices(
    indices: torch.Tensor,
    query_pids: torch.Tensor,
    gallery_pids: torch.Tensor,
    query_camids: torch.Tensor,
    gallery_camids: torch.Tensor,
    max_rank: int,
    empty_message: str,
) -> RetrievalMetrics:
    all_cmc: list[torch.Tensor] = []
    all_ap: list[float] = []
    _append_query_metrics(
        indices=indices,
        query_offset=0,
        query_pids=query_pids,
        gallery_pids=gallery_pids,
        query_camids=query_camids,
        gallery_camids=gallery_camids,
        max_rank=max_rank,
        all_cmc=all_cmc,
        all_ap=all_ap,
    )
    return _finalize_metrics(all_cmc, all_ap, empty_message=empty_message)


def _append_query_metrics(
    indices: torch.Tensor,
    query_offset: int,
    query_pids: torch.Tensor,
    gallery_pids: torch.Tensor,
    query_camids: torch.Tensor,
    gallery_camids: torch.Tensor,
    max_rank: int,
    all_cmc: list[torch.Tensor],
    all_ap: list[float],
) -> None:
    for row_index in range(indices.shape[0]):
        query_index = query_offset + row_index
        query_pid = query_pids[query_index]
        query_camid = query_camids[query_index]
        order = indices[row_index]

        remove = (gallery_pids[order] == -1) | (
            (gallery_pids[order] == query_pid) & (gallery_camids[order] == query_camid)
        )
        keep = ~remove
        matches = (gallery_pids[order][keep] == query_pid).to(torch.float32)
        if matches.numel() == 0 or matches.sum() == 0:
            continue

        cmc = matches.cumsum(dim=0).clamp(max=1)
        if cmc.numel() < max_rank:
            cmc = torch.nn.functional.pad(cmc, (0, max_rank - cmc.numel()), value=float(cmc[-1]))
        all_cmc.append(cmc[:max_rank])

        relevant_ranks = torch.nonzero(matches, as_tuple=False).flatten()
        precision_at_hits = matches.cumsum(dim=0)[relevant_ranks] / (relevant_ranks + 1).to(
            torch.float32
        )
        all_ap.append(float(precision_at_hits.mean()))


def _finalize_metrics(
    all_cmc: list[torch.Tensor],
    all_ap: list[float],
    empty_message: str,
) -> RetrievalMetrics:
    if not all_cmc:
        raise ValueError(empty_message)

    stacked_cmc = torch.stack(all_cmc, dim=0).mean(dim=0)
    mean_ap = float(torch.tensor(all_ap, dtype=torch.float32).mean())
    return RetrievalMetrics(cmc=stacked_cmc, mAP=mean_ap, num_valid_queries=len(all_cmc))
