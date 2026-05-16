"""Retrieval metrics for Market-1501 style evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


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
    all_cmc: list[torch.Tensor] = []
    all_ap: list[float] = []

    for query_index in range(query_pids.numel()):
        query_pid = query_pids[query_index]
        query_camid = query_camids[query_index]
        order = indices[query_index]

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

    if not all_cmc:
        raise ValueError("No valid Market-1501 query with at least one gallery match")

    stacked_cmc = torch.stack(all_cmc, dim=0).mean(dim=0)
    mean_ap = float(torch.tensor(all_ap, dtype=torch.float32).mean())
    return RetrievalMetrics(cmc=stacked_cmc, mAP=mean_ap, num_valid_queries=len(all_cmc))
