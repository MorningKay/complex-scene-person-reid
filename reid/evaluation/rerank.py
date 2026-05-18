"""Sparse k-reciprocal re-ranking for Market-style Re-ID evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from reid.evaluation.metrics import (
    DistanceName,
    RetrievalMetrics,
    _compute_distance_chunk,
    _validate_feature_inputs,
    _validate_query_gallery_lengths,
    evaluate_market1501,
)

DEFAULT_RERANK_K1 = 20
DEFAULT_RERANK_K2 = 6
DEFAULT_RERANK_LAMBDA = 0.3
DEFAULT_RERANK_NEIGHBOR_CHUNK_SIZE = 128
DEFAULT_RERANK_QUERY_CHUNK_SIZE = 64
_EPS = 1e-12


SparseWeights = tuple[torch.Tensor, torch.Tensor]


def evaluate_market_style_reranking(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_pids: Sequence[int] | torch.Tensor,
    gallery_pids: Sequence[int] | torch.Tensor,
    query_camids: Sequence[int] | torch.Tensor,
    gallery_camids: Sequence[int] | torch.Tensor,
    distance: DistanceName = "cosine",
    max_rank: int = 50,
    k1: int = DEFAULT_RERANK_K1,
    k2: int = DEFAULT_RERANK_K2,
    lambda_value: float = DEFAULT_RERANK_LAMBDA,
    neighbor_chunk_size: int = DEFAULT_RERANK_NEIGHBOR_CHUNK_SIZE,
    rerank_query_chunk_size: int = DEFAULT_RERANK_QUERY_CHUNK_SIZE,
    compute_device: str | torch.device | None = None,
) -> RetrievalMetrics:
    """Evaluate features with sparse k-reciprocal re-ranking.

    The implementation stores only top-k neighborhoods and sparse reciprocal
    weights. Query-gallery final distances are materialized in query chunks.
    """

    _validate_rerank_inputs(
        distance=distance,
        max_rank=max_rank,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value,
        neighbor_chunk_size=neighbor_chunk_size,
        rerank_query_chunk_size=rerank_query_chunk_size,
    )
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

    num_query = int(query_features.shape[0])
    num_gallery = int(gallery_features.shape[0])
    max_rank = min(max_rank, num_gallery)
    all_features = torch.cat([query_features, gallery_features], dim=0).to(
        dtype=torch.float32,
        device="cpu",
    )
    features_for_distance = _prepare_features(all_features, distance)
    device = torch.device(compute_device) if compute_device is not None else all_features.device

    top_k = min(
        all_features.shape[0],
        max(k1 + 1, k2, int(round(k1 / 2)) + 1),
    )
    neighbor_indices, row_max = _compute_topk_neighbors(
        features=features_for_distance,
        distance=distance,
        top_k=top_k,
        chunk_size=neighbor_chunk_size,
        compute_device=device,
    )
    reciprocal_weights = _build_reciprocal_weights(
        features=features_for_distance,
        neighbor_indices=neighbor_indices,
        row_max=row_max,
        distance=distance,
        k1=k1,
    )
    reciprocal_weights = _apply_query_expansion(
        reciprocal_weights,
        neighbor_indices=neighbor_indices,
        k2=k2,
    )
    inverted_indices, inverted_values = _build_gallery_inverted_index(
        reciprocal_weights[num_query:],
        num_total=all_features.shape[0],
    )

    gallery_on_device = features_for_distance[num_query:].to(device=device, dtype=torch.float32)
    cmc_sum: torch.Tensor | None = None
    ap_sum = 0.0
    num_valid_queries = 0
    for start in range(0, num_query, rerank_query_chunk_size):
        end = min(start + rerank_query_chunk_size, num_query)
        query_chunk = features_for_distance[start:end].to(device=device, dtype=torch.float32)
        original_dist = _compute_distance_chunk(query_chunk, gallery_on_device, distance).cpu()
        original_dist = original_dist / row_max[start:end].unsqueeze(1).clamp_min(_EPS)
        final_dist_rows = []
        for local_index, query_index in enumerate(range(start, end)):
            jaccard = _compute_jaccard_distances(
                reciprocal_weights[query_index],
                inverted_indices=inverted_indices,
                inverted_values=inverted_values,
                num_gallery=num_gallery,
            )
            final_dist_rows.append(
                lambda_value * original_dist[local_index] + (1 - lambda_value) * jaccard
            )
        final_dist = torch.stack(final_dist_rows, dim=0)
        try:
            chunk_metrics = evaluate_market1501(
                final_dist,
                query_pids=query_pids[start:end],
                gallery_pids=gallery_pids,
                query_camids=query_camids[start:end],
                gallery_camids=gallery_camids,
                max_rank=max_rank,
            )
        except ValueError as exc:
            if "No valid Market-1501 query" in str(exc):
                continue
            raise

        valid = chunk_metrics.num_valid_queries
        if cmc_sum is None:
            cmc_sum = torch.zeros_like(chunk_metrics.cmc)
        cmc_sum += chunk_metrics.cmc * valid
        ap_sum += chunk_metrics.mAP * valid
        num_valid_queries += valid

    if cmc_sum is None or num_valid_queries == 0:
        raise ValueError("No valid Market-style query with at least one gallery match")

    return RetrievalMetrics(
        cmc=cmc_sum / num_valid_queries,
        mAP=ap_sum / num_valid_queries,
        num_valid_queries=num_valid_queries,
    )


def _validate_rerank_inputs(
    distance: str,
    max_rank: int,
    k1: int,
    k2: int,
    lambda_value: float,
    neighbor_chunk_size: int,
    rerank_query_chunk_size: int,
) -> None:
    if distance not in {"cosine", "euclidean"}:
        raise ValueError("distance must be one of: cosine, euclidean")
    for value, name in (
        (max_rank, "max_rank"),
        (k1, "rerank_k1"),
        (k2, "rerank_k2"),
        (neighbor_chunk_size, "rerank_neighbor_chunk_size"),
        (rerank_query_chunk_size, "rerank_query_chunk_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(lambda_value, bool) or not isinstance(lambda_value, (int, float)):
        raise ValueError("rerank_lambda must be a number in the range [0, 1]")
    if float(lambda_value) < 0 or float(lambda_value) > 1:
        raise ValueError("rerank_lambda must be in the range [0, 1]")


def _as_1d_long_tensor(values: Sequence[int] | torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.long)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(tensor.shape)}")
    return tensor.cpu()


def _prepare_features(features: torch.Tensor, distance: DistanceName) -> torch.Tensor:
    features = features.to(dtype=torch.float32, device="cpu")
    if distance == "cosine":
        return F.normalize(features, p=2, dim=1, eps=_EPS)
    return features


def _compute_topk_neighbors(
    features: torch.Tensor,
    distance: DistanceName,
    top_k: int,
    chunk_size: int,
    compute_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features_on_device = features.to(device=compute_device, dtype=torch.float32)
    neighbor_indices: list[torch.Tensor] = []
    row_max: list[torch.Tensor] = []
    for start in range(0, features.shape[0], chunk_size):
        end = min(start + chunk_size, features.shape[0])
        chunk = features_on_device[start:end]
        distances = _compute_distance_chunk(chunk, features_on_device, distance)
        row_max.append(distances.max(dim=1).values.detach().cpu().clamp_min(_EPS))
        _values, indices = torch.topk(
            distances,
            k=top_k,
            dim=1,
            largest=False,
            sorted=True,
        )
        neighbor_indices.append(indices.detach().cpu())
    return torch.cat(neighbor_indices, dim=0), torch.cat(row_max, dim=0)


def _build_reciprocal_weights(
    features: torch.Tensor,
    neighbor_indices: torch.Tensor,
    row_max: torch.Tensor,
    distance: DistanceName,
    k1: int,
) -> list[SparseWeights]:
    num_total = int(neighbor_indices.shape[0])
    full_k = min(k1 + 1, int(neighbor_indices.shape[1]))
    half_k = min(int(round(k1 / 2)) + 1, int(neighbor_indices.shape[1]))
    full_neighbor_sets = [
        set(int(item) for item in row[:full_k].tolist()) for row in neighbor_indices
    ]
    half_neighbor_sets = [
        set(int(item) for item in row[:half_k].tolist()) for row in neighbor_indices
    ]

    rows: list[SparseWeights] = []
    for row_index in range(num_total):
        forward = [int(item) for item in neighbor_indices[row_index, :full_k].tolist()]
        reciprocal = {
            candidate for candidate in forward if row_index in full_neighbor_sets[candidate]
        }
        expansion = set(reciprocal)
        for candidate in reciprocal:
            candidate_forward = [
                int(item) for item in neighbor_indices[candidate, :half_k].tolist()
            ]
            candidate_reciprocal = {
                item for item in candidate_forward if candidate in half_neighbor_sets[item]
            }
            if not candidate_reciprocal:
                continue
            overlap = len(candidate_reciprocal.intersection(reciprocal))
            if overlap > (2 / 3) * len(candidate_reciprocal):
                expansion.update(candidate_reciprocal)

        if not expansion:
            expansion.add(row_index)
        indices = torch.tensor(sorted(expansion), dtype=torch.long)
        distances = _distance_to_indices(features, row_index, indices, row_max, distance)
        values = torch.exp(-distances)
        values = values / values.sum().clamp_min(_EPS)
        rows.append((indices, values.cpu()))
    return rows


def _distance_to_indices(
    features: torch.Tensor,
    row_index: int,
    indices: torch.Tensor,
    row_max: torch.Tensor,
    distance: DistanceName,
) -> torch.Tensor:
    row = features[row_index : row_index + 1]
    candidates = features[indices]
    distances = _compute_distance_chunk(row, candidates, distance).flatten().cpu()
    return distances / row_max[row_index].clamp_min(_EPS)


def _apply_query_expansion(
    rows: list[SparseWeights],
    neighbor_indices: torch.Tensor,
    k2: int,
) -> list[SparseWeights]:
    if k2 <= 1:
        return rows

    qe_k = min(k2, int(neighbor_indices.shape[1]))
    expanded_rows: list[SparseWeights] = []
    for row_index in range(len(rows)):
        accumulator: dict[int, float] = {}
        for neighbor in neighbor_indices[row_index, :qe_k].tolist():
            indices, values = rows[int(neighbor)]
            for index, value in zip(indices.tolist(), values.tolist()):
                accumulator[int(index)] = accumulator.get(int(index), 0.0) + float(value) / qe_k
        sorted_items = sorted(accumulator.items())
        expanded_rows.append(
            (
                torch.tensor([item[0] for item in sorted_items], dtype=torch.long),
                torch.tensor([item[1] for item in sorted_items], dtype=torch.float32),
            )
        )
    return expanded_rows


def _build_gallery_inverted_index(
    gallery_rows: list[SparseWeights],
    num_total: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    inverted_indices: list[list[int]] = [[] for _ in range(num_total)]
    inverted_values: list[list[float]] = [[] for _ in range(num_total)]
    for gallery_index, (indices, values) in enumerate(gallery_rows):
        for index, value in zip(indices.tolist(), values.tolist()):
            inverted_indices[int(index)].append(gallery_index)
            inverted_values[int(index)].append(float(value))

    return (
        [torch.tensor(items, dtype=torch.long) for items in inverted_indices],
        [torch.tensor(items, dtype=torch.float32) for items in inverted_values],
    )


def _compute_jaccard_distances(
    query_weights: SparseWeights,
    inverted_indices: list[torch.Tensor],
    inverted_values: list[torch.Tensor],
    num_gallery: int,
) -> torch.Tensor:
    intersections = torch.zeros(num_gallery, dtype=torch.float32)
    query_indices, query_values = query_weights
    for index, query_value in zip(query_indices.tolist(), query_values.tolist()):
        gallery_indices = inverted_indices[int(index)]
        if gallery_indices.numel() == 0:
            continue
        gallery_values = inverted_values[int(index)]
        intersections[gallery_indices] += torch.minimum(
            gallery_values,
            torch.full_like(gallery_values, float(query_value)),
        )

    intersections = intersections.clamp(min=0.0, max=1.0)
    return 1 - intersections / (2 - intersections).clamp_min(_EPS)
