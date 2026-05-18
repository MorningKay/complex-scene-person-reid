import pytest
import torch

from reid.evaluation.distance import pairwise_distance
from reid.evaluation.metrics import (
    RetrievalMetrics,
    evaluate_clothes_changing_retrieval,
    evaluate_market1501,
    evaluate_market_style_retrieval,
)
from reid.evaluation.rerank import evaluate_market_style_reranking


def test_evaluate_market1501_returns_perfect_scores_for_perfect_ranking() -> None:
    distmat = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]])
    metrics = evaluate_market1501(
        distmat,
        query_pids=[1, 2],
        gallery_pids=[1, 2, 3],
        query_camids=[0, 1],
        gallery_camids=[1, 0, 0],
        max_rank=3,
    )

    assert isinstance(metrics, RetrievalMetrics)
    torch.testing.assert_close(metrics.cmc, torch.tensor([1.0, 1.0, 1.0]))
    assert metrics.mAP == pytest.approx(1.0)
    assert metrics.num_valid_queries == 2


def test_evaluate_market1501_filters_same_identity_same_camera_gallery_entries() -> None:
    distmat = torch.tensor([[0.05, 0.1, 0.2, 0.3]])
    metrics = evaluate_market1501(
        distmat,
        query_pids=[7],
        gallery_pids=[7, 7, 8, -1],
        query_camids=[2],
        gallery_camids=[2, 3, 0, 1],
        max_rank=3,
    )

    torch.testing.assert_close(metrics.cmc, torch.tensor([1.0, 1.0, 1.0]))
    assert metrics.mAP == pytest.approx(1.0)
    assert metrics.num_valid_queries == 1


def test_evaluate_market1501_skips_queries_without_positive_gallery_matches() -> None:
    distmat = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]])
    metrics = evaluate_market1501(
        distmat,
        query_pids=[1, 2],
        gallery_pids=[3, 2, 4],
        query_camids=[0, 1],
        gallery_camids=[1, 0, 0],
        max_rank=3,
    )

    torch.testing.assert_close(metrics.cmc, torch.tensor([1.0, 1.0, 1.0]))
    assert metrics.mAP == pytest.approx(1.0)
    assert metrics.num_valid_queries == 1


def test_evaluate_market1501_ignores_gallery_entries_with_pid_minus_one() -> None:
    distmat = torch.tensor([[0.1, 0.2, 0.3]])
    metrics = evaluate_market1501(
        distmat,
        query_pids=[5],
        gallery_pids=[-1, 5, 6],
        query_camids=[0],
        gallery_camids=[1, 1, 0],
        max_rank=3,
    )

    torch.testing.assert_close(metrics.cmc, torch.tensor([1.0, 1.0, 1.0]))
    assert metrics.mAP == pytest.approx(1.0)
    assert metrics.num_valid_queries == 1


def test_evaluate_market1501_raises_when_no_query_has_valid_positive_match() -> None:
    distmat = torch.tensor([[0.1, 0.2]])

    with pytest.raises(ValueError):
        evaluate_market1501(
            distmat,
            query_pids=[1],
            gallery_pids=[1, 2],
            query_camids=[0],
            gallery_camids=[0, 1],
            max_rank=2,
        )


def test_evaluate_market1501_validates_input_shapes() -> None:
    with pytest.raises(ValueError):
        evaluate_market1501(
            torch.tensor([0.1, 0.2]),
            query_pids=[1],
            gallery_pids=[1, 2],
            query_camids=[0],
            gallery_camids=[0, 1],
        )

    with pytest.raises(ValueError):
        evaluate_market1501(
            torch.tensor([[0.1, 0.2]]),
            query_pids=[1, 2],
            gallery_pids=[1, 2],
            query_camids=[0],
            gallery_camids=[0, 1],
        )


def test_evaluate_market_style_retrieval_matches_full_matrix() -> None:
    query_features = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    gallery_features = torch.tensor(
        [
            [0.1, 0.0],
            [10.1, 10.0],
            [99.0, 99.0],
            [0.2, 0.0],
        ]
    )
    query_pids = [1, 2]
    gallery_pids = [1, 2, -1, 1]
    query_camids = [0, 1]
    gallery_camids = [1, 0, 0, 0]

    full_metrics = evaluate_market1501(
        pairwise_distance(query_features, gallery_features),
        query_pids=query_pids,
        gallery_pids=gallery_pids,
        query_camids=query_camids,
        gallery_camids=gallery_camids,
        max_rank=4,
    )
    chunked_metrics = evaluate_market_style_retrieval(
        query_features=query_features,
        gallery_features=gallery_features,
        query_pids=query_pids,
        gallery_pids=gallery_pids,
        query_camids=query_camids,
        gallery_camids=gallery_camids,
        distance="euclidean",
        max_rank=4,
        query_chunk_size=1,
    )

    torch.testing.assert_close(chunked_metrics.cmc, full_metrics.cmc)
    assert chunked_metrics.mAP == pytest.approx(full_metrics.mAP)
    assert chunked_metrics.num_valid_queries == full_metrics.num_valid_queries


def test_evaluate_market_style_retrieval_raises_without_valid_query() -> None:
    with pytest.raises(ValueError, match="No valid Market-style query"):
        evaluate_market_style_retrieval(
            query_features=torch.tensor([[0.0, 0.0]]),
            gallery_features=torch.tensor([[0.1, 0.0]]),
            query_pids=[1],
            gallery_pids=[1],
            query_camids=[0],
            gallery_camids=[0],
            query_chunk_size=1,
        )


def test_evaluate_market_style_retrieval_rejects_invalid_chunk_size() -> None:
    with pytest.raises(ValueError, match="query_chunk_size"):
        evaluate_market_style_retrieval(
            query_features=torch.tensor([[0.0, 0.0]]),
            gallery_features=torch.tensor([[0.1, 0.0]]),
            query_pids=[1],
            gallery_pids=[1],
            query_camids=[0],
            gallery_camids=[1],
            query_chunk_size=0,
        )


def test_evaluate_clothes_changing_retrieval_filters_same_clothes_positive() -> None:
    metrics = evaluate_clothes_changing_retrieval(
        query_features=torch.tensor([[0.0, 0.0]]),
        gallery_features=torch.tensor(
            [
                [0.0, 0.0],
                [0.05, 0.0],
                [0.1, 0.0],
            ]
        ),
        query_pids=[1],
        gallery_pids=[1, 2, 1],
        query_camids=[0],
        gallery_camids=[1, 1, 1],
        query_clothes_ids=[3],
        gallery_clothes_ids=[3, 9, 4],
        distance="euclidean",
        max_rank=3,
        query_chunk_size=1,
    )

    torch.testing.assert_close(metrics.cmc, torch.tensor([0.0, 1.0, 1.0]))
    assert metrics.mAP == pytest.approx(0.5)
    assert metrics.num_valid_queries == 1


def test_evaluate_clothes_changing_retrieval_skips_queries_without_changed_match() -> None:
    metrics = evaluate_clothes_changing_retrieval(
        query_features=torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
        gallery_features=torch.tensor(
            [
                [0.1, 0.0],
                [10.1, 10.0],
            ]
        ),
        query_pids=[1, 2],
        gallery_pids=[1, 2],
        query_camids=[0, 0],
        gallery_camids=[1, 1],
        query_clothes_ids=[5, 5],
        gallery_clothes_ids=[5, 6],
        distance="euclidean",
        max_rank=2,
        query_chunk_size=1,
    )

    torch.testing.assert_close(metrics.cmc, torch.tensor([1.0, 1.0]))
    assert metrics.mAP == pytest.approx(1.0)
    assert metrics.num_valid_queries == 1


def test_evaluate_clothes_changing_retrieval_requires_changed_gallery_match() -> None:
    with pytest.raises(ValueError, match="clothes-changing"):
        evaluate_clothes_changing_retrieval(
            query_features=torch.tensor([[0.0, 0.0]]),
            gallery_features=torch.tensor([[0.1, 0.0]]),
            query_pids=[1],
            gallery_pids=[1],
            query_camids=[0],
            gallery_camids=[1],
            query_clothes_ids=[5],
            gallery_clothes_ids=[5],
            query_chunk_size=1,
        )


def test_evaluate_market_style_reranking_matches_dense_reference() -> None:
    query_features = torch.tensor(
        [
            [1.0, 0.1, 0.0],
            [0.0, 0.2, 1.0],
        ]
    )
    gallery_features = torch.tensor(
        [
            [0.9, 0.0, 0.1],
            [0.1, 0.0, 0.9],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.1],
        ]
    )
    query_pids = [1, 2]
    gallery_pids = [1, 2, 3, 4]
    query_camids = [0, 0]
    gallery_camids = [1, 1, 1, 1]

    dense_dist = _dense_reference_rerank_distmat(
        query_features,
        gallery_features,
        distance="cosine",
        k1=2,
        k2=2,
        lambda_value=0.3,
    )
    dense_metrics = evaluate_market1501(
        dense_dist,
        query_pids=query_pids,
        gallery_pids=gallery_pids,
        query_camids=query_camids,
        gallery_camids=gallery_camids,
        max_rank=4,
    )
    sparse_metrics = evaluate_market_style_reranking(
        query_features=query_features,
        gallery_features=gallery_features,
        query_pids=query_pids,
        gallery_pids=gallery_pids,
        query_camids=query_camids,
        gallery_camids=gallery_camids,
        distance="cosine",
        max_rank=4,
        k1=2,
        k2=2,
        lambda_value=0.3,
        neighbor_chunk_size=2,
        rerank_query_chunk_size=1,
    )

    torch.testing.assert_close(sparse_metrics.cmc, dense_metrics.cmc)
    assert sparse_metrics.mAP == pytest.approx(dense_metrics.mAP)
    assert sparse_metrics.num_valid_queries == dense_metrics.num_valid_queries


def test_evaluate_market_style_reranking_lambda_one_matches_original_ranking() -> None:
    query_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gallery_features = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.7, 0.3]])
    query_pids = [1, 2]
    gallery_pids = [1, 2, 3]
    query_camids = [0, 0]
    gallery_camids = [1, 1, 1]

    original_metrics = evaluate_market_style_retrieval(
        query_features=query_features,
        gallery_features=gallery_features,
        query_pids=query_pids,
        gallery_pids=gallery_pids,
        query_camids=query_camids,
        gallery_camids=gallery_camids,
        distance="cosine",
        max_rank=3,
        query_chunk_size=1,
    )
    rerank_metrics = evaluate_market_style_reranking(
        query_features=query_features,
        gallery_features=gallery_features,
        query_pids=query_pids,
        gallery_pids=gallery_pids,
        query_camids=query_camids,
        gallery_camids=gallery_camids,
        distance="cosine",
        max_rank=3,
        k1=2,
        k2=1,
        lambda_value=1.0,
        neighbor_chunk_size=2,
        rerank_query_chunk_size=1,
    )

    torch.testing.assert_close(rerank_metrics.cmc, original_metrics.cmc)
    assert rerank_metrics.mAP == pytest.approx(original_metrics.mAP)
    assert rerank_metrics.num_valid_queries == original_metrics.num_valid_queries


def test_evaluate_market_style_reranking_filters_junk_gallery_entries() -> None:
    metrics = evaluate_market_style_reranking(
        query_features=torch.tensor([[1.0, 0.0]]),
        gallery_features=torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
            ]
        ),
        query_pids=[7],
        gallery_pids=[7, -1, 7],
        query_camids=[2],
        gallery_camids=[2, 0, 3],
        distance="cosine",
        max_rank=3,
        k1=2,
        k2=1,
        lambda_value=1.0,
        neighbor_chunk_size=2,
        rerank_query_chunk_size=1,
    )

    torch.testing.assert_close(metrics.cmc, torch.tensor([1.0, 1.0, 1.0]))
    assert metrics.mAP == pytest.approx(1.0)
    assert metrics.num_valid_queries == 1


def test_evaluate_market_style_reranking_raises_without_valid_query() -> None:
    with pytest.raises(ValueError, match="No valid Market-style query"):
        evaluate_market_style_reranking(
            query_features=torch.tensor([[1.0, 0.0]]),
            gallery_features=torch.tensor([[1.0, 0.0]]),
            query_pids=[1],
            gallery_pids=[1],
            query_camids=[0],
            gallery_camids=[0],
            k1=1,
            k2=1,
            neighbor_chunk_size=1,
            rerank_query_chunk_size=1,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"distance": "bad"}, "distance"),
        ({"max_rank": 0}, "max_rank"),
        ({"k1": 0}, "rerank_k1"),
        ({"k2": 0}, "rerank_k2"),
        ({"lambda_value": -0.1}, "rerank_lambda"),
        ({"lambda_value": 1.1}, "rerank_lambda"),
        ({"neighbor_chunk_size": 0}, "rerank_neighbor_chunk_size"),
        ({"rerank_query_chunk_size": 0}, "rerank_query_chunk_size"),
    ],
)
def test_evaluate_market_style_reranking_rejects_invalid_controls(
    kwargs: dict,
    message: str,
) -> None:
    base_kwargs = {
        "query_features": torch.tensor([[1.0, 0.0]]),
        "gallery_features": torch.tensor([[0.9, 0.1]]),
        "query_pids": [1],
        "gallery_pids": [1],
        "query_camids": [0],
        "gallery_camids": [1],
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        evaluate_market_style_reranking(**base_kwargs)


def _dense_reference_rerank_distmat(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    distance: str,
    k1: int,
    k2: int,
    lambda_value: float,
) -> torch.Tensor:
    features = torch.cat([query_features, gallery_features], dim=0).to(torch.float32)
    if distance == "cosine":
        features = torch.nn.functional.normalize(features, p=2, dim=1, eps=1e-12)
        original_dist = 1 - features @ features.t()
    else:
        original_dist = pairwise_distance(features, features)
    original_dist = original_dist / original_dist.max(dim=1).values.clamp_min(1e-12).unsqueeze(1)
    initial_rank = torch.argsort(original_dist, dim=1)

    all_num = features.shape[0]
    full_k = min(k1 + 1, all_num)
    half_k = min(int(round(k1 / 2)) + 1, all_num)
    dense_weights = torch.zeros((all_num, all_num), dtype=torch.float32)
    for row_index in range(all_num):
        forward = initial_rank[row_index, :full_k].tolist()
        reciprocal = {
            candidate
            for candidate in forward
            if row_index in initial_rank[candidate, :full_k].tolist()
        }
        expansion = set(reciprocal)
        for candidate in reciprocal:
            candidate_forward = initial_rank[candidate, :half_k].tolist()
            candidate_reciprocal = {
                item
                for item in candidate_forward
                if candidate in initial_rank[item, :half_k].tolist()
            }
            if candidate_reciprocal and len(candidate_reciprocal.intersection(reciprocal)) > (
                2 / 3
            ) * len(candidate_reciprocal):
                expansion.update(candidate_reciprocal)
        if not expansion:
            expansion.add(row_index)
        expansion_indices = torch.tensor(sorted(expansion), dtype=torch.long)
        values = torch.exp(-original_dist[row_index, expansion_indices])
        dense_weights[row_index, expansion_indices] = values / values.sum().clamp_min(1e-12)

    if k2 > 1:
        qe_k = min(k2, all_num)
        dense_weights = torch.stack(
            [
                dense_weights[initial_rank[row_index, :qe_k]].mean(dim=0)
                for row_index in range(all_num)
            ],
            dim=0,
        )

    num_query = query_features.shape[0]
    num_gallery = gallery_features.shape[0]
    final_dist = torch.empty((num_query, num_gallery), dtype=torch.float32)
    for query_index in range(num_query):
        for gallery_index in range(num_gallery):
            all_gallery_index = num_query + gallery_index
            intersection = torch.minimum(
                dense_weights[query_index],
                dense_weights[all_gallery_index],
            ).sum()
            jaccard = 1 - intersection / (2 - intersection).clamp_min(1e-12)
            final_dist[query_index, gallery_index] = (
                lambda_value * original_dist[query_index, all_gallery_index]
                + (1 - lambda_value) * jaccard
            )
    return final_dist
