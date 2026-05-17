import pytest
import torch

from reid.evaluation.distance import pairwise_distance
from reid.evaluation.metrics import (
    RetrievalMetrics,
    evaluate_market1501,
    evaluate_market_style_retrieval,
)


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
