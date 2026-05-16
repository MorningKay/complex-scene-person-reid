import pytest
import torch

from reid.evaluation.distance import cosine_distance, pairwise_distance


def test_pairwise_distance_returns_squared_euclidean_distances() -> None:
    query_features = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    gallery_features = torch.tensor([[1.0, 0.0], [2.0, 2.0], [0.0, 1.0]])

    distmat = pairwise_distance(query_features, gallery_features)

    expected = torch.tensor(
        [
            [1.0, 8.0, 1.0],
            [1.0, 2.0, 1.0],
        ]
    )
    assert distmat.shape == (2, 3)
    torch.testing.assert_close(distmat, expected)


def test_cosine_distance_returns_one_minus_cosine_similarity() -> None:
    query_features = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    gallery_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    distmat = cosine_distance(query_features, gallery_features)

    expected = torch.tensor(
        [
            [0.0, 1.0],
            [1 - 2**-0.5, 1 - 2**-0.5],
        ]
    )
    torch.testing.assert_close(distmat, expected)


@pytest.mark.parametrize(
    ("query_features", "gallery_features"),
    [
        (torch.zeros(2), torch.zeros(1, 2)),
        (torch.zeros(1, 2), torch.zeros(2)),
    ],
)
def test_distance_helpers_reject_non_2d_inputs(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        pairwise_distance(query_features, gallery_features)
    with pytest.raises(ValueError):
        cosine_distance(query_features, gallery_features)


def test_distance_helpers_reject_mismatched_feature_dimensions() -> None:
    query_features = torch.zeros(2, 3)
    gallery_features = torch.zeros(4, 5)

    with pytest.raises(ValueError):
        pairwise_distance(query_features, gallery_features)
    with pytest.raises(ValueError):
        cosine_distance(query_features, gallery_features)
