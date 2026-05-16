import pytest
import torch
from torch import nn
from torch.nn import functional as F

from reid.losses import (
    BatchHardTripletLoss,
    LabelSmoothingCrossEntropy,
    build_classification_loss,
)


def test_label_smoothing_zero_matches_cross_entropy() -> None:
    logits = torch.tensor([[3.0, 1.0, -1.0], [0.5, 2.0, 0.0]])
    targets = torch.tensor([0, 1], dtype=torch.long)
    criterion = LabelSmoothingCrossEntropy(epsilon=0.0)

    loss = criterion(logits, targets)

    assert torch.allclose(loss, F.cross_entropy(logits, targets))


def test_label_smoothing_returns_finite_scalar_and_backward() -> None:
    logits = torch.tensor(
        [[2.0, 0.0, -1.0], [0.1, 0.2, 1.3]],
        requires_grad=True,
    )
    targets = torch.tensor([0, 2], dtype=torch.long)
    criterion = LabelSmoothingCrossEntropy(epsilon=0.1)

    loss = criterion(logits, targets)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_build_classification_loss_selects_loss_type() -> None:
    assert isinstance(build_classification_loss(label_smoothing=0.0), nn.CrossEntropyLoss)
    assert isinstance(
        build_classification_loss(label_smoothing=0.1),
        LabelSmoothingCrossEntropy,
    )


@pytest.mark.parametrize(
    ("logits", "targets"),
    [
        (torch.randn(2, 3, 4), torch.tensor([0, 1], dtype=torch.long)),
        (torch.randn(2, 3), torch.tensor([[0], [1]], dtype=torch.long)),
        (torch.randn(2, 3), torch.tensor([0], dtype=torch.long)),
        (torch.randn(2, 3), torch.tensor([0, 1], dtype=torch.int32)),
    ],
)
def test_label_smoothing_rejects_invalid_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    criterion = LabelSmoothingCrossEntropy()

    with pytest.raises(ValueError):
        criterion(logits, targets)


def test_batch_hard_triplet_loss_is_zero_when_classes_are_separated() -> None:
    features = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
        ]
    )
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    criterion = BatchHardTripletLoss(margin=0.3)

    loss = criterion(features, targets)

    assert torch.allclose(loss, torch.tensor(0.0))


def test_batch_hard_triplet_loss_is_positive_and_backward_when_negative_is_close() -> None:
    features = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.2, 0.0],
            [3.0, 0.0],
        ],
        requires_grad=True,
    )
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    criterion = BatchHardTripletLoss(margin=0.3)

    loss = criterion(features, targets)
    loss.backward()

    assert loss.item() > 0
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


@pytest.mark.parametrize(
    ("features", "targets"),
    [
        (torch.randn(1, 4), torch.tensor([0], dtype=torch.long)),
        (torch.randn(3, 4), torch.tensor([0, 1, 2], dtype=torch.long)),
    ],
)
def test_batch_hard_triplet_loss_returns_zero_without_valid_anchors(
    features: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    features.requires_grad_(True)
    criterion = BatchHardTripletLoss()

    loss = criterion(features, targets)
    loss.backward()

    assert torch.allclose(loss, torch.tensor(0.0))
    assert features.grad is not None


@pytest.mark.parametrize(
    ("features", "targets"),
    [
        (torch.randn(2, 3, 4), torch.tensor([0, 1], dtype=torch.long)),
        (torch.randn(2, 3), torch.tensor([[0], [1]], dtype=torch.long)),
        (torch.randn(2, 3), torch.tensor([0], dtype=torch.long)),
        (torch.randn(2, 3), torch.tensor([0, 1], dtype=torch.int32)),
    ],
)
def test_batch_hard_triplet_loss_rejects_invalid_inputs(
    features: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    criterion = BatchHardTripletLoss()

    with pytest.raises(ValueError):
        criterion(features, targets)
