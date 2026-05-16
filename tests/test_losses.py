import pytest
import torch
from torch import nn
from torch.nn import functional as F

from reid.losses import LabelSmoothingCrossEntropy, build_classification_loss


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
