"""Classification losses for identity supervision."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _validate_classification_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if logits.ndim != 2:
        raise ValueError(f"logits must be a 2D tensor, got shape {tuple(logits.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets must be a 1D tensor, got shape {tuple(targets.shape)}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            "logits and targets must have the same batch size, "
            f"got {logits.shape[0]} and {targets.shape[0]}"
        )
    if targets.dtype != torch.long:
        raise ValueError(f"targets must have dtype torch.long, got {targets.dtype}")


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        if not 0 <= epsilon < 1:
            raise ValueError("epsilon must be in the range [0, 1)")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: 'mean', 'sum', 'none'")

        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_classification_inputs(logits, targets)

        log_probs = F.log_softmax(logits, dim=1)
        nll_loss = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=1)
        loss = (1 - self.epsilon) * nll_loss + self.epsilon * smooth_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_classification_loss(label_smoothing: float = 0.0) -> nn.Module:
    if label_smoothing < 0:
        raise ValueError("label_smoothing must be non-negative")
    if label_smoothing == 0:
        return nn.CrossEntropyLoss()
    return LabelSmoothingCrossEntropy(epsilon=label_smoothing)
