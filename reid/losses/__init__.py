"""Loss functions for Re-ID training."""

from reid.losses.classification import (
    LabelSmoothingCrossEntropy,
    build_classification_loss,
)

__all__ = [
    "LabelSmoothingCrossEntropy",
    "build_classification_loss",
]
