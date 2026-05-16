"""Loss functions for Re-ID training."""

from reid.losses.classification import (
    LabelSmoothingCrossEntropy,
    build_classification_loss,
)
from reid.losses.triplet import BatchHardTripletLoss

__all__ = [
    "BatchHardTripletLoss",
    "LabelSmoothingCrossEntropy",
    "build_classification_loss",
]
