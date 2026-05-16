"""Model architectures."""

from reid.models.resnet_reid import (
    ResNetReID,
    load_imagenet_resnet50_weights,
    load_resnet50_backbone_state_dict,
    resnet50_reid,
)

__all__ = [
    "ResNetReID",
    "load_imagenet_resnet50_weights",
    "load_resnet50_backbone_state_dict",
    "resnet50_reid",
]
