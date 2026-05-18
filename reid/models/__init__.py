"""Model architectures."""

from reid.models.osnet_reid import (
    ChannelGate,
    Conv1x1,
    Conv1x1Linear,
    ConvLayer,
    LightConv3x3,
    OSBlock,
    OSNetReID,
    load_imagenet_osnet_weights,
    load_osnet_backbone_state_dict,
    osnet_x1_0_reid,
)
from reid.models.resnet_reid import (
    ResNetReID,
    load_imagenet_resnet50_weights,
    load_resnet50_backbone_state_dict,
    resnet50_reid,
)
from reid.models.registry import build_reid_model, normalize_model_name

__all__ = [
    "ChannelGate",
    "Conv1x1",
    "Conv1x1Linear",
    "ConvLayer",
    "LightConv3x3",
    "OSBlock",
    "OSNetReID",
    "ResNetReID",
    "build_reid_model",
    "load_imagenet_osnet_weights",
    "load_imagenet_resnet50_weights",
    "load_osnet_backbone_state_dict",
    "load_resnet50_backbone_state_dict",
    "normalize_model_name",
    "osnet_x1_0_reid",
    "resnet50_reid",
]
