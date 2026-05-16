import torch
from torchvision.models import resnet50 as torchvision_resnet50

from reid.models import ResNetReID, load_resnet50_backbone_state_dict, resnet50_reid
from reid.models.resnet_reid import Bottleneck


def test_resnet50_reid_returns_logits_and_training_features() -> None:
    model = resnet50_reid(num_classes=751)
    x = torch.randn(2, 3, 256, 128)

    logits, features = model(x)

    assert logits.shape == (2, 751)
    assert features.shape == (2, 2048)


def test_resnet50_reid_supports_custom_feature_dim() -> None:
    model = resnet50_reid(num_classes=10, feature_dim=256)
    x = torch.randn(2, 3, 64, 32)

    logits, features = model(x)

    assert logits.shape == (2, 10)
    assert features.shape == (2, 256)


def test_resnet50_reid_returns_normalized_retrieval_features() -> None:
    model = ResNetReID(num_classes=10, feature_dim=256)
    model.eval()

    with torch.no_grad():
        features = model(torch.randn(2, 3, 64, 32), return_feature=True)

    assert features.shape == (2, 256)
    assert torch.allclose(features.norm(p=2, dim=1), torch.ones(2), atol=1e-5)


def test_resnet50_reid_backward_reaches_backbone() -> None:
    model = resnet50_reid(num_classes=10, feature_dim=128)
    x = torch.randn(2, 3, 64, 32)

    logits, features = model(x)
    loss = logits.mean() + features.mean()
    loss.backward()

    assert model.conv1.weight.grad is not None
    assert torch.isfinite(model.conv1.weight.grad).all()


def test_bottleneck_downsample_matches_residual_shape() -> None:
    block = Bottleneck(in_channels=64, bottleneck_channels=64, stride=2)
    x = torch.randn(2, 64, 16, 8)

    y = block(x)

    assert y.shape == (2, 256, 8, 4)


def test_load_resnet50_backbone_state_dict_loads_matching_backbone_weights() -> None:
    model = resnet50_reid(num_classes=10, feature_dim=256, last_stride=1)
    source_model = torchvision_resnet50(weights=None)
    source_state_dict = source_model.state_dict()

    loaded_keys = load_resnet50_backbone_state_dict(model, source_state_dict)

    assert "conv1.weight" in loaded_keys
    assert "layer4.2.bn3.running_var" in loaded_keys
    assert not any(key.startswith("fc.") for key in loaded_keys)
    assert torch.equal(model.conv1.weight, source_state_dict["conv1.weight"])


def test_load_resnet50_backbone_state_dict_keeps_reid_head_initialized() -> None:
    model = resnet50_reid(num_classes=10, feature_dim=256, last_stride=1)
    source_state_dict = torchvision_resnet50(weights=None).state_dict()
    classifier_before = model.classifier.weight.detach().clone()
    bnneck_before = model.bnneck.weight.detach().clone()
    embedding_before = model.embedding.weight.detach().clone()

    load_resnet50_backbone_state_dict(model, source_state_dict)

    assert torch.equal(model.classifier.weight, classifier_before)
    assert torch.equal(model.bnneck.weight, bnneck_before)
    assert torch.equal(model.embedding.weight, embedding_before)


def test_load_resnet50_backbone_state_dict_keeps_reid_last_stride() -> None:
    model = resnet50_reid(num_classes=10, last_stride=1)
    source_state_dict = torchvision_resnet50(weights=None).state_dict()

    load_resnet50_backbone_state_dict(model, source_state_dict)

    assert model.layer4[0].conv2.stride == (1, 1)
    assert model.layer4[0].downsample[0].stride == (1, 1)
