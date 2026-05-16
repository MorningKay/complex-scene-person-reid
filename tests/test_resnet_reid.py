import torch

from reid.models import ResNetReID, resnet50_reid
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
