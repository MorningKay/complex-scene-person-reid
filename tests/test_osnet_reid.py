from pathlib import Path

import pytest
import torch

from reid.models import (
    OSBlock,
    OSNetReID,
    load_imagenet_osnet_weights,
    load_osnet_backbone_state_dict,
    osnet_x1_0_reid,
)


def test_osnet_x1_0_returns_logits_and_training_features() -> None:
    model = osnet_x1_0_reid(num_classes=10, feature_dim=512)
    x = torch.randn(2, 3, 64, 32)

    logits, features = model(x)

    assert logits.shape == (2, 10)
    assert features.shape == (2, 512)


def test_osnet_x1_0_returns_normalized_retrieval_features() -> None:
    model = osnet_x1_0_reid(num_classes=10, feature_dim=256)
    model.eval()

    with torch.no_grad():
        features = model(torch.randn(2, 3, 64, 32), return_feature=True)

    assert features.shape == (2, 256)
    assert torch.allclose(features.norm(p=2, dim=1), torch.ones(2), atol=1e-5)


def test_osblock_downsample_matches_residual_shape() -> None:
    block = OSBlock(in_channels=64, out_channels=256)
    x = torch.randn(2, 64, 16, 8)

    y = block(x)

    assert y.shape == (2, 256, 16, 8)


def test_osnet_x1_0_backward_reaches_backbone() -> None:
    model = osnet_x1_0_reid(num_classes=10, feature_dim=128)
    x = torch.randn(2, 3, 64, 32)

    logits, features = model(x)
    loss = logits.mean() + features.mean()
    loss.backward()

    assert model.conv1.conv.weight.grad is not None
    assert torch.isfinite(model.conv1.conv.weight.grad).all()


def test_osnet_x1_0_pretrained_false_does_not_read_missing_path(tmp_path: Path) -> None:
    model = osnet_x1_0_reid(
        num_classes=10,
        feature_dim=512,
        pretrained=False,
        pretrained_path=tmp_path / "missing.pth",
    )

    assert isinstance(model, OSNetReID)


def test_load_osnet_backbone_state_dict_loads_matching_backbone_and_fc_weights() -> None:
    model = osnet_x1_0_reid(num_classes=10, feature_dim=512)
    source_state_dict = osnet_x1_0_reid(num_classes=10, feature_dim=512).state_dict()
    source_state_dict["conv1.conv.weight"] = torch.full_like(
        source_state_dict["conv1.conv.weight"],
        0.25,
    )
    source_state_dict["fc.0.weight"] = torch.full_like(source_state_dict["fc.0.weight"], 0.5)
    source_state_dict["bnneck.weight"] = torch.full_like(
        source_state_dict["bnneck.weight"],
        7.0,
    )
    source_state_dict["classifier.weight"] = torch.full_like(
        source_state_dict["classifier.weight"],
        9.0,
    )
    classifier_before = model.classifier.weight.detach().clone()

    loaded_keys = load_osnet_backbone_state_dict(model, source_state_dict)

    assert "conv1.conv.weight" in loaded_keys
    assert "fc.0.weight" in loaded_keys
    assert "classifier.weight" not in loaded_keys
    assert "bnneck.weight" not in loaded_keys
    assert torch.allclose(
        model.conv1.conv.weight,
        torch.full_like(model.conv1.conv.weight, 0.25),
    )
    assert torch.allclose(model.fc[0].weight, torch.full_like(model.fc[0].weight, 0.5))
    assert torch.allclose(model.bnneck.weight, torch.ones_like(model.bnneck.weight))
    assert torch.equal(model.classifier.weight, classifier_before)


def test_load_imagenet_osnet_weights_accepts_nested_state_dict(tmp_path: Path) -> None:
    model = osnet_x1_0_reid(num_classes=10, feature_dim=512)
    source_state_dict = osnet_x1_0_reid(num_classes=1000, feature_dim=512).state_dict()
    checkpoint_path = tmp_path / "osnet_x1_0_imagenet.pth"
    torch.save({"state_dict": source_state_dict}, checkpoint_path)

    loaded_keys = load_imagenet_osnet_weights(model, checkpoint_path)

    assert "conv1.conv.weight" in loaded_keys
    assert "classifier.weight" not in loaded_keys


def test_load_imagenet_osnet_weights_rejects_missing_file(tmp_path: Path) -> None:
    model = osnet_x1_0_reid(num_classes=10, feature_dim=512)

    with pytest.raises(FileNotFoundError, match="OSNet pretrained checkpoint"):
        load_imagenet_osnet_weights(model, tmp_path / "missing.pth")
