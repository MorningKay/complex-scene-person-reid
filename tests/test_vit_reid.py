import pytest
import torch

from reid.models import (
    ViTPatch16GlobalLocalReID,
    build_reid_model,
    vit_patch16_global_local_reid,
)


BACKBONE_NAME = "deit_tiny_patch16_224"


def test_vit_global_local_returns_logits_and_training_features() -> None:
    model = vit_patch16_global_local_reid(
        num_classes=10,
        backbone_name=BACKBONE_NAME,
        image_size=(64, 32),
        patch_size=16,
        num_parts=4,
        feature_dim=128,
        pretrained=False,
    )
    x = torch.randn(2, 3, 64, 32)

    logits, features = model(x)

    assert logits.shape == (2, 10)
    assert features.shape == (2, 128)
    assert model.grid_size == (4, 2)
    assert model.projection[0].in_features == model.embed_dim * 5


def test_vit_global_local_returns_normalized_retrieval_features() -> None:
    model = vit_patch16_global_local_reid(
        num_classes=10,
        backbone_name=BACKBONE_NAME,
        image_size=(64, 32),
        patch_size=16,
        num_parts=4,
        feature_dim=128,
        pretrained=False,
    )
    model.eval()

    with torch.no_grad():
        features = model(torch.randn(2, 3, 64, 32), return_feature=True)

    assert features.shape == (2, 128)
    assert torch.allclose(features.norm(p=2, dim=1), torch.ones(2), atol=1e-5)


def test_vit_global_local_backward_reaches_backbone() -> None:
    model = vit_patch16_global_local_reid(
        num_classes=10,
        backbone_name=BACKBONE_NAME,
        image_size=(64, 32),
        patch_size=16,
        num_parts=4,
        feature_dim=64,
        pretrained=False,
    )
    x = torch.randn(2, 3, 64, 32)

    logits, features = model(x)
    loss = logits.mean() + features.mean()
    loss.backward()

    assert model.backbone.patch_embed.proj.weight.grad is not None
    assert torch.isfinite(model.backbone.patch_embed.proj.weight.grad).all()


def test_vit_global_local_pretrained_false_does_not_download_weights() -> None:
    model = vit_patch16_global_local_reid(
        num_classes=10,
        backbone_name=BACKBONE_NAME,
        image_size=(64, 32),
        patch_size=16,
        num_parts=4,
        feature_dim=128,
        pretrained=False,
    )

    assert isinstance(model, ViTPatch16GlobalLocalReID)


def test_vit_global_local_rejects_image_size_not_divisible_by_patch_size() -> None:
    with pytest.raises(ValueError, match="divisible by patch_size"):
        vit_patch16_global_local_reid(
            num_classes=10,
            backbone_name=BACKBONE_NAME,
            image_size=(64, 30),
            patch_size=16,
            num_parts=4,
            pretrained=False,
        )


def test_vit_global_local_rejects_grid_height_not_divisible_by_parts() -> None:
    with pytest.raises(ValueError, match="grid height"):
        vit_patch16_global_local_reid(
            num_classes=10,
            backbone_name=BACKBONE_NAME,
            image_size=(48, 32),
            patch_size=16,
            num_parts=4,
            pretrained=False,
        )


def test_vit_global_local_rejects_runtime_image_size_mismatch() -> None:
    model = vit_patch16_global_local_reid(
        num_classes=10,
        backbone_name=BACKBONE_NAME,
        image_size=(64, 32),
        patch_size=16,
        num_parts=4,
        pretrained=False,
    )

    with pytest.raises(ValueError, match="configured ViT image_size"):
        model(torch.randn(2, 3, 80, 32))


def test_build_reid_model_accepts_vit_global_local_name() -> None:
    config = {
        "data": {"image_size": [64, 32]},
        "model": {
            "name": "vit_patch16_global_local",
            "backbone_name": BACKBONE_NAME,
            "num_classes": 10,
            "feature_dim": 128,
            "pretrained": False,
            "patch_size": 16,
            "num_parts": 4,
        },
    }

    model = build_reid_model(config, load_pretrained=False)

    assert isinstance(model, ViTPatch16GlobalLocalReID)
