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


def test_vit_global_local_sie_and_part_heads_return_auxiliary_logits() -> None:
    model = vit_patch16_global_local_reid(
        num_classes=10,
        backbone_name=BACKBONE_NAME,
        image_size=(64, 32),
        patch_size=16,
        num_parts=4,
        feature_dim=64,
        pretrained=False,
        sie_camera=True,
        sie_num_cameras=6,
        sie_coefficient=2.0,
        part_classifiers=True,
    )
    x = torch.randn(2, 3, 64, 32)
    camids = torch.tensor([0, 5])

    logits, features, part_logits = model(x, camids=camids)

    assert logits.shape == (2, 10)
    assert features.shape == (2, 64)
    assert len(part_logits) == 4
    assert all(logits_for_part.shape == (2, 10) for logits_for_part in part_logits)
    assert model.sie_camera_embeddings is not None
    assert model.sie_camera_embeddings.weight.shape == (6, model.embed_dim)


def test_vit_global_local_sie_requires_camids() -> None:
    model = vit_patch16_global_local_reid(
        num_classes=10,
        backbone_name=BACKBONE_NAME,
        image_size=(64, 32),
        patch_size=16,
        num_parts=4,
        feature_dim=64,
        pretrained=False,
        sie_camera=True,
        sie_num_cameras=6,
    )

    with pytest.raises(ValueError, match="Camera SIE requires camids"):
        model(torch.randn(2, 3, 64, 32))


def test_vit_global_local_sie_rejects_out_of_range_camids() -> None:
    model = vit_patch16_global_local_reid(
        num_classes=10,
        backbone_name=BACKBONE_NAME,
        image_size=(64, 32),
        patch_size=16,
        num_parts=4,
        feature_dim=64,
        pretrained=False,
        sie_camera=True,
        sie_num_cameras=6,
    )

    with pytest.raises(ValueError, match="camera range"):
        model(torch.randn(2, 3, 64, 32), camids=torch.tensor([0, 6]))


def test_vit_global_local_sie_backward_reaches_sie_and_part_heads() -> None:
    model = vit_patch16_global_local_reid(
        num_classes=10,
        backbone_name=BACKBONE_NAME,
        image_size=(64, 32),
        patch_size=16,
        num_parts=4,
        feature_dim=64,
        pretrained=False,
        sie_camera=True,
        sie_num_cameras=6,
        part_classifiers=True,
    )
    x = torch.randn(2, 3, 64, 32)
    camids = torch.tensor([0, 1])

    logits, features, part_logits = model(x, camids=camids)
    loss = logits.mean() + features.mean() + sum(item.mean() for item in part_logits)
    loss.backward()

    assert model.sie_camera_embeddings is not None
    assert model.sie_camera_embeddings.weight.grad is not None
    assert model.part_classifiers[0].weight.grad is not None


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


def test_build_reid_model_passes_vit_sie_and_part_head_controls() -> None:
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
            "sie_camera": True,
            "sie_num_cameras": 6,
            "sie_coefficient": 2.0,
            "part_classifiers": True,
        },
    }

    model = build_reid_model(config, load_pretrained=False)

    assert isinstance(model, ViTPatch16GlobalLocalReID)
    assert model.sie_camera is True
    assert model.sie_num_cameras == 6
    assert model.sie_coefficient == pytest.approx(2.0)
    assert model.part_classifiers_enabled is True
