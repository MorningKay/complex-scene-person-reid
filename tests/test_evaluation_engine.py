from pathlib import Path

import pytest
import torch

from reid.engine import run_evaluation
from reid.engine.train import run_training

DATA_ROOT = Path("data/Market-1501-v15.09.15")
MSMT17_ROOT = Path("data/MSMT17_V1")
OCCLUDED_REID_ROOT = Path("data/Occluded_REID")
VC_CLOTHES_ROOT = Path("data/VC-Clothes")


def make_smoke_config() -> dict:
    return {
        "run": {
            "name": "eval_pytest_smoke",
            "seed": 42,
            "smoke": True,
        },
        "data": {
            "root": str(DATA_ROOT),
            "image_size": [32, 16],
            "batch_size": 2,
            "num_workers": 0,
            "random_erasing": False,
            "pin_memory": False,
            "drop_last": False,
        },
        "model": {
            "num_classes": 751,
            "feature_dim": 2048,
            "last_stride": 1,
        },
        "loss": {
            "label_smoothing": 0.0,
        },
        "optimizer": {
            "lr": 0.0003,
            "weight_decay": 0.0005,
        },
        "train": {
            "epochs": 1,
            "max_batches": 1,
            "log_interval": 1,
        },
    }


def make_osnet_smoke_config() -> dict:
    config = make_smoke_config()
    config["run"]["name"] = "eval_osnet_pytest_smoke"
    config["data"]["image_size"] = [64, 32]
    config["model"] = {
        "name": "osnet_x1_0",
        "num_classes": 751,
        "feature_dim": 512,
        "pretrained": False,
    }
    return config


def make_vit_smoke_config() -> dict:
    config = make_smoke_config()
    config["run"]["name"] = "eval_vit_pytest_smoke"
    config["data"]["image_size"] = [64, 32]
    config["data"]["batch_size"] = 4
    config["model"] = {
        "name": "vit_patch16_global_local",
        "backbone_name": "deit_tiny_patch16_224",
        "num_classes": 751,
        "feature_dim": 128,
        "pretrained": False,
        "patch_size": 16,
        "num_parts": 4,
    }
    config["optimizer"]["name"] = "adamw"
    return config


def test_run_evaluation_writes_metrics_and_log(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    run_training(config=make_smoke_config(), output_dir=train_dir, device="cpu")

    metrics = run_evaluation(
        checkpoint_path=train_dir / "ckpt" / "best.pth",
        data_root=DATA_ROOT,
        output_dir=eval_dir,
        device="cpu",
        batch_size=4,
        num_workers=0,
        max_query=8,
        max_gallery=32,
    )

    assert (eval_dir / "eval_metrics.json").is_file()
    assert (eval_dir / "logs" / "eval.txt").is_file()
    assert metrics["dataset_name"] == "market1501"
    assert metrics["query_chunk_size"] == 256
    assert metrics["num_query"] == 8
    assert metrics["num_gallery"] == 32
    assert metrics["num_valid_queries"] > 0
    for key in ("rank1", "rank5", "rank10", "mAP"):
        assert 0 <= metrics[key] <= 1


def test_run_evaluation_accepts_msmt17_dataset_name(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")
    if not MSMT17_ROOT.is_dir():
        pytest.skip(f"MSMT17_V1 dataset not found at {MSMT17_ROOT}")

    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval_msmt17"
    run_training(config=make_smoke_config(), output_dir=train_dir, device="cpu")

    metrics = run_evaluation(
        checkpoint_path=train_dir / "ckpt" / "best.pth",
        dataset_name="msmt17_v1",
        data_root=MSMT17_ROOT,
        output_dir=eval_dir,
        device="cpu",
        batch_size=4,
        num_workers=0,
        max_query=8,
        max_gallery=64,
        query_chunk_size=2,
    )

    assert metrics["dataset_name"] == "msmt17_v1"
    assert metrics["query_chunk_size"] == 2
    assert metrics["num_query"] == 8
    assert metrics["num_gallery"] == 64
    assert metrics["num_valid_queries"] > 0
    for key in ("rank1", "rank5", "rank10", "mAP"):
        assert 0 <= metrics[key] <= 1

    eval_log = (eval_dir / "logs" / "eval.txt").read_text(encoding="utf-8")
    assert "dataset_name=msmt17_v1" in eval_log
    assert "query_chunk_size=2" in eval_log


def test_run_evaluation_accepts_occluded_reid_protocol(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")
    if not OCCLUDED_REID_ROOT.is_dir():
        pytest.skip(f"Occluded_REID dataset not found at {OCCLUDED_REID_ROOT}")

    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval_occluded"
    run_training(config=make_smoke_config(), output_dir=train_dir, device="cpu")

    metrics = run_evaluation(
        checkpoint_path=train_dir / "ckpt" / "best.pth",
        dataset_name="occluded_reid",
        data_root=OCCLUDED_REID_ROOT,
        output_dir=eval_dir,
        device="cpu",
        batch_size=4,
        num_workers=0,
        max_query=8,
        max_gallery=32,
        query_chunk_size=2,
    )

    assert metrics["dataset_name"] == "occluded_reid"
    assert metrics["protocol"] == "occluded_to_whole"
    assert metrics["query"] == "occluded_body_images"
    assert metrics["gallery"] == "whole_body_images"
    assert metrics["num_query"] == 8
    assert metrics["num_gallery"] == 32
    assert metrics["num_valid_queries"] > 0
    for key in ("rank1", "rank5", "rank10", "mAP"):
        assert 0 <= metrics[key] <= 1

    eval_log = (eval_dir / "logs" / "eval.txt").read_text(encoding="utf-8")
    assert "dataset_name=occluded_reid" in eval_log
    assert "protocol=occluded_to_whole" in eval_log


def test_run_evaluation_accepts_vc_clothes_protocol(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")
    if not VC_CLOTHES_ROOT.is_dir():
        pytest.skip(f"VC-Clothes dataset not found at {VC_CLOTHES_ROOT}")

    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval_vc"
    run_training(config=make_smoke_config(), output_dir=train_dir, device="cpu")

    metrics = run_evaluation(
        checkpoint_path=train_dir / "ckpt" / "best.pth",
        dataset_name="vc_clothes",
        data_root=VC_CLOTHES_ROOT,
        output_dir=eval_dir,
        device="cpu",
        batch_size=4,
        num_workers=0,
        max_query=16,
        max_gallery=256,
        query_chunk_size=2,
    )

    assert metrics["dataset_name"] == "vc_clothes"
    assert metrics["protocol"] == "standard"
    assert metrics["num_query"] == 16
    assert metrics["num_gallery"] == 256
    assert metrics["num_valid_queries"] > 0
    assert metrics["clothes_changing"]["dataset_name"] == "vc_clothes"
    assert metrics["clothes_changing"]["protocol"] == "clothes_changing"
    assert metrics["clothes_changing"]["num_query"] == 16
    assert metrics["clothes_changing"]["num_gallery"] == 256
    assert metrics["clothes_changing"]["num_valid_queries"] > 0
    for key in ("rank1", "rank5", "rank10", "mAP"):
        assert 0 <= metrics[key] <= 1
        assert 0 <= metrics["clothes_changing"][key] <= 1

    eval_log = (eval_dir / "logs" / "eval.txt").read_text(encoding="utf-8")
    assert "dataset_name=vc_clothes" in eval_log
    assert "protocol=standard" in eval_log
    assert "clothes_changing rank1=" in eval_log


def test_run_evaluation_rejects_reranking_for_special_protocols(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")
    if not VC_CLOTHES_ROOT.is_dir():
        pytest.skip(f"VC-Clothes dataset not found at {VC_CLOTHES_ROOT}")

    train_dir = tmp_path / "train"
    run_training(config=make_smoke_config(), output_dir=train_dir, device="cpu")

    with pytest.raises(ValueError, match="Re-ranking"):
        run_evaluation(
            checkpoint_path=train_dir / "ckpt" / "best.pth",
            dataset_name="vc_clothes",
            data_root=VC_CLOTHES_ROOT,
            output_dir=tmp_path / "eval_vc_rerank",
            device="cpu",
            batch_size=4,
            num_workers=0,
            max_query=4,
            max_gallery=32,
            query_chunk_size=2,
            rerank=True,
        )


def test_run_evaluation_reloads_osnet_checkpoint(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    train_dir = tmp_path / "osnet_train"
    eval_dir = tmp_path / "osnet_eval"
    run_training(config=make_osnet_smoke_config(), output_dir=train_dir, device="cpu")

    metrics = run_evaluation(
        checkpoint_path=train_dir / "ckpt" / "best.pth",
        data_root=DATA_ROOT,
        output_dir=eval_dir,
        device="cpu",
        batch_size=4,
        num_workers=0,
        max_query=8,
        max_gallery=32,
        query_chunk_size=2,
    )

    assert metrics["dataset_name"] == "market1501"
    assert metrics["query_chunk_size"] == 2
    assert metrics["num_query"] == 8
    assert metrics["num_gallery"] == 32
    for key in ("rank1", "rank5", "rank10", "mAP"):
        assert 0 <= metrics[key] <= 1

    eval_log = (eval_dir / "logs" / "eval.txt").read_text(encoding="utf-8")
    assert "dataset_name=market1501" in eval_log
    assert "query_chunk_size=2" in eval_log


def test_run_evaluation_reloads_vit_checkpoint(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    train_dir = tmp_path / "vit_train"
    eval_dir = tmp_path / "vit_eval"
    run_training(config=make_vit_smoke_config(), output_dir=train_dir, device="cpu")

    metrics = run_evaluation(
        checkpoint_path=train_dir / "ckpt" / "best.pth",
        data_root=DATA_ROOT,
        output_dir=eval_dir,
        device="cpu",
        batch_size=4,
        num_workers=0,
        max_query=8,
        max_gallery=32,
        query_chunk_size=2,
    )

    assert metrics["dataset_name"] == "market1501"
    assert metrics["query_chunk_size"] == 2
    assert metrics["num_query"] == 8
    assert metrics["num_gallery"] == 32
    for key in ("rank1", "rank5", "rank10", "mAP"):
        assert 0 <= metrics[key] <= 1

    checkpoint = torch.load(train_dir / "ckpt" / "best.pth", map_location="cpu")
    assert checkpoint["config"]["model"]["name"] == "vit_patch16_global_local"
    assert checkpoint["config"]["optimizer"]["name"] == "adamw"

    eval_log = (eval_dir / "logs" / "eval.txt").read_text(encoding="utf-8")
    assert "dataset_name=market1501" in eval_log
    assert "query_features=(8, 128)" in eval_log


def test_run_evaluation_writes_reranking_metrics(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")

    train_dir = tmp_path / "train_rerank"
    eval_dir = tmp_path / "eval_rerank"
    run_training(config=make_smoke_config(), output_dir=train_dir, device="cpu")

    metrics = run_evaluation(
        checkpoint_path=train_dir / "ckpt" / "best.pth",
        data_root=DATA_ROOT,
        output_dir=eval_dir,
        device="cpu",
        batch_size=4,
        num_workers=0,
        max_query=4,
        max_gallery=16,
        query_chunk_size=2,
        rerank=True,
        rerank_k1=3,
        rerank_k2=2,
        rerank_lambda=0.3,
        rerank_neighbor_chunk_size=4,
        rerank_query_chunk_size=2,
    )

    assert metrics["rerank_enabled"] is True
    assert metrics["rerank_k1"] == 3
    assert metrics["rerank_k2"] == 2
    assert metrics["rerank_lambda"] == pytest.approx(0.3)
    assert metrics["rerank_neighbor_chunk_size"] == 4
    assert metrics["rerank_query_chunk_size"] == 2
    assert metrics["pre_rerank"]["dataset_name"] == "market1501"
    assert metrics["pre_rerank"]["num_query"] == 4
    assert metrics["pre_rerank"]["num_gallery"] == 16
    for key in ("rank1", "rank5", "rank10", "mAP"):
        assert 0 <= metrics[key] <= 1
        assert 0 <= metrics["pre_rerank"][key] <= 1

    eval_log = (eval_dir / "logs" / "eval.txt").read_text(encoding="utf-8")
    assert "rerank=True" in eval_log
    assert "rerank_k1=3" in eval_log
    assert "rerank_query_chunk_size=2" in eval_log


def test_run_evaluation_accepts_msmt17_reranking_smoke(tmp_path: Path) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Market-1501 dataset not found at {DATA_ROOT}")
    if not MSMT17_ROOT.is_dir():
        pytest.skip(f"MSMT17_V1 dataset not found at {MSMT17_ROOT}")

    train_dir = tmp_path / "train_msmt17_rerank"
    eval_dir = tmp_path / "eval_msmt17_rerank"
    run_training(config=make_smoke_config(), output_dir=train_dir, device="cpu")

    metrics = run_evaluation(
        checkpoint_path=train_dir / "ckpt" / "best.pth",
        dataset_name="msmt17_v1",
        data_root=MSMT17_ROOT,
        output_dir=eval_dir,
        device="cpu",
        batch_size=4,
        num_workers=0,
        max_query=4,
        max_gallery=24,
        query_chunk_size=2,
        rerank=True,
        rerank_k1=3,
        rerank_k2=2,
        rerank_lambda=0.3,
        rerank_neighbor_chunk_size=4,
        rerank_query_chunk_size=2,
    )

    assert metrics["dataset_name"] == "msmt17_v1"
    assert metrics["rerank_enabled"] is True
    assert metrics["pre_rerank"]["dataset_name"] == "msmt17_v1"
    assert metrics["num_query"] == 4
    assert metrics["num_gallery"] == 24
    assert metrics["pre_rerank"]["num_query"] == 4
    assert metrics["pre_rerank"]["num_gallery"] == 24
    for key in ("rank1", "rank5", "rank10", "mAP"):
        assert 0 <= metrics[key] <= 1


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"config": {"model": {}}},
        {"model": {}},
    ],
)
def test_run_evaluation_rejects_malformed_checkpoints(
    tmp_path: Path,
    checkpoint: dict,
) -> None:
    checkpoint_path = tmp_path / "bad.pth"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError):
        run_evaluation(
            checkpoint_path=checkpoint_path,
            data_root=DATA_ROOT,
            output_dir=tmp_path / "eval",
            device="cpu",
            max_query=1,
            max_gallery=1,
        )
