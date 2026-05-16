"""Checkpoint evaluation engine for Market-1501 retrieval."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from reid.data import build_market1501_dataloader
from reid.data.market1501 import Market1501Sample
from reid.evaluation import cosine_distance, evaluate_market1501, pairwise_distance
from reid.models import resnet50_reid

Config = dict[str, Any]
DistanceName = Literal["cosine", "euclidean"]


@dataclass(frozen=True)
class FeatureSet:
    features: torch.Tensor
    pids: torch.Tensor
    camids: torch.Tensor
    paths: tuple[str, ...]


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any], Config]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a mapping")
    if "model" not in checkpoint:
        raise ValueError("Checkpoint is missing required key: model")
    if "config" not in checkpoint:
        raise ValueError("Checkpoint is missing required key: config")

    config = checkpoint["config"]
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError("Checkpoint config is missing required section: model")

    model_config = config["model"]
    model = resnet50_reid(
        num_classes=int(model_config["num_classes"]),
        feature_dim=int(model_config["feature_dim"]),
        last_stride=int(model_config["last_stride"]),
    )
    model.load_state_dict(checkpoint["model"])
    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    model.eval()
    return model, checkpoint, config


def extract_features(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str | torch.device,
) -> FeatureSet:
    resolved_device = torch.device(device)
    features: list[torch.Tensor] = []
    pids: list[torch.Tensor] = []
    camids: list[torch.Tensor] = []
    paths: list[str] = []

    model.eval()
    with torch.inference_mode():
        for images, batch_pids, batch_camids, batch_paths in dataloader:
            images = images.to(resolved_device, non_blocking=True)
            batch_features = model(images, return_feature=True)
            features.append(batch_features.detach().cpu())
            pids.append(batch_pids.detach().cpu())
            camids.append(batch_camids.detach().cpu())
            paths.extend(batch_paths)

    if not features:
        raise ValueError("No features were extracted")

    return FeatureSet(
        features=torch.cat(features, dim=0),
        pids=torch.cat(pids, dim=0),
        camids=torch.cat(camids, dim=0),
        paths=tuple(paths),
    )


def run_evaluation(
    checkpoint_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    device: str | torch.device | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    distance: DistanceName = "cosine",
    max_query: int | None = None,
    max_gallery: int | None = None,
) -> dict[str, Any]:
    if distance not in {"cosine", "euclidean"}:
        raise ValueError("distance must be one of: cosine, euclidean")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    output_path = Path(output_dir)
    logs_dir = output_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "eval.txt"
    if log_file.exists():
        log_file.unlink()

    resolved_device = _resolve_device(device)
    model, _checkpoint, config = load_model_from_checkpoint(checkpoint_path, resolved_device)
    image_size = tuple(config.get("data", {}).get("image_size", (256, 128)))

    _log(f"checkpoint={checkpoint_path}", log_file)
    _log(f"device={resolved_device}", log_file)
    _log(f"distance={distance}", log_file)

    query_loader = build_market1501_dataloader(
        root=data_root,
        split="query",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        num_workers=num_workers,
    )
    gallery_loader = build_market1501_dataloader(
        root=data_root,
        split="gallery",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        num_workers=num_workers,
    )
    _limit_eval_samples(query_loader, gallery_loader, max_query=max_query, max_gallery=max_gallery)

    start_time = time.time()
    query = extract_features(model, query_loader, resolved_device)
    gallery = extract_features(model, gallery_loader, resolved_device)
    _log(f"query_features={tuple(query.features.shape)}", log_file)
    _log(f"gallery_features={tuple(gallery.features.shape)}", log_file)

    if distance == "cosine":
        distmat = cosine_distance(query.features, gallery.features)
    else:
        distmat = pairwise_distance(query.features, gallery.features)

    retrieval_metrics = evaluate_market1501(
        distmat=distmat,
        query_pids=query.pids,
        gallery_pids=gallery.pids,
        query_camids=query.camids,
        gallery_camids=gallery.camids,
        max_rank=10,
    )
    metrics = {
        "checkpoint": str(checkpoint_path),
        "distance": distance,
        "rank1": _rank_at(retrieval_metrics.cmc, 1),
        "rank5": _rank_at(retrieval_metrics.cmc, 5),
        "rank10": _rank_at(retrieval_metrics.cmc, 10),
        "mAP": retrieval_metrics.mAP,
        "num_valid_queries": retrieval_metrics.num_valid_queries,
        "num_query": int(query.features.shape[0]),
        "num_gallery": int(gallery.features.shape[0]),
        "elapsed_seconds": time.time() - start_time,
    }
    _write_json(metrics, output_path / "eval_metrics.json")
    _log(
        "rank1={rank1:.6f} rank5={rank5:.6f} rank10={rank10:.6f} "
        "mAP={mAP:.6f} valid_queries={num_valid_queries}".format(**metrics),
        log_file,
    )
    return metrics


def _limit_eval_samples(
    query_loader: torch.utils.data.DataLoader,
    gallery_loader: torch.utils.data.DataLoader,
    max_query: int | None,
    max_gallery: int | None,
) -> None:
    if max_query is not None and max_query <= 0:
        raise ValueError("max_query must be positive when provided")
    if max_gallery is not None and max_gallery <= 0:
        raise ValueError("max_gallery must be positive when provided")

    if max_query is not None:
        query_loader.dataset.samples = query_loader.dataset.samples[:max_query]

    if max_gallery is not None:
        query_samples = list(query_loader.dataset.samples)
        gallery_samples = list(gallery_loader.dataset.samples)
        selected = _select_gallery_subset(gallery_samples, query_samples, max_gallery)
        gallery_loader.dataset.samples = selected


def _select_gallery_subset(
    gallery_samples: list[Market1501Sample],
    query_samples: list[Market1501Sample],
    max_gallery: int,
) -> list[Market1501Sample]:
    query_keys = {(sample.pid, sample.camid) for sample in query_samples}
    preferred: list[Market1501Sample] = []
    fallback: list[Market1501Sample] = []
    preferred_paths: set[Path] = set()

    for sample in gallery_samples:
        has_cross_camera_match = any(
            sample.pid == query_pid and sample.camid != query_camid
            for query_pid, query_camid in query_keys
        )
        if sample.pid >= 0 and has_cross_camera_match:
            preferred.append(sample)
            preferred_paths.add(sample.path)
        else:
            fallback.append(sample)

    return (preferred + [sample for sample in fallback if sample.path not in preferred_paths])[
        :max_gallery
    ]


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rank_at(cmc: torch.Tensor, rank: int) -> float:
    if cmc.numel() == 0:
        raise ValueError("cmc must not be empty")
    index = min(rank, cmc.numel()) - 1
    return float(cmc[index])


def _log(message: str, log_file: Path) -> None:
    print(message, flush=True)
    with log_file.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def _write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
