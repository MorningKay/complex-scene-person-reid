"""Minimal CE-only training engine."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from reid.data import build_market1501_dataloader
from reid.losses import build_classification_loss
from reid.models import resnet50_reid
from reid.utils import set_seed, validate_training_config, write_config

Config = dict[str, Any]


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    pid_to_label: dict[int, int],
    epoch: int,
    max_batches: int | None = None,
    log_interval: int = 20,
    log_file: Path | None = None,
) -> dict[str, float | int]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    num_batches = 0

    for batch_index, (images, pids, _camids, _paths) in enumerate(dataloader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = _map_pids_to_labels(pids, pid_to_label, device)

        optimizer.zero_grad(set_to_none=True)
        logits, _features = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        loss_value = float(loss.detach().cpu())
        total_loss += loss_value * batch_size
        total_samples += batch_size
        num_batches += 1

        if log_interval > 0 and batch_index % log_interval == 0:
            _log(
                f"epoch={epoch} batch={batch_index} loss={loss_value:.6f}",
                log_file,
            )

    if num_batches == 0:
        raise ValueError("No training batches were processed")

    avg_train_loss = total_loss / total_samples
    return {
        "epoch": epoch,
        "avg_train_loss": avg_train_loss,
        "num_batches": num_batches,
        "num_samples": total_samples,
    }


def run_training(
    config: Config,
    output_dir: str | Path,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    validate_training_config(config)
    output_path = Path(output_dir)
    logs_dir = output_path / "logs"
    ckpt_dir = output_path / "ckpt"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "train.txt"
    if log_file.exists():
        log_file.unlink()

    write_config(config, output_path / "config.yaml")
    set_seed(int(config["run"]["seed"]))
    resolved_device = _resolve_device(device)

    _log(f"run_name={config['run']['name']}", log_file)
    _log(f"device={resolved_device}", log_file)

    dataloader = build_market1501_dataloader(
        root=config["data"]["root"],
        split="train",
        batch_size=int(config["data"]["batch_size"]),
        image_size=tuple(config["data"].get("image_size", (256, 128))),
        random_erasing=bool(config["data"].get("random_erasing", False)),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"].get("pin_memory", False)),
        drop_last=bool(config["data"].get("drop_last", False)),
    )
    pid_to_label = _build_pid_to_label(dataloader)
    num_classes = int(config["model"]["num_classes"])
    if num_classes != len(pid_to_label):
        raise ValueError(
            "model.num_classes must match the number of train identities, "
            f"got {num_classes} and {len(pid_to_label)}"
        )

    model = resnet50_reid(
        num_classes=num_classes,
        feature_dim=int(config["model"]["feature_dim"]),
        last_stride=int(config["model"]["last_stride"]),
    ).to(resolved_device)
    criterion = build_classification_loss(
        label_smoothing=float(config["loss"]["label_smoothing"])
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        epoch_metrics = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=resolved_device,
            pid_to_label=pid_to_label,
            epoch=epoch,
            max_batches=config["train"].get("max_batches"),
            log_interval=int(config["train"].get("log_interval", 20)),
            log_file=log_file,
        )
        history.append(epoch_metrics)
        _log(
            "epoch={epoch} avg_train_loss={avg_train_loss:.6f} "
            "num_batches={num_batches} num_samples={num_samples}".format(**epoch_metrics),
            log_file,
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": epoch_metrics,
            "config": config,
            "pid_to_label": pid_to_label,
        }
        latest_path = ckpt_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        if float(epoch_metrics["avg_train_loss"]) < best_loss:
            best_loss = float(epoch_metrics["avg_train_loss"])
            best_epoch = epoch
            shutil.copyfile(latest_path, ckpt_dir / "best.pth")

    elapsed_seconds = time.time() - start_time
    final_epoch_metrics = history[-1]
    metrics = {
        "run_name": config["run"]["name"],
        "device": str(resolved_device),
        "epoch": final_epoch_metrics["epoch"],
        "avg_train_loss": final_epoch_metrics["avg_train_loss"],
        "num_batches": final_epoch_metrics["num_batches"],
        "num_samples": final_epoch_metrics["num_samples"],
        "best_epoch": best_epoch,
        "best_avg_train_loss": best_loss,
        "elapsed_seconds": elapsed_seconds,
        "history": history,
    }
    _write_json(metrics, output_path / "metrics.json")
    _write_run_summary(config, metrics, output_path)
    _log(f"done avg_train_loss={metrics['avg_train_loss']:.6f}", log_file)
    return metrics


def _map_pids_to_labels(
    pids: torch.Tensor,
    pid_to_label: dict[int, int],
    device: torch.device,
) -> torch.Tensor:
    try:
        labels = [pid_to_label[int(pid)] for pid in pids.tolist()]
    except KeyError as exc:
        raise ValueError(f"Unknown train pid in batch: {exc.args[0]}") from exc
    return torch.tensor(labels, dtype=torch.long, device=device)


def _build_pid_to_label(dataloader: torch.utils.data.DataLoader) -> dict[int, int]:
    samples = getattr(dataloader.dataset, "samples", None)
    if samples is None:
        raise ValueError("Dataloader dataset must expose Market-1501 samples")

    pids = sorted({int(sample.pid) for sample in samples if int(sample.pid) >= 0})
    return {pid: label for label, pid in enumerate(pids)}


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log(message: str, log_file: Path | None) -> None:
    print(message, flush=True)
    if log_file is not None:
        with log_file.open("a", encoding="utf-8") as file:
            file.write(message + "\n")


def _write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def _write_run_summary(config: Config, metrics: dict[str, Any], output_path: Path) -> None:
    summary = "\n".join(
        [
            "# Run Summary",
            "",
            f"- run_name: {config['run']['name']}",
            f"- device: {metrics['device']}",
            f"- epochs: {config['train']['epochs']}",
            f"- final_avg_train_loss: {metrics['avg_train_loss']:.6f}",
            f"- best_epoch: {metrics['best_epoch']}",
            f"- best_avg_train_loss: {metrics['best_avg_train_loss']:.6f}",
            f"- latest_checkpoint: {output_path / 'ckpt' / 'latest.pth'}",
            f"- best_checkpoint: {output_path / 'ckpt' / 'best.pth'}",
            f"- smoke: {bool(config['run'].get('smoke', False))}",
            "",
        ]
    )
    (output_path / "run_summary.md").write_text(summary, encoding="utf-8")
