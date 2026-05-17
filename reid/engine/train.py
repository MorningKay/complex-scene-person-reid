"""Minimal CE-only training engine."""

from __future__ import annotations

import json
import math
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from reid.data import build_reid_dataloader, normalize_dataset_name
from reid.engine.evaluate import DEFAULT_QUERY_CHUNK_SIZE, evaluate_model_on_reid_dataset
from reid.losses import build_classification_loss
from reid.models import resnet50_reid
from reid.utils import (
    configure_torch_multiprocessing_sharing,
    set_seed,
    validate_training_config,
    write_config,
)

Config = dict[str, Any]
_TRAINING_EVAL_DATASETS = {"market1501", "msmt17_v1"}


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
    amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip_norm: float | None = None,
) -> dict[str, float | int]:
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_samples = 0
    total_correct = 0
    num_batches = 0

    for batch_index, (images, pids, _camids, _paths) in enumerate(dataloader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = _map_pids_to_labels(pids, pid_to_label, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits, _features = model(images)
            ce_loss = criterion(logits, labels)
            loss = ce_loss
        if amp_enabled:
            if scaler is None:
                raise ValueError("AMP training requires a GradScaler")
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        batch_size = images.shape[0]
        loss_value = float(loss.detach().cpu())
        ce_loss_value = float(ce_loss.detach().cpu())
        batch_correct = int(logits.detach().argmax(dim=1).eq(labels).sum().cpu())
        batch_id_acc = batch_correct / batch_size
        lr = _current_lr(optimizer)
        total_loss += loss_value * batch_size
        total_ce_loss += ce_loss_value * batch_size
        total_samples += batch_size
        total_correct += batch_correct
        num_batches += 1

        if log_interval > 0 and batch_index % log_interval == 0:
            _log(
                f"epoch={epoch} batch={batch_index} lr={lr:.8f} "
                f"loss={loss_value:.6f} ce_loss={ce_loss_value:.6f} "
                f"id_acc={batch_id_acc:.6f}",
                log_file,
            )

    if num_batches == 0:
        raise ValueError("No training batches were processed")

    avg_train_loss = total_loss / total_samples
    avg_ce_loss = total_ce_loss / total_samples
    train_id_acc = total_correct / total_samples
    return {
        "epoch": epoch,
        "avg_train_loss": avg_train_loss,
        "avg_ce_loss": avg_ce_loss,
        "train_id_acc": train_id_acc,
        "lr": _current_lr(optimizer),
        "num_batches": num_batches,
        "num_samples": total_samples,
    }


def run_training(
    config: Config,
    output_dir: str | Path,
    device: str | torch.device | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    sharing_strategy = configure_torch_multiprocessing_sharing()
    validate_training_config(config)
    output_path = Path(output_dir)
    logs_dir = output_path / "logs"
    ckpt_dir = output_path / "ckpt"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "train.txt"
    if log_file.exists() and resume_checkpoint is None:
        log_file.unlink()

    resume_state = _load_resume_checkpoint(resume_checkpoint) if resume_checkpoint else None
    if resume_state is not None:
        _assert_resume_config_compatible(config, resume_state)

    write_config(config, output_path / "config.yaml")
    set_seed(int(config["run"]["seed"]))
    resolved_device = _resolve_device(device)

    _log(f"run_name={config['run']['name']}", log_file)
    _log(f"device={resolved_device}", log_file)
    _log(f"torch_sharing_strategy={sharing_strategy}", log_file)
    dataset_name = _dataset_name(config)
    amp_enabled = _amp_enabled(config, resolved_device)
    grad_clip_norm = _grad_clip_norm(config)
    random_erasing = bool(config["data"].get("random_erasing", False))
    random_erasing_prob = _random_erasing_prob(config)
    _log(f"dataset_name={dataset_name}", log_file)
    _log(f"model_pretrained={bool(config['model'].get('pretrained', False))}", log_file)
    _log(f"random_erasing={random_erasing}", log_file)
    _log(f"random_erasing_prob={random_erasing_prob:.6f}", log_file)
    _log(f"scheduler_name={_scheduler_name(config)}", log_file)
    _log(f"amp_enabled={amp_enabled}", log_file)
    _log(f"grad_clip_norm={_format_optional_metric(grad_clip_norm)}", log_file)
    if resume_checkpoint is not None:
        _log(f"resume_checkpoint={resume_checkpoint}", log_file)

    dataloader = build_reid_dataloader(
        name=dataset_name,
        root=config["data"]["root"],
        split="train",
        batch_size=int(config["data"]["batch_size"]),
        image_size=tuple(config["data"].get("image_size", (256, 128))),
        random_erasing=random_erasing,
        random_erasing_prob=random_erasing_prob,
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"].get("pin_memory", False)),
        drop_last=bool(config["data"].get("drop_last", False)),
    )
    pid_to_label = _build_pid_to_label(dataloader)
    num_train_ids = len(pid_to_label)
    num_classes = int(config["model"]["num_classes"])
    if num_classes != num_train_ids:
        raise ValueError(
            "model.num_classes must match the number of train identities, "
            f"got {num_classes} and {num_train_ids}"
        )

    model = resnet50_reid(
        num_classes=num_classes,
        feature_dim=int(config["model"]["feature_dim"]),
        last_stride=int(config["model"]["last_stride"]),
        pretrained=bool(config["model"].get("pretrained", False)),
    ).to(resolved_device)
    criterion = build_classification_loss(
        label_smoothing=float(config["loss"]["label_smoothing"])
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    if resume_state is not None:
        _load_training_state(
            resume_state=resume_state,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=resolved_device,
            pid_to_label=pid_to_label,
        )
        start_epoch = int(resume_state["epoch"]) + 1
        if start_epoch > int(config["train"]["epochs"]):
            raise ValueError(
                "Resume checkpoint has already reached or exceeded train.epochs, "
                f"got checkpoint epoch {resume_state['epoch']} and train.epochs "
                f"{config['train']['epochs']}"
            )

    eval_config = config.get("eval", {})
    eval_enabled = bool(eval_config.get("enabled", False))
    history = _resume_history(output_path, resume_state)
    best_epoch_metrics, best_epoch = _best_from_history(history, eval_enabled)
    best_loss_value = _best_metric_value(best_epoch_metrics, "avg_train_loss")
    best_map_value = _best_eval_metric_value(best_epoch_metrics, "mAP")
    best_rank1_value = _best_eval_metric_value(best_epoch_metrics, "rank1")
    best_loss = best_loss_value if best_loss_value is not None else float("inf")
    best_map = best_map_value if best_map_value is not None else -1.0
    best_rank1 = best_rank1_value if best_rank1_value is not None else 0.0
    best_metric_name = "mAP" if eval_enabled else "avg_train_loss"
    start_time = time.time()

    for epoch in range(start_epoch, int(config["train"]["epochs"]) + 1):
        _set_epoch_lr(optimizer, config, epoch)
        train_metrics = train_one_epoch(
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
            amp_enabled=amp_enabled,
            scaler=scaler,
            grad_clip_norm=grad_clip_norm,
        )
        epoch_metrics: dict[str, Any] = dict(train_metrics)
        history.append(epoch_metrics)
        _log(
            "epoch={epoch} avg_train_loss={avg_train_loss:.6f} "
            "avg_ce_loss={avg_ce_loss:.6f} train_id_acc={train_id_acc:.6f} "
            "lr={lr:.8f} num_batches={num_batches} num_samples={num_samples}".format(
                **epoch_metrics
            ),
            log_file,
        )

        if _should_evaluate_epoch(eval_config, epoch):
            if dataset_name not in _TRAINING_EVAL_DATASETS:
                valid = ", ".join(sorted(_TRAINING_EVAL_DATASETS))
                raise ValueError(
                    f"Training-time evaluation currently supports only: {valid}; "
                    f"disable eval.enabled for {dataset_name}"
                )
            eval_metrics = evaluate_model_on_reid_dataset(
                model=model,
                dataset_name=dataset_name,
                data_root=config["data"]["root"],
                image_size=tuple(config["data"].get("image_size", (256, 128))),
                device=resolved_device,
                batch_size=int(eval_config["batch_size"]),
                num_workers=int(eval_config["num_workers"]),
                distance=eval_config["distance"],
                max_query=eval_config.get("max_query"),
                max_gallery=eval_config.get("max_gallery"),
                query_chunk_size=int(
                    eval_config.get("query_chunk_size", DEFAULT_QUERY_CHUNK_SIZE)
                ),
                log_file=log_file,
            )
            epoch_metrics["eval"] = eval_metrics
            _log(
                "epoch={epoch} rank1={rank1:.6f} rank5={rank5:.6f} "
                "rank10={rank10:.6f} mAP={mAP:.6f}".format(
                    epoch=epoch,
                    **eval_metrics,
                ),
                log_file,
            )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler_state": _scheduler_state(config, optimizer, epoch),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "metrics": epoch_metrics,
            "history": history,
            "config": config,
            "pid_to_label": pid_to_label,
        }
        latest_path = ckpt_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        if _is_best_epoch(epoch_metrics, eval_enabled, best_loss, best_map, best_rank1):
            best_loss = float(epoch_metrics["avg_train_loss"])
            if eval_enabled:
                best_map = float(epoch_metrics["eval"]["mAP"])
                best_rank1 = float(epoch_metrics["eval"]["rank1"])
            best_epoch = epoch
            best_epoch_metrics = epoch_metrics
            shutil.copyfile(latest_path, ckpt_dir / "best.pth")

    elapsed_seconds = time.time() - start_time
    final_epoch_metrics = history[-1]
    metrics = {
        "run_name": config["run"]["name"],
        "dataset_name": dataset_name,
        "device": str(resolved_device),
        "epoch": final_epoch_metrics["epoch"],
        "avg_train_loss": final_epoch_metrics["avg_train_loss"],
        "avg_ce_loss": final_epoch_metrics["avg_ce_loss"],
        "train_id_acc": final_epoch_metrics["train_id_acc"],
        "lr": final_epoch_metrics["lr"],
        "random_erasing": random_erasing,
        "random_erasing_prob": random_erasing_prob,
        "scheduler_name": _scheduler_name(config),
        "scheduler_state": _scheduler_state(
            config, optimizer, int(final_epoch_metrics["epoch"])
        ),
        "amp_enabled": amp_enabled,
        "grad_clip_norm": grad_clip_norm,
        "num_batches": final_epoch_metrics["num_batches"],
        "num_samples": final_epoch_metrics["num_samples"],
        "num_train_ids": num_train_ids,
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_avg_train_loss": _best_metric_value(best_epoch_metrics, "avg_train_loss"),
        "best_mAP": _best_eval_metric_value(best_epoch_metrics, "mAP"),
        "best_rank1": _best_eval_metric_value(best_epoch_metrics, "rank1"),
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
        raise ValueError("Dataloader dataset must expose Re-ID samples")

    pids = sorted({int(sample.pid) for sample in samples if int(sample.pid) >= 0})
    return {pid: label for label, pid in enumerate(pids)}


def _load_resume_checkpoint(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Resume checkpoint must be a mapping")
    for key in ("model", "optimizer", "epoch", "config", "pid_to_label"):
        if key not in checkpoint:
            raise ValueError(f"Resume checkpoint is missing required key: {key}")
    return checkpoint


def _assert_resume_config_compatible(config: Config, checkpoint: dict[str, Any]) -> None:
    checkpoint_config = checkpoint["config"]
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Resume checkpoint config must be a mapping")

    current = deepcopy(config)
    previous = deepcopy(checkpoint_config)
    current_epochs = int(current["train"].pop("epochs"))
    previous_epochs = int(previous["train"].pop("epochs"))
    if current_epochs < previous_epochs:
        raise ValueError("Current train.epochs must be greater than or equal to checkpoint config")
    if current != previous:
        raise ValueError("Current config must match checkpoint config when resuming")


def _load_training_state(
    resume_state: dict[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    pid_to_label: dict[int, int],
) -> None:
    checkpoint_pid_to_label = resume_state["pid_to_label"]
    if checkpoint_pid_to_label != pid_to_label:
        raise ValueError("Resume checkpoint pid_to_label does not match the current dataset")

    model.load_state_dict(resume_state["model"])
    optimizer.load_state_dict(resume_state["optimizer"])
    _move_optimizer_state(optimizer, device)
    scaler_state = resume_state.get("scaler")
    if isinstance(scaler_state, dict):
        scaler.load_state_dict(scaler_state)


def _move_optimizer_state(optimizer: Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _resume_history(
    output_path: Path,
    resume_state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if resume_state is None:
        return []

    metrics_path = output_path / "metrics.json"
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        history = metrics.get("history", [])
        if isinstance(history, list):
            return list(history)

    history = resume_state.get("history")
    if isinstance(history, list):
        return list(history)

    metrics = resume_state.get("metrics")
    if isinstance(metrics, dict):
        return [metrics]

    return []


def _best_from_history(
    history: list[dict[str, Any]],
    eval_enabled: bool,
) -> tuple[dict[str, Any] | None, int]:
    best_epoch_metrics: dict[str, Any] | None = None
    best_loss = float("inf")
    best_map = -1.0
    best_rank1 = 0.0
    best_epoch = 0

    for epoch_metrics in history:
        if _is_best_epoch(epoch_metrics, eval_enabled, best_loss, best_map, best_rank1):
            best_epoch_metrics = epoch_metrics
            best_loss = float(epoch_metrics["avg_train_loss"])
            if eval_enabled:
                best_map = float(epoch_metrics["eval"]["mAP"])
                best_rank1 = float(epoch_metrics["eval"]["rank1"])
            best_epoch = int(epoch_metrics["epoch"])

    return best_epoch_metrics, best_epoch


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_enabled(config: Config, device: torch.device) -> bool:
    return bool(config["train"].get("amp", False)) and device.type == "cuda"


def _grad_clip_norm(config: Config) -> float | None:
    value = config["train"].get("grad_clip_norm")
    if value is None:
        return None
    return float(value)


def _random_erasing_prob(config: Config) -> float:
    return float(config["data"].get("random_erasing_prob", 0.5))


def _dataset_name(config: Config) -> str:
    return normalize_dataset_name(config.get("data", {}).get("name"))


def _current_lr(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _scheduler_name(config: Config) -> str:
    scheduler_config = config.get("scheduler")
    if not scheduler_config:
        return "constant"
    return str(scheduler_config["name"])


def _set_epoch_lr(optimizer: Optimizer, config: Config, epoch: int) -> float:
    lr = _compute_epoch_lr(config, epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def _compute_epoch_lr(config: Config, epoch: int) -> float:
    base_lr = float(config["optimizer"]["lr"])
    scheduler_config = config.get("scheduler")
    if not scheduler_config:
        return base_lr

    total_epochs = int(config["train"]["epochs"])
    min_lr = float(scheduler_config["min_lr"])
    warmup_epochs = int(scheduler_config["warmup_epochs"])
    warmup_factor = float(scheduler_config["warmup_factor"])

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        if warmup_epochs == 1:
            return base_lr
        progress = (epoch - 1) / (warmup_epochs - 1)
        return base_lr * (warmup_factor + progress * (1 - warmup_factor))

    decay_epochs = total_epochs - warmup_epochs
    if decay_epochs <= 1:
        return base_lr

    decay_index = epoch - warmup_epochs
    progress = (decay_index - 1) / (decay_epochs - 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _scheduler_state(
    config: Config,
    optimizer: Optimizer,
    epoch: int,
) -> dict[str, float | int | str]:
    return {
        "name": _scheduler_name(config),
        "last_epoch": epoch,
        "lr": _current_lr(optimizer),
    }


def _should_evaluate_epoch(eval_config: dict[str, Any], epoch: int) -> bool:
    if not bool(eval_config.get("enabled", False)):
        return False
    return epoch % int(eval_config["interval"]) == 0


def _is_best_epoch(
    epoch_metrics: dict[str, Any],
    eval_enabled: bool,
    best_loss: float,
    best_map: float,
    best_rank1: float,
) -> bool:
    if not eval_enabled:
        return float(epoch_metrics["avg_train_loss"]) < best_loss
    eval_metrics = epoch_metrics.get("eval")
    if not isinstance(eval_metrics, dict):
        return False

    current_map = float(eval_metrics["mAP"])
    current_rank1 = float(eval_metrics["rank1"])
    return (current_map > best_map) or (current_map == best_map and current_rank1 > best_rank1)


def _best_metric_value(epoch_metrics: dict[str, Any] | None, name: str) -> float | None:
    if epoch_metrics is None:
        return None
    return float(epoch_metrics[name])


def _best_eval_metric_value(epoch_metrics: dict[str, Any] | None, name: str) -> float | None:
    if epoch_metrics is None or "eval" not in epoch_metrics:
        return None
    return float(epoch_metrics["eval"][name])


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
            f"- dataset_name: {metrics['dataset_name']}",
            f"- device: {metrics['device']}",
            f"- model_pretrained: {bool(config['model'].get('pretrained', False))}",
            f"- random_erasing: {metrics['random_erasing']}",
            f"- random_erasing_prob: {metrics['random_erasing_prob']:.6f}",
            f"- scheduler_name: {metrics['scheduler_name']}",
            f"- amp_enabled: {metrics['amp_enabled']}",
            f"- grad_clip_norm: {_format_optional_metric(metrics['grad_clip_norm'])}",
            f"- epochs: {config['train']['epochs']}",
            f"- num_train_ids: {metrics['num_train_ids']}",
            f"- final_avg_train_loss: {metrics['avg_train_loss']:.6f}",
            f"- final_avg_ce_loss: {metrics['avg_ce_loss']:.6f}",
            f"- final_train_id_acc: {metrics['train_id_acc']:.6f}",
            f"- best_metric_name: {metrics['best_metric_name']}",
            f"- best_epoch: {metrics['best_epoch']}",
            f"- best_avg_train_loss: {_format_optional_metric(metrics['best_avg_train_loss'])}",
            f"- best_mAP: {_format_optional_metric(metrics['best_mAP'])}",
            f"- best_rank1: {_format_optional_metric(metrics['best_rank1'])}",
            f"- latest_checkpoint: {output_path / 'ckpt' / 'latest.pth'}",
            f"- best_checkpoint: {output_path / 'ckpt' / 'best.pth'}",
            f"- smoke: {bool(config['run'].get('smoke', False))}",
            "",
        ]
    )
    (output_path / "run_summary.md").write_text(summary, encoding="utf-8")


def _format_optional_metric(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:.6f}"
