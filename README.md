# Complex-Scene Person Re-Identification

[English](README.md) | [中文](README_zh.md)

This repository contains a compact course-project codebase for person re-identification
under complex scenarios. It supports standard and extended Re-ID experiments on
Market-1501, MSMT17, Occluded-ReID, and VC-Clothes, with ResNet-50, OSNet, and
ViT/DeiT-based recipes.

## Environment Setting

Recommended environment:

- Python 3.10
- [uv](https://docs.astral.sh/uv/) for dependency management
- CUDA GPU for full MSMT17 training and evaluation

Install dependencies:

```bash
uv sync --group dev --group cuda
```

For local macOS CPU/MPS development, use the mac group instead:

```bash
uv sync --group dev --group mac
```

If you are running on a server with a prepared environment and do not want `uv` to
resync packages every time, prefix commands with:

```bash
UV_NO_SYNC=1
```

The training wrapper selects the dependency group through `REID_UV_GROUP`. On CUDA
servers, set it once before training:

```bash
export REID_UV_GROUP=cuda
```

## Data Layout

Place datasets under `data/` with the following layout:

```text
data/
  Market-1501-v15.09.15/
  MSMT17_V1/
  Occluded_REID/
  VC-Clothes/
  pretrained/
    osnet_x1_0_imagenet.pth
```

The OSNet pretrained checkpoint is only required by OSNet pretrained configs. DeiT/ViT
recipes use `timm` pretrained weights, which are downloaded or loaded from the local
cache by `timm`.

## Project Structure

```text
.
├── configs/                         # YAML experiment recipes
│   ├── resnet50_ce_pretrained_msmt17.yaml
│   ├── osnet_x1_0_ce_triplet_pretrained_msmt17.yaml
│   └── deit_base_patch16_global_local_sie_part_msmt17_384.yaml
├── reid/                            # Main Python package
│   ├── data/                        # Dataset parsers, transforms, dataloaders
│   │   ├── common.py                # Shared ReIDSample structure
│   │   ├── dataloader.py            # Dataset registry and DataLoader builders
│   │   ├── market1501.py            # Market-1501 split parser
│   │   ├── msmt17.py                # MSMT17_V1 list-file parser
│   │   ├── occluded_reid.py         # Occluded-ReID query/gallery parser
│   │   ├── vc_clothes.py            # VC-Clothes parser with clothes metadata
│   │   ├── samplers.py              # PK identity batch sampler
│   │   └── transforms.py            # Train/eval image preprocessing
│   ├── engine/                      # Training and evaluation entry logic
│   │   ├── train.py                 # Training loop, checkpointing, resume
│   │   └── evaluate.py              # Checkpoint loading and feature extraction
│   ├── evaluation/                  # Retrieval metrics and post-processing
│   │   ├── distance.py              # Cosine and Euclidean distance helpers
│   │   ├── metrics.py               # CMC/mAP evaluation protocols
│   │   └── rerank.py                # K-reciprocal re-ranking
│   ├── losses/                      # Classification and metric learning losses
│   │   ├── classification.py        # Cross entropy and label smoothing
│   │   └── triplet.py               # Batch-hard triplet loss
│   ├── models/                      # Re-ID model definitions
│   │   ├── registry.py              # Model factory used by training/evaluation
│   │   ├── resnet_reid.py           # Self-written ResNet-50 Re-ID baseline
│   │   ├── osnet_reid.py            # Self-written OSNet x1.0 Re-ID model
│   │   └── vit_reid.py              # DeiT/ViT global-local Re-ID model
│   └── utils/                       # Config, seeding, multiprocessing helpers
├── scripts/
│   ├── train.py                     # Python training CLI
│   ├── train_resnet_ce.sh           # Shell wrapper for timestamped runs
│   └── evaluate.py                  # Python evaluation CLI
└── tests/                           # Pytest coverage for data/model/train/eval
```

## Quick Start

Run a small smoke test first:

```bash
UV_NO_SYNC=1 uv run --group dev --group cuda pytest tests/test_training_smoke.py
```

Train the Market-1501 ResNet-50 baseline:

```bash
bash scripts/train_resnet_ce.sh \
  configs/resnet50_ce_pretrained_market1501.yaml \
  resnet50_ce_pretrained_market1501 \
  cuda
```

Train the MSMT17 ResNet-50 baseline:

```bash
bash scripts/train_resnet_ce.sh \
  configs/resnet50_ce_pretrained_msmt17.yaml \
  resnet50_ce_pretrained_msmt17 \
  cuda
```

Train the MSMT17 OSNet final CNN recipe:

```bash
bash scripts/train_resnet_ce.sh \
  configs/osnet_x1_0_ce_triplet_pretrained_msmt17.yaml \
  osnet_x1_0_ce_triplet_pretrained_msmt17 \
  cuda
```

Train the DeiT-based transformer extension:

```bash
bash scripts/train_resnet_ce.sh \
  configs/deit_base_patch16_global_local_sie_part_msmt17_384.yaml \
  deit_base_patch16_global_local_sie_part_msmt17_384 \
  cuda
```

Resume a run from the latest checkpoint:

```bash
bash scripts/train_resnet_ce.sh \
  <config.yaml> \
  <run_name> \
  cuda \
  outputs/<timestamp>_<run_name>/ckpt/latest.pth
```

## Evaluation

Evaluate a checkpoint on MSMT17:

```bash
UV_NO_SYNC=1 uv run --group dev --group cuda python scripts/evaluate.py \
  --checkpoint outputs/<run>/ckpt/best.pth \
  --data-root data/MSMT17_V1 \
  --dataset-name msmt17_v1 \
  --output-dir outputs/<run>/eval_msmt17 \
  --device cuda \
  --batch-size 128 \
  --num-workers 4 \
  --distance cosine \
  --query-chunk-size 256
```

Run post-hoc k-reciprocal re-ranking:

```bash
UV_NO_SYNC=1 uv run --group dev --group cuda python scripts/evaluate.py \
  --checkpoint outputs/<run>/ckpt/best.pth \
  --data-root data/MSMT17_V1 \
  --dataset-name msmt17_v1 \
  --output-dir outputs/<run>/eval_rerank \
  --device cuda \
  --batch-size 128 \
  --num-workers 4 \
  --distance cosine \
  --query-chunk-size 256 \
  --rerank \
  --rerank-k1 20 \
  --rerank-k2 6 \
  --rerank-lambda 0.3
```

Occluded-ReID and VC-Clothes can be evaluated with the same script by changing
`--dataset-name` and `--data-root` to `occluded_reid` / `data/Occluded_REID` or
`vc_clothes` / `data/VC-Clothes`.

## Outputs

Training creates one directory per run:

```text
outputs/<timestamp>_<run_name>/
  ckpt/
    latest.pth
    best.pth
  logs/
  metrics.json
  run_summary.md
```

Evaluation outputs are written to the selected `--output-dir`, usually as
`eval_metrics.json` and log files.
