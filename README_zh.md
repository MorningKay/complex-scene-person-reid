# 复杂场景行人重识别

[English](README.md) | [中文](README_zh.md)

本仓库是一个面向课程大作业的行人重识别代码库，主要覆盖复杂场景下的训练与评估流程。当前支持
Market-1501、MSMT17、Occluded-ReID 和 VC-Clothes，并提供 ResNet-50、OSNet 以及
ViT/DeiT 扩展实验配置。

## 环境配置

推荐环境：

- Python 3.10
- 使用 [uv](https://docs.astral.sh/uv/) 管理依赖
- 完整 MSMT17 训练与评估建议使用 CUDA GPU

安装 CUDA 环境依赖：

```bash
uv sync --group dev --group cuda
```

如果是在 macOS 本地做 CPU/MPS 调试，可以使用 mac 依赖组：

```bash
uv sync --group dev --group mac
```

如果服务器环境已经安装完成，不希望每次运行时重新同步依赖，可以在命令前加上：

```bash
UV_NO_SYNC=1
```

训练脚本通过 `REID_UV_GROUP` 选择依赖组。在 CUDA 服务器上训练前，可以先设置：

```bash
export REID_UV_GROUP=cuda
```

## 数据目录

请将数据集放在 `data/` 目录下，并保持以下结构：

```text
data/
  Market-1501-v15.09.15/
  MSMT17_V1/
  Occluded_REID/
  VC-Clothes/
  pretrained/
    osnet_x1_0_imagenet.pth
```

其中 OSNet 的 ImageNet 预训练权重只在使用 OSNet 预训练配置时需要。DeiT/ViT 配置使用 `timm`
加载预训练权重，首次运行时会从网络下载或从本地缓存读取。

## 项目结构

```text
.
├── configs/                         # 实验配置文件
│   ├── resnet50_ce_pretrained_msmt17.yaml
│   ├── osnet_x1_0_ce_triplet_pretrained_msmt17.yaml
│   └── deit_base_patch16_global_local_sie_part_msmt17_384.yaml
├── reid/                            # 主要 Python 源码包
│   ├── data/                        # 数据集解析、变换和 DataLoader
│   │   ├── common.py                # 统一 ReIDSample 样本结构
│   │   ├── dataloader.py            # 数据集 registry 和 DataLoader 构建入口
│   │   ├── market1501.py            # Market-1501 数据划分解析
│   │   ├── msmt17.py                # MSMT17_V1 list 文件解析
│   │   ├── occluded_reid.py         # Occluded-ReID query/gallery 解析
│   │   ├── vc_clothes.py            # VC-Clothes 解析和衣服编号 metadata
│   │   ├── samplers.py              # PK identity batch sampler
│   │   └── transforms.py            # 训练和评估图像预处理
│   ├── engine/                      # 训练与评估主流程
│   │   ├── train.py                 # 训练循环、checkpoint 和断点恢复
│   │   └── evaluate.py              # checkpoint 加载和特征提取
│   ├── evaluation/                  # 检索指标和后处理
│   │   ├── distance.py              # cosine / Euclidean 距离计算
│   │   ├── metrics.py               # CMC/mAP 评估协议
│   │   └── rerank.py                # k-reciprocal re-ranking
│   ├── losses/                      # 分类损失与度量学习损失
│   │   ├── classification.py        # Cross entropy 和 label smoothing
│   │   └── triplet.py               # Batch-hard triplet loss
│   ├── models/                      # Re-ID 模型定义
│   │   ├── registry.py              # 训练/评估共用的模型构建入口
│   │   ├── resnet_reid.py           # 自写 ResNet-50 Re-ID baseline
│   │   ├── osnet_reid.py            # 自写 OSNet x1.0 Re-ID 模型
│   │   └── vit_reid.py              # DeiT/ViT global-local Re-ID 模型
│   └── utils/                       # 配置、随机种子、多进程辅助函数
├── scripts/
│   ├── train.py                     # Python 训练 CLI
│   ├── train_resnet_ce.sh           # 创建时间戳输出目录的训练脚本
│   └── evaluate.py                  # Python 评估 CLI
└── tests/                           # 数据、模型、训练和评估的 pytest 测试
```

## 快速开始

建议先运行一个小规模测试：

```bash
UV_NO_SYNC=1 uv run --group dev --group cuda pytest tests/test_training_smoke.py
```

训练 Market-1501 上的 ResNet-50 基线：

```bash
bash scripts/train_resnet_ce.sh \
  configs/resnet50_ce_pretrained_market1501.yaml \
  resnet50_ce_pretrained_market1501 \
  cuda
```

训练 MSMT17 上的 ResNet-50 基线：

```bash
bash scripts/train_resnet_ce.sh \
  configs/resnet50_ce_pretrained_msmt17.yaml \
  resnet50_ce_pretrained_msmt17 \
  cuda
```

训练 MSMT17 上的 OSNet 最终卷积模型：

```bash
bash scripts/train_resnet_ce.sh \
  configs/osnet_x1_0_ce_triplet_pretrained_msmt17.yaml \
  osnet_x1_0_ce_triplet_pretrained_msmt17 \
  cuda
```

训练基于 DeiT 的 Transformer 扩展模型：

```bash
bash scripts/train_resnet_ce.sh \
  configs/deit_base_patch16_global_local_sie_part_msmt17_384.yaml \
  deit_base_patch16_global_local_sie_part_msmt17_384 \
  cuda
```

从 checkpoint 继续训练：

```bash
bash scripts/train_resnet_ce.sh \
  <config.yaml> \
  <run_name> \
  cuda \
  outputs/<timestamp>_<run_name>/ckpt/latest.pth
```

## 评估

在 MSMT17 上评估 checkpoint：

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

运行 k-reciprocal 重排序后处理：

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

Occluded-ReID 和 VC-Clothes 也使用同一个评估脚本，只需要将 `--dataset-name` 和 `--data-root`
分别改为 `occluded_reid` / `data/Occluded_REID` 或 `vc_clothes` / `data/VC-Clothes`。

## 输出文件

每次训练会生成一个独立目录：

```text
outputs/<timestamp>_<run_name>/
  ckpt/
    latest.pth
    best.pth
  logs/
  metrics.json
  run_summary.md
```

评估结果会写入指定的 `--output-dir`，通常包括 `eval_metrics.json` 和对应日志。
