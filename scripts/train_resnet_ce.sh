#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/resnet50_ce_market1501.yaml}"
RUN_NAME="${2:-resnet50_ce}"
DEVICE="${3:-}"
UV_GROUP="${REID_UV_GROUP:-mac}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/${TIMESTAMP}_${RUN_NAME}"
RAW_LOG_PATH="${OUTPUT_DIR}/logs/raw_log.txt"

mkdir -p "${OUTPUT_DIR}/logs"

CMD=(
  uv run --group dev --group "${UV_GROUP}"
  python -u scripts/train.py
  --config "${CONFIG_PATH}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${DEVICE}" ]]; then
  CMD+=(--device "${DEVICE}")
fi

{
  echo "config=${CONFIG_PATH}"
  echo "run_name=${RUN_NAME}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "uv_group=${UV_GROUP}"
  if [[ -n "${DEVICE}" ]]; then
    echo "device=${DEVICE}"
  fi
  echo "command=${CMD[*]}"
  "${CMD[@]}"
} 2>&1 | tee "${RAW_LOG_PATH}"
