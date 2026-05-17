#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/resnet50_ce_market1501.yaml}"
RUN_NAME="${2:-resnet50_ce}"
DEVICE="${3:-}"
RESUME_CHECKPOINT="${4:-}"
UV_GROUP="${REID_UV_GROUP:-mac}"
NOFILE_LIMIT="${REID_NOFILE_LIMIT:-65535}"

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  CKPT_DIR="$(cd "$(dirname "${RESUME_CHECKPOINT}")" && pwd)"
  OUTPUT_DIR="$(cd "${CKPT_DIR}/.." && pwd)"
else
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_DIR="outputs/${TIMESTAMP}_${RUN_NAME}"
fi
RAW_LOG_PATH="${OUTPUT_DIR}/logs/raw_log.txt"

mkdir -p "${OUTPUT_DIR}/logs"

configure_nofile_limit() {
  local before
  local after
  before="$(ulimit -n 2>/dev/null || true)"
  echo "nofile_limit_before=${before:-unknown}"
  echo "nofile_limit_target=${NOFILE_LIMIT}"
  if [[ "${before}" =~ ^[0-9]+$ && "${NOFILE_LIMIT}" =~ ^[0-9]+$ ]] && (( before >= NOFILE_LIMIT )); then
    echo "nofile_limit_after=${before}"
    return 0
  fi
  if ! ulimit -n "${NOFILE_LIMIT}" 2>/dev/null; then
    echo "warning=failed_to_set_nofile_limit"
  fi
  after="$(ulimit -n 2>/dev/null || true)"
  echo "nofile_limit_after=${after:-unknown}"
}

CMD=(
  uv run --group dev --group "${UV_GROUP}"
  python -u scripts/train.py
  --config "${CONFIG_PATH}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${DEVICE}" ]]; then
  CMD+=(--device "${DEVICE}")
fi

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  CMD+=(--resume "${RESUME_CHECKPOINT}")
fi

{
  echo "config=${CONFIG_PATH}"
  echo "run_name=${RUN_NAME}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "uv_group=${UV_GROUP}"
  configure_nofile_limit
  if [[ -n "${DEVICE}" ]]; then
    echo "device=${DEVICE}"
  fi
  if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    echo "resume_checkpoint=${RESUME_CHECKPOINT}"
  fi
  echo "command=${CMD[*]}"
  "${CMD[@]}"
} 2>&1 | tee -a "${RAW_LOG_PATH}"
