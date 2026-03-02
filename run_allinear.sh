#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
CONFIG_PATH="$ROOT_DIR/model/allinear_16l.json"
RUN_NAME="allinear"
DEVICE="cuda:0"
BATCH_SIZE=16
MICRO_BATCH_SIZE=1
LEARNING_RATE=""
MAX_STEPS=""
EVAL_INTERVAL="10000"
EVAL_NUM_BATCHES=""
WANDB_ROOT="${WANDB_ROOT:-/dev/shm/wandb}"
WANDB_LOG_INTERVAL="${WANDB_LOG_INTERVAL:-10}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --run_name)
      RUN_NAME="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --micro_batch_size)
      MICRO_BATCH_SIZE="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --max_steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --eval_interval)
      EVAL_INTERVAL="$2"
      shift 2
      ;;
    --eval_num_batches)
      EVAL_NUM_BATCHES="$2"
      shift 2
      ;;
    --wandb_log_interval)
      WANDB_LOG_INTERVAL="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift 1
      ;;
  esac
done

if [[ "$DEVICE" == cuda:* ]]; then
  GPU_INDEX="${DEVICE#cuda:}"
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
elif [[ "$DEVICE" == cpu ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

WANDB_RUN_DIR="${WANDB_ROOT%/}/${RUN_NAME}"
mkdir -p "${WANDB_RUN_DIR}"
export WANDB_DIR="${WANDB_RUN_DIR}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export WANDB_DISABLE_CODE="${WANDB_DISABLE_CODE:-true}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python -m model.train --config "$CONFIG_PATH" \
  --run_name "$RUN_NAME" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --micro_batch_size "$MICRO_BATCH_SIZE" \
  ${LEARNING_RATE:+--learning_rate "$LEARNING_RATE"} \
  ${MAX_STEPS:+--max_steps "$MAX_STEPS"} \
  ${EVAL_INTERVAL:+--eval_interval "$EVAL_INTERVAL"} \
  ${EVAL_NUM_BATCHES:+--eval_num_batches "$EVAL_NUM_BATCHES"} \
  ${WANDB_LOG_INTERVAL:+--wandb_log_interval "$WANDB_LOG_INTERVAL"} \
  "${EXTRA_ARGS[@]}"
