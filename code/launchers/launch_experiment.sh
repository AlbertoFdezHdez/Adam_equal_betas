#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CODE_ROOT:-}" ]]; then
  CODE_ROOT="$(cd "$CODE_ROOT" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  CODE_ROOT="$(cd "$SLURM_SUBMIT_DIR/code" && pwd)"
else
  CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
else
  REPO_ROOT="$(cd "$CODE_ROOT/.." && pwd)"
fi

LOG_ROOT="$REPO_ROOT/descargas/logs"

EXPERIMENT_MODULE="${EXPERIMENT_MODULE:?Set EXPERIMENT_MODULE, e.g. resnet18_cifar100}"
BETA1="${BETA1:?Set BETA1}"
BETA2="${BETA2:?Set BETA2}"
SEED="${SEED:?Set SEED}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/scratch1/hernanal/datasets}"
NANOGPT_DATA_DIR="${NANOGPT_DATA_DIR:-/scratch1/hernanal/nanoGPT/data/slimpajama}"
NANOGPT_ROOT="${NANOGPT_ROOT:-/scratch1/hernanal/nanoGPT}"

export TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-$DATA_ROOT}"
export HF_HOME="${HF_HOME:-$TRAINING_DATA_DIR/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$TRAINING_DATA_DIR/torch_cache}"
export NANOGPT_DATA_DIR
export NANOGPT_ROOT

mkdir -p "$LOG_ROOT/$EXPERIMENT_MODULE"
LOG_FILE="$LOG_ROOT/$EXPERIMENT_MODULE/${EXPERIMENT_MODULE}_b1-${BETA1}_b2-${BETA2}_seed-${SEED}.log"

cd "$CODE_ROOT"
if [[ "$EXPERIMENT_MODULE" == "nanogpt_slimpajama" ]]; then
  "$PYTHON_BIN" -m "experiments.${EXPERIMENT_MODULE}" --beta1 "$BETA1" --beta2 "$BETA2" --seed "$SEED" --data-dir "$NANOGPT_DATA_DIR" 2>&1 | tee "$LOG_FILE"
else
  "$PYTHON_BIN" -m "experiments.${EXPERIMENT_MODULE}" --beta1 "$BETA1" --beta2 "$BETA2" --seed "$SEED" 2>&1 | tee "$LOG_FILE"
fi
