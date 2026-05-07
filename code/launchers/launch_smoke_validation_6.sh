#!/usr/bin/env bash
set -uo pipefail

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

LOG_ROOT="$REPO_ROOT/descargas/logs/smoke_validation_6"

PYTHON_BIN="${PYTHON_BIN:-python}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.99}"
SEED="${SEED:-1}"
DATA_ROOT="${DATA_ROOT:-/scratch1/hernanal/datasets}"
NANOGPT_DATA_DIR="${NANOGPT_DATA_DIR:-/scratch1/hernanal/nanoGPT/data/slimpajama}"
NANOGPT_ROOT="${NANOGPT_ROOT:-/scratch1/hernanal/nanoGPT}"

mkdir -p "$LOG_ROOT"

SUMMARY_LOG="$LOG_ROOT/summary.log"
STATUS=0

export TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-$DATA_ROOT}"
export HF_HOME="${HF_HOME:-$TRAINING_DATA_DIR/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$TRAINING_DATA_DIR/torch_cache}"
export NANOGPT_DATA_DIR
export NANOGPT_ROOT

declare -a EXPERIMENTS=(
  "resnet18_cifar100"
  "efficientnet_tinyimagenet"
  "t5_squad"
  "vit_cifar100"
  "nanogpt_wikitext"
  "nanogpt_slimpajama"
)

cd "$CODE_ROOT"

echo "[smoke_validation_6] starting smoke suite" > "$SUMMARY_LOG"

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
  LOG_FILE="$LOG_ROOT/${EXPERIMENT}_smoke_b1-${BETA1}_b2-${BETA2}_seed-${SEED}.log"
  echo "[smoke_validation_6] running $EXPERIMENT" | tee "$LOG_FILE"
  if [[ "$EXPERIMENT" == "nanogpt_slimpajama" ]]; then
    "$PYTHON_BIN" -m "experiments.${EXPERIMENT}" \
      --beta1 "$BETA1" \
      --beta2 "$BETA2" \
      --seed "$SEED" \
      --smoke-test \
      --data-dir "$NANOGPT_DATA_DIR" 2>&1 | tee -a "$LOG_FILE"
  else
    "$PYTHON_BIN" -m "experiments.${EXPERIMENT}" \
      --beta1 "$BETA1" \
      --beta2 "$BETA2" \
      --seed "$SEED" \
      --smoke-test 2>&1 | tee -a "$LOG_FILE"
  fi
  CMD_STATUS=${PIPESTATUS[0]}
  if [[ "$CMD_STATUS" -eq 0 ]]; then
    echo "[smoke_validation_6] status=OK experiment=$EXPERIMENT" | tee -a "$LOG_FILE" "$SUMMARY_LOG"
  else
    STATUS=1
    echo "[smoke_validation_6] status=FAIL experiment=$EXPERIMENT exit_code=$CMD_STATUS" | tee -a "$LOG_FILE" "$SUMMARY_LOG"
  fi
done

if [[ "$STATUS" -eq 0 ]]; then
  echo "[smoke_validation_6] all six smoke experiments finished" | tee -a "$SUMMARY_LOG"
else
  echo "[smoke_validation_6] smoke suite finished with failures" | tee -a "$SUMMARY_LOG"
fi

exit "$STATUS"
