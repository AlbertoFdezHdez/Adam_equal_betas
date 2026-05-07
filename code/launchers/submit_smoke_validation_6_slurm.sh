#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$CODE_ROOT/.." && pwd)"
LOG_ROOT="$REPO_ROOT/descargas/logs/smoke_validation_6"

PARTITION="${PARTITION:-q5}"
NODELIST="${NODELIST:-r4n02}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
GRES="${GRES:-gpu:1}"

BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.99}"
SEED="${SEED:-1}"
DATA_ROOT="${DATA_ROOT:-/scratch1/hernanal/datasets}"
NANOGPT_DATA_DIR="${NANOGPT_DATA_DIR:-/scratch1/hernanal/nanoGPT/data/slimpajama}"
NANOGPT_ROOT="${NANOGPT_ROOT:-/scratch1/hernanal/nanoGPT}"

mkdir -p "$LOG_ROOT"

sbatch \
  --partition "$PARTITION" \
  --nodelist "$NODELIST" \
  --nodes 1 \
  --cpus-per-task "$CPUS" \
  --mem "$MEM" \
  --gres "$GRES" \
  --job-name "smoke_validation_6" \
  --output "$LOG_ROOT/slurm_%j.out" \
  --chdir "$REPO_ROOT" \
  --export "ALL,CODE_ROOT=$CODE_ROOT,REPO_ROOT=$REPO_ROOT,PYTHON_BIN=$PYTHON_BIN,BETA1=$BETA1,BETA2=$BETA2,SEED=$SEED,DATA_ROOT=$DATA_ROOT,NANOGPT_DATA_DIR=$NANOGPT_DATA_DIR,NANOGPT_ROOT=$NANOGPT_ROOT" \
  "$CODE_ROOT/launchers/launch_smoke_validation_6.sh"
