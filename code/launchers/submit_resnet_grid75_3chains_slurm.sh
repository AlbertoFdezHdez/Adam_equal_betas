#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CODE_ROOT:-}" ]]; then
  CODE_ROOT="$(cd "$CODE_ROOT" && pwd)"
else
  CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
else
  REPO_ROOT="$(cd "$CODE_ROOT/.." && pwd)"
fi

PARTITION="${PARTITION:-q5}"
NODELIST="${NODELIST:-r4n02}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
GRES="${GRES:-gpu:1}"
DATA_ROOT="${DATA_ROOT:-/scratch1/hernanal/datasets}"

LOG_ROOT="$REPO_ROOT/descargas/logs/resnet18_cifar100_grid75"
mkdir -p "$LOG_ROOT"

for SEED in 0 1 2; do
  sbatch \
    --partition "$PARTITION" \
    --nodelist "$NODELIST" \
    --nodes 1 \
    --cpus-per-task "$CPUS" \
    --mem "$MEM" \
    --gres "$GRES" \
    --job-name "resnet18_cifar100_grid75_s${SEED}" \
    --output "$LOG_ROOT/seed_${SEED}_%j.out" \
    --chdir "$REPO_ROOT" \
    --export "ALL,CODE_ROOT=$CODE_ROOT,REPO_ROOT=$REPO_ROOT,PYTHON_BIN=$PYTHON_BIN,EXPERIMENT_MODULE=resnet18_cifar100,SEED=$SEED,DATA_ROOT=$DATA_ROOT" \
    "$CODE_ROOT/launchers/launch_resnet_grid75_chain.sh"
done
