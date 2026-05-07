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
NANOGPT_ROOT="${NANOGPT_ROOT:-/scratch1/hernanal/nanoGPT}"
NANOGPT_DATA_DIR="${NANOGPT_DATA_DIR:-/scratch1/hernanal/nanoGPT/data/slimpajama}"

RESNET_DEP_0="${RESNET_DEP_0:-4163636}"
RESNET_DEP_1="${RESNET_DEP_1:-4163637}"
RESNET_DEP_2="${RESNET_DEP_2:-4163638}"

LOG_ROOT="$REPO_ROOT/descargas/logs/nanogpt_wikitext_grid75"
mkdir -p "$LOG_ROOT"

seeds=(0 1 2)
deps=("$RESNET_DEP_0" "$RESNET_DEP_1" "$RESNET_DEP_2")

for idx in "${!seeds[@]}"; do
  SEED="${seeds[$idx]}"
  DEP="${deps[$idx]}"
  sbatch \
    --partition "$PARTITION" \
    --nodelist "$NODELIST" \
    --nodes 1 \
    --cpus-per-task "$CPUS" \
    --mem "$MEM" \
    --gres "$GRES" \
    --job-name "nanogpt_wikitext_grid75_s${SEED}" \
    --output "$LOG_ROOT/seed_${SEED}_%j.out" \
    --chdir "$REPO_ROOT" \
    --dependency "afterany:${DEP}" \
    --export "ALL,CODE_ROOT=$CODE_ROOT,REPO_ROOT=$REPO_ROOT,PYTHON_BIN=$PYTHON_BIN,EXPERIMENT_MODULE=nanogpt_wikitext,SEED=$SEED,DATA_ROOT=$DATA_ROOT,NANOGPT_ROOT=$NANOGPT_ROOT,NANOGPT_DATA_DIR=$NANOGPT_DATA_DIR" \
    "$CODE_ROOT/launchers/launch_nanogpt_wikitext_grid75_chain.sh"
done
