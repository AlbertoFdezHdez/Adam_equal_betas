#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$CODE_ROOT/.." && pwd)"
LOG_ROOT="$REPO_ROOT/descargas/logs/dataset_prepare"

PYTHON_BIN="${PYTHON_BIN:-python}"
PARTITION="${PARTITION:-q5}"
NODELIST="${NODELIST:-r4n02}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
GRES="${GRES:-gpu:0}"
DATA_ROOT="${DATA_ROOT:-/scratch1/hernanal/datasets}"
RUNTIME_ASSETS_DIR="${RUNTIME_ASSETS_DIR:-$CODE_ROOT/runtime_assets}"
KAGGLE_JSON_PATH="${KAGGLE_JSON_PATH:-$RUNTIME_ASSETS_DIR/kaggle.json}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$RUNTIME_ASSETS_DIR/hf_token.txt}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
SLIMPAJAMA_DIR="${SLIMPAJAMA_DIR:-/data/slimpajama-mini}"
NANOGPT_ROOT="${NANOGPT_ROOT:-/opt/nanoGPT}"

if [[ -z "$HF_TOKEN" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
fi

mkdir -p "$LOG_ROOT"

sbatch \
  --partition "$PARTITION" \
  --nodelist "$NODELIST" \
  --nodes 1 \
  --cpus-per-task "$CPUS" \
  --mem "$MEM" \
  --gres "$GRES" \
  --job-name "prepare_datasets" \
  --output "$LOG_ROOT/prepare_datasets_%j.out" \
  --chdir "$CODE_ROOT" \
  --export "ALL,TRAINING_DATA_DIR=$DATA_ROOT,HF_HOME=$DATA_ROOT/hf_cache,HF_DATASETS_CACHE=$DATA_ROOT/hf_cache/datasets,HF_HUB_CACHE=$DATA_ROOT/hf_cache/hub,TRANSFORMERS_CACHE=$DATA_ROOT/hf_cache/transformers,TORCH_HOME=$DATA_ROOT/torch_cache,HF_TOKEN=$HF_TOKEN,HUGGINGFACE_HUB_TOKEN=$HF_TOKEN,SLIMPAJAMA_DIR=$SLIMPAJAMA_DIR,NANOGPT_ROOT=$NANOGPT_ROOT" \
  --wrap "$PYTHON_BIN launchers/prepare_datasets.py --data-root '$DATA_ROOT' --kaggle-json '$KAGGLE_JSON_PATH' --hf-token-file '$HF_TOKEN_FILE' --slimpajama-dir '$SLIMPAJAMA_DIR' --nanogpt-root '$NANOGPT_ROOT'"
