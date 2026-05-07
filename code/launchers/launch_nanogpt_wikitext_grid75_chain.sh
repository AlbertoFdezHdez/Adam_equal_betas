#!/usr/bin/env bash
set -uo pipefail

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

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_MODULE="${EXPERIMENT_MODULE:-nanogpt_wikitext}"
SEED="${SEED:?Set SEED}"
DATA_ROOT="${DATA_ROOT:-/scratch1/hernanal/datasets}"
NANOGPT_ROOT="${NANOGPT_ROOT:-/scratch1/hernanal/nanoGPT}"
NANOGPT_DATA_DIR="${NANOGPT_DATA_DIR:-/scratch1/hernanal/nanoGPT/data/slimpajama}"

LOG_ROOT="$REPO_ROOT/descargas/logs/nanogpt_wikitext_grid75/seed_${SEED}"
SUMMARY_FILE="$LOG_ROOT/summary.tsv"

BETA_VALUES=("0.900" "0.968" "0.990" "0.9968" "0.999")

mkdir -p "$LOG_ROOT"
printf "position\tbeta1\tbeta2\tseed\tstatus\texit_code\twall_time_total_sec\n" > "$SUMMARY_FILE"

export TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-$DATA_ROOT}"
export HF_HOME="${HF_HOME:-$TRAINING_DATA_DIR/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$TRAINING_DATA_DIR/torch_cache}"
export NANOGPT_ROOT
export NANOGPT_DATA_DIR

cd "$CODE_ROOT"

position=0
for beta1 in "${BETA_VALUES[@]}"; do
  for beta2 in "${BETA_VALUES[@]}"; do
    position=$((position + 1))
    run_log="$LOG_ROOT/pos_${position}_b1-${beta1}_b2-${beta2}.log"
    echo "[nanogpt_wikitext_grid75][seed=${SEED}] running position=${position} beta1=${beta1} beta2=${beta2}" | tee "$run_log"
    set +e
    "$PYTHON_BIN" -m "experiments.${EXPERIMENT_MODULE}" \
      --beta1 "$beta1" \
      --beta2 "$beta2" \
      --seed "$SEED" 2>&1 | tee -a "$run_log"
    exit_code=${PIPESTATUS[0]}
    set -e

    wall_time_total_sec="NA"
    if [[ "$exit_code" -eq 0 ]]; then
      json_path="$REPO_ROOT/descargas/results/nanogpt_wikitext/runs/nanogpt_wikitext__NanoGPT__WikiText__b1-${beta1//./p}__b2-${beta2//./p}__seed-${SEED}.json"
      if [[ -f "$json_path" ]]; then
        wall_time_total_sec="$(
          python - <<'PY' "$json_path"
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text(encoding='utf-8'))
print(data.get('wall_time_total_sec', 'NA'))
PY
        )"
      fi
      echo "[nanogpt_wikitext_grid75][seed=${SEED}] status=OK position=${position} beta1=${beta1} beta2=${beta2}" | tee -a "$run_log"
    else
      echo "[nanogpt_wikitext_grid75][seed=${SEED}] status=FAIL position=${position} beta1=${beta1} beta2=${beta2} exit_code=${exit_code}" | tee -a "$run_log"
    fi

    if [[ "$exit_code" -eq 0 ]]; then
      status_label="OK"
    else
      status_label="FAIL"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$position" "$beta1" "$beta2" "$SEED" "$status_label" "$exit_code" "$wall_time_total_sec" >> "$SUMMARY_FILE"
  done
done

echo "[nanogpt_wikitext_grid75][seed=${SEED}] chain finished"
