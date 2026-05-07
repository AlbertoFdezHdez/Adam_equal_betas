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

EXPERIMENT_MODULE="${EXPERIMENT_MODULE:?Set EXPERIMENT_MODULE, e.g. resnet18_cifar100}"

LOG_ROOT="$REPO_ROOT/descargas/logs/grid24_seed0/${EXPERIMENT_MODULE}"
SUMMARY_FILE="$LOG_ROOT/submitted_jobs.tsv"

PARTITION="${PARTITION:-q5}"
NODELIST="${NODELIST:-r4n02}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
GRES="${GRES:-gpu:1}"

DATA_ROOT="${DATA_ROOT:-/scratch1/hernanal/datasets}"
NANOGPT_DATA_DIR="${NANOGPT_DATA_DIR:-/scratch1/hernanal/nanoGPT/data/slimpajama}"
NANOGPT_ROOT="${NANOGPT_ROOT:-/scratch1/hernanal/nanoGPT}"

SEED="${SEED:-0}"
BETA_VALUES=("0.9" "0.968" "0.99" "0.9968" "0.999")
COMBINATIONS=()

for beta1 in "${BETA_VALUES[@]}"; do
  for beta2 in "${BETA_VALUES[@]}"; do
    if [[ "$beta1" == "0.9" && "$beta2" == "0.9" ]]; then
      continue
    fi
    COMBINATIONS+=("${beta1},${beta2}")
  done
done

if [[ "${#COMBINATIONS[@]}" -ne 24 ]]; then
  echo "Expected 24 beta combinations, found ${#COMBINATIONS[@]}"
  exit 1
fi

mkdir -p "$LOG_ROOT"
printf "chain\tposition\texperiment\tbeta1\tbeta2\tseed\tjobid\tdependency\n" > "$SUMMARY_FILE"

submit_job() {
  local chain_name="$1"
  local position="$2"
  local beta1="$3"
  local beta2="$4"
  local dependency_jobid="${5:-}"

  local -a dependency_args=()
  local dependency_label="none"
  if [[ -n "$dependency_jobid" ]]; then
    dependency_args=(--dependency "afterany:${dependency_jobid}")
    dependency_label="$dependency_jobid"
  fi

  local job_name="${EXPERIMENT_MODULE}_s${SEED}_grid24"
  local output_file="$LOG_ROOT/${chain_name}_${position}_b1-${beta1}_b2-${beta2}_%j.out"

  local jobid
  jobid="$(
    sbatch --parsable \
      --partition "$PARTITION" \
      --nodelist "$NODELIST" \
      --nodes 1 \
      --cpus-per-task "$CPUS" \
      --mem "$MEM" \
      --gres "$GRES" \
      --job-name "$job_name" \
      --output "$output_file" \
      --chdir "$REPO_ROOT" \
      --export "ALL,CODE_ROOT=$CODE_ROOT,REPO_ROOT=$REPO_ROOT,PYTHON_BIN=$PYTHON_BIN,EXPERIMENT_MODULE=$EXPERIMENT_MODULE,BETA1=$beta1,BETA2=$beta2,SEED=$SEED,DATA_ROOT=$DATA_ROOT,NANOGPT_DATA_DIR=$NANOGPT_DATA_DIR,NANOGPT_ROOT=$NANOGPT_ROOT" \
      "${dependency_args[@]}" \
      "$CODE_ROOT/launchers/launch_experiment.sh"
  )"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$chain_name" "$position" "$EXPERIMENT_MODULE" "$beta1" "$beta2" "$SEED" "$jobid" "$dependency_label" >> "$SUMMARY_FILE"
  echo "[grid24] submitted chain=${chain_name} position=${position} beta1=${beta1} beta2=${beta2} jobid=${jobid} dependency=${dependency_label}" >&2
  printf "%s" "$jobid"
}

submit_chain_slice() {
  local chain_name="$1"
  local start_index="$2"
  local end_index="$3"
  local previous_jobid=""
  local position=1
  local idx
  for (( idx=start_index; idx<=end_index; idx++ )); do
    IFS=',' read -r beta1 beta2 <<< "${COMBINATIONS[$idx]}"
    previous_jobid="$(submit_job "$chain_name" "$position" "$beta1" "$beta2" "$previous_jobid")"
    position=$((position + 1))
  done
}

submit_chain_slice "chain1" 0 7
submit_chain_slice "chain2" 8 15
submit_chain_slice "chain3" 16 23

echo "[grid24] submissions written to $SUMMARY_FILE"
