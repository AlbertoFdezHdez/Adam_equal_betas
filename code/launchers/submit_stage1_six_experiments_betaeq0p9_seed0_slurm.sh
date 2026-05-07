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

LOG_ROOT="$REPO_ROOT/descargas/logs/stage1_six_experiments_betaeq0p9_seed0"
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

BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.9}"
SEED="${SEED:-0}"

CHAIN_1=("t5_squad" "resnet18_cifar100")
CHAIN_2=("nanogpt_slimpajama" "vit_cifar100")
CHAIN_3=("efficientnet_tinyimagenet" "nanogpt_wikitext")

mkdir -p "$LOG_ROOT"
printf "chain\tposition\texperiment\tbeta1\tbeta2\tseed\tjobid\tdependency\n" > "$SUMMARY_FILE"

submit_job() {
  local chain_name="$1"
  local position="$2"
  local experiment="$3"
  local dependency_jobid="${4:-}"

  local -a dependency_args=()
  local dependency_label="none"
  if [[ -n "$dependency_jobid" ]]; then
    dependency_args=(--dependency "afterany:${dependency_jobid}")
    dependency_label="$dependency_jobid"
  fi

  local job_name="${experiment}_s${SEED}_b0p9eq"
  local output_file="$LOG_ROOT/${chain_name}_${position}_${experiment}_%j.out"

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
      --export "ALL,CODE_ROOT=$CODE_ROOT,REPO_ROOT=$REPO_ROOT,PYTHON_BIN=$PYTHON_BIN,EXPERIMENT_MODULE=$experiment,BETA1=$BETA1,BETA2=$BETA2,SEED=$SEED,DATA_ROOT=$DATA_ROOT,NANOGPT_DATA_DIR=$NANOGPT_DATA_DIR,NANOGPT_ROOT=$NANOGPT_ROOT" \
      "${dependency_args[@]}" \
      "$CODE_ROOT/launchers/launch_experiment.sh"
  )"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$chain_name" "$position" "$experiment" "$BETA1" "$BETA2" "$SEED" "$jobid" "$dependency_label" >> "$SUMMARY_FILE"
  echo "[stage1] submitted chain=${chain_name} position=${position} experiment=${experiment} jobid=${jobid} dependency=${dependency_label}" >&2
  printf "%s" "$jobid"
}

submit_chain() {
  local chain_name="$1"
  shift
  local previous_jobid=""
  local position=1
  local experiment
  for experiment in "$@"; do
    previous_jobid="$(submit_job "$chain_name" "$position" "$experiment" "$previous_jobid")"
    position=$((position + 1))
  done
}

submit_chain "chain1" "${CHAIN_1[@]}"
submit_chain "chain2" "${CHAIN_2[@]}"
submit_chain "chain3" "${CHAIN_3[@]}"

echo "[stage1] submissions written to $SUMMARY_FILE"
