# Reproduction guide

Run every command from the repository root. Commands that train large models or download large datasets are marked **server/full**.

## 1. Quick offline check

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python analysis/generate_controlled_figures.py --output-dir outputs/controlled
```

Expected controlled outputs are `figure_equal_betas.pdf`, `figure_unequal_betas.pdf`, and `controlled_signal_summary.json`. The PDFs correspond to Figures 1 and 2. Pass `--use-tex` only when a working LaTeX installation is available.

A very short Digits pipeline checks file formats and numerical identities without reproducing the published table:

```bash
python analysis/train_digits_gradient_trajectory.py \
  --output-dir outputs/smoke_digits --steps 12 --seed 0 --beta1 0.9 --beta2 0.999
python analysis/literal_decomposition.py \
  --gradients outputs/smoke_digits/gradients.npy \
  --output outputs/smoke_digits/decomposition.json \
  --beta1 0.9 --beta2 0.999 --windows 2 6 --burn-in 0
```

## 2. Figures 1 and 2

```bash
python analysis/generate_controlled_figures.py --output-dir outputs/controlled
```

The signal is deterministic and has no random seed. Index 0 stores the zero initial Adam state; gradient updates are replayed for indices 1 through 600 with bias-correction exponent equal to the index. This convention has negligible visual impact in the plotted range but is stated for exactness.

## 3. Table 1 and Appendix B: Digits decomposition

The following is a complete local run of 9 beta pairs by 3 seeds. It performs 27 independent full-batch trainings and writes one approximately 19.8 MiB gradient array per run.

```bash
python analysis/run_digits_grid.py --output-root outputs/digits --steps 1500
python analysis/summarize_literal_decomposition.py \
  --input-root outputs/digits \
  --output-dir outputs/decomposition_tables
```

Per-run files:

- `outputs/digits/b1-<beta1>_b2-<beta2>_seed-<seed>/gradients.npy`: pre-update gradients, `(1500, 3466)`, `float32`.
- `.../metadata.json`: split, model, optimizer, seed, accuracy, and storage conventions.
- `.../decomposition.json`: statistics for windows 6, 10, 100, and 1000.

Aggregated outputs include `table1_main.csv`, `appendix_short_window.csv`, `appendix_reverse_order.csv`, and all-pair summaries. These generated files contain the values to be inserted into the corresponding manuscript tables; no result CSV is bundled with this code-only archive.

To reanalyze existing gradient arrays without retraining:

```bash
python analysis/run_digits_grid.py --output-root outputs/digits --skip-training --force
python analysis/summarize_literal_decomposition.py \
  --input-root outputs/digits --output-dir outputs/decomposition_tables
```

## 4. Six full training experiments

Install server dependencies and configure data/cache paths:

```bash
python -m pip install -r requirements-training.txt
export TRAINING_DATA_DIR="$PWD/data"
export HF_HOME="$PWD/data/hf_cache"
export TORCH_HOME="$PWD/data/torch_cache"
export ADAM_BETA_OUTPUT_ROOT="$PWD/outputs"
```

For NanoGPT, also clone its public repository separately and set:

```bash
export NANOGPT_ROOT=/absolute/path/to/nanoGPT
```

For SlimPajama:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='cerebras/SlimPajama-627B', repo_type='dataset', local_dir='data/slimpajama-raw', allow_patterns=['train/chunk1/*','validation/chunk1/*'])"
export SLIMPAJAMA_DIR="$PWD/data/slimpajama-raw"
export NANOGPT_DATA_DIR="$PWD/data/slimpajama"
python code/runtime_assets/nanogpt_slimpajama_prepare.py
```

### Single run

```bash
EXPERIMENT_MODULE=resnet18_cifar100 BETA1=0.9 BETA2=0.9 SEED=0 \
  bash code/launchers/launch_experiment.sh
```

Each run writes JSON to `outputs/results/<experiment>/runs/`. The JSON contains beta values, seed, protocol metadata, epoch losses, validation metrics, learning-rate history, per-step `updates_norm`, and per-step gradient norms.

### SLURM grids

Replace `gpu_partition` with a valid local partition. The historical 3-by-3 experiments used seeds 1, 10, and 100; SlimPajama used seed 1 only. ResNet18 used a 5-by-5 grid and seeds 0, 1, and 2.

```bash
PARTITION=gpu_partition EXPERIMENT_MODULE=efficientnet_tinyimagenet \
  BETA1S_CSV=0.9,0.99,0.999 SEEDS_CSV=1,10,100 \
  bash code/launchers/launch_grid_slurm.sh

PARTITION=gpu_partition EXPERIMENT_MODULE=t5_squad \
  BETA1S_CSV=0.9,0.99,0.999 SEEDS_CSV=1,10,100 \
  bash code/launchers/launch_grid_slurm.sh

PARTITION=gpu_partition EXPERIMENT_MODULE=nanogpt_wikitext \
  BETA1S_CSV=0.9,0.99,0.999 SEEDS_CSV=1,10,100 \
  bash code/launchers/launch_grid_slurm.sh

PARTITION=gpu_partition EXPERIMENT_MODULE=nanogpt_slimpajama \
  BETA1S_CSV=0.9,0.99,0.999 SEEDS_CSV=1 \
  bash code/launchers/launch_grid_slurm.sh

PARTITION=gpu_partition EXPERIMENT_MODULE=vit_cifar100 \
  BETA1S_CSV=0.9,0.99,0.999 SEEDS_CSV=1,10,100 \
  bash code/launchers/launch_grid_slurm.sh

PARTITION=gpu_partition EXPERIMENT_MODULE=resnet18_cifar100 \
  BETA1S_CSV=0.9,0.968,0.99,0.9968,0.999 SEEDS_CSV=0,1,2 \
  bash code/launchers/launch_grid_slurm.sh
```

The SLURM grid is a Cartesian product. `BETA2S_CSV` defaults to `BETA1S_CSV`. Use `ARRAY_CONCURRENCY` to cap simultaneous jobs and override `CPUS`, `MEM`, or `GRES` for the target cluster.

## 5. Table 2, Appendix C figures, and Table 4

### From fresh JSON runs

After all runs are present at the locations specified by `configs/training_sources_json.json`:

```bash
python analysis/analyze_training_dynamics.py \
  --config configs/training_sources_json.json \
  --data-root . \
  --output-dir outputs/training_dynamics
```

This writes:

- `table2_omega.csv`: window-200 first-difference oscillation values and diagonal-selection rates;
- `table4_ablation.csv`: selection rates for both metrics and all five smoothing windows;
- one PDF grid per experiment unless `--skip-plots` is passed.

### From historical raw results

Place trusted historical pickles under `raw_results/legacy/` and ResNet JSON files under `raw_results/json/resnet18_cifar100/runs/`, exactly as listed in `configs/training_sources_legacy.json`, then run:

```bash
python analysis/analyze_training_dynamics.py \
  --config configs/training_sources_legacy.json \
  --data-root . \
  --output-dir outputs/training_dynamics_legacy
```

Only load pickle files from a trusted source. The analyzer propagates the top-level historical `epochs` field into each run before removing the first three epochs; this detail is required to reproduce the published values.

Table 4 uses interior second differences by default, matching the displayed formula:

```bash
python analysis/analyze_training_dynamics.py \
  --config configs/training_sources_json.json --data-root . \
  --output-dir outputs/training_dynamics_boundaries \
  --omega2-convention one-sided-boundaries
```

The optional command implements the prose convention with one-sided boundaries. On the archived six-task results, both conventions gave exactly the same published selection rates.

## 6. What cannot be reproduced from this archive alone

- All data and result files are excluded. Costly training outputs must be regenerated and placed according to one of the configuration files.
- The precise NanoGPT Git commit used originally was not recorded.
- Exact package versions for all original server runs were not recoverable.
- Historical output filenames say EfficientNet-B0 although the executed constructor is EfficientNet-V2-S; the package does not relabel old numerical results.
- The ViT run used the default 1000-output head. Changing it to 100 outputs would be a new experiment, not a reproduction.
