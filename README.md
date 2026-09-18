# Why $\beta_1=\beta_2$ Is Dynamically Special in Adam

This repository studies the dynamical role of Adam's two exponential-memory parameters. Its central question is why tying the first- and second-moment memories, $\beta_1=\beta_2$, produces qualitatively different normalized updates from using unequal memories.

The repository contains the complete implementation of three complementary empirical analyses:

1. a controlled three-dimensional gradient signal that isolates sensitivity to changing coordinate magnitudes;
2. a literal discrete decomposition of Adam's normalized update on gradients produced by training a small neural network;
3. six model--dataset experiments that compare update-norm oscillations across equal and unequal momentum parameters.

## Research objectives

- Isolate the discrete magnitude-lag contribution created by different first- and second-moment memories.
- Check the exact decomposition $R=S+L^{\mathrm{disc}}+E$ on real training gradients.
- Quantify how the smoothness of the normalized update changes across the $(\beta_1,\beta_2)$ grid.
- Make every numerical convention explicit: indexing, bias correction, epsilon placement, valid-window filtering, aggregation, and smoothing.
- Provide reproducible entry points for all models and datasets used in the project.

## Experiments and outputs

| Analysis | Main implementation | Generated outputs |
|---|---|---|
| Controlled magnitude variation | `analysis/generate_controlled_figures.py` | Equal- and unequal-memory figures plus a protocol summary |
| Discrete decomposition | `analysis/train_digits_gradient_trajectory.py`, `analysis/literal_decomposition.py`, `analysis/run_digits_grid.py` | Per-run gradients and decomposition statistics |
| Decomposition aggregation | `analysis/summarize_literal_decomposition.py` | Component-budget and robustness tables |
| Training dynamics | `code/experiments/` | Per-run JSON histories for six model--dataset pairs |
| Oscillation analysis | `analysis/analyze_training_dynamics.py` | Update-smoothness tables, ablations, and experiment grids |

The detailed result-to-code map is in [`RESULTS_TO_CODE.md`](RESULTS_TO_CODE.md). Exact commands are in [`REPRODUCE.md`](REPRODUCE.md), executed checks are recorded in [`VALIDATION.md`](VALIDATION.md), and the full file inventory is in [`INVENTORY.md`](INVENTORY.md).

## Repository structure

```text
analysis/                 Numerical analyses and figure/table generation
code/experiments/         Six full training entry points
code/utils/               Models, datasets, training utilities, and I/O
code/launchers/           Local and SLURM launchers
code/runtime_assets/      SlimPajama preparation utility
configs/                  Experiment protocols and analysis input maps
tests/                    Lightweight numerical tests
```

Generated datasets, checkpoints, gradients, training histories, figures, tables, logs, credentials, and caches are intentionally excluded. Their expected locations and generation procedures are documented in [`LARGE_FILES.md`](LARGE_FILES.md).

## Quick start

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Generate the deterministic controlled experiment:

```bash
python analysis/generate_controlled_figures.py --output-dir outputs/controlled
```

Run a short end-to-end check of the discrete decomposition:

```bash
python analysis/train_digits_gradient_trajectory.py \
  --output-dir outputs/smoke_digits --steps 12 \
  --seed 0 --beta1 0.9 --beta2 0.999

python analysis/literal_decomposition.py \
  --gradients outputs/smoke_digits/gradients.npy \
  --output outputs/smoke_digits/decomposition.json \
  --beta1 0.9 --beta2 0.999 --windows 2 6 --burn-in 0
```

Install the additional dependencies required by the six full training experiments:

```bash
python -m pip install -r requirements-training.txt
```

## Full training entry points

The available experiment modules are:

- `resnet18_cifar100`
- `efficientnet_tinyimagenet`
- `t5_squad`
- `nanogpt_wikitext`
- `nanogpt_slimpajama`
- `vit_cifar100`

Example single run:

```bash
EXPERIMENT_MODULE=resnet18_cifar100 BETA1=0.9 BETA2=0.9 SEED=0 \
  bash code/launchers/launch_experiment.sh
```

Every entry point accepts `--smoke-test`. Smoke tests check the execution path but do not replace the complete experiment grids. Full local and SLURM commands are provided in [`REPRODUCE.md`](REPRODUCE.md).

The two NanoGPT experiments require a separate checkout of `karpathy/nanoGPT` containing `model.py`, `train.py`, `sample.py`, and `config/`. Set `NANOGPT_ROOT` to that checkout before running them.

## Datasets

All paths default to `data/` and can be redirected with `TRAINING_DATA_DIR`. Hugging Face caches can be redirected with `HF_HOME`, and TorchVision caches with `TORCH_HOME`.

| Experiment | Dataset |
|---|---|
| Literal decomposition | `sklearn.datasets.load_digits`, stratified 80/20 split |
| ResNet18 and ViT-B/16 | CIFAR-100 through TorchVision |
| EfficientNet-V2-S | Tiny ImageNet-200 through the Kaggle mirror `xiataokang/tinyimagenettorch` |
| T5-small | SQuAD through Hugging Face Datasets |
| NanoGPT | WikiText-2 or SlimPajama-627B with GPT-2 tokenization |

Tiny ImageNet requires a locally configured Kaggle account. SlimPajama raw data must be downloaded separately and converted into `uint16` token binaries with `code/runtime_assets/nanogpt_slimpajama_prepare.py`. Credentials and access tokens must remain outside the repository.

## Numerical conventions

### Literal discrete decomposition

Each $(\beta_1,\beta_2)$ pair is trained independently for seeds 0, 1, and 2. Each trajectory stores the complete pre-update gradient for 1500 full-batch steps as a `(1500, 3466)` `float32` array; the analysis loads it as `float64`.

The first moment, second moment, and log-magnitude EMAs start from zero before step 0, retain the full history, and use bias correction $1-\beta^{k+1}$. Adam uses $\varepsilon=10^{-8}$ outside the square root. The log recurrence evaluates `log(max(abs(g), float64.tiny))`, while windows containing zero or non-finite gradients remain invalid.

After a 200-step burn-in, endpoint $k$ is valid only when all $w$ gradients from $k-w+1$ through $k$ are finite, nonzero, and have the same sign. All coordinates are used; there is no coordinate subsampling or magnitude threshold.

Component percentages are global absolute budgets. The additional diagnostics are

```text
F2    = 1 - sum(E^2) / sum((R-S)^2)
kappa = (sum(|L_disc|) + sum(|E|)) / sum(|R-S|)
```

Sign agreement excludes samples where either compared value is exactly zero.

### Training dynamics

`updates_norm` is the Euclidean norm of

```text
(parameters_after - parameters_before) / current_learning_rate
```

restricted to parameters owned directly by `Linear` and `Conv1d/2d/3d` modules. It includes biases from those modules and, for AdamW, the realized decoupled weight-decay contribution.

The analysis discards the first three epochs, smooths the retained sequence with an EMA initialized at its first value, and measures mean absolute first differences. The main smoothing window is 200; robustness analyses use windows 1, 10, 100, 200, and 500 and also evaluate interior second differences.

## Implementation details that affect exact reproduction

- EfficientNet uses TorchVision `efficientnet_v2_s`. Some historical input and figure filenames contain `EfficientNet-B0`; those filenames are retained only for compatibility with existing artifacts.
- EfficientNet's final 200-class classifier is created after sinusoidal initialization and therefore keeps PyTorch's default initialization.
- ViT-B/16 retains TorchVision's default 1000-output head; CIFAR-100 labels occupy indices 0--99.
- T5-small is loaded from its pretrained checkpoint before the module-wise sinusoidal initializer is applied.
- ResNet18 uses the backend TF32 default; the other five full-training entry points explicitly disable TF32.
- The NanoGPT learning-rate scheduler advances every 20 batches during the verbose first epoch and through the epoch-level callback.
- The exact historical NanoGPT revision and a fully frozen environment for every original run were not preserved. See [`ENVIRONMENT.md`](ENVIRONMENT.md).

## Output and security notes

Fresh runs are stored as JSON under `outputs/results/<experiment>/runs/`. Historical pickle inputs remain supported through `configs/training_sources_legacy.json`, but pickle files can execute code while loading and must come from a trusted source.

The tracked project files contain no datasets, results, credentials, local machine paths, or checkpoints. The checksum list in `FILE_MANIFEST.sha256` covers every project file except the manifest itself.
