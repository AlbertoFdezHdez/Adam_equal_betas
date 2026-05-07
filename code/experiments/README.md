# Experiments

This directory contains one training entrypoint per model/dataset experiment.

Current experiment modules:

- `resnet18_cifar100`
- `efficientnet_tinyimagenet`
- `t5_squad`
- `vit_cifar100`
- `nanogpt_wikitext`
- `nanogpt_slimpajama`

## Full campaign

The complete campaign is:

- `6` experiments.
- `25` beta pairs in `{0.900, 0.968, 0.990, 0.9968, 0.999}^2`.
- `3` seeds: `0, 1, 2`.

Total: `6 x 25 x 3 = 450` training runs.

## Recommended execution order

### 1. Smoke validation

Run the six short validation jobs before launching the full campaign:

```bash
PARTITION=q5 NODELIST=r4n02 bash code/launchers/submit_smoke_validation_6_slurm.sh
```

### 2. First real batch

Run the six experiments with `beta1 = beta2 = 0.9` and `seed = 0`, using at most three GPUs:

```bash
PARTITION=q5 NODELIST=r4n02 bash code/launchers/submit_stage1_six_experiments_betaeq0p9_seed0_slurm.sh
```

The launcher writes submitted job ids to:

```text
descargas/logs/stage1_six_experiments_betaeq0p9_seed0/submitted_jobs.tsv
```

### 3. Remaining seed-0 beta pairs for one experiment

For a single experiment, run the remaining `24` beta pairs for `seed = 0` in three sequential chains:

```bash
EXPERIMENT_MODULE=resnet18_cifar100 PARTITION=q5 NODELIST=r4n02 bash code/launchers/submit_single_experiment_grid24_seed0_3x8_slurm.sh
```

SlimPajama/NanoGPT requires external dataset paths:

```bash
EXPERIMENT_MODULE=nanogpt_slimpajama \
PARTITION=q5 NODELIST=r4n02 \
NANOGPT_ROOT=/scratch1/hernanal/nanoGPT \
NANOGPT_DATA_DIR=/scratch1/hernanal/nanoGPT/data/slimpajama \
bash code/launchers/submit_single_experiment_grid24_seed0_3x8_slurm.sh
```

Submitted job ids are written to:

```text
descargas/logs/grid24_seed0/<EXPERIMENT_MODULE>/submitted_jobs.tsv
```

## Full 75-run grids

For the full ResNet18/CIFAR-100 grid:

```bash
PARTITION=q5 NODELIST=r4n02 bash code/launchers/submit_resnet_grid75_3chains_slurm.sh
```

For the full NanoGPT/WikiText grid:

```bash
PARTITION=q5 NODELIST=r4n02 bash code/launchers/submit_nanogpt_wikitext_grid75_3chains_slurm.sh
```

Each launcher submits three long jobs, one per seed. Each job runs its `25` beta pairs sequentially on a single GPU.

## Timing fields

Every run JSON includes:

- `training_wall_time_sec`
- `wall_time_total_sec`

These fields are also included in merged CSV summaries:

```bash
python code/launchers/merge_results.py <experiment>
```
