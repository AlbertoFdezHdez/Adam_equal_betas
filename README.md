# Adam beta-grid experiments

This is a compact GitHub-ready release of the Adam beta-grid project. It contains the code needed to run the experiments, the plotting scripts used for the paper figures, and a small set of lightweight paper artifacts.

The main question studied here is how Adam's `(beta1, beta2)` choices affect training dynamics, especially the oscillation of the normalized Adam update direction `||R_k||`.

## What is included

```text
code/
  experiments/          Training entrypoints, one per model/dataset pair.
  launchers/            Local and SLURM launch helpers.
  utils/                Shared models, datasets, runtime, metrics, and result writers.

display/
  display_code/         Figure/table generation scripts and notebooks.

validacion_teoria/
  code/                 Gradient-probe and Adam decomposition diagnostics.

paper_artifacts/
  figures/              Selected PDF figures for the paper.
  tables/grid5/         5x5 beta-grid summary tables.
  tables/legacy_grid3/  Lightweight summaries derived from legacy 3x3 outputs.
  theory_diagnostics/   A small subset of compact diagnostic summaries.

docs/
  repository_structure.md
  reproduce_figures.md
  run_training.md
  github_release_plan.md

tools/
  collect_public_artifacts.py
```

This compact upload has roughly 100 files and is meant to be easy to inspect on GitHub.

## What is intentionally not included

Large or private files are excluded from this upload:

- Raw training trajectories under `descargas/results/`.
- Cluster logs under `descargas/logs/`.
- Checkpoints and model states.
- Large NumPy arrays from gradient-probe runs.
- Legacy PKL trajectory blobs.
- Private credentials such as `kaggle.json` and `hf_token.txt`.

If exact full reproduction is required, publish those heavy files separately as a GitHub Release asset, Zenodo archive, or institutional storage bundle.

Expected external raw-output layout:

```text
descargas/results/<experiment>/runs/*.json
legacy/for_visualizing/results/*.pkl
validacion_teoria/descargas/
```

## Quick start: inspect paper artifacts

The most useful files to inspect first are:

```text
paper_artifacts/figures/resnet18_cifar100_grid5.pdf
paper_artifacts/figures/nanogpt_wikitext_grid5.pdf
paper_artifacts/tables/grid5/omega1_grid5_report.txt
paper_artifacts/tables/grid5/hypothesis_tests.csv
```

The 5x5 tables report oscillation statistics for:

- `resnet18_cifar100`
- `nanogpt_wikitext`

over beta values:

```text
{0.900, 0.968, 0.990, 0.9968, 0.999}
```

and seeds:

```text
0, 1, 2
```

## Reproducing the 5x5 figures from raw run JSON files

Install display dependencies:

```bash
pip install -r requirements-display.txt
```

Place raw run JSON files under:

```text
descargas/results/
```

with paths such as:

```text
descargas/results/resnet18_cifar100/runs/*.json
descargas/results/nanogpt_wikitext/runs/*.json
```

Then run:

```bash
cd display/display_code
python analyze_grid5_results.py
```

Generated outputs go to:

```text
display/plots/
display/display_code/grid5_outputs/
```

To refresh the compact public artifacts:

```bash
python tools/collect_public_artifacts.py
```

## Running a training experiment

Install training dependencies:

```bash
pip install -r code/requirements-train.txt
```

Example:

```bash
cd code
python -m experiments.resnet18_cifar100 --beta1 0.9 --beta2 0.99 --seed 1
```

The full campaign consists of:

- `6` model/dataset experiments.
- `25` beta pairs in `{0.900, 0.968, 0.990, 0.9968, 0.999}^2`.
- `3` seeds: `0, 1, 2`.

Experiment entrypoints:

- `resnet18_cifar100`
- `efficientnet_tinyimagenet`
- `t5_squad`
- `vit_cifar100`
- `nanogpt_wikitext`
- `nanogpt_slimpajama`

See `docs/run_training.md` and `code/experiments/README.md` for launcher details.

## Notes on cluster-specific launchers

The SLURM launchers were written for the original cluster environment. Before reuse, check:

- partition and node names,
- Python environment path,
- dataset paths,
- scratch/output paths,
- NanoGPT repository location.

## License

Add the intended license before making the repository public.
