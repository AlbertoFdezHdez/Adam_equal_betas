# Adam beta-grid experiments

This repository contains training code, analysis scripts, and lightweight paper artifacts for a set of Adam optimizer experiments over a beta-grid.

The central experimental question is how the choice of `(beta1, beta2)` affects training dynamics, especially the oscillation of the normalized Adam update direction `||R_k||`. The public layout is designed so that readers can either inspect the paper artifacts directly or reproduce the figures from downloaded run outputs.

## Repository layout

```text
code/
  experiments/          Training entrypoints, one per model/dataset pair.
  launchers/            Local and SLURM launch helpers.
  utils/                Shared models, datasets, runtime, metrics, and result writers.

display/
  display_code/         Figure/table generation code.
  plots/                Generated local plots.

paper_artifacts/
  figures/              Curated PDF figures for the paper.
  tables/               Lightweight CSV/TXT/TeX tables used in the paper.
  theory_diagnostics/   Lightweight diagnostic summaries from the theory validation runs.

validacion_teoria/
  code/                 Gradient-probe and Adam decomposition diagnostics.
  results/              Local diagnostic outputs. Heavy arrays are not meant for GitHub.

legacy/
  for_training/         Original training snapshot, preserved for provenance.
  for_visualizing/      Original visualization snapshot, preserved for provenance.
```

## Quick start: reproduce paper tables and figures from existing outputs

Install plotting dependencies:

```bash
pip install -r requirements-display.txt
```

Regenerate the 5x5 beta-grid analysis for the downloaded ResNet/CIFAR-100 and NanoGPT/WikiText runs:

```bash
cd display/display_code
python analyze_grid5_results.py
```

Outputs are written to:

```text
display/plots/
display/display_code/grid5_outputs/
```

To collect the lightweight public artifacts into `paper_artifacts/`:

```bash
python tools/collect_public_artifacts.py
```

## Training experiments

The main training campaign uses:

- `6` model/dataset experiments.
- `25` beta pairs in `{0.900, 0.968, 0.990, 0.9968, 0.999}^2`.
- `3` seeds: `0, 1, 2`.

The six experiment entrypoints are:

- `resnet18_cifar100`
- `efficientnet_tinyimagenet`
- `t5_squad`
- `vit_cifar100`
- `nanogpt_wikitext`
- `nanogpt_slimpajama`

Example single run:

```bash
cd code
python -m experiments.resnet18_cifar100 --beta1 0.9 --beta2 0.99 --seed 1
```

See `code/README.md` and `code/experiments/README.md` for the SLURM launchers.

## Data and artifact policy

GitHub is not a good place for multi-GB raw training traces, checkpoints, or gradient arrays. This repository therefore keeps:

- Source code.
- Lightweight CSV/TXT/TeX summaries.
- Curated PDF figures.
- Scripts that regenerate figures when raw run outputs are available.

The following are intentionally excluded by `.gitignore`:

- Private tokens such as `kaggle.json` and `hf_token.txt`.
- Local logs.
- Checkpoints.
- Large NumPy arrays.
- Legacy PKL trajectory blobs.

For a fully reproducible public release, upload raw run outputs as a separate release asset or Zenodo archive, then document the download location in `paper_artifacts/README.md`.

## Legacy provenance

The `legacy/` folder is preserved to keep the original handoff intact. Public-facing summaries derived from it are copied into `paper_artifacts/` so users do not need to inspect the legacy tree for normal figure/table reproduction.

## License

Add the intended license before making the repository public.
