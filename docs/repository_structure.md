# Repository structure

The repository separates runnable code, visualization code, lightweight paper artifacts, and preserved legacy material.

## Training code

`code/` contains the current training implementation.

- `code/experiments/`: one Python module per experiment.
- `code/launchers/`: local and SLURM launch scripts.
- `code/utils/`: shared model, dataset, metric, callback, path, and result helpers.

## Visualization code

`display/display_code/` contains notebooks and scripts for reproducing figures and tables.

The most important script for the current 5x5 beta-grid results is:

```bash
python display/display_code/analyze_grid5_results.py
```

## Public artifacts

`paper_artifacts/` contains lightweight, GitHub-friendly outputs:

- PDF figures.
- CSV/TXT/TeX tables.
- Small diagnostic summaries.

It does not contain raw training curves, checkpoints, or gradient arrays when those files are too large for a normal GitHub repository.

## Legacy material

`legacy/` is preserved for provenance only. It should not be the first place users look. If a legacy result is relevant to the paper, export a lightweight summary into `paper_artifacts/`.
