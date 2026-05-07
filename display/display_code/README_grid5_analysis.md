# Grid 5x5 analysis

This folder keeps the new analysis code separate from the legacy notebooks.

Run from this directory:

```bash
python analyze_grid5_results.py
```

By default the script reads:

```text
../../descargas/results
```

and analyzes:

```text
resnet18_cifar100
nanogpt_wikitext
```

It writes plots to:

```text
../plots
```

and CSV tables to:

```text
grid5_outputs
```

Main outputs:

- `omega1_grid5_report.txt`: two readable 5x5 tables, one per experiment, with rate and p-value.
- `run_summary.csv`: one row per run with final train/validation losses, metrics and wall time.
- `updates_norm_oscillations.csv`: per-run oscillation values of `||R_k||` for several EMA windows.
- `updates_norm_oscillation_grid_mean.csv`: mean/std oscillation grids over seeds.
- `hypothesis_tests.csv`: row-min hypothesis-test rates and exact binomial p-values.
- `../plots/resnet18_cifar100_grid5.png`
- `../plots/nanogpt_wikitext_grid5.png`

The main hypothesis test uses `omega1_updates_norm` with EMA window 200, matching the logic used in the previous notebook.

The plots use the previous notebook style: LaTeX text, Computer Modern, shared bottom legend, blue `Train loss`, red `||R_k||`, soft standard-deviation bands, and titles containing `beta1`, `beta2`, and `omega`.

If a local machine does not have LaTeX installed, use:

```bash
python analyze_grid5_results.py --no-latex
```
