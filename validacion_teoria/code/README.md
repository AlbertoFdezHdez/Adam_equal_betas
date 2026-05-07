# Gradient-probe and Adam decomposition diagnostics

This folder contains small-scale diagnostic scripts used to inspect Adam moments, gradient sign stability, and decomposition terms. The scripts are separate from the large training campaign under `code/`.

The default setup trains a small CNN on CIFAR-10 and stores per-batch gradients so later analyses can inspect gradient norms, signs, ratios, and moment decompositions without rerunning training.

## Install

```bash
pip install -r requirements.txt
```

## Estimate storage

```bash
python train_adam_gradient_grid.py --estimate-only --epochs 20 --batch-size 128 --grad-dtype float16
```

## Run the base Adam probe

```bash
python train_adam_gradient_grid.py \
  --mode single \
  --beta1 0.9 \
  --beta2 0.999 \
  --epochs 20 \
  --batch-size 128 \
  --lr 0.001 \
  --weight-decay 0.0 \
  --grad-dtype float16
```

Main run outputs are written to:

```text
../descargas/adam_gradient_cifar10/runs/<run_name>/
```

Typical files:

- `gradients.npy`: memory-mappable gradient matrix.
- `param_metadata.json`: parameter names, shapes, and offsets.
- `grad_stats_batch.csv`: per-batch gradient statistics.
- `metrics_batch.csv`: per-batch loss and accuracy.
- `metrics_epoch.csv`: per-epoch train/validation metrics.
- `summary.json`: configuration and final metrics.

## Main diagnostic scripts

Sign-stability analysis:

```bash
python analyze_gradient_sign_stability.py --windows 2 10 100
```

Gradient-ratio threshold tables:

```bash
python analyze_gradient_ratio_threshold_table.py \
  --run-dir "../descargas/adam_gradient_cifar10/runs/<run_name>" \
  --output-name <short_name>
```

Probe EMA decomposition:

```bash
python analyze_probe_ema_decomposition.py
python summarize_ema_decomposition_results.py
```

Multi-beta EMA replay:

```bash
python analyze_ema_decomposition_multi_beta.py --coord-sample-size 4096
python analyze_ema_decomposition_multi_beta_R.py
```

Direct `R` diagnostics:

```bash
python analyze_direct_R_decomposition_check.py
python analyze_direct_R_decomposition_v2.py
python analyze_direct_R_cancellation_diagnostics.py
python analyze_grouped_R_table.py
python validate_R_discretization_and_regime.py
python analyze_R_theorem_level_budget.py
```

Lightweight summaries from these scripts can be collected into `paper_artifacts/` with:

```bash
python ../../tools/collect_public_artifacts.py
```
