# Manuscript-to-code map

Status vocabulary:

- **Executed**: the final analysis was run during the audit against available original artifacts.
- **Inspected**: formulas, configurations, or stored outputs were checked without rerunning training.
- **Server pending**: implementation is present but the costly training was not rerun.
- **Unavailable**: exact reproduction is blocked by missing provenance or files.

## Figures 1 and 2: controlled magnitude variation

| Stage | Implementation | Input | Intermediate/output | Status |
|---|---|---|---|---|
| Construct signal | `analysis/generate_controlled_figures.py` | Analytic 601-point signal; no files and no randomness | In-memory `float64` gradient array | Executed in final package check |
| Replay Adam | same script | Equal betas `0.9,0.95,0.99,0.999`; unequal pairs `(0.9,0.999),(0.9,0.95),(0.9,0.99),(0.99,0.999)` | Bias-corrected normalized updates, epsilon `1e-8` outside sqrt | Executed |
| Render figures | same script | Gradient and replay arrays | `figure_equal_betas.pdf`, `figure_unequal_betas.pdf`, summary JSON | Executed during the audit; generated files are not included |

## Table 1 and Appendix B: literal discrete decomposition

| Stage | Implementation | Input/configuration | Intermediate/output | Status |
|---|---|---|---|---|
| Train trajectory | `analysis/train_digits_gradient_trajectory.py` | scikit-learn Digits; stratified 80/20 split; standardized train statistics; MLP 64-32-32-10 Tanh; 1500 full-batch steps; lr `3e-4`; epsilon `1e-8`; seeds `0,1,2`; each beta pair trained separately | `gradients.npy` `(1500,3466)` float32 and `metadata.json` | One complete original trajectory inspected and analyzed; full 27-run retraining not run |
| Grid orchestration | `analysis/run_digits_grid.py` | `beta1,beta2 in {0.9,0.99,0.999}`, seeds `0,1,2` | 27 run directories | Inspected; server/local full run pending |
| Exact decomposition | `analysis/literal_decomposition.py` | Gradient array loaded as float64; burn-in 200; windows 6, 10, 100, 1000; all coordinates | Per-run `decomposition.json` with budgets, `F2`, cancellation, sign agreement, identity error | Executed on one full 1500-by-3466 original trajectory for w=6 and w=1000; exact identity error zero in that run |
| Aggregate | `analysis/summarize_literal_decomposition.py` | 27 decomposition JSON files | Main Table 1, w=6 control, reverse-order table, all-pair summaries | Inspected privately against the original aggregate CSVs; those files are not included |

The current implementation is the full-history log-magnitude version. It does not use the obsolete deterministic 1024-coordinate probe or a local-window EMA. Every beta pair has its own independently trained gradient trajectory.

## Table 2 and Appendix C: six training experiments

| Manuscript row | Training entry point | Actual dataset/model implementation | Grid and seeds | Analysis input | Status |
|---|---|---|---|---|---|
| NanoGPT / SlimPajama | `code/experiments/nanogpt_slimpajama.py` | External nanoGPT GPT-2-small configuration; SlimPajama `uint16` bins | 3x3; seed 1 | JSON run directory or historical pickle | Analysis executed; training server pending; exact nanoGPT commit unavailable |
| EfficientNet-V2-S / TinyImageNet | `code/experiments/efficientnet_tinyimagenet.py` | TorchVision EfficientNet-V2-S, Tiny ImageNet Kaggle mirror | 3x3; seeds 1,10,100 | same | Analysis executed; training server pending |
| T5 / SQuAD | `code/experiments/t5_squad.py` | pretrained `t5-small`, partial custom reinitialization; SQuAD | 3x3; seeds 1,10,100 | same | Analysis executed; training server pending |
| ResNet18 / CIFAR-100 | `code/experiments/resnet18_cifar100.py` | TorchVision ResNet18, 100-class head | 5x5 over `0.9,0.968,0.99,0.9968,0.999`; seeds 0,1,2 | JSON run directory | Analysis executed; training server pending |
| NanoGPT / WikiText-2 | `code/experiments/nanogpt_wikitext.py` | external nanoGPT GPT-2-small configuration; WikiText-2 | 3x3; seeds 1,10,100 | JSON or historical pickle | Analysis executed; training server pending; exact nanoGPT commit unavailable |
| ViT-B16 / CIFAR-100 | `code/experiments/vit_cifar100.py` | TorchVision ViT-B/16 with **default 1000-output head** | 3x3; seeds 1,10,100 | same | Analysis executed; training server pending |

`analysis/analyze_training_dynamics.py` removes the first three epochs, applies the exact EMA, computes per-run oscillation values, averages finite per-seed values for display, and calculates the diagonal-selection rates. It also generates the six Appendix-C grids. During the private audit, the script reproduced every displayed Table-2 value from the available original raw files, including the reported `NaN` cells and the one-seed SlimPajama exception. Neither the raw files nor the resulting tables are included in this code-only package.

## Table 4: smoothing and metric ablations

| Result | Implementation | Inputs | Output | Status |
|---|---|---|---|---|
| Windows 1,10,100,200,500 | `analysis/analyze_training_dynamics.py` | Same six raw result sets | `table4_ablation.csv` | Executed; all published rates reproduced |
| First-difference metric | same | EMA trajectories | Mean absolute first difference | Executed |
| Second-difference metric | same, default `--omega2-convention interior` | EMA trajectories | Mean absolute interior second difference | Executed |
| Boundary robustness | same, `--omega2-convention one-sided-boundaries` | Same trajectories | Alternate rates | Executed; all selection rates identical to the default on archived data |

## Artifact flow

```text
controlled formula -> generate_controlled_figures.py -> Figures 1-2

Digits training -> gradients.npy + metadata.json
               -> literal_decomposition.py -> per-run decomposition.json
               -> summarize_literal_decomposition.py -> Table 1 / Appendix B CSVs

six training entry points -> per-run JSON files
                          -> analyze_training_dynamics.py
                          -> Table 2 CSV + Appendix C PDFs + Table 4 CSV
```

No published numeric result lacks a code path. Exact end-to-end reproduction of the six costly training experiments remains subject to the implementation/provenance limitations described in the README and private audit.
