# Result-to-code map

This map connects each experimental result to its training, analysis, and output path.

## Controlled magnitude variation

| Stage | Implementation | Input | Output |
|---|---|---|---|
| Construct signal | `analysis/generate_controlled_figures.py` | Deterministic 601-point, three-dimensional gradient sequence | In-memory `float64` gradients |
| Replay Adam | same script | Equal betas `0.9,0.95,0.99,0.999`; unequal pairs `(0.9,0.999)`, `(0.9,0.95)`, `(0.9,0.99)`, `(0.99,0.999)` | Bias-corrected normalized updates with epsilon `1e-8` outside the square root |
| Render | same script | Gradient and update arrays | `figure_equal_betas.pdf`, `figure_unequal_betas.pdf`, and `controlled_signal_summary.json` |

This experiment is deterministic and is included in the routine validation checks.

## Literal discrete decomposition

| Stage | Implementation | Input/configuration | Output |
|---|---|---|---|
| Train trajectories | `analysis/train_digits_gradient_trajectory.py` | Digits, stratified 80/20 split, standardized training statistics, 64-32-32-10 Tanh MLP, 1500 full-batch steps | `gradients.npy` `(1500,3466)` in `float32` plus metadata |
| Run grid | `analysis/run_digits_grid.py` | `beta1,beta2 in {0.9,0.99,0.999}`, seeds `0,1,2` | 27 independent run directories |
| Decompose | `analysis/literal_decomposition.py` | Gradients loaded as `float64`, burn-in 200, windows 6, 10, 100, 1000, all coordinates | Per-run `decomposition.json` with budgets, `F2`, cancellation, sign agreement, and identity error |
| Aggregate | `analysis/summarize_literal_decomposition.py` | The 27 decomposition files | Main component table, short-window control, reverse-order table, and all-pair summaries |

The implementation uses full-history log-magnitude EMAs. It does not use coordinate probing, coordinate subsampling, or local-window EMAs. Every beta pair is trained independently.

## Training-dynamics experiments

| Experiment | Entry point | Actual implementation | Grid and seeds |
|---|---|---|---|
| NanoGPT / SlimPajama | `code/experiments/nanogpt_slimpajama.py` | External NanoGPT GPT-2-small configuration; `uint16` SlimPajama token bins | 3x3; seed 1 |
| EfficientNet-V2-S / Tiny ImageNet | `code/experiments/efficientnet_tinyimagenet.py` | TorchVision EfficientNet-V2-S; Tiny ImageNet Kaggle mirror | 3x3; seeds 1, 10, 100 |
| T5-small / SQuAD | `code/experiments/t5_squad.py` | Pretrained `t5-small` followed by module-wise sinusoidal initialization | 3x3; seeds 1, 10, 100 |
| ResNet18 / CIFAR-100 | `code/experiments/resnet18_cifar100.py` | TorchVision ResNet18 with a 100-class head | 5x5; seeds 0, 1, 2 |
| NanoGPT / WikiText-2 | `code/experiments/nanogpt_wikitext.py` | External NanoGPT GPT-2-small configuration; WikiText-2 | 3x3; seeds 1, 10, 100 |
| ViT-B/16 / CIFAR-100 | `code/experiments/vit_cifar100.py` | TorchVision ViT-B/16 with its default 1000-output head | 3x3; seeds 1, 10, 100 |

Fresh runs write JSON files under `outputs/results/<experiment>/runs/`. `analysis/analyze_training_dynamics.py` removes the first three epochs, applies the configured EMA, computes per-run oscillation metrics, averages finite seed values, calculates diagonal-selection rates, and generates the experiment grids.

## Smoothing and metric robustness

| Result | Implementation | Output |
|---|---|---|
| EMA windows 1, 10, 100, 200, 500 | `analysis/analyze_training_dynamics.py` | `table4_ablation.csv` |
| Mean absolute first difference | same | Per-configuration `omega1` values and selection rates |
| Mean absolute interior second difference | same, default `--omega2-convention interior` | Per-configuration `omega2` values and selection rates |
| One-sided boundary comparison | same, `--omega2-convention one-sided-boundaries` | Alternate second-difference results |

## Data flow

```text
controlled signal -> generate_controlled_figures.py -> controlled figures

Digits training -> gradients.npy + metadata.json
               -> literal_decomposition.py -> decomposition.json
               -> summarize_literal_decomposition.py -> decomposition tables

six training entry points -> per-run JSON files
                          -> analyze_training_dynamics.py
                          -> smoothness tables + ablations + experiment grids
```

Generated artifacts are intentionally excluded from version control. Each one has a documented generation or acquisition path in `REPRODUCE.md` and `LARGE_FILES.md`.
