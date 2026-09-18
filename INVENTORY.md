# File inventory and status

All files listed here are current unless explicitly marked as compatibility-only. Obsolete notebooks, probe-coordinate decompositions, logs, and intermediate exploratory scripts are intentionally not tracked.

## Top level

| File | Role |
|---|---|
| `README.md` | Project overview, setup, datasets, conventions, fidelity notes, and verification scope. |
| `REPRODUCE.md` | Ordered quick and full reproduction commands. |
| `RESULTS_TO_CODE.md` | Result-by-result implementation and output map. |
| `LARGE_FILES.md` | Excluded artifact schemas, locations, consumers, sizes, and regeneration. |
| `ENVIRONMENT.md` | Honest environment provenance and missing version information. |
| `requirements.txt` | Lightweight analysis and Digits dependencies. |
| `requirements-training.txt` | Additional full-training dependencies. |
| `.gitignore` | Excludes credentials, data, raw results, caches, checkpoints, and generated outputs. |

## Analysis

| File | Role |
|---|---|
| `analysis/__init__.py` | Marks the analysis module. |
| `analysis/generate_controlled_figures.py` | Current deterministic Figures 1--2 implementation. |
| `analysis/train_digits_gradient_trajectory.py` | Current independent Digits training/gradient capture implementation. |
| `analysis/literal_decomposition.py` | Current full-history discrete decomposition; replaces all earlier local-window and 1024-probe variants. |
| `analysis/run_digits_grid.py` | Current 3x3 beta by 3-seed Digits orchestrator. |
| `analysis/summarize_literal_decomposition.py` | Current Table-1 and Appendix-B aggregation. |
| `analysis/analyze_training_dynamics.py` | Current unified Table-2, Appendix-C, and Table-4 analyzer. Replaces the historical analyzer that mishandled top-level epoch metadata. |

## Training entry points

| File | Role |
|---|---|
| `code/experiments/resnet18_cifar100.py` | ResNet18/CIFAR-100 5x5 experiment. |
| `code/experiments/efficientnet_tinyimagenet.py` | EfficientNet-V2-S/TinyImageNet experiment; historical result filenames retain the earlier `EfficientNet-B0` label for compatibility. |
| `code/experiments/t5_squad.py` | T5/SQuAD experiment. |
| `code/experiments/nanogpt_wikitext.py` | NanoGPT/WikiText-2 experiment. |
| `code/experiments/nanogpt_slimpajama.py` | NanoGPT/SlimPajama experiment. |
| `code/experiments/vit_cifar100.py` | ViT-B/16/CIFAR-100 experiment; actual head has 1000 outputs. |
| `code/experiments/__init__.py` | Marks the experiment module. |

## Shared code

| File | Role |
|---|---|
| `code/utils/model.py` | Shared training/evaluation loop and callback integration. |
| `code/utils/callbacks.py` | Learning-rate scheduler and training callbacks. |
| `code/utils/io.py` | JSON/pickle I/O and parameter/gradient flattening. |
| `code/utils/results.py` | Per-run JSON naming/saving and optional legacy merge support. |
| `code/utils/metrics.py` | Accuracy metric utilities. |
| `code/utils/paths.py` | Repository-relative, environment-overridable output paths. |
| `code/utils/runtime.py` | External nanoGPT discovery and import checks. |
| `code/utils/__init__.py` | Public utility exports. |
| `code/utils/datasets/cifar.py` | CIFAR-10/100 and ViT-specific CIFAR-100 transforms/loaders. |
| `code/utils/datasets/tiny_imagenet.py` | Tiny ImageNet Kaggle acquisition and transforms. |
| `code/utils/datasets/squad.py` | SQuAD text formatting, tokenization, and loaders. |
| `code/utils/datasets/wikitext.py` | WikiText-2 filtering, tokenization, block construction, and loaders. |
| `code/utils/datasets/slimpajama.py` | NanoGPT binary dataset reader. |
| `code/utils/datasets/utils.py` | Dataset/cache paths and reduced-loader helpers. |
| `code/utils/datasets/__init__.py` | Dataset loader exports. |
| `code/utils/models/resnet.py` | ResNet18 constructor. |
| `code/utils/models/efficientnet.py` | Actual EfficientNet-V2-S constructor used by archived runs. |
| `code/utils/models/vit.py` | Actual historical ViT loader plus an unused 100-class alternative. |
| `code/utils/models/t5.py` | Pretrained T5-small loader. |
| `code/utils/models/sinusoidal.py` | Custom sinusoidal initialization. |
| `code/utils/models/mlp.py` | Small shared MLP utility; retained because it belongs to the active utility package, though the Digits script defines its exact model locally. |
| `code/utils/models/__init__.py` | Model constructor exports. |

## Launch and preparation

| File | Role |
|---|---|
| `code/launchers/launch_experiment.sh` | Portable single-run launcher. |
| `code/launchers/launch_grid_slurm.sh` | Cartesian-product SLURM submission launcher. |
| `code/launchers/slurm_array_runner.sh` | Maps a SLURM array index to beta pair and seed. |
| `code/launchers/merge_results.py` | Optional merger from fresh JSON runs to the historical pickle/CSV format; analysis does not require merging. |
| `code/launchers/__init__.py` | Marks the launcher helper module. |
| `code/runtime_assets/nanogpt_slimpajama_prepare.py` | Tokenizes previously downloaded SlimPajama chunk files into NanoGPT binary files. No tokens or credentials are bundled. |

## Configurations and tests

| Path | Role |
|---|---|
| `configs/experiment_protocols.json` | Exact protocol, including actual constructors and observable definitions. |
| `configs/training_sources_json.json` | Input map for newly generated JSON runs. |
| `configs/training_sources_legacy.json` | Compatibility-only input map for trusted archived pickle results plus ResNet JSON. |
| `tests/test_numerics.py` | Lightweight synthetic tests for EMA, masks, exact decomposition identity, differences, and non-finite selection. |

## Excluded development artifacts

The following development artifacts are intentionally excluded: old common-gradient replay experiments; deterministic 1024-coordinate calibration scripts; local-window rather than full-history log-EMA decompositions; exploratory MLP/CNN sweeps; notebooks; duplicate display scripts; stale tables; generated CSV/PDF files; raw results; downloaded datasets; credentials; server logs; and caches.
