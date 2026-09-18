# Validation

This document records which parts of the repository have been executed and which require the complete datasets or GPU training grids.

## Executed checks

1. `python -m unittest discover -s tests -v` passes all five numerical tests. The tests cover bias-corrected EMAs, inclusive sign windows, zero filtering, the exact decomposition identity, component budgets, smoothing, first and second differences, and non-finite selection behavior.
2. `analysis/generate_controlled_figures.py` generates both controlled-experiment figures and the protocol-summary JSON.
3. A 12-step full-batch Digits trajectory has been trained and decomposed for windows 2 and 6. Both rows reconstruct $R=S+L^{\mathrm{disc}}+E$ with zero identity error in `float64`.
4. One complete `(1500,3466)` `float32` Digits trajectory for `(beta1,beta2)=(0.9,0.999)` has been analyzed over all coordinates. For `w=1000`, it produced `N=995852`, `S=41.949278%`, `L_disc=44.841481%`, `E=13.209241%`, `kappa=1.663556`, `F2=63.5175%`, sign agreement `99.62735%`, and identity error `0.0`.
5. The training-dynamics analyzer has been checked against the available historical result sets. It reproduces all stored update-smoothness and ablation values at their displayed precision.
6. Both supported second-difference conventions produce the same configuration-selection rates on the available six-task result sets.
7. All Python sources compile, all JSON configurations parse, and all shell launchers pass syntax validation.

## Source-level checks

- The discrete decomposition recurrence, zero initialization, bias correction, epsilon placement, dtype conversion, window indexing, component budgets, `F2`, cancellation, and sign-agreement definitions have been traced directly through the implementation.
- The beta grids, seeds, preprocessing, optimizer epsilon, batch sizes, sequence lengths, initialization order, scheduler cadence, and gradient-accumulation behavior of all six training entry points are recorded in `configs/experiment_protocols.json`.
- Every generated figure and table has an explicit production path in `RESULTS_TO_CODE.md`.
- The project tree contains no datasets, checkpoints, result arrays, logs, credentials, machine-specific paths, or generated binary outputs.

## Expensive runs not repeated as part of the lightweight checks

- The complete 27-run, 1500-step Digits grid.
- The six full model--dataset training grids.
- Dataset and pretrained-model downloads.
- The external NanoGPT checkout.

These runs are implemented and documented in `REPRODUCE.md`; they are omitted from routine validation because of their compute, storage, and network requirements.

## Reproduction boundaries

- Historical files whose names contain `EfficientNet-B0` were generated with EfficientNet-V2-S. Current code and documentation use the correct model name while retaining those paths for compatibility.
- EfficientNet's final 200-class classifier uses its default PyTorch initialization because it is constructed after the sinusoidal initializer is applied.
- ViT-B/16 uses the default 1000-output TorchVision head.
- `updates_norm` covers parameters owned directly by linear and convolutional modules and includes weight-decay effects.
- The exact historical NanoGPT revision and a completely frozen package environment are not available.
- ResNet18 leaves TF32 at the backend default.
- The NanoGPT scheduler cadence follows the implemented callback schedule rather than a conventional per-optimizer-step warm-up.
