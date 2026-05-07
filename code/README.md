# Training Layout

This folder contains the runnable training code.

## Structure

- `experiments/`: one entrypoint per experiment.
- `launchers/`: local and SLURM launch helpers.
- `utils/`: shared training runtime, models, datasets, paths, and result utilities.

## Running a single experiment

From this `code/` directory:

```bash
python -m experiments.resnet18_cifar100 --beta1 0.9 --beta2 0.99 --seed 1
```

## Results and logs

- Logs: `../descargas/logs/<experiment>/`
- Per-run JSON: `../descargas/results/<experiment>/runs/`
- Merged PKL/CSV: `../descargas/results/<experiment>/merged/`

## Merging runs

```bash
python launchers/merge_results.py resnet18_cifar100
```
