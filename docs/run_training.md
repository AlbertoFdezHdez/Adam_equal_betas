# Running training experiments

Install training dependencies:

```bash
pip install -r code/requirements-train.txt
```

Run a single experiment from `code/`:

```bash
cd code
python -m experiments.resnet18_cifar100 --beta1 0.9 --beta2 0.99 --seed 1
```

Outputs are written to:

```text
descargas/results/<experiment>/runs/
descargas/logs/<experiment>/
```

## Full beta grid

The beta grid is:

```text
{0.900, 0.968, 0.990, 0.9968, 0.999}^2
```

with seeds:

```text
0, 1, 2
```

This gives `75` runs per experiment.

## SLURM launchers

The project includes SLURM launchers in `code/launchers/`. They were written for the local cluster used during the experiments, so check partition, node list, Python environment, and scratch paths before reuse.

Examples:

```bash
PARTITION=q5 NODELIST=r4n02 bash code/launchers/submit_resnet_grid75_3chains_slurm.sh
PARTITION=q5 NODELIST=r4n02 bash code/launchers/submit_nanogpt_wikitext_grid75_3chains_slurm.sh
```

Each launcher uses three long chains rather than submitting all runs independently, so the queue remains readable and at most three GPUs are used.
