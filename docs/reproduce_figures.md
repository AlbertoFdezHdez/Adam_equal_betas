# Reproducing figures

## 5x5 beta-grid figures

The current public plotting script reads downloaded run JSON files from:

```text
descargas/results/
```

and writes figures/tables to:

```text
display/plots/
display/display_code/grid5_outputs/
```

Run:

```bash
cd display/display_code
python analyze_grid5_results.py
```

The script currently analyzes:

- `resnet18_cifar100`
- `nanogpt_wikitext`

Main outputs:

- `display/plots/resnet18_cifar100_grid5.pdf`
- `display/plots/nanogpt_wikitext_grid5.pdf`
- `display/display_code/grid5_outputs/omega1_grid5_report.txt`
- `display/display_code/grid5_outputs/hypothesis_tests.csv`

## If raw run JSON files are not present

The repository includes lightweight summaries and final PDF figures under `paper_artifacts/`. To regenerate the trajectory panels exactly, download the external raw run-output bundle and place it at:

```text
descargas/results/
```

The bundle should contain directories such as:

```text
descargas/results/resnet18_cifar100/runs/*.json
descargas/results/nanogpt_wikitext/runs/*.json
```

## Collecting public artifacts

After regenerating figures/tables, run:

```bash
python tools/collect_public_artifacts.py
```

This copies lightweight outputs into `paper_artifacts/`.
