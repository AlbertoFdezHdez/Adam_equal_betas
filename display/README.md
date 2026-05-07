# Display and figure generation

This folder contains public-facing plotting code and generated figures.

## Structure

- `display_code/`: notebooks and Python scripts for figures and tables.
- `plots/`: locally generated figures.

## Current 5x5 beta-grid analysis

Run:

```bash
cd display/display_code
python analyze_grid5_results.py
```

Main outputs:

- `../plots/resnet18_cifar100_grid5.pdf`
- `../plots/nanogpt_wikitext_grid5.pdf`
- `grid5_outputs/omega1_grid5_report.txt`
- `grid5_outputs/hypothesis_tests.csv`

The script expects raw run JSON files under:

```text
../../descargas/results/
```

If those files are unavailable, use the precomputed lightweight outputs in `../paper_artifacts/`.
