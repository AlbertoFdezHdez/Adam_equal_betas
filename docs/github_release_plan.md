# GitHub release plan

Recommended release contents:

1. Source code under `code/`, `display/`, `validacion_teoria/code/`, and `tools/`.
2. Lightweight paper artifacts under `paper_artifacts/`.
3. Documentation under `README.md` and `docs/`.
4. No private tokens, checkpoints, local logs, or large raw arrays.

Recommended external artifact bundle:

1. Raw training JSON traces from `descargas/results/`.
2. Legacy PKL trajectory blobs, if exact legacy reproduction is required.
3. Large gradient arrays from `validacion_teoria/descargas/`.
4. Any model checkpoints needed for additional checks.

A practical workflow is:

```bash
python display/display_code/analyze_grid5_results.py
python tools/collect_public_artifacts.py
```

Then commit the source files and `paper_artifacts/`, and upload the heavy raw-output bundle to a GitHub Release or Zenodo.
