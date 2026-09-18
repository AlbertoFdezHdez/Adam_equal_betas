"""Aggregate independently trained Digits decompositions across seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BETAS = (0.9, 0.99, 0.999)
SEEDS = (0, 1, 2)
METRICS = ("n", "S_pct", "L_pct", "E_pct", "kappa", "F2", "sign_agreement")


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def run_dir(root: Path, beta1: float, beta2: float, seed: int) -> Path:
    return root / f"b1-{tag(beta1)}_b2-{tag(beta2)}_seed-{seed}"


def matched_window(beta1: float, beta2: float) -> int:
    return max(round(beta1 / (1.0 - beta1)), round(beta2 / (1.0 - beta2))) + 1


def load_rows(root: Path) -> dict[tuple[float, float, int, int], dict]:
    output = {}
    for beta1 in BETAS:
        for beta2 in BETAS:
            for seed in SEEDS:
                directory = run_dir(root, beta1, beta2, seed)
                metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                if metadata["betas"] != [beta1, beta2] or metadata["seed"] != seed:
                    raise RuntimeError(f"Metadata mismatch in {directory}")
                analysis = json.loads((directory / "analysis.json").read_text(encoding="utf-8"))
                if analysis["coordinates"] != "all":
                    raise RuntimeError(f"Paper aggregation requires all coordinates: {directory}")
                for row in analysis["rows"]:
                    output[(beta1, beta2, seed, int(row["window"]))] = {
                        **row,
                        "train_accuracy": metadata["train_accuracy"],
                        "test_accuracy": metadata["test_accuracy"],
                    }
    return output


def summarize(raw: dict, selector) -> list[dict]:
    grouped: defaultdict[tuple[float, float], list[dict]] = defaultdict(list)
    for (beta1, beta2, seed, window), row in raw.items():
        if window == selector(beta1, beta2):
            grouped[(beta1, beta2)].append(row)
    output = []
    for (beta1, beta2), rows in sorted(grouped.items()):
        if len(rows) != len(SEEDS):
            raise RuntimeError(f"Expected three seeds for {(beta1, beta2)}")
        summary = {"beta1": beta1, "beta2": beta2, "window": selector(beta1, beta2)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[f"{metric}_mean"] = float(np.nanmean(values))
            summary[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
        for metric in ("train_accuracy", "test_accuracy"):
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1))
        output.append(summary)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paper_row(row: dict) -> dict:
    diagonal = row["beta1"] == row["beta2"]
    return {
        "beta1": row["beta1"],
        "beta2": row["beta2"],
        "window": row["window"],
        "N_per_seed": row["n_mean"],
        "S_pct": row["S_pct_mean"],
        "L_disc_pct": row["L_pct_mean"],
        "E_pct": row["E_pct_mean"],
        "kappa": row["kappa_mean"],
        "F2_pct": "" if diagonal else 100.0 * row["F2_mean"],
        "sign_agreement_pct": "" if diagonal else 100.0 * row["sign_agreement_mean"],
        "test_accuracy_pct": 100.0 * row["test_accuracy_mean"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs/digits"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/decomposition_tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_rows(args.input_root)
    fixed_1000 = summarize(raw, lambda _b1, _b2: 1000)
    fixed_6 = summarize(raw, lambda _b1, _b2: 6)
    matched = summarize(raw, matched_window)
    write_csv(args.output_dir / "all_pairs_w1000.csv", fixed_1000)
    write_csv(args.output_dir / "all_pairs_w6.csv", fixed_6)
    write_csv(args.output_dir / "all_pairs_matched_window.csv", matched)
    write_csv(
        args.output_dir / "table1_main.csv",
        [paper_row(row) for row in fixed_1000 if row["beta1"] <= row["beta2"]],
    )
    write_csv(
        args.output_dir / "appendix_short_window.csv",
        [paper_row(row) for row in fixed_6 if row["beta1"] < row["beta2"]],
    )
    write_csv(
        args.output_dir / "appendix_reverse_order.csv",
        [paper_row(row) for row in matched if row["beta1"] > row["beta2"]],
    )
    print(f"Wrote decomposition summaries to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
