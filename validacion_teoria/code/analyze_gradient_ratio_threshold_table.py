from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    BASE_DIR
    / "descargas"
    / "adam_gradient_cifar10"
    / "runs"
    / "cifar10_tinycnn_adam_b1-0p9_b2-0p999_bs-128_grad-probe_seed-1234"
)
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "tables" / "gradient_ratio_threshold"


def rolling_sign_stable(signs: np.ndarray, window: int) -> np.ndarray:
    nonzero = signs != 0
    sign_change = signs[1:, :] != signs[:-1, :]

    nonzero_cum = np.empty((nonzero.shape[0] + 1, nonzero.shape[1]), dtype=np.int32)
    nonzero_cum[0, :] = 0
    np.cumsum(nonzero, axis=0, dtype=np.int32, out=nonzero_cum[1:, :])
    nonzero_count = nonzero_cum[window:, :] - nonzero_cum[:-window, :]

    change_cum = np.empty((sign_change.shape[0] + 1, sign_change.shape[1]), dtype=np.int32)
    change_cum[0, :] = 0
    np.cumsum(sign_change, axis=0, dtype=np.int32, out=change_cum[1:, :])
    change_count = change_cum[window - 1 :, :] - change_cum[: -(window - 1), :]

    return (nonzero_count == window) & (change_count == 0)


def compute_table(
    gradients_path: Path,
    windows: list[int],
    chunk_size: int,
    threshold: float,
    eps: float,
) -> list[dict[str, float | int]]:
    gradients = np.load(gradients_path, mmap_mode="r")
    if gradients.ndim != 2:
        raise ValueError(f"Expected gradients with shape (steps, parameters), got {gradients.shape}")

    steps, n_params = gradients.shape
    for window in windows:
        if window < 2 or window > steps:
            raise ValueError(f"Invalid window {window} for {steps} steps.")

    stable_counts = {window: 0 for window in windows}
    stable_and_small_counts = {window: 0 for window in windows}
    total_counts = {window: (steps - window + 1) * n_params for window in windows}

    for start in range(0, n_params, chunk_size):
        stop = min(start + chunk_size, n_params)
        chunk = np.asarray(gradients[:, start:stop])
        signs = np.sign(chunk).astype(np.int8, copy=False)
        abs_chunk = np.abs(chunk).astype(np.float32, copy=False)

        for window in windows:
            stable = rolling_sign_stable(signs, window)
            abs_windows = np.lib.stride_tricks.sliding_window_view(
                abs_chunk, window_shape=window, axis=0
            )
            g_min = np.min(abs_windows, axis=-1)
            g_max = np.max(abs_windows, axis=-1)
            valid_stable = stable & (g_min > eps)
            stable_counts[window] += int(valid_stable.sum())
            stable_and_small_counts[window] += int((valid_stable & (g_max < threshold * g_min)).sum())

        print(f"Processed coordinates {start}:{stop} / {n_params}", flush=True)

    rows: list[dict[str, float | int]] = []
    for window in windows:
        stable = stable_counts[window]
        small = stable_and_small_counts[window]
        total = total_counts[window]
        same_sign_percent = 100.0 * stable / total
        ratio_small_given_same_sign_percent = 100.0 * small / stable if stable else float("nan")
        product_percent = 100.0 * small / total
        rows.append(
            {
                "window": window,
                "total_cases": total,
                "same_sign_cases": stable,
                "same_sign_percent": same_sign_percent,
                "ratio_lt_sqrt2_given_same_sign_cases": small,
                "ratio_lt_sqrt2_given_same_sign_percent": ratio_small_given_same_sign_percent,
                "product_total_percent": product_percent,
            }
        )
    return rows


def write_outputs(output_dir: Path, rows: list[dict[str, float | int]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "ratio_threshold_table_long.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = [
        ("same_sign_percent", "% ventanas/coordenadas con signo constante"),
        (
            "ratio_lt_sqrt2_given_same_sign_percent",
            "% con max(|g|)/min(|g|) < sqrt(2) entre las de signo constante",
        ),
        ("product_total_percent", "% total que cumple ambas condiciones"),
    ]
    windows = [int(row["window"]) for row in rows]
    with (output_dir / "ratio_threshold_table_wide.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric"] + [f"w={window}" for window in windows])
        for key, label in metrics:
            writer.writerow([label] + [row[key] for row in rows])

    with (output_dir / "ratio_threshold_table.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each window, count same-sign gradient windows and, among them, "
            "how often max(|g|)/min(|g|) is below sqrt(2)."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--windows", type=int, nargs="+", default=list(range(2, 11)))
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=math.sqrt(2.0))
    parser.add_argument("--eps", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    gradients_path = run_dir / "gradients.npy"
    if not gradients_path.is_file():
        raise FileNotFoundError(gradients_path)

    rows = compute_table(
        gradients_path=gradients_path,
        windows=args.windows,
        chunk_size=args.chunk_size,
        threshold=args.threshold,
        eps=args.eps,
    )
    output_name = args.output_name or run_dir.name
    output_dir = args.results_dir.resolve() / output_name
    write_outputs(output_dir, rows)
    print(json.dumps(rows, indent=2), flush=True)
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
