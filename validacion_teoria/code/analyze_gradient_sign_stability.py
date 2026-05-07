from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    BASE_DIR
    / "descargas"
    / "adam_gradient_cifar10"
    / "runs"
    / "cifar10_tinycnn_adam_b1-0p9_b2-0p999_seed-1234"
)
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "sign_stability"


def rolling_sum_bool(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling sums along time axis for a boolean matrix with shape (T, C)."""
    cumulative = np.empty((values.shape[0] + 1, values.shape[1]), dtype=np.int32)
    cumulative[0, :] = 0
    np.cumsum(values, axis=0, dtype=np.int32, out=cumulative[1:, :])
    return cumulative[window:, :] - cumulative[:-window, :]


def compute_sign_stability(
    gradients_path: Path,
    windows: list[int],
    chunk_size: int,
) -> dict[int, np.ndarray]:
    """Return mean sign-stability over coordinates for each window.

    For a window [j, j+w-1] and coordinate i, the indicator is 1 iff every
    gradient sign in the window is equal and non-zero. The returned arrays have
    length T-w+1 and contain the mean of this indicator over coordinates.
    """
    gradients = np.load(gradients_path, mmap_mode="r")
    if gradients.ndim != 2:
        raise ValueError(f"Expected gradients with shape (steps, parameters), got {gradients.shape}")

    steps, n_params = gradients.shape
    for window in windows:
        if window < 2:
            raise ValueError("Windows must be at least 2 to detect sign changes.")
        if window > steps:
            raise ValueError(f"Window {window} is larger than the number of steps {steps}.")

    stable_counts = {window: np.zeros(steps - window + 1, dtype=np.int64) for window in windows}

    for start in range(0, n_params, chunk_size):
        stop = min(start + chunk_size, n_params)
        signs = np.sign(gradients[:, start:stop]).astype(np.int8, copy=False)
        nonzero = signs != 0
        sign_change = signs[1:, :] != signs[:-1, :]

        for window in windows:
            nonzero_count = rolling_sum_bool(nonzero, window)
            change_count = rolling_sum_bool(sign_change, window - 1)
            stable = (nonzero_count == window) & (change_count == 0)
            stable_counts[window] += stable.sum(axis=1, dtype=np.int64)

        print(f"Processed coordinates {start}:{stop} / {n_params}", flush=True)

    return {window: stable_counts[window] / n_params for window in windows}


def write_series_csv(path: Path, series_by_window: dict[int, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_len = max(len(values) for values in series_by_window.values())
    fieldnames = ["window_start_step"]
    for window in series_by_window:
        fieldnames.append(f"w{window}_stable_fraction")
        fieldnames.append(f"w{window}_stable_percent")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(max_len):
            row: dict[str, float | int | str] = {"window_start_step": step}
            for window, values in series_by_window.items():
                if step < len(values):
                    row[f"w{window}_stable_fraction"] = float(values[step])
                    row[f"w{window}_stable_percent"] = float(100.0 * values[step])
                else:
                    row[f"w{window}_stable_fraction"] = ""
                    row[f"w{window}_stable_percent"] = ""
            writer.writerow(row)


def summarize(series_by_window: dict[int, np.ndarray]) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for window, values in series_by_window.items():
        rows.append(
            {
                "window": window,
                "n_time_windows": int(len(values)),
                "temporal_mean_percent": float(100.0 * np.mean(values)),
                "temporal_median_percent": float(100.0 * np.median(values)),
                "first_window_percent": float(100.0 * values[0]),
                "last_window_percent": float(100.0 * values[-1]),
                "min_percent": float(100.0 * np.min(values)),
                "max_percent": float(100.0 * np.max(values)),
            }
        )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_series(
    plots_dir: Path,
    series_by_window: dict[int, np.ndarray],
    run_name: str,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(11, 6))
    for window, values in series_by_window.items():
        ax.plot(np.arange(len(values)), 100.0 * values, linewidth=1.4, label=f"w={window}")
    ax.set_title(f"Sign stability over time: {run_name}")
    ax.set_xlabel("Window start batch j")
    ax.set_ylabel("% coordinates with constant non-zero sign in [j, j+w-1]")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "sign_stability_w2_w10_w100.png", dpi=180)
    fig.savefig(plots_dir / "sign_stability_w2_w10_w100.pdf")
    plt.close(fig)

    for window, values in series_by_window.items():
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(np.arange(len(values)), 100.0 * values, color="#2563eb", linewidth=1.2)
        ax.axhline(100.0 * float(np.mean(values)), color="#dc2626", linestyle="--", linewidth=1.1)
        ax.set_title(f"Sign stability, w={window}: {run_name}")
        ax.set_xlabel("Window start batch j")
        ax.set_ylabel("% stable coordinates")
        ax.set_ylim(0, 100)
        fig.tight_layout()
        fig.savefig(plots_dir / f"sign_stability_w{window}.png", dpi=180)
        fig.savefig(plots_dir / f"sign_stability_w{window}.pdf")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-coordinate gradient sign stability.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--windows", type=int, nargs="+", default=[2, 10, 100])
    parser.add_argument("--chunk-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gradients_path = args.run_dir / "gradients.npy"
    if not gradients_path.is_file():
        raise FileNotFoundError(gradients_path)

    run_name = args.run_dir.name
    output_dir = args.results_dir / run_name
    plots_dir = output_dir / "plots"

    series_by_window = compute_sign_stability(
        gradients_path=gradients_path,
        windows=args.windows,
        chunk_size=args.chunk_size,
    )
    summary_rows = summarize(series_by_window)

    write_series_csv(output_dir / "sign_stability_timeseries.csv", series_by_window)
    write_summary_csv(output_dir / "sign_stability_summary.csv", summary_rows)
    with (output_dir / "sign_stability_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2, sort_keys=True)
    plot_series(plots_dir, series_by_window, run_name)

    print(json.dumps(summary_rows, indent=2), flush=True)
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
