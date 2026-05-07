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
    / "cifar10_tinycnn_adam_b1-0p9_b2-0p999_bs-128_grad-probe_seed-1234"
)
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "window_gradient_ratios"


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


def compute_ratio_chunk(
    chunk: np.ndarray,
    window: int,
    ratio_mode: str,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ratio matrix and valid mask with shape (T-window+1, C_chunk)."""
    windows = np.lib.stride_tricks.sliding_window_view(chunk, window_shape=window, axis=0)

    if ratio_mode == "signed":
        g_max = np.max(windows, axis=-1).astype(np.float32)
        g_min = np.min(windows, axis=-1).astype(np.float32)
        valid = np.abs(g_min) > eps
        ratio = np.full(g_max.shape, np.nan, dtype=np.float32)
        ratio[valid] = g_max[valid] / g_min[valid]
        return ratio, valid

    abs_windows = np.abs(windows).astype(np.float32, copy=False)
    g_max = np.max(abs_windows, axis=-1)
    g_min = np.min(abs_windows, axis=-1)
    valid = g_min > eps

    if ratio_mode == "same-sign-abs":
        signs = np.sign(chunk).astype(np.int8, copy=False)
        valid &= rolling_sign_stable(signs, window)

    ratio = np.full(g_max.shape, np.nan, dtype=np.float32)
    ratio[valid] = g_max[valid] / g_min[valid]
    return ratio, valid


def write_percentiles_csv(
    path: Path,
    starts: np.ndarray,
    percentiles: dict[int, np.ndarray],
    valid_counts: np.ndarray,
    total_params: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["window_start_step", "valid_count", "valid_fraction"]
    fieldnames.extend([f"p{q}" for q in sorted(percentiles)])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, start in enumerate(starts):
            row: dict[str, float | int] = {
                "window_start_step": int(start),
                "valid_count": int(valid_counts[idx]),
                "valid_fraction": float(valid_counts[idx] / total_params),
            }
            for q in sorted(percentiles):
                row[f"p{q}"] = float(percentiles[q][idx])
            writer.writerow(row)


def plot_percentiles(
    plots_dir: Path,
    starts: np.ndarray,
    percentiles: dict[int, np.ndarray],
    valid_counts: np.ndarray,
    total_params: int,
    run_name: str,
    window: int,
    ratio_mode: str,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {5: "#7c3aed", 25: "#2563eb", 50: "#111827", 75: "#f59e0b", 95: "#dc2626"}
    for q in sorted(percentiles):
        ax.plot(starts, percentiles[q], linewidth=1.25, color=colors.get(q), label=f"p{q}")
    ax.set_title(f"Gradient window ratios, w={window}, mode={ratio_mode}: {run_name}")
    ax.set_xlabel("Window start batch j")
    ax.set_ylabel("Gradient ratio")
    ax.set_yscale("log")
    ax.legend(ncol=5)
    fig.tight_layout()
    fig.savefig(plots_dir / f"gradient_ratio_percentiles_w{window}_{ratio_mode}.png", dpi=180)
    fig.savefig(plots_dir / f"gradient_ratio_percentiles_w{window}_{ratio_mode}.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(starts, 100.0 * valid_counts / total_params, color="#059669", linewidth=1.2)
    ax.set_title(f"Valid coordinates, w={window}, mode={ratio_mode}: {run_name}")
    ax.set_xlabel("Window start batch j")
    ax.set_ylabel("% valid coordinates")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(plots_dir / f"gradient_ratio_valid_fraction_w{window}_{ratio_mode}.png", dpi=180)
    fig.savefig(plots_dir / f"gradient_ratio_valid_fraction_w{window}_{ratio_mode}.pdf")
    plt.close(fig)


def analyze_window(
    gradients_path: Path,
    output_dir: Path,
    window: int,
    percentiles_to_compute: list[int],
    ratio_mode: str,
    chunk_size: int,
    eps: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gradients = np.load(gradients_path, mmap_mode="r")
    if gradients.ndim != 2:
        raise ValueError(f"Expected gradients with shape (steps, parameters), got {gradients.shape}")
    steps, n_params = gradients.shape
    if window < 2 or window > steps:
        raise ValueError(f"Invalid window {window} for {steps} steps.")

    n_windows = steps - window + 1
    ratios_path = output_dir / f"ratios_w{window}_{ratio_mode}.npy"
    ratios = np.lib.format.open_memmap(
        ratios_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_windows, n_params),
    )
    valid_counts = np.zeros(n_windows, dtype=np.int64)

    for start in range(0, n_params, chunk_size):
        stop = min(start + chunk_size, n_params)
        ratio_chunk, valid_chunk = compute_ratio_chunk(
            chunk=np.asarray(gradients[:, start:stop]),
            window=window,
            ratio_mode=ratio_mode,
            eps=eps,
        )
        ratios[:, start:stop] = ratio_chunk
        valid_counts += valid_chunk.sum(axis=1, dtype=np.int64)
        print(f"Processed coordinates {start}:{stop} / {n_params}", flush=True)
    ratios.flush()

    percentile_series = {q: np.empty(n_windows, dtype=np.float32) for q in percentiles_to_compute}
    for row_idx in range(n_windows):
        row = ratios[row_idx, :]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            values = np.full(len(percentiles_to_compute), np.nan, dtype=np.float32)
        else:
            values = np.percentile(finite, percentiles_to_compute).astype(np.float32)
        for q, value in zip(percentiles_to_compute, values):
            percentile_series[q][row_idx] = value

    starts = np.arange(n_windows)
    write_percentiles_csv(
        output_dir / f"gradient_ratio_percentiles_w{window}_{ratio_mode}.csv",
        starts,
        percentile_series,
        valid_counts,
        n_params,
    )

    plot_percentiles(
        output_dir / "plots",
        starts,
        percentile_series,
        valid_counts,
        n_params,
        gradients_path.parent.name,
        window,
        ratio_mode,
    )

    summary = {
        "run_name": gradients_path.parent.name,
        "window": window,
        "ratio_mode": ratio_mode,
        "eps": eps,
        "n_steps": steps,
        "n_parameters": n_params,
        "n_time_windows": n_windows,
        "ratios_path": str(ratios_path),
        "valid_fraction_mean": float(np.mean(valid_counts / n_params)),
        "valid_fraction_min": float(np.min(valid_counts / n_params)),
        "valid_fraction_max": float(np.max(valid_counts / n_params)),
        "percentile_temporal_means": {
            f"p{q}": float(np.nanmean(values)) for q, values in percentile_series.items()
        },
        "percentile_first_window": {
            f"p{q}": float(values[0]) for q, values in percentile_series.items()
        },
        "percentile_last_window": {
            f"p{q}": float(values[-1]) for q, values in percentile_series.items()
        },
    }
    with (output_dir / f"gradient_ratio_summary_w{window}_{ratio_mode}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze max/min gradient ratios in rolling windows.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--windows", type=int, nargs="+", default=[100])
    parser.add_argument("--percentiles", type=int, nargs="+", default=[5, 25, 50, 75, 95])
    parser.add_argument("--ratio-mode", choices=["abs", "signed", "same-sign-abs"], default="abs")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--eps", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    results_dir = args.results_dir.resolve()
    gradients_path = run_dir / "gradients.npy"
    if not gradients_path.is_file():
        raise FileNotFoundError(gradients_path)

    output_name = args.output_name or run_dir.name
    output_dir = results_dir / output_name / args.ratio_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for window in args.windows:
        summaries.append(
            analyze_window(
                gradients_path=gradients_path,
                output_dir=output_dir,
                window=window,
                percentiles_to_compute=args.percentiles,
                ratio_mode=args.ratio_mode,
                chunk_size=args.chunk_size,
                eps=args.eps,
            )
        )
    with (output_dir / "latest_gradient_ratio_summaries.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
