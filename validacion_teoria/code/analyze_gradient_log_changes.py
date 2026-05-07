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
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "log_gradient_changes"


def finite_update(count: int, mean: float, m2: float, values: np.ndarray) -> tuple[int, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return count, mean, m2

    batch_count = int(values.size)
    batch_mean = float(values.mean(dtype=np.float64))
    batch_m2 = float(((values.astype(np.float64) - batch_mean) ** 2).sum())

    if count == 0:
        return batch_count, batch_mean, batch_m2

    delta = batch_mean - mean
    new_count = count + batch_count
    new_mean = mean + delta * batch_count / new_count
    new_m2 = m2 + batch_m2 + delta * delta * count * batch_count / new_count
    return new_count, new_mean, new_m2


def append_sample(
    sample: np.ndarray,
    filled: int,
    seen_before: int,
    values: np.ndarray,
    rng: np.random.Generator,
) -> int:
    values = values[np.isfinite(values)]
    if values.size == 0 or sample.size == 0:
        return filled

    remaining = sample.size - filled
    if remaining > 0:
        take = min(remaining, values.size)
        sample[filled : filled + take] = values[:take]
        filled += take
        seen_before += take
        values = values[take:]

    if values.size > 0:
        totals = seen_before + np.arange(1, values.size + 1, dtype=np.int64)
        keep = rng.random(values.size) < (sample.size / totals)
        if np.any(keep):
            replace_idx = rng.integers(0, sample.size, size=int(np.count_nonzero(keep)))
            sample[replace_idx] = values[keep]
    return filled


def build_change_matrix(
    gradients_path: Path,
    output_path: Path,
    mode: str,
    chunk_size: int,
    eps: float,
    sample_size: int,
    seed: int,
) -> dict[str, object]:
    gradients = np.load(gradients_path, mmap_mode="r")
    if gradients.ndim != 2:
        raise ValueError(f"Expected gradients with shape (steps, parameters), got {gradients.shape}")

    steps, n_params = gradients.shape
    if mode == "pair_s":
        n_rows = steps - 1
    elif mode in {"triple_q", "triple_r"}:
        n_rows = steps - 2
    else:
        raise ValueError(mode)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    changes = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(n_rows, n_params))
    changes[:, :] = np.nan

    total_cases = n_rows * n_params
    valid_cases = 0
    count = 0
    mean = 0.0
    m2 = 0.0
    min_value = float("inf")
    max_value = float("-inf")
    sample = np.empty(sample_size, dtype=np.float32)
    sample_filled = 0
    rng = np.random.default_rng(seed)

    for start in range(0, n_params, chunk_size):
        stop = min(start + chunk_size, n_params)
        chunk = np.asarray(gradients[:, start:stop], dtype=np.float32)
        signs = np.sign(chunk).astype(np.int8, copy=False)
        abs_chunk = np.abs(chunk)
        log_abs = np.full_like(abs_chunk, np.nan, dtype=np.float32)
        positive = abs_chunk > eps
        log_abs[positive] = np.log(abs_chunk[positive])

        if mode == "pair_s":
            valid = positive[1:, :] & positive[:-1, :] & (signs[1:, :] == signs[:-1, :])
            values = log_abs[1:, :] - log_abs[:-1, :]
        elif mode == "triple_q":
            valid = (
                positive[2:, :]
                & positive[1:-1, :]
                & positive[:-2, :]
                & (signs[2:, :] == signs[1:-1, :])
                & (signs[1:-1, :] == signs[:-2, :])
            )
            values = (log_abs[2:, :] - log_abs[:-2, :]) / 2.0
        else:
            valid = (
                positive[2:, :]
                & positive[1:-1, :]
                & positive[:-2, :]
                & (signs[2:, :] == signs[1:-1, :])
                & (signs[1:-1, :] == signs[:-2, :])
            )
            values = log_abs[2:, :] - 2.0 * log_abs[1:-1, :] + log_abs[:-2, :]

        values = values.astype(np.float32, copy=False)
        values[~valid] = np.nan
        changes[:, start:stop] = values

        finite = values[np.isfinite(values)]
        seen_before = valid_cases
        valid_cases += int(finite.size)
        if finite.size > 0:
            min_value = min(min_value, float(finite.min()))
            max_value = max(max_value, float(finite.max()))
            count, mean, m2 = finite_update(count, mean, m2, finite)
            sample_filled = append_sample(sample, sample_filled, seen_before, finite, rng)

        print(f"[{mode}] processed coordinates {start}:{stop} / {n_params}", flush=True)

    changes.flush()
    sample = sample[:sample_filled]
    sample_path = output_path.with_name(output_path.stem + "_sample.npy")
    np.save(sample_path, sample)

    variance = m2 / (count - 1) if count > 1 else float("nan")
    sample_percentiles = {
        f"p{q:g}": float(v)
        for q, v in zip(
            [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5],
            np.percentile(sample, [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5]),
        )
    }
    return {
        "mode": mode,
        "matrix_path": str(output_path),
        "sample_path": str(sample_path),
        "n_steps": steps,
        "n_parameters": n_params,
        "n_time_positions": n_rows,
        "total_cases": total_cases,
        "valid_cases": valid_cases,
        "valid_percent": 100.0 * valid_cases / total_cases,
        "mean": mean,
        "std": float(np.sqrt(variance)),
        "min": min_value,
        "max": max_value,
        "sample_size": int(sample.size),
        "sample_percentiles": sample_percentiles,
    }


def temporal_percentiles(
    matrix_path: Path,
    percentiles: list[float],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    matrix = np.load(matrix_path, mmap_mode="r")
    n_rows = matrix.shape[0]
    series = {f"p{q:g}": np.empty(n_rows, dtype=np.float32) for q in percentiles}
    valid_fraction = np.empty(n_rows, dtype=np.float32)

    for row_idx in range(n_rows):
        row = np.asarray(matrix[row_idx, :])
        finite = row[np.isfinite(row)]
        valid_fraction[row_idx] = finite.size / row.size
        if finite.size == 0:
            values = np.full(len(percentiles), np.nan, dtype=np.float32)
        else:
            values = np.percentile(finite, percentiles).astype(np.float32)
        for key, value in zip(series, values):
            series[key][row_idx] = value

    return np.arange(n_rows), series, valid_fraction


def write_temporal_csv(
    path: Path,
    steps: np.ndarray,
    series: dict[str, np.ndarray],
    valid_fraction: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_index", "valid_fraction"] + list(series.keys()))
        for idx, step in enumerate(steps):
            writer.writerow([int(step), float(valid_fraction[idx])] + [float(series[key][idx]) for key in series])


def plot_temporal(
    path: Path,
    steps: np.ndarray,
    series: dict[str, np.ndarray],
    valid_fraction: np.ndarray,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(11, 6))
    for key, values in series.items():
        ax.plot(steps, values, linewidth=1.1, label=key)
    ax.axhline(0.0, color="#111827", linewidth=0.9, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("time index k")
    ax.set_ylabel("log-gradient change")
    ax.legend(ncol=min(5, len(series)))
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(steps, 100.0 * valid_fraction, color="#059669", linewidth=1.1)
    ax.set_title(title + " - valid same-sign fraction")
    ax.set_xlabel("time index k")
    ax.set_ylabel("% valid")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    valid_path = path.with_name(path.stem + "_valid_fraction")
    fig.savefig(valid_path.with_suffix(".png"), dpi=180)
    fig.savefig(valid_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_distribution(path: Path, sample_path: Path, title: str) -> None:
    sample = np.load(sample_path)
    lo, hi = np.percentile(sample, [0.5, 99.5])
    sample_clip = sample[(sample >= lo) & (sample <= hi)]

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(sample_clip, bins=160, density=True, color="#2563eb", alpha=0.75)
    for q, color in [(25, "#9333ea"), (50, "#111827"), (75, "#f59e0b")]:
        ax.axvline(np.percentile(sample, q), color=color, linewidth=1.2, label=f"p{q}")
    ax.axvline(0.0, color="#dc2626", linewidth=1.0, linestyle="--", label="0")
    ax.set_title(title + " - global sampled distribution")
    ax.set_xlabel("log-gradient change")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def analyze_mode(
    gradients_path: Path,
    output_dir: Path,
    mode: str,
    chunk_size: int,
    eps: float,
    sample_size: int,
    seed: int,
    percentiles: list[float],
) -> dict[str, object]:
    mode_dir = output_dir / mode
    matrix_path = mode_dir / f"{mode}.npy"
    summary = build_change_matrix(
        gradients_path=gradients_path,
        output_path=matrix_path,
        mode=mode,
        chunk_size=chunk_size,
        eps=eps,
        sample_size=sample_size,
        seed=seed,
    )
    steps, series, valid_fraction = temporal_percentiles(matrix_path, percentiles)
    write_temporal_csv(mode_dir / f"{mode}_temporal_percentiles.csv", steps, series, valid_fraction)
    plot_temporal(mode_dir / "plots" / f"{mode}_temporal_percentiles", steps, series, valid_fraction, mode)
    plot_distribution(mode_dir / "plots" / f"{mode}_global_distribution", Path(summary["sample_path"]), mode)

    summary["temporal_percentile_means"] = {key: float(np.nanmean(values)) for key, values in series.items()}
    summary["temporal_valid_fraction_mean"] = float(np.mean(valid_fraction))
    summary["temporal_valid_fraction_min"] = float(np.min(valid_fraction))
    summary["temporal_valid_fraction_max"] = float(np.max(valid_fraction))
    with (mode_dir / f"{mode}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze log changes of same-sign gradient coordinates.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["pair_s", "triple_q", "triple_r"],
        default=["pair_s", "triple_q", "triple_r"],
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--sample-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--percentiles", type=float, nargs="+", default=[5, 25, 50, 75, 95])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    gradients_path = run_dir / "gradients.npy"
    if not gradients_path.is_file():
        raise FileNotFoundError(gradients_path)

    output_name = args.output_name or run_dir.name
    output_dir = args.results_dir.resolve() / output_name
    summaries = []
    for mode in args.modes:
        summaries.append(
            analyze_mode(
                gradients_path=gradients_path,
                output_dir=output_dir,
                mode=mode,
                chunk_size=args.chunk_size,
                eps=args.eps,
                sample_size=args.sample_size,
                seed=args.seed,
                percentiles=args.percentiles,
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "latest_log_change_summaries.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
