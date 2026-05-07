from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


BETA_VALUES = [0.900, 0.968, 0.990, 0.9968, 0.999]
SEED_VALUES = [0, 1, 2]
EMA_WINDOWS = [1, 10, 100, 200, 500]
MAIN_EMA_WINDOW = 200
SKIP_EPOCHS = 3
PLOT_STRIDE = 100
MAX_PLOT_POINTS = 700

EXPERIMENT_TITLES = {
    "resnet18_cifar100": "ResNet18 on CIFAR-100",
    "nanogpt_wikitext": "NanoGPT on WikiText",
}


def beta_label(value: float) -> str:
    if value == 0.9968:
        return "0.9968"
    return f"{value:.3f}"


def beta_key(value: float) -> float:
    return round(float(value), 4)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_matplotlib(use_latex: bool = True) -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use("default")
    plt.rcParams.update(
        {
            "text.usetex": use_latex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_runs(results_root: Path, experiment: str) -> list[dict]:
    runs_dir = results_root / experiment / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Missing runs directory: {runs_dir}")

    runs = []
    for path in sorted(runs_dir.glob("*.json")):
        run = load_json(path)
        run["_path"] = str(path)
        if run.get("smoke"):
            continue
        runs.append(run)
    return runs


def index_runs(runs: list[dict]) -> dict[tuple[float, float, int], dict]:
    indexed = {}
    for run in runs:
        key = (beta_key(run["beta1"]), beta_key(run["beta2"]), int(run["seed"]))
        indexed[key] = run
    return indexed


def validate_grid(indexed: dict[tuple[float, float, int], dict]) -> tuple[list[tuple[float, float, int]], list[tuple[float, float, int]]]:
    expected = {
        (beta_key(beta1), beta_key(beta2), seed)
        for beta1 in BETA_VALUES
        for beta2 in BETA_VALUES
        for seed in SEED_VALUES
    }
    observed = set(indexed)
    return sorted(expected - observed), sorted(observed - expected)


def ema(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.astype(float, copy=False)
    alpha = 2.0 / (window + 1.0)
    out = np.empty(values.size, dtype=float)
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def safe_start_index(run: dict, series_len: int, skip_epochs: int = SKIP_EPOCHS) -> int:
    epochs = run.get("epochs")
    if not epochs or skip_epochs <= 0:
        return 0
    return min(series_len, int(skip_epochs * series_len / float(epochs)))


def processed_series(run: dict, key: str, ema_window: int = MAIN_EMA_WINDOW) -> np.ndarray:
    if key not in run:
        return np.array([], dtype=float)
    values = np.asarray(run[key], dtype=float)
    start = safe_start_index(run, values.size)
    return ema(values[start:], ema_window)


def truncate_stack(series_list: list[np.ndarray]) -> np.ndarray:
    series_list = [arr for arr in series_list if arr.size > 0]
    if not series_list:
        return np.empty((0, 0), dtype=float)
    min_len = min(arr.size for arr in series_list)
    return np.vstack([arr[:min_len] for arr in series_list])


def downsample(x: np.ndarray, *ys: np.ndarray, max_points: int = MAX_PLOT_POINTS) -> tuple[np.ndarray, ...]:
    if x.size <= max_points:
        return (x, *ys)
    step = int(math.ceil(x.size / max_points))
    return (x[::step], *(y[::step] for y in ys))


def oscillation_first_difference(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(values))))


def oscillation_second_difference(values: np.ndarray) -> float:
    if values.size < 3:
        return float("nan")
    return float(np.mean(np.abs(np.diff(values, n=2))))


def binom_sf(k: int, n: int, p: float) -> float:
    if k <= 0:
        return 1.0
    return float(sum(math.comb(n, j) * (p**j) * ((1.0 - p) ** (n - j)) for j in range(k, n + 1)))


def row_min_test(grid: np.ndarray, tie_mode: str = "include") -> dict:
    n_success = 0
    n_trials = 0
    n_beta2 = grid.shape[1]

    for i in range(grid.shape[0]):
        for s in range(grid.shape[2]):
            row = grid[i, :, s]
            if np.all(np.isnan(row)):
                continue
            finite = np.isfinite(row)
            if not finite[i]:
                continue
            if finite.sum() != n_beta2:
                continue
            row_min = np.nanmin(row)
            diagonal = row[i]
            if tie_mode == "include":
                success = diagonal <= row_min
            elif tie_mode == "strict":
                success = diagonal == row_min and np.sum(row == row_min) == 1
            else:
                raise ValueError(f"Unknown tie_mode: {tie_mode}")
            n_success += int(success)
            n_trials += 1

    p0 = 1.0 / n_beta2
    p_value = binom_sf(n_success, n_trials, p0) if n_trials else float("nan")
    return {
        "n_success": n_success,
        "n_trials": n_trials,
        "rate": float(n_success / n_trials) if n_trials else float("nan"),
        "p0": p0,
        "p_value": p_value,
    }


def min_val_loss(run: dict) -> float:
    values = run.get("val_loss", [])
    if not values:
        return float("nan")
    return float(np.nanmin(np.asarray(values, dtype=float)))


def final_or_nan(run: dict, key: str) -> float:
    value = run.get(key)
    if isinstance(value, list) and value:
        return float(value[-1])
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_grid_plot(
    experiment: str,
    indexed: dict[tuple[float, float, int], dict],
    plots_dir: Path,
    *,
    use_latex: bool,
) -> None:
    configure_matplotlib(use_latex=use_latex)

    color_loss = "#1f4e79"
    color_r = "#8b0000"
    fill_alpha = 0.22
    lw_loss = 2.0
    lw_r = 1.8
    n = len(BETA_VALUES)

    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(7.4, 10.2), sharex=True, sharey=False)

    loss_handle = None
    r_handle = None
    cache: list[list[dict | None]] = [[None] * n for _ in range(n)]

    for row_idx, beta1 in enumerate(BETA_VALUES):
        for col_idx, beta2 in enumerate(BETA_VALUES):
            loss_runs = []
            r_runs = []
            for seed in SEED_VALUES:
                run = indexed.get((beta_key(beta1), beta_key(beta2), seed))
                if not run:
                    continue
                loss = processed_series(run, "train_step_loss", MAIN_EMA_WINDOW)
                r_norm = processed_series(run, "updates_norm", MAIN_EMA_WINDOW)
                if loss.size > 0 and r_norm.size > 0:
                    loss_runs.append(loss)
                    r_runs.append(r_norm)

            loss_stack = truncate_stack(loss_runs)
            r_stack = truncate_stack(r_runs)
            if loss_stack.size == 0 or r_stack.size == 0:
                continue

            min_len = min(loss_stack.shape[1], r_stack.shape[1])
            loss_stack = loss_stack[:, :min_len]
            r_stack = r_stack[:, :min_len]

            loss_mean = np.mean(loss_stack, axis=0)
            loss_std = np.std(loss_stack, axis=0)
            r_mean = np.mean(r_stack, axis=0)
            r_std = np.std(r_stack, axis=0)
            omega = float(np.nanmean([oscillation_first_difference(x) for x in r_stack]))

            sl = slice(None, None, PLOT_STRIDE)
            cache[row_idx][col_idx] = {
                "x": np.arange(min_len)[sl],
                "loss_mean": loss_mean[sl],
                "loss_lo": (loss_mean - loss_std)[sl],
                "loss_hi": (loss_mean + loss_std)[sl],
                "r_mean": r_mean[sl],
                "r_lo": (r_mean - r_std)[sl],
                "r_hi": (r_mean + r_std)[sl],
                "omega": omega,
            }

    for row_idx, beta1 in enumerate(BETA_VALUES):
        loss_min = +np.inf
        loss_max = -np.inf
        r_min = +np.inf
        r_max = -np.inf

        for col_idx in range(n):
            d = cache[row_idx][col_idx]
            if d is None:
                continue
            loss_min = max(0.0, min(loss_min, float(np.percentile(d["loss_lo"], 1))))
            loss_max = max(loss_max, float(np.percentile(d["loss_hi"], 99)))
            r_min = max(0.0, min(r_min, float(np.percentile(d["r_lo"], 1))))
            r_max = max(r_max, float(np.percentile(d["r_hi"], 99)))

        if not np.isfinite(loss_min) or not np.isfinite(loss_max):
            loss_min, loss_max = 0.0, 1.0
        if not np.isfinite(r_min) or not np.isfinite(r_max):
            r_min, r_max = 0.0, 1.0
        loss_pad = 0.1 * (loss_max - loss_min + 1e-12)
        r_pad = 0.1 * (r_max - r_min + 1e-12)

        for col_idx, beta2 in enumerate(BETA_VALUES):
            ax = axes[row_idx, col_idx]
            d = cache[row_idx][col_idx]
            if d is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            ax.fill_between(d["x"], d["loss_lo"], d["loss_hi"], linewidth=0.0, color=color_loss, alpha=fill_alpha)
            loss_handle = ax.plot(
                d["x"], d["loss_mean"], linestyle="-", linewidth=lw_loss, color=color_loss, label=r"Train loss"
            )[0]

            if row_idx == n - 1:
                ax.set_xlabel(r"Steps")
                ax.tick_params(axis="x", which="both", labelbottom=True)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", which="both", labelbottom=False)

            if col_idx == 0:
                ax.set_ylabel(r"Loss")
                ax.tick_params(axis="y", which="both", labelleft=True)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", which="both", labelleft=False)

            ax.set_ylim(loss_min - loss_pad, loss_max + loss_pad)

            ax2 = ax.twinx()
            ax2.fill_between(d["x"], d["r_lo"], d["r_hi"], linewidth=0.0, color=color_r, alpha=fill_alpha)
            r_handle = ax2.plot(
                d["x"],
                d["r_mean"],
                linestyle="-",
                linewidth=lw_r,
                color=color_r,
                label=r"$\|\mathbf{R}_k\|$",
            )[0]

            if col_idx == n - 1:
                ax2.set_ylabel(r"Norm of $\mathbf{R}_k$")
                ax2.tick_params(axis="y", which="both", labelright=True)
            else:
                ax2.set_ylabel("")
                ax2.tick_params(axis="y", which="both", labelright=False)

            ax2.set_ylim(r_min - r_pad, r_max + r_pad)
            ax.set_title(
                rf"$\beta_1={beta_label(beta1)},\ \beta_2={beta_label(beta2)}$" + "\n" + rf"$\omega={d['omega']:.4g}$",
                fontsize=8,
            )

    if loss_handle is not None and r_handle is not None:
        fig.legend(
            [loss_handle, r_handle],
            [loss_handle.get_label(), r_handle.get_label()],
            loc="lower center",
            ncol=2,
            frameon=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.015),
        )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    ensure_dir(plots_dir)
    fig.savefig(plots_dir / f"{experiment}_grid5.png", bbox_inches="tight", dpi=300)
    fig.savefig(plots_dir / f"{experiment}_grid5.pdf", bbox_inches="tight")
    plt.close(fig)


def analyze_experiment(
    experiment: str,
    results_root: Path,
    plots_dir: Path,
    *,
    use_latex: bool,
) -> dict[str, list[dict]]:
    runs = discover_runs(results_root, experiment)
    indexed = index_runs(runs)
    missing, extra = validate_grid(indexed)
    print(f"[{experiment}] runs={len(runs)} missing={len(missing)} extra={len(extra)}")
    if missing:
        print(f"[{experiment}] missing keys: {missing}")
    if extra:
        print(f"[{experiment}] extra keys: {extra}")

    make_grid_plot(experiment, indexed, plots_dir, use_latex=use_latex)

    run_rows = []
    osc_rows = []
    grid_rows = []
    test_rows = []

    for key, run in sorted(indexed.items()):
        beta1, beta2, seed = key
        run_rows.append(
            {
                "experiment": experiment,
                "beta1": beta1,
                "beta2": beta2,
                "seed": seed,
                "final_train_loss": final_or_nan(run, "train_loss"),
                "final_val_loss": final_or_nan(run, "val_loss"),
                "min_val_loss": min_val_loss(run),
                "final_train_metric": final_or_nan(run, "train_metric"),
                "final_val_metric": final_or_nan(run, "val_metric"),
                "wall_time_total_sec": final_or_nan(run, "wall_time_total_sec"),
                "training_wall_time_sec": final_or_nan(run, "training_wall_time_sec"),
                "path": run.get("_path", ""),
            }
        )

        for window in EMA_WINDOWS:
            values = processed_series(run, "updates_norm", window)
            osc_rows.append(
                {
                    "experiment": experiment,
                    "beta1": beta1,
                    "beta2": beta2,
                    "seed": seed,
                    "ema_window": window,
                    "omega1_updates_norm": oscillation_first_difference(values),
                    "omega2_updates_norm": oscillation_second_difference(values),
                }
            )

    for window in EMA_WINDOWS:
        omega1_grid = np.full((len(BETA_VALUES), len(BETA_VALUES), len(SEED_VALUES)), np.nan, dtype=float)
        omega2_grid = np.full_like(omega1_grid, np.nan)

        for i, beta1 in enumerate(BETA_VALUES):
            for j, beta2 in enumerate(BETA_VALUES):
                for k, seed in enumerate(SEED_VALUES):
                    run = indexed.get((beta_key(beta1), beta_key(beta2), seed))
                    if not run:
                        continue
                    values = processed_series(run, "updates_norm", window)
                    omega1_grid[i, j, k] = oscillation_first_difference(values)
                    omega2_grid[i, j, k] = oscillation_second_difference(values)

                omega1_values = omega1_grid[i, j, :]
                omega2_values = omega2_grid[i, j, :]
                grid_rows.append(
                    {
                        "experiment": experiment,
                        "beta1": beta_key(beta1),
                        "beta2": beta_key(beta2),
                        "ema_window": window,
                        "omega1_mean": float(np.nanmean(omega1_values)),
                        "omega1_std": float(np.nanstd(omega1_values)),
                        "omega2_mean": float(np.nanmean(omega2_values)),
                        "omega2_std": float(np.nanstd(omega2_values)),
                    }
                )

        for name, grid in [("omega1_updates_norm", omega1_grid), ("omega2_updates_norm", omega2_grid)]:
            test = row_min_test(grid)
            test_rows.append(
                {
                    "experiment": experiment,
                    "test": name,
                    "ema_window": window,
                    "n_success": test["n_success"],
                    "n_trials": test["n_trials"],
                    "rate": test["rate"],
                    "p0": test["p0"],
                    "p_value": test["p_value"],
                }
            )

    loss_grid = np.full((len(BETA_VALUES), len(BETA_VALUES), len(SEED_VALUES)), np.nan, dtype=float)
    for i, beta1 in enumerate(BETA_VALUES):
        for j, beta2 in enumerate(BETA_VALUES):
            for k, seed in enumerate(SEED_VALUES):
                run = indexed.get((beta_key(beta1), beta_key(beta2), seed))
                if run:
                    loss_grid[i, j, k] = min_val_loss(run)
    loss_test = row_min_test(loss_grid)
    test_rows.append(
        {
            "experiment": experiment,
            "test": "min_val_loss",
            "ema_window": "",
            "n_success": loss_test["n_success"],
            "n_trials": loss_test["n_trials"],
            "rate": loss_test["rate"],
            "p0": loss_test["p0"],
            "p_value": loss_test["p_value"],
        }
    )

    return {
        "run_rows": run_rows,
        "osc_rows": osc_rows,
        "grid_rows": grid_rows,
        "test_rows": test_rows,
    }


def print_summary(run_rows: list[dict], test_rows: list[dict]) -> None:
    by_experiment: dict[str, list[dict]] = {}
    for row in run_rows:
        by_experiment.setdefault(row["experiment"], []).append(row)

    for experiment, rows in by_experiment.items():
        val_losses = np.asarray([float(row["final_val_loss"]) for row in rows], dtype=float)
        train_losses = np.asarray([float(row["final_train_loss"]) for row in rows], dtype=float)
        times = np.asarray([float(row["wall_time_total_sec"]) for row in rows], dtype=float)
        best = min(rows, key=lambda row: float(row["final_val_loss"]))
        print(
            f"[{experiment}] final_val_loss min/mean/max="
            f"{np.nanmin(val_losses):.6g}/{np.nanmean(val_losses):.6g}/{np.nanmax(val_losses):.6g}; "
            f"final_train_loss min/mean/max="
            f"{np.nanmin(train_losses):.6g}/{np.nanmean(train_losses):.6g}/{np.nanmax(train_losses):.6g}; "
            f"wall_time_sec min/mean/max="
            f"{np.nanmin(times):.1f}/{np.nanmean(times):.1f}/{np.nanmax(times):.1f}"
        )
        print(
            f"[{experiment}] best final val: beta1={best['beta1']} beta2={best['beta2']} "
            f"seed={best['seed']} val_loss={float(best['final_val_loss']):.6g}"
        )

    for row in test_rows:
        if row["test"] == "omega1_updates_norm" and str(row["ema_window"]) == str(MAIN_EMA_WINDOW):
            print(
                f"[{row['experiment']}] hypothesis {row['test']} ema={row['ema_window']}: "
                f"rate={float(row['rate']):.4f}, p_value={float(row['p_value']):.6g}, "
                f"success={row['n_success']}/{row['n_trials']}"
            )


def format_report_table(experiment: str, grid_rows: list[dict], test_rows: list[dict]) -> str:
    values = {
        (float(row["beta1"]), float(row["beta2"])): float(row["omega1_mean"])
        for row in grid_rows
        if row["experiment"] == experiment and int(row["ema_window"]) == MAIN_EMA_WINDOW
    }
    test = next(
        row
        for row in test_rows
        if row["experiment"] == experiment
        and row["test"] == "omega1_updates_norm"
        and int(row["ema_window"]) == MAIN_EMA_WINDOW
    )

    lines = []
    lines.append(EXPERIMENT_TITLES.get(experiment, experiment))
    lines.append(f"Omega = mean over seeds of osc1(EMA_{MAIN_EMA_WINDOW}(||R_k||))")
    header = ["beta1 \\ beta2"] + [beta_label(beta2) for beta2 in BETA_VALUES]
    widths = [13] + [12] * len(BETA_VALUES)
    lines.append(" ".join(item.rjust(width) for item, width in zip(header, widths)))
    lines.append("-" * (sum(widths) + len(widths) - 1))

    for beta1 in BETA_VALUES:
        row = [beta_label(beta1)]
        for beta2 in BETA_VALUES:
            value = values.get((beta_key(beta1), beta_key(beta2)), float("nan"))
            row.append("nan" if not np.isfinite(value) else f"{value:.4f}")
        lines.append(" ".join(item.rjust(width) for item, width in zip(row, widths)))

    lines.append("")
    lines.append(
        "rate = "
        f"{float(test['rate']):.4f} "
        f"({test['n_success']}/{test['n_trials']}), "
        f"p-value = {float(test['p_value']):.6g}"
    )
    return "\n".join(lines)


def write_text_report(output_dir: Path, grid_rows: list[dict], test_rows: list[dict], experiments: list[str]) -> Path:
    path = output_dir / "omega1_grid5_report.txt"
    blocks = [
        "5x5 beta-grid report",
        f"Values shown with 4 decimals. EMA window = {MAIN_EMA_WINDOW}.",
        "",
    ]
    for experiment in experiments:
        blocks.append(format_report_table(experiment, grid_rows, test_rows))
        blocks.append("")
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze downloaded 5x5 Adam beta-grid results.")
    parser.add_argument("--results-root", type=Path, default=Path("../../descargas/results"))
    parser.add_argument("--plots-dir", type=Path, default=Path("../plots"))
    parser.add_argument("--output-dir", type=Path, default=Path("grid5_outputs"))
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["resnet18_cifar100", "nanogpt_wikitext"],
        help="Experiment directory names under results root.",
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Disable LaTeX rendering for plots if the local machine has no LaTeX installation.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    results_root = (script_dir / args.results_root).resolve()
    plots_dir = (script_dir / args.plots_dir).resolve()
    output_dir = ensure_dir((script_dir / args.output_dir).resolve())

    print(f"results_root={results_root}")
    print(f"plots_dir={plots_dir}")
    print(f"output_dir={output_dir}")

    all_run_rows = []
    all_osc_rows = []
    all_grid_rows = []
    all_test_rows = []

    for experiment in args.experiments:
        result = analyze_experiment(experiment, results_root, plots_dir, use_latex=not args.no_latex)
        all_run_rows.extend(result["run_rows"])
        all_osc_rows.extend(result["osc_rows"])
        all_grid_rows.extend(result["grid_rows"])
        all_test_rows.extend(result["test_rows"])

    write_csv(
        output_dir / "run_summary.csv",
        [
            "experiment",
            "beta1",
            "beta2",
            "seed",
            "final_train_loss",
            "final_val_loss",
            "min_val_loss",
            "final_train_metric",
            "final_val_metric",
            "wall_time_total_sec",
            "training_wall_time_sec",
            "path",
        ],
        all_run_rows,
    )
    write_csv(
        output_dir / "updates_norm_oscillations.csv",
        [
            "experiment",
            "beta1",
            "beta2",
            "seed",
            "ema_window",
            "omega1_updates_norm",
            "omega2_updates_norm",
        ],
        all_osc_rows,
    )
    write_csv(
        output_dir / "updates_norm_oscillation_grid_mean.csv",
        [
            "experiment",
            "beta1",
            "beta2",
            "ema_window",
            "omega1_mean",
            "omega1_std",
            "omega2_mean",
            "omega2_std",
        ],
        all_grid_rows,
    )
    write_csv(
        output_dir / "hypothesis_tests.csv",
        ["experiment", "test", "ema_window", "n_success", "n_trials", "rate", "p0", "p_value"],
        all_test_rows,
    )
    report_path = write_text_report(output_dir, all_grid_rows, all_test_rows, args.experiments)

    print_summary(all_run_rows, all_test_rows)
    print("Wrote:")
    print(f"  {report_path}")
    print(f"  {output_dir / 'run_summary.csv'}")
    print(f"  {output_dir / 'updates_norm_oscillations.csv'}")
    print(f"  {output_dir / 'updates_norm_oscillation_grid_mean.csv'}")
    print(f"  {output_dir / 'hypothesis_tests.csv'}")
    print(f"  {plots_dir / 'resnet18_cifar100_grid5.png'}")
    print(f"  {plots_dir / 'nanogpt_wikitext_grid5.png'}")


if __name__ == "__main__":
    main()
