"""Regenerate Table 2, Table 4, and Appendix C training-dynamics plots.

Inputs may be the historical merged PKL files or directories of per-run JSON
files produced by ``code/experiments``.  PKL files must be trusted because the
Python pickle format is executable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


WINDOWS = (1, 10, 100, 200, 500)
MAIN_WINDOW = 200
SKIP_EPOCHS = 3


def ema(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or window <= 1:
        return values
    output = np.empty_like(values)
    alpha = 2.0 / (window + 1.0)
    output[0] = values[0]
    for index in range(1, values.size):
        output[index] = alpha * values[index] + (1.0 - alpha) * output[index - 1]
    return output


def start_index(length: int, epochs: int | float | None) -> int:
    if not epochs or epochs <= SKIP_EPOCHS:
        return 0
    return min(length, int(SKIP_EPOCHS * length / float(epochs)))


def processed(run: dict, window: int) -> np.ndarray:
    values = np.asarray(run.get("updates_norm", []), dtype=np.float64)
    values = values[start_index(values.size, run.get("epochs")) :]
    return ema(values, window)


def omega1(values: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(values)))) if values.size >= 2 else float("nan")


def omega2(values: np.ndarray, convention: str = "interior") -> float:
    if values.size < 3:
        return float("nan")
    if convention == "interior":
        # Exactly the T-2 terms in the equation in Appendix C.
        return float(np.mean(np.abs(np.diff(values, n=2))))
    if convention == "one-sided-boundaries":
        second = np.empty_like(values)
        second[1:-1] = values[2:] - 2.0 * values[1:-1] + values[:-2]
        second[0] = values[0] - 2.0 * values[1] + values[2]
        second[-1] = values[-1] - 2.0 * values[-2] + values[-3]
        return float(np.mean(np.abs(second)))
    raise ValueError(f"Unknown omega2 convention: {convention}")


def load_pickle(path: Path) -> dict[tuple[float, float, int], dict]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    epochs = payload.get("epochs")
    output = {}
    for key, run in payload["runs"].items():
        copied = dict(run)
        copied.setdefault("epochs", epochs)
        copied.setdefault("beta1", float(key[0]))
        copied.setdefault("beta2", float(key[1]))
        copied.setdefault("seed", int(key[2]))
        output[(float(key[0]), float(key[1]), int(key[2]))] = copied
    return output


def load_json_directory(path: Path) -> dict[tuple[float, float, int], dict]:
    output = {}
    for file_path in sorted(path.glob("*.json")):
        run = json.loads(file_path.read_text(encoding="utf-8"))
        if run.get("smoke") or run.get("smoke_test"):
            continue
        key = (float(run["beta1"]), float(run["beta2"]), int(run["seed"]))
        output[key] = run
    if not output:
        raise FileNotFoundError(f"No non-smoke JSON runs found in {path}")
    return output


def load_runs(specification: dict, data_root: Path) -> dict[tuple[float, float, int], dict]:
    path = data_root / specification["path"]
    if specification["format"] == "pickle":
        return load_pickle(path)
    if specification["format"] == "json_dir":
        return load_json_directory(path)
    raise ValueError(f"Unknown source format {specification['format']}")


def expected_axes(specification: dict) -> tuple[list[float], list[float], list[int]]:
    beta1 = [float(value) for value in specification["beta1"]]
    beta2 = [float(value) for value in specification["beta2"]]
    seeds = [int(value) for value in specification["seeds"]]
    return beta1, beta2, seeds


def validate_grid(specification: dict, runs: dict) -> None:
    beta1, beta2, seeds = expected_axes(specification)
    expected = {(a, b, seed) for a in beta1 for b in beta2 for seed in seeds}
    missing = expected - set(runs)
    if missing:
        raise RuntimeError(f"{specification['id']} is missing runs: {sorted(missing)}")


def metric_grid(
    specification: dict,
    runs: dict,
    window: int,
    metric: str,
    omega2_convention: str,
) -> np.ndarray:
    beta1, beta2, seeds = expected_axes(specification)
    grid = np.full((len(beta1), len(beta2), len(seeds)), np.nan, dtype=np.float64)
    for i, first in enumerate(beta1):
        for j, second in enumerate(beta2):
            for k, seed in enumerate(seeds):
                values = processed(runs[(first, second, seed)], window)
                grid[i, j, k] = (
                    omega1(values)
                    if metric == "omega1"
                    else omega2(values, omega2_convention)
                )
    return grid


def diagonal_selection_rate(grid: np.ndarray) -> tuple[int, int, float]:
    successes = 0
    trials = 0
    for row_index in range(grid.shape[0]):
        for seed_index in range(grid.shape[2]):
            row = np.where(np.isfinite(grid[row_index, :, seed_index]), grid[row_index, :, seed_index], np.inf)
            if np.all(np.isinf(row)):
                continue
            successes += int(row[row_index] == np.min(row))
            trials += 1
    return successes, trials, successes / trials if trials else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib(use_tex: bool) -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use("default")
    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        }
    )


def stack_truncated(series: list[np.ndarray]) -> np.ndarray:
    length = min(array.size for array in series)
    return np.vstack([array[:length] for array in series])


def plot_experiment(
    specification: dict,
    runs: dict,
    output_dir: Path,
    use_tex: bool,
) -> None:
    configure_matplotlib(use_tex)
    beta1, beta2, seeds = expected_axes(specification)
    rows, columns = len(beta1), len(beta2)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(9, 5) if rows == 3 else (7.4, 10.2),
        squeeze=False,
        sharex=True,
    )
    loss_handle = update_handle = None
    cache: dict[tuple[int, int], dict] = {}
    for i, first in enumerate(beta1):
        for j, second in enumerate(beta2):
            loss_series = []
            update_series = []
            run_omegas = []
            for seed in seeds:
                run = runs[(first, second, seed)]
                update = processed(run, MAIN_WINDOW)
                loss = np.asarray(run.get("train_step_loss", []), dtype=np.float64)
                loss = loss[start_index(loss.size, run.get("epochs")) :]
                loss = ema(loss, MAIN_WINDOW)
                length = min(loss.size, update.size)
                loss_series.append(loss[:length])
                update_series.append(update[:length])
                run_omegas.append(omega1(update))
            loss_stack = stack_truncated(loss_series)
            update_stack = stack_truncated(update_series)
            cache[(i, j)] = {
                "loss_mean": np.mean(loss_stack, axis=0),
                "loss_std": np.std(loss_stack, axis=0),
                "update_mean": np.mean(update_stack, axis=0),
                "update_std": np.std(update_stack, axis=0),
                "omega": float(np.mean(run_omegas)),
            }
    for i, first in enumerate(beta1):
        finite_loss = np.concatenate(
            [cache[(i, j)]["loss_mean"][np.isfinite(cache[(i, j)]["loss_mean"])] for j in range(columns)]
        )
        finite_update = np.concatenate(
            [cache[(i, j)]["update_mean"][np.isfinite(cache[(i, j)]["update_mean"])] for j in range(columns)]
        )
        loss_limits = (max(0.0, float(np.percentile(finite_loss, 1))), float(np.percentile(finite_loss, 99)))
        update_limits = (max(0.0, float(np.percentile(finite_update, 1))), float(np.percentile(finite_update, 99)))
        for j, second in enumerate(beta2):
            axis = axes[i, j]
            item = cache[(i, j)]
            stride = max(1, math.ceil(item["loss_mean"].size / 700))
            x = np.arange(item["loss_mean"].size)[::stride]
            loss_mean = item["loss_mean"][::stride]
            loss_std = item["loss_std"][::stride]
            update_mean = item["update_mean"][::stride]
            update_std = item["update_std"][::stride]
            axis.fill_between(x, loss_mean - loss_std, loss_mean + loss_std, color="#1f4e79", alpha=0.22, linewidth=0)
            loss_handle = axis.plot(x, loss_mean, color="#1f4e79", linewidth=2, label="Train loss")[0]
            axis.set_ylim(*loss_limits)
            twin = axis.twinx()
            twin.fill_between(x, update_mean - update_std, update_mean + update_std, color="#8b0000", alpha=0.22, linewidth=0)
            update_handle = twin.plot(x, update_mean, color="#8b0000", linewidth=1.8, label=r"$\|\mathbf{R}_k\|$")[0]
            twin.set_ylim(*update_limits)
            axis.set_title(rf"$\beta_1={first:g},\ \beta_2={second:g},\ \omega={item['omega']:.4g}$", fontsize=8)
            if i == rows - 1:
                axis.set_xlabel("Steps")
            if j == 0:
                axis.set_ylabel("Loss")
            else:
                axis.tick_params(axis="y", labelleft=False)
            if j == columns - 1:
                twin.set_ylabel(r"Norm of $\mathbf{R}_k$")
            else:
                twin.tick_params(axis="y", labelright=False)
    if loss_handle is not None and update_handle is not None:
        figure.legend(
            [loss_handle, update_handle],
            [loss_handle.get_label(), update_handle.get_label()],
            loc="lower center",
            ncol=2,
            frameon=True,
        )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / specification["figure"], bbox_inches="tight", dpi=300)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training_dynamics"))
    parser.add_argument(
        "--omega2-convention",
        choices=("interior", "one-sided-boundaries"),
        default="interior",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--use-tex", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    table2_rows = []
    table4_rows = []
    for specification in config["experiments"]:
        runs = load_runs(specification, args.data_root)
        validate_grid(specification, runs)
        main_grid = metric_grid(
            specification,
            runs,
            MAIN_WINDOW,
            "omega1",
            args.omega2_convention,
        )
        successes, trials, rate = diagonal_selection_rate(main_grid)
        beta1, beta2, _seeds = expected_axes(specification)
        for i, first in enumerate(beta1):
            for j, second in enumerate(beta2):
                # Deliberately propagate non-finite seeds: the manuscript reports
                # a cell as NaN if any repeated run is non-finite.
                table2_rows.append(
                    {
                        "experiment": specification["id"],
                        "beta1": first,
                        "beta2": second,
                        "omega_mean": float(np.mean(main_grid[i, j, :])),
                        "omega_std": float(np.std(main_grid[i, j, :])),
                        "n_seeds": main_grid.shape[2],
                        "selection_successes": successes,
                        "selection_trials": trials,
                        "selection_rate": rate,
                    }
                )
        for window in WINDOWS:
            first_grid = metric_grid(
                specification, runs, window, "omega1", args.omega2_convention
            )
            second_grid = metric_grid(
                specification, runs, window, "omega2", args.omega2_convention
            )
            first_successes, first_trials, first_rate = diagonal_selection_rate(first_grid)
            second_successes, second_trials, second_rate = diagonal_selection_rate(second_grid)
            table4_rows.append(
                {
                    "experiment": specification["id"],
                    "window": window,
                    "omega1_successes": first_successes,
                    "omega1_trials": first_trials,
                    "omega1_rate": first_rate,
                    "omega2_successes": second_successes,
                    "omega2_trials": second_trials,
                    "omega2_rate": second_rate,
                    "omega2_convention": args.omega2_convention,
                }
            )
        if not args.skip_plots:
            plot_experiment(specification, runs, args.output_dir / "figures", args.use_tex)
        print(f"Analyzed {specification['id']}: {len(runs)} runs", flush=True)
    write_csv(args.output_dir / "table2_omega.csv", table2_rows)
    write_csv(args.output_dir / "table4_ablation.csv", table4_rows)
    print(f"Wrote training-dynamics outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
