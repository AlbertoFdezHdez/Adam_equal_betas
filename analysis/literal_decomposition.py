"""Literal discrete decomposition R = S + L_disc + E for one trajectory.

The implementation follows Appendix B exactly: full-history bias-corrected
EMAs, float64 analysis, epsilon outside the square root, and inclusive
sign-stable windows.  All coordinates are analyzed by default; ``--coordinates``
is intended only for quick checks and is never used for manuscript values.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def corrected_ema(values: np.ndarray, beta: float) -> np.ndarray:
    state = np.zeros(values.shape[1], dtype=np.float64)
    output = np.empty_like(values, dtype=np.float64)
    beta_power = 1.0
    for k in range(values.shape[0]):
        state = beta * state + (1.0 - beta) * values[k]
        beta_power *= beta
        output[k] = state / (1.0 - beta_power)
    return output


def sign_stable_mask(gradients: np.ndarray, window: int, burn_in: int) -> np.ndarray:
    """Return valid endpoints k for gradients[k-window+1:k+1]."""
    if window < 1:
        raise ValueError("window must be positive")
    steps, coordinates = gradients.shape
    result = np.zeros((steps, coordinates), dtype=bool)
    zeros = gradients == 0.0
    nonfinite = ~np.isfinite(gradients)
    changes = np.zeros_like(result)
    changes[1:] = np.signbit(gradients[1:]) != np.signbit(gradients[:-1])
    invalid_prefix = np.vstack(
        [
            np.zeros((1, coordinates), dtype=np.int32),
            np.cumsum(zeros | nonfinite, axis=0, dtype=np.int32),
        ]
    )
    change_prefix = np.vstack(
        [
            np.zeros((1, coordinates), dtype=np.int32),
            np.cumsum(changes, axis=0, dtype=np.int32),
        ]
    )
    for endpoint in range(window - 1, steps):
        start = endpoint - window + 1
        invalid_count = invalid_prefix[endpoint + 1] - invalid_prefix[start]
        # The first sign in a block has no within-block predecessor.
        change_count = change_prefix[endpoint + 1] - change_prefix[start + 1]
        result[endpoint] = (invalid_count == 0) & (change_count == 0)
    result[:burn_in] = False
    return result


def new_stats() -> defaultdict[str, float]:
    return defaultdict(float)


def accumulate(
    stats: defaultdict[str, float],
    normalized_update: np.ndarray,
    sign: np.ndarray,
    lag: np.ndarray,
    valid: np.ndarray,
) -> None:
    error = normalized_update - sign - lag
    safe = (
        valid
        & np.isfinite(normalized_update)
        & np.isfinite(sign)
        & np.isfinite(lag)
        & np.isfinite(error)
    )
    r = normalized_update[safe]
    s = sign[safe]
    ell = lag[safe]
    residual = error[safe]
    if r.size == 0:
        return
    deviation = r - s
    stats["n"] += int(r.size)
    stats["sum_abs_S"] += float(np.abs(s).sum())
    stats["sum_abs_L"] += float(np.abs(ell).sum())
    stats["sum_abs_E"] += float(np.abs(residual).sum())
    stats["sum_abs_LplusE"] += float(np.abs(ell + residual).sum())
    stats["sum_sq_D"] += float(deviation @ deviation)
    stats["sum_sq_E"] += float(residual @ residual)
    directional = (ell != 0.0) & (deviation != 0.0)
    stats["n_directional"] += int(directional.sum())
    stats["n_directional_agree"] += int(
        np.count_nonzero(ell[directional] * deviation[directional] > 0.0)
    )
    stats["max_identity_error"] = max(
        stats["max_identity_error"],
        float(np.max(np.abs(r - s - ell - residual))),
    )


def finalize(stats: defaultdict[str, float], window: int) -> dict[str, float | int]:
    tiny = np.finfo(np.float64).tiny
    budget = stats["sum_abs_S"] + stats["sum_abs_L"] + stats["sum_abs_E"]
    return {
        "window": window,
        "n": int(stats["n"]),
        "S_pct": 100.0 * stats["sum_abs_S"] / max(budget, tiny),
        "L_pct": 100.0 * stats["sum_abs_L"] / max(budget, tiny),
        "E_pct": 100.0 * stats["sum_abs_E"] / max(budget, tiny),
        "kappa": (stats["sum_abs_L"] + stats["sum_abs_E"])
        / max(stats["sum_abs_LplusE"], tiny),
        "F2": 1.0 - stats["sum_sq_E"] / max(stats["sum_sq_D"], tiny),
        "sign_agreement": (
            stats["n_directional_agree"] / stats["n_directional"]
            if stats["n_directional"]
            else float("nan")
        ),
        "max_identity_error": stats["max_identity_error"],
    }


def analyze(
    gradient_path: Path,
    beta1: float,
    beta2: float,
    windows: tuple[int, ...],
    burn_in: int,
    epsilon: float,
    chunk_size: int,
    coordinates: int | None,
    coordinate_seed: int,
) -> list[dict[str, float | int]]:
    source = np.load(gradient_path, mmap_mode="r")
    if source.ndim != 2:
        raise ValueError(f"Expected a two-dimensional gradient array, got {source.shape}")
    if coordinates is None or coordinates >= source.shape[1]:
        indices = np.arange(source.shape[1])
    else:
        rng = np.random.default_rng(coordinate_seed)
        indices = np.sort(rng.choice(source.shape[1], coordinates, replace=False))
    stats = {window: new_stats() for window in windows}
    tiny = np.finfo(np.float64).tiny
    for start in range(0, indices.size, chunk_size):
        selected = indices[start : start + chunk_size]
        gradients = np.asarray(source[:, selected], dtype=np.float64)
        first = corrected_ema(gradients, beta1)
        second = corrected_ema(gradients * gradients, beta2)
        log_magnitude = np.log(np.maximum(np.abs(gradients), tiny))
        log_first = corrected_ema(log_magnitude, beta1)
        log_second = corrected_ema(log_magnitude, beta2)
        normalized_update = first / (np.sqrt(second) + epsilon)
        sign = np.sign(gradients)
        lag = sign * (log_first - log_second)
        for window in windows:
            accumulate(
                stats[window],
                normalized_update,
                sign,
                lag,
                sign_stable_mask(gradients, window, burn_in),
            )
    return [finalize(stats[window], window) for window in windows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradients", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beta1", type=float, required=True)
    parser.add_argument("--beta2", type=float, required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=(6, 100, 1000))
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--coordinates",
        type=int,
        default=None,
        help="Optional deterministic subsample for smoke checks; omit for paper results.",
    )
    parser.add_argument("--coordinate-seed", type=int, default=777)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = tuple(sorted(set(args.windows)))
    rows = analyze(
        args.gradients,
        args.beta1,
        args.beta2,
        windows,
        args.burn_in,
        args.epsilon,
        args.chunk_size,
        args.coordinates,
        args.coordinate_seed,
    )
    payload = {
        "gradients": str(args.gradients),
        "gradient_storage_dtype": str(np.load(args.gradients, mmap_mode="r").dtype),
        "analysis_dtype": "float64",
        "beta1": args.beta1,
        "beta2": args.beta2,
        "burn_in": args.burn_in,
        "epsilon": args.epsilon,
        "epsilon_placement": "outside_sqrt",
        "coordinates": args.coordinates if args.coordinates is not None else "all",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
