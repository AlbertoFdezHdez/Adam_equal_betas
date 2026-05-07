from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analyze_direct_R_decomposition_v2 import (
    BETA_PAIRS,
    OUTPUT_DIR,
    SELECTED_JS,
    WINDOW,
    decompose_direct_R,
    selected_previous_ratio,
    shadow_ema_with_initial,
)
from analyze_ema_decomposition_multi_beta_R import decompose_ema_selected
from analyze_probe_ema_decomposition import BASE_DIR, DEFAULT_RUN_DIR, shadow_ema, valid_and_drift, write_json


VALIDATION_DIR = BASE_DIR / "results" / "R_validation" / "discretization_regime"
R_BETA_PAIRS = ((0.9, 0.9), (0.99, 0.99), (0.999, 0.999), (0.9, 0.999))
GROUPS = ("base", "lag", "scale", "residual")
FINE_TERMS = ("base", "lag", "transport", "curvature", "residual")


@dataclass
class RValidationStats:
    n_values: int = 0
    n_nonpositive: int = 0
    sum_abs_target: float = 0.0
    sum_target_sq: float = 0.0
    sum_abs_terms_fine: float = 0.0
    sum_abs_transport: float = 0.0
    sum_abs_curvature: float = 0.0
    sum_abs_scale: float = 0.0
    sum_abs_group: dict[str, float] = field(default_factory=lambda: {group: 0.0 for group in GROUPS})
    residual_sq: float = 0.0

    def add(self, r_terms: dict[str, np.ndarray], target: np.ndarray, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        safe = mask & np.isfinite(target)
        if not np.any(safe):
            return
        target_vals = target[safe]
        base = r_terms["base"][safe]
        lag = r_terms["lag"][safe]
        transport = r_terms["transport"][safe]
        curvature = r_terms["curvature"][safe]
        residual = r_terms["residual"][safe]
        scale = transport + curvature

        self.n_values += int(target_vals.size)
        self.sum_abs_target += float(np.sum(np.abs(target_vals), dtype=np.float64))
        self.sum_target_sq += float(np.sum(target_vals * target_vals, dtype=np.float64))
        self.sum_abs_terms_fine += float(
            np.sum(
                np.abs(base) + np.abs(lag) + np.abs(transport) + np.abs(curvature) + np.abs(residual),
                dtype=np.float64,
            )
        )
        self.sum_abs_transport += float(np.sum(np.abs(transport), dtype=np.float64))
        self.sum_abs_curvature += float(np.sum(np.abs(curvature), dtype=np.float64))
        self.sum_abs_scale += float(np.sum(np.abs(scale), dtype=np.float64))
        self.sum_abs_group["base"] += float(np.sum(np.abs(base), dtype=np.float64))
        self.sum_abs_group["lag"] += float(np.sum(np.abs(lag), dtype=np.float64))
        self.sum_abs_group["scale"] += float(np.sum(np.abs(scale), dtype=np.float64))
        self.sum_abs_group["residual"] += float(np.sum(np.abs(residual), dtype=np.float64))
        self.residual_sq += float(np.sum(residual * residual, dtype=np.float64))

    def row(self, eps: float) -> dict[str, float | int]:
        if self.n_values == 0:
            return {
                "n_values": 0,
                "R_target_abs": 0.0,
                "R_target_l2": 0.0,
                "residual_l1_rel": float("nan"),
                "residual_l2_rel": float("nan"),
                "cancellation_ratio": float("nan"),
                "tc_cancellation": float("nan"),
                "base_group_pct": float("nan"),
                "lag_group_pct": float("nan"),
                "scale_group_pct": float("nan"),
                "residual_group_pct": float("nan"),
            }
        target_l2 = math.sqrt(self.sum_target_sq)
        residual_l2 = math.sqrt(self.residual_sq)
        denom_group = sum(self.sum_abs_group.values())
        return {
            "n_values": self.n_values,
            "R_target_abs": self.sum_abs_target,
            "R_target_l2": target_l2,
            "residual_l1_rel": self.sum_abs_group["residual"] / (self.sum_abs_target + eps),
            "residual_l2_rel": residual_l2 / (target_l2 + eps),
            "cancellation_ratio": self.sum_abs_terms_fine / (self.sum_abs_target + eps),
            "tc_cancellation": (self.sum_abs_transport + self.sum_abs_curvature) / (self.sum_abs_scale + eps),
            "base_group_pct": 100.0 * self.sum_abs_group["base"] / denom_group if denom_group else float("nan"),
            "lag_group_pct": 100.0 * self.sum_abs_group["lag"] / denom_group if denom_group else float("nan"),
            "scale_group_pct": 100.0 * self.sum_abs_group["scale"] / denom_group if denom_group else float("nan"),
            "residual_group_pct": 100.0 * self.sum_abs_group["residual"] / denom_group if denom_group else float("nan"),
        }

    def merge(self, other: "RValidationStats") -> None:
        self.n_values += other.n_values
        self.n_nonpositive += other.n_nonpositive
        self.sum_abs_target += other.sum_abs_target
        self.sum_target_sq += other.sum_target_sq
        self.sum_abs_terms_fine += other.sum_abs_terms_fine
        self.sum_abs_transport += other.sum_abs_transport
        self.sum_abs_curvature += other.sum_abs_curvature
        self.sum_abs_scale += other.sum_abs_scale
        self.residual_sq += other.residual_sq
        for group in GROUPS:
            self.sum_abs_group[group] += other.sum_abs_group[group]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def beta_from_tau_dt(tau: float, dt: float) -> float:
    return float(math.exp(-dt / tau))


def tau_from_beta_dt(beta: float, dt: float = 1.0) -> float:
    return float(-dt / math.log(beta))


def r_terms_for_values(
    values: np.ndarray,
    beta1: float,
    beta2: float,
    *,
    adam_eps: float,
    stationary_init: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    y = values * values
    if stationary_init:
        m_shadow = shadow_ema_with_initial(values, beta1, values[0, :])
        v_shadow = shadow_ema_with_initial(y, beta2, y[0, :])
    else:
        m_shadow = shadow_ema(values, beta1)
        v_shadow = shadow_ema(y, beta2)
    m_raw = decompose_ema_selected(values, m_shadow, beta1, SELECTED_JS)[0]
    v_raw = decompose_ema_selected(y, v_shadow, beta2, SELECTED_JS)[0]
    m_terms = {"base": m_raw["base"], "lag": m_raw["lag"], "transport": m_raw["trans"], "curvature": m_raw["curv"]}
    v_terms = {"base": v_raw["base"], "lag": v_raw["lag"], "transport": v_raw["trans"], "curvature": v_raw["curv"]}
    prev_ratio = selected_previous_ratio(values)
    return decompose_direct_R(m_terms, v_terms, prev_ratio, beta1, beta2, adam_eps)


def full_selected_mask(values: np.ndarray) -> np.ndarray:
    valid, _, _, _ = valid_and_drift(values, window=WINDOW, eps_g=0.0)
    return np.broadcast_to(valid[:, :, None], (valid.shape[0], valid.shape[1], len(SELECTED_JS)))


def evaluate_values(
    values: np.ndarray,
    beta1: float,
    beta2: float,
    *,
    adam_eps: float,
    stationary_init: bool,
    mask: np.ndarray | None = None,
    eps: float = 1e-30,
) -> dict[str, object]:
    r_terms, target, nonpositive = r_terms_for_values(
        values,
        beta1,
        beta2,
        adam_eps=adam_eps,
        stationary_init=stationary_init,
    )
    if mask is None:
        mask = full_selected_mask(values)
    stats = RValidationStats(n_nonpositive=nonpositive)
    stats.add(r_terms, target, mask)
    row = stats.row(eps)
    row["nonpositive_R_denominator"] = nonpositive
    return row


def dimensional_rows() -> list[dict[str, object]]:
    return [
        {
            "term": "R_target",
            "formula": "M_j / sqrt(V_j + eps_adam)",
            "units": "[g] / sqrt([g]^2) = 1",
            "status": "dimensionless",
        },
        {"term": "R_base", "formula": "sign(g_j)", "units": "1", "status": "dimensionless"},
        {
            "term": "R_lag",
            "formula": "sign(g_j) * (ell2 - ell1) * (1 - |g_{j-1}|/|g_j|)",
            "units": "1 * (tau2/dt - tau1/dt) * 1 = 1",
            "status": "dimensionless; ell=beta/(1-beta) is tau/dt in step units",
        },
        {
            "term": "R_transport",
            "formula": "transport_m/|g_j| - 0.5*sign(g_j)*transport_v/g_j^2",
            "units": "[g]/[g] - [g]^2/[g]^2 = 1",
            "status": "dimensionless",
        },
        {
            "term": "R_curvature",
            "formula": "curvature_m/|g_j| - 0.5*sign(g_j)*curvature_v/g_j^2 - 0.5*sign(g_j)*ell2*(1-ratio)^2",
            "units": "[g]/[g] - [g]^2/[g]^2 - (tau2/dt)*1 = 1",
            "status": "dimensionless",
        },
        {
            "term": "R_residual",
            "formula": "R_target - (base + lag + transport + curvature)",
            "units": "1",
            "status": "dimensionless",
        },
    ]


def synthetic_signal(kind: str, dt: float, steps: int, coords: int, *, a: float, b: float = 0.0) -> np.ndarray:
    t = (np.arange(steps, dtype=np.float64) * dt)[:, None]
    offsets = np.linspace(-0.15, 0.15, coords, dtype=np.float64)[None, :]
    if kind == "exp_linear":
        return np.exp(a * t + offsets)
    if kind == "exp_quadratic":
        return np.exp(a * t + b * t * t + offsets)
    raise ValueError(f"unknown synthetic signal kind: {kind}")


def synthetic_dt_convergence(adam_eps: float, eps: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_pairs = ((0.9, 0.9), (0.9, 0.999))
    dts = (1.0, 0.5, 0.25, 0.125)
    physical_horizon = 128.0
    coords = 32
    for kind, a, b in (("exp_linear", 0.002, 0.0), ("exp_quadratic", 0.001, 0.00001)):
        for base_beta1, base_beta2 in base_pairs:
            tau1 = tau_from_beta_dt(base_beta1)
            tau2 = tau_from_beta_dt(base_beta2)
            previous_l2: float | None = None
            previous_dt: float | None = None
            for dt in dts:
                steps = int(physical_horizon / dt) + WINDOW + 8
                values = synthetic_signal(kind, dt, steps, coords, a=a, b=b)
                beta1 = beta_from_tau_dt(tau1, dt)
                beta2 = beta_from_tau_dt(tau2, dt)
                row = evaluate_values(values, beta1, beta2, adam_eps=adam_eps, stationary_init=True, eps=eps)
                residual_l2 = float(row["residual_l2_rel"])
                rate = float("nan")
                if previous_l2 is not None and residual_l2 > 0.0:
                    rate = math.log(previous_l2 / residual_l2) / math.log(previous_dt / dt)  # type: ignore[arg-type]
                rows.append(
                    {
                        "signal": kind,
                        "base_beta1_at_dt1": base_beta1,
                        "base_beta2_at_dt1": base_beta2,
                        "dt": dt,
                        "beta1": beta1,
                        "beta2": beta2,
                        "residual_l1_rel": row["residual_l1_rel"],
                        "residual_l2_rel": residual_l2,
                        "cancellation_ratio": row["cancellation_ratio"],
                        "base_group_pct": row["base_group_pct"],
                        "lag_group_pct": row["lag_group_pct"],
                        "scale_group_pct": row["scale_group_pct"],
                        "residual_group_pct": row["residual_group_pct"],
                        "dt_convergence_rate_l2_vs_previous": rate,
                        "n_values": row["n_values"],
                    }
                )
                previous_l2 = residual_l2
                previous_dt = dt
    return rows


def stationary_tests(adam_eps: float, eps: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    steps = 64
    coords = 32
    t = np.arange(steps, dtype=np.float64)[:, None]
    for beta1, beta2 in R_BETA_PAIRS:
        for value, name in ((1.0, "constant_positive"), (-1.0, "constant_negative")):
            row = evaluate_values(
                np.full((steps, coords), value, dtype=np.float64),
                beta1,
                beta2,
                adam_eps=adam_eps,
                stationary_init=True,
                eps=eps,
            )
            rows.append({"signal": name, "eps_amplitude": 0.0, "beta1": beta1, "beta2": beta2, **row})
        for amp in (1e-3, 1e-2, 1e-1):
            values = 1.0 * (1.0 + amp * np.sin(0.25 * t + np.linspace(0.0, 0.5, coords)[None, :]))
            row = evaluate_values(values, beta1, beta2, adam_eps=adam_eps, stationary_init=True, eps=eps)
            rows.append({"signal": "near_stationary_sine", "eps_amplitude": amp, "beta1": beta1, "beta2": beta2, **row})
    return rows


def window_level_real_masks(values: np.ndarray, p10_abs_g: float) -> dict[str, np.ndarray]:
    valid, _, _, _ = valid_and_drift(values, window=WINDOW, eps_g=0.0)
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=WINDOW + 1, axis=0)
    abs_windows = np.abs(windows)
    logs = np.full_like(abs_windows, np.nan, dtype=np.float64)
    np.log(abs_windows, out=logs, where=abs_windows > 0.0)
    delta = logs[:, :, 1:] - logs[:, :, :-1]
    second_delta = delta[:, :, 1:] - delta[:, :, :-1]
    min_abs = np.min(abs_windows, axis=-1)
    masks_2d = {
        "A_sign_stable": valid,
        "max_abs_delta_lt_1e-2": valid & (np.max(np.abs(delta), axis=-1) < 1e-2),
        "max_abs_delta_lt_1e-3": valid & (np.max(np.abs(delta), axis=-1) < 1e-3),
        "max_abs_second_delta_lt_1e-3": valid & (np.max(np.abs(second_delta), axis=-1) < 1e-3),
        "delta_lt_1e-2_second_lt_1e-3_abs_g_gt_p10": (
            valid
            & (np.max(np.abs(delta), axis=-1) < 1e-2)
            & (np.max(np.abs(second_delta), axis=-1) < 1e-3)
            & (min_abs > p10_abs_g)
        ),
    }
    return {
        name: np.broadcast_to(mask[:, :, None], (mask.shape[0], mask.shape[1], len(SELECTED_JS)))
        for name, mask in masks_2d.items()
    }


def real_regime_bins(args: argparse.Namespace, eps: float) -> list[dict[str, object]]:
    run_dir = args.run_dir.resolve()
    gradients = np.load(run_dir / "gradients.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    coord_indices = np.sort(
        rng.choice(gradients.shape[1], size=min(args.coord_sample_size, gradients.shape[1]), replace=False)
    )
    sampled = np.asarray(gradients[:, coord_indices], dtype=np.float64)
    nonzero_abs = np.abs(sampled[np.nonzero(sampled)])
    p10_abs_g = float(np.percentile(nonzero_abs, 10)) if nonzero_abs.size else 0.0

    total_coord_windows = int((sampled.shape[0] - WINDOW) * sampled.shape[1])
    stats_by_key = {
        (bin_name, beta1, beta2): RValidationStats()
        for bin_name in (
            "A_sign_stable",
            "max_abs_delta_lt_1e-2",
            "max_abs_delta_lt_1e-3",
            "max_abs_second_delta_lt_1e-3",
            "delta_lt_1e-2_second_lt_1e-3_abs_g_gt_p10",
        )
        for beta1, beta2 in R_BETA_PAIRS
    }
    nonpositive_by_pair = {pair: 0 for pair in R_BETA_PAIRS}
    for start in range(0, sampled.shape[1], args.chunk_size):
        stop = min(start + args.chunk_size, sampled.shape[1])
        values = sampled[:, start:stop]
        masks = window_level_real_masks(values, p10_abs_g)
        for beta1, beta2 in R_BETA_PAIRS:
            r_terms, target, nonpositive = r_terms_for_values(
                values,
                beta1,
                beta2,
                adam_eps=args.adam_eps,
                stationary_init=False,
            )
            nonpositive_by_pair[(beta1, beta2)] += int(nonpositive)
            for bin_name, mask in masks.items():
                stats = RValidationStats(n_nonpositive=nonpositive)
                stats.add(r_terms, target, mask)
                stats_by_key[(bin_name, beta1, beta2)].merge(stats)
        print(f"Processed real-bin coordinates {start}:{stop} / {sampled.shape[1]}", flush=True)

    final_rows: list[dict[str, object]] = []
    for beta1, beta2 in R_BETA_PAIRS:
        for bin_name in (
            "A_sign_stable",
            "max_abs_delta_lt_1e-2",
            "max_abs_delta_lt_1e-3",
            "max_abs_second_delta_lt_1e-3",
            "delta_lt_1e-2_second_lt_1e-3_abs_g_gt_p10",
        ):
            stats = stats_by_key[(bin_name, beta1, beta2)]
            final_rows.append(
                {
                    "bin": bin_name,
                    "beta1": beta1,
                    "beta2": beta2,
                    "p10_abs_g_threshold": p10_abs_g,
                    "total_coord_windows": total_coord_windows,
                    "valid_regime_fraction": (stats.n_values / len(SELECTED_JS)) / total_coord_windows,
                    "nonpositive_R_denominator": nonpositive_by_pair[(beta1, beta2)],
                    **stats.row(eps),
                }
            )
    return final_rows


def summarize_status(dt_rows: list[dict[str, object]], stationary_rows: list[dict[str, object]], real_rows: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    constant_rows = [row for row in stationary_rows if str(row["signal"]).startswith("constant")]
    constants_pass = all(float(row["base_group_pct"]) > 99.999 and float(row["residual_l1_rel"]) < 1e-7 for row in constant_rows)
    lines.append(f"- Stationary constant test: {'passed' if constants_pass else 'failed'}; base is expected to explain R.")

    tiny_rows = [row for row in stationary_rows if row["signal"] == "near_stationary_sine" and float(row["eps_amplitude"]) == 1e-3]
    tiny_base_min = min(float(row["base_group_pct"]) for row in tiny_rows)
    tiny_res_max = max(float(row["residual_l1_rel"]) for row in tiny_rows)
    lines.append(
        f"- Near-stationary eps=1e-3: minimum base grouped share is {tiny_base_min:.3g}%, "
        f"maximum residual_l1_rel is {tiny_res_max:.3g}."
    )

    last_dt_rows = [row for row in dt_rows if float(row["dt"]) == 0.125]
    rates = [float(row["dt_convergence_rate_l2_vs_previous"]) for row in last_dt_rows if np.isfinite(float(row["dt_convergence_rate_l2_vs_previous"]))]
    if rates:
        lines.append(f"- Synthetic dt refinement: final-step L2 convergence rates range from {min(rates):.3g} to {max(rates):.3g}.")

    tight_rows = [
        row
        for row in real_rows
        if row["bin"] == "delta_lt_1e-2_second_lt_1e-3_abs_g_gt_p10" and int(row["n_values"]) > 0
    ]
    if tight_rows:
        best = min(tight_rows, key=lambda row: float(row["residual_l2_rel"]))
        lines.append(
            "- Real tight bin: best residual_l2_rel is "
            f"{float(best['residual_l2_rel']):.3g} at beta=({float(best['beta1']):g},{float(best['beta2']):g}); "
            f"valid fraction {100.0 * float(best['valid_regime_fraction']):.3g}%."
        )
    else:
        lines.append("- Real tight bin: no sampled coordinate-window survives the combined drift/curvature/abs(g) filter.")
    lines.append(
        "- Decision rule: R should not be used as an additive budget unless residuals are small, dt refinement decreases the residual, and the near-stationary regime is base-dominated."
    )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R discretization/units versus regime limitations.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=VALIDATION_DIR)
    parser.add_argument("--coord-sample-size", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--eps", type=float, default=1e-30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dim_rows = dimensional_rows()
    stationary_rows = stationary_tests(args.adam_eps, args.eps)
    dt_rows = synthetic_dt_convergence(args.adam_eps, args.eps)
    real_rows = real_regime_bins(args, args.eps)

    dim_csv = output_dir / "table_K1_R_dimensional_consistency.csv"
    stationary_csv = output_dir / "table_K2_R_stationary_and_near_stationary_tests.csv"
    dt_csv = output_dir / "table_K3_R_synthetic_dt_convergence.csv"
    real_csv = output_dir / "table_K4_R_real_regime_bins.csv"
    summary_md = output_dir / "R_validation_discretization_regime_summary.md"
    write_csv(dim_csv, dim_rows)
    write_csv(stationary_csv, stationary_rows)
    write_csv(dt_csv, dt_rows)
    write_csv(real_csv, real_rows)

    status_lines = summarize_status(dt_rows, stationary_rows, real_rows)
    summary = "\n".join(
        [
            "# R Discretization and Regime Validation",
            "",
            "This validation treats the previous grouped R table as a negative diagnostic, not as the main paper result.",
            "",
            "## Formula and units",
            "",
            "`R_target = M_j / sqrt(V_j + eps_adam)` is dimensionless. The implemented terms are also dimensionless: `ell=beta/(1-beta)` plays the role of `tau/dt` in step units, while all finite ratios such as `|g_{j-1}|/|g_j|` are dimensionless.",
            "",
            "## Outputs",
            "",
            f"- Dimensional consistency: `{dim_csv}`",
            f"- Stationary and near-stationary tests: `{stationary_csv}`",
            f"- Synthetic dt convergence: `{dt_csv}`",
            f"- Real-data validity bins: `{real_csv}`",
            "",
            "## Automatic reading",
            "",
            *status_lines,
            "",
            "## Recommendation",
            "",
            "Keep the additive budgets for M and V. For R, report only residual/cancellation/regime diagnostics unless the synthetic and real-regime checks clearly support a small residual.",
        ]
    )
    summary_md.write_text(summary, encoding="utf-8")
    write_json(
        output_dir / "R_validation_discretization_regime_summary.json",
        {
            "dimensional_consistency_csv": str(dim_csv),
            "stationary_tests_csv": str(stationary_csv),
            "dt_convergence_csv": str(dt_csv),
            "real_regime_bins_csv": str(real_csv),
            "summary_md": str(summary_md),
            "coord_sample_size": args.coord_sample_size,
            "selected_j": list(SELECTED_JS),
            "window": WINDOW,
            "adam_eps": args.adam_eps,
            "beta_pairs": [list(pair) for pair in R_BETA_PAIRS],
        },
    )
    print(summary)
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
