from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analyze_direct_R_decomposition_v2 import (
    BETA_PAIRS,
    OUTPUT_DIR,
    SELECTED_JS,
    TERMS,
    WINDOW,
    decompose_direct_R,
    decompose_ema_selected,
    selected_previous_ratio,
)
from analyze_probe_ema_decomposition import DEFAULT_RUN_DIR, shadow_ema, valid_and_drift, write_json


GROUPS = ("base", "lag", "scale", "residual")
FINE_TERMS = ("base", "lag", "transport", "curvature", "residual")


@dataclass
class GroupStats:
    n_values: int = 0
    sum_abs_target: float = 0.0
    sum_target_sq: float = 0.0
    sum_target: float = 0.0
    sum_abs_group: dict[str, float] = field(default_factory=lambda: {group: 0.0 for group in GROUPS})
    sum_group: dict[str, float] = field(default_factory=lambda: {group: 0.0 for group in GROUPS})
    sum_group_sq: dict[str, float] = field(default_factory=lambda: {group: 0.0 for group in GROUPS})
    sum_group_target: dict[str, float] = field(default_factory=lambda: {group: 0.0 for group in GROUPS})
    sum_abs_fine: dict[str, float] = field(default_factory=lambda: {term: 0.0 for term in FINE_TERMS})
    residual_sq: float = 0.0

    def add(self, r_terms: dict[str, np.ndarray], target: np.ndarray, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        target_vals = target[mask]
        finite = np.isfinite(target_vals)
        if not np.any(finite):
            return
        target_vals = target_vals[finite]
        fine = {term: r_terms[term][mask][finite] for term in FINE_TERMS}
        grouped = {
            "base": fine["base"],
            "lag": fine["lag"],
            "scale": fine["transport"] + fine["curvature"],
            "residual": fine["residual"],
        }
        self.n_values += int(target_vals.size)
        self.sum_abs_target += float(np.sum(np.abs(target_vals), dtype=np.float64))
        self.sum_target += float(np.sum(target_vals, dtype=np.float64))
        self.sum_target_sq += float(np.sum(target_vals * target_vals, dtype=np.float64))
        self.residual_sq += float(np.sum(grouped["residual"] * grouped["residual"], dtype=np.float64))
        for group in GROUPS:
            vals = grouped[group]
            self.sum_abs_group[group] += float(np.sum(np.abs(vals), dtype=np.float64))
            self.sum_group[group] += float(np.sum(vals, dtype=np.float64))
            self.sum_group_sq[group] += float(np.sum(vals * vals, dtype=np.float64))
            self.sum_group_target[group] += float(np.sum(vals * target_vals, dtype=np.float64))
        for term in FINE_TERMS:
            self.sum_abs_fine[term] += float(np.sum(np.abs(fine[term]), dtype=np.float64))

    def corr(self, group: str, eps: float) -> float:
        n = max(self.n_values, 1)
        group_mean = self.sum_group[group] / n
        target_mean = self.sum_target / n
        cov = self.sum_group_target[group] - n * group_mean * target_mean
        var_group = self.sum_group_sq[group] - n * group_mean * group_mean
        var_target = self.sum_target_sq - n * target_mean * target_mean
        return cov / np.sqrt(max(var_group, 0.0) * max(var_target, 0.0) + eps)

    def row(self, beta1: float, beta2: float, n_coordinates: int, n_windows: int, n_valid_windows: int, eps: float) -> dict[str, object]:
        denom_group = sum(self.sum_abs_group.values())
        fine_denom = sum(self.sum_abs_fine.values())
        target_l2 = float(np.sqrt(self.sum_target_sq))
        residual_l2 = float(np.sqrt(self.residual_sq))
        row: dict[str, object] = {
            "beta1": beta1,
            "beta2": beta2,
            "R_target_abs": self.sum_abs_target,
            "R_target_l2": target_l2,
            "base_group_pct": 100.0 * self.sum_abs_group["base"] / denom_group if denom_group else float("nan"),
            "lag_group_pct": 100.0 * self.sum_abs_group["lag"] / denom_group if denom_group else float("nan"),
            "scale_group_pct": 100.0 * self.sum_abs_group["scale"] / denom_group if denom_group else float("nan"),
            "residual_group_pct": 100.0 * self.sum_abs_group["residual"] / denom_group if denom_group else float("nan"),
            "residual_l1_rel": self.sum_abs_group["residual"] / (self.sum_abs_target + eps),
            "residual_l2_rel": residual_l2 / (target_l2 + eps),
            "total_cancellation": fine_denom / (self.sum_abs_target + eps),
            "tc_cancellation": (
                (self.sum_abs_fine["transport"] + self.sum_abs_fine["curvature"])
                / (self.sum_abs_group["scale"] + eps)
            ),
            "proj_base": self.sum_group_target["base"] / (self.sum_target_sq + eps),
            "proj_lag": self.sum_group_target["lag"] / (self.sum_target_sq + eps),
            "proj_scale": self.sum_group_target["scale"] / (self.sum_target_sq + eps),
            "proj_residual": self.sum_group_target["residual"] / (self.sum_target_sq + eps),
            "corr_base_R": self.corr("base", eps),
            "corr_lag_R": self.corr("lag", eps),
            "corr_scale_R": self.corr("scale", eps),
            "corr_residual_R": self.corr("residual", eps),
            "sum_abs_base_group": self.sum_abs_group["base"],
            "sum_abs_lag_group": self.sum_abs_group["lag"],
            "sum_abs_scale_group": self.sum_abs_group["scale"],
            "sum_abs_residual_group": self.sum_abs_group["residual"],
            "sum_abs_base_fine": self.sum_abs_fine["base"],
            "sum_abs_lag_fine": self.sum_abs_fine["lag"],
            "sum_abs_transport_fine": self.sum_abs_fine["transport"],
            "sum_abs_curvature_fine": self.sum_abs_fine["curvature"],
            "sum_abs_residual_fine": self.sum_abs_fine["residual"],
            "n_coordinates": n_coordinates,
            "n_windows": n_windows,
            "n_valid_windows": n_valid_windows,
            "n_values": self.n_values,
        }
        return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped R table: base, lag, scale=transport+curvature, residual.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--coord-sample-size", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--eps", type=float, default=1e-30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    gradients = np.load(run_dir / "gradients.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    coord_indices = np.sort(
        rng.choice(gradients.shape[1], size=min(args.coord_sample_size, gradients.shape[1]), replace=False)
    )

    stats = {pair: GroupStats() for pair in BETA_PAIRS}
    unique_betas = tuple(sorted({beta for pair in BETA_PAIRS for beta in pair}))
    n_valid_windows = 0
    n_windows = int(gradients.shape[0] - WINDOW)

    for start in range(0, coord_indices.size, args.chunk_size):
        stop = min(start + args.chunk_size, coord_indices.size)
        chunk = np.asarray(gradients[:, coord_indices[start:stop]], dtype=np.float64)
        y = chunk * chunk
        valid, _, _, _ = valid_and_drift(chunk, window=WINDOW, eps_g=0.0)
        n_valid_windows += int(np.count_nonzero(valid))
        mask = np.broadcast_to(valid[:, :, None], (valid.shape[0], valid.shape[1], len(SELECTED_JS)))
        prev_ratio = selected_previous_ratio(chunk)

        m_by_beta = {}
        v_by_beta = {}
        for beta in unique_betas:
            m_by_beta[beta] = decompose_ema_selected(chunk, shadow_ema(chunk, beta), beta, SELECTED_JS)[0]
            v_by_beta[beta] = decompose_ema_selected(y, shadow_ema(y, beta), beta, SELECTED_JS)[0]

        for beta1, beta2 in BETA_PAIRS:
            m_raw = m_by_beta[beta1]
            v_raw = v_by_beta[beta2]
            m_terms = {"base": m_raw["base"], "lag": m_raw["lag"], "transport": m_raw["trans"], "curvature": m_raw["curv"]}
            v_terms = {"base": v_raw["base"], "lag": v_raw["lag"], "transport": v_raw["trans"], "curvature": v_raw["curv"]}
            r_terms, target, _ = decompose_direct_R(m_terms, v_terms, prev_ratio, beta1, beta2, args.adam_eps)
            stats[(beta1, beta2)].add(r_terms, target, mask)
        print(f"Processed analyzed coordinates {start}:{stop} / {coord_indices.size}", flush=True)

    rows = [
        stats[(beta1, beta2)].row(beta1, beta2, int(coord_indices.size), n_windows, n_valid_windows, args.eps)
        for beta1, beta2 in BETA_PAIRS
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped_csv = output_dir / "table_J_R_grouped_budget.csv"
    fine_debug_csv = output_dir / "table_J_R_grouped_budget_with_fine_debug.csv"
    write_csv(grouped_csv, [
        {
            k: v
            for k, v in row.items()
            if k
            in {
                "beta1",
                "beta2",
                "base_group_pct",
                "lag_group_pct",
                "scale_group_pct",
                "residual_group_pct",
                "residual_l1_rel",
                "residual_l2_rel",
                "total_cancellation",
                "tc_cancellation",
                "proj_base",
                "proj_lag",
                "proj_scale",
                "proj_residual",
                "corr_base_R",
                "corr_lag_R",
                "corr_scale_R",
                "corr_residual_R",
                "R_target_abs",
                "R_target_l2",
                "n_coordinates",
                "n_windows",
                "n_valid_windows",
            }
        }
        for row in rows
    ])
    write_csv(fine_debug_csv, rows)
    write_json(
        output_dir / "R_grouped_budget_summary.json",
        {
            "grouped_csv": str(grouped_csv),
            "fine_debug_csv": str(fine_debug_csv),
            "coord_sample_size": int(coord_indices.size),
            "n_windows": n_windows,
            "n_valid_windows": int(n_valid_windows),
            "selected_j": list(SELECTED_JS),
            "adam_eps": args.adam_eps,
        },
    )
    print("# Grouped R budget")
    for row in rows:
        print(
            f"beta=({row['beta1']:g},{row['beta2']:g}) "
            f"base={row['base_group_pct']:.3g}% lag={row['lag_group_pct']:.3g}% "
            f"scale={row['scale_group_pct']:.3g}% residual={row['residual_group_pct']:.3g}% "
            f"res_l1={row['residual_l1_rel']:.3g} cancel={row['total_cancellation']:.3g} "
            f"TC={row['tc_cancellation']:.3g}"
        )
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
