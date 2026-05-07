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
from analyze_probe_ema_decomposition import DEFAULT_RUN_DIR, Reservoir, shadow_ema, stable_seed, valid_and_drift, write_json


R_TERMS = ("base", "lag", "transport", "curvature", "residual")
DRIFT_THRESHOLDS = {
    "tiny_drift_abs_lt_0p01": 0.01,
    "very_small_drift_abs_lt_0p1": 0.1,
    "small_drift_abs_lt_log_sqrt2": float(np.log(np.sqrt(2.0))),
}


@dataclass
class TermMoment:
    sum: float = 0.0
    sum_sq: float = 0.0
    sum_abs: float = 0.0
    sum_target_dot: float = 0.0
    sample_abs: Reservoir | None = None


@dataclass
class RDiagnostic:
    terms: dict[str, TermMoment]
    n: int = 0
    sum_target: float = 0.0
    sum_target_sq: float = 0.0
    sum_abs_target: float = 0.0
    sum_abs_transport_plus_curvature: float = 0.0

    @classmethod
    def create(cls, sample_size: int, seed: int, *parts: object) -> "RDiagnostic":
        terms = {
            term: TermMoment(sample_abs=Reservoir(sample_size, np.random.default_rng(stable_seed(seed, *parts, term))))
            for term in R_TERMS
        }
        return cls(terms=terms)

    def add(self, r_terms: dict[str, np.ndarray], target: np.ndarray, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        target_vals = target[mask]
        finite = np.isfinite(target_vals)
        if not np.any(finite):
            return
        target_vals = target_vals[finite]
        term_vals = {term: r_terms[term][mask][finite] for term in R_TERMS}
        n = int(target_vals.size)
        self.n += n
        self.sum_target += float(np.sum(target_vals, dtype=np.float64))
        self.sum_target_sq += float(np.sum(target_vals * target_vals, dtype=np.float64))
        self.sum_abs_target += float(np.sum(np.abs(target_vals), dtype=np.float64))
        tc = term_vals["transport"] + term_vals["curvature"]
        self.sum_abs_transport_plus_curvature += float(np.sum(np.abs(tc), dtype=np.float64))
        for term in R_TERMS:
            vals = term_vals[term]
            moment = self.terms[term]
            moment.sum += float(np.sum(vals, dtype=np.float64))
            moment.sum_sq += float(np.sum(vals * vals, dtype=np.float64))
            moment.sum_abs += float(np.sum(np.abs(vals), dtype=np.float64))
            moment.sum_target_dot += float(np.sum(vals * target_vals, dtype=np.float64))
            if moment.sample_abs is not None and term in {"base", "transport", "curvature"}:
                moment.sample_abs.add(np.abs(vals))

    def row(self, quantity: str, beta1: float, beta2: float, n_coordinates: int, n_windows: int, n_valid_windows: int, eps: float) -> dict[str, object]:
        sum_abs_terms = sum(self.terms[term].sum_abs for term in R_TERMS)
        row: dict[str, object] = {
            "quantity": quantity,
            "beta1": beta1,
            "beta2": beta2,
            "base_pct_abs": 100.0 * self.terms["base"].sum_abs / sum_abs_terms if sum_abs_terms else float("nan"),
            "lag_pct_abs": 100.0 * self.terms["lag"].sum_abs / sum_abs_terms if sum_abs_terms else float("nan"),
            "transport_pct_abs": 100.0 * self.terms["transport"].sum_abs / sum_abs_terms if sum_abs_terms else float("nan"),
            "curvature_pct_abs": 100.0 * self.terms["curvature"].sum_abs / sum_abs_terms if sum_abs_terms else float("nan"),
            "residual_pct_abs": 100.0 * self.terms["residual"].sum_abs / sum_abs_terms if sum_abs_terms else float("nan"),
            "sum_abs_target": self.sum_abs_target,
            "sum_abs_terms": sum_abs_terms,
            "cancellation_ratio": sum_abs_terms / (self.sum_abs_target + eps),
            "cancellation_transport_curvature": (
                (self.terms["transport"].sum_abs + self.terms["curvature"].sum_abs)
                / (self.sum_abs_transport_plus_curvature + eps)
            ),
            "sum_abs_transport_plus_curvature": self.sum_abs_transport_plus_curvature,
            "n_coordinates": n_coordinates,
            "n_windows": n_windows,
            "n_valid_windows": n_valid_windows,
            "n_values": self.n,
        }
        for term in R_TERMS:
            moment = self.terms[term]
            row[f"signed_{term}"] = moment.sum_target_dot / (self.sum_target_sq + eps)
            term_mean = moment.sum / max(self.n, 1)
            target_mean = self.sum_target / max(self.n, 1)
            cov = moment.sum_target_dot - self.n * term_mean * target_mean
            var_term = moment.sum_sq - self.n * term_mean * term_mean
            var_target = self.sum_target_sq - self.n * target_mean * target_mean
            row[f"corr_{term}"] = cov / np.sqrt(max(var_term, 0.0) * max(var_target, 0.0) + eps)
            row[f"net_{term}_pct"] = 100.0 * abs(moment.sum) / (self.sum_abs_target + eps)
            row[f"net_{term}_sign"] = 1 if moment.sum > 0 else (-1 if moment.sum < 0 else 0)
        for term in ("base", "transport", "curvature"):
            sample = self.terms[term].sample_abs.sample() if self.terms[term].sample_abs else np.empty(0)
            if sample.size:
                q = np.percentile(sample, [50, 90, 99, 99.9])
            else:
                q = [float("nan")] * 4
            row[f"median_abs_{term}"] = float(q[0])
            row[f"q90_abs_{term}"] = float(q[1])
            row[f"q99_abs_{term}"] = float(q[2])
            row[f"q999_abs_{term}"] = float(q[3])
        return row


@dataclass
class DiffDiagnostic:
    sum_abs_diff: dict[str, float] = field(default_factory=lambda: {term: 0.0 for term in (*R_TERMS, "target_R")})
    sum_abs_ref: dict[str, float] = field(default_factory=lambda: {term: 0.0 for term in (*R_TERMS, "target_R")})
    n: int = 0

    def add(self, left_terms: dict[str, np.ndarray], left_target: np.ndarray, right_terms: dict[str, np.ndarray], right_target: np.ndarray, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        self.n += int(np.count_nonzero(mask))
        for term in R_TERMS:
            lv = left_terms[term][mask]
            rv = right_terms[term][mask]
            self.sum_abs_diff[term] += float(np.sum(np.abs(lv - rv), dtype=np.float64))
            self.sum_abs_ref[term] += float(np.sum(np.abs(lv), dtype=np.float64))
        self.sum_abs_diff["target_R"] += float(np.sum(np.abs(left_target[mask] - right_target[mask]), dtype=np.float64))
        self.sum_abs_ref["target_R"] += float(np.sum(np.abs(left_target[mask]), dtype=np.float64))

    def rows(self, eps: float) -> list[dict[str, object]]:
        rows = []
        for term in (*R_TERMS, "target_R"):
            rows.append(
                {
                    "comparison": "(0.999,0.999) - (0.9,0.999)",
                    "term": term,
                    "mean_abs_diff": self.sum_abs_diff[term] / max(self.n, 1),
                    "relative_abs_diff": self.sum_abs_diff[term] / (self.sum_abs_ref[term] + eps),
                    "n_values": self.n,
                }
            )
        return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def estimate_abs_g_thresholds(
    gradients: np.ndarray,
    coord_indices: np.ndarray,
    chunk_size: int,
    sample_size: int,
    seed: int,
) -> dict[str, float]:
    reservoir = Reservoir(sample_size, np.random.default_rng(seed))
    for start in range(0, coord_indices.size, chunk_size):
        stop = min(start + chunk_size, coord_indices.size)
        chunk = np.asarray(gradients[:, coord_indices[start:stop]], dtype=np.float64)
        valid, _, _, _ = valid_and_drift(chunk, window=6, eps_g=0.0)
        windows = np.lib.stride_tricks.sliding_window_view(chunk, window_shape=7, axis=0)
        g_selected = windows[:, :, 1:][:, :, SELECTED_JS]
        mask = np.broadcast_to(valid[:, :, None], g_selected.shape)
        reservoir.add(np.abs(g_selected[mask]))
    sample = reservoir.sample()
    return {
        "p1": float(np.percentile(sample, 1)),
        "p5": float(np.percentile(sample, 5)),
        "p10": float(np.percentile(sample, 10)),
    }


def compute_diagnostics(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    run_dir = args.run_dir.resolve()
    gradients = np.load(run_dir / "gradients.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    coord_indices = np.sort(
        rng.choice(gradients.shape[1], size=min(args.coord_sample_size, gradients.shape[1]), replace=False)
    )
    abs_g_thresholds = estimate_abs_g_thresholds(
        gradients, coord_indices, args.chunk_size, args.threshold_sample_size, args.seed + 1
    )

    groups: dict[tuple[str, str, float, float], RDiagnostic] = {}
    for group_type, group_name in [("overall", "all")]:
        for beta1, beta2 in BETA_PAIRS:
            groups[(group_type, group_name, beta1, beta2)] = RDiagnostic.create(
                args.sample_size, args.seed, group_type, group_name, beta1, beta2
            )
    for name in DRIFT_THRESHOLDS:
        for beta1, beta2 in BETA_PAIRS:
            groups[("drift", name, beta1, beta2)] = RDiagnostic.create(args.sample_size, args.seed, "drift", name, beta1, beta2)
    for name in abs_g_thresholds:
        for beta1, beta2 in BETA_PAIRS:
            groups[("abs_g_filter", f"abs_g_ge_{name}", beta1, beta2)] = RDiagnostic.create(
                args.sample_size, args.seed, "abs_g_filter", name, beta1, beta2
            )

    unique_betas = tuple(sorted({beta for pair in BETA_PAIRS for beta in pair}))
    n_valid_windows = 0
    diff_diag = DiffDiagnostic()
    n_windows = int(gradients.shape[0] - WINDOW)

    for start in range(0, coord_indices.size, args.chunk_size):
        stop = min(start + args.chunk_size, coord_indices.size)
        chunk = np.asarray(gradients[:, coord_indices[start:stop]], dtype=np.float64)
        y = chunk * chunk
        valid, _, _, _ = valid_and_drift(chunk, window=WINDOW, eps_g=0.0)
        n_valid_windows += int(np.count_nonzero(valid))
        base_mask = np.broadcast_to(valid[:, :, None], (valid.shape[0], valid.shape[1], len(SELECTED_JS)))
        windows = np.lib.stride_tricks.sliding_window_view(chunk, window_shape=WINDOW + 1, axis=0)
        g_prev = windows[:, :, :-1][:, :, SELECTED_JS]
        g_cur = windows[:, :, 1:][:, :, SELECTED_JS]
        with np.errstate(divide="ignore", invalid="ignore"):
            drift_abs = np.abs(np.log(np.abs(g_cur)) - np.log(np.abs(g_prev)))
        abs_g = np.abs(g_cur)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_prev_ratio = np.abs(g_prev) / np.abs(g_cur)
        prev_ratio = np.where(np.isfinite(raw_prev_ratio), raw_prev_ratio, 0.0)

        m_by_beta = {}
        v_by_beta = {}
        for beta in unique_betas:
            m_by_beta[beta] = decompose_ema_selected(chunk, shadow_ema(chunk, beta), beta, SELECTED_JS)[0]
            v_by_beta[beta] = decompose_ema_selected(y, shadow_ema(y, beta), beta, SELECTED_JS)[0]

        r_cache: dict[tuple[float, float], tuple[dict[str, np.ndarray], np.ndarray]] = {}
        for beta1, beta2 in BETA_PAIRS:
            m_raw = m_by_beta[beta1]
            v_raw = v_by_beta[beta2]
            m_terms = {"base": m_raw["base"], "lag": m_raw["lag"], "transport": m_raw["trans"], "curvature": m_raw["curv"]}
            v_terms = {"base": v_raw["base"], "lag": v_raw["lag"], "transport": v_raw["trans"], "curvature": v_raw["curv"]}
            r_terms, target, _ = decompose_direct_R(m_terms, v_terms, prev_ratio, beta1, beta2, args.adam_eps)
            r_cache[(beta1, beta2)] = (r_terms, target)

            groups[("overall", "all", beta1, beta2)].add(r_terms, target, base_mask)
            for name, threshold in DRIFT_THRESHOLDS.items():
                groups[("drift", name, beta1, beta2)].add(r_terms, target, base_mask & (drift_abs < threshold))
            for name, threshold in abs_g_thresholds.items():
                groups[("abs_g_filter", f"abs_g_ge_{name}", beta1, beta2)].add(
                    r_terms, target, base_mask & (abs_g >= threshold)
                )

        left_terms, left_target = r_cache[(0.999, 0.999)]
        right_terms, right_target = r_cache[(0.9, 0.999)]
        diff_diag.add(left_terms, left_target, right_terms, right_target, base_mask)
        print(f"Processed analyzed coordinates {start}:{stop} / {coord_indices.size}", flush=True)

    rows = []
    for (group_type, group_name, beta1, beta2), diagnostic in groups.items():
        row = diagnostic.row("R", beta1, beta2, int(coord_indices.size), n_windows, n_valid_windows, args.eps)
        row["group_type"] = group_type
        row["group"] = group_name
        rows.append(row)

    primary = [row for row in rows if row["group_type"] == "overall" and row["group"] == "all"]
    summary = {
        "coord_sample_size": int(coord_indices.size),
        "seed": args.seed,
        "n_windows": n_windows,
        "n_valid_windows": int(n_valid_windows),
        "selected_j": list(SELECTED_JS),
        "adam_eps": args.adam_eps,
        "abs_g_thresholds": abs_g_thresholds,
    }
    return primary, rows, diff_diag.rows(args.eps), summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R cancellation diagnostics for direct R v2 decomposition.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--coord-sample-size", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--threshold-sample-size", type=int, default=300_000)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--eps", type=float, default=1e-30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_rows, group_rows, diff_rows, summary = compute_diagnostics(args)

    primary_path = output_dir / "table_I_R_cancellation_diagnostics.csv"
    group_path = output_dir / "table_I_R_cancellation_by_group.csv"
    diff_path = output_dir / "table_I_R_beta1_sensitivity.csv"
    write_csv(primary_path, primary_rows)
    write_csv(group_path, group_rows)
    write_csv(diff_path, diff_rows)

    lines = [
        "# R Cancellation Diagnostics",
        "",
        f"- Coordinates analyzed: `{summary['coord_sample_size']}`.",
        f"- Sign-stable windows: `{summary['n_valid_windows']}`.",
        f"- `|g|` thresholds: {summary['abs_g_thresholds']}.",
        "",
        "## Overall cancellation ratios",
        "",
    ]
    for row in primary_rows:
        lines.append(
            f"- beta=({float(row['beta1']):g},{float(row['beta2']):g}): "
            f"cancellation_ratio={float(row['cancellation_ratio']):.3g}, "
            f"TC_cancel={float(row['cancellation_transport_curvature']):.3g}, "
            f"corr_transport={float(row['corr_transport']):.3g}, "
            f"corr_curvature={float(row['corr_curvature']):.3g}"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{primary_path.name}`",
            f"- `{group_path.name}`",
            f"- `{diff_path.name}`",
        ]
    )
    report_path = output_dir / "R_cancellation_diagnostics_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    write_json(output_dir / "R_cancellation_diagnostics_summary.json", summary)
    print("\n".join(lines))
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
