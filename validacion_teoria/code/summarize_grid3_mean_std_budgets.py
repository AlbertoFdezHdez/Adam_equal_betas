from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analyze_direct_R_decomposition_v2 import OUTPUT_DIR, SELECTED_JS, WINDOW, selected_previous_ratio
from analyze_ema_decomposition_multi_beta_R import decompose_ema_selected
from analyze_probe_ema_decomposition import DEFAULT_RUN_DIR, shadow_ema, valid_and_drift, write_json


BETA_VALUES = (0.9, 0.99, 0.999)
BETA_PAIRS = tuple((beta1, beta2) for beta1 in BETA_VALUES for beta2 in BETA_VALUES)
MV_TERMS = ("base", "lag", "transition", "curvature", "residual")
R_TERMS = ("S", "L", "E")


@dataclass
class ShareStats:
    terms: tuple[str, ...]
    n_values: int = 0
    sum_share: dict[str, float] = field(init=False)
    sum_share_sq: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.sum_share = {term: 0.0 for term in self.terms}
        self.sum_share_sq = {term: 0.0 for term in self.terms}

    def add(self, shares: dict[str, np.ndarray], mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        n = int(np.count_nonzero(mask))
        self.n_values += n
        for term, values in shares.items():
            vals = values[mask]
            self.sum_share[term] += float(np.sum(vals, dtype=np.float64))
            self.sum_share_sq[term] += float(np.sum(vals * vals, dtype=np.float64))

    def mean(self, term: str) -> float:
        return self.sum_share[term] / self.n_values if self.n_values else float("nan")

    def std(self, term: str) -> float:
        if self.n_values == 0:
            return float("nan")
        mean = self.mean(term)
        var = max(self.sum_share_sq[term] / self.n_values - mean * mean, 0.0)
        return math.sqrt(var)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pm(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f} +- {100.0 * std:.2f}%"


def mv_share_arrays(raw_terms: dict[str, np.ndarray], obs: np.ndarray, recon: np.ndarray, eps: float) -> dict[str, np.ndarray]:
    residual = obs - recon
    abs_terms = {
        "base": np.abs(raw_terms["base"]),
        "lag": np.abs(raw_terms["lag"]),
        "transition": np.abs(raw_terms["trans"]),
        "curvature": np.abs(raw_terms["curv"]),
        "residual": np.abs(residual),
    }
    denom = sum(abs_terms.values()) + eps
    return {term: values / denom for term, values in abs_terms.items()}


def r_theorem_share_arrays(target: np.ndarray, S: np.ndarray, L: np.ndarray, eps: float) -> dict[str, np.ndarray]:
    E = target - S - L
    abs_terms = {"S": np.abs(S), "L": np.abs(L), "E": np.abs(E)}
    denom = abs_terms["S"] + abs_terms["L"] + abs_terms["E"] + eps
    return {term: values / denom for term, values in abs_terms.items()}


def markdown_mv(rows: list[dict[str, object]]) -> str:
    headers = ["quantity", "beta1", "beta2", "base", "lag", "transition", "curvature", "residual"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["quantity"]),
                    f"{float(row['beta1']):g}",
                    f"{float(row['beta2']):g}",
                    str(row["base_mean_std"]),
                    str(row["lag_mean_std"]),
                    str(row["transition_mean_std"]),
                    str(row["curvature_mean_std"]),
                    str(row["residual_mean_std"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def markdown_r(rows: list[dict[str, object]]) -> str:
    headers = ["beta1", "beta2", "S", "L", "E", "n_values"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(row['beta1']):g}",
                    f"{float(row['beta2']):g}",
                    str(row["S_mean_std"]),
                    str(row["L_mean_std"]),
                    str(row["E_mean_std"]),
                    str(row["n_values"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid 3x3 per-value mean/std share budgets for M/V and theorem-level R.")
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
    gradients = np.load(args.run_dir / "gradients.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    coord_indices = np.sort(
        rng.choice(gradients.shape[1], size=min(args.coord_sample_size, gradients.shape[1]), replace=False)
    )
    unique_betas = tuple(sorted(BETA_VALUES))
    mv_stats = {(quantity, beta1, beta2): ShareStats(MV_TERMS) for quantity in ("M", "V") for beta1, beta2 in BETA_PAIRS}
    r_stats = {(beta1, beta2): ShareStats(R_TERMS) for beta1, beta2 in BETA_PAIRS}
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
        one_minus_ratio = np.where(np.isfinite(prev_ratio), 1.0 - prev_ratio, 0.0)

        m_by_beta = {}
        v_by_beta = {}
        for beta in unique_betas:
            m_by_beta[beta] = decompose_ema_selected(chunk, shadow_ema(chunk, beta), beta, SELECTED_JS)
            v_by_beta[beta] = decompose_ema_selected(y, shadow_ema(y, beta), beta, SELECTED_JS)
        S = np.sign(m_by_beta[0.9][0]["base"])

        for beta1, beta2 in BETA_PAIRS:
            m_raw, m_obs, m_recon = m_by_beta[beta1]
            v_raw, v_obs, v_recon = v_by_beta[beta2]
            mv_stats[("M", beta1, beta2)].add(mv_share_arrays(m_raw, m_obs, m_recon, args.eps), mask)
            mv_stats[("V", beta1, beta2)].add(v_share := mv_share_arrays(v_raw, v_obs, v_recon, args.eps), mask)

            ell1 = beta1 / (1.0 - beta1)
            ell2 = beta2 / (1.0 - beta2)
            target = m_obs / np.sqrt(v_obs + args.adam_eps)
            L = S * (ell2 - ell1) * one_minus_ratio
            r_stats[(beta1, beta2)].add(r_theorem_share_arrays(target, S, L, args.eps), mask)
        print(f"Processed coordinates {start}:{stop} / {coord_indices.size}", flush=True)

    mv_rows: list[dict[str, object]] = []
    for quantity in ("M", "V"):
        for beta1, beta2 in BETA_PAIRS:
            stats = mv_stats[(quantity, beta1, beta2)]
            row: dict[str, object] = {
                "quantity": quantity,
                "beta1": beta1,
                "beta2": beta2,
                "n_coordinates": int(coord_indices.size),
                "n_windows": n_windows,
                "n_valid_windows": int(n_valid_windows),
                "n_values": stats.n_values,
            }
            for term in MV_TERMS:
                row[f"{term}_mean"] = stats.mean(term)
                row[f"{term}_std"] = stats.std(term)
                row[f"{term}_mean_std"] = fmt_pm(stats.mean(term), stats.std(term))
            mv_rows.append(row)

    r_rows: list[dict[str, object]] = []
    for beta1, beta2 in BETA_PAIRS:
        stats = r_stats[(beta1, beta2)]
        row = {
            "beta1": beta1,
            "beta2": beta2,
            "n_coordinates": int(coord_indices.size),
            "n_windows": n_windows,
            "n_valid_windows": int(n_valid_windows),
            "n_values": stats.n_values,
        }
        for term in R_TERMS:
            row[f"{term}_mean"] = stats.mean(term)
            row[f"{term}_std"] = stats.std(term)
            row[f"{term}_mean_std"] = fmt_pm(stats.mean(term), stats.std(term))
        r_rows.append(row)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mv_csv = output_dir / "table_P_MV_grid3_share_mean_std.csv"
    r_csv = output_dir / "table_Q_R_theorem_grid3_share_mean_std.csv"
    md_path = output_dir / "table_PQ_grid3_share_mean_std.md"
    write_csv(mv_csv, mv_rows)
    write_csv(r_csv, r_rows)
    md = "\n\n".join(
        [
            "# Grid 3x3 Share Mean +- Std",
            "Each cell is the mean +- standard deviation of the local absolute share over valid coordinate-window-offset values.",
            "## M/V",
            markdown_mv(mv_rows),
            "## R theorem-level",
            markdown_r(r_rows),
            f"n_coordinates={coord_indices.size}, n_windows={n_windows}, n_valid_windows={n_valid_windows}, selected_j={SELECTED_JS}.",
        ]
    )
    md_path.write_text(md, encoding="utf-8")
    write_json(
        output_dir / "table_PQ_grid3_share_mean_std_summary.json",
        {
            "mv_csv": str(mv_csv),
            "r_csv": str(r_csv),
            "markdown": str(md_path),
            "coord_sample_size": int(coord_indices.size),
            "seed": args.seed,
            "n_windows": n_windows,
            "n_valid_windows": int(n_valid_windows),
            "selected_j": list(SELECTED_JS),
        },
    )
    print("## M/V")
    print(markdown_mv(mv_rows))
    print("\n## R theorem-level")
    print(markdown_r(r_rows))
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
