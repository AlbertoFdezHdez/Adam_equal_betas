from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analyze_direct_R_decomposition_v2 import OUTPUT_DIR, SELECTED_JS, WINDOW
from analyze_ema_decomposition_multi_beta_R import decompose_ema_selected
from analyze_probe_ema_decomposition import DEFAULT_RUN_DIR, shadow_ema, valid_and_drift, write_json


TERMS = ("base", "lag", "transition", "curvature", "residual")
DEFAULT_BETA_VALUES = (0.9, 0.99, 0.999)
GRID_BETA_PAIRS = tuple((beta1, beta2) for beta1 in DEFAULT_BETA_VALUES for beta2 in DEFAULT_BETA_VALUES)


@dataclass
class MVStats:
    sum_abs: dict[str, float] = field(default_factory=lambda: {term: 0.0 for term in TERMS})
    sum_abs_target: float = 0.0
    sum_target_sq: float = 0.0
    sum_residual_sq: float = 0.0
    n_values: int = 0

    def add(self, raw_terms: dict[str, np.ndarray], obs: np.ndarray, recon: np.ndarray, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        residual = obs - recon
        terms = {
            "base": raw_terms["base"],
            "lag": raw_terms["lag"],
            "transition": raw_terms["trans"],
            "curvature": raw_terms["curv"],
            "residual": residual,
        }
        target = obs[mask]
        residual_vals = residual[mask]
        self.n_values += int(target.size)
        self.sum_abs_target += float(np.sum(np.abs(target), dtype=np.float64))
        self.sum_target_sq += float(np.sum(target * target, dtype=np.float64))
        self.sum_residual_sq += float(np.sum(residual_vals * residual_vals, dtype=np.float64))
        for term, values in terms.items():
            vals = values[mask]
            self.sum_abs[term] += float(np.sum(np.abs(vals), dtype=np.float64))

    def row(
        self,
        quantity: str,
        beta1: float,
        beta2: float,
        *,
        n_coordinates: int,
        n_windows: int,
        n_valid_windows: int,
        eps: float,
    ) -> dict[str, object]:
        denom = sum(self.sum_abs.values())
        target_l2 = math.sqrt(self.sum_target_sq)
        residual_l2 = math.sqrt(self.sum_residual_sq)
        return {
            "quantity": quantity,
            "beta1": beta1,
            "beta2": beta2,
            "base_pct": 100.0 * self.sum_abs["base"] / denom if denom else float("nan"),
            "lag_pct": 100.0 * self.sum_abs["lag"] / denom if denom else float("nan"),
            "transition_pct": 100.0 * self.sum_abs["transition"] / denom if denom else float("nan"),
            "curvature_pct": 100.0 * self.sum_abs["curvature"] / denom if denom else float("nan"),
            "residual_pct": 100.0 * self.sum_abs["residual"] / denom if denom else float("nan"),
            "sum_abs_base": self.sum_abs["base"],
            "sum_abs_lag": self.sum_abs["lag"],
            "sum_abs_transition": self.sum_abs["transition"],
            "sum_abs_curvature": self.sum_abs["curvature"],
            "sum_abs_residual": self.sum_abs["residual"],
            "sum_abs_target": self.sum_abs_target,
            "target_l2": target_l2,
            "residual_l1_rel": self.sum_abs["residual"] / (self.sum_abs_target + eps),
            "residual_l2_rel": residual_l2 / (target_l2 + eps),
            "n_coordinates": n_coordinates,
            "n_windows": n_windows,
            "n_valid_windows": n_valid_windows,
            "n_values": self.n_values,
        }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float) -> str:
    if value != 0.0 and abs(value) < 0.001:
        return f"{value:.2e}%"
    return f"{value:.3f}%"


def markdown_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "quantity",
        "beta1",
        "beta2",
        "base %",
        "lag %",
        "transition %",
        "curvature %",
        "residual %",
        "residual_l1_rel",
        "residual_l2_rel",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["quantity"]),
                    f"{float(row['beta1']):g}",
                    f"{float(row['beta2']):g}",
                    fmt_pct(float(row["base_pct"])),
                    fmt_pct(float(row["lag_pct"])),
                    fmt_pct(float(row["transition_pct"])),
                    fmt_pct(float(row["curvature_pct"])),
                    fmt_pct(float(row["residual_pct"])),
                    f"{float(row['residual_l1_rel']):.3e}",
                    f"{float(row['residual_l2_rel']):.3e}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final M/V absolute budget table with target and residual diagnostics.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--coord-sample-size", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--eps", type=float, default=1e-30)
    parser.add_argument("--output-name", type=str, default="table_N_final_MV_budget_grid3_with_residuals")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gradients = np.load(args.run_dir / "gradients.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    coord_indices = np.sort(
        rng.choice(gradients.shape[1], size=min(args.coord_sample_size, gradients.shape[1]), replace=False)
    )
    beta_pairs = GRID_BETA_PAIRS
    unique_betas = tuple(sorted({beta for pair in beta_pairs for beta in pair}))
    stats = {(quantity, beta1, beta2): MVStats() for quantity in ("M", "V") for beta1, beta2 in beta_pairs}
    n_valid_windows = 0
    n_windows = int(gradients.shape[0] - WINDOW)

    for start in range(0, coord_indices.size, args.chunk_size):
        stop = min(start + args.chunk_size, coord_indices.size)
        chunk = np.asarray(gradients[:, coord_indices[start:stop]], dtype=np.float64)
        valid, _, _, _ = valid_and_drift(chunk, window=WINDOW, eps_g=0.0)
        n_valid_windows += int(np.count_nonzero(valid))
        mask = np.broadcast_to(valid[:, :, None], (valid.shape[0], valid.shape[1], len(SELECTED_JS)))
        y = chunk * chunk
        m_by_beta = {}
        v_by_beta = {}
        for beta in unique_betas:
            m_by_beta[beta] = decompose_ema_selected(chunk, shadow_ema(chunk, beta), beta, SELECTED_JS)
            v_by_beta[beta] = decompose_ema_selected(y, shadow_ema(y, beta), beta, SELECTED_JS)
        for beta1, beta2 in beta_pairs:
            m_raw, m_obs, m_recon = m_by_beta[beta1]
            v_raw, v_obs, v_recon = v_by_beta[beta2]
            stats[("M", beta1, beta2)].add(m_raw, m_obs, m_recon, mask)
            stats[("V", beta1, beta2)].add(v_raw, v_obs, v_recon, mask)
        print(f"Processed coordinates {start}:{stop} / {coord_indices.size}", flush=True)

    rows = [
        stats[(quantity, beta1, beta2)].row(
            quantity,
            beta1,
            beta2,
            n_coordinates=int(coord_indices.size),
            n_windows=n_windows,
            n_valid_windows=int(n_valid_windows),
            eps=args.eps,
        )
        for quantity in ("M", "V")
        for beta1, beta2 in beta_pairs
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_name}.csv"
    md_path = output_dir / f"{args.output_name}.md"
    write_csv(csv_path, rows)
    md_path.write_text(
        "\n".join(
            [
                "# Final M/V Budget With Target Residuals",
                "",
                markdown_table(rows),
                "",
                f"- Coordinates: `{coord_indices.size}` deterministic coordinates, seed `{args.seed}`.",
                f"- Windows per coordinate: `{n_windows}`.",
                f"- Valid sign-stable coordinate-windows: `{n_valid_windows}`.",
                f"- Values aggregate selected offsets `j={SELECTED_JS}`.",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        output_dir / f"{args.output_name}_summary.json",
        {
            "csv": str(csv_path),
            "markdown": str(md_path),
            "coord_sample_size": int(coord_indices.size),
            "seed": args.seed,
            "n_windows": n_windows,
            "n_valid_windows": int(n_valid_windows),
            "selected_j": list(SELECTED_JS),
            "beta_pairs": [list(pair) for pair in beta_pairs],
        },
    )
    print(markdown_table(rows))
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
