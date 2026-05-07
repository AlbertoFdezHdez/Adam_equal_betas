from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analyze_direct_R_decomposition_v2 import BETA_PAIRS, OUTPUT_DIR, SELECTED_JS, WINDOW, selected_previous_ratio
from analyze_ema_decomposition_multi_beta_R import decompose_ema_selected
from analyze_probe_ema_decomposition import DEFAULT_RUN_DIR, shadow_ema, valid_and_drift, write_json


DEFAULT_BETA_VALUES = (0.9, 0.99, 0.999)
THEOREM_BETA_PAIRS = tuple((beta1, beta2) for beta1 in DEFAULT_BETA_VALUES for beta2 in DEFAULT_BETA_VALUES)


@dataclass
class TheoremRStats:
    n_values: int = 0
    sum_abs: dict[str, float] = field(default_factory=lambda: {"S": 0.0, "L": 0.0, "E": 0.0})
    sum_abs_target: float = 0.0
    sum_target: float = 0.0
    sum_target_sq: float = 0.0
    sum_term: dict[str, float] = field(default_factory=lambda: {"S": 0.0, "L": 0.0, "E": 0.0})
    sum_term_sq: dict[str, float] = field(default_factory=lambda: {"S": 0.0, "L": 0.0, "E": 0.0})
    sum_term_target: dict[str, float] = field(default_factory=lambda: {"S": 0.0, "L": 0.0, "E": 0.0})
    sum_abs_target_minus_S: float = 0.0
    sum_abs_target_minus_SL: float = 0.0
    sum_sq_target_minus_S: float = 0.0
    sum_sq_target_minus_SL: float = 0.0

    def add(self, target: np.ndarray, S: np.ndarray, L: np.ndarray, mask: np.ndarray) -> None:
        safe = mask & np.isfinite(target) & np.isfinite(S) & np.isfinite(L)
        if not np.any(safe):
            return
        target_vals = target[safe]
        S_vals = S[safe]
        L_vals = L[safe]
        E_vals = target_vals - S_vals - L_vals
        terms = {"S": S_vals, "L": L_vals, "E": E_vals}

        self.n_values += int(target_vals.size)
        self.sum_abs_target += float(np.sum(np.abs(target_vals), dtype=np.float64))
        self.sum_target += float(np.sum(target_vals, dtype=np.float64))
        self.sum_target_sq += float(np.sum(target_vals * target_vals, dtype=np.float64))

        target_minus_S = target_vals - S_vals
        target_minus_SL = E_vals
        self.sum_abs_target_minus_S += float(np.sum(np.abs(target_minus_S), dtype=np.float64))
        self.sum_abs_target_minus_SL += float(np.sum(np.abs(target_minus_SL), dtype=np.float64))
        self.sum_sq_target_minus_S += float(np.sum(target_minus_S * target_minus_S, dtype=np.float64))
        self.sum_sq_target_minus_SL += float(np.sum(target_minus_SL * target_minus_SL, dtype=np.float64))

        for name, vals in terms.items():
            self.sum_abs[name] += float(np.sum(np.abs(vals), dtype=np.float64))
            self.sum_term[name] += float(np.sum(vals, dtype=np.float64))
            self.sum_term_sq[name] += float(np.sum(vals * vals, dtype=np.float64))
            self.sum_term_target[name] += float(np.sum(vals * target_vals, dtype=np.float64))

    def corr(self, name: str, eps: float) -> float:
        if self.n_values == 0:
            return float("nan")
        n = self.n_values
        target_mean = self.sum_target / n
        term_mean = self.sum_term[name] / n
        cov = self.sum_term_target[name] - n * target_mean * term_mean
        var_target = self.sum_target_sq - n * target_mean * target_mean
        var_term = self.sum_term_sq[name] - n * term_mean * term_mean
        return cov / math.sqrt(max(var_target, 0.0) * max(var_term, 0.0) + eps)

    def row(
        self,
        beta1: float,
        beta2: float,
        *,
        n_coordinates: int,
        n_windows: int,
        n_valid_windows: int,
        eps: float,
    ) -> dict[str, object]:
        denom = self.sum_abs["S"] + self.sum_abs["L"] + self.sum_abs["E"]
        target_l2 = math.sqrt(self.sum_target_sq)
        err_S_l1 = self.sum_abs_target_minus_S / (self.sum_abs_target + eps)
        err_SL_l1 = self.sum_abs_target_minus_SL / (self.sum_abs_target + eps)
        err_S_l2 = math.sqrt(self.sum_sq_target_minus_S) / (target_l2 + eps)
        err_SL_l2 = math.sqrt(self.sum_sq_target_minus_SL) / (target_l2 + eps)
        return {
            "beta1": beta1,
            "beta2": beta2,
            "S_pct": 100.0 * self.sum_abs["S"] / denom if denom else float("nan"),
            "L_pct": 100.0 * self.sum_abs["L"] / denom if denom else float("nan"),
            "E_pct": 100.0 * self.sum_abs["E"] / denom if denom else float("nan"),
            "err_S_l1": err_S_l1,
            "err_SL_l1": err_SL_l1,
            "lag_gain_l1": err_S_l1 - err_SL_l1,
            "err_S_l2": err_S_l2,
            "err_SL_l2": err_SL_l2,
            "lag_gain_l2": err_S_l2 - err_SL_l2,
            "proj_S": self.sum_term_target["S"] / (self.sum_target_sq + eps),
            "proj_L": self.sum_term_target["L"] / (self.sum_target_sq + eps),
            "proj_E": self.sum_term_target["E"] / (self.sum_target_sq + eps),
            "proj_sum": (
                self.sum_term_target["S"] + self.sum_term_target["L"] + self.sum_term_target["E"]
            )
            / (self.sum_target_sq + eps),
            "corr_S_R": self.corr("S", eps),
            "corr_L_R": self.corr("L", eps),
            "corr_E_R": self.corr("E", eps),
            "sum_abs_S": self.sum_abs["S"],
            "sum_abs_L": self.sum_abs["L"],
            "sum_abs_E": self.sum_abs["E"],
            "sum_abs_target": self.sum_abs_target,
            "target_l2": target_l2,
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
        "beta1",
        "beta2",
        "S %",
        "L %",
        "E %",
        "err S L1",
        "err S+L L1",
        "lag gain L1",
        "err S L2",
        "err S+L L2",
        "lag gain L2",
        "proj sum",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(row['beta1']):g}",
                    f"{float(row['beta2']):g}",
                    fmt_pct(float(row["S_pct"])),
                    fmt_pct(float(row["L_pct"])),
                    fmt_pct(float(row["E_pct"])),
                    f"{float(row['err_S_l1']):.3e}",
                    f"{float(row['err_SL_l1']):.3e}",
                    f"{float(row['lag_gain_l1']):.3e}",
                    f"{float(row['err_S_l2']):.3e}",
                    f"{float(row['err_SL_l2']):.3e}",
                    f"{float(row['lag_gain_l2']):.3e}",
                    f"{float(row['proj_sum']):.12f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Theorem-level R diagnostic: R = sign(g) + explicit lag + rest.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--coord-sample-size", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--eps", type=float, default=1e-30)
    parser.add_argument("--output-name", type=str, default="table_O_R_theorem_level_sign_lag_rest_grid3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gradients = np.load(args.run_dir / "gradients.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    coord_indices = np.sort(
        rng.choice(gradients.shape[1], size=min(args.coord_sample_size, gradients.shape[1]), replace=False)
    )
    n_windows = int(gradients.shape[0] - WINDOW)
    unique_betas = tuple(sorted({beta for pair in THEOREM_BETA_PAIRS for beta in pair}))
    stats = {pair: TheoremRStats() for pair in THEOREM_BETA_PAIRS}
    n_valid_windows = 0

    for start in range(0, coord_indices.size, args.chunk_size):
        stop = min(start + args.chunk_size, coord_indices.size)
        chunk = np.asarray(gradients[:, coord_indices[start:stop]], dtype=np.float64)
        y = chunk * chunk
        valid, _, _, _ = valid_and_drift(chunk, window=WINDOW, eps_g=0.0)
        n_valid_windows += int(np.count_nonzero(valid))
        mask = np.broadcast_to(valid[:, :, None], (valid.shape[0], valid.shape[1], len(SELECTED_JS)))
        prev_ratio = selected_previous_ratio(chunk)
        sign = np.sign(decompose_ema_selected(chunk, shadow_ema(chunk, 0.9), 0.9, SELECTED_JS)[0]["base"])
        one_minus_ratio = np.where(np.isfinite(prev_ratio), 1.0 - prev_ratio, 0.0)

        m_obs_by_beta = {}
        v_obs_by_beta = {}
        for beta in unique_betas:
            m_obs_by_beta[beta] = decompose_ema_selected(chunk, shadow_ema(chunk, beta), beta, SELECTED_JS)[1]
            v_obs_by_beta[beta] = decompose_ema_selected(y, shadow_ema(y, beta), beta, SELECTED_JS)[1]

        for beta1, beta2 in THEOREM_BETA_PAIRS:
            ell1 = beta1 / (1.0 - beta1)
            ell2 = beta2 / (1.0 - beta2)
            target = m_obs_by_beta[beta1] / np.sqrt(v_obs_by_beta[beta2] + args.adam_eps)
            S = sign
            L = S * (ell2 - ell1) * one_minus_ratio
            if beta1 == beta2 and np.any(np.abs(L[mask]) > 1e-12):
                raise RuntimeError(f"Diagonal theorem-level L is not zero for beta={beta1}")
            stats[(beta1, beta2)].add(target, S, L, mask)
        print(f"Processed coordinates {start}:{stop} / {coord_indices.size}", flush=True)

    rows = [
        stats[pair].row(
            pair[0],
            pair[1],
            n_coordinates=int(coord_indices.size),
            n_windows=n_windows,
            n_valid_windows=int(n_valid_windows),
            eps=args.eps,
        )
        for pair in THEOREM_BETA_PAIRS
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_name}.csv"
    md_path = output_dir / f"{args.output_name}.md"
    write_csv(csv_path, rows)
    md_path.write_text(
        "\n".join(
            [
                "# Theorem-Level R Diagnostic",
                "",
                "`R_target = M / sqrt(V + eps_adam)`.",
                "`S = sign(g)`.",
                "`L = sign(g) * (ell2 - ell1) * (1 - |g_{j-1}|/|g_j|)`.",
                "`E = R_target - S - L`.",
                "",
                markdown_table(rows),
                "",
                f"- Coordinates: `{coord_indices.size}` deterministic coordinates, seed `{args.seed}`.",
                f"- Windows per coordinate: `{n_windows}`.",
                f"- Valid sign-stable coordinate-windows: `{n_valid_windows}`.",
                f"- Values aggregate selected offsets `j={SELECTED_JS}`.",
                f"- Adam eps: `{args.adam_eps}`.",
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
            "adam_eps": args.adam_eps,
            "beta_pairs": [list(pair) for pair in THEOREM_BETA_PAIRS],
        },
    )
    print(markdown_table(rows))
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
