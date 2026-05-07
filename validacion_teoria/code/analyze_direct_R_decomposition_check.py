from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analyze_ema_decomposition_multi_beta_R import decompose_ema_selected
from analyze_probe_ema_decomposition import (
    BASE_DIR,
    DEFAULT_RUN_DIR,
    load_json,
    shadow_ema,
    valid_and_drift,
    write_json,
)


OUTPUT_DIR = BASE_DIR / "results" / "ema_decomposition_multi_beta" / "mb_probe_W6_sample4096"
WINDOW = 6
SELECTED_JS = (0, 2, 5)
BETA_PAIRS = ((0.9, 0.9), (0.99, 0.99), (0.999, 0.999), (0.9, 0.999))
QUANTITIES = ("M", "V", "R")
TERMS = ("base", "lag", "transport", "curvature", "residual")


@dataclass
class Aggregate:
    sum_abs: dict[str, float] = field(default_factory=lambda: {term: 0.0 for term in TERMS})
    n_values: int = 0

    def add(self, terms: dict[str, np.ndarray], mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        self.n_values += int(np.count_nonzero(mask))
        for term in TERMS:
            self.sum_abs[term] += float(np.sum(np.abs(terms[term][mask]), dtype=np.float64))

    def pct(self, term: str) -> float:
        denom = sum(self.sum_abs.values())
        return 100.0 * self.sum_abs[term] / denom if denom > 0 else float("nan")


def fmt_pct(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value != 0.0 and abs(value) < 0.001:
        return f"{value:.2e}\\%"
    return f"{value:.3f}\\%"


def decompose_direct_R(
    m_terms: dict[str, np.ndarray],
    v_terms: dict[str, np.ndarray],
    delta_selected: np.ndarray,
    beta1: float,
    beta2: float,
    adam_eps: float,
) -> tuple[dict[str, np.ndarray], int]:
    ell1 = beta1 / (1.0 - beta1)
    ell2 = beta2 / (1.0 - beta2)

    g = m_terms["base"]
    y = v_terms["base"]
    sign = np.sign(g)
    abs_g = np.abs(g)

    m_full = m_terms["base"] + m_terms["lag"] + m_terms["transport"] + m_terms["curvature"]
    v_full = v_terms["base"] + v_terms["lag"] + v_terms["transport"] + v_terms["curvature"]
    denom_full = v_full + adam_eps
    nonpositive = int(np.count_nonzero(denom_full <= 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        linear_lag_from_mv = m_terms["lag"] / abs_g - 0.5 * sign * v_terms["lag"] / y
        transport = m_terms["transport"] / abs_g - 0.5 * sign * v_terms["transport"] / y
        curvature_from_mv = m_terms["curvature"] / abs_g - 0.5 * sign * v_terms["curvature"] / y
        full_R = m_full / np.sqrt(denom_full)

    base = sign
    # Direct first-order R lag. With dt=1 in this discrete replay, ell is the
    # effective time constant in steps, so the explicit factor is ell2-ell1.
    lag = sign * (ell2 - ell1) * delta_selected
    curvature = curvature_from_mv + (linear_lag_from_mv - lag)
    residual = full_R - base - lag - transport - curvature

    return {
        "base": base,
        "lag": lag,
        "transport": transport,
        "curvature": curvature,
        "residual": residual,
    }, nonpositive


def aggregate_quantity(
    aggregate: Aggregate,
    *,
    base: np.ndarray,
    lag: np.ndarray,
    transport: np.ndarray,
    curvature: np.ndarray,
    obs: np.ndarray,
    recon: np.ndarray,
    mask: np.ndarray,
) -> float:
    residual = obs - recon
    aggregate.add(
        {
            "base": base,
            "lag": lag,
            "transport": transport,
            "curvature": curvature,
            "residual": residual,
        },
        mask,
    )
    return float(np.nanmax(np.abs(residual)))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_rows(
    aggregates: dict[tuple[str, float, float], Aggregate],
    *,
    n_coordinates: int,
    n_windows: int,
    n_valid_windows: int,
) -> list[dict[str, object]]:
    rows = []
    for quantity in QUANTITIES:
        for beta1, beta2 in BETA_PAIRS:
            agg = aggregates[(quantity, beta1, beta2)]
            row = {
                "quantity": quantity,
                "beta1": beta1,
                "beta2": beta2,
                "base_pct": agg.pct("base"),
                "lag_pct": agg.pct("lag"),
                "transport_pct": agg.pct("transport"),
                "curvature_pct": agg.pct("curvature"),
                "residual_pct": agg.pct("residual"),
                "sum_abs_base": agg.sum_abs["base"],
                "sum_abs_lag": agg.sum_abs["lag"],
                "sum_abs_transport": agg.sum_abs["transport"],
                "sum_abs_curvature": agg.sum_abs["curvature"],
                "sum_abs_residual": agg.sum_abs["residual"],
                "n_coordinates": n_coordinates,
                "n_windows": n_windows,
                "n_valid_windows": n_valid_windows,
            }
            rows.append(row)
    return rows


def latex_table(rows: list[dict[str, object]]) -> str:
    out = [
        r"\begin{tabular}{lccrrrrr}",
        r"\toprule",
        r"Quantity & $\beta_1$ & $\beta_2$ & Base & Lag & Transport & Curvature & Residual \\",
        r"\midrule",
    ]
    for row in rows:
        out.append(
            f"{row['quantity']} & {float(row['beta1']):g} & {float(row['beta2']):g} & "
            f"{fmt_pct(float(row['base_pct']))} & {fmt_pct(float(row['lag_pct']))} & "
            f"{fmt_pct(float(row['transport_pct']))} & {fmt_pct(float(row['curvature_pct']))} & "
            f"{fmt_pct(float(row['residual_pct']))} \\\\"
        )
    out.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct R decomposition check with diagonal-zero lag.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--coord-sample-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--diagonal-lag-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    gradients = np.load(run_dir / "gradients.npy", mmap_mode="r")
    config = load_json(run_dir / "config.json")

    rng = np.random.default_rng(args.seed)
    coord_indices = np.sort(
        rng.choice(gradients.shape[1], size=min(args.coord_sample_size, gradients.shape[1]), replace=False)
    )

    unique_betas = tuple(sorted({beta for pair in BETA_PAIRS for beta in pair}))
    aggregates = {(quantity, beta1, beta2): Aggregate() for quantity in QUANTITIES for beta1, beta2 in BETA_PAIRS}
    n_valid_windows = 0
    max_recon_error = {"M": 0.0, "V": 0.0}
    r_nonpositive = 0

    for start in range(0, coord_indices.size, args.chunk_size):
        stop = min(start + args.chunk_size, coord_indices.size)
        idx = coord_indices[start:stop]
        chunk = np.asarray(gradients[:, idx], dtype=np.float64)
        y = chunk * chunk
        valid, delta_windows, _, _ = valid_and_drift(chunk, window=WINDOW, eps_g=0.0)
        n_valid_windows += int(np.count_nonzero(valid))
        mask = np.broadcast_to(valid[:, :, None], (valid.shape[0], valid.shape[1], len(SELECTED_JS)))
        delta_selected = delta_windows[:, :, SELECTED_JS]

        m_by_beta = {}
        v_by_beta = {}
        for beta in unique_betas:
            m_terms_raw, obs_m, recon_m = decompose_ema_selected(
                chunk, shadow_ema(chunk, beta), beta, SELECTED_JS
            )
            v_terms_raw, obs_v, recon_v = decompose_ema_selected(y, shadow_ema(y, beta), beta, SELECTED_JS)
            m_by_beta[beta] = (m_terms_raw, obs_m, recon_m)
            v_by_beta[beta] = (v_terms_raw, obs_v, recon_v)

        for beta1, beta2 in BETA_PAIRS:
            m_raw, obs_m, recon_m = m_by_beta[beta1]
            v_raw, obs_v, recon_v = v_by_beta[beta2]
            m_terms = {
                "base": m_raw["base"],
                "lag": m_raw["lag"],
                "transport": m_raw["trans"],
                "curvature": m_raw["curv"],
            }
            v_terms = {
                "base": v_raw["base"],
                "lag": v_raw["lag"],
                "transport": v_raw["trans"],
                "curvature": v_raw["curv"],
            }

            max_recon_error["M"] = max(
                max_recon_error["M"],
                aggregate_quantity(
                    aggregates[("M", beta1, beta2)],
                    base=m_terms["base"],
                    lag=m_terms["lag"],
                    transport=m_terms["transport"],
                    curvature=m_terms["curvature"],
                    obs=obs_m,
                    recon=recon_m,
                    mask=mask,
                ),
            )
            max_recon_error["V"] = max(
                max_recon_error["V"],
                aggregate_quantity(
                    aggregates[("V", beta1, beta2)],
                    base=v_terms["base"],
                    lag=v_terms["lag"],
                    transport=v_terms["transport"],
                    curvature=v_terms["curvature"],
                    obs=obs_v,
                    recon=recon_v,
                    mask=mask,
                ),
            )

            r_terms, bad = decompose_direct_R(
                m_terms,
                v_terms,
                delta_selected=delta_selected,
                beta1=beta1,
                beta2=beta2,
                adam_eps=args.adam_eps,
            )
            r_nonpositive += bad
            aggregates[("R", beta1, beta2)].add(r_terms, mask)

        print(f"Processed analyzed coordinates {start}:{stop} / {coord_indices.size}", flush=True)

    rows = make_rows(
        aggregates,
        n_coordinates=int(coord_indices.size),
        n_windows=int(gradients.shape[0] - WINDOW),
        n_valid_windows=n_valid_windows,
    )

    warnings: list[str] = []
    for row in rows:
        if row["quantity"] != "R":
            continue
        if abs(float(row["beta1"]) - float(row["beta2"])) > 0.0:
            continue
        total = sum(float(row[f"sum_abs_{term}"]) for term in TERMS)
        rel = float(row["sum_abs_lag"]) / max(total, 1e-300)
        if rel > args.diagonal_lag_tolerance:
            raise RuntimeError(
                f"Diagonal R lag check failed for beta={row['beta1']}: rel={rel:.3e}, "
                f"sum_abs_lag={row['sum_abs_lag']:.3e}"
            )
        warnings.append(
            f"Diagonal beta={float(row['beta1']):g}: R lag rel={rel:.3e}, sum_abs_lag={float(row['sum_abs_lag']):.3e}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "table_G_direct_M_V_R_decomposition_grouped.csv"
    tex_path = output_dir / "table_G_direct_M_V_R_decomposition_grouped.tex"
    summary_path = output_dir / "direct_R_decomposition_check_summary.md"
    write_csv(csv_path, rows)
    tex_path.write_text(latex_table(rows), encoding="utf-8")

    summary = "\n".join(
        [
            "# Direct R Decomposition Check",
            "",
            f"- Source run: `{run_dir}`",
            f"- Dataset: `{config.get('dataset', 'unknown')}`",
            f"- Coordinates analyzed: `{coord_indices.size}` deterministic coordinates, seed `{args.seed}`.",
            f"- Windows per coordinate: `{gradients.shape[0] - WINDOW}`.",
            f"- Sign-stable coordinate-windows: `{n_valid_windows}`.",
            f"- Contributions are aggregated over `j={SELECTED_JS}`.",
            f"- Adam epsilon for R: `{args.adam_eps}`.",
            f"- Cases with `V+epsilon <= 0`: `{r_nonpositive}`.",
            f"- Max reconstruction error M: `{max_recon_error['M']:.3e}`.",
            f"- Max reconstruction error V: `{max_recon_error['V']:.3e}`.",
            "",
            "## Diagonal lag_R checks",
            "",
            *[f"- {line}" for line in warnings],
            "",
            "## CSV",
            "",
            f"- `{csv_path.name}`",
            f"- `{tex_path.name}`",
            "",
            "## LaTeX",
            "",
            "```latex",
            latex_table(rows),
            "```",
        ]
    )
    summary_path.write_text(summary, encoding="utf-8")
    write_json(
        output_dir / "direct_R_decomposition_check_summary.json",
        {
            "csv": str(csv_path),
            "latex": str(tex_path),
            "summary": str(summary_path),
            "n_coordinates": int(coord_indices.size),
            "n_windows": int(gradients.shape[0] - WINDOW),
            "n_valid_windows": int(n_valid_windows),
            "selected_j": list(SELECTED_JS),
            "adam_eps": args.adam_eps,
            "max_recon_error": max_recon_error,
            "r_nonpositive_count": int(r_nonpositive),
            "diagonal_checks": warnings,
        },
    )
    print(summary)
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
