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
TERMS = ("base", "lag", "transport", "curvature", "residual")
QUANTITIES = ("M", "V", "R")


@dataclass
class Aggregate:
    sum_abs: dict[str, float] = field(default_factory=lambda: {term: 0.0 for term in TERMS})
    sum_signed: dict[str, float] = field(default_factory=lambda: {term: 0.0 for term in TERMS})
    n_values: int = 0

    def add(self, terms: dict[str, np.ndarray], mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        self.n_values += int(np.count_nonzero(mask))
        for term in TERMS:
            vals = terms[term][mask]
            self.sum_abs[term] += float(np.sum(np.abs(vals), dtype=np.float64))
            self.sum_signed[term] += float(np.sum(vals, dtype=np.float64))

    def pct(self, term: str) -> float:
        denom = sum(self.sum_abs.values())
        return 100.0 * self.sum_abs[term] / denom if denom > 0 else float("nan")

    def mean(self, term: str) -> float:
        return self.sum_signed[term] / self.n_values if self.n_values else float("nan")


def shadow_ema_with_initial(signal: np.ndarray, beta: float, initial: np.ndarray) -> np.ndarray:
    moment = initial.astype(np.float64, copy=True)
    out = np.empty(signal.shape, dtype=np.float64)
    one_minus_beta = 1.0 - beta
    for idx in range(signal.shape[0]):
        moment = beta * moment + one_minus_beta * signal[idx, :]
        out[idx, :] = moment
    return out


def selected_previous_ratio(values: np.ndarray) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=WINDOW + 1, axis=0)
    g_prev = windows[:, :, :-1][:, :, SELECTED_JS]
    g_cur = windows[:, :, 1:][:, :, SELECTED_JS]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(g_prev) / np.abs(g_cur)
    return ratio


def decompose_direct_R(
    m_terms: dict[str, np.ndarray],
    v_terms: dict[str, np.ndarray],
    prev_abs_ratio: np.ndarray,
    beta1: float,
    beta2: float,
    adam_eps: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    ell1 = beta1 / (1.0 - beta1)
    ell2 = beta2 / (1.0 - beta2)

    g = m_terms["base"]
    y = v_terms["base"]
    sign = np.sign(g)
    abs_g = np.abs(g)

    m_full = m_terms["base"] + m_terms["lag"] + m_terms["transport"] + m_terms["curvature"]
    v_full = v_terms["base"] + v_terms["lag"] + v_terms["transport"] + v_terms["curvature"]
    denom = v_full + adam_eps
    nonpositive = int(np.count_nonzero(denom <= 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        target = m_full / np.sqrt(denom)
        transport = m_terms["transport"] / abs_g - 0.5 * sign * v_terms["transport"] / y
        curvature_linear = m_terms["curvature"] / abs_g - 0.5 * sign * v_terms["curvature"] / y

    one_minus_ratio = np.where(np.isfinite(prev_abs_ratio), 1.0 - prev_abs_ratio, 0.0)
    base = sign
    # Direct discrete first-order R lag. This is the term that carries the
    # required factor (tau2-tau1)/dt; here dt=1 and ell=beta/(1-beta).
    lag = sign * (ell2 - ell1) * one_minus_ratio
    # The exact first-order Taylor contribution of the lag blocks contains
    # an additional quadratic-in-drift piece from Delta(g^2). That piece is
    # not part of lag_R because it does not carry ell2-ell1, so it is assigned
    # to curvature.
    curvature_from_lag_quadratic = -0.5 * sign * ell2 * (one_minus_ratio**2)
    curvature = curvature_linear + curvature_from_lag_quadratic
    residual = target - base - lag - transport - curvature
    return {
        "base": base,
        "lag": lag,
        "transport": transport,
        "curvature": curvature,
        "residual": residual,
    }, target, nonpositive


def add_m_or_v(
    aggregate: Aggregate,
    raw_terms: dict[str, np.ndarray],
    obs: np.ndarray,
    recon: np.ndarray,
    mask: np.ndarray,
) -> float:
    terms = {
        "base": raw_terms["base"],
        "lag": raw_terms["lag"],
        "transport": raw_terms["trans"],
        "curvature": raw_terms["curv"],
        "residual": obs - recon,
    }
    aggregate.add(terms, mask)
    return float(np.nanmax(np.abs(terms["residual"])))


def rows_from_aggregates(
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
                "mean_base": agg.mean("base"),
                "mean_lag": agg.mean("lag"),
                "mean_transport": agg.mean("transport"),
                "mean_curvature": agg.mean("curvature"),
                "mean_residual": agg.mean("residual"),
                "n_coordinates": n_coordinates,
                "n_windows": n_windows,
                "n_valid_windows": n_valid_windows,
                "n_values": agg.n_values,
            }
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct_latex(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value != 0.0 and abs(value) < 0.001:
        return f"{value:.2e}\\%"
    return f"{value:.3f}\\%"


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
            f"{pct_latex(float(row['base_pct']))} & {pct_latex(float(row['lag_pct']))} & "
            f"{pct_latex(float(row['transport_pct']))} & {pct_latex(float(row['curvature_pct']))} & "
            f"{pct_latex(float(row['residual_pct']))} \\\\"
        )
    out.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(out)


def run_decomposition_on_array(
    values: np.ndarray,
    *,
    stationary_init: bool,
    adam_eps: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    valid, _, _, _ = valid_and_drift(values, window=WINDOW, eps_g=0.0)
    mask = np.broadcast_to(valid[:, :, None], (valid.shape[0], valid.shape[1], len(SELECTED_JS)))
    prev_ratio = selected_previous_ratio(values)
    y = values * values

    unique_betas = tuple(sorted({beta for pair in BETA_PAIRS for beta in pair}))
    m_by_beta = {}
    v_by_beta = {}
    for beta in unique_betas:
        if stationary_init:
            m_shadow = shadow_ema_with_initial(values, beta, values[0, :])
            v_shadow = shadow_ema_with_initial(y, beta, y[0, :])
        else:
            m_shadow = shadow_ema(values, beta)
            v_shadow = shadow_ema(y, beta)
        m_by_beta[beta] = decompose_ema_selected(values, m_shadow, beta, SELECTED_JS)
        v_by_beta[beta] = decompose_ema_selected(y, v_shadow, beta, SELECTED_JS)

    aggregates = {(quantity, beta1, beta2): Aggregate() for quantity in QUANTITIES for beta1, beta2 in BETA_PAIRS}
    nonpositive = 0
    max_err = {"M": 0.0, "V": 0.0}

    for beta1, beta2 in BETA_PAIRS:
        m_raw, obs_m, recon_m = m_by_beta[beta1]
        v_raw, obs_v, recon_v = v_by_beta[beta2]
        max_err["M"] = max(max_err["M"], add_m_or_v(aggregates[("M", beta1, beta2)], m_raw, obs_m, recon_m, mask))
        max_err["V"] = max(max_err["V"], add_m_or_v(aggregates[("V", beta1, beta2)], v_raw, obs_v, recon_v, mask))
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
        r_terms, _, bad = decompose_direct_R(m_terms, v_terms, prev_ratio, beta1, beta2, adam_eps)
        nonpositive += bad
        aggregates[("R", beta1, beta2)].add(r_terms, mask)

    rows = rows_from_aggregates(
        aggregates,
        n_coordinates=values.shape[1],
        n_windows=values.shape[0] - WINDOW,
        n_valid_windows=int(np.count_nonzero(valid)),
    )
    diagnostics = {
        "n_valid_windows": int(np.count_nonzero(valid)),
        "n_values": int(np.count_nonzero(mask)),
        "nonpositive_R_denominator": int(nonpositive),
        "max_recon_error_M": max_err["M"],
        "max_recon_error_V": max_err["V"],
    }
    return rows, diagnostics


def synthetic_tests(adam_eps: float, tolerance: float) -> list[str]:
    messages: list[str] = []
    steps = 32
    coords = 16
    for value, name in [(1.0, "constant_positive"), (-1.0, "constant_negative")]:
        rows, _ = run_decomposition_on_array(
            np.full((steps, coords), value, dtype=np.float64),
            stationary_init=True,
            adam_eps=adam_eps,
        )
        for row in rows:
            if row["quantity"] != "R":
                continue
            base = float(row["base_pct"])
            others = float(row["lag_pct"]) + float(row["transport_pct"]) + float(row["curvature_pct"]) + float(row["residual_pct"])
            if abs(base - 100.0) > tolerance or others > tolerance:
                raise RuntimeError(f"{name} failed for beta=({row['beta1']},{row['beta2']}): base={base}, others={others}")
        messages.append(f"{name}: passed, R base_pct ~= 100%, other terms ~= 0")

    t = np.arange(steps, dtype=np.float64)[:, None]
    c = np.arange(coords, dtype=np.float64)[None, :]
    base_signal = np.exp(0.03 * np.sin(0.3 * t + 0.2 * c) + 0.01 * t)
    scale_rows = []
    for scale in (0.1, 1.0, 10.0):
        rows, _ = run_decomposition_on_array(scale * base_signal, stationary_init=True, adam_eps=adam_eps)
        scale_rows.append(rows)
    ref = {
        (row["quantity"], row["beta1"], row["beta2"], term): float(row[f"{term}_pct"])
        for row in scale_rows[1]
        for term in TERMS
    }
    max_diff = 0.0
    for rows in (scale_rows[0], scale_rows[2]):
        for row in rows:
            for term in TERMS:
                max_diff = max(
                    max_diff,
                    abs(float(row[f"{term}_pct"]) - ref[(row["quantity"], row["beta1"], row["beta2"], term)]),
                )
    if max_diff > 1e-3:
        raise RuntimeError(f"scale test failed: max pct diff={max_diff}")
    messages.append(f"scale_test: passed, max percentage difference={max_diff:.3e}")

    diag_rows = [row for row in scale_rows[1] if row["quantity"] == "R" and float(row["beta1"]) == float(row["beta2"])]
    for row in diag_rows:
        if float(row["sum_abs_lag"]) > 1e-12:
            raise RuntimeError(f"diagonal synthetic lag_R failed for beta={row['beta1']}: {row['sum_abs_lag']}")
    messages.append("diagonal_test: passed, synthetic lag_R is exactly zero on beta1=beta2")

    row_a = next(row for row in scale_rows[1] if row["quantity"] == "R" and row["beta1"] == 0.999 and row["beta2"] == 0.999)
    row_b = next(row for row in scale_rows[1] if row["quantity"] == "R" and row["beta1"] == 0.9 and row["beta2"] == 0.999)
    diffs = {term: abs(float(row_a[f"{term}_pct"]) - float(row_b[f"{term}_pct"])) for term in TERMS}
    if max(diffs.values()) <= 1e-8:
        raise RuntimeError("beta1 sensitivity test failed: (0.999,0.999) and (0.9,0.999) are identical")
    messages.append(
        "beta1_sensitivity: passed, (0.999,0.999) differs from (0.9,0.999); "
        + ", ".join(f"{term}_diff={diffs[term]:.3e}" for term in TERMS)
    )
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct R decomposition v2 with explicit target and synthetic tests.")
    parser.add_argument("--run-dir", type=Path, default=BASE_DIR / "descargas" / "adam_gradient_cifar10" / "runs" / "cifar10_tinycnn_adam_b1-0p9_b2-0p999_bs-128_grad-probe_seed-1234")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--coord-sample-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--test-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_messages = synthetic_tests(args.adam_eps, args.test_tolerance)

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    gradients = np.load(run_dir / "gradients.npy", mmap_mode="r")
    config = load_json(run_dir / "config.json")

    rng = np.random.default_rng(args.seed)
    coord_indices = np.sort(
        rng.choice(gradients.shape[1], size=min(args.coord_sample_size, gradients.shape[1]), replace=False)
    )

    all_rows: list[dict[str, object]] = []
    real_diagnostics: dict[str, object] | None = None
    for start in range(0, coord_indices.size, args.chunk_size):
        stop = min(start + args.chunk_size, coord_indices.size)
        idx = coord_indices[start:stop]
        rows, diagnostics = run_decomposition_on_array(
            np.asarray(gradients[:, idx], dtype=np.float64),
            stationary_init=False,
            adam_eps=args.adam_eps,
        )
        if not all_rows:
            all_rows = rows
            real_diagnostics = diagnostics
        else:
            for target, source in zip(all_rows, rows):
                for term in TERMS:
                    target[f"sum_abs_{term}"] = float(target[f"sum_abs_{term}"]) + float(source[f"sum_abs_{term}"])
                    target[f"mean_{term}"] = 0.0
                target["n_coordinates"] = int(target["n_coordinates"]) + int(source["n_coordinates"])
                target["n_valid_windows"] = int(target["n_valid_windows"]) + int(source["n_valid_windows"])
                target["n_values"] = int(target["n_values"]) + int(source["n_values"])
            if real_diagnostics is not None:
                real_diagnostics["n_valid_windows"] = int(real_diagnostics["n_valid_windows"]) + int(diagnostics["n_valid_windows"])
                real_diagnostics["n_values"] = int(real_diagnostics["n_values"]) + int(diagnostics["n_values"])
                real_diagnostics["nonpositive_R_denominator"] = int(real_diagnostics["nonpositive_R_denominator"]) + int(diagnostics["nonpositive_R_denominator"])
                real_diagnostics["max_recon_error_M"] = max(float(real_diagnostics["max_recon_error_M"]), float(diagnostics["max_recon_error_M"]))
                real_diagnostics["max_recon_error_V"] = max(float(real_diagnostics["max_recon_error_V"]), float(diagnostics["max_recon_error_V"]))
        print(f"Processed analyzed coordinates {start}:{stop} / {coord_indices.size}", flush=True)

    # Recompute percentages after chunk aggregation.
    for row in all_rows:
        denom = sum(float(row[f"sum_abs_{term}"]) for term in TERMS)
        for term in TERMS:
            row[f"{term}_pct"] = 100.0 * float(row[f"sum_abs_{term}"]) / denom if denom > 0 else float("nan")
        row["n_coordinates"] = int(coord_indices.size)
        row["n_windows"] = int(gradients.shape[0] - WINDOW)

    for row in all_rows:
        if row["quantity"] == "R" and abs(float(row["beta1"]) - float(row["beta2"])) == 0.0:
            total = sum(float(row[f"sum_abs_{term}"]) for term in TERMS)
            rel = float(row["sum_abs_lag"]) / max(total, 1e-300)
            if rel > 1e-12:
                raise RuntimeError(f"Real diagonal lag_R failed for beta={row['beta1']}: rel={rel:.3e}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "table_H_direct_M_V_R_decomposition_v2.csv"
    tex_path = output_dir / "table_H_direct_M_V_R_decomposition_v2.tex"
    formula_csv = output_dir / "table_H_R_formula_terms_v2.csv"
    summary_path = output_dir / "direct_R_decomposition_v2_summary.md"
    write_csv(csv_path, all_rows)
    tex_path.write_text(latex_table(all_rows), encoding="utf-8")
    formula_rows = [
        {
            "R_target": "M_j / sqrt(V_j + eps_adam)",
            "R_base": "sign(g_j)",
            "R_lag": "sign(g_j) * (ell2 - ell1) * (1 - |g_{j-1}|/|g_j|)",
            "R_transport": "transport_m/|g_j| - 0.5*sign(g_j)*transport_v/g_j^2",
            "R_curvature": "curvature_m/|g_j| - 0.5*sign(g_j)*curvature_v/g_j^2 - 0.5*sign(g_j)*ell2*(1-|g_{j-1}|/|g_j|)^2",
            "R_residual": "R_target - (R_base + R_lag + R_transport + R_curvature)",
        }
    ]
    write_csv(formula_csv, formula_rows)

    summary = "\n".join(
        [
            "# Direct R Decomposition v2",
            "",
            "## Explicit target and terms",
            "",
            "- `R_target = M_j / sqrt(V_j + eps_adam)`.",
            "- `R_base = sign(g_j)`.",
            "- `R_lag = sign(g_j) * (ell2 - ell1) * (1 - |g_{j-1}|/|g_j|)`.",
            "- `R_transport = transport_m/|g_j| - 0.5 sign(g_j) transport_v/g_j^2`.",
            "- `R_curvature = curvature_m/|g_j| - 0.5 sign(g_j) curvature_v/g_j^2 - 0.5 sign(g_j) ell2 (1-|g_{j-1}|/|g_j|)^2`.",
            "- `R_residual = R_target - (R_base + R_lag + R_transport + R_curvature)`.",
            "",
            "## Synthetic tests",
            "",
            *[f"- {message}" for message in test_messages],
            "",
            "## Real replay setup",
            "",
            f"- Source run: `{run_dir}`.",
            f"- Dataset: `{config.get('dataset', 'unknown')}`.",
            f"- Coordinates analyzed: `{coord_indices.size}` deterministic coordinates, seed `{args.seed}`.",
            f"- Windows per coordinate: `{gradients.shape[0] - WINDOW}`.",
            f"- Sign-stable coordinate-windows: `{real_diagnostics['n_valid_windows'] if real_diagnostics else 'NA'}`.",
            f"- Values aggregated include `j={SELECTED_JS}`.",
            f"- Adam eps: `{args.adam_eps}`.",
            f"- Cases with `V+eps <= 0`: `{real_diagnostics['nonpositive_R_denominator'] if real_diagnostics else 'NA'}`.",
            f"- Max reconstruction error M: `{float(real_diagnostics['max_recon_error_M']):.3e}`.",
            f"- Max reconstruction error V: `{float(real_diagnostics['max_recon_error_V']):.3e}`.",
            "",
            "## LaTeX",
            "",
            "```latex",
            latex_table(all_rows),
            "```",
        ]
    )
    summary_path.write_text(summary, encoding="utf-8")
    write_json(
        output_dir / "direct_R_decomposition_v2_summary.json",
        {
            "csv": str(csv_path),
            "latex": str(tex_path),
            "formula_csv": str(formula_csv),
            "summary": str(summary_path),
            "synthetic_tests": test_messages,
            "real_diagnostics": real_diagnostics,
            "coord_sample_size": int(coord_indices.size),
            "seed": args.seed,
            "selected_j": list(SELECTED_JS),
            "adam_eps": args.adam_eps,
        },
    )
    print(summary)
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
