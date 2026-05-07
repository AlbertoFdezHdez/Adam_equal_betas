from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analyze_ema_decomposition_multi_beta import (
    MINIMUM_BETA_PAIRS,
    REGIMES,
    SELECTED_JS,
    TERMS,
    fmt_num,
    fmt_pct,
    markdown_table,
    pct,
)
from analyze_probe_ema_decomposition import (
    BASE_DIR,
    DEFAULT_RUN_DIR,
    Reservoir,
    load_json,
    shadow_ema,
    stable_seed,
    valid_and_drift,
    write_json,
)


DEFAULT_MULTI_BETA_DIR = (
    BASE_DIR / "results" / "ema_decomposition_multi_beta" / "mb_probe_W6_sample4096"
)
WINDOW = 6
R_TERMS = ("base", "lag", "trans", "curv", "residual")


@dataclass
class RStats:
    share_samples: dict[str, Reservoir]
    residual_ratio_sample: Reservoir
    count: int = 0
    share_sums: dict[str, float] = field(default_factory=lambda: {term: 0.0 for term in R_TERMS})
    dom_counts: dict[str, int] = field(default_factory=lambda: {term: 0 for term in R_TERMS})
    denom_nonpositive_count: int = 0


def make_reservoir(seed: int, sample_size: int, *parts: object) -> Reservoir:
    return Reservoir(sample_size, np.random.default_rng(stable_seed(seed, *parts)))


def empty_r_stats(seed: int, sample_size: int, *parts: object) -> RStats:
    return RStats(
        share_samples={term: make_reservoir(seed, sample_size, *parts, "share", term) for term in R_TERMS},
        residual_ratio_sample=make_reservoir(seed, sample_size, *parts, "residual_ratio"),
    )


def decompose_ema_selected(
    values: np.ndarray,
    ema_post: np.ndarray,
    beta: float,
    selected_js: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    ell = beta / (1.0 - beta)
    kappa = beta * beta / (1.0 - beta)
    value_windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=WINDOW + 1, axis=0)
    ema_windows = np.lib.stride_tricks.sliding_window_view(ema_post[1:, :], window_shape=WINDOW, axis=0)

    base0 = value_windows[:, :, 1]
    lag0 = -ell * (value_windows[:, :, 1] - value_windows[:, :, 0])
    d0 = ema_windows[:, :, 0] - (base0 + lag0)

    selected_set = set(selected_js)
    term_lists = {term: [] for term in TERMS}
    recon_lists = []
    obs_lists = []
    curv_current = np.zeros_like(base0)

    for j in range(WINDOW):
        if j > 0:
            d2 = value_windows[:, :, j + 1] - 2.0 * value_windows[:, :, j] + value_windows[:, :, j - 1]
            curv_current = beta * curv_current + kappa * d2
        if j not in selected_set:
            continue
        base = value_windows[:, :, j + 1]
        lag = -ell * (value_windows[:, :, j + 1] - value_windows[:, :, j])
        trans = (beta**j) * d0
        curv = curv_current.copy() if j > 0 else np.zeros_like(base0)
        recon = base + lag + trans + curv
        term_lists["base"].append(base)
        term_lists["lag"].append(lag)
        term_lists["trans"].append(trans)
        term_lists["curv"].append(curv)
        recon_lists.append(recon)
        obs_lists.append(ema_windows[:, :, j])

    terms = {term: np.stack(values_list, axis=-1) for term, values_list in term_lists.items()}
    return terms, np.stack(obs_lists, axis=-1), np.stack(recon_lists, axis=-1)


def update_r_stats(
    stats: RStats,
    *,
    m_terms: dict[str, np.ndarray],
    v_terms: dict[str, np.ndarray],
    mask: np.ndarray,
    adam_eps: float,
    eps_r: float,
    share_eps: float,
) -> None:
    if not np.any(mask):
        return
    m_full = sum(m_terms[term] for term in TERMS)
    v_full = sum(v_terms[term] for term in TERMS)
    denom = v_full + adam_eps
    nonpositive = denom <= 0.0
    safe = mask & (~nonpositive)
    stats.denom_nonpositive_count += int(np.count_nonzero(mask & nonpositive))
    if not np.any(safe):
        return

    denom_safe = denom[safe]
    sqrt_denom = np.sqrt(denom_safe)
    denom_32 = denom_safe * sqrt_denom
    m_safe = m_full[safe]
    r_full = m_safe / sqrt_denom

    attrs: dict[str, np.ndarray] = {}
    for term in TERMS:
        attrs[term] = m_terms[term][safe] / sqrt_denom - 0.5 * m_safe * v_terms[term][safe] / denom_32
    residual = r_full - sum(attrs[term] for term in TERMS)
    attrs["residual"] = residual

    abs_attrs = {term: np.abs(attrs[term]) for term in R_TERMS}
    total = sum(abs_attrs[term] for term in R_TERMS) + share_eps
    dom = np.argmax(np.stack([abs_attrs[term] for term in R_TERMS], axis=0), axis=0)

    n = int(total.size)
    stats.count += n
    for idx, term in enumerate(R_TERMS):
        share = abs_attrs[term] / total
        stats.share_sums[term] += float(np.sum(share, dtype=np.float64))
        stats.share_samples[term].add(share)
        stats.dom_counts[term] += int(np.count_nonzero(dom == idx))
    stats.residual_ratio_sample.add(abs_attrs["residual"] / (np.abs(r_full) + eps_r))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_r_rows(stats_by_key: dict[tuple[float, float, str, int], RStats]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for beta1, beta2 in MINIMUM_BETA_PAIRS:
        for regime in REGIMES:
            for j in SELECTED_JS:
                stats = stats_by_key[(beta1, beta2, regime, j)]
                row: dict[str, object] = {
                    "beta1": beta1,
                    "beta2": beta2,
                    "regime": regime,
                    "j": j,
                    "coord_windows": stats.count,
                    "denom_nonpositive_count": stats.denom_nonpositive_count,
                }
                for term in R_TERMS:
                    row[f"R_{term}_share_mean"] = stats.share_sums[term] / stats.count if stats.count else float("nan")
                    row[f"R_{term}_share_p50"] = stats.share_samples[term].percentile(50)
                    row[f"R_{term}_dom_percent"] = pct(stats.dom_counts[term], stats.count)
                qs = stats.residual_ratio_sample.percentiles((50, 90))
                row["R_residual_ratio_mean"] = float(np.mean(stats.residual_ratio_sample.sample())) if stats.residual_ratio_sample.filled else float("nan")
                row["R_residual_ratio_p50"] = qs["p50"]
                row["R_residual_ratio_p90"] = qs["p90"]
                rows.append(row)
    return rows


def table_r_markdown(rows: list[dict[str, object]]) -> str:
    headers = [
        "beta1",
        "beta2",
        "regime",
        "j",
        "R base share mean",
        "R lag share mean",
        "R trans share mean",
        "R curv share mean",
        "R residual share mean",
        "R base p50",
        "R lag p50",
        "R trans p50",
        "R curv p50",
        "R residual p50",
        "R base dom %",
        "R lag dom %",
        "R trans dom %",
        "R curv dom %",
        "R residual dom %",
    ]
    md_rows = []
    for row in rows:
        md_rows.append(
            [
                fmt_num(row["beta1"]),
                fmt_num(row["beta2"]),
                str(row["regime"]),
                str(row["j"]),
                fmt_pct(100.0 * float(row["R_base_share_mean"])),
                fmt_pct(100.0 * float(row["R_lag_share_mean"])),
                fmt_pct(100.0 * float(row["R_trans_share_mean"])),
                fmt_pct(100.0 * float(row["R_curv_share_mean"])),
                fmt_pct(100.0 * float(row["R_residual_share_mean"])),
                fmt_pct(100.0 * float(row["R_base_share_p50"])),
                fmt_pct(100.0 * float(row["R_lag_share_p50"])),
                fmt_pct(100.0 * float(row["R_trans_share_p50"])),
                fmt_pct(100.0 * float(row["R_curv_share_p50"])),
                fmt_pct(100.0 * float(row["R_residual_share_p50"])),
                fmt_pct(row["R_base_dom_percent"]),
                fmt_pct(row["R_lag_dom_percent"]),
                fmt_pct(row["R_trans_dom_percent"]),
                fmt_pct(row["R_curv_dom_percent"]),
                fmt_pct(row["R_residual_dom_percent"]),
            ]
        )
    return markdown_table(headers, md_rows)


def read_mv_shares(multi_beta_dir: Path) -> dict[tuple[float, float, str], dict[str, float]]:
    path = multi_beta_dir / "table_A_absolute_share_budget_selected_j.csv"
    out: dict[tuple[float, float, str], dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["regime"] != "A" or int(float(row["j"])) != 5:
                continue
            key = (float(row["beta1"]), float(row["beta2"]), row["signal"])
            out[key] = {term: float(row[f"{term}_share_mean"]) for term in TERMS}
    return out


def build_compact_rows(
    r_rows: list[dict[str, object]],
    mv_shares: dict[tuple[float, float, str], dict[str, float]],
) -> list[dict[str, object]]:
    r_by_pair = {
        (float(row["beta1"]), float(row["beta2"])): row
        for row in r_rows
        if row["regime"] == "A" and int(row["j"]) == 5
    }
    rows = []
    for beta1, beta2 in MINIMUM_BETA_PAIRS:
        row: dict[str, object] = {"beta1": beta1, "beta2": beta2}
        for signal in ("m", "v"):
            shares = mv_shares[(beta1, beta2, signal)]
            for term in TERMS:
                row[f"{signal}_{term}"] = shares[term]
        r_row = r_by_pair[(beta1, beta2)]
        for term in R_TERMS:
            row[f"R_{term}"] = float(r_row[f"R_{term}_share_mean"])
        rows.append(row)
    return rows


def compact_markdown(rows: list[dict[str, object]]) -> str:
    headers = [
        "beta1",
        "beta2",
        "m base",
        "m lag",
        "m trans",
        "m curv",
        "v base",
        "v lag",
        "v trans",
        "v curv",
        "R base",
        "R lag",
        "R trans",
        "R curv",
        "R residual",
    ]
    md_rows = []
    for row in rows:
        md_rows.append(
            [
                fmt_num(row["beta1"]),
                fmt_num(row["beta2"]),
                *[fmt_pct(100.0 * float(row[f"m_{term}"])) for term in TERMS],
                *[fmt_pct(100.0 * float(row[f"v_{term}"])) for term in TERMS],
                *[fmt_pct(100.0 * float(row[f"R_{term}"])) for term in R_TERMS],
            ]
        )
    return markdown_table(headers, md_rows)


def write_compact_csv(path: Path, rows: list[dict[str, object]]) -> None:
    write_csv(path, rows)


def build_interpretation(r_rows: list[dict[str, object]], compact_rows: list[dict[str, object]]) -> str:
    r_a5 = [row for row in r_rows if row["regime"] == "A" and int(row["j"]) == 5]
    best_base = max(r_a5, key=lambda row: float(row["R_base_share_mean"]))
    current = next(row for row in r_a5 if float(row["beta1"]) == 0.9 and float(row["beta2"]) == 0.999)
    dom_terms = {
        term: float(current[f"R_{term}_dom_percent"])
        for term in R_TERMS
    }
    main_dom = max(dom_terms, key=dom_terms.get)
    residual_mean = float(current["R_residual_share_mean"])
    residual_ratio_p90 = float(current["R_residual_ratio_p90"])
    diag_09 = next(row for row in r_a5 if float(row["beta1"]) == 0.9 and float(row["beta2"]) == 0.9)
    return "\n".join(
        [
            f"- In regime A, j=5, the largest R base share occurs at ({best_base['beta1']:g},{best_base['beta2']:g}) with {fmt_pct(100.0 * float(best_base['R_base_share_mean']))}.",
            f"- For the original pair (0.9,0.999), the largest R attribution by dominance is `{main_dom}`: "
            + ", ".join(f"{term}={fmt_pct(value)}" for term, value in dom_terms.items())
            + ".",
            f"- The R residual share mean for (0.9,0.999), regime A, j=5 is {fmt_pct(100.0 * residual_mean)}; the p90 of |R_res|/(|R_full|+eps_R) is {fmt_num(residual_ratio_p90)}.",
            f"- The diagonal short-memory pair (0.9,0.9) has R base share {fmt_pct(100.0 * float(diag_09['R_base_share_mean']))} versus {fmt_pct(100.0 * float(current['R_base_share_mean']))} for (0.9,0.999). This suggests, but does not prove, that shorter/balanced EMA memories make the normalized update attribution more base-like on this fixed gradient stream.",
            "- Because this is a linearized attribution around the full (M,V), a large residual means the four-term R budget should be read as descriptive rather than as an exact decomposition.",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Light R attribution for multi-beta EMA replay.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--multi-beta-dir", type=Path, default=DEFAULT_MULTI_BETA_DIR)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--coord-sample-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--eps-r", type=float, default=1e-12)
    parser.add_argument("--share-eps", type=float, default=1e-30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    multi_beta_dir = args.multi_beta_dir.resolve()
    gradients = np.load(run_dir / "gradients.npy", mmap_mode="r")
    summary = load_json(multi_beta_dir / "multi_beta_diagnostic_summary.json")
    drift_threshold = float(summary["drift_threshold_for_regime_B"])

    rng = np.random.default_rng(args.seed)
    coord_indices = np.sort(
        rng.choice(gradients.shape[1], size=min(args.coord_sample_size, gradients.shape[1]), replace=False)
    )
    total_coord_windows = (gradients.shape[0] - WINDOW) * int(coord_indices.size)

    stats_by_key: dict[tuple[float, float, str, int], RStats] = {}
    unique_betas = tuple(sorted({beta for pair in MINIMUM_BETA_PAIRS for beta in pair}))
    sign_stable_count = 0
    denom_nonpositive_total = 0
    max_recon_m = 0.0
    max_recon_v = 0.0

    for start in range(0, coord_indices.size, args.chunk_size):
        stop = min(start + args.chunk_size, coord_indices.size)
        idx = coord_indices[start:stop]
        chunk = np.asarray(gradients[:, idx], dtype=np.float64)
        y = chunk * chunk
        valid, _, max_abs_delta, _ = valid_and_drift(chunk, window=WINDOW, eps_g=0.0)
        sign_stable_count += int(np.count_nonzero(valid))
        regimes = {"A": valid, "B": valid & (max_abs_delta <= drift_threshold)}

        m_by_beta = {}
        v_by_beta = {}
        for beta in unique_betas:
            m_terms, obs_m, recon_m = decompose_ema_selected(chunk, shadow_ema(chunk, beta), beta, SELECTED_JS)
            v_terms, obs_v, recon_v = decompose_ema_selected(y, shadow_ema(y, beta), beta, SELECTED_JS)
            max_recon_m = max(max_recon_m, float(np.max(np.abs(obs_m - recon_m))))
            max_recon_v = max(max_recon_v, float(np.max(np.abs(obs_v - recon_v))))
            m_by_beta[beta] = m_terms
            v_by_beta[beta] = v_terms

        for beta1, beta2 in MINIMUM_BETA_PAIRS:
            for regime in REGIMES:
                base_mask = regimes[regime]
                for pos, j in enumerate(SELECTED_JS):
                    key = (beta1, beta2, regime, j)
                    if key not in stats_by_key:
                        stats_by_key[key] = empty_r_stats(args.seed, args.sample_size, *key)
                    mask = base_mask[:, :, None]
                    j_mask = np.broadcast_to(mask, m_by_beta[beta1]["base"][:, :, pos : pos + 1].shape)[:, :, 0]
                    m_terms_j = {term: m_by_beta[beta1][term][:, :, pos] for term in TERMS}
                    v_terms_j = {term: v_by_beta[beta2][term][:, :, pos] for term in TERMS}
                    before_bad = stats_by_key[key].denom_nonpositive_count
                    update_r_stats(
                        stats_by_key[key],
                        m_terms=m_terms_j,
                        v_terms=v_terms_j,
                        mask=j_mask,
                        adam_eps=args.adam_eps,
                        eps_r=args.eps_r,
                        share_eps=args.share_eps,
                    )
                    denom_nonpositive_total += stats_by_key[key].denom_nonpositive_count - before_bad

        print(f"Processed analyzed coordinates {start}:{stop} / {coord_indices.size}", flush=True)

    r_rows = build_r_rows(stats_by_key)
    r_csv = multi_beta_dir / "table_E_R_linearized_share_budget_selected_j.csv"
    write_csv(r_csv, r_rows)

    mv_shares = read_mv_shares(multi_beta_dir)
    compact_rows = build_compact_rows(r_rows, mv_shares)
    compact_csv = multi_beta_dir / "table_F_compact_m_v_R_regimeA_j5.csv"
    write_compact_csv(compact_csv, compact_rows)

    current_a5 = next(
        row for row in r_rows if float(row["beta1"]) == 0.9 and float(row["beta2"]) == 0.999 and row["regime"] == "A" and int(row["j"]) == 5
    )
    report = f"""# EMA Decomposition Multi-Beta R Interpretive Summary

## 0. Setup

- Mode: same-gradient-stream replay on stored probe gradients.
- Coordinates analyzed: `{coord_indices.size}` deterministic coordinates, seed `{args.seed}`.
- Coordinate-windows analyzed: `{total_coord_windows}`.
- Sign-stable coordinate-windows: `{sign_stable_count}` = {fmt_pct(pct(sign_stable_count, total_coord_windows))}.
- W: `{WINDOW}`.
- j values: `{', '.join(str(j) for j in SELECTED_JS)}`.
- Regimes: A sign-stable; B sign-stable plus `max|delta| <= {drift_threshold:.6g}`.
- Adam epsilon used for R: `{args.adam_eps}`.
- eps_R used for residual ratios: `{args.eps_r}`.
- Cases with `V + epsilon <= 0`: `{denom_nonpositive_total}`.
- Reconstruction max error m: `{max_recon_m:.3e}`.
- Reconstruction max error v: `{max_recon_v:.3e}`.

This is not an exact decomposition of R. It is a linearized attribution around the full `(M,V)`:

`A_t = m_t / sqrt(V+eps) - 0.5 * M * v_t / (V+eps)^(3/2)`.

The nonlinear residual is reported explicitly.

## 1. Main R Share Budget

{table_r_markdown(r_rows)}

## 2. Compact m/v/R Table, Regime A, j=5

{compact_markdown(compact_rows)}

## 3. Sanity Checks

- Residual ratio `|R_res|/(|R_full|+eps_R)` for `(0.9,0.999)`, regime A, j=5:
  - mean: `{fmt_num(current_a5['R_residual_ratio_mean'])}`
  - p50: `{fmt_num(current_a5['R_residual_ratio_p50'])}`
  - p90: `{fmt_num(current_a5['R_residual_ratio_p90'])}`
- R residual share mean for `(0.9,0.999)`, regime A, j=5: `{fmt_pct(100.0 * float(current_a5['R_residual_share_mean']))}`.

## 4. Automatic Interpretation

{build_interpretation(r_rows, compact_rows)}

## 5. Outputs

- `table_E_R_linearized_share_budget_selected_j.csv`
- `table_F_compact_m_v_R_regimeA_j5.csv`
"""
    report_path = multi_beta_dir / "ema_decomposition_multi_beta_R_interpretive_summary.md"
    report_path.write_text(report, encoding="utf-8")
    write_json(
        multi_beta_dir / "multi_beta_R_diagnostic_summary.json",
        {
            "coord_sample_size": int(coord_indices.size),
            "coord_sample_seed": args.seed,
            "total_coord_windows": int(total_coord_windows),
            "sign_stable_count": int(sign_stable_count),
            "sign_stable_percent": pct(sign_stable_count, total_coord_windows),
            "adam_eps": args.adam_eps,
            "eps_r": args.eps_r,
            "denom_nonpositive_total": int(denom_nonpositive_total),
            "max_recon_error_m": max_recon_m,
            "max_recon_error_v": max_recon_v,
        },
    )
    print(report)
    print(f"Results written to: {multi_beta_dir}", flush=True)


if __name__ == "__main__":
    main()
