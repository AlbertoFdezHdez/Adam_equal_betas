from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_probe_ema_decomposition import (
    BASE_DIR,
    DEFAULT_RUN_DIR,
    TERMS,
    Reservoir,
    decompose_ema,
    format_float_for_name,
    load_json,
    shadow_ema,
    stable_seed,
    valid_and_drift,
    write_json,
)


DEFAULT_SINGLE_BETA_RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "ema_decomposition"
    / "probe_b10p9_b20p999_bs128_pbs128_s1234"
    / "W6"
)
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "ema_decomposition_multi_beta"

WINDOW = 6
SELECTED_JS = (0, 2, 5)
REGIMES = ("A", "B")
SIGNALS = ("m", "v")
RATIO_TERMS = ("lag", "trans", "curv")
THRESHOLDS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
MINIMUM_BETA_PAIRS = (
    (0.9, 0.9),
    (0.99, 0.99),
    (0.999, 0.999),
    (0.9, 0.99),
    (0.99, 0.999),
    (0.9, 0.999),
)
FULL_GRID_BETA_PAIRS = (
    (0.9, 0.9),
    (0.9, 0.99),
    (0.9, 0.999),
    (0.99, 0.9),
    (0.99, 0.99),
    (0.99, 0.999),
    (0.999, 0.9),
    (0.999, 0.99),
    (0.999, 0.999),
)


@dataclass
class BudgetStats:
    share_samples: dict[str, Reservoir]
    rel_samples: dict[str, Reservoir]
    count: int = 0
    share_sums: dict[str, float] = field(default_factory=lambda: {term: 0.0 for term in TERMS})
    dom_counts: dict[str, int] = field(default_factory=lambda: {term: 0 for term in TERMS})


@dataclass
class CoverageCounts:
    c_m: int = 0
    c_v: int = 0
    c_both: int = 0


@dataclass
class DriftStats:
    sample: Reservoir
    total_coord_windows: int = 0
    sign_stable_count: int = 0
    moderate_count: int = 0


def make_reservoir(seed: int, sample_size: int, *parts: object) -> Reservoir:
    return Reservoir(sample_size, np.random.default_rng(stable_seed(seed, *parts)))


def empty_budget_stats(seed: int, sample_size: int, *parts: object) -> BudgetStats:
    return BudgetStats(
        share_samples={
            term: make_reservoir(seed, sample_size, *parts, "share", term) for term in TERMS
        },
        rel_samples={
            term: make_reservoir(seed, sample_size, *parts, "relative", term)
            for term in RATIO_TERMS
        },
    )


def pct(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else float("nan")


def fmt_num(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return "NA"
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax >= 1e5 or ax < 1e-3:
        return f"{x:.2e}"
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def fmt_pct(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return "NA"
    if x != 0.0 and abs(x) < 0.1:
        return f"{x:.3g}%"
    if abs(x) < 1.0:
        return f"{x:.3f}%"
    return f"{x:.1f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    aligns = []
    for header in headers:
        if (
            header.endswith("%")
            or header.endswith("mean")
            or header.endswith("p50")
            or header.endswith("p90")
            or header.startswith("beta")
            or header in {"j", "W", "coord-windows", "threshold"}
        ):
            aligns.append("---:")
        else:
            aligns.append("---")
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def write_text_allow_long_path(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path_str = str(path.resolve())
    if len(path_str) >= 240 and not path_str.startswith("\\\\?\\"):
        path_str = "\\\\?\\" + path_str
    with open(path_str, "w", encoding="utf-8") as handle:
        handle.write(text)


def beta_label(beta1: float, beta2: float) -> str:
    return f"({beta1:g},{beta2:g})"


def output_name_for_run(config: dict[str, object], window: int) -> str:
    signal = str(config.get("saved_gradient_source", "probe"))
    beta1 = format_float_for_name(float(config.get("beta1", 0.9)))
    beta2 = format_float_for_name(float(config.get("beta2", 0.999)))
    batch_size = int(config.get("batch_size", 128))
    probe_batch = int(config.get("probe_batch_size") or batch_size)
    seed = int(config.get("seed", 1234))
    return f"{signal}_stream_b1{beta1}_b2{beta2}_bs{batch_size}_pbs{probe_batch}_s{seed}_W{window}"


def update_budget_stats(
    stats: BudgetStats,
    *,
    base: np.ndarray,
    lag: np.ndarray,
    trans: np.ndarray,
    curv: np.ndarray,
    mask: np.ndarray,
    relative_eps: float,
    share_eps: float,
) -> None:
    if not np.any(mask):
        return

    abs_base = np.abs(base[mask])
    abs_lag = np.abs(lag[mask])
    abs_trans = np.abs(trans[mask])
    abs_curv = np.abs(curv[mask])
    n = int(abs_base.size)
    if n == 0:
        return

    abs_terms = {
        "base": abs_base,
        "lag": abs_lag,
        "trans": abs_trans,
        "curv": abs_curv,
    }
    total = abs_base + abs_lag + abs_trans + abs_curv + share_eps
    dom = np.argmax(np.stack((abs_base, abs_lag, abs_trans, abs_curv), axis=0), axis=0)

    stats.count += n
    for term_idx, term in enumerate(TERMS):
        share = abs_terms[term] / total
        stats.share_sums[term] += float(np.sum(share, dtype=np.float64))
        stats.share_samples[term].add(share)
        stats.dom_counts[term] += int(np.count_nonzero(dom == term_idx))

    denom = abs_base + relative_eps
    stats.rel_samples["lag"].add(abs_lag / denom)
    stats.rel_samples["trans"].add(abs_trans / denom)
    stats.rel_samples["curv"].add(abs_curv / denom)


def add_signal_stats(
    *,
    budgets: dict[tuple[str, float, float, str, str, int], BudgetStats],
    beta_pairs: tuple[tuple[float, float], ...],
    active_beta: float,
    signal: str,
    regimes: dict[str, np.ndarray],
    base: np.ndarray,
    lag: np.ndarray,
    trans: np.ndarray,
    curv: np.ndarray,
    sample_size: int,
    seed: int,
    relative_eps: float,
    share_eps: float,
    js_to_process: tuple[int, ...],
) -> None:
    beta_index = 0 if signal == "m" else 1
    matching_pairs = [pair for pair in beta_pairs if pair[beta_index] == active_beta]
    for beta1, beta2 in matching_pairs:
        for regime in REGIMES:
            mask = regimes[regime]
            for j in js_to_process:
                key = ("replay", beta1, beta2, signal, regime, j)
                if key not in budgets:
                    budgets[key] = empty_budget_stats(seed, sample_size, *key)
                update_budget_stats(
                    budgets[key],
                    base=base[:, :, j],
                    lag=lag[:, :, j],
                    trans=trans[:, :, j],
                    curv=curv[:, :, j],
                    mask=mask,
                    relative_eps=relative_eps,
                    share_eps=share_eps,
                )


def scan_real_training_runs(
    runs_root: Path,
    requested_pairs: tuple[tuple[float, float], ...],
    source_run_dir: Path,
) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    if not runs_root.is_dir():
        return found
    requested_set = {(round(b1, 12), round(b2, 12)) for b1, b2 in requested_pairs}
    for run_dir in runs_root.iterdir():
        if run_dir.resolve() == source_run_dir.resolve():
            continue
        config_path = run_dir / "config.json"
        gradients_path = run_dir / "gradients.npy"
        if not config_path.is_file() or not gradients_path.is_file():
            continue
        try:
            config = load_json(config_path)
            pair = (round(float(config["beta1"]), 12), round(float(config["beta2"]), 12))
        except Exception:
            continue
        if pair in requested_set:
            found.append(
                {
                    "run_dir": str(run_dir),
                    "beta1": float(config["beta1"]),
                    "beta2": float(config["beta2"]),
                    "saved_gradient_source": str(config.get("saved_gradient_source", "unknown")),
                }
            )
    return found


def compute_replay(
    *,
    gradients: np.ndarray,
    beta_pairs: tuple[tuple[float, float], ...],
    window: int,
    chunk_size: int,
    eps_g: float,
    eps_v: float,
    relative_eps: float,
    share_eps: float,
    drift_threshold: float,
    sample_size: int,
    seed: int,
    js_to_process: tuple[int, ...],
    coord_indices: np.ndarray | None,
) -> tuple[
    dict[tuple[str, float, float, str, str, int], BudgetStats],
    dict[tuple[str, float, float, float], CoverageCounts],
    DriftStats,
    dict[tuple[str, float, float], dict[str, float]],
]:
    unique_betas = tuple(sorted({beta for pair in beta_pairs for beta in pair}))
    steps, n_params_total = gradients.shape
    if coord_indices is None:
        coord_indices = np.arange(n_params_total, dtype=np.int64)
    n_params = int(coord_indices.size)
    n_windows = steps - window
    total_coord_windows = n_windows * n_params
    drift_stats = DriftStats(
        sample=make_reservoir(seed, sample_size, "drift"),
        total_coord_windows=total_coord_windows,
    )
    budgets: dict[tuple[str, float, float, str, str, int], BudgetStats] = {}
    coverage = {
        ("replay", beta1, beta2, threshold): CoverageCounts()
        for beta1, beta2 in beta_pairs
        for threshold in THRESHOLDS
    }
    recon_errors = {
        ("replay", beta1, beta2): {"m": 0.0, "v": 0.0} for beta1, beta2 in beta_pairs
    }

    for start in range(0, n_params, chunk_size):
        stop = min(start + chunk_size, n_params)
        idx = coord_indices[start:stop]
        chunk = np.asarray(gradients[:, idx], dtype=np.float64)
        y = chunk * chunk
        valid, delta_windows, max_abs_delta, _ = valid_and_drift(chunk, window=window, eps_g=eps_g)
        moderate = valid & (max_abs_delta <= drift_threshold)
        regimes = {"A": valid, "B": moderate}

        drift_stats.sign_stable_count += int(np.count_nonzero(valid))
        drift_stats.moderate_count += int(np.count_nonzero(moderate))
        drift_stats.sample.add(max_abs_delta[valid])

        first_order_m = {}
        first_order_v = {}
        for beta in unique_betas:
            ell = beta / (1.0 - beta)
            first_order_m[beta] = np.max(np.abs(ell * (1.0 - np.exp(-delta_windows))), axis=-1)
            first_order_v[beta] = np.max(np.abs(ell * (1.0 - np.exp(-2.0 * delta_windows))), axis=-1)

        for beta1, beta2 in beta_pairs:
            for threshold in THRESHOLDS:
                c_m = valid & (first_order_m[beta1] <= threshold)
                c_v = valid & (first_order_v[beta2] <= threshold)
                counts = coverage[("replay", beta1, beta2, threshold)]
                counts.c_m += int(np.count_nonzero(c_m))
                counts.c_v += int(np.count_nonzero(c_v))
                counts.c_both += int(np.count_nonzero(c_m & c_v))

        for beta in unique_betas:
            shadow_m = shadow_ema(chunk, beta)
            base_m, lag_m, trans_m, curv_m, obs_m, recon_m = decompose_ema(
                chunk, shadow_m, beta=beta, window=window
            )
            err_m = float(np.max(np.abs(recon_m - obs_m)))
            for beta1, beta2 in beta_pairs:
                if beta1 == beta:
                    recon_errors[("replay", beta1, beta2)]["m"] = max(
                        recon_errors[("replay", beta1, beta2)]["m"], err_m
                    )
            add_signal_stats(
                budgets=budgets,
                beta_pairs=beta_pairs,
                active_beta=beta,
                signal="m",
                regimes=regimes,
                base=base_m,
                lag=lag_m,
                trans=trans_m,
                curv=curv_m,
                sample_size=sample_size,
                seed=seed,
                relative_eps=relative_eps if relative_eps > 0 else eps_g,
                share_eps=share_eps,
                js_to_process=js_to_process,
            )
            del shadow_m, base_m, lag_m, trans_m, curv_m, obs_m, recon_m

        for beta in unique_betas:
            shadow_v = shadow_ema(y, beta)
            base_v, lag_v, trans_v, curv_v, obs_v, recon_v = decompose_ema(
                y, shadow_v, beta=beta, window=window
            )
            err_v = float(np.max(np.abs(recon_v - obs_v)))
            for beta1, beta2 in beta_pairs:
                if beta2 == beta:
                    recon_errors[("replay", beta1, beta2)]["v"] = max(
                        recon_errors[("replay", beta1, beta2)]["v"], err_v
                    )
            add_signal_stats(
                budgets=budgets,
                beta_pairs=beta_pairs,
                active_beta=beta,
                signal="v",
                regimes=regimes,
                base=base_v,
                lag=lag_v,
                trans=trans_v,
                curv=curv_v,
                sample_size=sample_size,
                seed=seed + 10_000,
                relative_eps=relative_eps if relative_eps > 0 else eps_v,
                share_eps=share_eps,
                js_to_process=js_to_process,
            )
            del shadow_v, base_v, lag_v, trans_v, curv_v, obs_v, recon_v

        print(f"Processed analyzed coordinates {start}:{stop} / {n_params}", flush=True)

    return budgets, coverage, drift_stats, recon_errors


def budget_row(
    budgets: dict[tuple[str, float, float, str, str, int], BudgetStats],
    mode: str,
    beta1: float,
    beta2: float,
    signal: str,
    regime: str,
    j: int,
) -> dict[str, object]:
    stats = budgets[(mode, beta1, beta2, signal, regime, j)]
    row: dict[str, object] = {
        "mode": mode,
        "beta1": beta1,
        "beta2": beta2,
        "signal": signal,
        "regime": regime,
        "j": j,
        "coord_windows": stats.count,
    }
    for term in TERMS:
        row[f"{term}_share_mean"] = stats.share_sums[term] / stats.count if stats.count else float("nan")
    for term in TERMS:
        row[f"{term}_share_p50"] = stats.share_samples[term].percentile(50)
    for term in TERMS:
        row[f"{term}_dom_percent"] = pct(stats.dom_counts[term], stats.count)
    return row


def relative_row(
    budgets: dict[tuple[str, float, float, str, str, int], BudgetStats],
    mode: str,
    beta1: float,
    beta2: float,
    signal: str,
    regime: str,
    j: int,
) -> dict[str, object]:
    stats = budgets[(mode, beta1, beta2, signal, regime, j)]
    row: dict[str, object] = {
        "mode": mode,
        "beta1": beta1,
        "beta2": beta2,
        "signal": signal,
        "regime": regime,
        "j": j,
        "coord_windows": stats.count,
    }
    for term in RATIO_TERMS:
        qs = stats.rel_samples[term].percentiles((50, 90))
        row[f"{term}_base_p50"] = qs["p50"]
        row[f"{term}_base_p90"] = qs["p90"]
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def table_a_markdown(rows: list[dict[str, object]]) -> str:
    headers = [
        "mode",
        "beta1",
        "beta2",
        "signal",
        "regime",
        "j",
        "base share mean",
        "lag share mean",
        "trans share mean",
        "curv share mean",
        "base share p50",
        "lag share p50",
        "trans share p50",
        "curv share p50",
        "base dom %",
        "lag dom %",
        "trans dom %",
        "curv dom %",
    ]
    md_rows = []
    for row in rows:
        md_rows.append(
            [
                str(row["mode"]),
                fmt_num(row["beta1"]),
                fmt_num(row["beta2"]),
                str(row["signal"]),
                str(row["regime"]),
                str(row["j"]),
                fmt_pct(100.0 * float(row["base_share_mean"])),
                fmt_pct(100.0 * float(row["lag_share_mean"])),
                fmt_pct(100.0 * float(row["trans_share_mean"])),
                fmt_pct(100.0 * float(row["curv_share_mean"])),
                fmt_pct(100.0 * float(row["base_share_p50"])),
                fmt_pct(100.0 * float(row["lag_share_p50"])),
                fmt_pct(100.0 * float(row["trans_share_p50"])),
                fmt_pct(100.0 * float(row["curv_share_p50"])),
                fmt_pct(row["base_dom_percent"]),
                fmt_pct(row["lag_dom_percent"]),
                fmt_pct(row["trans_dom_percent"]),
                fmt_pct(row["curv_dom_percent"]),
            ]
        )
    return markdown_table(headers, md_rows)


def table_b_markdown(rows: list[dict[str, object]]) -> str:
    headers = [
        "mode",
        "beta1",
        "beta2",
        "signal",
        "regime",
        "j",
        "lag/base p50",
        "lag/base p90",
        "trans/base p50",
        "trans/base p90",
        "curv/base p50",
        "curv/base p90",
    ]
    md_rows = []
    for row in rows:
        md_rows.append(
            [
                str(row["mode"]),
                fmt_num(row["beta1"]),
                fmt_num(row["beta2"]),
                str(row["signal"]),
                str(row["regime"]),
                str(row["j"]),
                fmt_num(row["lag_base_p50"]),
                fmt_num(row["lag_base_p90"]),
                fmt_num(row["trans_base_p50"]),
                fmt_num(row["trans_base_p90"]),
                fmt_num(row["curv_base_p50"]),
                fmt_num(row["curv_base_p90"]),
            ]
        )
    return markdown_table(headers, md_rows)


def build_coverage_rows(
    *,
    beta_pairs: tuple[tuple[float, float], ...],
    coverage: dict[tuple[str, float, float, float], CoverageCounts],
    drift_stats: DriftStats,
    window: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    drift_qs = drift_stats.sample.percentiles((50, 90, 99))
    compact_rows = []
    threshold_rows = []
    for beta1, beta2 in beta_pairs:
        compact: dict[str, object] = {
            "mode": "replay",
            "beta1": beta1,
            "beta2": beta2,
            "W": window,
            "coord_windows": drift_stats.total_coord_windows,
            "sign_stable_percent": pct(
                drift_stats.sign_stable_count, drift_stats.total_coord_windows
            ),
            "drift_p50": drift_qs["p50"],
            "drift_p90": drift_qs["p90"],
            "drift_p99": drift_qs["p99"],
        }
        for threshold in THRESHOLDS:
            counts = coverage[("replay", beta1, beta2, threshold)]
            threshold_rows.append(
                {
                    "mode": "replay",
                    "beta1": beta1,
                    "beta2": beta2,
                    "threshold": threshold,
                    "C_m_coverage_percent": pct(counts.c_m, drift_stats.total_coord_windows),
                    "C_v_coverage_percent": pct(counts.c_v, drift_stats.total_coord_windows),
                    "C_both_coverage_percent": pct(counts.c_both, drift_stats.total_coord_windows),
                }
            )
        for threshold in (1.0, 10.0, 100.0):
            counts = coverage[("replay", beta1, beta2, threshold)]
            compact[f"C_m@{threshold:g}_percent"] = pct(counts.c_m, drift_stats.total_coord_windows)
            compact[f"C_v@{threshold:g}_percent"] = pct(counts.c_v, drift_stats.total_coord_windows)
            compact[f"C_both@{threshold:g}_percent"] = pct(
                counts.c_both, drift_stats.total_coord_windows
            )
        compact_rows.append(compact)
    return compact_rows, threshold_rows


def table_c_markdown(rows: list[dict[str, object]]) -> str:
    headers = [
        "mode",
        "beta1",
        "beta2",
        "W",
        "coord-windows",
        "sign-stable %",
        "drift p50",
        "drift p90",
        "drift p99",
        "C_m@1 %",
        "C_v@1 %",
        "C_both@1 %",
        "C_m@10 %",
        "C_v@10 %",
        "C_both@10 %",
        "C_m@100 %",
        "C_v@100 %",
        "C_both@100 %",
    ]
    md_rows = []
    for row in rows:
        md_rows.append(
            [
                str(row["mode"]),
                fmt_num(row["beta1"]),
                fmt_num(row["beta2"]),
                str(row["W"]),
                str(row["coord_windows"]),
                fmt_pct(row["sign_stable_percent"]),
                fmt_num(row["drift_p50"]),
                fmt_num(row["drift_p90"]),
                fmt_num(row["drift_p99"]),
                fmt_pct(row["C_m@1_percent"]),
                fmt_pct(row["C_v@1_percent"]),
                fmt_pct(row["C_both@1_percent"]),
                fmt_pct(row["C_m@10_percent"]),
                fmt_pct(row["C_v@10_percent"]),
                fmt_pct(row["C_both@10_percent"]),
                fmt_pct(row["C_m@100_percent"]),
                fmt_pct(row["C_v@100_percent"]),
                fmt_pct(row["C_both@100_percent"]),
            ]
        )
    return markdown_table(headers, md_rows)


def table_d_rows(
    beta_pairs: tuple[tuple[float, float], ...],
    recon_errors: dict[tuple[str, float, float], dict[str, float]],
) -> list[dict[str, object]]:
    rows = []
    for beta1, beta2 in beta_pairs:
        errors = recon_errors[("replay", beta1, beta2)]
        rows.append(
            {
                "mode": "replay",
                "beta1": beta1,
                "beta2": beta2,
                "max_recon_error_m": errors["m"],
                "max_recon_error_v": errors["v"],
            }
        )
    return rows


def table_d_markdown(rows: list[dict[str, object]]) -> str:
    headers = ["mode", "beta1", "beta2", "max recon error m", "max recon error v"]
    md_rows = [
        [
            str(row["mode"]),
            fmt_num(row["beta1"]),
            fmt_num(row["beta2"]),
            fmt_num(row["max_recon_error_m"]),
            fmt_num(row["max_recon_error_v"]),
        ]
        for row in rows
    ]
    return markdown_table(headers, md_rows)


def selected_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row.get("signal") in SIGNALS and row.get("regime") in REGIMES and int(row.get("j", -1)) in SELECTED_JS
    ]


def plot_share_stacked(
    path: Path,
    table_a_rows: list[dict[str, object]],
    beta_pairs: tuple[tuple[float, float], ...],
    regime: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"base": "#4c78a8", "lag": "#f58518", "trans": "#54a24b", "curv": "#e45756"}
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    labels = [beta_label(*pair) for pair in beta_pairs]
    x = np.arange(len(beta_pairs))
    for ax, signal in zip(axes, SIGNALS):
        bottom = np.zeros(len(beta_pairs), dtype=np.float64)
        for term in TERMS:
            vals = []
            for beta1, beta2 in beta_pairs:
                row = next(
                    r
                    for r in table_a_rows
                    if r["beta1"] == beta1
                    and r["beta2"] == beta2
                    and r["signal"] == signal
                    and r["regime"] == regime
                    and r["j"] == 5
                )
                vals.append(100.0 * float(row[f"{term}_share_mean"]))
            vals_arr = np.asarray(vals)
            ax.bar(x, vals_arr, bottom=bottom, label=term, color=colors[term], width=0.75)
            bottom += vals_arr
        ax.set_title(f"{signal}, regime {regime}, j=5")
        ax.set_ylabel("mean absolute share (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=40, ha="right")
    axes[0].legend(ncols=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_share_metric(
    path: Path,
    table_a_rows: list[dict[str, object]],
    beta_pairs: tuple[tuple[float, float], ...],
    term: str,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [beta_label(*pair) for pair in beta_pairs]
    x = np.arange(len(beta_pairs))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    for offset, signal in [(-width / 2, "m"), (width / 2, "v")]:
        vals = []
        for beta1, beta2 in beta_pairs:
            row = next(
                r
                for r in table_a_rows
                if r["beta1"] == beta1
                and r["beta2"] == beta2
                and r["signal"] == signal
                and r["regime"] == "A"
                and r["j"] == 5
            )
            vals.append(100.0 * float(row[f"{term}_share_mean"]))
        ax.bar(x + offset, vals, width=width, label=signal)
    ax.set_title(title)
    ax.set_ylabel("mean absolute share (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_coverage(
    path: Path,
    threshold_rows: list[dict[str, object]],
    beta_pairs: tuple[tuple[float, float], ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [beta_label(*pair) for pair in beta_pairs]
    x = np.arange(len(beta_pairs))
    width = 0.25
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10.0), sharex=True)
    colors = {"C_m": "#4c78a8", "C_v": "#f58518", "C_both": "#54a24b"}
    for ax, threshold in zip(axes, (1.0, 10.0, 100.0)):
        for offset, key in [(-width, "C_m"), (0.0, "C_v"), (width, "C_both")]:
            vals = []
            for beta1, beta2 in beta_pairs:
                row = next(
                    r
                    for r in threshold_rows
                    if r["beta1"] == beta1
                    and r["beta2"] == beta2
                    and r["threshold"] == threshold
                )
                vals.append(float(row[f"{key}_coverage_percent"]))
            ax.bar(x + offset, vals, width=width, label=key, color=colors[key])
        ax.set_title(f"threshold = {threshold:g}")
        ax.set_ylabel("coverage (%)")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=40, ha="right")
    axes[0].legend(ncols=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.35))
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def get_a_row(
    rows: list[dict[str, object]],
    beta1: float,
    beta2: float,
    signal: str,
    regime: str = "A",
    j: int = 5,
) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row["mode"] == "replay"
        and row["beta1"] == beta1
        and row["beta2"] == beta2
        and row["signal"] == signal
        and row["regime"] == regime
        and row["j"] == j
    )


def interpret_report(
    *,
    table_a_rows: list[dict[str, object]],
    coverage_compact_rows: list[dict[str, object]],
    beta_pairs: tuple[tuple[float, float], ...],
) -> tuple[str, str, str, str]:
    m_base_best = max(
        beta_pairs, key=lambda pair: float(get_a_row(table_a_rows, *pair, "m")["base_share_mean"])
    )
    v_base_best = max(
        beta_pairs, key=lambda pair: float(get_a_row(table_a_rows, *pair, "v")["base_share_mean"])
    )
    current = (0.9, 0.999)
    diagonals = [pair for pair in beta_pairs if pair[0] == pair[1]]
    current_m = get_a_row(table_a_rows, *current, "m") if current in beta_pairs else None
    current_v = get_a_row(table_a_rows, *current, "v") if current in beta_pairs else None
    best_diag_m = max(diagonals, key=lambda pair: float(get_a_row(table_a_rows, *pair, "m")["base_share_mean"]))
    best_diag_v = max(diagonals, key=lambda pair: float(get_a_row(table_a_rows, *pair, "v")["base_share_mean"]))

    def non_base_main(row: dict[str, object]) -> str:
        values = {
            "lag": float(row["lag_share_mean"]),
            "trans": float(row["trans_share_mean"]),
            "curv": float(row["curv_share_mean"]),
        }
        return max(values, key=values.get)

    lines_coverage = [
        "The sign-stable and drift columns are identical across beta pairs because this is a replay on the same probe-gradient stream. The perturbative columns change because ell1 and ell2 depend on beta."
    ]
    cov_by_pair = {
        (float(row["beta1"]), float(row["beta2"])): row for row in coverage_compact_rows
    }
    smaller_beta2 = [pair for pair in beta_pairs if pair[1] < 0.999]
    best_cv10 = max(smaller_beta2, key=lambda pair: float(cov_by_pair[pair]["C_v@10_percent"]))
    lines_coverage.append(
        f"At threshold 10, the largest C_v coverage among beta2<0.999 is {beta_label(*best_cv10)} with {fmt_pct(cov_by_pair[best_cv10]['C_v@10_percent'])}."
    )

    lines_share = [
        f"The largest base share for m at j=5, regime A, occurs at beta={beta_label(*m_base_best)} with {fmt_pct(100.0 * float(get_a_row(table_a_rows, *m_base_best, 'm')['base_share_mean']))}.",
        f"The largest base share for v at j=5, regime A, occurs at beta={beta_label(*v_base_best)} with {fmt_pct(100.0 * float(get_a_row(table_a_rows, *v_base_best, 'v')['base_share_mean']))}.",
    ]
    if current_m and current_v:
        lines_share.append(
            f"Compared with (0.9,0.999), the best diagonal m base share is {fmt_pct(100.0 * float(get_a_row(table_a_rows, *best_diag_m, 'm')['base_share_mean']))} versus {fmt_pct(100.0 * float(current_m['base_share_mean']))}; for v it is {fmt_pct(100.0 * float(get_a_row(table_a_rows, *best_diag_v, 'v')['base_share_mean']))} versus {fmt_pct(100.0 * float(current_v['base_share_mean']))}."
        )

    current_text = []
    if current_m and current_v:
        current_text.append(
            f"For the original beta pair (0.9,0.999), the dominant non-base mean-share term at j=5 is {non_base_main(current_m)} for m and {non_base_main(current_v)} for v."
        )

    trans_dominant_pairs = []
    curv_dominant_pairs = []
    for pair in beta_pairs:
        for signal in SIGNALS:
            row = get_a_row(table_a_rows, *pair, signal)
            main = non_base_main(row)
            if main == "trans":
                trans_dominant_pairs.append((pair, signal))
            if main == "curv":
                curv_dominant_pairs.append((pair, signal))
    if trans_dominant_pairs:
        current_text.append(
            "The transient is the largest non-base mean-share term in some cases: "
            + ", ".join(f"{sig}@{beta_label(*pair)}" for pair, sig in trans_dominant_pairs)
            + "."
        )
    else:
        current_text.append(
            "The transient is not the largest non-base mean-share term at j=5, regime A, for any replayed pair in this grid."
        )
    if curv_dominant_pairs:
        current_text.append(
            "Curvature is the largest non-base mean-share term in: "
            + ", ".join(f"{sig}@{beta_label(*pair)}" for pair, sig in curv_dominant_pairs[:12])
            + ("." if len(curv_dominant_pairs) <= 12 else ", ...")
        )

    v_base_by_pair = [
        float(get_a_row(table_a_rows, *pair, "v")["base_share_mean"]) for pair in beta_pairs
    ]
    m_base_by_pair = [
        float(get_a_row(table_a_rows, *pair, "m")["base_share_mean"]) for pair in beta_pairs
    ]
    bottleneck_text = (
        f"Across the grid, v base share ranges from {fmt_pct(100.0 * min(v_base_by_pair))} to {fmt_pct(100.0 * max(v_base_by_pair))}, while m ranges from {fmt_pct(100.0 * min(m_base_by_pair))} to {fmt_pct(100.0 * max(m_base_by_pair))}. This is consistent with v being the more beta-sensitive bottleneck, especially at beta2=0.999."
    )

    main_conclusions = "\n".join(
        [
            f"- Same-gradient replay suggests the largest m base share at j=5 is obtained by {beta_label(*m_base_best)}.",
            f"- Same-gradient replay suggests the largest v base share at j=5 is obtained by {beta_label(*v_base_best)}.",
            f"- {bottleneck_text}",
            "- These replay results isolate the EMA mechanism. They do not by themselves prove that a real training run with those betas would follow the same gradient trajectory.",
        ]
    )
    theory_text = "\n".join(
        [
            "- The results distinguish the transient from curvature: a large transient would weaken the claim that local history has decayed, while large curvature indicates that W=6 is not quantitatively first-order.",
            "- If smaller beta2 produces substantially larger v base share and perturbative coverage, that supports the view that the practical validity of the expansion depends strongly on EMA time scale.",
            "- If beta2=0.999 remains non-perturbative even on the diagonal, this is consistent with the second-moment expansion being the main quantitative bottleneck at very long memory.",
            "- Same-gradient replay is deliberately controlled: it isolates the EMA parameters on the same g_k stream, but it is not a substitute for per-beta training diagnostics.",
        ]
    )
    return (
        "\n".join(lines_coverage),
        "\n".join(lines_share),
        "\n".join(current_text),
        main_conclusions + "\n\n" + theory_text,
    )


def build_report(
    *,
    output_dir: Path,
    run_dir: Path,
    config: dict[str, object],
    probe_metadata: dict[str, object] | None,
    real_runs_found: list[dict[str, object]],
    beta_pairs: tuple[tuple[float, float], ...],
    drift_threshold: float,
    n_params_total: int,
    n_params_analyzed: int,
    coord_sample_seed: int | None,
    table_a_rows_selected: list[dict[str, object]],
    table_b_rows_selected: list[dict[str, object]],
    table_c_rows: list[dict[str, object]],
    threshold_rows: list[dict[str, object]],
    table_d: list[dict[str, object]],
    js_to_process: tuple[int, ...],
) -> str:
    coverage_interp, share_interp, nonbase_interp, conclusions = interpret_report(
        table_a_rows=table_a_rows_selected,
        coverage_compact_rows=table_c_rows,
        beta_pairs=beta_pairs,
    )
    steps = int(np.load(run_dir / "gradients.npy", mmap_mode="r").shape[0])
    n_params = n_params_analyzed
    n_windows = steps - WINDOW
    coord_windows = n_windows * n_params
    probe_batch = probe_metadata.get("batch_size") if probe_metadata else config.get("probe_batch_size") or config.get("batch_size")

    requested_pair_set = {(round(b1, 12), round(b2, 12)) for b1, b2 in beta_pairs}
    found_pair_set = {
        (round(float(run["beta1"]), 12), round(float(run["beta2"]), 12)) for run in real_runs_found
    }
    if not real_runs_found or not requested_pair_set.issubset(found_pair_set):
        mode2_note = (
            "A complete separate real per-beta probe-gradient grid was not found, so this report only uses Mode 1."
        )
        if real_runs_found:
            mode2_note += (
                " Additional runs were detected but not mixed into the replay tables: "
                + ", ".join(
                    f"({run['beta1']},{run['beta2']})[{run['saved_gradient_source']}]"
                    for run in real_runs_found
                )
                + "."
            )
    else:
        mode2_note = (
            "A complete real per-beta grid was detected, but this script keeps it separate and does not mix it into the replay tables."
        )

    report = f"""# EMA Decomposition Multi-Beta Interpretive Summary

## 0. Setup and caveats

- Mode analyzed in the tables: **same-gradient-stream replay**.
- Source run: `{run_dir}`.
- Dataset: `{config.get('dataset', 'unknown')}`.
- Model: `TinyCIFARCNN`.
- Stored gradient source: `{config.get('saved_gradient_source', 'unknown')}`.
- Probe batch size: `{probe_batch}`.
- Stored gradient matrix: `{steps}` steps x `{n_params_total}` coordinates.
- Total trainable coordinates in the stored matrix: `{n_params_total}`.
- Coordinates analyzed in this report: `{n_params_analyzed}`{f" (deterministic sample, seed={coord_sample_seed})" if coord_sample_seed is not None else " (all coordinates)"}.
- Window length: `W={WINDOW}`, using `g_{{k0-1}},...,g_{{k0+5}}`.
- Coordinate-windows analyzed: `{coord_windows}`.
- Regime A: sign-stable and no effective zeros.
- Regime B: Regime A plus `max|delta| <= {drift_threshold:.6g}`. This is the same drift threshold as the previous single-beta diagnostic.
- Beta pairs replayed: {', '.join(beta_label(*pair) for pair in beta_pairs)}.
- j values stored in the main CSV tables: {', '.join(str(j) for j in js_to_process)}.

This is a same-gradient-stream replay diagnostic. It isolates the effect of the EMA parameters on the same probe-gradient trajectory. It does not claim that the model trajectory would be the same under those betas.

Per-beta training diagnostics reflect the actual optimizer trajectory for each beta pair, but are less controlled because `g_k` changes with the training dynamics. {mode2_note}

## 1. Reconstruction sanity check

Table D. Reconstruction sanity check

{table_d_markdown(table_d)}

The exact discrete identity is numerically satisfied if these errors are near floating-point precision.

## 2. Window and perturbative coverage

Table C. Window and perturbative coverage

{table_c_markdown(table_c_rows)}

Short interpretation:

{coverage_interp}

The full threshold table is saved as `table_C_full_threshold_coverage.csv`.

## 3. Absolute share budget

Table A. Absolute share budget

{table_a_markdown(table_a_rows_selected)}

Short interpretation:

{share_interp}

{nonbase_interp}

The selected-j table is saved as `table_A_absolute_share_budget_selected_j.csv`. Re-run with `--all-j` to also store every `j=0,...,5`.

## 4. Relative-to-base error magnitudes

Table B. Relative-to-base error magnitudes

{table_b_markdown(table_b_rows_selected)}

Short interpretation:

Relative-to-base magnitudes can be large even when absolute shares are moderate. The share budget is the most direct way to ask whether the decomposition is dominated by base, lag, transitory history, or curvature; the relative table asks whether each correction is small compared with `g` or `g^2`.

The selected-j table is saved as `table_B_relative_to_base_selected_j.csv`. Re-run with `--all-j` to also store every `j=0,...,5`.

## 5. Main conclusions

{conclusions.split("\n\n")[0]}

## 6. What this says about the theory

{conclusions.split("\n\n")[1]}

## Plots

- `plots/plot1_share_budget_stacked_j5_regimeA.png`
- `plots/plot1_share_budget_stacked_j5_regimeB.png`
- `plots/plot2_base_share_vs_beta_j5_regimeA.png`
- `plots/plot3_curv_share_vs_beta_j5_regimeA.png`
- `plots/plot4_perturbative_coverage_vs_beta.png`
"""
    report_path = output_dir / "ema_decomposition_multi_beta_interpretive_summary.md"
    write_text_allow_long_path(report_path, report)
    return report


def parse_beta_pairs(raw: list[str] | None) -> tuple[tuple[float, float], ...]:
    if not raw:
        return MINIMUM_BETA_PAIRS
    pairs = []
    for item in raw:
        if "," not in item:
            raise ValueError(f"Expected beta pair as beta1,beta2, got {item!r}")
        left, right = item.split(",", maxsplit=1)
        pairs.append((float(left), float(right)))
    return tuple(pairs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Same-gradient-stream replay EMA decomposition for multiple beta pairs."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--single-beta-results-dir", type=Path, default=DEFAULT_SINGLE_BETA_RESULTS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--beta-pair", action="append", default=None, help="Format: beta1,beta2")
    parser.add_argument("--full-grid", action="store_true", help="Use all 9 pairs in {0.9,0.99,0.999}^2.")
    parser.add_argument("--all-j", action="store_true", help="Store all j=0,...,5 instead of only j=0,2,5.")
    parser.add_argument(
        "--coord-sample-size",
        type=int,
        default=0,
        help="If >0, analyze a deterministic sample of this many coordinates instead of all coordinates.",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--sample-size", type=int, default=40_000)
    parser.add_argument("--eps-g", type=float, default=0.0)
    parser.add_argument("--eps-v", type=float, default=0.0)
    parser.add_argument("--relative-eps", type=float, default=1e-30)
    parser.add_argument("--share-eps", type=float, default=1e-30)
    parser.add_argument("--seed", type=int, default=20260506)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = load_json(run_dir / "config.json")
    probe_metadata_path = run_dir / "probe_batch_metadata.json"
    probe_metadata = load_json(probe_metadata_path) if probe_metadata_path.is_file() else None
    beta_pairs = FULL_GRID_BETA_PAIRS if args.full_grid and not args.beta_pair else parse_beta_pairs(args.beta_pair)
    js_to_process = tuple(range(WINDOW)) if args.all_j else SELECTED_JS
    gradients = np.load(run_dir / "gradients.npy", mmap_mode="r")
    if gradients.ndim != 2:
        raise ValueError(f"Expected gradients with shape (steps, parameters), got {gradients.shape}")
    if gradients.shape[0] <= WINDOW:
        raise ValueError(f"Need more than W={WINDOW} steps, got {gradients.shape[0]}")
    if args.coord_sample_size > 0:
        sample_size_coords = min(args.coord_sample_size, gradients.shape[1])
        rng = np.random.default_rng(args.seed)
        coord_indices = np.sort(rng.choice(gradients.shape[1], size=sample_size_coords, replace=False))
        coord_sample_seed: int | None = args.seed
    else:
        coord_indices = None
        coord_sample_seed = None

    single_summary_path = args.single_beta_results_dir / "diagnostic_summary.json"
    if single_summary_path.is_file():
        drift_threshold = float(load_json(single_summary_path)["drift_threshold_for_regime_B"])
    else:
        raise FileNotFoundError(
            f"Need previous diagnostic summary for comparable Regime B threshold: {single_summary_path}"
        )

    output_name = args.output_name or output_name_for_run(config, WINDOW)
    output_dir = args.results_dir.resolve() / output_name
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    real_runs_found = scan_real_training_runs(run_dir.parent, beta_pairs, source_run_dir=run_dir)

    print(
        f"Loaded replay stream shape={gradients.shape}, dtype={gradients.dtype}; beta_pairs={beta_pairs}",
        flush=True,
    )
    budgets, coverage, drift_stats, recon_errors = compute_replay(
        gradients=gradients,
        beta_pairs=beta_pairs,
        window=WINDOW,
        chunk_size=args.chunk_size,
        eps_g=args.eps_g,
        eps_v=args.eps_v,
        relative_eps=args.relative_eps,
        share_eps=args.share_eps,
        drift_threshold=drift_threshold,
        sample_size=args.sample_size,
        seed=args.seed,
        js_to_process=js_to_process,
        coord_indices=coord_indices,
    )

    table_a_all = [
        budget_row(budgets, "replay", beta1, beta2, signal, regime, j)
        for beta1, beta2 in beta_pairs
        for signal in SIGNALS
        for regime in REGIMES
        for j in js_to_process
    ]
    table_a_selected = [
        row for row in table_a_all if row["regime"] in REGIMES and int(row["j"]) in SELECTED_JS
    ]
    table_b_all = [
        relative_row(budgets, "replay", beta1, beta2, signal, regime, j)
        for beta1, beta2 in beta_pairs
        for signal in SIGNALS
        for regime in REGIMES
        for j in js_to_process
    ]
    table_b_selected = [
        row for row in table_b_all if row["regime"] in REGIMES and int(row["j"]) in SELECTED_JS
    ]
    table_c_compact, table_c_thresholds = build_coverage_rows(
        beta_pairs=beta_pairs,
        coverage=coverage,
        drift_stats=drift_stats,
        window=WINDOW,
    )
    table_d = table_d_rows(beta_pairs, recon_errors)
    max_err_m = max(float(row["max_recon_error_m"]) for row in table_d)
    max_err_v = max(float(row["max_recon_error_v"]) for row in table_d)
    if max_err_m > 1e-8 or max_err_v > 1e-6:
        raise RuntimeError(
            f"Reconstruction error too large: max m={max_err_m:.3e}, max v={max_err_v:.3e}"
        )

    if args.all_j:
        write_csv(output_dir / "table_A_absolute_share_budget_all_j.csv", table_a_all)
        write_csv(output_dir / "table_B_relative_to_base_all_j.csv", table_b_all)
    write_csv(output_dir / "table_A_absolute_share_budget_selected_j.csv", table_a_selected)
    write_csv(output_dir / "table_B_relative_to_base_selected_j.csv", table_b_selected)
    write_csv(output_dir / "table_C_window_and_perturbative_coverage.csv", table_c_compact)
    write_csv(output_dir / "table_C_full_threshold_coverage.csv", table_c_thresholds)
    write_csv(output_dir / "table_D_reconstruction_sanity_check.csv", table_d)

    plot_share_stacked(
        plots_dir / "plot1_share_budget_stacked_j5_regimeA",
        table_a_selected,
        beta_pairs,
        regime="A",
    )
    plot_share_stacked(
        plots_dir / "plot1_share_budget_stacked_j5_regimeB",
        table_a_selected,
        beta_pairs,
        regime="B",
    )
    plot_share_metric(
        plots_dir / "plot2_base_share_vs_beta_j5_regimeA",
        table_a_selected,
        beta_pairs,
        term="base",
        title="Base share versus beta pair, regime A, j=5",
    )
    plot_share_metric(
        plots_dir / "plot3_curv_share_vs_beta_j5_regimeA",
        table_a_selected,
        beta_pairs,
        term="curv",
        title="Curvature share versus beta pair, regime A, j=5",
    )
    plot_coverage(
        plots_dir / "plot4_perturbative_coverage_vs_beta",
        table_c_thresholds,
        beta_pairs,
    )

    report = build_report(
        output_dir=output_dir,
        run_dir=run_dir,
        config=config,
        probe_metadata=probe_metadata,
        real_runs_found=real_runs_found,
        beta_pairs=beta_pairs,
        drift_threshold=drift_threshold,
        n_params_total=int(gradients.shape[1]),
        n_params_analyzed=int(coord_indices.size) if coord_indices is not None else int(gradients.shape[1]),
        coord_sample_seed=coord_sample_seed,
        table_a_rows_selected=table_a_selected,
        table_b_rows_selected=table_b_selected,
        table_c_rows=table_c_compact,
        threshold_rows=table_c_thresholds,
        table_d=table_d,
        js_to_process=js_to_process,
    )

    summary = {
        "mode": "same-gradient-stream replay",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "beta_pairs": beta_pairs,
        "window": WINDOW,
        "gradient_shape": [int(gradients.shape[0]), int(gradients.shape[1])],
        "n_parameters_analyzed": int(coord_indices.size) if coord_indices is not None else int(gradients.shape[1]),
        "coord_sample_seed": coord_sample_seed,
        "drift_threshold_for_regime_B": drift_threshold,
        "sample_size": args.sample_size,
        "chunk_size": args.chunk_size,
        "j_values_processed": js_to_process,
        "real_training_runs_found": real_runs_found,
    }
    write_json(output_dir / "multi_beta_diagnostic_summary.json", summary)
    print(report, flush=True)
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
