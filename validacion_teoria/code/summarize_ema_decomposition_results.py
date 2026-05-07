from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analyze_probe_ema_decomposition import (
    BASE_DIR,
    REGIMES,
    Reservoir,
    decompose_ema,
    load_json,
    shadow_ema,
    valid_and_drift,
)


DEFAULT_RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "ema_decomposition"
    / "probe_b10p9_b20p999_bs128_pbs128_s1234"
    / "W6"
)
SELECTED_JS = (0, 2, 5)
SELECTED_REGIMES = ("A", "B")
RATIO_NAMES = ("trans/lag", "trans/curv", "curv/lag")
THRESHOLDS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


@dataclass
class JointStats:
    count: int = 0
    dom_lag: int = 0
    dom_trans: int = 0
    dom_curv: int = 0
    trans_gt_lag: int = 0
    trans_gt_curv: int = 0
    ratio_samples: dict[str, Reservoir] = field(default_factory=dict)


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
    aligns = ["---"] * len(headers)
    numeric_suffixes = ("%", "p50", "p90", "j", "coverage %")
    for idx, header in enumerate(headers):
        if header.strip() == "j" or header.endswith(numeric_suffixes):
            aligns[idx] = "---:"
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def read_decomposition_rows(results_dir: Path) -> dict[tuple[str, str, int, str], dict[str, str]]:
    rows_by_key: dict[tuple[str, str, int, str], dict[str, str]] = {}
    for moment, filename in (
        ("m", "table2_decomposition_m_j025.csv"),
        ("v", "table3_decomposition_v_j025.csv"),
    ):
        path = results_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (moment, row["regime"], int(row["j"]), row["term"])
                rows_by_key[key] = row
    return rows_by_key


def ratio_reservoir(seed: int, sample_size: int, *parts: object) -> Reservoir:
    text = "|".join(str(part) for part in parts)
    offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 1_000_000_000
    return Reservoir(sample_size, np.random.default_rng(seed + offset))


def empty_joint_stats(seed: int, sample_size: int, signal: str, regime: str, j: int) -> JointStats:
    return JointStats(
        ratio_samples={
            name: ratio_reservoir(seed, sample_size, signal, regime, j, name) for name in RATIO_NAMES
        }
    )


def add_joint_stats(
    stats: JointStats,
    *,
    abs_lag: np.ndarray,
    abs_trans: np.ndarray,
    abs_curv: np.ndarray,
    mask: np.ndarray,
    ratio_eps: float,
) -> None:
    if not np.any(mask):
        return
    lag_vals = abs_lag[mask]
    trans_vals = abs_trans[mask]
    curv_vals = abs_curv[mask]
    n = int(lag_vals.size)
    stats.count += n

    dom = np.argmax(np.stack((lag_vals, trans_vals, curv_vals), axis=0), axis=0)
    stats.dom_lag += int(np.count_nonzero(dom == 0))
    stats.dom_trans += int(np.count_nonzero(dom == 1))
    stats.dom_curv += int(np.count_nonzero(dom == 2))
    stats.trans_gt_lag += int(np.count_nonzero(trans_vals > lag_vals))
    stats.trans_gt_curv += int(np.count_nonzero(trans_vals > curv_vals))

    stats.ratio_samples["trans/lag"].add(trans_vals / (lag_vals + ratio_eps))
    stats.ratio_samples["trans/curv"].add(trans_vals / (curv_vals + ratio_eps))
    stats.ratio_samples["curv/lag"].add(curv_vals / (lag_vals + ratio_eps))


def accumulate_signal_joint_stats(
    all_stats: dict[tuple[str, str, int], JointStats],
    *,
    signal: str,
    regimes: dict[str, np.ndarray],
    lag: np.ndarray,
    trans: np.ndarray,
    curv: np.ndarray,
    ratio_eps: float,
    sample_size: int,
    seed: int,
) -> None:
    for regime in SELECTED_REGIMES:
        mask = regimes[regime]
        for j in SELECTED_JS:
            key = (signal, regime, j)
            if key not in all_stats:
                all_stats[key] = empty_joint_stats(seed, sample_size, signal, regime, j)
            add_joint_stats(
                all_stats[key],
                abs_lag=np.abs(lag[:, :, j]),
                abs_trans=np.abs(trans[:, :, j]),
                abs_curv=np.abs(curv[:, :, j]),
                mask=mask,
                ratio_eps=ratio_eps,
            )


def compute_joint_stats_and_coverage(
    *,
    summary: dict[str, object],
    chunk_size: int,
    ratio_eps: float,
    sample_size: int,
    seed: int,
) -> tuple[dict[tuple[str, str, int], JointStats], dict[float, dict[str, int]], int]:
    gradients_path = Path(str(summary["gradients_path"]))
    gradients = np.load(gradients_path, mmap_mode="r")
    if gradients.ndim != 2:
        raise ValueError(f"Expected gradients with shape (steps, parameters), got {gradients.shape}")

    beta1 = float(summary["beta1"])
    beta2 = float(summary["beta2"])
    ell1 = float(summary["ell1"])
    ell2 = float(summary["ell2"])
    window = int(summary["window"])
    eps_g = float(summary["eps_g"])
    drift_threshold = float(summary["drift_threshold_for_regime_B"])

    steps, n_params = gradients.shape
    n_windows = steps - window
    total_coord_windows = n_windows * n_params
    joint_stats: dict[tuple[str, str, int], JointStats] = {}
    coverage = {threshold: {"C_m": 0, "C_v": 0, "C_both": 0} for threshold in THRESHOLDS}

    for start in range(0, n_params, chunk_size):
        stop = min(start + chunk_size, n_params)
        chunk = np.asarray(gradients[:, start:stop], dtype=np.float64)
        valid, delta_windows, max_abs_delta, _ = valid_and_drift(chunk, window=window, eps_g=eps_g)
        first_order_m = np.max(np.abs(ell1 * (1.0 - np.exp(-delta_windows))), axis=-1)
        first_order_v = np.max(np.abs(ell2 * (1.0 - np.exp(-2.0 * delta_windows))), axis=-1)

        for threshold in THRESHOLDS:
            c_m = valid & (first_order_m <= threshold)
            c_v = valid & (first_order_v <= threshold)
            coverage[threshold]["C_m"] += int(np.count_nonzero(c_m))
            coverage[threshold]["C_v"] += int(np.count_nonzero(c_v))
            coverage[threshold]["C_both"] += int(np.count_nonzero(c_m & c_v))

        regimes = {
            "A": valid,
            "B": valid & (max_abs_delta <= drift_threshold),
        }

        shadow_m = shadow_ema(chunk, beta1)
        _, lag_m, trans_m, curv_m, _, _ = decompose_ema(chunk, shadow_m, beta=beta1, window=window)
        accumulate_signal_joint_stats(
            joint_stats,
            signal="m",
            regimes=regimes,
            lag=lag_m,
            trans=trans_m,
            curv=curv_m,
            ratio_eps=ratio_eps,
            sample_size=sample_size,
            seed=seed,
        )

        y = chunk * chunk
        shadow_v = shadow_ema(y, beta2)
        _, lag_v, trans_v, curv_v, _, _ = decompose_ema(y, shadow_v, beta=beta2, window=window)
        accumulate_signal_joint_stats(
            joint_stats,
            signal="v",
            regimes=regimes,
            lag=lag_v,
            trans=trans_v,
            curv=curv_v,
            ratio_eps=ratio_eps,
            sample_size=sample_size,
            seed=seed + 10_000,
        )

        print(f"Processed coordinates {start}:{stop} / {n_params}", flush=True)

    return joint_stats, coverage, total_coord_windows


def build_term_budget_table(
    rows_by_key: dict[tuple[str, str, int, str], dict[str, str]],
    joint_stats: dict[tuple[str, str, int], JointStats],
) -> str:
    headers = [
        "signal",
        "regime",
        "j",
        "lag p50",
        "lag p90",
        "trans p50",
        "trans p90",
        "curv p50",
        "curv p90",
        "dom lag %",
        "dom trans %",
        "dom curv %",
        "trans>lag %",
        "trans>curv %",
    ]
    rows: list[list[str]] = []
    for signal in ("m", "v"):
        for regime in SELECTED_REGIMES:
            for j in SELECTED_JS:
                lag = rows_by_key[(signal, regime, j, "lag")]
                trans = rows_by_key[(signal, regime, j, "trans")]
                curv = rows_by_key[(signal, regime, j, "curv")]
                stats = joint_stats[(signal, regime, j)]
                rows.append(
                    [
                        signal,
                        regime,
                        str(j),
                        fmt_num(lag["rel_p50"]),
                        fmt_num(lag["rel_p90"]),
                        fmt_num(trans["rel_p50"]),
                        fmt_num(trans["rel_p90"]),
                        fmt_num(curv["rel_p50"]),
                        fmt_num(curv["rel_p90"]),
                        fmt_pct(pct(stats.dom_lag, stats.count)),
                        fmt_pct(pct(stats.dom_trans, stats.count)),
                        fmt_pct(pct(stats.dom_curv, stats.count)),
                        fmt_pct(pct(stats.trans_gt_lag, stats.count)),
                        fmt_pct(pct(stats.trans_gt_curv, stats.count)),
                    ]
                )
    return markdown_table(headers, rows)


def build_ratio_table(joint_stats: dict[tuple[str, str, int], JointStats]) -> str:
    headers = [
        "signal",
        "regime",
        "j",
        "trans/lag p50",
        "trans/lag p90",
        "trans/curv p50",
        "trans/curv p90",
        "curv/lag p50",
        "curv/lag p90",
    ]
    rows: list[list[str]] = []
    for signal in ("m", "v"):
        for regime in SELECTED_REGIMES:
            for j in SELECTED_JS:
                stats = joint_stats[(signal, regime, j)]
                ratio_values = []
                for name in RATIO_NAMES:
                    qs = stats.ratio_samples[name].percentiles((50, 90))
                    ratio_values.extend([fmt_num(qs["p50"]), fmt_num(qs["p90"])])
                rows.append([signal, regime, str(j), *ratio_values])
    return markdown_table(headers, rows)


def build_coverage_table(coverage: dict[float, dict[str, int]], total_coord_windows: int) -> str:
    headers = ["threshold", "C_m coverage %", "C_v coverage %", "C_both coverage %"]
    rows = []
    for threshold in THRESHOLDS:
        counts = coverage[threshold]
        rows.append(
            [
                fmt_num(threshold),
                fmt_pct(pct(counts["C_m"], total_coord_windows)),
                fmt_pct(pct(counts["C_v"], total_coord_windows)),
                fmt_pct(pct(counts["C_both"], total_coord_windows)),
            ]
        )
    return markdown_table(headers, rows)


def interpretation(
    rows_by_key: dict[tuple[str, str, int, str], dict[str, str]],
    joint_stats: dict[tuple[str, str, int], JointStats],
    coverage: dict[float, dict[str, int]],
    total_coord_windows: int,
) -> str:
    lines = ["Interpretation:"]
    for signal in ("m", "v"):
        stats = joint_stats[(signal, "A", 5)]
        dom = {
            "lag": pct(stats.dom_lag, stats.count),
            "trans": pct(stats.dom_trans, stats.count),
            "curv": pct(stats.dom_curv, stats.count),
        }
        main = max(dom, key=dom.get)
        lag_p90 = float(rows_by_key[(signal, "A", 5, "lag")]["rel_p90"])
        trans_p90 = float(rows_by_key[(signal, "A", 5, "trans")]["rel_p90"])
        curv_p90 = float(rows_by_key[(signal, "A", 5, "curv")]["rel_p90"])
        lines.append(
            f"- For {signal}, at j=5 in regime A, the largest non-base error term is usually "
            f"{main}: dom lag={dom['lag']:.1f}%, dom trans={dom['trans']:.1f}%, dom curv={dom['curv']:.1f}%. "
            f"The p90 values are lag={fmt_num(lag_p90)}, trans={fmt_num(trans_p90)}, curv={fmt_num(curv_p90)}."
        )

    m_stats = joint_stats[("m", "A", 5)]
    v_stats = joint_stats[("v", "A", 5)]
    if pct(m_stats.dom_trans, m_stats.count) < pct(m_stats.dom_curv, m_stats.count):
        lines.append(
            "- The transient term is present, but for m it is not the main obstruction at the end of W=6 windows; curvature dominates more often."
        )
    else:
        lines.append(
            "- For m, the transient term is competitive with the other error terms, so saying it has decayed would require care."
        )

    if pct(v_stats.dom_curv, v_stats.count) > max(pct(v_stats.dom_lag, v_stats.count), pct(v_stats.dom_trans, v_stats.count)):
        lines.append(
            "- For v, curvature is the clearest obstruction in this diagnostic: it dominates more often than lag or transient at j=5."
        )
    else:
        lines.append(
            "- For v, no single correction term cleanly dominates; lag, transient and curvature should all be treated as relevant."
        )

    c_m_1 = pct(coverage[1.0]["C_m"], total_coord_windows)
    c_v_1 = pct(coverage[1.0]["C_v"], total_coord_windows)
    c_v_100 = pct(coverage[100.0]["C_v"], total_coord_windows)
    lines.append(
        f"- The first-order perturbative coverage at threshold 1 is C_m={c_m_1:.3g}% and C_v={c_v_1:.3g}% of all coordinate-windows."
    )
    lines.append(
        f"- Even at threshold 100, C_v coverage is {c_v_100:.3g}%, consistent with beta2=0.999 making ell2 about 999 and amplifying small log-drifts."
    )
    lines.append(
        "- Descriptively, these numbers point more toward case C/D than case A: the transient is not absent, but the harder practical issue is curvature and the second moment v."
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print compact interpretive tables for the EMA decomposition diagnostic."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--sample-size", type=int, default=80_000)
    parser.add_argument("--ratio-eps", type=float, default=1e-30)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    summary_path = results_dir / "diagnostic_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    summary = load_json(summary_path)
    rows_by_key = read_decomposition_rows(results_dir)
    joint_stats, coverage, total_coord_windows = compute_joint_stats_and_coverage(
        summary=summary,
        chunk_size=args.chunk_size,
        ratio_eps=args.ratio_eps,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    sections = [
        "# EMA Decomposition Interpretive Summary",
        "",
        "Coverage percentages in the perturbative table are relative to all coordinate-windows and include the sign-stable condition.",
        "",
        "## 1. Term Budget",
        build_term_budget_table(rows_by_key, joint_stats),
        "",
        "## 2. Ratios Between Error Terms",
        build_ratio_table(joint_stats),
        "",
        "## 3. Relaxed Perturbative Coverage",
        build_coverage_table(coverage, total_coord_windows),
        "",
        "## 4. Automatic Interpretation",
        interpretation(rows_by_key, joint_stats, coverage, total_coord_windows),
    ]
    output = "\n".join(sections)
    print(output)

    if not args.no_save:
        output_path = results_dir / "interpretive_console_summary.md"
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
