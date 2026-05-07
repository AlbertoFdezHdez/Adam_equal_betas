from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    BASE_DIR
    / "descargas"
    / "adam_gradient_cifar10"
    / "runs"
    / "cifar10_tinycnn_adam_b1-0p9_b2-0p999_bs-128_grad-probe_seed-1234"
)
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "ema_decomposition"

TERMS = ("base", "lag", "trans", "curv")
RATIO_TERMS = ("lag", "trans", "curv")
REGIMES = ("A", "B", "C")
PHASES = ("early", "mid", "late", "all")


class Reservoir:
    def __init__(self, max_size: int, rng: np.random.Generator) -> None:
        self.max_size = int(max_size)
        self.rng = rng
        self.data = np.empty(self.max_size, dtype=np.float64) if self.max_size > 0 else np.empty(0)
        self.filled = 0
        self.seen = 0

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values).reshape(-1)
        if values.size == 0:
            return
        values = values[np.isfinite(values)]
        if values.size == 0:
            return

        start_seen = self.seen
        self.seen += int(values.size)
        if self.max_size <= 0:
            return

        remaining = self.max_size - self.filled
        if remaining > 0:
            take = min(remaining, values.size)
            self.data[self.filled : self.filled + take] = values[:take]
            self.filled += take
            start_seen += take
            values = values[take:]

        if values.size == 0:
            return

        totals = start_seen + np.arange(1, values.size + 1, dtype=np.int64)
        keep = self.rng.random(values.size) < (self.max_size / totals)
        if np.any(keep):
            replace_idx = self.rng.integers(0, self.max_size, size=int(np.count_nonzero(keep)))
            self.data[replace_idx] = values[keep]

    def sample(self) -> np.ndarray:
        return self.data[: self.filled]

    def percentile(self, q: float) -> float:
        sample = self.sample()
        if sample.size == 0:
            return float("nan")
        return float(np.percentile(sample, q))

    def percentiles(self, qs: tuple[float, ...]) -> dict[str, float]:
        sample = self.sample()
        if sample.size == 0:
            return {f"p{q:g}": float("nan") for q in qs}
        vals = np.percentile(sample, qs)
        return {f"p{q:g}": float(v) for q, v in zip(qs, vals)}


class PairReservoir:
    def __init__(self, max_size: int, rng: np.random.Generator) -> None:
        self.max_size = int(max_size)
        self.rng = rng
        self.x = np.empty(self.max_size, dtype=np.float64) if self.max_size > 0 else np.empty(0)
        self.y = np.empty(self.max_size, dtype=np.float64) if self.max_size > 0 else np.empty(0)
        self.filled = 0
        self.seen = 0

    def add(self, x_values: np.ndarray, y_values: np.ndarray) -> None:
        x_values = np.asarray(x_values).reshape(-1)
        y_values = np.asarray(y_values).reshape(-1)
        if x_values.size != y_values.size:
            raise ValueError("Pair reservoir received arrays with different sizes.")
        if x_values.size == 0:
            return
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[finite]
        y_values = y_values[finite]
        if x_values.size == 0:
            return

        start_seen = self.seen
        self.seen += int(x_values.size)
        if self.max_size <= 0:
            return

        remaining = self.max_size - self.filled
        if remaining > 0:
            take = min(remaining, x_values.size)
            self.x[self.filled : self.filled + take] = x_values[:take]
            self.y[self.filled : self.filled + take] = y_values[:take]
            self.filled += take
            start_seen += take
            x_values = x_values[take:]
            y_values = y_values[take:]

        if x_values.size == 0:
            return

        totals = start_seen + np.arange(1, x_values.size + 1, dtype=np.int64)
        keep = self.rng.random(x_values.size) < (self.max_size / totals)
        if np.any(keep):
            replace_idx = self.rng.integers(0, self.max_size, size=int(np.count_nonzero(keep)))
            self.x[replace_idx] = x_values[keep]
            self.y[replace_idx] = y_values[keep]

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x[: self.filled], self.y[: self.filled]


@dataclass
class TermStats:
    rel_sample: Reservoir
    count: int = 0
    share_sum: float = 0.0
    dominate_count: int = 0

    def add(self, rel_values: np.ndarray, share_values: np.ndarray, dominate_mask: np.ndarray) -> None:
        rel_values = np.asarray(rel_values).reshape(-1)
        share_values = np.asarray(share_values).reshape(-1)
        dominate_mask = np.asarray(dominate_mask).reshape(-1)
        if rel_values.size == 0:
            return
        self.count += int(rel_values.size)
        self.share_sum += float(np.sum(share_values, dtype=np.float64))
        self.dominate_count += int(np.count_nonzero(dominate_mask))
        self.rel_sample.add(rel_values)


@dataclass
class ComparisonStats:
    count: int = 0
    trans_gt_lag: int = 0
    trans_gt_curv: int = 0
    curv_gt_lag: int = 0

    def add(self, abs_lag: np.ndarray, abs_trans: np.ndarray, abs_curv: np.ndarray) -> None:
        if abs_lag.size == 0:
            return
        self.count += int(abs_lag.size)
        self.trans_gt_lag += int(np.count_nonzero(abs_trans > abs_lag))
        self.trans_gt_curv += int(np.count_nonzero(abs_trans > abs_curv))
        self.curv_gt_lag += int(np.count_nonzero(abs_curv > abs_lag))


@dataclass
class PhaseStats:
    drift_sample: Reservoir
    total_coord_windows: int = 0
    sign_stable_count: int = 0
    moderate_drift_count: int = 0
    perturbative_count: int = 0


@dataclass
class Aggregates:
    term_stats: dict[tuple[str, str, int, str], TermStats] = field(default_factory=dict)
    comparison_stats: dict[tuple[str, str, int], ComparisonStats] = field(default_factory=dict)
    dominance_counts: dict[tuple[str, str, int, str], int] = field(default_factory=dict)
    dominance_totals: dict[tuple[str, str, int], int] = field(default_factory=dict)
    trans_curv_pairs: dict[tuple[str, str], PairReservoir] = field(default_factory=dict)
    curv_drift_pairs: dict[tuple[str, str], PairReservoir] = field(default_factory=dict)
    max_reconstruction_error: dict[str, float] = field(default_factory=lambda: {"m": 0.0, "v": 0.0})
    max_relative_reconstruction_error: dict[str, float] = field(default_factory=lambda: {"m": 0.0, "v": 0.0})


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_float_for_name(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def default_output_name(config: dict[str, object], fallback: str) -> str:
    try:
        signal = str(config.get("saved_gradient_source", "probe"))
        beta1 = format_float_for_name(float(config["beta1"]))
        beta2 = format_float_for_name(float(config["beta2"]))
        batch_size = int(config["batch_size"])
        probe_batch = int(config.get("probe_batch_size") or batch_size)
        seed = int(config["seed"])
        return f"{signal}_b1{beta1}_b2{beta2}_bs{batch_size}_pbs{probe_batch}_s{seed}"
    except Exception:
        return fallback[:80]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def make_reservoirs(seed: int, sample_size: int) -> tuple[np.random.Generator, callable]:
    seed_sequence = np.random.SeedSequence(seed)
    counter = {"value": 0}

    def next_rng() -> np.random.Generator:
        child = seed_sequence.spawn(1)[0]
        counter["value"] += 1
        return np.random.default_rng(child)

    return np.random.default_rng(seed), lambda: Reservoir(sample_size, next_rng())


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    offset = int.from_bytes(digest, byteorder="little") % 1_000_000_000
    return int(base_seed + offset)


def phase_masks(n_windows: int, steps: int) -> dict[str, np.ndarray]:
    starts = np.arange(1, n_windows + 1, dtype=np.float64)
    progress = starts / max(steps - 1, 1)
    return {
        "early": progress < 0.10,
        "mid": (progress >= 0.45) & (progress < 0.55),
        "late": progress >= 0.90,
        "all": np.ones(n_windows, dtype=bool),
    }


def valid_and_drift(
    chunk: np.ndarray,
    window: int,
    eps_g: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    signs = np.sign(chunk).astype(np.int8, copy=False)
    abs_chunk = np.abs(chunk)
    positive = abs_chunk > eps_g

    sign_windows = np.lib.stride_tricks.sliding_window_view(signs, window_shape=window + 1, axis=0)
    positive_windows = np.lib.stride_tricks.sliding_window_view(
        positive, window_shape=window + 1, axis=0
    )
    same_sign = np.all(sign_windows == sign_windows[:, :, :1], axis=-1)
    valid = np.all(positive_windows, axis=-1) & same_sign

    log_abs = np.full(abs_chunk.shape, np.nan, dtype=np.float64)
    log_abs[positive] = np.log(abs_chunk[positive].astype(np.float64) + eps_g)
    delta = log_abs[1:, :] - log_abs[:-1, :]
    delta_windows = np.lib.stride_tricks.sliding_window_view(delta, window_shape=window, axis=0)
    abs_delta_windows = np.abs(delta_windows)
    max_abs_delta = np.nanmax(abs_delta_windows, axis=-1)
    max_delta_diff = np.nanmax(np.abs(delta_windows[:, :, 1:] - delta_windows[:, :, :-1]), axis=-1)
    drift_metric = max_abs_delta**2 + max_delta_diff
    return valid, delta_windows, max_abs_delta, drift_metric


def shadow_ema(signal: np.ndarray, beta: float) -> np.ndarray:
    moment = np.zeros(signal.shape[1], dtype=np.float64)
    out = np.empty(signal.shape, dtype=np.float64)
    one_minus_beta = 1.0 - beta
    for idx in range(signal.shape[0]):
        moment = beta * moment + one_minus_beta * signal[idx, :]
        out[idx, :] = moment
    return out


def decompose_ema(
    values: np.ndarray,
    ema_post: np.ndarray,
    beta: float,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ell = beta / (1.0 - beta)
    kappa = beta * beta / (1.0 - beta)

    value_windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=window + 1, axis=0
    )
    ema_windows = np.lib.stride_tricks.sliding_window_view(ema_post[1:, :], window_shape=window, axis=0)

    base = value_windows[:, :, 1:]
    delta = base - value_windows[:, :, :-1]
    lag = -ell * delta
    slow0 = base[:, :, 0] + lag[:, :, 0]
    d0 = ema_windows[:, :, 0] - slow0
    powers = np.power(beta, np.arange(window, dtype=np.float64))
    trans = d0[:, :, None] * powers[None, None, :]

    curv = np.zeros_like(base)
    for j in range(1, window):
        d2 = value_windows[:, :, j + 1] - 2.0 * value_windows[:, :, j] + value_windows[:, :, j - 1]
        curv[:, :, j] = beta * curv[:, :, j - 1] + kappa * d2

    recon = base + lag + trans + curv
    return base, lag, trans, curv, ema_windows, recon


def count_first_pass(
    gradients: np.ndarray,
    window: int,
    chunk_size: int,
    eps_g: float,
    phase_mask_by_name: dict[str, np.ndarray],
    sample_size: int,
    seed: int,
) -> tuple[dict[str, PhaseStats], Reservoir]:
    _, new_reservoir = make_reservoirs(seed, sample_size)
    phase_stats = {phase: PhaseStats(new_reservoir()) for phase in PHASES}
    global_drift = new_reservoir()
    steps, n_params = gradients.shape

    for start in range(0, n_params, chunk_size):
        stop = min(start + chunk_size, n_params)
        chunk = np.asarray(gradients[:, start:stop], dtype=np.float64)
        valid, _, max_abs_delta, _ = valid_and_drift(chunk, window=window, eps_g=eps_g)
        n_chunk_params = stop - start

        for phase in PHASES:
            phase_mask = phase_mask_by_name[phase]
            stats = phase_stats[phase]
            stats.total_coord_windows += int(np.count_nonzero(phase_mask) * n_chunk_params)
            phase_valid = valid[phase_mask, :]
            stats.sign_stable_count += int(np.count_nonzero(phase_valid))
            stats.drift_sample.add(max_abs_delta[phase_mask, :][phase_valid])

        global_drift.add(max_abs_delta[valid])
        print(f"[pass 1] processed coordinates {start}:{stop} / {n_params}", flush=True)

    return phase_stats, global_drift


def get_term_stats(
    aggregates: Aggregates,
    signal: str,
    regime: str,
    j: int,
    term: str,
    sample_size: int,
    seed: int,
) -> TermStats:
    key = (signal, regime, j, term)
    if key not in aggregates.term_stats:
        rng = np.random.default_rng(stable_seed(seed, *key))
        aggregates.term_stats[key] = TermStats(rel_sample=Reservoir(sample_size, rng))
    return aggregates.term_stats[key]


def get_pair_reservoir(
    target: dict[tuple[str, str], PairReservoir],
    signal: str,
    regime: str,
    sample_size: int,
    seed: int,
    salt: int,
) -> PairReservoir:
    key = (signal, regime)
    if key not in target:
        rng = np.random.default_rng(stable_seed(seed + salt, *key))
        target[key] = PairReservoir(sample_size, rng)
    return target[key]


def aggregate_decomposition(
    *,
    aggregates: Aggregates,
    signal: str,
    regimes: dict[str, np.ndarray],
    drift_metric: np.ndarray,
    base: np.ndarray,
    lag: np.ndarray,
    trans: np.ndarray,
    curv: np.ndarray,
    obs: np.ndarray,
    recon: np.ndarray,
    denom_eps: float,
    share_eps: float,
    sample_size: int,
    pair_sample_size: int,
    seed: int,
) -> None:
    err = np.abs(recon - obs)
    max_err = float(np.max(err))
    aggregates.max_reconstruction_error[signal] = max(aggregates.max_reconstruction_error[signal], max_err)
    rel_err = err / (np.abs(obs) + denom_eps + share_eps)
    aggregates.max_relative_reconstruction_error[signal] = max(
        aggregates.max_relative_reconstruction_error[signal],
        float(np.max(rel_err)),
    )

    window = base.shape[-1]
    for regime, mask in regimes.items():
        regime_count = int(np.count_nonzero(mask))
        if regime_count == 0:
            continue

        for j in range(window):
            base_j = base[:, :, j]
            lag_j = lag[:, :, j]
            trans_j = trans[:, :, j]
            curv_j = curv[:, :, j]

            abs_base = np.abs(base_j)
            abs_lag = np.abs(lag_j)
            abs_trans = np.abs(trans_j)
            abs_curv = np.abs(curv_j)
            abs_total = abs_base + abs_lag + abs_trans + abs_curv + share_eps
            abs_stack = np.stack((abs_base, abs_lag, abs_trans, abs_curv), axis=0)
            dominance = np.argmax(abs_stack, axis=0)
            denominator = abs_base + denom_eps
            safe_denominator = np.where(denominator > 0.0, denominator, np.nan)

            key_comp = (signal, regime, j)
            comp = aggregates.comparison_stats.setdefault(key_comp, ComparisonStats())
            comp.add(abs_lag[mask], abs_trans[mask], abs_curv[mask])
            aggregates.dominance_totals[key_comp] = aggregates.dominance_totals.get(key_comp, 0) + regime_count

            rel_by_term = {
                "lag": abs_lag / safe_denominator,
                "trans": abs_trans / safe_denominator,
                "curv": abs_curv / safe_denominator,
            }
            share_by_term = {
                "base": abs_base / abs_total,
                "lag": abs_lag / abs_total,
                "trans": abs_trans / abs_total,
                "curv": abs_curv / abs_total,
            }

            for term_idx, term in enumerate(TERMS):
                dom_key = (signal, regime, j, term)
                dom_count = int(np.count_nonzero((dominance == term_idx) & mask))
                aggregates.dominance_counts[dom_key] = aggregates.dominance_counts.get(dom_key, 0) + dom_count

            for term in RATIO_TERMS:
                stats = get_term_stats(aggregates, signal, regime, j, term, sample_size, seed)
                stats.add(rel_by_term[term][mask], share_by_term[term][mask], (dominance == TERMS.index(term))[mask])

            trans_curv = get_pair_reservoir(
                aggregates.trans_curv_pairs, signal, regime, pair_sample_size, seed, salt=17_000
            )
            trans_curv.add(rel_by_term["curv"][mask], rel_by_term["trans"][mask])

            curv_drift = get_pair_reservoir(
                aggregates.curv_drift_pairs, signal, regime, pair_sample_size, seed, salt=31_000
            )
            curv_drift.add(drift_metric[mask], rel_by_term["curv"][mask])


def second_pass(
    *,
    gradients: np.ndarray,
    window: int,
    chunk_size: int,
    beta1: float,
    beta2: float,
    eps_g: float,
    eps_v: float,
    share_eps: float,
    drift_threshold: float,
    perturb_threshold: float,
    phase_mask_by_name: dict[str, np.ndarray],
    phase_stats: dict[str, PhaseStats],
    sample_size: int,
    pair_sample_size: int,
    seed: int,
) -> Aggregates:
    aggregates = Aggregates()
    steps, n_params = gradients.shape
    ell1 = beta1 / (1.0 - beta1)
    ell2 = beta2 / (1.0 - beta2)

    for start in range(0, n_params, chunk_size):
        stop = min(start + chunk_size, n_params)
        chunk = np.asarray(gradients[:, start:stop], dtype=np.float64)
        valid, delta_windows, max_abs_delta, drift_metric = valid_and_drift(
            chunk, window=window, eps_g=eps_g
        )

        first_order_m = np.max(np.abs(ell1 * (1.0 - np.exp(-delta_windows))), axis=-1)
        first_order_v = np.max(np.abs(ell2 * (1.0 - np.exp(-2.0 * delta_windows))), axis=-1)
        moderate = valid & (max_abs_delta <= drift_threshold)
        perturbative = valid & (first_order_m <= perturb_threshold) & (first_order_v <= perturb_threshold)
        regimes = {"A": valid, "B": moderate, "C": perturbative}

        for phase in PHASES:
            phase_mask = phase_mask_by_name[phase]
            stats = phase_stats[phase]
            stats.moderate_drift_count += int(np.count_nonzero(moderate[phase_mask, :]))
            stats.perturbative_count += int(np.count_nonzero(perturbative[phase_mask, :]))

        shadow_m = shadow_ema(chunk, beta1)
        base_m, lag_m, trans_m, curv_m, obs_m, recon_m = decompose_ema(
            chunk, shadow_m, beta=beta1, window=window
        )
        aggregate_decomposition(
            aggregates=aggregates,
            signal="m",
            regimes=regimes,
            drift_metric=drift_metric,
            base=base_m,
            lag=lag_m,
            trans=trans_m,
            curv=curv_m,
            obs=obs_m,
            recon=recon_m,
            denom_eps=eps_g,
            share_eps=share_eps,
            sample_size=sample_size,
            pair_sample_size=pair_sample_size,
            seed=seed,
        )

        y = chunk * chunk
        shadow_v = shadow_ema(y, beta2)
        base_v, lag_v, trans_v, curv_v, obs_v, recon_v = decompose_ema(
            y, shadow_v, beta=beta2, window=window
        )
        aggregate_decomposition(
            aggregates=aggregates,
            signal="v",
            regimes=regimes,
            drift_metric=drift_metric,
            base=base_v,
            lag=lag_v,
            trans=trans_v,
            curv=curv_v,
            obs=obs_v,
            recon=recon_v,
            denom_eps=eps_v,
            share_eps=share_eps,
            sample_size=sample_size,
            pair_sample_size=pair_sample_size,
            seed=seed + 10_000,
        )

        print(f"[pass 2] processed coordinates {start}:{stop} / {n_params}", flush=True)

    return aggregates


def write_window_table(
    path: Path,
    *,
    phase_stats: dict[str, PhaseStats],
    beta1: float,
    beta2: float,
    model: str,
    signal: str,
    window: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for phase in PHASES:
        stats = phase_stats[phase]
        total = stats.total_coord_windows
        sign_percent = 100.0 * stats.sign_stable_count / total if total else float("nan")
        moderate_percent = 100.0 * stats.moderate_drift_count / total if total else float("nan")
        perturb_percent = 100.0 * stats.perturbative_count / total if total else float("nan")
        perturb_given_sign = (
            100.0 * stats.perturbative_count / stats.sign_stable_count
            if stats.sign_stable_count
            else float("nan")
        )
        drift_qs = stats.drift_sample.percentiles((50, 90, 99))
        rows.append(
            {
                "signal": signal,
                "model": model,
                "beta1": beta1,
                "beta2": beta2,
                "phase": phase,
                "W": window,
                "coord_windows": total,
                "sign_stable_count": stats.sign_stable_count,
                "sign_stable_percent": sign_percent,
                "drift_p50": drift_qs["p50"],
                "drift_p90": drift_qs["p90"],
                "drift_p99": drift_qs["p99"],
                "moderate_drift_count": stats.moderate_drift_count,
                "moderate_drift_percent": moderate_percent,
                "perturbative_count": stats.perturbative_count,
                "perturbative_percent": perturb_percent,
                "perturbative_given_sign_stable_percent": perturb_given_sign,
                "drift_sample_seen": stats.drift_sample.seen,
                "drift_sample_size": stats.drift_sample.filled,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_decomposition_table(
    path: Path,
    *,
    aggregates: Aggregates,
    signal: str,
    beta1: float,
    beta2: float,
    window: int,
    selected_js: tuple[int, ...] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    js = selected_js if selected_js is not None else tuple(range(window))
    for regime in REGIMES:
        for j in js:
            comp = aggregates.comparison_stats.get((signal, regime, j), ComparisonStats())
            comp_count = comp.count
            for term in RATIO_TERMS:
                stats = aggregates.term_stats.get((signal, regime, j, term))
                if stats is None or stats.count == 0:
                    q = {"p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
                    share_mean = float("nan")
                    dominates_percent = float("nan")
                    count = 0
                else:
                    q = stats.rel_sample.percentiles((50, 90, 99))
                    share_mean = stats.share_sum / stats.count
                    dominates_percent = 100.0 * stats.dominate_count / stats.count
                    count = stats.count
                rows.append(
                    {
                        "signal": "probe",
                        "moment": signal,
                        "beta1": beta1,
                        "beta2": beta2,
                        "regime": regime,
                        "j": j,
                        "term": term,
                        "coord_windows": count,
                        "rel_p50": q["p50"],
                        "rel_p90": q["p90"],
                        "rel_p99": q["p99"],
                        "abs_share_mean": share_mean,
                        "dominates_percent": dominates_percent,
                        "trans_gt_lag_percent": 100.0 * comp.trans_gt_lag / comp_count
                        if comp_count
                        else float("nan"),
                        "trans_gt_curv_percent": 100.0 * comp.trans_gt_curv / comp_count
                        if comp_count
                        else float("nan"),
                        "curv_gt_lag_percent": 100.0 * comp.curv_gt_lag / comp_count
                        if comp_count
                        else float("nan"),
                        "rel_sample_seen": stats.rel_sample.seen if stats else 0,
                        "rel_sample_size": stats.rel_sample.filled if stats else 0,
                    }
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_dominance_table(path: Path, *, aggregates: Aggregates, window: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for signal in ("m", "v"):
        for regime in REGIMES:
            for j in range(window):
                total = aggregates.dominance_totals.get((signal, regime, j), 0)
                for term in TERMS:
                    count = aggregates.dominance_counts.get((signal, regime, j, term), 0)
                    rows.append(
                        {
                            "moment": signal,
                            "regime": regime,
                            "j": j,
                            "term": term,
                            "dominates_count": count,
                            "coord_windows": total,
                            "dominates_percent": 100.0 * count / total if total else float("nan"),
                        }
                    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_trans_decay(
    path: Path,
    *,
    aggregates: Aggregates,
    signal: str,
    window: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = {"A": "#1f77b4", "B": "#2ca02c", "C": "#d62728"}
    x = np.arange(window)
    for regime in REGIMES:
        p25: list[float] = []
        p50: list[float] = []
        p75: list[float] = []
        p90: list[float] = []
        for j in range(window):
            stats = aggregates.term_stats.get((signal, regime, j, "trans"))
            if stats is None or stats.count == 0:
                p25.append(np.nan)
                p50.append(np.nan)
                p75.append(np.nan)
                p90.append(np.nan)
            else:
                qs = stats.rel_sample.percentiles((25, 50, 75, 90))
                p25.append(qs["p25"])
                p50.append(qs["p50"])
                p75.append(qs["p75"])
                p90.append(qs["p90"])
        p25_arr = np.asarray(p25, dtype=np.float64)
        p50_arr = np.asarray(p50, dtype=np.float64)
        p75_arr = np.asarray(p75, dtype=np.float64)
        p90_arr = np.asarray(p90, dtype=np.float64)
        if np.all(~np.isfinite(p50_arr)):
            continue
        ax.plot(x, np.maximum(p50_arr, 1e-16), color=colors[regime], marker="o", label=f"{regime} median")
        ax.fill_between(
            x,
            np.maximum(p25_arr, 1e-16),
            np.maximum(p75_arr, 1e-16),
            color=colors[regime],
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(x, np.maximum(p90_arr, 1e-16), color=colors[regime], linestyle="--", alpha=0.75)

    ylabel = r"$|\beta^j D_0| / (|g_j|+\epsilon_g)$" if signal == "m" else r"$|\beta^j D_0| / (g_j^2+\epsilon_v)$"
    ax.set_title(f"Real transient size for {signal}")
    ax.set_xlabel("j inside W=6 window")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_dominance(
    path: Path,
    *,
    aggregates: Aggregates,
    signal: str,
    selected_js: tuple[int, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    colors = {"base": "#4c78a8", "lag": "#f58518", "trans": "#54a24b", "curv": "#e45756"}
    labels: list[str] = []
    positions: list[int] = []
    values_by_term = {term: [] for term in TERMS}
    pos = 0
    for regime in REGIMES:
        for j in selected_js:
            labels.append(f"{regime}\nj={j}")
            positions.append(pos)
            total = aggregates.dominance_totals.get((signal, regime, j), 0)
            for term in TERMS:
                count = aggregates.dominance_counts.get((signal, regime, j, term), 0)
                values_by_term[term].append(100.0 * count / total if total else 0.0)
            pos += 1

    bottom = np.zeros(len(positions), dtype=np.float64)
    for term in TERMS:
        vals = np.asarray(values_by_term[term], dtype=np.float64)
        ax.bar(positions, vals, bottom=bottom, color=colors[term], label=term, width=0.72)
        bottom += vals

    ax.set_title(f"Dominant term for {signal}")
    ax.set_ylabel("% coord-windows")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_hexbin_pairs(
    path: Path,
    *,
    pair_dict: dict[tuple[str, str], PairReservoir],
    title_prefix: str,
    xlabel: str,
    ylabel: str,
    diagonal: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.5), sharex=False, sharey=False)
    for row, signal in enumerate(("m", "v")):
        for col, regime in enumerate(REGIMES):
            ax = axes[row, col]
            reservoir = pair_dict.get((signal, regime))
            if reservoir is None or reservoir.filled == 0:
                ax.set_axis_off()
                continue
            x, y = reservoir.sample()
            finite = np.isfinite(x) & np.isfinite(y)
            x = x[finite]
            y = y[finite]
            if x.size == 0:
                ax.set_axis_off()
                continue
            lx = np.log10(np.maximum(x, 1e-16))
            ly = np.log10(np.maximum(y, 1e-16))
            hb = ax.hexbin(lx, ly, gridsize=45, mincnt=1, cmap="viridis", bins="log")
            if diagonal:
                lo = float(min(np.min(lx), np.min(ly)))
                hi = float(max(np.max(lx), np.max(ly)))
                ax.plot([lo, hi], [lo, hi], color="white", linewidth=1.0, alpha=0.85)
            ax.set_title(f"{signal}, regime {regime}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.15)
    fig.suptitle(title_prefix, y=0.995)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact discrete EMA decomposition on fixed probe-gradient windows."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--eps-g", type=float, default=0.0)
    parser.add_argument("--eps-v", type=float, default=0.0)
    parser.add_argument("--share-eps", type=float, default=1e-30)
    parser.add_argument("--drift-regime-quantile", type=float, default=90.0)
    parser.add_argument("--perturb-threshold", type=float, default=1.0)
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--drift-sample-size", type=int, default=300_000)
    parser.add_argument("--pair-sample-size", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=20260505)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    gradients_path = run_dir / "gradients.npy"
    config_path = run_dir / "config.json"
    if not gradients_path.is_file():
        raise FileNotFoundError(gradients_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    config = load_json(config_path)
    beta1 = float(config["beta1"])
    beta2 = float(config["beta2"])
    model = "TinyCIFARCNN"
    signal_name = str(config.get("saved_gradient_source", "probe"))
    if signal_name != "probe":
        print(f"Warning: expected probe gradients, found signal={signal_name!r}.", flush=True)

    gradients = np.load(gradients_path, mmap_mode="r")
    if gradients.ndim != 2:
        raise ValueError(f"Expected gradients with shape (steps, parameters), got {gradients.shape}")
    steps, n_params = gradients.shape
    if args.window < 2 or args.window + 1 > steps:
        raise ValueError(f"Invalid W={args.window} for {steps} steps.")
    n_windows = steps - args.window
    phase_mask_by_name = phase_masks(n_windows=n_windows, steps=steps)

    output_name = args.output_name or default_output_name(config, run_dir.name)
    output_dir = args.results_dir.resolve() / output_name / f"W{args.window}"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loaded gradients {gradients_path} with shape={gradients.shape}, dtype={gradients.dtype}",
        flush=True,
    )
    phase_stats, global_drift = count_first_pass(
        gradients=gradients,
        window=args.window,
        chunk_size=args.chunk_size,
        eps_g=args.eps_g,
        phase_mask_by_name=phase_mask_by_name,
        sample_size=args.drift_sample_size,
        seed=args.seed,
    )
    drift_threshold = global_drift.percentile(args.drift_regime_quantile)
    print(
        f"Regime B threshold: max|delta| <= p{args.drift_regime_quantile:g} = {drift_threshold:.6g}",
        flush=True,
    )

    aggregates = second_pass(
        gradients=gradients,
        window=args.window,
        chunk_size=args.chunk_size,
        beta1=beta1,
        beta2=beta2,
        eps_g=args.eps_g,
        eps_v=args.eps_v,
        share_eps=args.share_eps,
        drift_threshold=drift_threshold,
        perturb_threshold=args.perturb_threshold,
        phase_mask_by_name=phase_mask_by_name,
        phase_stats=phase_stats,
        sample_size=args.sample_size,
        pair_sample_size=args.pair_sample_size,
        seed=args.seed,
    )

    table1 = write_window_table(
        output_dir / "table1_window_validity_and_drift.csv",
        phase_stats=phase_stats,
        beta1=beta1,
        beta2=beta2,
        model=model,
        signal=signal_name,
        window=args.window,
    )
    table2 = write_decomposition_table(
        output_dir / "table2_decomposition_m_j025.csv",
        aggregates=aggregates,
        signal="m",
        beta1=beta1,
        beta2=beta2,
        window=args.window,
        selected_js=(0, 2, 5),
    )
    table3 = write_decomposition_table(
        output_dir / "table3_decomposition_v_j025.csv",
        aggregates=aggregates,
        signal="v",
        beta1=beta1,
        beta2=beta2,
        window=args.window,
        selected_js=(0, 2, 5),
    )
    write_decomposition_table(
        output_dir / "table2_decomposition_m_all_j.csv",
        aggregates=aggregates,
        signal="m",
        beta1=beta1,
        beta2=beta2,
        window=args.window,
        selected_js=None,
    )
    write_decomposition_table(
        output_dir / "table3_decomposition_v_all_j.csv",
        aggregates=aggregates,
        signal="v",
        beta1=beta1,
        beta2=beta2,
        window=args.window,
        selected_js=None,
    )
    dominance_rows = write_dominance_table(
        output_dir / "dominance_counts_all_j.csv",
        aggregates=aggregates,
        window=args.window,
    )

    plot_trans_decay(plots_dir / "plot1_transient_decay_m", aggregates=aggregates, signal="m", window=args.window)
    plot_trans_decay(plots_dir / "plot1_transient_decay_v", aggregates=aggregates, signal="v", window=args.window)
    plot_dominance(
        plots_dir / "plot2_term_dominance_m",
        aggregates=aggregates,
        signal="m",
        selected_js=(0, 2, 5),
    )
    plot_dominance(
        plots_dir / "plot2_term_dominance_v",
        aggregates=aggregates,
        signal="v",
        selected_js=(0, 2, 5),
    )
    plot_hexbin_pairs(
        plots_dir / "plot3_transient_vs_curvature",
        pair_dict=aggregates.trans_curv_pairs,
        title_prefix="Transient versus curvature",
        xlabel=r"$\log_{10}(\rho^{curv})$",
        ylabel=r"$\log_{10}(\rho^{trans})$",
        diagonal=True,
    )
    plot_hexbin_pairs(
        plots_dir / "plot4_curv_vs_drift",
        pair_dict=aggregates.curv_drift_pairs,
        title_prefix="Curvature versus discrete drift proxy",
        xlabel=r"$\log_{10}(\max|\delta|^2+\max|\Delta\delta|)$",
        ylabel=r"$\log_{10}(\rho^{curv})$",
        diagonal=False,
    )

    summary = {
        "run_dir": str(run_dir),
        "gradients_path": str(gradients_path),
        "steps": int(steps),
        "n_parameters": int(n_params),
        "window": int(args.window),
        "n_window_starts": int(n_windows),
        "signal": signal_name,
        "model": model,
        "beta1": beta1,
        "beta2": beta2,
        "ell1": beta1 / (1.0 - beta1),
        "ell2": beta2 / (1.0 - beta2),
        "drift_regime_quantile": args.drift_regime_quantile,
        "drift_threshold_for_regime_B": drift_threshold,
        "perturb_threshold_for_regime_C": args.perturb_threshold,
        "eps_g": args.eps_g,
        "eps_v": args.eps_v,
        "share_eps": args.share_eps,
        "chunk_size": args.chunk_size,
        "sample_size_per_decomposition_key": args.sample_size,
        "drift_sample_size_per_phase": args.drift_sample_size,
        "pair_sample_size": args.pair_sample_size,
        "max_reconstruction_error": aggregates.max_reconstruction_error,
        "max_relative_reconstruction_error": aggregates.max_relative_reconstruction_error,
        "outputs": {
            "table1": str(output_dir / "table1_window_validity_and_drift.csv"),
            "table2_m_j025": str(output_dir / "table2_decomposition_m_j025.csv"),
            "table3_v_j025": str(output_dir / "table3_decomposition_v_j025.csv"),
            "table2_m_all_j": str(output_dir / "table2_decomposition_m_all_j.csv"),
            "table3_v_all_j": str(output_dir / "table3_decomposition_v_all_j.csv"),
            "dominance": str(output_dir / "dominance_counts_all_j.csv"),
            "plots_dir": str(plots_dir),
        },
    }
    write_json(output_dir / "diagnostic_summary.json", summary)
    write_json(output_dir / "diagnostic_config.json", {"args": vars(args), "training_config": config})
    write_json(output_dir / "table1_window_validity_and_drift.json", table1)
    write_json(output_dir / "table2_decomposition_m_j025.json", table2)
    write_json(output_dir / "table3_decomposition_v_j025.json", table3)
    write_json(output_dir / "dominance_counts_all_j.json", dominance_rows)

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
