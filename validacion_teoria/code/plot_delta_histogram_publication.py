from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_PATH = BASE_DIR / "results" / "log_gradient_changes" / "probe_bs128" / "pair_s" / "pair_s_sample.npy"
DEFAULT_SUMMARY_PATH = BASE_DIR / "results" / "log_gradient_changes" / "probe_bs128" / "pair_s" / "pair_s_summary.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "plots"


def configure_matplotlib(use_latex: bool) -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use("default")
    plt.rcParams.update(
        {
            "text.usetex": use_latex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def make_plot(sample_path: Path, summary_path: Path, output_dir: Path, use_latex: bool) -> None:
    configure_matplotlib(use_latex)
    sample = np.load(sample_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    lo, hi = np.percentile(sample, [0.5, 99.5])
    clipped = sample[(sample >= lo) & (sample <= hi)]
    threshold = float(np.log(np.sqrt(2.0)))

    color_hist = "#1f4e79"
    color_threshold = "#8b0000"
    color_zero = "#222222"
    color_median = "#666666"

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.hist(
        clipped,
        bins=120,
        density=True,
        color=color_hist,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.2,
    )
    ax.axvline(0.0, color=color_zero, linewidth=1.0, label=r"$0$")
    ax.axvline(threshold, color=color_threshold, linewidth=1.0, linestyle="--", label=r"$\pm\log\sqrt{2}$")
    ax.axvline(-threshold, color=color_threshold, linewidth=1.0, linestyle="--")
    ax.axvline(summary["sample_percentiles"]["p50"], color=color_median, linewidth=0.9, linestyle=":")

    ax.set_xlabel(r"$\Delta_{k,i}=\log |g_{k+1,i}|-\log |g_{k,i}|$")
    ax.set_ylabel("Density")
    ax.set_title(r"Same-sign consecutive log-gradient changes")
    ax.legend(frameon=False, fontsize=9)
    ax.text(
        0.03,
        0.96,
        rf"$N={summary['valid_cases']:,}$" + "\n" + rf"$\sigma={summary['std']:.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "delta_log_gradient_histogram_probe_bs128.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "delta_log_gradient_histogram_probe_bs128.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publication-style histogram for same-sign log-gradient changes.")
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-latex", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        make_plot(args.sample_path, args.summary_path, args.output_dir, use_latex=not args.no_latex)
    except RuntimeError:
        if args.no_latex:
            raise
        make_plot(args.sample_path, args.summary_path, args.output_dir, use_latex=False)


if __name__ == "__main__":
    main()
