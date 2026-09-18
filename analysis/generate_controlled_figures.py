"""Generate Figures 1 and 2 from the deterministic gradient signal.

This is the script version of the controlled experiment. To remain
bit-for-bit faithful to that figure, index 0 stores the zero-initialized Adam
state and gradients 1,...,600 are replayed with bias correction exponent k.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


EQUAL_BETAS = (0.9, 0.95, 0.99, 0.999)
UNEQUAL_BETAS = ((0.9, 0.999), (0.9, 0.95), (0.9, 0.99), (0.99, 0.999))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def scale(step: float) -> float:
    return float(
        1.0
        + 0.5
        * (sigmoid((step - 200.0) / 20.0) - sigmoid((step - 400.0) / 20.0))
    )


def gradient(step: float) -> np.ndarray:
    direction = np.asarray(
        [10.0, -8.0 + np.sin(step / 10.0), 9.0 + np.cos(step / 10.0)],
        dtype=np.float64,
    )
    return scale(step) * direction / np.linalg.norm(direction)


def adam_normalized_update(
    gradients: np.ndarray, beta1: float, beta2: float, epsilon: float = 1e-8
) -> np.ndarray:
    """Replay the exact convention used by the reference controlled figures."""
    steps, dimensions = gradients.shape
    first = np.zeros(dimensions, dtype=np.float64)
    second = np.zeros(dimensions, dtype=np.float64)
    result = np.zeros_like(gradients, dtype=np.float64)
    for k in range(1, steps):
        first = beta1 * first + (1.0 - beta1) * gradients[k]
        second = beta2 * second + (1.0 - beta2) * gradients[k] ** 2
        corrected_first = first / (1.0 - beta1**k)
        corrected_second = second / (1.0 - beta2**k)
        result[k] = corrected_first / (np.sqrt(corrected_second) + epsilon)
    return result


def configure_matplotlib(use_tex: bool) -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use("default")
    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        }
    )


def make_figure(
    times: np.ndarray,
    gradient_norm: np.ndarray,
    curves: list[tuple[str, np.ndarray]],
    output: Path,
    y_limits: tuple[float, float],
) -> None:
    colors = ("#006400", "#1f4e79", "#8b0000", "#a05000")
    markers = ("o", "s", "^", "D")
    figure, left = plt.subplots(figsize=(4, 3))
    handles = [
        left.plot(
            times,
            gradient_norm,
            linestyle="-",
            linewidth=2.0,
            color="#222222",
            label=r"$\|\mathbf{g}_k\|$",
        )[0]
    ]
    left.set_xlabel("Steps")
    left.set_ylabel(r"Norm of $\mathbf{g}_k$")
    left.set_xlim(50, 550)
    right = left.twinx()
    right.set_ylabel(r"Norm of $\mathbf{R}_k$")
    right.set_ylim(*y_limits)
    for index, (label, values) in enumerate(curves):
        handles.append(
            right.plot(
                times,
                values,
                linewidth=1.8,
                color=colors[index],
                marker=markers[index],
                markersize=3.0,
                markevery=(4 * index, 16),
                label=label,
            )[0]
        )
    left.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower left",
        frameon=True,
        fontsize=7,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/controlled"))
    parser.add_argument("--use-tex", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib(args.use_tex)
    times = np.arange(0, 601, dtype=np.float64)
    gradients = np.stack([gradient(step) for step in times])
    gradient_norm = np.linalg.norm(gradients, axis=1)

    equal = {
        beta: np.linalg.norm(adam_normalized_update(gradients, beta, beta), axis=1)
        for beta in EQUAL_BETAS
    }
    unequal = {
        pair: np.linalg.norm(adam_normalized_update(gradients, *pair), axis=1)
        for pair in UNEQUAL_BETAS
    }
    pooled = np.concatenate(
        [values[1:] for values in (*equal.values(), *unequal.values())]
    )
    span = float(np.max(pooled) - np.min(pooled))
    if span == 0.0:
        span = max(1.0, float(np.max(pooled)))
    y_limits = (
        max(0.0, float(np.min(pooled)) - 0.1 * span),
        float(np.max(pooled)) + 0.1 * span,
    )

    equal_curves = [
        (
            rf"$\|\mathbf{{R}}_k\|$ for $\beta_1 = \beta_2 = {beta}$",
            equal[beta],
        )
        for beta in EQUAL_BETAS
    ]
    unequal_curves = [
        (
            rf"$\|\mathbf{{R}}_k\|$ for $\beta_1 = {b1},\,\beta_2 = {b2}$",
            unequal[(b1, b2)],
        )
        for b1, b2 in UNEQUAL_BETAS
    ]
    make_figure(
        times,
        gradient_norm,
        equal_curves,
        args.output_dir / "figure_equal_betas.pdf",
        y_limits,
    )
    make_figure(
        times,
        gradient_norm,
        unequal_curves,
        args.output_dir / "figure_unequal_betas.pdf",
        y_limits,
    )
    summary = {
        "steps": int(times.size),
        "indexing": "index 0 is the zero initial state; gradients 1..600 are replayed",
        "epsilon": 1e-8,
        "epsilon_placement": "outside_sqrt",
        "equal_betas": list(EQUAL_BETAS),
        "unequal_betas": [list(pair) for pair in UNEQUAL_BETAS],
        "gradient_norm_min": float(np.min(gradient_norm)),
        "gradient_norm_max": float(np.max(gradient_norm)),
        "shared_update_norm_limits": list(y_limits),
    }
    (args.output_dir / "controlled_signal_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote controlled figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
