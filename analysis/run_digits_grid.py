"""Run the 3x3-by-3-seed Digits training and literal decomposition pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BETAS = (0.9, 0.99, 0.999)
SEEDS = (0, 1, 2)


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs/digits"))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    trainer = here / "train_digits_gradient_trajectory.py"
    analyzer = here / "literal_decomposition.py"
    for beta1 in BETAS:
        for beta2 in BETAS:
            for seed in SEEDS:
                run_dir = (
                    args.output_root
                    / f"b1-{tag(beta1)}_b2-{tag(beta2)}_seed-{seed}"
                )
                gradients = run_dir / "gradients.npy"
                metadata = run_dir / "metadata.json"
                analysis = run_dir / "analysis.json"
                if not args.skip_training and (
                    args.force or not (gradients.exists() and metadata.exists())
                ):
                    run(
                        [
                            sys.executable,
                            str(trainer),
                            "--output-dir",
                            str(run_dir),
                            "--steps",
                            str(args.steps),
                            "--learning-rate",
                            "3e-4",
                            "--beta1",
                            str(beta1),
                            "--beta2",
                            str(beta2),
                            "--seed",
                            str(seed),
                            "--epsilon",
                            "1e-8",
                            "--weight-decay",
                            "0",
                            "--batch-size",
                            "0",
                        ]
                    )
                if not args.skip_analysis and (args.force or not analysis.exists()):
                    if not gradients.exists():
                        raise FileNotFoundError(
                            f"Missing {gradients}; run training or remove --skip-training"
                        )
                    run(
                        [
                            sys.executable,
                            str(analyzer),
                            "--gradients",
                            str(gradients),
                            "--output",
                            str(analysis),
                            "--beta1",
                            str(beta1),
                            "--beta2",
                            str(beta2),
                            "--windows",
                            "6",
                            "10",
                            "100",
                            "1000",
                        ]
                    )


if __name__ == "__main__":
    main()
