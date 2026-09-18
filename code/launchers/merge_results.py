import argparse
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils import merge_experiment_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="Experiment directory name inside descargas/results")
    args = parser.parse_args()
    merged = merge_experiment_runs(args.experiment)
    print(f"Merged {len(merged.get('runs', {}))} runs for experiment '{args.experiment}'.")


if __name__ == "__main__":
    main()
