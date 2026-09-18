import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
OUTPUT_ROOT = Path(os.environ.get("ADAM_BETA_OUTPUT_ROOT", PROJECT_ROOT / "outputs")).expanduser()
LOGS_ROOT = OUTPUT_ROOT / "logs"
RESULTS_ROOT = OUTPUT_ROOT / "results"
DISPLAY_ROOT = OUTPUT_ROOT / "figures"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_experiment_results_dir(experiment_name: str) -> Path:
    return ensure_dir(RESULTS_ROOT / experiment_name)


def get_experiment_runs_dir(experiment_name: str) -> Path:
    return ensure_dir(get_experiment_results_dir(experiment_name) / "runs")


def get_experiment_merged_dir(experiment_name: str) -> Path:
    return ensure_dir(get_experiment_results_dir(experiment_name) / "merged")
