from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
DESCARGAS_ROOT = PROJECT_ROOT / "descargas"
LOGS_ROOT = DESCARGAS_ROOT / "logs"
RESULTS_ROOT = DESCARGAS_ROOT / "results"
DISPLAY_ROOT = PROJECT_ROOT / "display"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_experiment_results_dir(experiment_name: str) -> Path:
    return ensure_dir(RESULTS_ROOT / experiment_name)


def get_experiment_runs_dir(experiment_name: str) -> Path:
    return ensure_dir(get_experiment_results_dir(experiment_name) / "runs")


def get_experiment_merged_dir(experiment_name: str) -> Path:
    return ensure_dir(get_experiment_results_dir(experiment_name) / "merged")
