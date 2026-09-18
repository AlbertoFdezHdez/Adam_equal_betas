from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .io import load_dict, save_dict, save_dict_pickle
from .paths import get_experiment_merged_dir, get_experiment_runs_dir


BASE_KEYS = ("model", "dataset", "epochs", "initialization", "optimizer", "wd", "loss")
VAR_KEYS = ("beta1", "beta2", "seed")


def _sanitize_float(value: Any) -> str:
    return str(value).replace(".", "p")


def build_run_name(experiment_name: str, history: dict[str, Any]) -> str:
    return (
        f"{experiment_name}"
        f"__{history['model']}"
        f"__{history['dataset']}"
        f"__b1-{_sanitize_float(history['beta1'])}"
        f"__b2-{_sanitize_float(history['beta2'])}"
        f"__seed-{history['seed']}"
    )


def save_experiment_run(experiment_name: str, history: dict[str, Any]) -> Path:
    history = dict(history)
    history.setdefault("experiment", experiment_name)
    run_name = build_run_name(experiment_name, history)
    output_path = get_experiment_runs_dir(experiment_name) / run_name
    save_dict(output_path, history)
    return output_path.with_suffix(".json")


def _summary_row(data: dict[str, Any]) -> dict[str, Any]:
    val_loss = data.get("val_loss", [])
    val_metric = data.get("val_metric", [])
    train_loss = data.get("train_loss", [])
    return {
        "experiment": data.get("experiment"),
        "model": data.get("model"),
        "dataset": data.get("dataset"),
        "seed": data.get("seed"),
        "beta1": data.get("beta1"),
        "beta2": data.get("beta2"),
        "epochs": data.get("epochs"),
        "optimizer": data.get("optimizer"),
        "wd": data.get("wd"),
        "lr": data.get("lr"),
        "training_wall_time_sec": data.get("training_wall_time_sec"),
        "wall_time_total_sec": data.get("wall_time_total_sec"),
        "final_train_loss": train_loss[-1] if train_loss else None,
        "final_val_loss": val_loss[-1] if val_loss else None,
        "best_val_loss": min(val_loss) if val_loss else None,
        "best_val_metric": max(val_metric) if val_metric else None,
    }


def merge_experiment_runs(experiment_name: str) -> dict[str, Any]:
    runs_dir = get_experiment_runs_dir(experiment_name)
    merged_dir = get_experiment_merged_dir(experiment_name)

    files = sorted(path for path in runs_dir.glob("*.json") if path.is_file())
    if not files:
        return {"runs": {}}

    first = load_dict(files[0])
    merged: dict[str, Any] = {k: first[k] for k in BASE_KEYS if k in first}
    merged["experiment"] = experiment_name
    merged["runs"] = {}

    summary_rows = []
    for file_path in files:
        run_data = load_dict(file_path)
        summary_rows.append(_summary_row(run_data))
        run_key = tuple(run_data[k] for k in VAR_KEYS)
        payload = {
            k: v
            for k, v in run_data.items()
            if k not in BASE_KEYS and k not in VAR_KEYS and k != "experiment"
        }
        merged["runs"][run_key] = payload

    if "model" in merged and "dataset" in merged:
        stem = f"{merged['model']}_{merged['dataset']}"
    else:
        stem = experiment_name

    save_dict_pickle(merged_dir / stem, merged)

    csv_path = merged_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    return merged
