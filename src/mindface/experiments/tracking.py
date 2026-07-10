from __future__ import annotations

import csv
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from mindface.utils.config import resolve_path


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_experiment_dir(root: str | Path, name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in name).strip("-")
    experiment_dir = resolve_path(root) / f"{timestamp_id()}_{safe_name}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def write_config_snapshot(experiment_dir: str | Path, cfg: dict[str, Any]) -> Path:
    path = Path(experiment_dir) / "config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def write_history_csv(experiment_dir: str | Path, history: list[dict[str, Any]]) -> Path:
    path = Path(experiment_dir) / "history.csv"
    fieldnames = ["epoch", "train_loss", "val_loss", "best_val_loss", "is_best"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    return path


def write_metrics_json(experiment_dir: str | Path, metrics: dict[str, Any]) -> Path:
    path = Path(experiment_dir) / "metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path


def write_latest_marker(root: str | Path, experiment_dir: str | Path) -> Path:
    marker = resolve_path(root) / "latest_train_run.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(Path(experiment_dir)), encoding="utf-8")
    return marker


def base_runtime_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

