from __future__ import annotations

import csv
import logging

import numpy as np

from mindface.artifacts.model_bundle import load_model_bundle
from mindface.training.trainer import train_from_config


def _write_tiny_dataset(root) -> None:
    rows = []
    rng = np.random.default_rng(7)
    for index, split in enumerate(("train", "val")):
        features_name = f"features_{index}.npy"
        targets_name = f"targets_{index}.npy"
        np.save(root / features_name, rng.random((4, 70), dtype=np.float32))
        np.save(root / targets_name, rng.random((4, 3), dtype=np.float32))
        rows.append({"split": split, "features": features_name, "targets": targets_name})
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "features", "targets"])
        writer.writeheader()
        writer.writerows(rows)


def test_tiny_training_saves_loads_and_resumes_optimizer_state(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_tiny_dataset(dataset_dir)
    best_path = tmp_path / "best.pt"
    last_path = tmp_path / "last.pt"
    cfg = {
        "dataset": {"dir": str(dataset_dir)},
        "features": {"fps": 25, "frame_ms": 40, "feature_dim": 70},
        "model": {
            "type": "mlp",
            "params": {"input_dim": 70, "hidden_dim": 8, "output_dim": 3, "dropout": 0.0},
        },
        "train": {"seed": 3, "device": "cpu", "epochs": 1, "batch_size": 2, "lr": 0.001, "num_workers": 0},
        "output": {
            "checkpoint_path": str(best_path),
            "last_checkpoint_path": str(last_path),
            "experiment_root": str(tmp_path / "experiments"),
        },
    }
    logger = logging.getLogger("test_training_resume")

    train_from_config(cfg, logger)
    first = load_model_bundle(last_path)

    assert best_path.exists()
    assert first.epoch == 1
    assert first.optimizer_state is not None
    assert first.feature_spec.feature_dim == 70

    cfg["train"] = {**cfg["train"], "epochs": 2, "resume_from": str(last_path)}
    train_from_config(cfg, logger)
    resumed = load_model_bundle(last_path)

    assert resumed.epoch == 2
    assert resumed.optimizer_state is not None
    assert resumed.metadata["resumed_from_epoch"] == 1

