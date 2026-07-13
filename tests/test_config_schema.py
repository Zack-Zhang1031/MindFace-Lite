from __future__ import annotations

import yaml

from mindface.configuration.schema import validate_all_configs, validate_config


def test_all_project_yaml_configs_match_their_schemas() -> None:
    reports = validate_all_configs()

    assert len(reports) == 32
    assert all(report.valid for report in reports), {
        str(report.path): report.errors for report in reports if not report.valid
    }


def test_training_schema_reports_missing_nested_field(tmp_path) -> None:
    path = tmp_path / "train-broken.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"dir": "data/example"},
                "features": {"fps": 25, "frame_ms": 40, "feature_dim": 70},
                "model": {"type": "mlp"},
                "train": {"epochs": 1, "batch_size": 2, "lr": 0.001},
                "output": {"checkpoint_path": "outputs/checkpoints/test.pt"},
            }
        ),
        encoding="utf-8",
    )

    report = validate_config(path, schema_name="training")

    assert not report.valid
    assert any("model.params" in error for error in report.errors)


def test_dataset_schema_rejects_split_ratios_that_do_not_sum_to_one(tmp_path) -> None:
    path = tmp_path / "prepare-grid.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "raw_grid_dir": "data/raw/grid",
                "output_dir": "data/processed/grid",
                "scan": {},
                "features": {"fps": 25, "frame_ms": 40, "feature_dim": 70},
                "labels": {},
                "split": {"train_ratio": 0.8, "val_ratio": 0.3, "test_ratio": 0.1},
                "output": {},
            }
        ),
        encoding="utf-8",
    )

    report = validate_config(path, schema_name="datasets")

    assert not report.valid
    assert any("split ratios" in error for error in report.errors)

