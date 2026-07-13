from __future__ import annotations

import logging

from mindface.data.synthetic_generation import generate_synthetic_dataset
from mindface.data.grid_landmarks import landmark_config_from_mapping
from mindface.deploy.benchmark import latency_summary
from mindface.deploy.rknn_pipeline import run_rknn_pipeline


def test_synthetic_generation_lives_in_package_module(tmp_path) -> None:
    cfg = {
        "dataset": {"output_dir": str(tmp_path / "data"), "seed": 1, "num_samples": 2, "frames_per_sample": 4, "train_ratio": 0.5},
        "audio": {"sample_rate": 16000, "fps": 25, "frame_ms": 40},
        "features": {"feature_dim": 70},
    }

    manifest = generate_synthetic_dataset(cfg, logging.getLogger("test_synthetic_generation"))

    assert manifest.exists()
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 3


def test_benchmark_latency_summary_is_reusable() -> None:
    report = latency_summary([1.0, 2.0, 3.0, 4.0])

    assert report["mean_ms"] == 2.5
    assert report["fps"] == 400.0


def test_rknn_dry_run_pipeline_is_reusable(tmp_path) -> None:
    cfg = {
        "model": {"onnx_path": str(tmp_path / "model.onnx"), "rknn_path": str(tmp_path / "model.rknn"), "target_platform": "rk3588"},
        "quantization": {"enabled": False, "dataset_path": None},
        "input": {"frames": 16, "feature_dim": 70},
        "runtime": {"run_inference": False, "verbose": False, "target": None},
        "output": {"report_path": str(tmp_path / "report.json")},
        "logging": {"log_path": str(tmp_path / "rknn.log")},
    }

    report, report_path = run_rknn_pipeline(cfg, dry_run=True)

    assert report["dry_run"] is True
    assert report_path == tmp_path / "report.json"
    assert report_path.exists()


def test_grid_landmark_config_construction_lives_in_package(tmp_path) -> None:
    cfg = {
        "grid": {
            "video_dir": str(tmp_path / "video"),
            "output_dir": str(tmp_path / "output"),
            "max_videos": None,
            "split_ratios": [0.8, 0.1, 0.1],
            "split_strategy": "auto",
            "seed": 42,
        },
        "landmarks": {"delegate": "cpu", "model_path": str(tmp_path / "face.task")},
        "quality": {"report_path": str(tmp_path / "quality.json"), "min_detection_rate": 0.95},
    }

    result = landmark_config_from_mapping(cfg, max_videos=8, output_dir=None, delegate="gpu")

    assert result.max_videos == 8
    assert result.delegate == "gpu"
    assert result.split_strategy == "auto"
