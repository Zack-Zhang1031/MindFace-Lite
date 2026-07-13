from __future__ import annotations

import argparse
import json

from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.logger import setup_logger


def run_synthetic(args: argparse.Namespace) -> int:
    from mindface.data.synthetic_generation import generate_synthetic_dataset

    manifest_path = generate_synthetic_dataset(
        load_yaml(args.config),
        setup_logger("generate_synthetic_dataset", "outputs/logs/generate_synthetic_dataset.log"),
    )
    print(f"Dataset manifest: {manifest_path}")
    return 0


def run_prepare_grid(args: argparse.Namespace) -> int:
    from mindface.data.grid_audio import prepare_grid_dataset

    manifest_path = prepare_grid_dataset(
        load_yaml(args.config),
        max_samples_override=args.max_samples,
        output_dir_override=args.output_dir,
    )
    print(f"Manifest: {manifest_path}")
    print("Train with: mindface train --config configs/training/train-grid-mlp.yaml")
    return 0


def run_extract_landmarks(args: argparse.Namespace) -> int:
    from mindface.data.grid_landmarks import (
        check_mediapipe_available,
        landmark_config_from_mapping,
        prepare_grid_video_landmarks,
    )
    from mindface.data.grid_quality import build_grid_landmark_quality_report

    cfg = load_yaml(args.config)
    output_dir = resolve_path(args.output_dir or cfg["grid"]["output_dir"])
    if args.quality_only:
        configured_report = cfg.get("quality", {}).get("report_path")
        report_path = output_dir / "quality_report.json" if args.output_dir else configured_report
        report = build_grid_landmark_quality_report(
            output_dir,
            report_path=report_path,
            min_detection_rate=float(cfg.get("quality", {}).get("min_detection_rate", 0.95)),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    ok, message = check_mediapipe_available()
    if args.check_deps:
        print(message)
        return 0
    if not ok:
        raise RuntimeError(message)
    landmark_cfg = landmark_config_from_mapping(cfg, args.max_videos, args.output_dir, args.delegate)
    manifest_path = prepare_grid_video_landmarks(
        landmark_cfg,
        setup_logger("grid_video_landmarks", cfg["logging"]["log_path"]),
    )
    print(f"GRID video landmark manifest: {manifest_path}")
    if landmark_cfg.quality_report_path is not None:
        print(f"Quality report: {landmark_cfg.quality_report_path}")
    return 0


def run_align_landmarks(args: argparse.Namespace) -> int:
    from mindface.data.grid_supervised import prepare_grid_landmark_dataset

    cfg = load_yaml(args.config)
    logger = setup_logger("prepare_grid_landmark", cfg["output"]["log_path"])
    manifest_path = prepare_grid_landmark_dataset(cfg, max_samples_override=args.max_samples)
    logger.info("Prepared GRID landmark supervised dataset: %s", manifest_path)
    print(f"Manifest: {manifest_path}")
    print("Train with: mindface train --config configs/training/train-grid-landmark-mlp.yaml")
    return 0
