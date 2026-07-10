from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.data.grid_quality import build_grid_landmark_quality_report
from mindface.data.grid_landmarks import (
    DEFAULT_FACE_LANDMARKER_URL,
    LandmarkConfig,
    check_mediapipe_available,
    prepare_grid_video_landmarks,
)
from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.logger import setup_logger


def build_landmark_config(
    cfg: dict,
    max_videos: int | None,
    output_dir: str | None,
    delegate: str | None,
) -> LandmarkConfig:
    grid_cfg = cfg["grid"]
    landmark_cfg = cfg["landmarks"]
    configured_max = grid_cfg.get("max_videos")
    resolved_output_dir = resolve_path(output_dir if output_dir is not None else grid_cfg["output_dir"])
    configured_quality_path = cfg.get("quality", {}).get("report_path")
    quality_report_path = (
        resolved_output_dir / "quality_report.json"
        if output_dir is not None
        else (resolve_path(configured_quality_path) if configured_quality_path else None)
    )
    return LandmarkConfig(
        video_dir=resolve_path(grid_cfg["video_dir"]),
        output_dir=resolved_output_dir,
        max_videos=max_videos if max_videos is not None else configured_max,
        split_ratios=tuple(float(x) for x in grid_cfg["split_ratios"]),
        seed=int(grid_cfg.get("seed", 42)),
        refine_landmarks=bool(landmark_cfg.get("refine_landmarks", True)),
        min_detection_confidence=float(landmark_cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(landmark_cfg.get("min_tracking_confidence", 0.5)),
        model_path=resolve_path(landmark_cfg.get("model_path", "models/mediapipe/face_landmarker.task")),
        model_url=str(landmark_cfg.get("model_url", DEFAULT_FACE_LANDMARKER_URL)),
        auto_download_model=bool(landmark_cfg.get("auto_download_model", True)),
        delegate=str(delegate if delegate is not None else landmark_cfg.get("delegate", "cpu")),
        quality_report_path=quality_report_path,
        quality_min_detection_rate=float(cfg.get("quality", {}).get("min_detection_rate", 0.95)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract mouth landmark labels from GRID videos.")
    parser.add_argument("--config", default="configs/grid_video_landmarks.yaml")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit videos for a quick debug run.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--delegate", choices=["cpu", "gpu"], default=None, help="Override MediaPipe delegate.")
    parser.add_argument("--check-deps", action="store_true", help="Only check optional dependencies.")
    parser.add_argument("--quality-only", action="store_true", help="Build a quality report from an existing landmark manifest.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_dir = resolve_path(args.output_dir if args.output_dir is not None else cfg["grid"]["output_dir"])

    if args.quality_only:
        report_path = output_dir / "quality_report.json" if args.output_dir is not None else cfg.get("quality", {}).get("report_path")
        report = build_grid_landmark_quality_report(
            output_dir,
            report_path=report_path,
            min_detection_rate=float(cfg.get("quality", {}).get("min_detection_rate", 0.95)),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    ok, message = check_mediapipe_available()
    if args.check_deps:
        print(message)
        return
    if not ok:
        raise SystemExit(message)

    logger = setup_logger("grid_video_landmarks", cfg["logging"]["log_path"])
    landmark_cfg = build_landmark_config(cfg, args.max_videos, args.output_dir, args.delegate)
    try:
        manifest_path = prepare_grid_video_landmarks(landmark_cfg, logger)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"GRID video landmark manifest: {manifest_path}")
    if landmark_cfg.quality_report_path is not None:
        print(f"Quality report: {landmark_cfg.quality_report_path}")


if __name__ == "__main__":
    main()
