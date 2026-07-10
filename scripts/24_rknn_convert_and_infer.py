from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.deploy.rknn_tools import (
    check_rknn_available,
    convert_onnx_to_rknn,
    create_feature_calibration_dataset,
    make_dry_run_report,
)
from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.logger import setup_logger


def write_report(report: dict, report_path: str | Path | None) -> Path | None:
    if report_path is None:
        return None
    output_path = resolve_path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ONNX to RKNN and optionally run RKNN inference.")
    parser.add_argument("--config", default="configs/rknn_deploy.yaml")
    parser.add_argument("--quantize", action="store_true", help="Enable RKNN build-time quantization.")
    parser.add_argument("--run-inference", action="store_true", help="Run RKNN runtime inference after conversion.")
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print planned RKNN steps without importing RKNN.")
    parser.add_argument("--report-path", default=None, help="Override RKNN deployment report JSON path.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    quantization_enabled = bool(args.quantize or cfg["quantization"].get("enabled", False))
    dataset_path = cfg["quantization"].get("dataset_path")
    report_path = args.report_path if args.report_path is not None else cfg.get("output", {}).get("report_path")

    if args.dry_run:
        report = make_dry_run_report(
            onnx_path=cfg["model"]["onnx_path"],
            output_path=cfg["model"]["rknn_path"],
            target_platform=str(cfg["model"]["target_platform"]),
            do_quantization=quantization_enabled,
            dataset_path=dataset_path,
            run_inference=bool(args.run_inference or cfg["runtime"].get("run_inference", False)),
        )
        saved_path = write_report(report, report_path)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if saved_path is not None:
            print(f"Report: {saved_path}")
        return

    ok, message = check_rknn_available()
    if args.check_deps:
        print(message)
        return
    if not ok:
        raise RuntimeError(
            message
            + "\nUse --dry-run on Windows to validate the config, or run this script inside an Ubuntu RKNN environment."
        )

    logger = setup_logger("rknn_convert", cfg["logging"]["log_path"])
    if quantization_enabled and dataset_path is None and bool(cfg["quantization"].get("auto_create_dummy_dataset", False)):
        dataset_path = create_feature_calibration_dataset(
            cfg["quantization"]["dummy_dataset_dir"],
            num_samples=int(cfg["quantization"]["dummy_samples"]),
            frames=int(cfg["input"]["frames"]),
            feature_dim=int(cfg["input"]["feature_dim"]),
            seed=42,
        )
        logger.warning("Using dummy RKNN calibration dataset: %s", dataset_path)

    report = convert_onnx_to_rknn(
        onnx_path=cfg["model"]["onnx_path"],
        output_path=cfg["model"]["rknn_path"],
        target_platform=str(cfg["model"]["target_platform"]),
        do_quantization=quantization_enabled,
        dataset_path=dataset_path,
        verbose=bool(cfg["runtime"].get("verbose", False)),
        runtime_target=cfg["runtime"].get("target"),
        run_inference=bool(args.run_inference or cfg["runtime"].get("run_inference", False)),
        dummy_frames=int(cfg["input"]["frames"]),
        feature_dim=int(cfg["input"]["feature_dim"]),
    )
    logger.info("RKNN report: %s", report)
    saved_path = write_report(report, report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if saved_path is not None:
        print(f"Report: {saved_path}")


if __name__ == "__main__":
    main()
