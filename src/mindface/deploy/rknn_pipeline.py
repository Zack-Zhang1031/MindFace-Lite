from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mindface.deploy.rknn_tools import (
    check_rknn_available,
    convert_onnx_to_rknn,
    create_feature_calibration_dataset,
    make_dry_run_report,
)
from mindface.utils.config import resolve_path
from mindface.utils.logger import setup_logger


def _write_report(report: dict[str, Any], report_path: str | Path | None) -> Path | None:
    if report_path is None:
        return None
    output_path = resolve_path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def run_rknn_pipeline(
    cfg: dict[str, Any],
    *,
    quantize: bool = False,
    run_inference: bool = False,
    dry_run: bool = False,
    report_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path | None]:
    quantization_enabled = bool(quantize or cfg["quantization"].get("enabled", False))
    dataset_path = cfg["quantization"].get("dataset_path")
    selected_report_path = report_path if report_path is not None else cfg.get("output", {}).get("report_path")
    runtime_inference = bool(run_inference or cfg["runtime"].get("run_inference", False))
    if dry_run:
        report = make_dry_run_report(
            onnx_path=cfg["model"]["onnx_path"],
            output_path=cfg["model"]["rknn_path"],
            target_platform=str(cfg["model"]["target_platform"]),
            do_quantization=quantization_enabled,
            dataset_path=dataset_path,
            run_inference=runtime_inference,
        )
        return report, _write_report(report, selected_report_path)

    ok, message = check_rknn_available()
    if not ok:
        raise RuntimeError(message)
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
        run_inference=runtime_inference,
        dummy_frames=int(cfg["input"]["frames"]),
        feature_dim=int(cfg["input"]["feature_dim"]),
    )
    return report, _write_report(report, selected_report_path)

