from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from mindface.utils.config import resolve_path


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_grid_landmark_quality_report(
    output_dir: str | Path,
    report_path: str | Path | None = None,
    min_detection_rate: float = 0.95,
) -> dict[str, Any]:
    output_dir = resolve_path(output_dir)
    manifest_path = output_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"GRID landmark manifest not found: {manifest_path}")

    manifest_rows = _read_manifest(manifest_path)
    mouth_open_values: list[float] = []
    mouth_width_values: list[float] = []
    mouth_round_values: list[float] = []
    sample_reports: list[dict[str, Any]] = []
    total_frames = 0
    detected_frames = 0
    missing_csv = 0

    for row in manifest_rows:
        sample_id = row.get("sample_id", "")
        landmarks_rel = row.get("landmarks_csv", "")
        csv_path = output_dir / landmarks_rel
        if not csv_path.exists():
            missing_csv += 1
            sample_reports.append(
                {
                    "sample_id": sample_id,
                    "status": "missing_csv",
                    "landmarks_csv": str(csv_path),
                    "num_frames": 0,
                    "detected_frames": 0,
                    "detection_rate": 0.0,
                }
            )
            continue

        sample_frames = 0
        sample_detected = 0
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for frame in reader:
                sample_frames += 1
                detected = int(float(frame.get("face_detected", 0)))
                sample_detected += detected
                mouth_open_values.append(float(frame.get("mouth_open", 0.0)))
                mouth_width_values.append(float(frame.get("mouth_width", 0.0)))
                mouth_round_values.append(float(frame.get("mouth_round", 0.0)))
        total_frames += sample_frames
        detected_frames += sample_detected
        detection_rate = sample_detected / sample_frames if sample_frames else 0.0
        status = "pass" if detection_rate >= min_detection_rate else "low_detection"
        sample_reports.append(
            {
                "sample_id": sample_id,
                "status": status,
                "landmarks_csv": str(csv_path),
                "num_frames": sample_frames,
                "detected_frames": sample_detected,
                "detection_rate": float(detection_rate),
            }
        )

    detection_rates = [float(item["detection_rate"]) for item in sample_reports]
    low_detection = [item for item in sample_reports if item["status"] != "pass"]
    report = {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "num_samples": len(manifest_rows),
        "missing_landmark_csv": missing_csv,
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "overall_detection_rate": float(detected_frames / total_frames) if total_frames else 0.0,
        "sample_detection_rate": {
            "min": float(min(detection_rates)) if detection_rates else None,
            "max": float(max(detection_rates)) if detection_rates else None,
            "mean": float(mean(detection_rates)) if detection_rates else None,
        },
        "mouth_open": _stats(mouth_open_values),
        "mouth_width": _stats(mouth_width_values),
        "mouth_round": _stats(mouth_round_values),
        "low_detection_sample_count": len(low_detection),
        "low_detection_samples": low_detection[:30],
        "notes": [
            "High detection rate means MediaPipe found a face on most frames.",
            "This report checks label extraction quality; it does not prove audio-video lip-sync alignment.",
            "Samples with low detection should be inspected before using landmark targets for supervised training.",
        ],
    }

    if report_path is not None:
        resolved_report_path = resolve_path(report_path)
    else:
        resolved_report_path = output_dir / "quality_report.json"
    report["report_path"] = str(resolved_report_path)
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report
