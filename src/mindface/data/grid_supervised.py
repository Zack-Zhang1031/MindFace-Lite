from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mindface.audio.features import extract_audio_features, read_wav_mono
from mindface.utils.config import resolve_path


@dataclass(frozen=True, slots=True)
class GridLandmarkSample:
    sample_id: str
    split: str
    audio_path: Path
    target_path: Path
    landmark_csv: Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _relative_to_project(path: Path) -> str:
    try:
        from mindface.utils.config import project_root

        return str(path.relative_to(project_root())).replace("\\", "/")
    except ValueError:
        return str(path)


def _build_audio_index(raw_grid_dir: Path) -> dict[str, Path]:
    audio_root = raw_grid_dir / "audio"
    if not audio_root.exists():
        raise FileNotFoundError(f"GRID audio directory not found: {audio_root}")
    index: dict[str, Path] = {}
    for wav_path in sorted(audio_root.rglob("*.wav")):
        index[wav_path.stem] = wav_path
    if not index:
        raise RuntimeError(f"No WAV files found under {audio_root}")
    return index


def _detection_rate(landmark_csv: Path) -> float:
    if not landmark_csv.exists():
        return 0.0
    rows = _read_csv(landmark_csv)
    if not rows:
        return 0.0
    detected = sum(1 for row in rows if int(float(row.get("face_detected", 0))) == 1)
    return detected / len(rows)


def find_grid_landmark_samples(
    raw_grid_dir: str | Path,
    landmark_dir: str | Path,
    min_detection_rate: float = 0.95,
    max_samples: int | None = None,
) -> list[GridLandmarkSample]:
    raw_grid_dir = resolve_path(raw_grid_dir)
    landmark_dir = resolve_path(landmark_dir)
    manifest_path = landmark_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"GRID landmark manifest not found: {manifest_path}. "
            "Run 'mindface data extract-landmarks' first."
        )

    audio_by_utterance = _build_audio_index(raw_grid_dir)
    samples: list[GridLandmarkSample] = []
    for row in _read_csv(manifest_path):
        sample_id = row["sample_id"]
        audio_path = audio_by_utterance.get(sample_id)
        if audio_path is None:
            continue
        landmark_csv = landmark_dir / row["landmarks_csv"]
        if _detection_rate(landmark_csv) < min_detection_rate:
            continue
        samples.append(
            GridLandmarkSample(
                sample_id=sample_id,
                split=row["split"],
                audio_path=audio_path,
                target_path=landmark_dir / row["targets"],
                landmark_csv=landmark_csv,
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    if not samples:
        raise RuntimeError("No aligned GRID audio + landmark samples selected.")
    return samples


def prepare_grid_landmark_dataset(cfg: dict[str, Any], max_samples_override: int | None = None) -> Path:
    raw_grid_dir = resolve_path(cfg["raw_grid_dir"])
    landmark_dir = resolve_path(cfg["landmark_dir"])
    output_dir = resolve_path(cfg["output_dir"])
    missing_inputs = []
    if not (raw_grid_dir / "audio").exists():
        missing_inputs.append(f"GRID audio directory: {raw_grid_dir / 'audio'}")
    if not (landmark_dir / "manifest.csv").exists():
        missing_inputs.append(f"GRID landmark manifest: {landmark_dir / 'manifest.csv'}")
    if missing_inputs:
        raise FileNotFoundError(
            "Cannot prepare GRID landmark supervised dataset. Missing required inputs:\n- "
            + "\n- ".join(missing_inputs)
            + "\nRestore data/raw/grid, run 'mindface data extract-landmarks', then rerun this command."
        )

    overwrite = bool(cfg.get("output", {}).get("overwrite", True))
    if output_dir.exists() and overwrite:
        resolved = output_dir.resolve()
        root = resolve_path(".").resolve()
        if not str(resolved).lower().startswith(str(root).lower()):
            raise RuntimeError(f"Refusing to remove output outside project: {resolved}")
        if output_dir.name in {"data", "raw", "grid"}:
            raise RuntimeError(f"Refusing to remove suspicious output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_samples = max_samples_override
    if max_samples is None and cfg.get("max_samples") is not None:
        max_samples = int(cfg["max_samples"])
    samples = find_grid_landmark_samples(
        raw_grid_dir=raw_grid_dir,
        landmark_dir=landmark_dir,
        min_detection_rate=float(cfg.get("min_detection_rate", 0.95)),
        max_samples=max_samples,
    )

    fps = int(cfg["features"]["fps"])
    frame_ms = float(cfg["features"]["frame_ms"])
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        sample_rate, waveform = read_wav_mono(sample.audio_path)
        features = extract_audio_features(waveform, sample_rate, fps=fps, frame_ms=frame_ms)
        targets = np.load(sample.target_path).astype(np.float32)
        frame_count = min(len(features), len(targets))
        if frame_count <= 0:
            continue
        features = features[:frame_count].astype(np.float32)
        targets = targets[:frame_count].astype(np.float32)

        feature_name = f"features_{index:06d}.npy"
        target_name = f"targets_{index:06d}.npy"
        np.save(output_dir / feature_name, features)
        np.save(output_dir / target_name, targets)
        rows.append(
            {
                "split": sample.split,
                "features": feature_name,
                "targets": target_name,
                "sample_id": sample.sample_id,
                "frames": frame_count,
                "sample_rate": sample_rate,
                "label_source": "grid_video_landmark",
                "source_audio": _relative_to_project(sample.audio_path),
                "source_landmarks": _relative_to_project(sample.landmark_csv),
                "source_targets": _relative_to_project(sample.target_path),
            }
        )

    if not rows:
        raise RuntimeError("No non-empty samples were produced after feature/target alignment.")
    manifest_path = output_dir / str(cfg.get("output", {}).get("manifest_name", "manifest.csv"))
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "split",
            "features",
            "targets",
            "sample_id",
            "frames",
            "sample_rate",
            "label_source",
            "source_audio",
            "source_landmarks",
            "source_targets",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path
