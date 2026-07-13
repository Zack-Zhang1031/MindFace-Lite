from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from mindface.audio.features import compute_rms, rms_to_mouth_open, write_wav_mono
from mindface.audio.spec import FeatureSpec
from mindface.utils.config import ensure_dir


def _synthetic_waveform(rng: np.random.Generator, sample_rate: int, frames: int, fps: int) -> np.ndarray:
    duration = frames / float(fps)
    time_values = np.arange(int(sample_rate * duration), dtype=np.float32) / float(sample_rate)
    waveform = np.zeros_like(time_values)
    cursor = 0.05
    while cursor < duration - 0.1:
        syllable = float(rng.uniform(0.10, 0.32))
        pause = float(rng.uniform(0.02, 0.12))
        end = min(duration, cursor + syllable)
        mask = (time_values >= cursor) & (time_values < end)
        local_time = time_values[mask] - cursor
        count = max(1, int(mask.sum()))
        amplitude = float(rng.uniform(0.15, 0.65))
        frequency = float(rng.uniform(130.0, 320.0))
        envelope = np.clip(np.sin(np.linspace(0, np.pi, count, dtype=np.float32)), 0.0, 1.0) ** float(
            rng.uniform(0.6, 1.1)
        )
        waveform[mask] += amplitude * envelope * (
            np.sin(2 * np.pi * frequency * local_time)
            + 0.25 * np.sin(2 * np.pi * frequency * 2.0 * local_time)
        )
        cursor += syllable + pause
    waveform += 0.004 * rng.normal(size=waveform.shape).astype(np.float32)
    return np.clip(waveform, -0.95, 0.95).astype(np.float32)


def _targets_from_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    feature_spec: FeatureSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    rms = compute_rms(
        waveform,
        sample_rate,
        fps=feature_spec.fps,
        frame_ms=feature_spec.frame_ms,
    )
    mouth_open = rms_to_mouth_open(rms, noise_floor=0.008, gamma=0.7, smoothing=0.45)
    delayed = np.concatenate([[mouth_open[0]], mouth_open[:-1]])
    mouth_width = np.clip(0.45 + 0.30 * mouth_open + 0.05 * rng.normal(size=len(mouth_open)), 0.0, 1.0)
    lip_round = np.clip(0.18 + 0.35 * delayed + 0.04 * rng.normal(size=len(mouth_open)), 0.0, 1.0)
    return np.stack([mouth_open, mouth_width, lip_round], axis=1).astype(np.float32)


def generate_synthetic_dataset(cfg: dict[str, Any], logger: Any) -> Path:
    out_dir = ensure_dir(cfg["dataset"]["output_dir"])
    ensure_dir(out_dir / "wavs")
    rng = np.random.default_rng(int(cfg["dataset"]["seed"]))
    num_samples = int(cfg["dataset"]["num_samples"])
    frames = int(cfg["dataset"]["frames_per_sample"])
    train_ratio = float(cfg["dataset"]["train_ratio"])
    sample_rate = int(cfg["audio"]["sample_rate"])
    feature_spec = FeatureSpec.from_mapping(
        {
            "fps": cfg["audio"]["fps"],
            "frame_ms": cfg["audio"]["frame_ms"],
            "feature_dim": cfg.get("features", {}).get("feature_dim", 70),
        }
    )

    rows: list[dict[str, str | int]] = []
    for index in range(num_samples):
        waveform = _synthetic_waveform(rng, sample_rate, frames, feature_spec.fps)
        features = feature_spec.extract(waveform, sample_rate)
        targets = _targets_from_waveform(waveform, sample_rate, feature_spec, rng)
        frame_count = min(len(features), len(targets), frames)
        feature_name = f"features_{index:03d}.npy"
        target_name = f"targets_{index:03d}.npy"
        wav_name = f"wavs/sample_{index:03d}.wav"
        np.save(out_dir / feature_name, features[:frame_count])
        np.save(out_dir / target_name, targets[:frame_count])
        write_wav_mono(out_dir / wav_name, sample_rate, waveform)
        split = "train" if index < int(num_samples * train_ratio) else "val"
        rows.append(
            {
                "schema_version": 1,
                "split": split,
                "features": feature_name,
                "targets": target_name,
                "wav": wav_name,
            }
        )

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["schema_version", "split", "features", "targets", "wav"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Generated synthetic dataset: %s samples=%d", manifest_path, num_samples)
    return manifest_path

