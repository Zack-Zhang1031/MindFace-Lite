from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.audio.features import compute_rms, extract_audio_features, rms_to_mouth_open, write_wav_mono
from mindface.utils.config import ensure_dir, load_yaml
from mindface.utils.logger import setup_logger


def _synthetic_waveform(rng: np.random.Generator, sample_rate: int, frames: int, fps: int) -> np.ndarray:
    duration = frames / float(fps)
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / float(sample_rate)
    waveform = np.zeros_like(t)
    cursor = 0.05
    while cursor < duration - 0.1:
        syllable = float(rng.uniform(0.10, 0.32))
        pause = float(rng.uniform(0.02, 0.12))
        start = cursor
        end = min(duration, cursor + syllable)
        mask = (t >= start) & (t < end)
        local_t = t[mask] - start
        n = max(1, mask.sum())
        amp = float(rng.uniform(0.15, 0.65))
        freq = float(rng.uniform(130.0, 320.0))
        envelope = np.clip(np.sin(np.linspace(0, np.pi, n, dtype=np.float32)), 0.0, 1.0) ** float(rng.uniform(0.6, 1.1))
        waveform[mask] += amp * envelope * (
            np.sin(2 * np.pi * freq * local_t) + 0.25 * np.sin(2 * np.pi * freq * 2.0 * local_t)
        )
        cursor += syllable + pause
    waveform += 0.004 * rng.normal(size=waveform.shape).astype(np.float32)
    return np.clip(waveform, -0.95, 0.95).astype(np.float32)


def _targets_from_waveform(waveform: np.ndarray, sample_rate: int, fps: int, frame_ms: float, rng: np.random.Generator) -> np.ndarray:
    rms = compute_rms(waveform, sample_rate, fps=fps, frame_ms=frame_ms)
    mouth_open = rms_to_mouth_open(rms, noise_floor=0.008, gamma=0.7, smoothing=0.45)
    delayed = np.concatenate([[mouth_open[0]], mouth_open[:-1]])
    mouth_width = np.clip(0.45 + 0.30 * mouth_open + 0.05 * rng.normal(size=len(mouth_open)), 0.0, 1.0)
    lip_round = np.clip(0.18 + 0.35 * delayed + 0.04 * rng.normal(size=len(mouth_open)), 0.0, 1.0)
    return np.stack([mouth_open, mouth_width, lip_round], axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic mouth parameter dataset.")
    parser.add_argument("--config", default="configs/synthetic_dataset.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("generate_synthetic_dataset", "outputs/logs/generate_synthetic_dataset.log")
    out_dir = ensure_dir(cfg["dataset"]["output_dir"])
    wav_dir = ensure_dir(out_dir / "wavs")

    rng = np.random.default_rng(int(cfg["dataset"]["seed"]))
    num_samples = int(cfg["dataset"]["num_samples"])
    frames = int(cfg["dataset"]["frames_per_sample"])
    train_ratio = float(cfg["dataset"]["train_ratio"])
    sample_rate = int(cfg["audio"]["sample_rate"])
    fps = int(cfg["audio"]["fps"])
    frame_ms = float(cfg["audio"]["frame_ms"])

    rows = []
    for idx in range(num_samples):
        waveform = _synthetic_waveform(rng, sample_rate, frames, fps)
        features = extract_audio_features(waveform, sample_rate, fps=fps, frame_ms=frame_ms)
        targets = _targets_from_waveform(waveform, sample_rate, fps, frame_ms, rng)
        min_len = min(len(features), len(targets), frames)
        features = features[:min_len]
        targets = targets[:min_len]

        feature_name = f"features_{idx:03d}.npy"
        target_name = f"targets_{idx:03d}.npy"
        wav_name = f"wavs/sample_{idx:03d}.wav"
        np.save(out_dir / feature_name, features)
        np.save(out_dir / target_name, targets)
        write_wav_mono(out_dir / wav_name, sample_rate, waveform)
        split = "train" if idx < int(num_samples * train_ratio) else "val"
        rows.append({"split": split, "features": feature_name, "targets": target_name, "wav": wav_name})

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "features", "targets", "wav"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Generated synthetic dataset: %s samples=%d", manifest_path, num_samples)
    print(f"Dataset manifest: {manifest_path}")


if __name__ == "__main__":
    main()
