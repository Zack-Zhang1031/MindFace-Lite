from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.audio.features import write_wav_mono
from mindface.utils.logger import setup_logger


def build_test_voice(sample_rate: int, duration_sec: float) -> np.ndarray:
    """Generate speech-like test audio with changing energy."""
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / float(sample_rate)
    waveform = np.zeros_like(t)
    syllables = [
        (0.20, 0.55, 180.0, 0.28),
        (0.70, 1.05, 240.0, 0.45),
        (1.20, 1.55, 155.0, 0.22),
        (1.80, 2.35, 210.0, 0.62),
        (2.55, 2.95, 300.0, 0.35),
        (3.20, 3.80, 170.0, 0.55),
    ]
    for start, end, freq, amp in syllables:
        mask = (t >= start) & (t < end)
        local_t = t[mask] - start
        n = max(1, mask.sum())
        envelope = np.clip(np.sin(np.linspace(0, np.pi, n, dtype=np.float32)), 0.0, 1.0) ** 0.8
        voiced = np.sin(2 * np.pi * freq * local_t) + 0.35 * np.sin(2 * np.pi * freq * 2.0 * local_t)
        waveform[mask] += amp * envelope * voiced
    waveform += 0.006 * np.random.default_rng(7).normal(size=waveform.shape).astype(np.float32)
    return np.clip(waveform, -0.95, 0.95).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a speech-like WAV for MindFace-Lite demos.")
    parser.add_argument("--output", default="outputs/audio/test_voice.wav")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration-sec", type=float, default=4.2)
    args = parser.parse_args()

    logger = setup_logger("generate_test_audio", "outputs/logs/generate_test_audio.log")
    waveform = build_test_voice(args.sample_rate, args.duration_sec)
    write_wav_mono(args.output, args.sample_rate, waveform)
    logger.info("Generated test audio: %s", ROOT / args.output)
    print(f"Generated: {ROOT / args.output}")


if __name__ == "__main__":
    main()
