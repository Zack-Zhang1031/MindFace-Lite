from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.audio.features import compute_rms, frame_times, read_wav_mono, rms_to_mouth_open
from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.csv_io import write_rule_csv
from mindface.utils.logger import setup_logger
from mindface.visual.mouth_drawer import params_to_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based RMS to mouth_open demo.")
    parser.add_argument("--config", default="configs/rule_demo.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("rule_mouth_demo", cfg["logging"]["log_path"])

    audio_path = resolve_path(cfg["audio"]["input_path"])
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing input WAV: {audio_path}. Run scripts/00_generate_test_audio.py first.")

    sample_rate, waveform = read_wav_mono(audio_path)
    fps = int(cfg["audio"]["fps"])
    frame_ms = float(cfg["audio"]["frame_ms"])
    rms = compute_rms(waveform, sample_rate, fps=fps, frame_ms=frame_ms)
    mouth_open = rms_to_mouth_open(
        rms,
        noise_floor=float(cfg["mouth"]["noise_floor"]),
        gamma=float(cfg["mouth"]["gamma"]),
        smoothing=float(cfg["mouth"]["smoothing"]),
    )
    times = frame_times(len(rms), fps)
    params = np.stack([mouth_open], axis=1)

    write_rule_csv(cfg["csv"]["output_path"], times, rms, mouth_open)
    params_to_video(
        params,
        cfg["video"]["output_path"],
        fps=fps,
        width=int(cfg["video"]["width"]),
        height=int(cfg["video"]["height"]),
    )

    logger.info("Input audio: %s", audio_path)
    logger.info("Frames: %d RMS range: %.6f..%.6f mouth range: %.3f..%.3f", len(rms), rms.min(), rms.max(), mouth_open.min(), mouth_open.max())
    print(f"CSV: {resolve_path(cfg['csv']['output_path'])}")
    print(f"Video: {resolve_path(cfg['video']['output_path'])}")


if __name__ == "__main__":
    main()
