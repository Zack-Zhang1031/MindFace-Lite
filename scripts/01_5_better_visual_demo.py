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
from mindface.visual.better_mouth_drawer import params_to_better_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1.5 better visual RMS mouth demo.")
    parser.add_argument("--config", default="configs/better_visual_demo.yaml")
    parser.add_argument("--input-wav", default=None, help="Override audio.input_path from the config.")
    parser.add_argument("--output-video", default=None, help="Override video.output_path from the config.")
    parser.add_argument("--output-csv", default=None, help="Override csv.output_path from the config.")
    parser.add_argument("--debug-overlay", action="store_true", help="Show mouth_open debug overlay in the video.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("better_visual_demo", cfg["logging"]["log_path"])
    audio_path = resolve_path(args.input_wav if args.input_wav is not None else cfg["audio"]["input_path"])
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

    csv_path = args.output_csv if args.output_csv is not None else cfg["csv"]["output_path"]
    video_path = args.output_video if args.output_video is not None else cfg["video"]["output_path"]
    show_debug_overlay = bool(args.debug_overlay or cfg["video"].get("show_debug_overlay", False))
    write_rule_csv(csv_path, times, rms, mouth_open)
    params_to_better_video(
        params,
        video_path,
        fps=fps,
        width=int(cfg["video"]["width"]),
        height=int(cfg["video"]["height"]),
        show_debug_overlay=show_debug_overlay,
    )

    logger.info("Input audio: %s", audio_path)
    logger.info(
        "Better visual frames=%d RMS range=%.6f..%.6f mouth range=%.3f..%.3f",
        len(rms),
        rms.min(),
        rms.max(),
        mouth_open.min(),
        mouth_open.max(),
    )
    print(f"CSV: {resolve_path(csv_path)}")
    print(f"Video: {resolve_path(video_path)}")


if __name__ == "__main__":
    main()

