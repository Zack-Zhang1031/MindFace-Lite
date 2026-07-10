from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.audio.features import compute_rms, frame_times, read_wav_mono, rms_to_mouth_open
from mindface.tts.simple_tts import write_text_wav
from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.csv_io import write_rule_csv
from mindface.visual.mouth_drawer import params_to_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TTS-like audio and drive the rule mouth demo.")
    parser.add_argument("--config", default="configs/tts_demo.yaml")
    parser.add_argument("--text", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    rule_cfg = load_yaml(cfg["rule_config"])
    text = args.text if args.text is not None else str(cfg["text"])
    write_text_wav(text, cfg["audio"]["output_path"], sample_rate=int(cfg["audio"]["sample_rate"]))

    sample_rate, waveform = read_wav_mono(cfg["audio"]["output_path"])
    fps = int(rule_cfg["audio"]["fps"])
    rms = compute_rms(waveform, sample_rate, fps=fps, frame_ms=float(rule_cfg["audio"]["frame_ms"]))
    mouth_open = rms_to_mouth_open(
        rms,
        noise_floor=float(rule_cfg["mouth"]["noise_floor"]),
        gamma=float(rule_cfg["mouth"]["gamma"]),
        smoothing=float(rule_cfg["mouth"]["smoothing"]),
    )
    times = frame_times(len(rms), fps)
    video_path = "outputs/videos/tts_rule_mouth_demo.mp4"
    csv_path = "outputs/logs/tts_rule_demo.csv"
    write_rule_csv(csv_path, times, rms, mouth_open)
    params_to_video(np.stack([mouth_open], axis=1), video_path, fps=fps)
    print(f"Audio: {resolve_path(cfg['audio']['output_path'])}")
    print(f"CSV: {resolve_path(csv_path)}")
    print(f"Video: {resolve_path(video_path)}")


if __name__ == "__main__":
    main()
