from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.audio.features import compute_rms, frame_times, read_wav_mono, rms_to_mouth_open
from mindface.tts.real_tts import check_tts_backend, generate_real_tts_wav
from mindface.utils.config import load_yaml
from mindface.utils.csv_io import write_rule_csv
from mindface.utils.logger import setup_logger
from mindface.visual.mouth_drawer import params_to_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real TTS audio and drive the rule mouth demo.")
    parser.add_argument("--config", default="configs/real_tts.yaml")
    parser.add_argument("--text", default=None)
    parser.add_argument("--engine", choices=["pyttsx3", "edge_tts"], default=None)
    parser.add_argument("--check-deps", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.engine is not None:
        cfg["engine"] = args.engine
    ok, message = check_tts_backend(str(cfg["engine"]))
    if args.check_deps:
        print(message)
        return
    if not ok:
        raise RuntimeError(message)

    logger = setup_logger("real_tts_mouth_demo", cfg["logging"]["log_path"])
    text = args.text if args.text is not None else str(cfg["text"])
    audio_path = generate_real_tts_wav(text, cfg)

    rule_cfg = load_yaml(cfg["rule_config"])
    sample_rate, waveform = read_wav_mono(audio_path)
    fps = int(rule_cfg["audio"]["fps"])
    rms = compute_rms(waveform, sample_rate, fps=fps, frame_ms=float(rule_cfg["audio"]["frame_ms"]))
    mouth_open = rms_to_mouth_open(
        rms,
        noise_floor=float(rule_cfg["mouth"]["noise_floor"]),
        gamma=float(rule_cfg["mouth"]["gamma"]),
        smoothing=float(rule_cfg["mouth"]["smoothing"]),
    )
    times = frame_times(len(rms), fps)
    write_rule_csv(cfg["outputs"]["csv_path"], times, rms, mouth_open)
    params_to_video(np.stack([mouth_open], axis=1), cfg["outputs"]["video_path"], fps=fps)
    logger.info("Real TTS mouth demo generated: audio=%s video=%s", audio_path, cfg["outputs"]["video_path"])
    print(f"Audio: {audio_path}")
    print(f"CSV: {cfg['outputs']['csv_path']}")
    print(f"Video: {cfg['outputs']['video_path']}")


if __name__ == "__main__":
    main()
