from __future__ import annotations

import argparse

import numpy as np

from mindface.audio.features import compute_rms, frame_times, read_wav_mono, rms_to_mouth_open, write_wav_mono
from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.csv_io import write_rule_csv
from mindface.utils.logger import setup_logger


def build_test_voice(sample_rate: int, duration_sec: float) -> np.ndarray:
    """Generate deterministic speech-like audio with changing energy."""
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
    for start, end, frequency, amplitude in syllables:
        mask = (t >= start) & (t < end)
        local_t = t[mask] - start
        envelope = np.clip(
            np.sin(np.linspace(0, np.pi, max(1, int(mask.sum())), dtype=np.float32)),
            0.0,
            1.0,
        ) ** 0.8
        voiced = np.sin(2 * np.pi * frequency * local_t) + 0.35 * np.sin(
            2 * np.pi * frequency * 2.0 * local_t
        )
        waveform[mask] += amplitude * envelope * voiced
    waveform += 0.006 * np.random.default_rng(7).normal(size=waveform.shape).astype(np.float32)
    return np.clip(waveform, -0.95, 0.95).astype(np.float32)


def _rule_features(cfg: dict, audio_path: str) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    sample_rate, waveform = read_wav_mono(audio_path)
    fps = int(cfg["audio"]["fps"])
    rms = compute_rms(waveform, sample_rate, fps=fps, frame_ms=float(cfg["audio"]["frame_ms"]))
    mouth_open = rms_to_mouth_open(
        rms,
        noise_floor=float(cfg["mouth"]["noise_floor"]),
        gamma=float(cfg["mouth"]["gamma"]),
        smoothing=float(cfg["mouth"]["smoothing"]),
    )
    return fps, frame_times(len(rms), fps), rms, mouth_open


def run_generate_audio(args: argparse.Namespace) -> int:
    logger = setup_logger("generate_test_audio", "outputs/logs/generate_test_audio.log")
    waveform = build_test_voice(args.sample_rate, args.duration_sec)
    write_wav_mono(args.output, args.sample_rate, waveform)
    output_path = resolve_path(args.output)
    logger.info("Generated test audio: %s", output_path)
    print(f"Generated: {output_path}")
    return 0


def run_rule_demo(args: argparse.Namespace) -> int:
    from mindface.visual.mouth_drawer import params_to_video

    cfg = load_yaml(args.config)
    logger = setup_logger("rule_mouth_demo", cfg["logging"]["log_path"])
    audio_path = resolve_path(cfg["audio"]["input_path"])
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing input WAV: {audio_path}. Run 'mindface demo generate-audio' first.")
    fps, times, rms, mouth_open = _rule_features(cfg, str(audio_path))
    write_rule_csv(cfg["csv"]["output_path"], times, rms, mouth_open)
    params_to_video(
        np.stack([mouth_open], axis=1),
        cfg["video"]["output_path"],
        fps=fps,
        width=int(cfg["video"]["width"]),
        height=int(cfg["video"]["height"]),
    )
    logger.info(
        "Frames=%d RMS range=%.6f..%.6f mouth range=%.3f..%.3f",
        len(rms),
        rms.min(),
        rms.max(),
        mouth_open.min(),
        mouth_open.max(),
    )
    print(f"CSV: {resolve_path(cfg['csv']['output_path'])}")
    print(f"Video: {resolve_path(cfg['video']['output_path'])}")
    return 0


def run_better_visual_demo(args: argparse.Namespace) -> int:
    from mindface.visual.better_mouth_drawer import params_to_better_video

    cfg = load_yaml(args.config)
    logger = setup_logger("better_visual_demo", cfg["logging"]["log_path"])
    audio_path = resolve_path(args.input_wav or cfg["audio"]["input_path"])
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing input WAV: {audio_path}. Run 'mindface demo generate-audio' first.")
    fps, times, rms, mouth_open = _rule_features(cfg, str(audio_path))
    csv_path = args.output_csv or cfg["csv"]["output_path"]
    video_path = args.output_video or cfg["video"]["output_path"]
    write_rule_csv(csv_path, times, rms, mouth_open)
    params_to_better_video(
        np.stack([mouth_open], axis=1),
        video_path,
        fps=fps,
        width=int(cfg["video"]["width"]),
        height=int(cfg["video"]["height"]),
        show_debug_overlay=bool(args.debug_overlay or cfg["video"].get("show_debug_overlay", False)),
    )
    logger.info("Better visual frames=%d", len(rms))
    print(f"CSV: {resolve_path(csv_path)}")
    print(f"Video: {resolve_path(video_path)}")
    return 0


def run_expressive_avatar_demo(args: argparse.Namespace) -> int:
    from mindface.visual.expressive_avatar import run_expressive_avatar_demo as run_demo

    result = run_demo(
        load_yaml(args.config),
        input_wav=args.input_wav,
        output_video=args.output_video,
        output_csv=args.output_csv,
        debug_roi=args.debug_roi,
    )
    print(f"Avatar: {result.avatar_path}")
    print(f"CSV: {result.csv_path}")
    print(f"Video: {result.video_path}")
    print(f"Preview: {result.preview_path}")
    return 0
