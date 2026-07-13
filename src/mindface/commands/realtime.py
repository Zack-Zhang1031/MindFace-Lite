from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.csv_io import write_rule_csv
from mindface.utils.logger import setup_logger


def _mouth_from_audio(audio_path: str, rule_cfg: dict) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    from mindface.audio.features import compute_rms, frame_times, read_wav_mono, rms_to_mouth_open

    sample_rate, waveform = read_wav_mono(audio_path)
    fps = int(rule_cfg["audio"]["fps"])
    rms = compute_rms(
        waveform,
        sample_rate,
        fps=fps,
        frame_ms=float(rule_cfg["audio"]["frame_ms"]),
    )
    mouth_open = rms_to_mouth_open(
        rms,
        noise_floor=float(rule_cfg["mouth"]["noise_floor"]),
        gamma=float(rule_cfg["mouth"]["gamma"]),
        smoothing=float(rule_cfg["mouth"]["smoothing"]),
    )
    return fps, frame_times(len(rms), fps), rms, mouth_open


def run_queue(args: argparse.Namespace) -> int:
    from mindface.realtime.pipeline import run_rule_queue_pipeline

    cfg = load_yaml(args.config)
    logger = setup_logger("realtime_rule", cfg["logging"]["log_path"])
    rows, stats = run_rule_queue_pipeline(
        str(resolve_path(cfg["audio"]["input_path"])),
        fps=int(cfg["audio"]["fps"]),
        frame_ms=float(cfg["audio"]["frame_ms"]),
        realtime_sleep=bool(cfg["realtime"]["sleep_like_realtime"]),
    )
    csv_path = resolve_path(cfg["output"]["csv_path"])
    report_path = resolve_path(cfg["output"]["report_path"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["frame_index", "time_sec", "rms", "mouth_open", "latency_ms"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "frame_index": row.index,
                    "time_sec": f"{row.time_sec:.6f}",
                    "rms": f"{row.rms:.8f}",
                    "mouth_open": f"{row.mouth_open:.6f}",
                    "latency_ms": f"{row.latency_ms:.4f}",
                }
            )
    report_path.write_text(json.dumps({"frames": len(rows), "latency": stats}, indent=2), encoding="utf-8")
    logger.info("Realtime rule simulation frames=%d", len(rows))
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")
    return 0


def run_microphone(args: argparse.Namespace) -> int:
    from mindface.realtime.microphone import (
        check_sounddevice_available,
        format_audio_device_report,
        run_microphone_rule_stream,
    )

    if args.list_devices:
        print(format_audio_device_report())
        return 0
    ok, message = check_sounddevice_available()
    if args.check_deps:
        print(message)
        return 0
    if not ok:
        raise RuntimeError(message)
    cfg = load_yaml(args.config)
    logger = setup_logger("mic_stream_rule_demo", cfg["logging"]["log_path"])
    udp_enabled = bool(cfg["udp"].get("enabled", False))
    rows = run_microphone_rule_stream(
        sample_rate=int(cfg["audio"]["sample_rate"]),
        fps=int(cfg["audio"]["fps"]),
        duration_sec=float(args.duration_sec if args.duration_sec is not None else cfg["audio"]["duration_sec"]),
        channels=int(cfg["audio"].get("channels", 1)),
        device=cfg["audio"].get("device"),
        noise_floor=float(cfg["mouth"]["noise_floor"]),
        scale=float(cfg["mouth"]["scale"]),
        gamma=float(cfg["mouth"]["gamma"]),
        smoothing=float(cfg["mouth"]["smoothing"]),
        csv_path=cfg["output"]["csv_path"],
        show_window=bool(args.show or cfg["output"].get("show_window", False)),
        udp_host=str(cfg["udp"]["host"]) if udp_enabled else None,
        udp_port=int(cfg["udp"]["port"]) if udp_enabled else None,
        logger=logger,
    )
    print(f"Mic frames: {len(rows)}")
    print(f"CSV: {cfg['output']['csv_path']}")
    return 0


def run_pseudo_tts_generate(args: argparse.Namespace) -> int:
    from mindface.tts.simple_tts import write_text_wav

    cfg = load_yaml(args.config)
    text = args.text if args.text is not None else str(cfg["text"])
    write_text_wav(text, cfg["audio"]["output_path"], sample_rate=int(cfg["audio"]["sample_rate"]))
    print(f"TTS-like WAV: {resolve_path(cfg['audio']['output_path'])}")
    return 0


def run_pseudo_tts_demo(args: argparse.Namespace) -> int:
    from mindface.tts.simple_tts import write_text_wav
    from mindface.visual.mouth_drawer import params_to_video

    cfg = load_yaml(args.config)
    rule_cfg = load_yaml(cfg["rule_config"])
    text = args.text if args.text is not None else str(cfg["text"])
    write_text_wav(text, cfg["audio"]["output_path"], sample_rate=int(cfg["audio"]["sample_rate"]))
    fps, times, rms, mouth_open = _mouth_from_audio(cfg["audio"]["output_path"], rule_cfg)
    video_path = "outputs/videos/tts_rule_mouth_demo.mp4"
    csv_path = "outputs/logs/tts_rule_demo.csv"
    write_rule_csv(csv_path, times, rms, mouth_open)
    params_to_video(np.stack([mouth_open], axis=1), video_path, fps=fps)
    print(f"Audio: {resolve_path(cfg['audio']['output_path'])}")
    print(f"CSV: {resolve_path(csv_path)}")
    print(f"Video: {resolve_path(video_path)}")
    return 0


def _real_tts_config(args: argparse.Namespace) -> dict:
    from mindface.tts.real_tts import check_tts_backend

    cfg = load_yaml(args.config)
    if args.engine is not None:
        cfg["engine"] = args.engine
    ok, message = check_tts_backend(str(cfg["engine"]))
    if args.check_deps:
        print(message)
        return {"_dependency_check_only": True}
    if not ok:
        raise RuntimeError(message)
    return cfg


def run_real_tts_generate(args: argparse.Namespace) -> int:
    from mindface.tts.real_tts import generate_real_tts_wav

    cfg = _real_tts_config(args)
    if cfg.get("_dependency_check_only"):
        return 0
    logger = setup_logger("real_tts", cfg["logging"]["log_path"])
    text = args.text if args.text is not None else str(cfg["text"])
    output_path = generate_real_tts_wav(text, cfg)
    logger.info("Generated real TTS audio with engine=%s: %s", cfg["engine"], output_path)
    print(f"Real TTS WAV: {output_path}")
    return 0


def run_real_tts_demo(args: argparse.Namespace) -> int:
    from mindface.tts.real_tts import generate_real_tts_wav
    from mindface.visual.mouth_drawer import params_to_video

    cfg = _real_tts_config(args)
    if cfg.get("_dependency_check_only"):
        return 0
    logger = setup_logger("real_tts_mouth_demo", cfg["logging"]["log_path"])
    text = args.text if args.text is not None else str(cfg["text"])
    audio_path = generate_real_tts_wav(text, cfg)
    fps, times, rms, mouth_open = _mouth_from_audio(str(audio_path), load_yaml(cfg["rule_config"]))
    write_rule_csv(cfg["outputs"]["csv_path"], times, rms, mouth_open)
    params_to_video(np.stack([mouth_open], axis=1), cfg["outputs"]["video_path"], fps=fps)
    logger.info("Real TTS mouth demo generated: audio=%s video=%s", audio_path, cfg["outputs"]["video_path"])
    print(f"Audio: {audio_path}")
    print(f"CSV: {cfg['outputs']['csv_path']}")
    print(f"Video: {cfg['outputs']['video_path']}")
    return 0
