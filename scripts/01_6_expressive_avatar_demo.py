from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.audio.features import compute_rms, frame_times, read_wav_mono, rms_to_mouth_open
from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.logger import setup_logger
from mindface.visual.static_avatar_warper import (
    load_avatar_config,
    load_static_avatar,
    params_to_static_avatar_video,
    render_static_avatar_frame,
)


def _viseme_for_time(time_sec: float, events: list[dict]) -> str:
    for event in events:
        if float(event["start"]) <= time_sec < float(event["end"]):
            return str(event["shape"]).lower()
    return "neutral"


def _build_expressive_params(mouth_open: np.ndarray, times: np.ndarray, cfg: dict) -> tuple[np.ndarray, list[str]]:
    viseme_cfg = cfg.get("viseme", {})
    events = viseme_cfg.get("events", []) if viseme_cfg.get("enabled", False) else []

    params = np.zeros((len(mouth_open), 3), dtype=np.float32)
    labels: list[str] = []
    for idx, (base_open, time_sec) in enumerate(zip(mouth_open, times)):
        shape = _viseme_for_time(float(time_sec), events)
        open_value = float(base_open)
        mouth_width = float(np.clip(0.48 + 0.18 * open_value, 0.0, 1.0))
        lip_round = float(np.clip(0.18 + 0.20 * open_value, 0.0, 1.0))

        if shape == "closed":
            open_value = min(open_value * 0.08, 0.025)
            mouth_width = 0.48
            lip_round = 0.08
        elif shape == "a":
            open_value = max(open_value, 0.68)
            mouth_width = 0.58
            lip_round = 0.12
        elif shape == "i":
            open_value = min(1.0, max(open_value * 0.55, 0.16))
            mouth_width = 0.96
            lip_round = 0.04
        elif shape == "o":
            open_value = max(open_value, 0.48)
            mouth_width = 0.36
            lip_round = 0.92
        elif shape == "u":
            open_value = max(open_value, 0.34)
            mouth_width = 0.26
            lip_round = 1.0

        params[idx] = [np.clip(open_value, 0.0, 1.0), mouth_width, lip_round]
        labels.append(shape)
    return params, labels


def _write_expressive_csv(
    output_path: str | Path,
    times: np.ndarray,
    rms: np.ndarray,
    params: np.ndarray,
    viseme_labels: list[str],
) -> Path:
    resolved = resolve_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_index", "time_sec", "rms", "mouth_open", "mouth_width", "lip_round", "viseme"])
        for frame_index, (time_sec, rms_value, row, viseme) in enumerate(zip(times, rms, params, viseme_labels)):
            writer.writerow(
                [
                    frame_index,
                    f"{float(time_sec):.6f}",
                    f"{float(rms_value):.8f}",
                    f"{float(row[0]):.6f}",
                    f"{float(row[1]):.6f}",
                    f"{float(row[2]):.6f}",
                    viseme,
                ]
            )
    return resolved


def _save_preview_frame(cfg: dict, params: np.ndarray, avatar_cfg) -> Path:
    preview_cfg = cfg.get("preview", {})
    output_path = resolve_path(preview_cfg.get("output_path", "outputs/videos/expressive_avatar_preview.png"))
    frame_index = int(preview_cfg.get("frame_index", min(48, len(params) - 1)))
    frame_index = max(0, min(frame_index, len(params) - 1))
    base = load_static_avatar(avatar_cfg.image_path, avatar_cfg.width, avatar_cfg.height)
    row = params[frame_index]
    frame = render_static_avatar_frame(
        base,
        float(row[0]),
        float(row[1]) if row.shape[0] > 1 else 0.55,
        float(row[2]) if row.shape[0] > 2 else 0.25,
        config=avatar_cfg,
        frame_index=frame_index,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1.6 static-avatar OpenCV mouth-warp demo.")
    parser.add_argument("--config", default="configs/expressive_avatar_demo.yaml")
    parser.add_argument("--input-wav", default=None, help="Override audio.input_path from the config.")
    parser.add_argument("--output-video", default=None, help="Override video.output_path from the config.")
    parser.add_argument("--output-csv", default=None, help="Override csv.output_path from the config.")
    parser.add_argument("--debug-roi", action="store_true", help="Draw the configured mouth ROI rectangle.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("expressive_avatar_demo", cfg["logging"]["log_path"])
    avatar_cfg = load_avatar_config(cfg)
    if args.debug_roi:
        avatar_cfg = type(avatar_cfg)(
            image_path=avatar_cfg.image_path,
            width=avatar_cfg.width,
            height=avatar_cfg.height,
            mouth_roi=avatar_cfg.mouth_roi,
            head_bob_px=avatar_cfg.head_bob_px,
            cheek_pulse=avatar_cfg.cheek_pulse,
            blink_interval_frames=avatar_cfg.blink_interval_frames,
            blink_duration_frames=avatar_cfg.blink_duration_frames,
            show_roi_debug=True,
        )

    audio_path = resolve_path(args.input_wav if args.input_wav is not None else cfg["audio"]["input_path"])
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing input WAV: {audio_path}. Run scripts/00_generate_test_audio.py first.")

    sample_rate, waveform = read_wav_mono(audio_path)
    fps = int(cfg["audio"]["fps"])
    rms = compute_rms(waveform, sample_rate, fps=fps, frame_ms=float(cfg["audio"]["frame_ms"]))
    mouth_open = rms_to_mouth_open(
        rms,
        noise_floor=float(cfg["mouth"]["noise_floor"]),
        gamma=float(cfg["mouth"]["gamma"]),
        smoothing=float(cfg["mouth"]["smoothing"]),
    )
    times = frame_times(len(rms), fps)
    params, viseme_labels = _build_expressive_params(mouth_open, times, cfg)

    csv_path = args.output_csv if args.output_csv is not None else cfg["csv"]["output_path"]
    video_path = args.output_video if args.output_video is not None else cfg["video"]["output_path"]
    csv_resolved = _write_expressive_csv(csv_path, times, rms, params, viseme_labels)
    params_to_static_avatar_video(params, video_path, avatar_cfg, fps=fps)
    preview_path = _save_preview_frame(cfg, params, avatar_cfg)

    logger.info("Avatar image: %s", avatar_cfg.image_path)
    logger.info("Input audio: %s", audio_path)
    logger.info(
        "Expressive avatar frames=%d RMS range=%.6f..%.6f mouth range=%.3f..%.3f",
        len(rms),
        rms.min(),
        rms.max(),
        mouth_open.min(),
        mouth_open.max(),
    )
    print(f"Avatar: {avatar_cfg.image_path}")
    print(f"CSV: {csv_resolved}")
    print(f"Video: {resolve_path(video_path)}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
