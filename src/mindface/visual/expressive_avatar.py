from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mindface.audio.features import compute_rms, frame_times, read_wav_mono, rms_to_mouth_open
from mindface.utils.config import resolve_path
from mindface.utils.logger import setup_logger
from mindface.visual.static_avatar_warper import (
    ExpressiveAvatarConfig,
    load_avatar_config,
    load_static_avatar,
    params_to_static_avatar_video,
    render_static_avatar_frame,
)


@dataclass(frozen=True, slots=True)
class ExpressiveAvatarResult:
    avatar_path: Path
    csv_path: Path
    video_path: Path
    preview_path: Path
    frame_count: int


def _viseme_for_time(time_sec: float, events: list[dict[str, Any]]) -> str:
    for event in events:
        if float(event["start"]) <= time_sec < float(event["end"]):
            return str(event["shape"]).lower()
    return "neutral"


def build_expressive_params(
    mouth_open: np.ndarray,
    times: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
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


def _save_preview_frame(
    cfg: dict[str, Any],
    params: np.ndarray,
    avatar_cfg: ExpressiveAvatarConfig,
) -> Path:
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
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to write avatar preview: {output_path}")
    return output_path


def run_expressive_avatar_demo(
    cfg: dict[str, Any],
    input_wav: str | None = None,
    output_video: str | None = None,
    output_csv: str | None = None,
    debug_roi: bool = False,
) -> ExpressiveAvatarResult:
    logger = setup_logger("expressive_avatar_demo", cfg["logging"]["log_path"])
    avatar_cfg = load_avatar_config(cfg)
    if debug_roi:
        avatar_cfg = replace(avatar_cfg, show_roi_debug=True)

    audio_path = resolve_path(input_wav if input_wav is not None else cfg["audio"]["input_path"])
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing input WAV: {audio_path}. Run 'mindface demo generate-audio' first.")

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
    params, viseme_labels = build_expressive_params(mouth_open, times, cfg)

    csv_path = output_csv if output_csv is not None else cfg["csv"]["output_path"]
    video_path = output_video if output_video is not None else cfg["video"]["output_path"]
    csv_resolved = _write_expressive_csv(csv_path, times, rms, params, viseme_labels)
    video_resolved = resolve_path(video_path)
    params_to_static_avatar_video(params, video_resolved, avatar_cfg, fps=fps)
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
    return ExpressiveAvatarResult(
        avatar_path=avatar_cfg.image_path,
        csv_path=csv_resolved,
        video_path=video_resolved,
        preview_path=preview_path,
        frame_count=len(rms),
    )
