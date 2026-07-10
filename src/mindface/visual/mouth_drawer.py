from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mindface.utils.config import resolve_path


def draw_face_frame(
    mouth_open: float,
    mouth_width: float = 0.55,
    lip_round: float = 0.25,
    width: int = 512,
    height: int = 512,
) -> np.ndarray:
    """Draw a simple digital face controlled by mouth parameters."""
    mouth_open = float(np.clip(mouth_open, 0.0, 1.0))
    mouth_width = float(np.clip(mouth_width, 0.0, 1.0))
    lip_round = float(np.clip(lip_round, 0.0, 1.0))

    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cx, cy = width // 2, height // 2

    face_color = (225, 214, 196)
    line_color = (25, 25, 25)
    cheek_color = (190, 205, 225)
    mouth_color = (35, 35, 45)

    radius = int(min(width, height) * 0.36)
    cv2.circle(canvas, (cx, cy), radius, face_color, -1)
    cv2.circle(canvas, (cx, cy), radius, line_color, 3)

    eye_y = cy - int(radius * 0.32)
    eye_dx = int(radius * 0.36)
    cv2.circle(canvas, (cx - eye_dx, eye_y), 14, line_color, -1)
    cv2.circle(canvas, (cx + eye_dx, eye_y), 14, line_color, -1)
    cv2.circle(canvas, (cx - eye_dx - 45, cy + 10), 18, cheek_color, -1)
    cv2.circle(canvas, (cx + eye_dx + 45, cy + 10), 18, cheek_color, -1)

    mouth_center = (cx, cy + int(radius * 0.38))
    base_width = int(50 + mouth_width * 130 - lip_round * 35)
    open_height = int(8 + mouth_open * 95 + lip_round * 22)
    base_width = max(24, base_width)
    open_height = max(8, open_height)
    cv2.ellipse(canvas, mouth_center, (base_width // 2, open_height // 2), 0, 0, 360, mouth_color, -1)
    cv2.ellipse(canvas, mouth_center, (base_width // 2, open_height // 2), 0, 0, 360, line_color, 2)

    bar_w = int((width - 80) * mouth_open)
    cv2.rectangle(canvas, (40, height - 44), (width - 40, height - 24), (210, 210, 210), 1)
    cv2.rectangle(canvas, (40, height - 44), (40 + bar_w, height - 24), (90, 140, 220), -1)
    cv2.putText(
        canvas,
        f"open={mouth_open:.2f}",
        (40, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        line_color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def params_to_video(
    params: np.ndarray,
    output_path: str | Path,
    fps: int = 25,
    width: int = 512,
    height: int = 512,
) -> None:
    """Render mouth parameters to an MP4 video."""
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    params = np.asarray(params, dtype=np.float32)
    if params.ndim == 1:
        params = params[:, None]

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for row in params:
            mouth_open = float(row[0])
            mouth_width = float(row[1]) if row.shape[0] > 1 else 0.55
            lip_round = float(row[2]) if row.shape[0] > 2 else 0.25
            writer.write(draw_face_frame(mouth_open, mouth_width, lip_round, width, height))
    finally:
        writer.release()
