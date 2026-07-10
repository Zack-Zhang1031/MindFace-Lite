from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mindface.utils.config import resolve_path


Color = tuple[int, int, int]


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _vertical_gradient(width: int, height: int, top: Color, bottom: Color) -> np.ndarray:
    top_arr = np.asarray(top, dtype=np.float32)
    bottom_arr = np.asarray(bottom, dtype=np.float32)
    weights = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    row = top_arr * (1.0 - weights) + bottom_arr * weights
    return np.repeat(row, width, axis=1).astype(np.uint8)


def _blend_color(canvas: np.ndarray, color: Color, mask: np.ndarray, alpha: float = 1.0) -> None:
    mask_3c = np.clip(mask.astype(np.float32) * float(alpha), 0.0, 1.0)[:, :, None]
    color_arr = np.asarray(color, dtype=np.float32)
    blended = canvas.astype(np.float32) * (1.0 - mask_3c) + color_arr * mask_3c
    canvas[:] = np.clip(blended, 0, 255).astype(np.uint8)


def _soft_ellipse_mask(
    shape: tuple[int, int],
    center: tuple[int, int],
    axes: tuple[int, int],
    blur: int = 0,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.float32)
    axes = (max(1, int(axes[0])), max(1, int(axes[1])))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    if blur > 0:
        kernel = blur if blur % 2 == 1 else blur + 1
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    return np.clip(mask, 0.0, 1.0)


def _draw_soft_ellipse(
    canvas: np.ndarray,
    center: tuple[int, int],
    axes: tuple[int, int],
    color: Color,
    alpha: float = 1.0,
    blur: int = 0,
) -> None:
    mask = _soft_ellipse_mask(canvas.shape[:2], center, axes, blur=blur)
    _blend_color(canvas, color, mask, alpha=alpha)


def _draw_eye(canvas: np.ndarray, center: tuple[int, int], scale: float = 1.0) -> None:
    cx, cy = center
    eye_axes = (int(28 * scale), int(15 * scale))
    cv2.ellipse(canvas, (cx, cy), eye_axes, 0, 0, 360, (248, 248, 250), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), eye_axes, 0, 0, 360, (74, 62, 54), 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy + 1), int(9 * scale), (126, 86, 54), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy + 1), int(5 * scale), (32, 26, 22), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx - int(4 * scale), cy - int(4 * scale)), int(2 * scale), (255, 255, 255), -1, cv2.LINE_AA)


def _draw_better_mouth(
    canvas: np.ndarray,
    center: tuple[int, int],
    mouth_open: float,
    mouth_width: float,
    lip_round: float,
) -> None:
    height, width = canvas.shape[:2]
    cx, cy = center
    half_width = int(width * (0.070 + 0.095 * mouth_width - 0.020 * lip_round))
    inner_height = int(height * (0.012 + 0.095 * mouth_open + 0.018 * lip_round))
    half_width = max(32, half_width)
    inner_height = max(6, inner_height)

    lip_shadow = (64, 58, 96)
    lip_dark = (92, 70, 154)
    lip_main = (112, 86, 190)
    lip_light = (148, 117, 218)
    mouth_dark = (24, 19, 28)
    teeth = (236, 238, 232)
    tongue = (91, 74, 190)

    _draw_soft_ellipse(canvas, (cx, cy + 4), (half_width + 16, inner_height + 16), lip_shadow, alpha=0.22, blur=17)
    cv2.ellipse(canvas, (cx, cy), (half_width + 12, inner_height + 11), 0, 0, 360, lip_dark, -1, cv2.LINE_AA)

    left = (cx - half_width - 6, cy)
    right = (cx + half_width + 6, cy)
    upper = np.asarray(
        [
            left,
            (cx - half_width // 2, cy - int(12 + lip_round * 12)),
            (cx, cy - int(6 + lip_round * 18)),
            (cx + half_width // 2, cy - int(12 + lip_round * 12)),
            right,
            (cx + half_width // 3, cy + max(2, inner_height // 5)),
            (cx, cy + max(3, inner_height // 7)),
            (cx - half_width // 3, cy + max(2, inner_height // 5)),
        ],
        dtype=np.int32,
    )
    lower = np.asarray(
        [
            left,
            (cx - half_width // 2, cy + int(9 + inner_height * 0.36)),
            (cx, cy + int(12 + inner_height * 0.48)),
            (cx + half_width // 2, cy + int(9 + inner_height * 0.36)),
            right,
            (cx + half_width // 3, cy + max(2, inner_height // 5)),
            (cx, cy + max(3, inner_height // 7)),
            (cx - half_width // 3, cy + max(2, inner_height // 5)),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [upper], lip_main, cv2.LINE_AA)
    cv2.fillPoly(canvas, [lower], lip_dark, cv2.LINE_AA)

    cv2.ellipse(canvas, (cx, cy + 1), (half_width - 8, inner_height), 0, 0, 360, mouth_dark, -1, cv2.LINE_AA)
    if mouth_open > 0.22:
        teeth_h = max(4, int(inner_height * 0.28))
        cv2.ellipse(
            canvas,
            (cx, cy - int(inner_height * 0.33)),
            (max(12, half_width - 22), teeth_h),
            0,
            0,
            180,
            teeth,
            -1,
            cv2.LINE_AA,
        )
    if mouth_open > 0.46:
        cv2.ellipse(
            canvas,
            (cx, cy + int(inner_height * 0.38)),
            (max(12, half_width - 26), max(5, int(inner_height * 0.26))),
            0,
            180,
            360,
            tongue,
            -1,
            cv2.LINE_AA,
        )

    cv2.ellipse(canvas, (cx, cy), (half_width + 12, inner_height + 11), 0, 0, 360, (58, 43, 80), 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx - half_width // 4, cy - inner_height // 2), (half_width // 5, 3), 0, 0, 360, lip_light, -1, cv2.LINE_AA)


def _draw_debug_overlay(canvas: np.ndarray, mouth_open: float) -> None:
    height, width = canvas.shape[:2]
    bar_width = int((width - 80) * mouth_open)
    cv2.rectangle(canvas, (40, height - 42), (width - 40, height - 24), (210, 210, 210), 1)
    cv2.rectangle(canvas, (40, height - 42), (40 + bar_width, height - 24), (120, 160, 230), -1)
    cv2.putText(
        canvas,
        f"mouth_open={mouth_open:.2f}",
        (40, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (48, 48, 48),
        2,
        cv2.LINE_AA,
    )


def draw_better_face_frame(
    mouth_open: float,
    mouth_width: float = 0.55,
    lip_round: float = 0.25,
    width: int = 768,
    height: int = 768,
    show_debug_overlay: bool = False,
) -> np.ndarray:
    """Draw a more polished 2D avatar face controlled by mouth parameters."""
    mouth_open = _clip01(mouth_open)
    mouth_width = _clip01(mouth_width)
    lip_round = _clip01(lip_round)

    canvas = _vertical_gradient(width, height, top=(236, 238, 232), bottom=(226, 232, 238))
    cx, cy = width // 2, height // 2
    face_w = int(width * 0.47)
    face_h = int(height * 0.60)
    face_center = (cx, cy + int(height * 0.005))

    skin_shadow = (154, 178, 203)
    skin = (176, 204, 229)
    skin_light = (205, 225, 244)
    skin_warm = (158, 186, 216)
    hair = (47, 42, 38)
    hair_light = (75, 66, 58)

    cv2.rectangle(canvas, (cx - int(width * 0.07), cy + int(height * 0.28)), (cx + int(width * 0.07), height), skin_warm, -1)
    _draw_soft_ellipse(canvas, (cx, cy + int(height * 0.54)), (int(width * 0.24), int(height * 0.09)), (180, 188, 198), alpha=0.25, blur=35)

    _draw_soft_ellipse(canvas, (cx, cy + 18), (face_w // 2 + 24, face_h // 2 + 26), skin_shadow, alpha=0.33, blur=25)
    _draw_soft_ellipse(canvas, (cx - face_w // 2, cy + 5), (int(width * 0.045), int(height * 0.085)), skin, alpha=1.0, blur=0)
    _draw_soft_ellipse(canvas, (cx + face_w // 2, cy + 5), (int(width * 0.045), int(height * 0.085)), skin, alpha=1.0, blur=0)
    _draw_soft_ellipse(canvas, face_center, (face_w // 2, face_h // 2), skin, alpha=1.0, blur=0)
    _draw_soft_ellipse(canvas, (cx - face_w // 6, cy - face_h // 6), (face_w // 3, face_h // 4), skin_light, alpha=0.23, blur=45)

    cv2.ellipse(canvas, (cx, cy - face_h // 2 + 30), (face_w // 2 + 26, int(height * 0.13)), 0, 175, 365, hair, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx - int(width * 0.12), cy - face_h // 2 + 78), (int(width * 0.13), int(height * 0.13)), -20, 190, 350, hair, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx + int(width * 0.14), cy - face_h // 2 + 72), (int(width * 0.12), int(height * 0.12)), 24, 190, 350, hair, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx - int(width * 0.06), cy - face_h // 2 + 44), (int(width * 0.12), int(height * 0.055)), -14, 180, 360, hair_light, 3, cv2.LINE_AA)

    eye_y = cy - int(height * 0.085)
    eye_dx = int(width * 0.115)
    cv2.ellipse(canvas, (cx - eye_dx, eye_y - 34), (40, 8), -8, 180, 360, (59, 51, 44), 4, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx + eye_dx, eye_y - 34), (40, 8), 8, 180, 360, (59, 51, 44), 4, cv2.LINE_AA)
    _draw_eye(canvas, (cx - eye_dx, eye_y), scale=1.0)
    _draw_eye(canvas, (cx + eye_dx, eye_y), scale=1.0)

    nose_top = (cx, eye_y + int(height * 0.035))
    nose_tip = (cx - 2, eye_y + int(height * 0.120))
    cv2.line(canvas, nose_top, nose_tip, (168, 151, 141), 3, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx + 10, nose_tip[1] + 2), (17, 8), 0, 0, 180, (158, 143, 133), 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx - int(width * 0.16), cy + int(height * 0.08)), (32, 18), 0, 0, 360, (184, 164, 222), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx + int(width * 0.16), cy + int(height * 0.08)), (32, 18), 0, 0, 360, (184, 164, 222), -1, cv2.LINE_AA)

    mouth_center = (cx, cy + int(height * 0.195))
    _draw_better_mouth(canvas, mouth_center, mouth_open, mouth_width, lip_round)

    if show_debug_overlay:
        _draw_debug_overlay(canvas, mouth_open)
    return canvas


def params_to_better_video(
    params: np.ndarray,
    output_path: str | Path,
    fps: int = 25,
    width: int = 768,
    height: int = 768,
    show_debug_overlay: bool = False,
) -> None:
    """Render mouth parameters to a polished MP4 avatar video."""
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
            frame = draw_better_face_frame(
                mouth_open,
                mouth_width,
                lip_round,
                width=width,
                height=height,
                show_debug_overlay=show_debug_overlay,
            )
            writer.write(frame)
    finally:
        writer.release()
