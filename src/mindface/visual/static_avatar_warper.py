from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mindface.utils.config import resolve_path


Color = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class MouthRoi:
    center_x: float = 0.5
    center_y: float = 0.60
    width: float = 0.30
    height: float = 0.16
    max_open_ratio: float = 0.105


@dataclass(frozen=True, slots=True)
class ExpressiveAvatarConfig:
    image_path: Path
    width: int = 768
    height: int = 768
    mouth_roi: MouthRoi = MouthRoi()
    head_bob_px: float = 5.0
    cheek_pulse: bool = True
    show_roi_debug: bool = False


def load_avatar_config(cfg: dict) -> ExpressiveAvatarConfig:
    asset_cfg = cfg["asset"]
    video_cfg = cfg["video"]
    roi_cfg = cfg["mouth_roi"]
    expressive_cfg = cfg.get("expressive", {})
    return ExpressiveAvatarConfig(
        image_path=resolve_path(asset_cfg["image_path"]),
        width=int(video_cfg["width"]),
        height=int(video_cfg["height"]),
        mouth_roi=MouthRoi(
            center_x=float(roi_cfg["center_x"]),
            center_y=float(roi_cfg["center_y"]),
            width=float(roi_cfg["width"]),
            height=float(roi_cfg["height"]),
            max_open_ratio=float(roi_cfg["max_open_ratio"]),
        ),
        head_bob_px=float(expressive_cfg.get("head_bob_px", 5.0)),
        cheek_pulse=bool(expressive_cfg.get("cheek_pulse", True)),
        show_roi_debug=bool(video_cfg.get("show_roi_debug", False)),
    )


def load_static_avatar(image_path: str | Path, width: int, height: int) -> np.ndarray:
    resolved = resolve_path(image_path)
    image = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read avatar image: {resolved}")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _roi_bounds(frame_shape: tuple[int, int, int], roi: MouthRoi) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    roi_w = int(width * roi.width)
    roi_h = int(height * roi.height)
    cx = int(width * roi.center_x)
    cy = int(height * roi.center_y)
    x0 = max(0, cx - roi_w // 2)
    y0 = max(0, cy - roi_h // 2)
    x1 = min(width, cx + roi_w // 2)
    y1 = min(height, cy + roi_h // 2)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid mouth ROI. Check mouth_roi in the config.")
    return x0, y0, x1, y1


def _soft_mask(shape: tuple[int, int], power: float = 2.2) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w / 2.0) / max(w / 2.0, 1.0)
    ny = (yy - h / 2.0) / max(h / 2.0, 1.0)
    radius = nx * nx + ny * ny
    mask = np.clip(1.0 - radius, 0.0, 1.0) ** power
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, w * 0.025))


def _warp_mouth_roi(roi_image: np.ndarray, mouth_open: float, mouth_width: float, lip_round: float) -> np.ndarray:
    h, w = roi_image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    center_y = h * 0.52
    center_x = w * 0.50
    nx = (xx - center_x) / max(w * 0.5, 1.0)
    ny = (yy - center_y) / max(h * 0.5, 1.0)

    ellipse = np.exp(-((nx / 0.82) ** 2 + (ny / 0.72) ** 2) * 2.2).astype(np.float32)
    vertical_delta = mouth_open * (h * (0.10 + 0.34 * lip_round))
    horizontal_gain = mouth_open * (0.04 + 0.10 * mouth_width)
    direction = np.where(yy < center_y, 1.0, -1.0).astype(np.float32)

    map_x = xx - (xx - center_x) * horizontal_gain * ellipse
    map_y = yy + direction * vertical_delta * ellipse
    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)
    return cv2.remap(roi_image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _blend_roi(frame: np.ndarray, warped_roi: np.ndarray, bounds: tuple[int, int, int, int], alpha_scale: float) -> None:
    x0, y0, x1, y1 = bounds
    mask = _soft_mask((y1 - y0, x1 - x0))[:, :, None]
    alpha = np.clip(mask * alpha_scale, 0.0, 1.0)
    original = frame[y0:y1, x0:x1].astype(np.float32)
    blended = original * (1.0 - alpha) + warped_roi.astype(np.float32) * alpha
    frame[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)


def _draw_transparent_ellipse(
    frame: np.ndarray,
    center: tuple[int, int],
    axes: tuple[int, int],
    color: Color,
    alpha: float,
    start_angle: int = 0,
    end_angle: int = 360,
) -> None:
    overlay = frame.copy()
    cv2.ellipse(overlay, center, axes, 0, start_angle, end_angle, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)


def _draw_open_mouth_details(
    frame: np.ndarray,
    roi: MouthRoi,
    mouth_open: float,
    mouth_width: float,
    lip_round: float,
) -> None:
    if mouth_open < 0.035:
        return
    h, w = frame.shape[:2]
    cx = int(w * roi.center_x)
    cy = int(h * roi.center_y)
    open_px = int(h * roi.max_open_ratio * mouth_open)
    half_width = int(w * roi.width * (0.20 + 0.26 * mouth_width - 0.10 * lip_round))
    inner_h = max(3, int(open_px * (0.38 + 0.52 * mouth_open)))
    inner_w = max(22, half_width)

    lip_color = (72, 82, 168)
    lip_shadow = (48, 40, 72)
    mouth_dark = (16, 13, 20)
    teeth_color = (238, 240, 235)
    tongue_color = (84, 80, 190)

    _draw_transparent_ellipse(frame, (cx, cy + inner_h // 5), (inner_w + 8, inner_h + 6), lip_shadow, 0.22)
    _draw_transparent_ellipse(frame, (cx, cy), (inner_w + 5, inner_h + 4), lip_color, 0.42)
    _draw_transparent_ellipse(frame, (cx, cy + 1), (inner_w, inner_h), mouth_dark, 0.86)

    if mouth_open > 0.25:
        teeth_h = max(3, int(inner_h * 0.26))
        _draw_transparent_ellipse(
            frame,
            (cx, cy - int(inner_h * 0.38)),
            (max(10, inner_w - 15), teeth_h),
            teeth_color,
            0.95,
            start_angle=0,
            end_angle=180,
        )
    if mouth_open > 0.52:
        _draw_transparent_ellipse(
            frame,
            (cx, cy + int(inner_h * 0.43)),
            (max(12, inner_w - 18), max(3, int(inner_h * 0.28))),
            tongue_color,
            0.76,
            start_angle=180,
            end_angle=360,
        )


def _apply_head_bob(frame: np.ndarray, mouth_open: float, head_bob_px: float) -> np.ndarray:
    if abs(head_bob_px) < 0.1:
        return frame
    shift_y = -float(head_bob_px) * mouth_open
    matrix = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, shift_y]], dtype=np.float32)
    return cv2.warpAffine(
        frame,
        matrix,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _draw_cheek_pulse(frame: np.ndarray, roi: MouthRoi, mouth_open: float) -> None:
    if mouth_open < 0.08:
        return
    h, w = frame.shape[:2]
    cx = int(w * roi.center_x)
    cy = int(h * (roi.center_y - 0.065))
    dx = int(w * 0.115)
    axes = (int(w * 0.035), int(h * 0.018))
    alpha = min(0.30, 0.08 + 0.18 * mouth_open)
    _draw_transparent_ellipse(frame, (cx - dx, cy), axes, (154, 142, 230), alpha)
    _draw_transparent_ellipse(frame, (cx + dx, cy), axes, (154, 142, 230), alpha)


def render_static_avatar_frame(
    base_image: np.ndarray,
    mouth_open: float,
    mouth_width: float = 0.55,
    lip_round: float = 0.25,
    config: ExpressiveAvatarConfig | None = None,
) -> np.ndarray:
    """Render one frame by locally warping the static avatar mouth region."""
    if config is None:
        config = ExpressiveAvatarConfig(image_path=Path(""))
    mouth_open = _clip01(mouth_open)
    mouth_width = _clip01(mouth_width)
    lip_round = _clip01(lip_round)

    frame = base_image.copy()
    frame = _apply_head_bob(frame, mouth_open, config.head_bob_px)
    bounds = _roi_bounds(frame.shape, config.mouth_roi)
    x0, y0, x1, y1 = bounds
    roi_image = frame[y0:y1, x0:x1].copy()
    warped_roi = _warp_mouth_roi(roi_image, mouth_open, mouth_width, lip_round)
    _blend_roi(frame, warped_roi, bounds, alpha_scale=0.25 + 0.72 * mouth_open)
    _draw_open_mouth_details(frame, config.mouth_roi, mouth_open, mouth_width, lip_round)
    if config.cheek_pulse:
        _draw_cheek_pulse(frame, config.mouth_roi, mouth_open)
    if config.show_roi_debug:
        cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 220, 60), 2)
    return frame


def params_to_static_avatar_video(
    params: np.ndarray,
    output_path: str | Path,
    config: ExpressiveAvatarConfig,
    fps: int = 25,
) -> None:
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_image = load_static_avatar(config.image_path, config.width, config.height)
    params = np.asarray(params, dtype=np.float32)
    if params.ndim == 1:
        params = params[:, None]

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (config.width, config.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    try:
        for row in params:
            mouth_open = float(row[0])
            mouth_width = float(row[1]) if row.shape[0] > 1 else 0.55
            lip_round = float(row[2]) if row.shape[0] > 2 else 0.25
            writer.write(render_static_avatar_frame(base_image, mouth_open, mouth_width, lip_round, config))
    finally:
        writer.release()
