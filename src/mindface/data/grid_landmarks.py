from __future__ import annotations

import csv
import ctypes
from dataclasses import dataclass
from pathlib import Path
import platform
import urllib.request

import cv2
import numpy as np

from mindface.data.grid_quality import build_grid_landmark_quality_report
from mindface.utils.config import resolve_path


VIDEO_EXTS = {".mpg", ".mpeg", ".mp4", ".avi", ".mov"}
DEFAULT_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


@dataclass(frozen=True)
class LandmarkConfig:
    video_dir: Path
    output_dir: Path
    max_videos: int | None
    split_ratios: tuple[float, float, float]
    seed: int
    refine_landmarks: bool
    min_detection_confidence: float
    min_tracking_confidence: float
    model_path: Path
    model_url: str
    auto_download_model: bool
    delegate: str
    quality_report_path: Path | None = None
    quality_min_detection_rate: float = 0.95


def check_mediapipe_available() -> tuple[bool, str]:
    try:
        import mediapipe  # noqa: F401
        from mediapipe.tasks import python as mp_python  # noqa: F401
        from mediapipe.tasks.python import vision  # noqa: F401
    except ImportError as exc:
        return (
            False,
            "MediaPipe FaceLandmarker dependencies are unavailable. "
            f"Original error: {exc}. Install mediapipe in the Windows training env "
            "or a separate optional-feature WSL env. Do not install it into "
            "mindface-rknn because it can break the RKNN numpy dependency.",
        )
    if platform.system() == "Linux":
        missing_libs = []
        for lib_name in ("libGLESv2.so.2", "libEGL.so.1"):
            try:
                ctypes.CDLL(lib_name)
            except OSError:
                missing_libs.append(lib_name)
        if missing_libs:
            return (
                False,
                "MediaPipe is installed, but Linux shared libraries are missing: "
                f"{', '.join(missing_libs)}. Install with: "
                "sudo apt install -y libgles2 libegl1",
            )
    return True, "mediapipe FaceLandmarker Tasks API is available."


def find_grid_videos(video_dir: str | Path, max_videos: int | None = None) -> list[Path]:
    video_dir = resolve_path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"GRID video directory not found: {video_dir}")
    videos = sorted(path for path in video_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTS)
    if max_videos is not None:
        videos = videos[: max(0, max_videos)]
    if not videos:
        raise RuntimeError(f"No video files found under {video_dir}")
    return videos


def split_for_index(index: int, total: int, ratios: tuple[float, float, float]) -> str:
    train_ratio, val_ratio, _ = ratios
    if total <= 1:
        return "train"
    train_count = max(1, int(round(total * train_ratio)))
    val_count = int(round(total * val_ratio))
    if total >= 3:
        val_count = max(1, val_count)
    if train_count + val_count >= total:
        train_count = max(1, total - val_count - 1)
    test_count = total - train_count - val_count
    if total >= 3 and test_count < 1:
        train_count = max(1, train_count - (1 - test_count))
        test_count = total - train_count - val_count
    train_end = train_count
    val_end = train_count + val_count
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def _point(landmarks, index: int) -> np.ndarray:
    point = landmarks[index]
    return np.asarray([point.x, point.y], dtype=np.float32)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def mouth_params_from_facemesh(landmarks) -> tuple[float, float, float, dict[str, float]]:
    upper_lip = _point(landmarks, 13)
    lower_lip = _point(landmarks, 14)
    left_corner = _point(landmarks, 61)
    right_corner = _point(landmarks, 291)
    left_face = _point(landmarks, 234)
    right_face = _point(landmarks, 454)

    face_width = max(_dist(left_face, right_face), 1e-6)
    mouth_vertical = _dist(upper_lip, lower_lip)
    mouth_width_raw = _dist(left_corner, right_corner)

    mouth_open = float(np.clip(mouth_vertical / (face_width * 0.12), 0.0, 1.0))
    mouth_width = float(np.clip(mouth_width_raw / (face_width * 0.55), 0.0, 1.0))
    mouth_round = float(np.clip((mouth_vertical / max(mouth_width_raw, 1e-6)) * 2.5, 0.0, 1.0))
    debug = {
        "upper_y": float(upper_lip[1]),
        "lower_y": float(lower_lip[1]),
        "left_x": float(left_corner[0]),
        "right_x": float(right_corner[0]),
        "face_width": float(face_width),
        "mouth_vertical": float(mouth_vertical),
        "mouth_width_raw": float(mouth_width_raw),
    }
    return mouth_open, mouth_width, mouth_round, debug


def ensure_face_landmarker_model(model_path: str | Path, model_url: str, auto_download: bool) -> Path:
    model_path = resolve_path(model_path)
    if model_path.exists():
        return model_path
    if not auto_download:
        raise FileNotFoundError(
            f"MediaPipe FaceLandmarker model not found: {model_path}. "
            "Set landmarks.auto_download_model=true or download face_landmarker.task manually."
        )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(model_url, timeout=60) as response:
        data = response.read()
    model_path.write_bytes(data)
    return model_path


def extract_landmarks_for_video(video_path: str | Path, output_csv_path: str | Path, cfg: LandmarkConfig) -> np.ndarray:
    ok, message = check_mediapipe_available()
    if not ok:
        raise RuntimeError(message)

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    video_path = resolve_path(video_path)
    output_csv_path = resolve_path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = ensure_face_landmarker_model(cfg.model_path, cfg.model_url, cfg.auto_download_model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    rows: list[dict[str, float | int]] = []
    targets: list[tuple[float, float, float]] = []
    frame_index = 0

    base_options = create_base_options(mp_python, model_path, cfg.delegate)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=cfg.min_detection_confidence,
        min_face_presence_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            timestamp_ms = int(round(frame_index * 1000.0 / fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            time_sec = frame_index / fps
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                mouth_open, mouth_width, mouth_round, debug = mouth_params_from_facemesh(landmarks)
                detected = 1
            else:
                mouth_open = mouth_width = mouth_round = 0.0
                debug = {
                    "upper_y": 0.0,
                    "lower_y": 0.0,
                    "left_x": 0.0,
                    "right_x": 0.0,
                    "face_width": 0.0,
                    "mouth_vertical": 0.0,
                    "mouth_width_raw": 0.0,
                }
                detected = 0
            rows.append(
                {
                    "frame_index": frame_index,
                    "time_sec": time_sec,
                    "mouth_open": mouth_open,
                    "mouth_width": mouth_width,
                    "mouth_round": mouth_round,
                    "face_detected": detected,
                    **debug,
                }
            )
            targets.append((mouth_open, mouth_width, mouth_round))
            frame_index += 1
    cap.release()

    fieldnames = [
        "frame_index",
        "time_sec",
        "mouth_open",
        "mouth_width",
        "mouth_round",
        "face_detected",
        "upper_y",
        "lower_y",
        "left_x",
        "right_x",
        "face_width",
        "mouth_vertical",
        "mouth_width_raw",
    ]
    with output_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return np.asarray(targets, dtype=np.float32)


def create_base_options(mp_python, model_path: Path, delegate: str):
    normalized = str(delegate or "cpu").strip().lower()
    if normalized == "cpu":
        return mp_python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
    if normalized == "gpu":
        return mp_python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp_python.BaseOptions.Delegate.GPU,
        )
    raise ValueError("landmarks.delegate must be 'cpu' or 'gpu'.")


def prepare_grid_video_landmarks(cfg: LandmarkConfig, logger) -> Path:
    videos = find_grid_videos(cfg.video_dir, cfg.max_videos)
    rng = np.random.default_rng(cfg.seed)
    order = np.arange(len(videos))
    rng.shuffle(order)
    shuffled_videos = [videos[int(i)] for i in order]

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(shuffled_videos)
    for index, video_path in enumerate(shuffled_videos):
        sample_id = video_path.stem
        landmarks_rel = Path("landmarks") / f"{sample_id}.csv"
        targets_rel = Path(f"targets_{index:06d}.npy")
        landmarks_csv = cfg.output_dir / landmarks_rel
        targets_path = cfg.output_dir / targets_rel
        try:
            targets = extract_landmarks_for_video(video_path, landmarks_csv, cfg)
        except RuntimeError as exc:
            if str(cfg.delegate).strip().lower() == "gpu":
                raise RuntimeError(
                    "MediaPipe GPU delegate failed during GRID landmark extraction. "
                    "Python GPU delegate support is platform-dependent; on Windows it may be unavailable. "
                    "Set landmarks.delegate=cpu in configs/grid_video_landmarks.yaml and rerun."
                ) from exc
            raise
        np.save(targets_path, targets)
        split = split_for_index(index, total, cfg.split_ratios)
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "video": str(video_path),
                "landmarks_csv": str(landmarks_rel).replace("\\", "/"),
                "targets": str(targets_rel).replace("\\", "/"),
                "num_frames": int(targets.shape[0]),
                "mouth_dim": int(targets.shape[1]) if targets.ndim == 2 else 0,
            }
        )
        logger.info("Processed GRID video landmarks %d/%d: %s frames=%d", index + 1, total, sample_id, targets.shape[0])

    manifest_path = cfg.output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["sample_id", "split", "video", "landmarks_csv", "targets", "num_frames", "mouth_dim"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    quality_report = build_grid_landmark_quality_report(
        cfg.output_dir,
        report_path=cfg.quality_report_path,
        min_detection_rate=cfg.quality_min_detection_rate,
    )
    logger.info("GRID landmark quality report: %s", quality_report["report_path"])
    return manifest_path
