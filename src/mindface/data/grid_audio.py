from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mindface.audio.features import compute_rms, extract_audio_features, read_wav_mono, rms_to_mouth_open
from mindface.data.splitting import speaker_disjoint_split
from mindface.utils.config import project_root, resolve_path
from mindface.utils.logger import setup_logger


@dataclass(frozen=True, slots=True)
class GridItem:
    speaker: str
    utterance: str
    audio_path: Path
    alignment_path: Path | None
    video_path: Path | None


def _relative(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(project_root())).replace("\\", "/")
    except ValueError:
        return str(path)


def scan_grid(raw_grid_dir: Path, require_alignment: bool, require_video: bool) -> list[GridItem]:
    audio_root = raw_grid_dir / "audio"
    align_root = raw_grid_dir / "alignments"
    video_root = raw_grid_dir / "video"

    if not audio_root.exists():
        raise FileNotFoundError(f"GRID audio directory not found: {audio_root}")
    if require_alignment and not align_root.exists():
        raise FileNotFoundError(f"GRID alignment directory not found: {align_root}")
    if require_video and not video_root.exists():
        raise FileNotFoundError(f"GRID video directory not found: {video_root}")

    video_by_name: dict[str, Path] = {}
    if video_root.exists():
        video_by_name = {path.stem: path for path in video_root.glob("*.mpg")}

    items: list[GridItem] = []
    for audio_path in sorted(audio_root.glob("*/*.wav")):
        speaker = audio_path.parent.name
        utterance = audio_path.stem
        alignment_path = align_root / speaker / f"{utterance}.align"
        video_path = video_by_name.get(utterance)

        if require_alignment and not alignment_path.exists():
            continue
        if require_video and video_path is None:
            continue

        items.append(
            GridItem(
                speaker=speaker,
                utterance=utterance,
                audio_path=audio_path,
                alignment_path=alignment_path if alignment_path.exists() else None,
                video_path=video_path,
            )
        )
    return items


def build_splits(total: int, train_ratio: float, val_ratio: float) -> list[str]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    if total >= 2:
        val_count = max(1, val_count)
    if total >= 3:
        test_count = max(1, total - train_count - val_count)
    else:
        test_count = 0
    train_count = max(1, total - val_count - test_count)
    return (["train"] * train_count + ["val"] * val_count + ["test"] * test_count)[:total]


def make_rule_targets(
    waveform: np.ndarray,
    sample_rate: int,
    fps: int,
    frame_ms: float,
    noise_floor: float,
    gamma: float,
    smoothing: float,
) -> np.ndarray:
    rms = compute_rms(waveform, sample_rate, fps=fps, frame_ms=frame_ms)
    mouth_open = rms_to_mouth_open(rms, noise_floor=noise_floor, gamma=gamma, smoothing=smoothing)
    previous = np.concatenate([[mouth_open[0]], mouth_open[:-1]])
    delta = mouth_open - previous

    # Pseudo labels keep GRID audio trainable before video landmarks are available.
    mouth_width = np.clip(0.45 + 0.30 * mouth_open + 0.05 * np.maximum(delta, 0.0), 0.0, 1.0)
    lip_round = np.clip(0.18 + 0.28 * previous + 0.04 * np.maximum(-delta, 0.0), 0.0, 1.0)
    return np.stack([mouth_open, mouth_width, lip_round], axis=1).astype(np.float32)


def _clean_output_dir(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists() or not overwrite:
        return
    resolved_output = output_dir.resolve()
    resolved_root = project_root().resolve()
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to clean output outside project: {resolved_output}") from exc
    if output_dir.name in {"raw", "grid", "data"}:
        raise RuntimeError(f"Refusing to clean suspicious output directory: {output_dir}")
    shutil.rmtree(output_dir)


def prepare_grid_dataset(
    cfg: dict[str, Any],
    max_samples_override: int | None = None,
    output_dir_override: str | None = None,
) -> Path:
    raw_grid_dir = resolve_path(cfg["raw_grid_dir"])
    output_dir = resolve_path(output_dir_override or cfg["output_dir"])
    _clean_output_dir(output_dir, overwrite=bool(cfg["output"].get("overwrite", True)))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("prepare_grid", cfg["output"]["log_path"])

    items = scan_grid(
        raw_grid_dir,
        require_alignment=bool(cfg["scan"]["require_alignment"]),
        require_video=bool(cfg["scan"]["require_video"]),
    )
    if not items:
        raise RuntimeError(f"No GRID items found in {raw_grid_dir}")

    rng = np.random.default_rng(int(cfg["scan"].get("seed", 42)))
    order = np.arange(len(items))
    rng.shuffle(order)
    items = [items[int(index)] for index in order]

    max_samples = max_samples_override
    if max_samples is None:
        raw_max = cfg["scan"].get("max_samples")
        max_samples = int(raw_max) if raw_max is not None else None
    if max_samples is not None:
        items = items[: max(0, max_samples)]
    if not items:
        raise RuntimeError("No samples selected after max_samples filtering.")

    fps = int(cfg["features"]["fps"])
    frame_ms = float(cfg["features"]["frame_ms"])
    if int(cfg["features"]["feature_dim"]) != 70:
        raise ValueError("Current feature extractor supports feature_dim=70")

    train_ratio = float(cfg["split"]["train_ratio"])
    val_ratio = float(cfg["split"]["val_ratio"])
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")
    split_strategy = str(cfg["split"].get("strategy", "speaker_disjoint"))
    if split_strategy == "speaker_disjoint":
        speaker_splits = speaker_disjoint_split(
            (item.speaker for item in items),
            (train_ratio, val_ratio, float(cfg["split"].get("test_ratio", 1.0 - train_ratio - val_ratio))),
            seed=int(cfg["scan"].get("seed", 42)),
        )
        splits = [speaker_splits[item.speaker] for item in items]
    elif split_strategy == "sample":
        splits = build_splits(len(items), train_ratio, val_ratio)
    else:
        raise ValueError("split.strategy must be 'speaker_disjoint' or 'sample'")

    rows: list[dict[str, Any]] = []
    logger.info("Preparing GRID dataset from %s", raw_grid_dir)
    logger.info("Selected samples: %d output_dir=%s", len(items), output_dir)

    for index, item in enumerate(items):
        sample_rate, waveform = read_wav_mono(item.audio_path)
        features = extract_audio_features(waveform, sample_rate, fps=fps, frame_ms=frame_ms)
        targets = make_rule_targets(
            waveform,
            sample_rate,
            fps=fps,
            frame_ms=frame_ms,
            noise_floor=float(cfg["labels"]["noise_floor"]),
            gamma=float(cfg["labels"]["gamma"]),
            smoothing=float(cfg["labels"]["smoothing"]),
        )
        frame_count = min(len(features), len(targets))
        features = features[:frame_count]
        targets = targets[:frame_count]

        feature_name = f"features_{index:06d}.npy"
        target_name = f"targets_{index:06d}.npy"
        np.save(output_dir / feature_name, features.astype(np.float32))
        np.save(output_dir / target_name, targets.astype(np.float32))

        rows.append(
            {
                "schema_version": 1,
                "split": splits[index],
                "split_strategy": split_strategy,
                "features": feature_name,
                "targets": target_name,
                "speaker": item.speaker,
                "utterance": item.utterance,
                "frames": frame_count,
                "sample_rate": sample_rate,
                "label_source": cfg["labels"]["source"],
                "source_audio": _relative(item.audio_path),
                "source_alignment": _relative(item.alignment_path),
                "source_video": _relative(item.video_path),
            }
        )
        if (index + 1) % 250 == 0:
            logger.info("Processed %d/%d samples", index + 1, len(items))

    manifest_path = output_dir / str(cfg["output"].get("manifest_name", "manifest.csv"))
    fieldnames = [
        "schema_version",
        "split",
        "split_strategy",
        "features",
        "targets",
        "speaker",
        "utterance",
        "frames",
        "sample_rate",
        "label_source",
        "source_audio",
        "source_alignment",
        "source_video",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote manifest: %s", manifest_path)
    return manifest_path
