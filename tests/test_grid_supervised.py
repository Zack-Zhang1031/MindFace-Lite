from __future__ import annotations

import csv

import numpy as np

from mindface.audio.features import write_wav_mono
from mindface.data.grid_supervised import prepare_grid_landmark_dataset


def test_prepare_grid_landmark_dataset_from_tiny_fixture(tmp_path) -> None:
    raw_grid_dir = tmp_path / "raw_grid"
    audio_dir = raw_grid_dir / "audio" / "s1"
    audio_dir.mkdir(parents=True)
    sample_rate = 16000
    waveform = np.sin(np.linspace(0, np.pi * 4.0, sample_rate // 2, dtype=np.float32)) * 0.2
    write_wav_mono(audio_dir / "tiny.wav", sample_rate, waveform)

    landmark_dir = tmp_path / "landmarks"
    landmark_dir.mkdir()
    target_path = landmark_dir / "tiny_targets.npy"
    np.save(target_path, np.ones((20, 3), dtype=np.float32) * 0.25)

    landmarks_csv = landmark_dir / "tiny_landmarks.csv"
    with landmarks_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_index", "face_detected"])
        for frame_index in range(20):
            writer.writerow([frame_index, 1])

    with (landmark_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "split", "landmarks_csv", "targets"])
        writer.writerow(["tiny", "train", landmarks_csv.name, target_path.name])

    output_dir = tmp_path / "processed"
    manifest_path = prepare_grid_landmark_dataset(
        {
            "raw_grid_dir": raw_grid_dir,
            "landmark_dir": landmark_dir,
            "output_dir": output_dir,
            "min_detection_rate": 0.95,
            "features": {"fps": 25, "frame_ms": 40, "feature_dim": 70},
            "output": {"overwrite": True, "manifest_name": "manifest.csv"},
        }
    )

    assert manifest_path.exists()
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))
    assert rows[0]["label_source"] == "grid_video_landmark"
    assert (output_dir / rows[0]["features"]).exists()
    assert (output_dir / rows[0]["targets"]).exists()
