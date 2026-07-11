from __future__ import annotations

import csv

import numpy as np

from mindface.data.synthetic_dataset import SyntheticMouthDataset


def test_dataset_frame_and_sequence_modes(tmp_path) -> None:
    features = np.zeros((4, 70), dtype=np.float32)
    targets = np.zeros((4, 3), dtype=np.float32)
    np.save(tmp_path / "features_000000.npy", features)
    np.save(tmp_path / "targets_000000.npy", targets)
    with (tmp_path / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "features", "targets"])
        writer.writeheader()
        writer.writerow({"split": "train", "features": "features_000000.npy", "targets": "targets_000000.npy"})

    frame_ds = SyntheticMouthDataset(tmp_path, split="train", sequence_mode=False)
    seq_ds = SyntheticMouthDataset(tmp_path, split="train", sequence_mode=True)

    x, y = frame_ds[0]
    assert len(frame_ds) == 4
    assert x.shape == (70,)
    assert y.shape == (3,)
    seq_x, seq_y = seq_ds[0]
    assert len(seq_ds) == 1
    assert seq_x.shape == (4, 70)
    assert seq_y.shape == (4, 3)

