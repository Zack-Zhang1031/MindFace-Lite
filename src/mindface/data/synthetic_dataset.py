from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from mindface.utils.config import resolve_path


class MouthParameterDataset(Dataset):
    """Manifest-based audio-feature to mouth-parameter dataset.

    For MLP training, sequence_mode=False returns individual frames.
    For LSTM/TCN/Transformer training, sequence_mode=True returns full sequences.
    """

    def __init__(self, dataset_dir: str | Path, split: str = "train", sequence_mode: bool = False):
        self.dataset_dir = resolve_path(dataset_dir)
        self.split = split
        self.sequence_mode = sequence_mode
        self.manifest_path = self.dataset_dir / "manifest.csv"
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Dataset manifest not found: {self.manifest_path}. "
                "Run scripts/02_generate_synthetic_dataset.py first."
            )

        self.samples: list[tuple[Path, Path]] = []
        with self.manifest_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.samples.append((self.dataset_dir / row["features"], self.dataset_dir / row["targets"]))

        if not self.samples:
            raise RuntimeError(f"No samples for split={split} in {self.manifest_path}")

        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.frame_index: list[tuple[int, int]] = []
        if not sequence_mode:
            for sample_idx, _ in enumerate(self.samples):
                features, targets = self._load_sample(sample_idx)
                if len(features) != len(targets):
                    raise ValueError(f"Length mismatch in sample {sample_idx}")
                self.frame_index.extend((sample_idx, frame_idx) for frame_idx in range(len(features)))

    def __len__(self) -> int:
        return len(self.samples) if self.sequence_mode else len(self.frame_index)

    def _load_sample(self, sample_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if sample_idx not in self._cache:
            feature_path, target_path = self.samples[sample_idx]
            features = np.load(feature_path).astype(np.float32)
            targets = np.load(target_path).astype(np.float32)
            if features.ndim != 2:
                raise ValueError(f"Features must be [T, F], got {features.shape} from {feature_path}")
            if targets.ndim != 2:
                raise ValueError(f"Targets must be [T, C], got {targets.shape} from {target_path}")
            self._cache[sample_idx] = (features, targets)
        return self._cache[sample_idx]

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if self.sequence_mode:
            return self._load_sample(idx)
        sample_idx, frame_idx = self.frame_index[idx]
        features, targets = self._load_sample(sample_idx)
        return features[frame_idx], targets[frame_idx]


# Backward-compatible name used by earlier stages and older docs.
SyntheticMouthDataset = MouthParameterDataset
