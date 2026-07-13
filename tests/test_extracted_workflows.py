from __future__ import annotations

import numpy as np

from mindface.data.grid_audio import build_splits, make_rule_targets
from mindface.visual.expressive_avatar import build_expressive_params


def test_grid_audio_helpers_live_in_package_module() -> None:
    waveform = np.zeros(1600, dtype=np.float32)

    targets = make_rule_targets(
        waveform,
        sample_rate=16000,
        fps=25,
        frame_ms=40.0,
        noise_floor=0.01,
        gamma=1.0,
        smoothing=0.0,
    )

    assert build_splits(4, train_ratio=0.5, val_ratio=0.25) == ["train", "train", "val", "test"]
    assert targets.shape[1] == 3


def test_closed_viseme_forces_mouth_closed() -> None:
    mouth_open = np.asarray([0.9], dtype=np.float32)
    times = np.asarray([0.25], dtype=np.float32)
    config = {
        "viseme": {
            "enabled": True,
            "events": [{"start": 0.0, "end": 0.5, "shape": "closed"}],
        }
    }

    params, labels = build_expressive_params(mouth_open, times, config)

    assert labels == ["closed"]
    assert float(params[0, 0]) <= 0.025001
