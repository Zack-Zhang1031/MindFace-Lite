from __future__ import annotations

import numpy as np

from mindface.audio.features import compute_rms, rms_to_mouth_open


def test_compute_rms_constant_signal() -> None:
    waveform = np.ones(1600, dtype=np.float32) * 0.5
    rms = compute_rms(waveform, sample_rate=16000, fps=25, frame_ms=40)
    assert rms.shape[0] > 0
    assert np.allclose(rms[:2], 0.5, atol=1e-6)
    assert np.all(rms <= 0.5 + 1e-6)


def test_mouth_open_is_bounded() -> None:
    rms = np.asarray([0.0, 0.01, 0.1, 1.0], dtype=np.float32)
    mouth = rms_to_mouth_open(rms, noise_floor=0.01, gamma=0.65, smoothing=0.0)
    assert float(mouth.min()) >= 0.0
    assert float(mouth.max()) <= 1.0
