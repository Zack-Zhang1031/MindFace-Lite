from __future__ import annotations

import numpy as np

from mindface.deploy.consistency import _summary_error


def test_summary_error_zero_for_identical_arrays() -> None:
    values = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)
    report = _summary_error(values, values.copy())
    assert report["mae"] == 0.0
    assert report["max_abs_error"] == 0.0
    assert report["rmse"] == 0.0

