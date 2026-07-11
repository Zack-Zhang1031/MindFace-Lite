from __future__ import annotations

from pathlib import Path

import numpy as np

from mindface.visual.static_avatar_warper import ExpressiveAvatarConfig, MouthRoi, render_static_avatar_frame


def test_static_avatar_frame_changes_with_mouth_open() -> None:
    base = np.full((128, 128, 3), 210, dtype=np.uint8)
    cfg = ExpressiveAvatarConfig(
        image_path=Path("unused.png"),
        width=128,
        height=128,
        mouth_roi=MouthRoi(center_x=0.5, center_y=0.6, width=0.35, height=0.2, max_open_ratio=0.08),
        head_bob_px=0.0,
        cheek_pulse=False,
    )
    closed = render_static_avatar_frame(base, 0.0, config=cfg)
    open_frame = render_static_avatar_frame(base, 1.0, config=cfg)
    assert closed.shape == base.shape
    assert open_frame.shape == base.shape
    assert np.mean(np.abs(open_frame.astype(np.float32) - closed.astype(np.float32))) > 0.1

