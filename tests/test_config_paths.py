from __future__ import annotations

from mindface.utils.config import resolve_path


def test_legacy_config_path_resolves_to_grouped_config() -> None:
    resolved = resolve_path("configs/train_mlp.yaml")

    assert resolved.as_posix().endswith("configs/training/train-mlp.yaml")
