from __future__ import annotations

from torch import nn

from mindface.models.lstm import MouthLSTM
from mindface.models.mlp import MouthMLP
from mindface.models.tcn import MouthTCN
from mindface.models.transformer import MouthTransformer


SEQUENCE_MODELS = {"lstm", "tcn", "transformer"}


def build_model(model_type: str, model_config: dict) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "mlp":
        return MouthMLP(**model_config)
    if model_type == "lstm":
        return MouthLSTM(**model_config)
    if model_type == "tcn":
        return MouthTCN(**model_config)
    if model_type == "transformer":
        return MouthTransformer(**model_config)
    raise ValueError(f"Unsupported model_type={model_type}")


def is_sequence_model(model_type: str) -> bool:
    return model_type.lower() in SEQUENCE_MODELS
