from __future__ import annotations

import torch
from torch import nn


class MouthLSTM(nn.Module):
    """Sequence baseline for temporal smoothing and context.

    Input:  [B, T, F]
    Output: [B, T, C]
    """

    def __init__(
        self,
        input_dim: int = 70,
        hidden_dim: int = 96,
        num_layers: int = 1,
        output_dim: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.head(y)
