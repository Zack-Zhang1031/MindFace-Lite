from __future__ import annotations

import torch
from torch import nn


class MouthMLP(nn.Module):
    """Frame-wise baseline model.

    Input:  [B, F]
    Output: [B, C]
    """

    def __init__(self, input_dim: int = 70, hidden_dim: int = 128, output_dim: int = 3, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
