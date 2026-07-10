from __future__ import annotations

import torch
from torch import nn


class MouthTCN(nn.Module):
    """Tiny temporal convolution network.

    Input:  [B, T, F]
    Output: [B, T, C]
    """

    def __init__(self, input_dim: int = 70, hidden_dim: int = 96, output_dim: int = 3, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        y = self.net(x)
        return y.transpose(1, 2)
