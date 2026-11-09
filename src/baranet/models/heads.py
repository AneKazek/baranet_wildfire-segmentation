"""Dual prediction heads: BA mask (primary) and boundary (auxiliary)."""

from __future__ import annotations

import torch
import torch.nn as nn


class BAMaskHead(nn.Module):
    def __init__(self, in_channels: int = 32, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(groups, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BoundaryHead(nn.Module):
    def __init__(self, in_channels: int = 32, hidden_channels: int = 16, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
