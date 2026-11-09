"""Multi-scale ConvGRU temporal fusion at E2..E5."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    """3x3 ConvGRU cell; h_t = (1 - z) * h_{t-1} + z * h_hat."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        inner = input_channels + hidden_channels
        self.conv_z = nn.Conv2d(inner, hidden_channels, kernel_size, padding=pad)
        self.conv_r = nn.Conv2d(inner, hidden_channels, kernel_size, padding=pad)
        self.conv_h = nn.Conv2d(inner, hidden_channels, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        xh = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.conv_z(xh))
        r = torch.sigmoid(self.conv_r(xh))
        h_hat = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * h_hat


class MultiScaleConvGRU(nn.Module):
    """Temporal fusion for a list of encoder levels.

    Input : list of [B, T, C_l, H_l, W_l], chronological order on dim=1.
    Output: list of [B, C_l, H_l, W_l] with F_l = Conv1x1([E_l^t || h_l^t]).
    E1 is intentionally not fused and is handled by the caller.
    """

    def __init__(
        self,
        in_channels: list[int],
        hidden_channels: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.proj = nn.ModuleList(
            nn.Conv2d(ch, hidden_channels, 1) for ch in in_channels
        )
        self.cells = nn.ModuleList(
            ConvGRUCell(hidden_channels, hidden_channels, kernel_size) for _ in in_channels
        )
        # [E_l^t (native width) || hidden state] -> native width
        self.fuse = nn.ModuleList(
            nn.Conv2d(ch + hidden_channels, ch, 1) for ch in in_channels
        )

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        for _level, (feat, proj, cell, fuse) in enumerate(
            zip(feats, self.proj, self.cells, self.fuse, strict=False)
        ):
            b, t = feat.shape[:2]
            hidden = None
            for step in range(t):
                x = proj(feat[:, step])
                if hidden is None:
                    hidden = torch.zeros_like(x)
                hidden = cell(x, hidden)
            out.append(fuse(torch.cat([feat[:, -1], hidden], dim=1)))
        return out
