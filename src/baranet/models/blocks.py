"""Shared building blocks: convolution-norm-activation and residual block."""

from __future__ import annotations

import torch
import torch.nn as nn


def _groups(channels: int, requested: int) -> int:
    if channels % requested != 0:
        raise ValueError(
            f"GroupNorm needs channels divisible by groups: {channels=} {requested=}"
        )
    return requested


def conv_norm_act(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 8,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.GroupNorm(_groups(out_channels, groups), out_channels),
        nn.GELU(),
    )


class ResidualBlock(nn.Module):
    """conv-GN-GELU x2 with an identity or 1x1 projected residual path."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_channels, groups), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels, groups), out_channels)
        self.act = nn.GELU()
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.proj(x))
