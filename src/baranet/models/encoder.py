"""Shared multispectral 2D residual encoder (E1..E5)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ResidualBlock, conv_norm_act


class SharedResidualEncoder(nn.Module):
    """One encoder instance, applied to every frame with shared weights.

    E1 32xHxW, E2 64xH/2xW/2, E3 128xH/4xW/4, E4 192xH/8xW/8, E5 256xH/16xW/16.
    """

    def __init__(self, in_channels: int = 8, channels: list[int] | None = None, groups: int = 8):
        super().__init__()
        channels = channels or [32, 64, 128, 192, 256]
        if len(channels) != 5:
            raise ValueError(f"encoder.channels must have 5 stages, got {channels}")
        self.channels = channels

        self.stem = conv_norm_act(in_channels, channels[0], groups=groups)
        self.stem_block = ResidualBlock(channels[0], channels[0], groups)

        self.down = nn.ModuleList()
        for in_ch, out_ch in zip(channels[:-1], channels[1:], strict=False):
            self.down.append(
                nn.Sequential(
                    conv_norm_act(in_ch, out_ch, stride=2, groups=groups),
                    ResidualBlock(out_ch, out_ch, groups),
                )
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: [N, C, H, W] -> [E1, E2, E3, E4, E5]."""
        feats = [self.stem_block(self.stem(x))]
        for block in self.down:
            feats.append(block(feats[-1]))
        return feats
