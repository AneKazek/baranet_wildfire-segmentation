"""Attention-gated U-Net decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlock


class AttentionGate(nn.Module):
    """alpha = sigmoid(psi(ReLU(Wx x + Wg g))); returns alpha * x."""

    def __init__(self, skip_channels: int, gate_channels: int, out_channels: int, min_width: int):
        super().__init__()
        inner = max(min_width, out_channels // 2)
        self.wx = nn.Conv2d(skip_channels, inner, 1)
        self.wg = nn.Conv2d(gate_channels, inner, 1)
        self.psi = nn.Conv2d(inner, 1, 1)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.psi(F.relu(self.wx(x) + self.wg(g))))
        return x * alpha


class DecoderBlock(nn.Module):
    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        out_channels: int,
        align_corners: bool,
        min_attention_width: int,
        groups: int = 8,
    ):
        super().__init__()
        self.align_corners = align_corners
        self.up = nn.Conv2d(decoder_channels, out_channels, 3, padding=1, bias=False)
        self.gate = AttentionGate(
            skip_channels, out_channels, out_channels, min_attention_width
        )
        self.fuse = ResidualBlock(out_channels + skip_channels, out_channels, groups)

    def forward(self, decoder: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = F.interpolate(
            decoder,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        g = self.up(g)
        return self.fuse(torch.cat([g, self.gate(skip, g)], dim=1))


class AttentionDecoder(nn.Module):
    """D4 -> D3 -> D2 -> D1; D1 has out_channels[3] at full resolution."""

    def __init__(
        self,
        encoder_channels: list[int],
        out_channels: list[int],
        align_corners: bool = False,
        min_attention_width: int = 16,
        groups: int = 8,
    ):
        super().__init__()
        if len(out_channels) != 4:
            raise ValueError(f"decoder.channels must have 4 stages, got {out_channels}")
        sources = [encoder_channels[-1], *out_channels[:-1]]
        skips = encoder_channels[-2::-1]  # E4, E3, E2, E1
        self.blocks = nn.ModuleList(
            DecoderBlock(src, skip, out, align_corners, min_attention_width, groups)
            for src, skip, out in zip(sources, skips, out_channels, strict=False)
        )
        self.out_channels = out_channels[-1]

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        d = feats[-1]
        for block, skip in zip(self.blocks, feats[-2::-1], strict=False):
            d = block(d, skip)
        return d
