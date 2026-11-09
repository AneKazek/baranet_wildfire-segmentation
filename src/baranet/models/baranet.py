"""Full BARA-Net assembly."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..config import ModelConfig, load_model_config
from .decoder import AttentionDecoder
from .encoder import SharedResidualEncoder
from .graph import SuperpixelGraphRefinement, slic_map
from .heads import BAMaskHead, BoundaryHead
from .temporal import MultiScaleConvGRU

LEVEL_INDEX = {"E1": 0, "E2": 1, "E3": 2, "E4": 3, "E5": 4}
# Latest-date logical channels used for SLIC / graph node spectra: I2, I3, M11.
SPECTRAL_CHANNELS = [1, 2, 5]


class BaraNet(nn.Module):
    """Temporal encoder + ConvGRU + attention decoder (+ optional SGR + heads).

    Input : x [B, T, C, H, W] (T=3, C=8) and optionally sp_map [B, H, W].
            If sp_map is None and SGR is enabled, SLIC is computed on the fly
            from the latest frame's [I2, I3, M11] channels.
    Output: dict with `ba_logits` [B,1,H,W] and `boundary_logits` [B,1,H,W] or None.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        channels = cfg.encoder.channels
        groups = cfg.encoder.norm_groups

        self.encoder = SharedResidualEncoder(cfg.input_channels, channels, groups)

        self.temporal: MultiScaleConvGRU | None = None
        if cfg.temporal.enabled:
            levels = [LEVEL_INDEX[name] for name in cfg.temporal.levels]
            if any(i == 0 for i in levels):
                raise ValueError("temporal.levels must not contain E1")
            self.temporal_levels = levels
            self.temporal = MultiScaleConvGRU(
                [channels[i] for i in levels],
                cfg.temporal.hidden_channels,
                cfg.temporal.kernel_size,
            )
        else:
            self.temporal_levels = []

        self.decoder = AttentionDecoder(
            channels,
            cfg.decoder.channels,
            cfg.decoder.align_corners,
            cfg.decoder.attention_intermediate_min,
            groups,
        )

        self.sgr: SuperpixelGraphRefinement | None = None
        if cfg.sgr.enabled:
            self.sgr = SuperpixelGraphRefinement(
                cfg.sgr, feature_channels=cfg.decoder.channels[-1]
            )

        self.ba_head = BAMaskHead(cfg.decoder.channels[-1], groups) if cfg.heads.ba_mask else None
        self.boundary_head: BoundaryHead | None = None
        if cfg.heads.boundary:
            self.boundary_head = BoundaryHead(
                cfg.decoder.channels[-1], cfg.heads.boundary_hidden_channels, groups
            )

    def forward(
        self, x: torch.Tensor, sp_map: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | None]:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"expected [B,T,C,H,W] or [B,C,H,W], got {tuple(x.shape)}")
        batch, steps = x.shape[:2]

        frames = self.encoder(x.flatten(0, 1))
        frames = [f.view(batch, steps, *f.shape[1:]) for f in frames]

        fused = [f[:, -1] for f in frames]
        if self.temporal is not None:
            levels = [frames[i] for i in self.temporal_levels]
            for i, out in zip(self.temporal_levels, self.temporal(levels), strict=False):
                fused[i] = out

        d1 = self.decoder(fused)

        if self.sgr is not None:
            spectral = x[:, -1][:, SPECTRAL_CHANNELS]
            if sp_map is None:
                sp_map = torch.stack([slic_map(image, self.cfg.sgr) for image in spectral])
            sp_map = sp_map.to(device=d1.device, dtype=torch.long)
            d1 = self.sgr(d1, spectral, sp_map)

        if self.ba_head is None:
            raise ValueError("heads.ba_mask must be enabled for BARA-Net")
        return {
            "ba_logits": self.ba_head(d1),
            "boundary_logits": None if self.boundary_head is None else self.boundary_head(d1),
        }


def build_model(cfg: ModelConfig | str | Path) -> BaraNet:
    """Build BARA-Net or a baseline from a config object or YAML path."""
    if isinstance(cfg, (str, Path)):
        cfg = load_model_config(cfg)
    if cfg.name == "unet_frame":
        from .baselines import UNetFrame

        return UNetFrame(cfg)
    if cfg.name == "unet_convgru":
        from .baselines import UNetConvGRU

        return UNetConvGRU(cfg)
    if cfg.name == "baranet":
        return BaraNet(cfg)
    raise ValueError(f"unknown model.name: {cfg.name!r}")
