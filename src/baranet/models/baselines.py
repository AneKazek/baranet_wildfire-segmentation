"""Baselines B1/B2. They reuse the exact same encoder/decoder as BARA-Net."""

from __future__ import annotations

from ..config import ModelConfig
from .baranet import BaraNet


class UNetFrame(BaraNet):
    """B1: latest frame only, no temporal module, no SGR, no boundary head."""

    def __init__(self, cfg: ModelConfig):
        cfg = cfg.model_copy(deep=True)
        cfg.name = "unet_frame"
        cfg.temporal.enabled = False
        cfg.sgr.enabled = False
        cfg.heads.boundary = False
        super().__init__(cfg)


class UNetConvGRU(BaraNet):
    """B2: shared encoder + multi-scale ConvGRU + attention decoder, no SGR."""

    def __init__(self, cfg: ModelConfig):
        cfg = cfg.model_copy(deep=True)
        cfg.name = "unet_convgru"
        cfg.temporal.enabled = True
        cfg.sgr.enabled = False
        cfg.heads.boundary = False
        super().__init__(cfg)
