"""Loss suite: Focal-Tversky + Dice + Boundary, all masked by `valid_mask`."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LossConfig


def _masked(prob: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None):
    y = target.float()
    if valid is None:
        return prob, y
    m = valid.float()
    return prob * m, y * m


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    alpha: float = 0.30,
    beta: float = 0.70,
    gamma: float = 0.75,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Per-sample Focal Tversky (beta weights false negatives), averaged over the batch."""
    prob, y = _masked(torch.sigmoid(logits), target, valid_mask)
    dims = (1, 2, 3)
    tp = (prob * y).sum(dims)
    fp = (prob * (1.0 - y)).sum(dims)
    fn = ((1.0 - prob) * y).sum(dims)
    ti = (tp + epsilon) / (tp + alpha * fp + beta * fn + epsilon)
    return (1.0 - ti).pow(gamma).mean()


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    prob, y = _masked(torch.sigmoid(logits), target, valid_mask)
    dims = (1, 2, 3)
    tp = (prob * y).sum(dims)
    fp = (prob * (1.0 - y)).sum(dims)
    fn = ((1.0 - prob) * y).sum(dims)
    return (1.0 - (2.0 * tp + epsilon) / (2.0 * tp + fp + fn + epsilon)).mean()


def morphological_gradient(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Dilate3x3(mask) - Erode3x3(mask), valid for [B, 1, H, W] binary masks."""
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask.float(), kernel_size, 1, pad)
    eroded = -F.max_pool2d(-mask.float(), kernel_size, 1, pad)
    return (dilated - eroded).clamp(0.0, 1.0)


class BoundaryLoss(nn.Module):
    """0.5 * masked BCEWithLogits + 0.5 * masked Dice on the boundary map."""

    def __init__(self, pos_weight: float | None = None, epsilon: float = 1e-7):
        super().__init__()
        self.pos_weight = pos_weight
        self.epsilon = epsilon

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = target.float()
        weight = None if self.pos_weight is None else torch.as_tensor(
            self.pos_weight, dtype=logits.dtype, device=logits.device
        )
        bce = F.binary_cross_entropy_with_logits(
            logits, y, reduction="none", pos_weight=weight
        )
        prob = torch.sigmoid(logits)
        if valid_mask is not None:
            m = valid_mask.float()
            bce = bce * m
            prob, y = prob * m, y * m
            norm = m.sum().clamp_min(1.0)
        else:
            norm = torch.as_tensor(y.numel(), dtype=bce.dtype, device=bce.device)
        bce = bce.sum() / norm

        dims = (1, 2, 3)
        tp = (prob * y).sum(dims)
        fp = (prob * (1.0 - y)).sum(dims)
        fn = ((1.0 - prob) * y).sum(dims)
        dice = (1.0 - (2.0 * tp + self.epsilon) /
                (2.0 * tp + fp + fn + self.epsilon)).mean()
        return 0.5 * bce + 0.5 * dice


class CombinedLoss(nn.Module):
    """0.50 Focal-Tversky + 0.35 Dice + 0.15 Boundary.

    When the boundary head/target is absent, the remaining weights are
    renormalized to sum to 1.0 (A2 ablation: 0.5882 / 0.4118).
    """

    def __init__(self, cfg: LossConfig, pos_weight: float | None = None):
        super().__init__()
        self.cfg = cfg
        self.boundary = BoundaryLoss(pos_weight, cfg.epsilon)

    def forward(
        self,
        ba_logits: torch.Tensor,
        ba_target: torch.Tensor,
        boundary_logits: torch.Tensor | None = None,
        boundary_target: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        ft = focal_tversky_loss(
            ba_logits, ba_target, valid_mask,
            cfg.tversky_alpha, cfg.tversky_beta, cfg.tversky_gamma, cfg.epsilon,
        )
        dice = dice_loss(ba_logits, ba_target, valid_mask, cfg.epsilon)

        use_boundary = boundary_logits is not None and boundary_target is not None
        if use_boundary:
            bnd = self.boundary(boundary_logits, boundary_target, valid_mask)
            total = cfg.focal_tversky * ft + cfg.dice * dice + cfg.boundary * bnd
        else:
            bnd = torch.zeros((), dtype=ft.dtype, device=ft.device)
            weights = cfg.focal_tversky + cfg.dice
            total = (cfg.focal_tversky / weights) * ft + (cfg.dice / weights) * dice
        return {
            "loss": total,
            "focal_tversky": ft.detach(),
            "dice": dice.detach(),
            "boundary": bnd.detach(),
        }
