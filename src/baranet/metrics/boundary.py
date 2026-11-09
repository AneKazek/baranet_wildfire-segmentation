"""Boundary F1 with a configurable pixel tolerance."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..losses.combined import morphological_gradient


def boundary_f1(
    prob: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
    tolerance: int = 1,
) -> dict[str, float]:
    """Boundary precision/recall/F1; a prediction matches if within `tolerance` pixels."""
    pred = (prob.detach().float() >= threshold).float()
    gt = morphological_gradient((target.detach().float() > 0.5).float())
    pr = morphological_gradient(pred)
    if valid_mask is not None:
        m = valid_mask.detach().float()
        gt, pr = gt * m, pr * m

    kernel = 2 * tolerance + 1
    gt_tol = F.max_pool2d(gt, kernel, 1, tolerance)
    pr_tol = F.max_pool2d(pr, kernel, 1, tolerance)
    precision = float((pr * gt_tol).sum() / pr.sum().clamp_min(1e-7))
    recall = float((gt * pr_tol).sum() / gt.sum().clamp_min(1e-7))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "tolerance_px": float(tolerance),
    }
