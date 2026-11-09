"""Pixel-level counts and fire-level macro / pixel-level micro segmentation metrics."""

from __future__ import annotations

from typing import Any

import torch

ArrayLike = torch.Tensor | list[Any]


def _as_2d(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    if x.dim() == 4:
        x = x[:, 0]
    if x.dim() != 3:
        raise ValueError(f"expected [B,1,H,W] or [B,H,W], got {tuple(x.shape)}")
    return x.reshape(x.shape[0], -1)


def _valid_2d(valid: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    if valid is None:
        return torch.ones_like(like)
    v = valid.detach().float()
    if v.dim() == 4:
        v = v[:, 0]
    return v.reshape(v.shape[0], -1)


def sample_counts(
    prob: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample TP/FP/FN counts as float tensors."""
    p = _as_2d(prob)
    t = (_as_2d(target) > 0.5).float()
    m = _valid_2d(valid_mask, t)
    pred = (p >= threshold).float() * m
    tp = (pred * t).sum(1)
    fp = (pred * (1.0 - t) * m).sum(1)
    fn = ((1.0 - pred) * t).sum(1)
    return tp, fp, fn


def _f1(tp: float, fp: float, fn: float) -> float:
    denom = 2.0 * tp + fp + fn
    return 0.0 if denom == 0 else 2.0 * tp / denom


def _iou(tp: float, fp: float, fn: float) -> float:
    denom = tp + fp + fn
    return 0.0 if denom == 0 else tp / denom


def _precision(tp: float, fp: float) -> float:
    denom = tp + fp
    return 0.0 if denom == 0 else tp / denom


def _recall(tp: float, fn: float) -> float:
    denom = tp + fn
    return 0.0 if denom == 0 else tp / denom


class SegmentationMetrics:
    """Accumulates TP/FP/FN per fire ID.

    Macro metrics average over fires that contain at least one positive target
    pixel; zero-positive windows still contribute to micro counts and are
    reported separately.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.per_fire: dict[Any, list[float]] = {}
        self.n_windows = 0
        self.zero_positive_windows = 0

    def update(
        self,
        prob: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        fire_id: ArrayLike | None = None,
    ) -> None:
        tp, fp, fn = sample_counts(prob, target, valid_mask, self.threshold)
        t = _as_2d(target)
        ids = list(range(tp.shape[0])) if fire_id is None else list(fire_id)
        if len(ids) != tp.shape[0]:
            raise ValueError(f"fire_id has {len(ids)} entries for batch {tp.shape[0]}")
        for i, fid in enumerate(ids):
            counts = self.per_fire.setdefault(fid, [0.0, 0.0, 0.0])
            counts[0] += float(tp[i])
            counts[1] += float(fp[i])
            counts[2] += float(fn[i])
            self.n_windows += 1
            if float(t[i].sum()) == 0.0:
                self.zero_positive_windows += 1

    def compute(self) -> dict[str, float]:
        total = [0.0, 0.0, 0.0]
        macro: list[list[float]] = []
        for counts in self.per_fire.values():
            for i in range(3):
                total[i] += counts[i]
            if counts[0] + counts[2] > 0:  # fire has positive target pixels
                macro.append(counts)
        tp, fp, fn = total

        def mean(rows: list[list[float]], fn_metric) -> float:
            return float(sum(fn_metric(*r) for r in rows) / len(rows)) if rows else 0.0

        return {
            "macro_f1": mean(macro, _f1),
            "micro_f1": _f1(tp, fp, fn),
            "macro_iou": mean(macro, _iou),
            "micro_iou": _iou(tp, fp, fn),
            "macro_precision": mean(macro, lambda a, b, _: _precision(a, b)),
            "macro_recall": mean(macro, lambda a, _, c: _recall(a, c)),
            "micro_precision": _precision(tp, fp),
            "micro_recall": _recall(tp, fn),
            "n_fires": len(macro),
            "n_windows": self.n_windows,
            "zero_positive_windows": self.zero_positive_windows,
        }


def select_threshold(
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, ArrayLike | None]],
    low: float = 0.05,
    high: float = 0.95,
    step: float = 0.01,
) -> float:
    """Pick the threshold with the best fire-level macro F1; ties go to the higher one."""
    thresholds = torch.arange(low, high + step / 2, step).tolist()
    best_threshold, best_score = high, -1.0
    for threshold in thresholds:
        metrics = SegmentationMetrics(threshold)
        for prob, target, valid, fire_id in samples:
            metrics.update(prob, target, valid, fire_id)
        score = metrics.compute()["macro_f1"]
        if score >= best_score:
            best_threshold, best_score = threshold, score
    return round(float(best_threshold), 4)
