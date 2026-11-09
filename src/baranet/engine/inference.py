"""Full-scene sliding-window inference with weighted overlap blending."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def patch_positions(size: int, patch: int, stride: int) -> list[int]:
    """Grid positions whose last patch reaches the scene boundary."""
    if patch >= size:
        return [0]
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    positions = list(range(0, size - patch + 1, stride))
    if positions[-1] != size - patch:
        positions.append(size - patch)
    return positions


def blend_window(height: int, width: int, floor: float = 0.01) -> torch.Tensor:
    """Center-weighted 2D window with a small positive floor."""
    def ramp(n: int) -> torch.Tensor:
        return (1.0 - torch.linspace(-1.0, 1.0, n).abs()).clamp_min(0.0) + floor

    return ramp(height).unsqueeze(1) * ramp(width).unsqueeze(0)


def _pad(tensor: torch.Tensor, size: int) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    pad_h, pad_w = max(0, size - height), max(0, size - width)
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return tensor


@torch.no_grad()
def predict_scene(
    model: nn.Module,
    frames: torch.Tensor,
    patch_size: int = 256,
    stride: int = 192,
    floor: float = 0.01,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """frames: normalized [T, C, H, W] scene -> probability map [H, W].

    Every original pixel is covered by at least one patch; otherwise this raises.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().to(device)
    _, _, height, width = frames.shape
    accumulator = torch.zeros(height, width)
    weights = torch.zeros(height, width)

    for row in patch_positions(height, patch_size, stride):
        for col in patch_positions(width, patch_size, stride):
            patch = frames[:, :, row : row + patch_size, col : col + patch_size]
            patch_h, patch_w = patch.shape[-2:]
            patch = _pad(patch, patch_size)

            out = model(patch.unsqueeze(0).to(device))
            prob = torch.sigmoid(out["ba_logits"])[0, 0].float().cpu()
            window = blend_window(patch_h, patch_w, floor)

            accumulator[row : row + patch_h, col : col + patch_w] += (
                prob[:patch_h, :patch_w] * window
            )
            weights[row : row + patch_h, col : col + patch_w] += window

    if bool((weights <= 0).any()):
        raise RuntimeError(
            f"inference coverage gap for scene {height}x{width}: denominator has zeros"
        )
    return accumulator / weights
