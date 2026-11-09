import torch
import torch.nn as nn

from baranet.engine.inference import blend_window, patch_positions, predict_scene


class MeanModel(nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"ba_logits": x.mean(dim=(1, 2), keepdim=True).squeeze(2)}


def test_patch_positions_reach_the_boundary():
    positions = patch_positions(100, patch=64, stride=48)
    assert positions[0] == 0
    assert positions[-1] == 36  # 100 - 64
    assert positions == sorted(positions)


def test_blend_window_is_positive_everywhere():
    window = blend_window(24, 32, floor=0.01)
    assert window.shape == (24, 32)
    assert float(window.min()) > 0.0


def test_predict_scene_covers_full_non_multiple_scene():
    frames = torch.randn(3, 8, 100, 130)
    prob = predict_scene(
        MeanModel(), frames, patch_size=64, stride=48, device="cpu"
    )
    assert prob.shape == (100, 130)
    assert torch.isfinite(prob).all()
    assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0


def test_predict_scene_handles_scene_smaller_than_patch():
    frames = torch.randn(3, 8, 40, 40)
    prob = predict_scene(MeanModel(), frames, patch_size=64, stride=32, device="cpu")
    assert prob.shape == (40, 40)
