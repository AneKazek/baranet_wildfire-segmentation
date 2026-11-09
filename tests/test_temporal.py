import torch

from baranet.models.temporal import MultiScaleConvGRU


def _features(batch=2, steps=3):
    channels = [32, 64, 128, 192, 256]
    return [
        torch.randn(batch, steps, ch, 64 // (2 ** i), 64 // (2 ** i))
        for i, ch in enumerate(channels)
    ]


def test_temporal_output_shapes():
    fusion = MultiScaleConvGRU([64, 128, 192, 256], hidden_channels=64)
    feats = _features()[1:]
    out = fusion(feats)
    assert [o.shape[1] for o in out] == [64, 128, 192, 256]
    assert [o.shape[-1] for o in out] == [32, 16, 8, 4]


def test_earlier_frame_changes_final_hidden_state():
    fusion = MultiScaleConvGRU([64, 128, 192, 256], hidden_channels=64)
    feats = _features()[1:]
    base = fusion(feats)

    changed = [f.clone() for f in feats]
    changed[0][:, 0] += 1.0  # earliest frame at E2 only
    perturbed = fusion(changed)
    assert not torch.allclose(base[0], perturbed[0])

    unchanged = [f.clone() for f in feats]
    unchanged[1][:, -1] += 0.0
    same = fusion(unchanged)
    assert torch.allclose(base[1], same[1])
