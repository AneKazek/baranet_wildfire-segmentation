import torch

from baranet import CombinedLoss, LossConfig, dice_loss, focal_tversky_loss
from baranet.losses import morphological_gradient


def _batch(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    target = (torch.rand(2, 1, 16, 16, generator=gen) > 0.7).float()
    logits = torch.randn(2, 1, 16, 16, generator=gen)
    return logits, target


def test_losses_are_finite_and_differentiable():
    logits, target = _batch()
    logits.requires_grad_(True)
    losses = CombinedLoss(LossConfig())(
        logits, target, logits, target
    )
    losses["loss"].backward()
    for key, value in losses.items():
        assert torch.isfinite(value).all(), key
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_valid_mask_ignores_invalid_pixels():
    logits, target = _batch()
    valid = torch.ones_like(target)
    valid[..., :8] = 0.0
    changed = target.clone()
    changed[..., :8] = 1.0 - changed[..., :8]

    for fn in (focal_tversky_loss, dice_loss):
        a = fn(logits, target, valid)
        b = fn(logits, changed, valid)
        assert torch.allclose(a, b, atol=1e-6)


def test_perfect_prediction_scores_better():
    logits, target = _batch()
    perfect = (target * 2 - 1) * 10
    assert focal_tversky_loss(perfect, target) < focal_tversky_loss(logits, target)
    assert dice_loss(perfect, target) < dice_loss(logits, target)


def test_all_negative_target_is_finite():
    target = torch.zeros(2, 1, 16, 16)
    logits = torch.zeros(2, 1, 16, 16)
    assert torch.isfinite(focal_tversky_loss(logits, target))
    assert torch.isfinite(dice_loss(logits, target))


def test_combined_loss_renormalizes_without_boundary():
    cfg = LossConfig()
    logits, target = _batch()
    total = CombinedLoss(cfg)(logits, target)["loss"]
    expected = (
        cfg.focal_tversky / 0.85 * focal_tversky_loss(logits, target)
        + cfg.dice / 0.85 * dice_loss(logits, target)
    )
    assert torch.allclose(total, expected, atol=1e-6)


def test_morphological_gradient_of_square():
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., 2:6, 2:6] = 1.0
    edge = morphological_gradient(mask)
    assert edge.shape == mask.shape
    assert edge[0, 0, 2, 2] == 1.0  # corner
    assert edge[0, 0, 3, 3] == 0.0  # interior
    assert edge[0, 0, 0, 0] == 0.0  # background
