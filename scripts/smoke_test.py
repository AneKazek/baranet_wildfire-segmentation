"""Clone-health check on a synthetic batch: forward + backward on CPU.

    uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import torch

from baranet import CombinedLoss, LossConfig, build_model, load_model_config


def main() -> None:
    cfg = load_model_config("configs/model/baranet.yaml")
    cfg.patch_size = 64
    cfg.sgr.slic.n_segments = 64
    model = build_model(cfg)

    x = torch.randn(2, 3, cfg.input_channels, cfg.patch_size, cfg.patch_size)
    target = (torch.rand(2, 1, cfg.patch_size, cfg.patch_size) > 0.8).float()
    boundary = (torch.rand_like(target) > 0.95).float()

    out = model(x)
    losses = CombinedLoss(LossConfig())(
        out["ba_logits"], target, out["boundary_logits"], boundary
    )
    losses["loss"].backward()

    grads = [p for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    print("ba_logits      ", tuple(out["ba_logits"].shape))
    print("boundary_logits", tuple(out["boundary_logits"].shape))
    print("loss           ", {k: round(float(v.detach()), 4) for k, v in losses.items()})
    print("params w/ grad ", len(grads), "/", sum(1 for p in model.parameters() if p.requires_grad))
    assert torch.isfinite(losses["loss"])
    assert len(grads) > 0


if __name__ == "__main__":
    main()
