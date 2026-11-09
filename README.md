# BARA-Net

**Burned-Area Region-Aware Network** — temporal multispectral burned-area segmentation on
[TS-SatFire](https://github.com/zhaoyutim/TS-SatFire) VIIRS data.

This repository is the **model + training-engine source code** (no dataset, no trained weights).
Clone it, put your TS-SatFire data wherever you like, and drive training from your own notebook.
The full specification lives in [`prd.md`](prd.md).

```
x [B, 3, 8, 256, 256]  →  shared residual encoder  →  ConvGRU @ E2–E5
                       →  attention U-Net decoder  →  superpixel graph refinement (SLIC + GATv2)
                       →  BA mask head + boundary head
```

## Install

```bash
git clone https://github.com/AneKazek/baranet_wildfire-segmentation.git
cd baranet_wildfire-segmentation
uv sync                 # creates .venv with Python 3.12 from uv.lock
uv run pytest           # 31 fast CPU tests
uv run python scripts/smoke_test.py
```

For CUDA, install the matching torch wheel once (the lock defaults to the PyPI build):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Quick start

```python
import torch
from baranet import (
    CombinedLoss, LossConfig, TrainingConfig, Trainer,
    build_model, load_model_config, morphological_gradient,
)

model = build_model(load_model_config("configs/model/baranet.yaml"))

x = torch.randn(2, 3, 8, 256, 256)        # 3 daily frames, 8 logical VIIRS channels
out = model(x)                            # sp_map=None -> SLIC is computed on the fly
out["ba_logits"].shape, out["boundary_logits"].shape   # both [2, 1, 256, 256]

target = (torch.rand(2, 1, 256, 256) > 0.8).float()
losses = CombinedLoss(LossConfig())(
    out["ba_logits"], target, out["boundary_logits"], morphological_gradient(target)
)
losses["loss"].backward()
```

Training (your `DataLoader` is the only thing you write):

```python
trainer = Trainer(
    model, train_loader, val_loader,
    cfg=TrainingConfig(),
    out_dir="artifacts/experiments/baranet_full_seed17",
    seed=17,
)
result = trainer.fit()   # best/latest checkpoints, CSV logs, config, threshold
```

Load it back with `baranet.load_checkpoint(path, model)`.

## Batch contract

| key | shape | required |
|---|---|---|
| `inputs` | `[B, 3, 8, H, W]` float | yes |
| `ba_target` | `[B, 1, H, W]` float (0/1) | yes |
| `sp_map` | `[B, H, W]` int64 SLIC labels | no — computed inside if absent |
| `boundary_target` | `[B, 1, H, W]` float | no |
| `valid_mask` | `[B, 1, H, W]` bool | no |
| `fire_id` | list of `B` strings | no (needed for fire-level macro F1) |

`H` and `W` must be multiples of 16 (256 in the paper config). Use `patch_size: 64` for tests.
Labels are never model inputs; masking happens in the loss.

## Layout

```
configs/model/      baranet, unet_convgru (B2), unet_frame (B1)
configs/train/      optimizer, schedule, seeds, loss weights
src/baranet/
  config.py         strict YAML schema (unknown/duplicate keys fail)
  models/           encoder, ConvGRU, attention decoder, SGR graph, heads, baselines
  losses/           Focal-Tversky + Dice + Boundary (valid-mask aware)
  metrics/          fire-macro / pixel-micro F1, IoU, precision, recall, boundary F1, threshold selection
  engine/           Trainer (AMP, grad-accum, warmup+cosine, early stop), checkpoints, full-scene inference
tests/              shapes, temporal causality, graph isolation, losses, metrics, inference, trainer
scripts/smoke_test.py
```

Baselines and ablations are config switches, not forks: `temporal.enabled`, `sgr.enabled`,
`heads.boundary` (see `configs/model/*.yaml`).

## Notes

- SLIC/edge topology are recomputed per forward pass unless you pass a cached `sp_map`.
- Graph batching is per patch and block-diagonal, so patches never exchange messages.
- Full-scene inference (`predict_scene`) uses 256/192 sliding windows with center-weighted blending
  and asserts that every pixel is covered.
- Reproducibility: `uv.lock`, seeded runs, config + git commit + environment saved per experiment.

## Citation

```bibtex
@misc{baranet2026,
  title  = {BARA-Net: Burned-Area Region-Aware Network for Temporal Multispectral Wildfire Segmentation},
  author = {AneKazek},
  year   = {2026},
  url    = {https://github.com/AneKazek/baranet_wildfire-segmentation}
}
```

Dataset: Zhao, Gerard & Ban (2025), *TS-SatFire*, Scientific Data,
https://doi.org/10.1038/s41597-025-06271-3.
