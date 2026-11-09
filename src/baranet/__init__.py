"""BARA-Net: temporal multispectral burned-area segmentation.

Quick start::

    from baranet import build_model, load_train_config, load_model_config, Trainer

    model = build_model(load_model_config("configs/model/baranet.yaml"))
    out = model(torch.randn(2, 3, 8, 256, 256))
    out["ba_logits"].shape  # [2, 1, 256, 256]
"""

from .config import (
    LossConfig,
    ModelConfig,
    TrainingConfig,
    load_model_config,
    load_train_config,
    load_yaml,
)
from .engine import (
    Trainer,
    load_checkpoint,
    predict_scene,
    save_checkpoint,
    seed_everything,
)
from .losses import CombinedLoss, dice_loss, focal_tversky_loss, morphological_gradient
from .metrics import SegmentationMetrics, boundary_f1, sample_counts, select_threshold
from .models import BaraNet, build_model

__all__ = [
    "BaraNet",
    "CombinedLoss",
    "LossConfig",
    "ModelConfig",
    "SegmentationMetrics",
    "Trainer",
    "TrainingConfig",
    "boundary_f1",
    "build_model",
    "dice_loss",
    "focal_tversky_loss",
    "load_checkpoint",
    "load_model_config",
    "load_train_config",
    "load_yaml",
    "morphological_gradient",
    "predict_scene",
    "sample_counts",
    "save_checkpoint",
    "seed_everything",
    "select_threshold",
]
