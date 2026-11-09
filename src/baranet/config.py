"""Configuration contract for BARA-Net.

Every architecture constant lives in YAML; this module validates it and rejects
unknown or duplicated keys so a typo fails loudly instead of silently training
a different model.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EncoderConfig(_Strict):
    channels: list[int] = Field(default_factory=lambda: [32, 64, 128, 192, 256])
    norm_groups: int = 8


class TemporalConfig(_Strict):
    enabled: bool = True
    levels: list[str] = Field(default_factory=lambda: ["E2", "E3", "E4", "E5"])
    hidden_channels: int = 64
    kernel_size: int = 3


class DecoderConfig(_Strict):
    channels: list[int] = Field(default_factory=lambda: [192, 128, 64, 32])
    align_corners: bool = False
    attention_intermediate_min: int = 16


class SlicConfig(_Strict):
    n_segments: int = 1024
    compactness: float = 10.0
    sigma: float = 1.0
    start_label: int = 0
    enforce_connectivity: bool = True
    convert2lab: bool = False


class Gatv2Config(_Strict):
    layers: int = 2
    first_heads: int = 4
    first_head_dim: int = 16
    second_heads: int = 1
    second_head_dim: int = 64
    dropout: float = 0.1


class SgrConfig(_Strict):
    enabled: bool = True
    node_input_dim: int = 35
    node_feature_dim: int = 64
    edge_feature_dim: int = 2
    slic: SlicConfig = Field(default_factory=SlicConfig)
    gatv2: Gatv2Config = Field(default_factory=Gatv2Config)


class HeadsConfig(_Strict):
    ba_mask: bool = True
    boundary: bool = True
    boundary_hidden_channels: int = 16


class ModelConfig(_Strict):
    name: str = "baranet"
    input_channels: int = 8
    sequence_length: int = 3
    patch_size: int = 256
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    sgr: SgrConfig = Field(default_factory=SgrConfig)
    heads: HeadsConfig = Field(default_factory=HeadsConfig)


class LossConfig(_Strict):
    focal_tversky: float = 0.50
    dice: float = 0.35
    boundary: float = 0.15
    tversky_alpha: float = 0.30
    tversky_beta: float = 0.70
    tversky_gamma: float = 0.75
    epsilon: float = 1.0e-7


class TrainingConfig(_Strict):
    epochs: int = 100
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    optimizer: str = "adamw"
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    warmup_epochs: int = 5
    scheduler: str = "cosine"
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    early_stopping_patience: int = 15
    seeds: list[int] = Field(default_factory=lambda: [17, 42, 2026])
    loss: LossConfig = Field(default_factory=LossConfig)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    seen: set = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ValueError(f"duplicate YAML key: {key!r}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.load(fh, Loader=_UniqueKeyLoader)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data


def load_model_config(path: str | Path) -> ModelConfig:
    data = load_yaml(path)
    model = data.pop("model", {})
    if not isinstance(model, dict):
        raise ValueError(f"{path}: 'model' must be a mapping")
    return ModelConfig(**{**model, **data})


def load_train_config(path: str | Path) -> TrainingConfig:
    data = load_yaml(path)
    training = data.pop("training", {})
    if not isinstance(training, dict):
        raise ValueError(f"{path}: 'training' must be a mapping")
    return TrainingConfig(**{**training, **data})
