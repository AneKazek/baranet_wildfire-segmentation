from pathlib import Path

import pytest
from pydantic import ValidationError

from baranet import load_model_config, load_train_config
from baranet.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_model_config_matches_prd_defaults():
    cfg = load_model_config(ROOT / "configs/model/baranet.yaml")
    assert cfg.encoder.channels == [32, 64, 128, 192, 256]
    assert cfg.temporal.hidden_channels == 64
    assert cfg.decoder.channels == [192, 128, 64, 32]
    assert cfg.sgr.slic.n_segments == 1024
    assert cfg.sgr.gatv2.first_heads == 4
    assert cfg.heads.boundary is True


def test_train_config_matches_prd_defaults():
    cfg = load_train_config(ROOT / "configs/train/default.yaml")
    assert cfg.learning_rate == 3e-4
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.seeds == [17, 42, 2026]
    assert cfg.loss.focal_tversky == 0.50
    assert cfg.loss.dice == 0.35
    assert cfg.loss.boundary == 0.15


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  name: baranet\n  typo_key: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_model_config(path)


def test_duplicate_key_is_rejected(tmp_path):
    path = tmp_path / "dup.yaml"
    path.write_text("model:\n  name: a\n  name: b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_yaml(path)


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_model_config(ROOT / "configs/model/does_not_exist.yaml")
