import pytest

from baranet import ModelConfig


@pytest.fixture
def small_cfg() -> ModelConfig:
    cfg = ModelConfig()
    cfg.patch_size = 64
    cfg.sgr.slic.n_segments = 64
    return cfg
