import torch

from baranet import build_model
from baranet.models.encoder import SharedResidualEncoder


def test_encoder_shapes():
    encoder = SharedResidualEncoder(8, [32, 64, 128, 192, 256])
    feats = encoder(torch.randn(2, 8, 64, 64))
    assert [f.shape[1] for f in feats] == [32, 64, 128, 192, 256]
    assert [f.shape[-1] for f in feats] == [64, 32, 16, 8, 4]


def test_full_model_output_shapes(small_cfg):
    model = build_model(small_cfg)
    out = model(torch.randn(2, 3, 8, 64, 64))
    assert out["ba_logits"].shape == (2, 1, 64, 64)
    assert out["boundary_logits"].shape == (2, 1, 64, 64)
    assert torch.isfinite(out["ba_logits"]).all()
    assert torch.isfinite(out["boundary_logits"]).all()


def test_decoder_skip_shapes(small_cfg):
    model = build_model(small_cfg)
    d1 = model.decoder(model.encoder(torch.randn(2, 8, 64, 64)))
    assert d1.shape == (2, 32, 64, 64)


def test_baselines_have_no_boundary_head(small_cfg):
    small_cfg.name = "unet_frame"
    b1 = build_model(small_cfg)
    out = b1(torch.randn(2, 3, 8, 64, 64))
    assert out["boundary_logits"] is None
    assert out["ba_logits"].shape == (2, 1, 64, 64)

    small_cfg.name = "unet_convgru"
    b2 = build_model(small_cfg)
    assert b2.temporal is not None and b2.sgr is None
    assert b2(torch.randn(2, 3, 8, 64, 64))["boundary_logits"] is None
