import torch

from baranet.config import SgrConfig
from baranet.models.graph import (
    SuperpixelGraphRefinement,
    build_graph,
    relabel,
    slic_map,
    spatial_adjacency,
)


def _block_map(size: int = 8, blocks: int = 4) -> torch.Tensor:
    step = size // blocks
    ids = torch.arange(blocks * blocks).reshape(blocks, blocks)
    return ids.repeat_interleave(step, 0).repeat_interleave(step, 1)


def test_relabel_is_contiguous():
    labels = torch.tensor([[5, 5, 9], [9, 20, 20]])
    out = relabel(labels)
    assert sorted(out.unique().tolist()) == list(range(out.unique().numel()))


def test_slic_map_is_contiguous_and_covers_pixels():
    cfg = SgrConfig()
    cfg.slic.n_segments = 16
    labels = slic_map(torch.randn(3, 32, 32), cfg)
    assert labels.shape == (32, 32)
    assert labels.min() == 0
    assert labels.unique().numel() == int(labels.max()) + 1
    assert int(labels.max()) >= 1


def test_adjacency_pairs_within_patch():
    sp = _block_map()
    pairs, counts, k = spatial_adjacency(sp)
    assert k == 16
    assert pairs.shape[1] == 2
    assert bool((pairs >= 0).all()) and bool((pairs < k).all())
    assert bool((pairs[:, 0] < pairs[:, 1]).all())
    assert bool((counts > 0).all())


def test_build_graph_has_both_directions_and_self_loops():
    sp = _block_map()
    spectral = torch.randn(int(sp.max()) + 1, 3)
    edge_index, edge_attr = build_graph(sp, spectral)
    k = int(sp.max()) + 1
    assert edge_index.shape[0] == 2
    assert edge_attr.shape == (edge_index.shape[1], 2)
    assert torch.isfinite(edge_attr).all()
    loops = (edge_index[0] == edge_index[1]).sum().item()
    assert loops == k
    directed = edge_index[:, edge_index[0] != edge_index[1]]
    reverse = torch.stack([directed[1], directed[0]])
    keys = set(zip(directed[0].tolist(), directed[1].tolist(), strict=False))
    reverse_keys = set(zip(reverse[0].tolist(), reverse[1].tolist(), strict=False))
    assert all((a, b) in keys for a, b in reverse_keys)


def test_sgr_output_shape_batch_isolation_and_gradient():
    sgr = SuperpixelGraphRefinement(SgrConfig()).eval()
    sp = _block_map().unsqueeze(0).repeat(2, 1, 1)
    spectral = torch.randn(2, 3, 8, 8)

    feature = torch.randn(2, 32, 8, 8)
    out = sgr(feature, spectral, sp)
    assert out.shape == (2, 32, 8, 8)
    assert not torch.allclose(out, feature)
    assert torch.isfinite(out).all()
    patch0 = out[0].detach().clone()

    perturbed = feature.clone()
    perturbed[1] += 5.0
    out_perturbed = sgr(perturbed, spectral, sp)
    assert torch.allclose(out_perturbed[0], patch0, atol=1e-6)

    grad_input = feature.clone().requires_grad_(True)
    sgr(grad_input, spectral, sp)[0].sum().backward()
    assert grad_input.grad is not None
    assert grad_input.grad[0].abs().sum() > 0
