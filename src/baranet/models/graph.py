"""Superpixel Graph Refinement (SGR): SLIC, adjacency, edge-aware GATv2.

Pure PyTorch: node pooling uses ``index_add_`` and attention uses a scattered
softmax over each destination node, so no torch-geometric / torch-scatter build
is required and everything stays differentiable.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import SgrConfig


def relabel(labels: torch.Tensor) -> torch.Tensor:
    """Map arbitrary integer labels to contiguous 0..K-1."""
    _, inverse = torch.unique(labels.reshape(-1), return_inverse=True)
    return inverse.view_as(labels)


def slic_map(image: torch.Tensor, cfg: SgrConfig) -> torch.Tensor:
    """SLIC on a normalized [3, H, W] latest-date tensor -> [H, W] int64 map.

    Scikit-image is imported lazily so the package imports without it.
    """
    from skimage.segmentation import slic

    arr = image.detach().float().cpu().permute(1, 2, 0).numpy()
    labels = slic(
        arr,
        n_segments=cfg.slic.n_segments,
        compactness=cfg.slic.compactness,
        sigma=cfg.slic.sigma,
        start_label=cfg.slic.start_label,
        enforce_connectivity=cfg.slic.enforce_connectivity,
        convert2lab=cfg.slic.convert2lab,
        channel_axis=-1,
    )
    out = relabel(torch.as_tensor(labels, dtype=torch.long))
    if int(out.max()) < 1:
        raise ValueError("SLIC produced a degenerate map with a single segment")
    return out


def spatial_adjacency(sp_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """4-neighbor superpixel adjacency for one patch.

    Returns (pairs [P, 2] with u < v, shared boundary pixel-pair counts [P], K).
    """
    h, w = sp_map.shape
    horizontal = torch.stack(
        [sp_map[:, :-1].reshape(-1), sp_map[:, 1:].reshape(-1)], dim=1
    )
    vertical = torch.stack(
        [sp_map[:-1, :].reshape(-1), sp_map[1:, :].reshape(-1)], dim=1
    )
    pairs = torch.cat([horizontal, vertical], dim=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    k = int(sp_map.max()) + 1
    if pairs.numel() == 0:
        empty = torch.zeros((0, 2), dtype=torch.long, device=sp_map.device)
        return empty, torch.zeros(0, dtype=torch.long, device=sp_map.device), k

    lo = pairs.min(dim=1).values
    hi = pairs.max(dim=1).values
    key, counts = torch.unique(lo * k + hi, return_counts=True)
    return torch.stack([key // k, key % k], dim=1), counts, k


def segment_mean(feature: torch.Tensor, sp_map: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Differentiable mean pooling of [C, H, W] over superpixels -> [K, C]."""
    channels = feature.shape[0]
    index = sp_map.reshape(-1)
    pooled = torch.zeros(
        num_nodes, channels, dtype=feature.dtype, device=feature.device
    )
    pooled.index_add_(0, index, feature.reshape(channels, -1).transpose(0, 1))
    counts = torch.bincount(index, minlength=num_nodes).clamp_min(1).to(feature.dtype)
    return pooled / counts.unsqueeze(1)


def edge_attributes(
    pairs: torch.Tensor,
    counts: torch.Tensor,
    node_spectral: torch.Tensor,
    areas: torch.Tensor,
) -> torch.Tensor:
    """[P, 2]: spectral distance / sqrt(3) clipped to [0, 2], shared ratio clipped to [0, 1]."""
    u, v = pairs[:, 0], pairs[:, 1]
    spectral = (node_spectral[u] - node_spectral[v]).norm(dim=1) / math.sqrt(3.0)
    shared = counts.to(node_spectral.dtype) / (
        areas[u].to(node_spectral.dtype) * areas[v].to(node_spectral.dtype)
    ).sqrt()
    return torch.stack([spectral.clamp(0.0, 2.0), shared.clamp(0.0, 1.0)], dim=1)


def build_graph(
    sp_map: torch.Tensor, node_spectral: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed edge_index [2, E] and edge_attr [E, 2] for one patch.

    Includes both directions and self-loops (spectral distance 0, shared ratio 1).
    """
    pairs, counts, k = spatial_adjacency(sp_map)
    areas = torch.bincount(sp_map.reshape(-1), minlength=k)
    attrs = edge_attributes(pairs, counts, node_spectral, areas)

    directed = torch.cat([pairs, pairs.flip(1)], dim=0).transpose(0, 1)
    directed_attrs = torch.cat([attrs, attrs], dim=0)
    loops = torch.arange(k, device=sp_map.device)
    loop_edges = torch.stack([loops, loops], dim=0)
    loop_attrs = torch.tensor(
        [[0.0, 1.0]], dtype=attrs.dtype, device=attrs.device
    ).expand(k, 2)
    edge_index = torch.cat([directed, loop_edges], dim=1)
    edge_attr = torch.cat([directed_attrs, loop_attrs], dim=0)
    return edge_index, edge_attr


class GATv2Layer(nn.Module):
    """Edge-aware GATv2: score_ij = a^T LeakyReLU(W_l h_i + W_r h_j + W_e e_ij)."""

    def __init__(
        self,
        in_channels: int,
        head_dim: int,
        heads: int,
        edge_channels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        out = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.lin_l = nn.Linear(in_channels, out, bias=False)
        self.lin_r = nn.Linear(in_channels, out, bias=False)
        self.edge_proj = nn.Linear(edge_channels, out, bias=False)
        self.att = nn.Parameter(torch.empty(heads, head_dim))
        nn.init.xavier_uniform_(self.att)
        self.res = nn.Linear(in_channels, out)
        self.norm = nn.LayerNorm(out)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index
        x_l = self.lin_l(x)
        x_r = self.lin_r(x)
        edge = self.edge_proj(edge_attr)

        msg = (x_l[src] + x_r[dst] + edge).view(-1, self.heads, self.head_dim)
        score = F.leaky_relu((msg * self.att).sum(dim=-1), negative_slope=0.2)

        # per-destination (and per-head) softmax, numerically stable
        nodes, heads, head_dim = x.shape[0], self.heads, self.head_dim
        flat_index = (dst.unsqueeze(1) * heads + torch.arange(heads, device=dst.device)).reshape(-1)
        peak = torch.full(
            (nodes * heads,), -torch.inf, dtype=score.dtype, device=score.device
        ).scatter_reduce(0, flat_index, score.reshape(-1), reduce="amax")
        peak = peak.view(nodes, heads)
        weights = torch.exp(score - peak[dst])
        denom = torch.zeros(nodes, heads, dtype=score.dtype, device=score.device)
        denom = denom.index_add_(0, dst, weights)
        alpha = weights / (denom[dst] + 1e-9)

        value = (x_r[dst] + edge).view(-1, heads, head_dim)
        out = torch.zeros(nodes, heads * head_dim, dtype=value.dtype, device=value.device)
        for head in range(heads):
            out[:, head * head_dim : (head + 1) * head_dim] = torch.zeros(
                nodes, head_dim, dtype=value.dtype, device=value.device
            ).index_add_(
                0,
                dst,
                (
                    alpha[:, head].unsqueeze(1) * value[:, head]
                ).to(dtype=value.dtype)
            )
        return self.dropout(F.gelu(self.norm(self.res(x) + out)))


class SuperpixelGraphRefinement(nn.Module):
    """Node pooling -> 2x edge-aware GATv2 -> broadcast -> residual fusion."""

    def __init__(self, cfg: SgrConfig, feature_channels: int = 32, spectral_channels: int = 3):
        super().__init__()
        self.cfg = cfg
        self.feature_channels = feature_channels
        self.node_mlp = nn.Sequential(
            nn.Linear(cfg.node_input_dim, cfg.node_feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        if cfg.node_input_dim != feature_channels + spectral_channels:
            raise ValueError(
                "sgr.node_input_dim must equal decoder D1 channels + 3 spectral "
                f"channels: {cfg.node_input_dim} != {feature_channels} + {spectral_channels}"
            )

        gat = cfg.gatv2
        layers: list[GATv2Layer] = []
        for i in range(gat.layers):
            head_dim = gat.first_head_dim if i == 0 else gat.second_head_dim
            heads = gat.first_heads if i == 0 else gat.second_heads
            layers.append(
                GATv2Layer(
                    cfg.node_feature_dim, head_dim, heads, cfg.edge_feature_dim, gat.dropout
                )
            )
        self.gat_layers = nn.ModuleList(layers)
        self.region_proj = nn.Conv2d(cfg.node_feature_dim, feature_channels, 1)
        self.fuse = nn.Conv2d(feature_channels * 2, feature_channels, 1)

    def forward(
        self, feature: torch.Tensor, spectral: torch.Tensor, sp_map: torch.Tensor
    ) -> torch.Tensor:
        """feature [B,C,H,W], spectral [B,3,H,W] (latest date, normalized), sp_map [B,H,W]."""
        batch, _, height, width = feature.shape
        node_inputs: list[torch.Tensor] = []
        edge_indices: list[torch.Tensor] = []
        edge_attrs: list[torch.Tensor] = []
        offsets: list[int] = []
        node_count = 0

        for i in range(batch):
            sp = sp_map[i]
            k = int(sp.max()) + 1
            if k < 2:
                raise ValueError(
                    f"degenerate superpixel map at batch index {i}: only {k} segment(s)"
                )
            pooled_feature = segment_mean(feature[i], sp, k)
            pooled_spectral = segment_mean(spectral[i], sp, k)
            node_inputs.append(torch.cat([pooled_feature, pooled_spectral], dim=1))

            edge_index, edge_attr = build_graph(sp, pooled_spectral)
            edge_indices.append(edge_index + node_count)
            edge_attrs.append(edge_attr)
            offsets.append(node_count)
            node_count += k

        z = self.node_mlp(torch.cat(node_inputs, dim=0))
        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)
        for layer in self.gat_layers:
            z = layer(z, edge_index, edge_attr)

        pixel_index = torch.cat(
            [sp_map[i].reshape(-1) + offset for i, offset in enumerate(offsets)]
        )
        region = z[pixel_index].view(batch, height, width, -1).permute(0, 3, 1, 2)
        region = self.region_proj(region)
        return feature + self.fuse(torch.cat([feature, region], dim=1))
