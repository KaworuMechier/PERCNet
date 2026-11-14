"""xCPD-style channel-patch graph spectral plugin (approximate).

This is a documented *approximation* of

    "Routing Channel-Patch Dependencies in Time Series Forecasting with
     Graph Spectral Decomposition" (ICLR 2026)

built from the public description, not from the official code or the paper's
exact equations.  The plugin treats non-overlapping forecast patches per
channel as graph nodes, builds a shared graph Fourier basis from a batch-
averaged normalised Laplacian, groups nodes into low/mid/high spectral bands,
routes them through band-specific experts with a soft MoE gate, and adds a
gated dual-path (spectral + channel-independent linear) residual to the
backbone forecast.

It is deliberately a separate model file.  The backbone is frozen by default;
the plugin is identity at initialisation (zero gate and zero output layer), so
on/off comparisons are clean.  Do not present it as an exact reproduction.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelPatchGraph(nn.Module):
    """Node features + shared spectral basis from a channel-patch graph."""

    def __init__(self, patch_len: int, hidden: int, knn: int = 2,
                 stat_dim: int = 3):
        super().__init__()
        self.patch_len = int(patch_len)
        self.knn = int(knn)
        self.node_proj = nn.Linear(self.patch_len + stat_dim, hidden)

    def forward(self, mu: torch.Tensor):
        """mu: (B,H,C) -> h (B,V,hidden), U (V,V), band masks (3,V)."""
        b, h, c = mu.shape
        p = self.patch_len
        if h % p != 0:
            raise ValueError(f"pred_len {h} not divisible by patch_len {p}")
        n = h // p
        v = c * n

        # (B, C, N, p) -> (B, V, p)
        patches = mu.transpose(1, 2).reshape(b, c, n, p).reshape(b, v, p)
        stats = torch.stack([
            patches.mean(dim=-1),
            (patches[..., -1] - patches[..., 0]) / max(p - 1, 1),
            patches.abs().mean(dim=-1),
        ], dim=-1)                                      # (B,V,3)
        raw = torch.cat([patches, stats], dim=-1)       # (B,V,p+3)
        raw = (raw - raw.mean(dim=1, keepdim=True)) / (
            raw.std(dim=1, keepdim=True) + 1e-6)
        h = self.node_proj(raw)                         # (B,V,d)

        # cosine kNN graph, sym, self loops
        hn = F.normalize(h, dim=-1)
        sim = hn @ hn.transpose(1, 2)                   # (B,V,V)
        k = min(self.knn, v - 1)
        idx = sim.topk(k + 1, dim=-1).indices           # includes self
        a = torch.zeros_like(sim)
        a.scatter_(-1, idx, 1.0)
        a = torch.clamp(a + a.transpose(1, 2), max=1.0)
        eye = torch.eye(v, device=mu.device, dtype=a.dtype).unsqueeze(0)
        a = torch.clamp(a + eye, max=1.0)

        deg = a.sum(dim=-1).clamp(min=1.0)
        d_inv = deg.pow(-0.5)
        lap = eye - d_inv.unsqueeze(-1) * a * d_inv.unsqueeze(-2)
        lap = lap.mean(dim=0)

        # Shared basis.  Detached: the basis is a fixed coordinate frame for
        # this batch, as in the plugin description.
        _, u = torch.linalg.eigh(lap)
        u = u.detach()

        # Three contiguous spectral bands (low/mid/high eigenvalues).
        cuts = torch.tensor_split(torch.arange(v, device=mu.device), 3)
        masks = torch.zeros(3, v, device=mu.device, dtype=h.dtype)
        for i, ix in enumerate(cuts):
            if len(ix):
                masks[i, ix] = 1.0
        return h, u, masks


class BandExpert(nn.Module):
    """Spectral diagonal filter + CI linear path for one frequency band."""

    def __init__(self, num_nodes: int, hidden: int):
        super().__init__()
        self.node_diag = nn.Parameter(torch.ones(num_nodes))
        self.ci = nn.Linear(hidden, hidden)
        self.spec = nn.Linear(hidden, hidden)

    def forward(self, h: torch.Tensor, h_spec: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        band = h_spec * mask.view(1, -1, 1)
        band = band * self.node_diag.view(1, -1, 1)
        return self.spec(F.gelu(band)) + self.ci(h)


class XCPDPlugin(nn.Module):
    def __init__(self, pred_len: int, n_series: int, patch_len: int = 6,
                 hidden: int = 16, knn: int = 2, router_hidden: int = 16,
                 gate_init: float = 1.0, route_threshold: float = 0.8):
        super().__init__()
        self.pred_len = int(pred_len)
        self.n_series = int(n_series)
        self.patch_len = int(patch_len)
        self.route_threshold = float(route_threshold)
        n = self.pred_len // self.patch_len
        self.num_nodes = self.n_series * n

        self.graph = ChannelPatchGraph(self.patch_len, hidden, knn)
        self.experts = nn.ModuleList([
            BandExpert(self.num_nodes, hidden) for _ in range(3)])
        self.router = nn.Sequential(
            nn.Linear(3, router_hidden), nn.GELU(),
            nn.Linear(router_hidden, 3))
        self.out = nn.Linear(hidden, self.patch_len)
        self.gate = nn.Parameter(torch.full((1, 1, self.n_series),
                                            float(gate_init)))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # Identity at init comes from the zero-init `out` alone.  `gate` must
        # NOT be zero as well: corr = tanh(gate) * out(fused) makes the gate
        # and the output layer each the other's only gradient path, so a
        # double-zero init has exactly zero grad everywhere and the plugin
        # never trains (see scripts/smoke_3090.py gradient check).

    def _route(self, energies: torch.Tensor) -> torch.Tensor:
        logits = self.router(energies)
        pi = torch.softmax(logits, dim=-1)
        if self.training:
            return pi
        # Inference: smallest set of experts whose cumulative gate exceeds
        # the threshold (the described variable-K routing).
        order = torch.argsort(pi, dim=-1, descending=True)
        sorted_pi = torch.gather(pi, -1, order)
        cum = sorted_pi.cumsum(dim=-1)
        keep_sorted = cum < self.route_threshold
        keep_sorted[..., 0] = True
        keep = torch.zeros_like(pi, dtype=torch.bool)
        keep.scatter_(-1, order, keep_sorted)
        pi = pi * keep
        return pi / (pi.sum(dim=-1, keepdim=True) + 1e-8)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        h, u, masks = self.graph(mu)
        h_spec = torch.einsum("ij,bjd->bid", u.t(), h)

        outs, energies = [], []
        for i, expert in enumerate(self.experts):
            outs.append(expert(h, h_spec, masks[i]))
            band = h_spec * masks[i].view(1, -1, 1)
            energies.append(band.pow(2).mean(dim=(1, 2)))
        energies = torch.stack(energies, dim=-1)        # (B,3)
        pi = self._route(energies).unsqueeze(-1)        # (B,3,1)
        fused = sum(pi[:, i].unsqueeze(-1) * outs[i] for i in range(3))
        corr = self.out(fused)                          # (B,V,p)
        b = mu.shape[0]
        corr = corr.view(b, self.n_series,
                         self.pred_len // self.patch_len, self.patch_len)
        corr = corr.reshape(b, self.n_series, self.pred_len).transpose(1, 2)
        return torch.tanh(self.gate) * corr             # (B,H,C)


class Model(nn.Module):
    """Frozen-backbone wrapper: v2 forecast + xCPD residual refinement."""

    def __init__(self, config, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.plugin = XCPDPlugin(
            pred_len=int(config.pred_len),
            n_series=int(getattr(config, "n_free", 3)),
            patch_len=int(getattr(config, "patch_len", 6)),
            hidden=int(getattr(config, "hidden", 16)),
            knn=int(getattr(config, "knn", 2)),
            router_hidden=int(getattr(config, "router_hidden", 16)),
            gate_init=float(getattr(config, "gate_init", 1.0)))
        self.residual_weight = float(getattr(config, "residual_weight", 0.0))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()          # frozen backbone stays deterministic
        return self

    def forward(self, x: torch.Tensor):
        base = self.backbone(x)
        mu = base["mu"]
        corr = self.plugin(mu)
        mu_ref = mu + corr
        out = dict(base)
        out["mu"] = mu_ref
        if "quantiles" in base:
            out["quantiles"] = base["quantiles"] + (mu_ref - mu).unsqueeze(-1)
        if getattr(self.backbone, "physics_identity", False):
            ls = self.backbone.load_scale.view(1, 1, -1)
            mu_raw = mu_ref * ls
            out["cmbbtu"] = (mu_raw * (
                self.backbone.cel_coef * self.backbone.load_scale)
                .view(1, 1, -1)).sum(dim=-1, keepdim=True)
        return out

    def compute_loss(self, out: dict, y_true: torch.Tensor,
                     scale3: torch.Tensor) -> torch.Tensor:
        loss = self.backbone.compute_loss(out, y_true, scale3)
        if self.residual_weight > 0.0:
            y_raw = y_true * scale3
            loss = loss + self.residual_weight * (
                (out["mu"] * scale3 - y_raw).abs().mean())
        return loss
