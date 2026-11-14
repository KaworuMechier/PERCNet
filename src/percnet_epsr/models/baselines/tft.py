"""Compact Temporal Fusion Transformer (TFT) — probabilistic baseline
======================================================================

A parameter-budgeted (≈34K) TFT for the Applied Energy comparison table.
The core TFT components are present:

  * lightweight variable-selection networks (softmax gate over variables)
  * LSTM encoder / decoder over selected past / future features
  * interpretable multi-head attention (decoder → encoder)
  * static enrichment + position-wise GRNs
  * quantile head (tau = 0.05 … 0.95, K = 9)

Differences from the original TFT (Limm et al., IJF 2021), documented for
transparency:
  * the VSN is a per-step softmax gate over the projected variables rather
    than the full per-variable GRN + selection network stack (parameter
    budget 20K–50K);
  * the decoder is non-autoregressive: the three target placeholders are the
    last observed target repeated over the horizon, while the future
    covariates drive the horizon.  This avoids exposure bias and keeps a
    single forward pass; it is a standard multi-horizon variant.

Input layout is the shared ASU 20-channel window from ``percnet_epsr.data``.
The decoder uses the known calendar only; future target values are never
read.

Contract: ``Model(config)``; ``forward(x) -> dict(mu, quantiles)``;
``compute_loss(out, y_true, scale3)`` → raw-scale pinball + monotonicity
(via sorting).  This keeps it compatible with the shared runner.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

COV_IDX = (3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19)
CAL_SLICE = slice(12, 16)
CAL_POS = slice(7, 11)


class GRN(nn.Module):
    """Gated residual network (TFT eq. 3-4, residual + LayerNorm)."""

    def __init__(self, hidden, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        return self.norm(x + h)


class VariableSelection(nn.Module):
    """Lightweight VSN: softmax gate over variables, then weighted sum."""

    def __init__(self, n_vars, hidden):
        super().__init__()
        self.n_vars = n_vars
        self.gate = nn.Linear(n_vars * hidden, n_vars)

    def forward(self, x):
        # x: (B, T, n_vars, hidden)
        b, t, v, h = x.shape
        w = torch.softmax(self.gate(x.reshape(b * t, v * h)), dim=-1)
        return (x * w.view(b, t, v, 1)).sum(dim=2)


class Model(nn.Module):
    """Compact non-autoregressive TFT with a quantile head."""

    def __init__(self, config):
        super().__init__()
        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.enc_in = int(config.enc_in)
        self.n_series = int(getattr(config, 'c_out', 3))
        self.hidden = int(getattr(config, 'hidden', 24))
        self.n_heads = int(getattr(config, 'n_heads', 2))
        self.dropout = float(getattr(config, 'dropout', 0.1))
        self.K = int(getattr(config, 'n_quantiles', 9))
        self.taus = torch.linspace(0.05, 0.95, self.K)
        self.use_calendar = bool(getattr(config, 'use_calendar', True))
        self.cov_idx = tuple(getattr(config, 'cov_idx', COV_IDX))
        self.n_cov = len(self.cov_idx)

        self.n_past_vars = self.n_series + self.n_cov      # 18
        self.n_fut_vars = self.n_cov + self.n_series       # 18

        # per-variable 1x1 projections (grouped Conv1d)
        self.past_proj = nn.Conv1d(
            self.n_past_vars, self.n_past_vars * self.hidden,
            kernel_size=1, groups=self.n_past_vars)
        self.fut_proj = nn.Conv1d(
            self.n_fut_vars, self.n_fut_vars * self.hidden,
            kernel_size=1, groups=self.n_fut_vars)
        self.past_vsn = VariableSelection(self.n_past_vars, self.hidden)
        self.fut_vsn = VariableSelection(self.n_fut_vars, self.hidden)

        self.enc = nn.LSTM(self.hidden, self.hidden, num_layers=1,
                           batch_first=True)
        self.dec = nn.LSTM(self.hidden, self.hidden, num_layers=1,
                           batch_first=True)
        self.attn = nn.MultiheadAttention(self.hidden, self.n_heads,
                                          dropout=self.dropout,
                                          batch_first=True)
        self.attn_norm = nn.LayerNorm(self.hidden)
        self.static_proj = nn.Linear(self.hidden, self.hidden)
        self.static_grn = GRN(self.hidden, self.dropout)
        self.pos_grn = GRN(self.hidden, self.dropout)
        self.quantile_head = nn.Linear(self.hidden, self.n_series * self.K)

    # ─────────────────────────────────────────────────────────────────────
    # Input assembly
    # ─────────────────────────────────────────────────────────────────────
    def _past_vars(self, x):
        p = self.seq_len
        tgt = x[:, :p, :self.n_series]                     # (B, P, nf)
        cov = x[:, :p, :][:, :, self.cov_idx]              # (B, P, n_cov)
        return torch.cat([tgt, cov], dim=-1)               # (B, P, Vp)

    def _future_calendar(self, x):
        """Known future calendar (B, H, 4); same convention as DeepAR."""
        last = x[:, self.seq_len - 1, CAL_SLICE]
        hod = torch.atan2(last[:, 0], last[:, 1]) / (2 * math.pi) * 24.0
        dow = torch.atan2(last[:, 2], last[:, 3]) / (2 * math.pi) * 7.0
        hod = torch.remainder(torch.round(hod), 24.0)
        dow = torch.remainder(torch.round(dow), 7.0)
        steps = torch.arange(1, self.pred_len + 1,
                             device=x.device, dtype=x.dtype)
        hours = hod[:, None] + steps[None, :]
        hh = torch.remainder(hours, 24.0)
        ww = torch.remainder(dow[:, None] + torch.floor(hours / 24.0), 7.0)
        ah = hh / 24.0 * 2.0 * math.pi
        aw = ww / 7.0 * 2.0 * math.pi
        return torch.stack([torch.sin(ah), torch.cos(ah),
                            torch.sin(aw), torch.cos(aw)], dim=-1)

    def _future_covariates(self, x):
        b, h = x.shape[0], self.pred_len
        cov = x.new_zeros(b, h, self.n_cov)
        if self.use_calendar and self.enc_in >= CAL_SLICE.stop:
            cov[..., CAL_POS] = self._future_calendar(x)
        return cov

    def _future_vars(self, x):
        p = self.seq_len
        last = x[:, p - 1, :self.n_series].unsqueeze(1).expand(
            -1, self.pred_len, -1)                         # (B, H, nf)
        cov = self._future_covariates(x)                   # (B, H, n_cov)
        return torch.cat([last, cov], dim=-1)              # (B, H, Vf)

    @staticmethod
    def _project(conv, z, n_vars, hidden):
        """(B, T, V) → (B, T, V, hidden) per-variable projection."""
        b, t, _ = z.shape
        out = conv(z.transpose(1, 2))                      # (B, V*h, T)
        return out.view(b, n_vars, hidden, t).permute(0, 3, 1, 2)

    # ─────────────────────────────────────────────────────────────────────
    # Forward / loss
    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x):
        b = x.shape[0]
        past = self._past_vars(x)                          # (B, P, Vp)
        fut = self._future_vars(x)                         # (B, H, Vf)
        pp = self._project(self.past_proj, past, self.n_past_vars,
                           self.hidden)
        fp = self._project(self.fut_proj, fut, self.n_fut_vars,
                           self.hidden)
        psel = self.past_vsn(pp)                           # (B, P, hidden)
        fsel = self.fut_vsn(fp)                            # (B, H, hidden)

        enc_out, (state_h, state_c) = self.enc(psel)
        dec_out, _ = self.dec(fsel, (state_h, state_c))
        attn_out, _ = self.attn(dec_out, enc_out, enc_out,
                                need_weights=False)
        dec_out = self.attn_norm(dec_out + attn_out)
        static = self.static_proj(enc_out[:, -1:, :])      # (B, 1, hidden)
        dec_out = self.static_grn(dec_out + static)
        dec_out = self.pos_grn(dec_out)

        q = self.quantile_head(dec_out).view(
            b, self.pred_len, self.n_series, self.K)
        # Last-value anchor (no extra parameters): the quantile head predicts
        # deviations from the last observed load, as in the other load models.
        anchor = x[:, self.seq_len - 1, :self.n_series].unsqueeze(1)
        q = q + anchor.unsqueeze(-1)
        # enforce monotone quantiles (rearrangement; gradients flow through)
        q = torch.sort(q, dim=-1).values
        return {'mu': q[..., self.K // 2], 'quantiles': q}

    def compute_loss(self, out, y_true, scale3):
        """Raw-scale pinball loss over the 9-tau grid."""
        nf = self.n_series
        q = out['quantiles']                                # (B,H,nf,K)
        q_raw = q * scale3.view(1, 1, nf, 1)
        y_raw = y_true * scale3.view(1, 1, nf)
        u = y_raw.unsqueeze(-1) - q_raw
        taus = self.taus.to(u.device).view(1, 1, 1, -1)
        return torch.maximum(taus * u, (taus - 1.0) * u).mean()
