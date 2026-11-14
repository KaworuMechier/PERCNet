"""Quantile-loss DeepAR (DeepARQ) — shared-objective probabilistic baseline
=========================================================================

DeepAR's autoregressive LSTM and covariate layout, but the Gaussian/Student-t
likelihood head is replaced by a **9-quantile head trained with the same
raw-scale pinball objective as TFT/PERC-Net**.  This isolates the contribution
of the *training target* from the contribution of the architecture.

Contract
--------
* ``teacher_forward(x, y)`` -> dict(quantiles (B,H,C,K), mu (B,H,C))
* ``forward(x)``            -> same shapes; autoregressive rollout feeding the
                              predictive **median** back as y_{t-1}
* ``compute_loss(out, y, scale3)`` -> mean pinball loss in physical units

The median-feedback rollout is deliberate: it is the cheap, deterministic
autoregressive analogue of TFT's non-autoregressive decoder.  It is a quantile
forecast conditional on the median path; it does **not** propagate future
target uncertainty through the recursion.  This is documented in the
supplement and is the same convention used by the compact quantile baselines.

Everything else (past/future covariates, series embedding, window-scale
feature, teacher forcing) is inherited from ``models_baselines.deepar`` so the
data/covariate protocol is identical.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .deepar import Model as DeepARModel


class Model(DeepARModel):
    """DeepAR with a K-quantile head and pinball loss."""

    def __init__(self, config):
        super().__init__(config)
        self.K = int(getattr(config, 'n_quantiles', 9))
        self.taus = torch.linspace(0.05, 0.95, self.K)
        # One K-vector per series step (the LSTM is unrolled per series, so
        # the head is shared across series and outputs K values, not C*K).
        self.head = nn.Linear(self.hidden, self.K)
        nn.init.xavier_uniform_(self.head.weight)
        with torch.no_grad():
            self.head.bias.zero_()

    # ─────────────────────────────────────────────────────────────────────
    # Teacher-forced path (training)
    # ─────────────────────────────────────────────────────────────────────
    def teacher_forward(self, x, y):
        """Quantiles for every horizon step, teacher forced.

        x: (B, P(+H), enc_in); y: (B, H, C), normalised targets.
        """
        b, h, nf = y.shape
        p = self.seq_len
        past = self._past_inputs(x)                         # (B, nf, P, d)
        y_prev = torch.cat([x[:, p - 1:p, :nf], y[:, :-1]], dim=1)
        fut = self._future_inputs(x, y_prev)                # (B, nf, H, d)
        seq = torch.cat([past, fut], dim=2)                 # (B, nf, P+H, d)
        out, _ = self.lstm(seq.reshape(b * nf, p + h, self.input_dim))
        raw = self.head(out[:, p:, :]).view(b, nf, h, self.K)
        q = torch.sort(raw.permute(0, 2, 1, 3), dim=-1).values
        return {'quantiles': q, 'mu': q[..., self.K // 2]}

    # ─────────────────────────────────────────────────────────────────────
    # Deterministic autoregressive rollout (validation / test quantiles)
    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x):
        b, nf, p = x.shape[0], self.n_series, self.seq_len
        h = self.pred_len
        past = self._past_inputs(x).reshape(b * nf, p, self.input_dim)
        _, (state_h, state_c) = self.lstm(past)

        emb = self.series_emb(self.series_ids)
        emb = emb.view(1, nf, 1, -1).expand(b, -1, 1, -1).reshape(
            b * nf, 1, self.emb_dim)
        cov_all = self._future_covariates(x)
        cov_all = cov_all.unsqueeze(1).expand(-1, nf, -1, -1).reshape(
            b * nf, h, self.n_cov)
        ws = self._window_scale(x)
        ws_t = ws.reshape(b * nf, 1) if ws is not None else None
        y_prev = x[:, p - 1, :nf].reshape(b * nf, 1)

        outs = []
        for t in range(h):
            step = torch.cat([y_prev, cov_all[:, t, :], emb.squeeze(1)],
                             dim=-1)
            if ws_t is not None:
                step = torch.cat([step, ws_t], dim=-1)
            out, (state_h, state_c) = self.lstm(step.unsqueeze(1),
                                                (state_h, state_c))
            q = self.head(out.squeeze(1)).view(b, nf, self.K)
            q = torch.sort(q, dim=-1).values
            y_prev = q[..., self.K // 2].reshape(b * nf, 1)
            outs.append(q.view(b, 1, nf, self.K))
        q = torch.cat(outs, dim=1)
        return {'quantiles': q, 'mu': q[..., self.K // 2]}

    # ─────────────────────────────────────────────────────────────────────
    # Shared pinball objective (identical to TFT / PERC-Net)
    # ─────────────────────────────────────────────────────────────────────
    def compute_loss(self, out, y_true, scale3):
        nf = self.n_series
        q = out['quantiles']                                # (B,H,nf,K)
        q_raw = q * scale3.view(1, 1, nf, 1)
        y_raw = y_true * scale3.view(1, 1, nf)
        u = y_raw.unsqueeze(-1) - q_raw
        taus = self.taus.to(u.device).view(1, 1, 1, -1)
        return torch.maximum(taus * u, (taus - 1.0) * u).mean()

    # DeepAR's sampling path is a Gaussian/Student-t likelihood path; it is
    # invalid for this head.  Fail loudly instead of silently reshaping.
    def sample(self, *args, **kwargs):
        raise NotImplementedError(
            'DeepARQ is evaluated from its quantile head; use forward() and '
            'read out["quantiles"] rather than sample().'
        )
