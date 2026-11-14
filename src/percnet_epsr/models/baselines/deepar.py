"""DeepAR (Salinas et al., IJF 2020) — compact multi-series probabilistic baseline
==================================================================================

The canonical autoregressive probabilistic forecaster, sized to the paper's
parameter budget (≈33K params at the default width):

    past 168 h ──► 2-layer LSTM ──► Gaussian / Student-t head (μ, log σ)
    decoder step t: [y_{t-1}, covariates_t, series embedding,
                     log window scale] ──► LSTM
    teacher forcing during training; autoregressive sampling at test time

Two configurable standard DeepAR ingredients (both on by default in the
"strong" recipe):
  * ``use_window_scale`` — per-window log mean |load| as an input feature
  * ``likelihood`` — 'gaussian' or 'student_t' (heavy-tailed residuals)

Input layout is the shared ASU 20-channel window built by ``util.DataLoaderS``:

    indices  0- 2  loads (KW, CHWTON, HTMBTU)  ← the only targets
    indices  3- 9  weather + DOY-COS
    indices 10-11  CEL / GHG derived           ← excluded (target-derived)
    indices 12-15  calendar sin/cos(HOD), sin/cos(DOW)
    indices 16-19  rolling 24/72 h TEMP/WBT means

Covariates fed to the network are ``COV_IDX`` (15 channels).  The decoder's
weather/rolling slots are zeroed and only the known calendar is used; future
target values are never read: teacher forcing uses ``y_{t-1}`` from the target
tensor during training, and the model's own sample during decoding.

Probabilistic output
--------------------
``sample(x, n_samples)`` returns (B, H, n_series, S) draws from the predictive
distribution — the Monte-Carlo ensemble used by ``train_deepar.py`` for CRPS
and pinball.  ``forward(x)`` is a deterministic open-loop mean path (μ fed
back), kept for the shared evaluate()/MAE helper; it is *not* the predictive
median and is not used for the probabilistic metrics.

The model returns dicts so it is compatible with the ``Model(config)``
contract used by the rest of the code base, but it is trained by
``train_deepar.py`` because DeepAR's teacher forcing needs the target window.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Covariates: weather+DOY (3-9), calendar (12-15), rolling weather (16-19).
COV_IDX = (3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19)
CAL_SLICE = slice(12, 16)          # calendar columns inside the 20-ch window
CAL_POS = slice(7, 11)             # their position inside the COV_IDX vector


class Model(nn.Module):
    """DeepAR with a Gaussian likelihood and a learned series embedding."""

    def __init__(self, config):
        super().__init__()
        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.enc_in = int(config.enc_in)
        self.use_calendar = bool(getattr(config, 'use_calendar', True))

        self.n_series = int(getattr(config, 'c_out', 3))
        self.hidden = int(getattr(config, 'hidden', 48))
        self.n_layers = int(getattr(config, 'n_layers', 2))
        self.emb_dim = int(getattr(config, 'emb_dim', 8))
        self.dropout = float(getattr(config, 'dropout', 0.1))
        self.K = int(getattr(config, 'n_quantiles', 9))
        self.loss_scale_weighted = bool(getattr(config, 'loss_scale_weighted',
                                                True))
        # Standard DeepAR scale handling: feed log(mean |window|) per series
        # as an input feature (the loader already removes the global scale).
        self.use_window_scale = bool(getattr(config, 'use_window_scale', True))
        # 'gaussian' | 'student_t' (heavy-tailed load residuals)
        self.likelihood = str(getattr(config, 'likelihood', 'gaussian')).lower()
        if self.likelihood not in ('gaussian', 'student_t'):
            raise ValueError(f'unknown likelihood {self.likelihood!r}')
        # σ floor in normalised units.  Student-t has almost no curvature for
        # large |z|, so without a floor the scale head collapses to ~0 and the
        # autoregressive rollout explodes (observed on the 3090).
        _floor_cfg = getattr(config, 'sigma_floor', None)
        _default_floor = 0.02 if self.likelihood == 'student_t' else 1e-3
        self.sigma_floor = float(_default_floor if _floor_cfg is None
                                 else _floor_cfg)

        self.cov_idx = tuple(getattr(config, 'cov_idx', COV_IDX))
        self.n_cov = len(self.cov_idx)
        self.input_dim = (1 + self.n_cov + self.emb_dim
                          + (1 if self.use_window_scale else 0))

        self.series_emb = nn.Embedding(self.n_series, self.emb_dim)
        nn.init.normal_(self.series_emb.weight, mean=0.0, std=0.02)

        self.lstm = nn.LSTM(
            self.input_dim, self.hidden, num_layers=self.n_layers,
            batch_first=True,
            dropout=self.dropout if self.n_layers > 1 else 0.0)
        self.head = nn.Linear(self.hidden, 2)
        nn.init.xavier_uniform_(self.head.weight)
        with torch.no_grad():
            self.head.bias[0] = 0.0
            self.head.bias[1] = -1.0        # initial σ ≈ softplus(-1) ≈ 0.31

        if self.likelihood == 'student_t':
            # per-series degrees of freedom; floor at 4 keeps the variance
            # finite and the tails less extreme than df→2.
            self.df_raw = nn.Parameter(torch.zeros(self.n_series))

        self.register_buffer('series_ids',
                             torch.arange(self.n_series, dtype=torch.long))

    # ─────────────────────────────────────────────────────────────────────
    # Input assembly (targets and covariates are already loader-normalised)
    # ─────────────────────────────────────────────────────────────────────
    def _series_emb(self, batch, n_steps):
        """(B, n_series, n_steps, emb_dim) broadcast series embedding."""
        emb = self.series_emb(self.series_ids)              # (nf, d)
        return emb.view(1, self.n_series, 1, self.emb_dim).expand(
            batch, -1, n_steps, -1)

    def _window_scale(self, x):
        """log of the history window's mean absolute load, per series.

        x is loader-normalised, so this is a *relative* level feature that
        lets the model adapt to the current window instead of relying on a
        single global scale.  (B, n_series); computed from history only.
        """
        if not self.use_window_scale:
            return None
        y = x[:, :self.seq_len, :self.n_series]
        return torch.log(y.abs().mean(dim=1) + 1e-3)

    def _scale_feat(self, ws, n_steps):
        """(B, nf) → (B, nf, n_steps, 1) or None."""
        if ws is None:
            return None
        return ws.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_steps, -1)

    def _past_inputs(self, x):
        """History inputs → (B, n_series, P, input_dim)."""
        b, nf, p = x.shape[0], self.n_series, self.seq_len
        xt = x[:, :p, :nf]                                  # (B, P, nf)
        # previous target: y_{t-1}, with y_0 repeated at the left edge
        prev = torch.cat([xt[:, :1], xt[:, :-1]], dim=1)    # (B, P, nf)
        cov = x[:, :p, :][:, :, self.cov_idx]               # (B, P, n_cov)
        prev = prev.permute(0, 2, 1).unsqueeze(-1)          # (B, nf, P, 1)
        cov = cov.unsqueeze(1).expand(-1, nf, -1, -1)       # (B, nf, P, n_cov)
        parts = [prev, cov, self._series_emb(b, p)]
        scale = self._scale_feat(self._window_scale(x), p)
        if scale is not None:
            parts.append(scale)
        return torch.cat(parts, dim=-1)

    def _future_calendar(self, x):
        """Known calendar for the H decoder steps (B, H, 4).

        Reconstructed from the last history row so the ``base`` setting sees
        the same sin/cos(HOD), sin/cos(DOW) columns the loader would have
        appended for future rows.  DOW advances only at midnight.
        """
        last = x[:, self.seq_len - 1, CAL_SLICE]            # (B, 4)
        hod = torch.atan2(last[:, 0], last[:, 1]) / (2 * math.pi) * 24.0
        dow = torch.atan2(last[:, 2], last[:, 3]) / (2 * math.pi) * 7.0
        # atan2 wraps to (-π, π]; normalise before adding the lead so that
        # the DOW rollover is counted at the true midnight, not at 0h.
        hod = torch.remainder(torch.round(hod), 24.0)       # [0, 24)
        dow = torch.remainder(torch.round(dow), 7.0)        # [0, 7)
        steps = torch.arange(1, self.pred_len + 1,
                             device=x.device, dtype=x.dtype)
        hours = hod[:, None] + steps[None, :]               # (B, H)
        hh = torch.remainder(hours, 24.0)
        ww = torch.remainder(dow[:, None] + torch.floor(hours / 24.0), 7.0)
        ah = hh / 24.0 * 2.0 * math.pi
        aw = ww / 7.0 * 2.0 * math.pi
        return torch.stack([torch.sin(ah), torch.cos(ah),
                            torch.sin(aw), torch.cos(aw)], dim=-1)

    def _future_covariates(self, x):
        """Decoder covariates (B, H, n_cov)."""
        b, h = x.shape[0], self.pred_len
        cov = x.new_zeros(b, h, self.n_cov)
        if self.use_calendar and self.enc_in >= CAL_SLICE.stop:
            cov[..., CAL_POS] = self._future_calendar(x)
        return cov

    def _future_inputs(self, x, y_prev):
        """Decoder inputs given the previous target (B, H, n_series).

        y_prev holds y_{t-1} for each decoder step (teacher forcing or the
        model's own sample). Returns (B, n_series, H, input_dim)."""
        b, h, nf = y_prev.shape
        cov = self._future_covariates(x)                    # (B, H, n_cov)
        prev = y_prev.permute(0, 2, 1).unsqueeze(-1)        # (B, nf, H, 1)
        cov = cov.unsqueeze(1).expand(-1, nf, -1, -1)       # (B, nf, H, n_cov)
        parts = [prev, cov, self._series_emb(b, h)]
        scale = self._scale_feat(self._window_scale(x), h)
        if scale is not None:
            parts.append(scale)
        return torch.cat(parts, dim=-1)

    def _split_head(self, raw):
        """(..., 2) → μ, σ with σ ≥ sigma_floor (normalised units)."""
        mu = raw[..., 0]
        sigma = F.softplus(raw[..., 1]) + self.sigma_floor
        return mu, sigma

    # ─────────────────────────────────────────────────────────────────────
    # Training path: full teacher forcing in one LSTM call
    # ─────────────────────────────────────────────────────────────────────
    def teacher_forward(self, x, y):
        """One-step-ahead distribution over the horizon, teacher forced.

        x: (B, P(+H), enc_in); y: (B, H, n_series), normalised targets.
        Returns dict(mu, sigma) with shapes (B, H, n_series).
        """
        b, h, nf = y.shape
        p = self.seq_len
        past = self._past_inputs(x)                         # (B, nf, P, d)
        y_prev = torch.cat([x[:, p - 1:p, :nf], y[:, :-1]], dim=1)
        fut = self._future_inputs(x, y_prev)                # (B, nf, H, d)
        seq = torch.cat([past, fut], dim=2)                 # (B, nf, P+H, d)
        out, _ = self.lstm(seq.reshape(b * nf, p + h, self.input_dim))
        raw = self.head(out[:, p:, :]).view(b, nf, h, 2)
        mu, sigma = self._split_head(raw.permute(0, 2, 1, 3))
        return {'mu': mu, 'sigma': sigma}

    def elementwise_nll(self, out, y_true, scale3):
        """Per-element NLL (Gaussian or Student-t), in physical units.

        y_true is normalised; scale3 = (1, 1, n_series) physical scales.
        The likelihood is evaluated after de-normalisation (σ_raw = σ·s), so
        the objective is scale-free per series unless ``loss_scale_weighted``
        is on — which aligns training with the raw-unit pinball/CRPS protocol.
        """
        s3 = scale3.view(1, 1, -1)
        mu = out['mu'] * s3
        sig = (out['sigma'] * s3).clamp(min=1e-3)
        y = y_true * s3
        z = (y - mu) / sig
        if self.likelihood == 'student_t':
            df = (4.0 + F.softplus(self.df_raw)).view(1, 1, -1)
            nll = (torch.lgamma((df + 1.0) / 2.0)
                   - torch.lgamma(df / 2.0)
                   - 0.5 * torch.log(df * math.pi)
                   - torch.log(sig)
                   - 0.5 * (df + 1.0) * torch.log1p(z * z / df))
        else:
            nll = torch.log(sig) + 0.5 * z * z + 0.5 * math.log(2.0 * math.pi)
        if self.loss_scale_weighted:
            w = s3 / s3.mean()
            nll = nll * w
        return nll

    def compute_loss(self, out, y_true, scale3):
        return self.elementwise_nll(out, y_true, scale3).mean()

    # ─────────────────────────────────────────────────────────────────────
    # Deterministic open-loop path (diagnostic / point metrics only)
    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x):
        """Feed the mean back autoregressively; returns dict(mu, sigma).

        This ignores the growth of predictive uncertainty through recursion,
        so it must not be used for interval/CRPS evaluation.  The predictive
        distribution is produced by :meth:`sample`.
        """
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
        mus, sigmas = [], []
        for t in range(h):
            step = torch.cat([y_prev, cov_all[:, t, :], emb.squeeze(1)],
                             dim=-1)
            if ws_t is not None:
                step = torch.cat([step, ws_t], dim=-1)
            inp = step.unsqueeze(1)
            out, (state_h, state_c) = self.lstm(inp, (state_h, state_c))
            mu, sigma = self._split_head(self.head(out.squeeze(1)))
            y_prev = mu.unsqueeze(1)
            mus.append(mu.view(b, nf))
            sigmas.append(sigma.view(b, nf))
        return {'mu': torch.stack(mus, dim=1),
                'sigma': torch.stack(sigmas, dim=1)}

    # ─────────────────────────────────────────────────────────────────────
    # Predictive sampling (the path used for CRPS / pinball)
    # ─────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, x, n_samples=200, chunk=64, seed=0):
        """Autoregressive predictive ensemble.

        Returns (B, H, n_series, S) samples in normalised units.  Windows are
        processed in chunks so the (chunk·n_series·S) LSTM state stays small.
        """
        self.eval()
        b_all, h, nf, p = x.shape[0], self.pred_len, self.n_series, \
            self.seq_len
        out = torch.empty(b_all, h, nf, n_samples,
                          device=x.device, dtype=x.dtype)
        emb = self.series_emb(self.series_ids)              # (nf, d)
        for ci, start in enumerate(range(0, b_all, chunk)):
            xb = x[start:start + chunk]
            b = xb.shape[0]
            torch.manual_seed(int(seed) * 100003 + ci)

            past = self._past_inputs(xb).reshape(b * nf, p, self.input_dim)
            _, (state_h, state_c) = self.lstm(past)
            state_h = state_h.repeat_interleave(n_samples, dim=1)
            state_c = state_c.repeat_interleave(n_samples, dim=1)

            y_prev = xb[:, p - 1, :nf].reshape(b * nf, 1) \
                .repeat_interleave(n_samples, dim=0)
            cov_all = self._future_covariates(xb)           # (b, h, n_cov)
            cov_all = cov_all.unsqueeze(1).expand(-1, nf, -1, -1).reshape(
                b * nf, h, self.n_cov).repeat_interleave(n_samples, dim=0)
            emb_t = emb.view(1, nf, self.emb_dim).expand(b, -1, -1).reshape(
                b * nf, self.emb_dim).repeat_interleave(n_samples, dim=0)
            ws = self._window_scale(xb)
            ws_t = (ws.reshape(b * nf, 1).repeat_interleave(n_samples, dim=0)
                    if ws is not None else None)
            if self.likelihood == 'student_t':
                df = (4.0 + F.softplus(self.df_raw)).to(
                    device=x.device, dtype=x.dtype)          # (nf,)
                df_flat = df.view(1, nf, 1).expand(
                    b, -1, n_samples).reshape(-1)            # (b*nf*S,)

            for t in range(h):
                step = torch.cat([y_prev, cov_all[:, t, :], emb_t], dim=-1)
                if ws_t is not None:
                    step = torch.cat([step, ws_t], dim=-1)
                inp = step.unsqueeze(1)
                out_t, (state_h, state_c) = self.lstm(inp,
                                                      (state_h, state_c))
                mu, sigma = self._split_head(self.head(out_t.squeeze(1)))
                # Keep the rollout on the data manifold: cap sigma and clamp
                # the feedback to the loader's normalised range.  Without
                # this, Student-t paths can explode and poison the recursion.
                sigma = sigma.clamp(max=5.0)
                if self.likelihood == 'student_t':
                    y = mu + sigma * torch.distributions.StudentT(
                        df_flat).sample()
                else:
                    y = mu + sigma * torch.randn(
                        mu.shape, device=x.device, dtype=x.dtype)
                y = y.clamp(-10.0, 10.0)
                y_prev = y.unsqueeze(1)
                out[start:start + b, t] = y.view(b, nf, n_samples)
        return out
