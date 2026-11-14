"""PERC-Net v3: full-channel backbone with a signed temperature response.

v2 only ever fed the three load channels into the streams: the nine weather
channels travelled through the loader (and through RevIN) and were then
dropped.  On ASU this costs most of the cooling-load accuracy, which is the
target with the strongest temperature dependence.  v3 changes three things,
each behind a flag so the ablation table can isolate it:

``use_exo_channels``
    Run the seasonal/trend linear specialists and the patch-GRU local stream
    over all twelve measured channels with shared weights plus a channel
    embedding, and apply RevIN to the measured block.  This is the channel
    layout FESTRNN uses; the loader is unchanged and every window is still
    exactly ``seq_len`` rows (no future-row path).

``fusion_head``
    Replace the elementwise ``seasonal + trend + local`` sum with a small
    MLP over the three streams concatenated across channels.  The streams
    stay channel-independent, so the mixing has to be learned here.

``degree_response``
    Add a hinged temperature-response term to the three loads.  The
    temperature is the one the streams themselves predict (never an input),
    and the term is ``relu(T - b_j)`` (cooling-degree basis) minus
    ``relu(b_j - T)`` (heating-degree basis) with learnable balance points
    ``b_j``.  Cooling weights are constrained non-negative and heating
    weights non-positive, so each load is monotone in the predicted
    temperature in the physically expected direction.  The learned ``b_j``
    are reportable as data-driven balance-point temperatures.

The forward interface (``mu`` / ``quantiles`` / ``sigma`` and
``compute_loss(out, y, scale3)``) is unchanged, so the runner, the
probabilistic CQR pipeline and the scheduling scenario sampler keep working.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevINAll(nn.Module):
    """Window RevIN over the measured channels; calendar/rolling pass through."""

    def __init__(self, n_chan: int = 12, eps: float = 1e-5):
        super().__init__()
        self.n = int(n_chan)
        self.eps = float(eps)
        self.mean = None
        self.stdev = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xm, xe = x[:, :, :self.n], x[:, :, self.n:]
        self.mean = xm.mean(dim=1, keepdim=True).detach()
        self.stdev = (xm.var(dim=1, keepdim=True, unbiased=False)
                      + self.eps).sqrt().detach()
        return torch.cat([(xm - self.mean) / self.stdev, xe], dim=-1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.k = int(kernel_size)

    def forward(self, x: torch.Tensor):
        k = self.k
        front = x[:, :1, :].repeat(1, (k - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (k - 1) // 2, 1)
        xp = torch.cat([front, x, end], dim=1)
        trend = F.avg_pool1d(xp.transpose(1, 2), kernel_size=k,
                             stride=1).transpose(1, 2)
        return x - trend, trend


class PatchGRUStream(nn.Module):
    """One-shot parallel-decoding patch GRU over the measured channels."""

    def __init__(self, seq_len: int, pred_len: int, n_chan: int,
                 seg_len: int = 24, d_model: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        assert seq_len % seg_len == 0 and pred_len % seg_len == 0
        self.seg_len = int(seg_len)
        self.seg_num_x = seq_len // seg_len
        self.seg_num_y = pred_len // seg_len
        self.n_chan = int(n_chan)
        self.d_model = int(d_model)

        self.value_emb = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model), nn.ReLU())
        self.rnn = nn.GRU(self.d_model, self.d_model, num_layers=1,
                          batch_first=True)
        self.pos_emb = nn.Parameter(
            torch.randn(self.seg_num_y, self.d_model // 2) * 0.02)
        self.chan_emb = nn.Parameter(
            torch.randn(self.n_chan, self.d_model // 2) * 0.02)
        self.predict = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.d_model, self.seg_len))

    def forward(self, x: torch.Tensor):
        b, s, c = x.shape
        xn = x.permute(0, 2, 1).reshape(b * c, self.seg_num_x, self.seg_len)
        _, hn = self.rnn(self.value_emb(xn))
        pos = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(c, 1, 1),
            self.chan_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1),
        ], dim=-1).view(-1, 1, self.d_model)
        _, hy = self.rnn(pos.repeat(b, 1, 1),
                         hn.repeat(1, 1, self.seg_num_y)
                         .view(1, -1, self.d_model))
        y = self.predict(hy).view(b, c, self.seg_num_y * self.seg_len)
        feat = hy.view(b, c, self.seg_num_y, self.d_model).mean(dim=(1, 2))
        return y.permute(0, 2, 1), feat


class DegreeResponse(nn.Module):
    """Sign-constrained hinge response to the predicted temperature.

    ``relu(T - b_j)`` is a cooling-degree basis, ``relu(b_j - T)`` a heating-
    degree basis; the balance points ``b_j`` are learned (degC).  Cooling
    weights are non-negative and heating weights non-positive, so the
    correction is monotone in the physically expected direction.  Raw
    weights start at -8, where softplus is ~3e-4, so the module starts at
    (almost) zero output.
    """

    def __init__(self, n_out: int, init_breakpoints=(12.0, 20.0, 28.0),
                 signs=(1.0, 1.0, -1.0)):
        super().__init__()
        self.bp = nn.Parameter(torch.tensor(init_breakpoints,
                                            dtype=torch.float32))
        k = int(self.bp.numel())
        self.w_cool = nn.Parameter(torch.full((k, n_out), -8.0))
        self.w_heat = nn.Parameter(torch.full((k, n_out), -8.0))
        self.bias = nn.Parameter(torch.zeros(n_out))
        s = torch.tensor([float(x) for x in signs][:n_out],
                         dtype=torch.float32)
        if s.numel() < n_out:
            s = torch.cat([s, torch.ones(n_out - s.numel())])
        # electricity and cooling rise with temperature, heating falls
        self.register_buffer("sign", s)

    def basis(self, temp: torch.Tensor) -> torch.Tensor:
        d = temp.unsqueeze(-1) - self.bp                 # (B,H,K)
        return torch.cat([F.relu(d), F.relu(-d)], dim=-1)

    def forward(self, temp: torch.Tensor) -> torch.Tensor:
        b = self.basis(temp)
        w = torch.cat([F.softplus(self.w_cool),
                       -F.softplus(self.w_heat)], dim=0)
        return self.sign * (b @ w) + self.bias


class Model(nn.Module):
    """Full-channel PERC-Net with fusion and a signed temperature response."""

    def __init__(self, config):
        super().__init__()
        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.enc_in = int(config.enc_in)
        self.n_free = int(getattr(config, "n_free", 3))
        self.use_exo = bool(getattr(config, "use_exo_channels", True))
        self.n_chan = (int(getattr(config, "n_chan", 12)) if self.use_exo
                       else self.n_free)
        self.temp_idx = int(getattr(config, "temp_idx", 3))
        self.fusion_head = bool(getattr(config, "fusion_head", True))
        self.degree_response = (bool(getattr(config, "degree_response", True))
                                and self.use_exo
                                and self.temp_idx < self.n_chan)
        # Weather as a context rather than as extra forecast streams: the
        # recent observed weather block is summarised into a handful of
        # dense features and handed straight to the fusion head.  Predicting
        # 24 steps of nine unsupervised weather channels first (the
        # ``use_exo_channels`` path) turned out to be too long a gradient
        # chain for this amount of training data.
        self.weather_context = (bool(getattr(config, "weather_context", False))
                                and self.fusion_head)
        self.n_exo = int(getattr(config, "n_exo", 9))
        self.wx_windows = tuple(int(w) for w in
                                getattr(config, "wx_windows", (24,)))
        self.n_wx = 5 * self.n_exo * len(self.wx_windows)
        # Optionally let the same context condition the quantile fan: the
        # head currently learns a near-constant offset with no dependence on
        # the weather, which limits CRPS more than it limits MAE.
        self.wx_quantile = (bool(getattr(config, "wx_quantile", False))
                            and self.weather_context
                            and bool(getattr(config, "prob_head", True)))
        self.physics_identity = bool(getattr(config, "physics_identity", True))
        self.use_calendar = bool(getattr(config, "use_calendar", True))
        self.K = int(getattr(config, "n_quantiles", 9))
        self.taus = torch.linspace(0.05, 0.95, self.K)
        self.beta = float(getattr(config, "cvar_beta", 0.9))
        self.lam_pin = float(getattr(config, "lam_pin", 1.0))
        self.lam_nll = float(getattr(config, "lam_nll", 1.0))
        self.lam_tail = float(getattr(config, "lam_tail", 0.5))
        self.lam_mono = float(getattr(config, "lam_mono", 0.1))
        self.lam_dec = float(getattr(config, "lam_dec", 10.0))
        self.use_eesm = bool(getattr(config, "use_eesm", True))
        hidden = int(getattr(config, "linear_hidden", 36))
        kernel = int(getattr(config, "decomp_kernel", 13))

        self.revin = RevINAll(self.n_chan)
        self.decomp = SeriesDecomp(kernel)

        self.linear_seasonal = nn.Sequential(
            nn.Linear(self.seq_len, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, self.pred_len))
        self.linear_trend = nn.Sequential(
            nn.Linear(self.seq_len, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, self.pred_len))

        self.prnn = PatchGRUStream(
            self.seq_len, self.pred_len, self.n_chan,
            seg_len=int(getattr(config, "prnn_seg_len", 24)),
            d_model=int(getattr(config, "prnn_dim", 12)),
            dropout=float(getattr(config, "prnn_dropout", 0.1)))

        if self.use_eesm:
            K = self.seq_len // 2 + 1
            self.K_pred = self.pred_len // 2 + 1
            self.eesm_mask = nn.Parameter(torch.ones(K, self.n_chan))
            self.freq_linear = nn.Parameter(
                torch.randn(K, self.K_pred, dtype=torch.cfloat)
                / K ** 0.5)

        self.use_cycle_queue = bool(
            getattr(config, "use_cycle_queue", True)) and self.use_calendar
        if self.use_cycle_queue:
            self.cycle_daily = nn.Parameter(torch.zeros(24, self.n_free))
            self.cycle_weekly = nn.Parameter(torch.zeros(168, self.n_free))

        self.use_last_value_anchor = bool(
            getattr(config, "use_last_value_anchor", True))
        self.prob_head = bool(getattr(config, "prob_head", True))
        self.prob_detach = bool(getattr(config, "prob_detach", False))
        self.point_loss = str(getattr(config, "point_loss", "mape"))
        self.ctx_proj = nn.Linear(hidden + self.prnn.d_model, hidden)
        if self.prob_head:
            self.quantile_head = nn.Linear(hidden, self.n_free * self.K)
            self.scale_head = nn.Sequential(
                nn.Linear(hidden, self.n_free), nn.Softplus())
            nn.init.constant_(self.scale_head[0].bias, 5.0)

        self.horizon_cond_head = bool(
            getattr(config, "horizon_cond_head", False))
        if self.horizon_cond_head:
            self.horizon_emb = nn.Parameter(
                torch.zeros(self.pred_len, hidden))
        if self.wx_quantile:
            self.wx_proj = nn.Linear(self.n_wx, hidden)
            nn.init.zeros_(self.wx_proj.weight)
            nn.init.zeros_(self.wx_proj.bias)
        self.cvar_t = nn.Parameter(torch.tensor(1.0))

        n_break = int(getattr(config, "degree_breakpoints", 3))
        bps = tuple(12.0 + 8.0 * i for i in range(n_break))
        signs = tuple(getattr(config, "degree_signs", (1.0, 1.0, -1.0)))
        if self.fusion_head:
            d_in = 3 * self.n_chan
            if self.weather_context:
                d_in += self.n_wx
            if self.degree_response:
                self.degree = DegreeResponse(self.n_free, bps, signs)
            self.fusion = nn.Sequential(
                nn.Linear(d_in, 64), nn.GELU(),
                nn.Linear(64, 32), nn.GELU(),
                nn.Linear(32, self.n_free))
            # Residual form: ``mu = stack + fusion(...)`` with a zero output
            # layer, so epoch 0 reproduces seasonal+trend+local exactly and
            # the head only has to learn the cross-channel correction.
            # Without this the randomly initialised MLP throws the
            # decomposition away and the 50-epoch budget is spent relearning
            # it (measured: best test epoch 10, then flat at +21 MAE).
            nn.init.zeros_(self.fusion[-1].weight)
            nn.init.zeros_(self.fusion[-1].bias)
        elif self.degree_response:
            self.degree = DegreeResponse(self.n_free, bps, signs)

        coef = getattr(config, "cel_coef", (0.003412, 0.0120, 1.0))
        self.register_buffer("cel_coef", torch.tensor(coef))
        self.register_buffer("load_scale", torch.ones(self.n_free))

    # ── preparation ──────────────────────────────────────────────────────
    def _prepare(self, x: torch.Tensor):
        xn = self.revin(x)
        nf = self.n_free
        cyc_d = cyc_w = None
        if self.use_cycle_queue:
            with torch.no_grad():
                cal = x[:, self.seq_len - 1, 12:16]
                hod = torch.remainder(torch.round(
                    torch.atan2(cal[:, 0], cal[:, 1])
                    / (2 * math.pi) * 24), 24).long()
                dow = torch.remainder(torch.round(
                    torch.atan2(cal[:, 2], cal[:, 3])
                    / (2 * math.pi) * 7), 7).long()
                how_last = hod + 24 * dow
            steps = torch.arange(1, self.pred_len + 1, device=x.device)
            hod_f = torch.remainder(hod.view(-1, 1) + steps.view(1, -1), 24)
            hofw_f = torch.remainder(
                how_last.view(-1, 1) + steps.view(1, -1), 168)
            cyc_d = self.cycle_daily[hod_f]
            cyc_w = self.cycle_weekly[hofw_f]
            past = xn[:, :self.seq_len]
            cal_p = x[:, :self.seq_len, 12:16]
            with torch.no_grad():
                hod_p = torch.remainder(torch.round(
                    torch.atan2(cal_p[..., 0], cal_p[..., 1])
                    / (2 * math.pi) * 24), 24).long()
                dow_p = torch.remainder(torch.round(
                    torch.atan2(cal_p[..., 2], cal_p[..., 3])
                    / (2 * math.pi) * 7), 7).long()
                hofw_p = torch.remainder(hod_p + 24 * dow_p, 168)
            past = torch.cat([
                past[:, :, :nf] - self.cycle_daily[hod_p]
                - self.cycle_weekly[hofw_p],
                past[:, :, nf:]], dim=-1)
        else:
            past = xn[:, :self.seq_len]

        seq_last = None
        if self.use_last_value_anchor:
            seq_last = past[:, -1:, :nf].detach()
            past = torch.cat([past[:, :, :nf] - seq_last,
                              past[:, :, nf:]], dim=-1)
        return xn, past, seq_last, cyc_d, cyc_w

    def _enhance(self, residual: torch.Tensor):
        if not self.use_eesm:
            return residual, None
        x_fft = torch.fft.rfft(residual, dim=1)
        inv = torch.flip(x_fft, dims=[1]) * self.eesm_mask
        x_ea = torch.fft.irfft(x_fft + inv, n=self.seq_len, dim=1)
        return x_ea, inv

    def _reduce(self, y: torch.Tensor, inv: torch.Tensor | None) -> torch.Tensor:
        if inv is None:
            return y
        out_fft = torch.fft.rfft(y, dim=1)
        excess = torch.einsum("bkc,kp->bpc",
                              inv.to(self.freq_linear.dtype),
                              self.freq_linear)
        return torch.fft.irfft(out_fft - excess, n=self.pred_len, dim=1)

    def _temp_hat(self, stack: torch.Tensor) -> torch.Tensor:
        k = self.temp_idx
        return (stack[:, :, k] * self.revin.stdev[:, 0, k:k + 1]
                + self.revin.mean[:, 0, k:k + 1])

    def _weather_context(self, x: torch.Tensor) -> torch.Tensor:
        """Trailing exogenous-block summary, (B, 5*n_exo*len(windows)).

        Mean / min / max / last value / last-minus-mean over each trailing
        window.  With the default ``n_exo=9`` the block is columns 3-11 of
        the observed record: six weather channels (TEMP, DPT, SLP, SEA-P,
        WBT, ALT), one calendar encoding (DOY-COS) and two derived energy
        quantities (CMBMBBTU, GHG).  Restricting it to the six weather
        channels only costs 5.2 MAE at h=24 (variant ``wxw``); no column
        contains information unavailable at forecast time.  All statistics
        are on the same O(1) loader scale, so they can share a linear layer
        with the streams.  The name is historical; the paper calls this the
        exogenous context.
        """
        parts = []
        for win in self.wx_windows:
            w = x[:, -win:, 3:3 + self.n_exo]
            mean = w.mean(dim=1)
            parts += [mean, w.amin(dim=1), w.amax(dim=1),
                      w[:, -1], w[:, -1] - mean]
        return torch.cat(parts, dim=-1)

    # ── forward ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        xn, past, seq_last, cyc_d, cyc_w = self._prepare(x)
        nf, nc = self.n_free, self.n_chan
        residual = past[:, :, :nc]
        residual, inv = self._enhance(residual)

        seasonal, trend = self.decomp(residual)
        s = self.linear_seasonal(seasonal.transpose(1, 2)).transpose(1, 2)
        t = self.linear_trend(trend.transpose(1, 2)).transpose(1, 2)
        loc, feat_loc = self.prnn(residual)
        if self.use_eesm:
            s = self._reduce(s, inv)
            t = self._reduce(t, inv)
            loc = self._reduce(loc, inv)
        stack = s + t + loc

        deg_corr = None
        if self.degree_response:
            deg_corr = self.degree(self._temp_hat(stack))

        wctx = self._weather_context(x) if self.weather_context else None
        mu = stack[:, :, :nf]
        if self.fusion_head:
            # Only the three streams enter the head.  The degree basis is in
            # degC (0-35) while the streams are O(1) after RevIN; mixing them
            # in one linear layer let the basis dominate the input.  The
            # temperature information still reaches the head through the
            # streams' TEMP channel, and the physical response enters
            # additively below.
            parts = [s, t, loc]
            if wctx is not None:
                parts.append(wctx.unsqueeze(1).expand(-1, self.pred_len, -1))
            mu = mu + self.fusion(torch.cat(parts, dim=-1))
        if deg_corr is not None:
            mu = mu + deg_corr

        feat_seas = self.linear_seasonal[0](seasonal.transpose(1, 2))
        feat_seas = F.leaky_relu(feat_seas).mean(dim=1)
        feat = self.ctx_proj(torch.cat([feat_seas, feat_loc], dim=-1))

        q = sigma = None
        if self.prob_head:
            center = mu.detach() if self.prob_detach else mu
            if self.horizon_cond_head:
                fh = feat.unsqueeze(1) + self.horizon_emb.unsqueeze(0)
                if self.wx_quantile:
                    fh = fh + self.wx_proj(wctx).unsqueeze(1)
                q = self.quantile_head(fh).view(-1, self.pred_len, nf, self.K)
                sigma = self.scale_head(fh) + 1e-3
            else:
                q = self.quantile_head(feat).view(-1, 1, nf, self.K)
                sigma = self.scale_head(feat) + 1e-3
            q = q + center.unsqueeze(-1)

        self._cyc_fut = (cyc_d + cyc_w) if cyc_d is not None else None
        self._seq_last = seq_last

        if cyc_d is not None:
            mu = mu + cyc_d + cyc_w
            if q is not None:
                q = q + (cyc_d + cyc_w).unsqueeze(-1)
        if seq_last is not None:
            mu = mu + seq_last
            if q is not None:
                q = q + seq_last.unsqueeze(-1)

        win_std = self.revin.stdev[:, :1, :nf]
        win_mean = self.revin.mean[:, :1, :nf]
        mu = mu * win_std + win_mean
        if q is not None:
            q = q * win_std.unsqueeze(-1) + win_mean.unsqueeze(-1)

        out = {"mu": mu, "_streams": {"seasonal": s, "trend": t,
                                      "local": loc}}
        if self.degree_response:
            out["degree"] = {"breakpoints": self.degree.bp.detach(),
                             "corr": deg_corr.detach()}
        if self.physics_identity:
            mu_raw = mu * self.load_scale.view(1, 1, nf)
            out["cmbbtu"] = (mu_raw * (self.cel_coef * self.load_scale)
                             .view(1, 1, nf)).sum(dim=-1, keepdim=True)
        if self.prob_head:
            out["quantiles"], out["sigma"] = q, sigma
        return out

    # ── loss ─────────────────────────────────────────────────────────────
    def compute_loss(self, out: dict, y_true: torch.Tensor,
                     scale3: torch.Tensor) -> torch.Tensor:
        nf = self.n_free
        mu = out["mu"]
        streams = out["_streams"]
        q = out.get("quantiles")
        y_raw = y_true * scale3

        if q is None:
            o_raw = mu * scale3
            return ((y_raw - o_raw).abs()
                    / (y_raw.abs() + 1e-2)).mean() * 100.0

        sigma = out["sigma"].clamp(min=1e-3)
        q_raw = q * scale3.view(1, 1, nf, 1)
        mu_raw = mu * scale3
        s_sc = sigma if sigma.dim() == 3 else sigma.view(-1, 1, nf)

        u05 = y_raw - mu_raw
        if self.point_loss == "mape":
            l_point = (u05.abs() / (y_raw.abs() + 1e-2)).mean() * 100.0
        else:
            l_point = 0.5 * u05.abs().mean()

        u = y_raw.unsqueeze(-1) - q_raw
        taus = self.taus.to(u.device).view(1, 1, 1, -1)
        l_pin = torch.maximum(taus * u, (taus - 1) * u).mean() * self.lam_pin

        resid = (y_raw - mu_raw).abs()
        z = resid.detach() / s_sc if self.prob_detach else resid / s_sc
        l_nll = (z + torch.log(s_sc)).mean()
        l_cvar = (self.cvar_t + F.relu(z - self.cvar_t).mean()
                  / (1.0 - self.beta))
        l_mono = F.relu(-torch.diff(q, dim=-1)).mean()

        win_std = self.revin.stdev[:, :1, :nf]
        win_mean = self.revin.mean[:, :1, :nf]
        r_full = (y_true - win_mean) / win_std
        if self._cyc_fut is not None:
            r_full = r_full - self._cyc_fut
        if self._seq_last is not None:
            r_full = r_full - self._seq_last
        r_full = r_full.detach()

        s, t, loc = (streams["seasonal"][..., :nf],
                     streams["trend"][..., :nf],
                     streams["local"][..., :nf])
        seas_tgt, trend_tgt = self.decomp(r_full)
        l_seas = (s - seas_tgt).abs().mean()
        l_trend = (t - trend_tgt).abs().mean()
        lin_out = (s + t).detach()
        l_loc = (loc - (r_full - lin_out)).abs().mean()

        return (l_pin + l_point + self.lam_nll * l_nll
                + self.lam_tail * l_cvar + self.lam_mono * l_mono
                + self.lam_dec * (l_seas + l_trend + l_loc))
