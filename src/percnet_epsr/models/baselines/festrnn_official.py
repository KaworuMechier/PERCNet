"""Official FESTRNN architecture, vendored for protocol reproduction.

Source: github.com/balloonplacewzy/FESTRNN (Wang et al., IEEE TII 2026),
``test_model/FESTRNN.py`` and ``layers/RevIN.py``.  The forward pass is kept
verbatim; ``Model`` only adds the ``(B,T,C) -> (B,C,T)`` transpose expected by
the official implementation so the rest of this project can treat it like any
other baseline.

This file exists to reproduce the official 24 h MAE (~611.7).  It is not the
development port used in the probabilistic tables.
"""

import warnings

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5,
                 affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce,
                                   keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(
            x, dim=dim2reduce, keepdim=True, unbiased=False)
            + self.eps).detach()

    def _normalize(self, x):
        x = x - self.last if self.subtract_last else x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev[:, :, :3]
        x = x + self.last if self.subtract_last else x + self.mean[:, :, :3]
        return x


class moving_avg(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride,
                                padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


class _OfficialFESTRNN(nn.Module):
    """Verbatim official forward; input (B,C,T), output (B,T,3)."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.channels = int(configs.enc_in)
        self.hidden_size = int(configs.hidden_size)
        self.d_model = int(configs.d_model)
        self.seg_len = int(configs.seg_len)
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len
        self.dropout = float(configs.dropout)

        self.revin_layer = RevIN(self.enc_in, affine=False,
                                 subtract_last=False)
        self.decompsition = series_decomp(kernel_size=13)
        self.endlinear = nn.Linear(2 * self.enc_in, 64)
        self.finallinear = nn.Sequential(
            nn.Linear(64, 32), nn.Linear(32, 3))
        self.linear_seasonal = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len))
        self.linear_trend = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len))
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model), nn.ReLU())
        self.rnn = nn.GRU(input_size=self.d_model,
                          hidden_size=self.d_model, num_layers=1,
                          bias=True, batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(
            torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(
            torch.randn(self.enc_in, self.d_model // 2))
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len))
        self.mask_matrix = nn.Parameter(
            torch.ones(int(self.seq_len / 2) + 1, self.channels))
        # PyTorch labels complex-valued nn.Modules "experimental"; the
        # official FESTRNN uses this complex spectral projection, so the
        # warning is expected and cosmetic.  Keep the exact initialisation.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Complex modules are a new feature")
            self.freq_linear = nn.Linear(
                int(self.seq_len / 2) + 1,
                int(self.pred_len / 2) + 1).to(torch.cfloat)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x = self.revin_layer(x, "norm")

        x_fft = torch.fft.rfft(x, dim=1)
        x_inverse_fft = torch.flip(x_fft, dims=[1]) * self.mask_matrix
        x = torch.fft.irfft(x_fft + x_inverse_fft, dim=1)

        seasonal, trend = self.decompsition(x)
        seasonal = self.linear_seasonal(
            seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        trend = self.linear_trend(
            trend.permute(0, 2, 1)).permute(0, 2, 1)
        z = seasonal + trend

        batch_size = x.size(0)
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1)
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))
        _, hn = self.rnn(x)
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1),
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)
        _, hy = self.rnn(pos_emb, hn.repeat(
            1, 1, self.seg_num_y).view(1, -1, self.d_model))
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) + seq_last

        out_fft = torch.fft.rfft(z, dim=1)
        inv_z = self.freq_linear(
            x_inverse_fft.permute(0, 2, 1)).permute(0, 2, 1)
        z = torch.fft.irfft(out_fft - inv_z, dim=1)
        out_fft = torch.fft.rfft(y, dim=1)
        inv_y = self.freq_linear(
            x_inverse_fft.permute(0, 2, 1)).permute(0, 2, 1)
        y = torch.fft.irfft(out_fft - inv_y, dim=1)
        z = torch.cat((z, y), dim=-1)

        z = self.endlinear(z)
        z = self.finallinear(z)
        z = self.revin_layer(z, "denorm")
        return z


class Model(nn.Module):
    """Baseline interface: input (B,T,C), output (B,T,3)."""

    def __init__(self, config):
        super().__init__()
        self.net = _OfficialFESTRNN(config)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        return self.net(x.transpose(1, 2))
