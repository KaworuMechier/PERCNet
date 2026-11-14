"""segRNN (segment recurrent network, LTSF baseline).

The history is cut into non-overlapping segments, each segment is embedded
into a recurrent state, and the horizon is decoded segment-by-segment in
parallel from the encoder's final state (one GRU pass over learned positional
queries), rather than step-by-step autoregressively.  ``seg_len`` must divide
both ``seq_len`` and ``pred_len`` (24 divides 168 and 24/48/72/96).

Interface matches the other point baselines: input (B, T, C), output
(B, H, C); the runner scores the first three channels.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, inverse=False):
        if not inverse:
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = (x.var(dim=1, keepdim=True, unbiased=False)
                          + self.eps).sqrt().detach()
            return (x - self.mean) / self.stdev
        return x * self.stdev + self.mean


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.enc_in = int(config.enc_in)
        self.seg_len = int(getattr(config, "seg_len", 24))
        d = int(getattr(config, "d_model", 32))
        dropout = float(getattr(config, "dropout", 0.1))
        assert self.seq_len % self.seg_len == 0, "seg_len must divide seq_len"
        assert self.pred_len % self.seg_len == 0, "seg_len must divide pred_len"
        self.n_in = self.seq_len // self.seg_len
        self.n_out = self.pred_len // self.seg_len

        self.revin = RevIN(self.enc_in)
        self.value_emb = nn.Sequential(nn.Linear(self.seg_len, d), nn.ReLU())
        self.encoder = nn.GRU(d, d, num_layers=1, batch_first=True)
        self.decoder = nn.GRU(d, d, num_layers=1, batch_first=True)
        self.pos = nn.Parameter(torch.randn(self.n_out, d) * 0.02)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d, self.seg_len))

    def forward(self, x):
        b, t, c = x.shape
        z = self.revin(x).permute(0, 2, 1).reshape(b * c, self.n_in,
                                                   self.seg_len)
        _, h = self.encoder(self.value_emb(z))
        q = self.pos.unsqueeze(0).repeat(b * c, 1, 1)
        dec, _ = self.decoder(q, h)
        y = self.head(dec).reshape(b, c, self.pred_len).permute(0, 2, 1)
        return self.revin(y, inverse=True)
