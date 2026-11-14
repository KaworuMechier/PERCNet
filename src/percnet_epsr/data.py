"""Base-only sliding-window loader.

The project is deliberately single-setting: every window is exactly
``seq_len`` rows long.  There is no future-row path, no exogenous forecast
input and no per-setting switch.  The runtime assertion in ``_batchify`` is
the guard that keeps it that way.

Channel layout (20 columns for ASU):

   0- 2   loads            KW, CHWTON, HTMBTU      (targets)
   3- 9   weather          TEMP DPT SLP SEA-P WBT ALT DOY-COS
  10-11   derived          CMBMBBTU, GHG
  12-15   calendar         sin/cos(HOD), sin/cos(DOW)   (appended)
  16-19   rolling weather  TEMP/WBT 24 h and 72 h means  (appended)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch


BASE_CHANNELS = 12
TARGET_IDX = (0, 1, 2)
WEATHER_IDX = (3, 4, 5, 6, 7, 8, 9)
CAL_SLICE = slice(12, 16)
ROLL_SLICE = slice(16, 20)
COV_IDX = (3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19)
ROLL_COLS = (3, 7)            # TEMP, WBT
ROLL_WINDOWS = (24, 72)


@dataclass(frozen=True)
class FeatureSpec:
    """Static description of the ASU feature layout."""

    dataset: str = "ASU"
    file: str = "data/dataset_input.csv"
    seq_len: int = 168
    train_frac: float = 0.8
    valid_frac: float = 0.1
    normalize: int = 2
    n_free: int = 3
    n_tgt: int = 3
    use_calendar: bool = True
    use_rolling_weather: bool = True
    physics_identity: bool = True
    cel_coef: tuple[float, float, float] = (0.003412, 0.0120, 1.0)

    @property
    def n_calendar(self) -> int:
        return 4 if self.use_calendar else 0

    @property
    def n_rolling(self) -> int:
        return 4 if self.use_rolling_weather else 0

    @property
    def enc_in(self) -> int:
        return BASE_CHANNELS + self.n_calendar + self.n_rolling

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "file": self.file,
            "seq_len": self.seq_len,
            "train_frac": self.train_frac,
            "valid_frac": self.valid_frac,
            "normalize": self.normalize,
            "n_free": self.n_free,
            "n_tgt": self.n_tgt,
            "use_calendar": self.use_calendar,
            "use_rolling_weather": self.use_rolling_weather,
            "physics_identity": self.physics_identity,
            "cel_coef": list(self.cel_coef),
            "enc_in": self.enc_in,
        }


def default_spec() -> FeatureSpec:
    return FeatureSpec()


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean; expanding at the left edge."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(len(x))
    lo = np.maximum(idx - window + 1, 0)
    return (c[idx + 1] - c[lo]) / (idx - lo + 1)


def _calendar(abs_rows: np.ndarray) -> np.ndarray:
    """(N, P) absolute row indices -> (N, P, 4) sin/cos(HOD, DOW)."""
    hod = (abs_rows % 24) / 24.0 * 2.0 * np.pi
    dow = ((abs_rows % 168) // 24) / 7.0 * 2.0 * np.pi
    return np.stack(
        [np.sin(hod), np.cos(hod), np.sin(dow), np.cos(dow)], axis=-1)


class DataLoaderS:
    """Contiguous 80/10/10 sliding windows, train-only robust scaling."""

    def __init__(self, spec: FeatureSpec, horizon: int,
                 device: str | torch.device = "cpu"):
        self.spec = spec
        self.P = int(spec.seq_len)
        self.h = int(horizon)
        self.device = torch.device(device)

        raw = pd.read_csv(spec.file)
        raw = raw.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] < BASE_CHANNELS:
            raise ValueError(f"expected >= {BASE_CHANNELS} numeric columns, "
                             f"got {raw.shape}")
        self.rawdat = raw
        self.n, self.m = raw.shape

        train_end = int(spec.train_frac * self.n)
        valid_end = int((spec.train_frac + spec.valid_frac) * self.n)
        self.train_end = train_end
        self.valid_end = valid_end

        offset = self.P + self.h - 1
        train_idx = list(range(offset, train_end))
        valid_idx = list(range(train_end, valid_end))
        test_idx = list(range(valid_end, self.n))
        self._normalized(spec.normalize, train_idx)

        # Causal rolling weather features (normalised units).
        self.roll = None
        if spec.use_rolling_weather:
            feats = [_rolling_mean(self.dat[:, c], w)
                     for c in ROLL_COLS for w in ROLL_WINDOWS]
            self.roll = np.stack(feats, axis=-1)

        self.train = self._batchify(train_idx)
        self.valid = self._batchify(valid_idx)
        self.test = self._batchify(test_idx)

        self.scale = torch.from_numpy(self.scale).float().to(self.device)
        self.offset = torch.from_numpy(self.offset).float().to(self.device)

    # ── normalisation ────────────────────────────────────────────────────
    def _normalized(self, normalize: int, train_idx: list[int]) -> None:
        self.dat = np.zeros_like(self.rawdat)
        self.scale = np.ones(self.m, dtype=np.float64)
        self.offset = np.zeros(self.m, dtype=np.float64)
        if not normalize:
            self.dat = self.rawdat.copy()
            return
        if normalize == 3:
            tr = self.rawdat[train_idx, :]
            for i in range(self.m):
                mu = tr[:, i].mean()
                sd = tr[:, i].std() + 1e-8
                self.scale[i] = sd
                self.offset[i] = mu
                self.dat[:, i] = (self.rawdat[:, i] - mu) / sd
            return
        if normalize != 2:
            raise ValueError(f"unsupported normalize={normalize}")
        tr = self.rawdat[train_idx, :]
        for i in range(self.m):
            col = tr[:, i]
            abs_col = np.abs(col)
            raw_max = float(abs_col.max())
            p95 = (float(np.percentile(abs_col, 95))
                   if len(abs_col) >= 20 else raw_max)
            robust = max(p95 * 2.0, 1e-8)
            self.scale[i] = robust if raw_max > robust * 50.0 else raw_max
            self.scale[i] = max(self.scale[i], 1e-8)
            self.dat[:, i] = np.clip(
                self.rawdat[:, i] / self.scale[i], -10.0, 10.0)

    # ── windows ──────────────────────────────────────────────────────────
    def _batchify(self, idx_set: Iterable[int]) -> list[torch.Tensor]:
        idx = list(idx_set)
        n = len(idx)
        ends = [idx[i] - self.h + 1 for i in range(n)]
        starts = [e - self.P for e in ends]
        X = np.stack([self.dat[s:e, :] for s, e in zip(starts, ends)])
        Y = np.stack([self.dat[i + 1 - self.h:i + 1, :] for i in idx])

        if self.spec.use_calendar:
            abs_rows = np.stack(
                [np.arange(s, e) for s, e in zip(starts, ends)])
            X = np.concatenate([X, _calendar(abs_rows)], axis=-1)
        if self.spec.use_rolling_weather:
            if self.roll is None:
                raise RuntimeError("rolling features requested but not built")
            X = np.concatenate(
                [X, np.stack([self.roll[s:e, :]
                              for s, e in zip(starts, ends)])], axis=-1)

        if X.shape[1] != self.P:
            raise AssertionError(
                f"window length {X.shape[1]} != seq_len {self.P}; "
                "the project is base-only")
        if X.shape[-1] != self.spec.enc_in:
            raise AssertionError(
                f"channel count {X.shape[-1]} != enc_in {self.spec.enc_in}")
        return [torch.from_numpy(X).float(), torch.from_numpy(Y).float()]

    # ── batching ─────────────────────────────────────────────────────────
    def get_batches(self, inputs: torch.Tensor, targets: torch.Tensor,
                    batch_size: int, shuffle: bool = True):
        length = len(inputs)
        index = torch.randperm(length) if shuffle else torch.arange(length)
        for start in range(0, length, batch_size):
            sel = index[start:start + batch_size]
            yield (inputs[sel].to(self.device),
                   targets[sel].to(self.device))

    def split_sizes(self) -> dict[str, int]:
        return {"train": len(self.train[0]),
                "valid": len(self.valid[0]),
                "test": len(self.test[0])}
