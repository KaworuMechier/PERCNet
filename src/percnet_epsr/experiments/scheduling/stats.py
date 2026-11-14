"""Paired statistics for scheduling cost/reliability comparisons."""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_bootstrap(a, b, n_boot: int = 10000, seed: int = 2024,
                     alpha: float = 0.05) -> dict:
    """Bootstrap the paired difference mean(a - b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"mean_diff": float(d.mean()),
            "ci_low": float(lo), "ci_high": float(hi),
            "p_two_sided": float(2.0 * min((boots <= 0).mean(),
                                           (boots >= 0).mean()))}


def wilcoxon(a, b) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.allclose(a, b):
        return {"stat": 0.0, "p_value": 1.0}
    res = stats.wilcoxon(a, b, zero_method="wilcox",
                         alternative="two-sided")
    return {"stat": float(res.statistic), "p_value": float(res.pvalue)}
