"""Split-conformal CQR on the 9-tau grid.

Calibration uses validation quantiles only; test is never touched.  For each
symmetric pair (tau_lo, tau_hi):
    s_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
    c   = Quantile_{ceil((n+1)(1-alpha))/n}(s)
    q_lo' = q_lo - max(c, 0),  q_hi' = q_hi + max(c, 0)
The calibrated grid is sorted to enforce monotonicity; the median is
unchanged.
"""

from __future__ import annotations

import numpy as np


def finite_sample_level(n: int, nominal: float) -> float:
    level = np.ceil((n + 1) * float(nominal)) / max(n, 1)
    return float(min(1.0, max(0.0, level)))


def symmetric_pairs(taus: np.ndarray):
    taus = np.asarray(taus, dtype=np.float64)
    for i, tau in enumerate(taus):
        if tau >= 0.5 - 1e-12:
            continue
        j = int(np.argmin(np.abs(taus - (1.0 - tau))))
        if abs(taus[i] + taus[j] - 1.0) > 1e-6:
            continue
        yield i, j, float(tau), float(1.0 - 2.0 * tau)


def cqr_calibrate(q_cal, y_cal, q_test, taus):
    """Return (q_test_calibrated, per-pair correction table)."""
    q_cal = np.asarray(q_cal, dtype=np.float64)
    y_cal = np.asarray(y_cal, dtype=np.float64)
    q_out = np.asarray(q_test, dtype=np.float64).copy()
    if q_cal.ndim != 4 or y_cal.ndim != 3:
        raise ValueError("expected q_cal (N,H,C,K), y_cal (N,H,C)")
    valid = np.isfinite(y_cal).all(axis=(1, 2))
    qv, yv = q_cal[valid], y_cal[valid]
    if len(yv) == 0:
        raise ValueError("no finite calibration windows")
    n = int(valid.sum())
    table = []
    for i, j, tau_lo, nominal in symmetric_pairs(taus):
        score = np.maximum(qv[..., i] - yv, yv - qv[..., j])
        level = finite_sample_level(n, nominal)
        c = np.quantile(score, level, axis=0, method="higher")
        c = np.maximum(c, 0.0)
        q_out[..., i] -= c
        q_out[..., j] += c
        table.append({"i_lo": i, "i_hi": j, "tau_lo": tau_lo,
                      "nominal": nominal, "level": level,
                      "corr_mean": float(np.mean(c)),
                      "corr_max": float(np.max(c))})
    q_out = np.sort(q_out, axis=-1)
    return q_out, table
