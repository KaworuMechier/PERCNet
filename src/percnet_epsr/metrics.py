"""Point and probabilistic metrics.

Point metrics follow the ASHRAE-style definitions used in the prior
manuscript (MAE / RMSE / MAPE / CV-RMSE / sMAPE / NMBE / ACCR, where ACCR is
the Pearson correlation, not a percentage).  Probabilistic metrics use the
9-tau grid: CRPS = 2 * mean pinball.
"""

from __future__ import annotations

import math

import numpy as np


TAUS = np.array([0.05, 0.1625, 0.275, 0.3875, 0.5,
                 0.6125, 0.725, 0.8375, 0.95], dtype=np.float64)


def _flat(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1)


def MAE(y_true, y_pred) -> float:
    return float(np.abs(_flat(y_true) - _flat(y_pred)).mean())


def RMSE(y_true, y_pred) -> float:
    e = _flat(y_true) - _flat(y_pred)
    return float(math.sqrt(float(np.mean(e * e))))


def MAPE(y_true, y_pred, zero_thr: float = 1e-3) -> float:
    yt, yp = _flat(y_true), _flat(y_pred)
    mask = np.abs(yt) > zero_thr
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)


def CV_RMSE(y_true, y_pred) -> float:
    rmse = RMSE(y_true, y_pred)
    mean = float(np.abs(_flat(y_true)).mean())
    return float(rmse / (mean + 1e-8) * 100.0)


def SMAPE(y_true, y_pred) -> float:
    yt, yp = _flat(y_true), _flat(y_pred)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    return float(np.mean(np.abs(yt - yp) / (denom + 1e-8)) * 100.0)


def NMBE(y_true, y_pred) -> float:
    yt, yp = _flat(y_true), _flat(y_pred)
    mean = float(np.abs(yt).mean())
    return float((yt - yp).sum() / (len(yt) * mean + 1e-8) * 100.0)


def ACCR(y_true, y_pred) -> float:
    yt, yp = _flat(y_true), _flat(y_pred)
    if yt.std() < 1e-8 or yp.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(yt, yp)[0, 1])


def calc_all_metrics(y_true, y_pred, n_tgt: int = 3):
    """Return (total, per_target) dictionaries."""
    total = {
        "MAE": MAE(y_true, y_pred),
        "RMSE": RMSE(y_true, y_pred),
        "MAPE": MAPE(y_true, y_pred),
        "CV_RMSE": CV_RMSE(y_true, y_pred),
        "sMAPE": SMAPE(y_true, y_pred),
        "NMBE": NMBE(y_true, y_pred),
        "ACCR": ACCR(y_true, y_pred),
    }
    per_target = {}
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    for i in range(min(n_tgt, yt.shape[-1])):
        per_target[i] = {
            "MAE": MAE(yt[:, :, i], yp[:, :, i]),
            "RMSE": RMSE(yt[:, :, i], yp[:, :, i]),
            "MAPE": MAPE(yt[:, :, i], yp[:, :, i]),
            "CV_RMSE": CV_RMSE(yt[:, :, i], yp[:, :, i]),
            "sMAPE": SMAPE(yt[:, :, i], yp[:, :, i]),
            "NMBE": NMBE(yt[:, :, i], yp[:, :, i]),
            "ACCR": ACCR(yt[:, :, i], yp[:, :, i]),
        }
    return total, per_target


# ── probabilistic ────────────────────────────────────────────────────────────
def probabilistic_metrics(q, y, taus=TAUS) -> dict:
    """q: (N,H,C,K), y: (N,H,C) in physical units."""
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64).reshape(-1)
    if q.shape[-1] != len(taus):
        raise ValueError(f"quantile axis {q.shape[-1]} != {len(taus)} taus")

    t = taus.reshape(1, 1, 1, -1)
    u = y[..., None] - q
    pin = np.maximum(t * u, (t - 1.0) * u)
    pin_mean = float(pin.mean())
    per_pin = pin.mean(axis=(0, 1, 3))

    i_lo = int(np.argmin(taus))
    i_hi = int(np.argmax(taus))
    i50 = int(np.argmin(np.abs(taus - 0.5)))
    q_lo, q_hi, q50 = q[..., i_lo], q[..., i_hi], q[..., i50]

    inside = (y >= q_lo) & (y <= q_hi)
    picp = float(inside.mean())
    pinaw = float((q_hi - q_lo).mean())

    alpha = 0.10
    wink = ((q_hi - q_lo)
            + (2.0 / alpha) * np.maximum(q_lo - y, 0.0)
            + (2.0 / alpha) * np.maximum(y - q_hi, 0.0))
    winkler = float(wink.mean())

    qce = float(np.mean([np.abs((y < q[..., i]).mean() - tau)
                         for i, tau in enumerate(taus)]))

    err = np.abs(y - q50)
    k = max(1, int(round(0.10 * err.size)))
    cvar = float(np.sort(err.ravel())[-k:].mean())

    return {
        "CRPS": 2.0 * pin_mean,
        "Pinball": pin_mean,
        "MAE": float(err.mean()),
        "PICP_90": picp,
        "PINAW_90": pinaw,
        "Winkler_90": winkler,
        "QCE": qce,
        "CVaR_90": cvar,
        "per_crps": 2.0 * per_pin,
        "per_mae": err.mean(axis=(0, 1)),
    }


def ensemble_crps(samples, y) -> float:
    """Near-exact CRPS from a predictive sample ensemble.

    samples: (..., S) float; y: (...) float.  Uses the sorted-sample
    estimator (Gneiting & Raftery).
    """
    s = np.sort(np.asarray(samples, dtype=np.float64), axis=-1)
    yy = np.asarray(y, dtype=np.float64)[..., None]
    S = s.shape[-1]
    k = np.arange(1, S + 1)
    term = (2.0 / S) * np.abs(s - yy).sum(axis=-1)
    spread = (1.0 / (S * (S - 1))) * (
        (2.0 * k - S - 1.0) * s).sum(axis=-1)
    return float((term - spread).mean())


def summarise_probs(per_model: dict[str, dict]) -> str:
    """Tiny text table used by the experiment entry points."""
    cols = ["CRPS", "Pinball", "MAE", "PICP_90", "PINAW_90",
            "Winkler_90", "QCE", "CVaR_90"]
    head = f"{'model':<24}" + "".join(f"{c:>12}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in per_model.items():
        lines.append(f"{name:<24}" + "".join(
            f"{m.get(c, float('nan')):>12.3f}" for c in cols))
    return "\n".join(lines)
