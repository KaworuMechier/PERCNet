"""Scenario-coherent joint sampling for the stochastic dispatch study.

The forecaster gives nine marginal quantiles per (horizon, energy).  A
stochastic unit-commitment model needs a *path*: a joint draw over the
H x 3 dimensions that keeps both the decay of forecast error over lead time
and the coupling between energies (a hot afternoon lifts electricity and
cooling together).  Three constructions are provided so the scheduling
study can price the difference:

``independent``
    Every (h, energy) sampled on its own.  Marginally calibrated, no joint
    structure at all.
``copula``
    One uniform draw per energy, reused at every horizon, plus a static
    3 x 3 residual correlation.  This is what the earlier pipeline did: the
    quantile level is perfectly persistent inside one energy, which is too
    strong, and only three dimensions of cross-energy structure are kept.
``path``
    Probability-integral transform of the validation residuals, a low-rank
    factor model of the Gaussianised PIT vector, then inverse-marginal
    mapping through the (conformal-calibrated) quantile grid.  Lead-time
    decorrelation and cross-energy coupling both come from the data.

The marginal map is interpolated in Gaussianised coordinates and
extrapolated linearly beyond the 5 % / 95 % grid points, so scenarios are
not truncated at the outer quantiles.

Scores: the energy score (multivariate CRPS) and the variogram score
(temporal dependence).  Both are proper; dimensions are standardised by the
validation marginal spread so the cooling load's larger units do not
dominate the norm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


# ── marginal maps ─────────────────────────────────────────────────────────
def _gauss(taus: np.ndarray) -> np.ndarray:
    return norm.ppf(np.clip(np.asarray(taus, dtype=float), 1e-6, 1 - 1e-6))


def inverse_map(q: np.ndarray, taus: np.ndarray,
                u: np.ndarray) -> np.ndarray:
    """u (..., H, C) -> values, through the per-dimension quantile grid.

    Interpolation happens in z = Phi^-1(u) space so the tails behave like a
    Gaussian continuation instead of a flat clamp.
    """
    taus = np.asarray(taus, dtype=float)
    zq = _gauss(taus)
    K = len(zq)
    shape = u.shape
    flat_q = np.sort(q.reshape(-1, K), axis=-1)
    zu = _gauss(u.reshape(-1))
    idx = np.clip(np.searchsorted(zq, zu) - 1, 0, K - 2)
    ar = np.arange(flat_q.shape[0])
    z0, z1 = zq[idx], zq[idx + 1]
    w = (zu - z0) / (z1 - z0)
    y = flat_q[ar, idx] + w * (flat_q[ar, idx + 1] - flat_q[ar, idx])
    return y.reshape(shape)


def pit(q: np.ndarray, taus: np.ndarray,
        y: np.ndarray) -> np.ndarray:
    """values (N, H, C) -> uniform PIT levels, through the same grid."""
    taus = np.asarray(taus, dtype=float)
    K = len(taus)
    zq = _gauss(taus)
    N, H, C = y.shape
    flat_q = np.sort(q.reshape(-1, K), axis=-1)
    flat_y = y.reshape(-1)
    # row-wise searchsorted: how many grid points fall below each observation
    idx = np.clip((flat_q < flat_y[:, None]).sum(axis=1) - 1, 0, K - 2)
    ar = np.arange(flat_y.shape[0])
    y0, y1 = flat_q[ar, idx], flat_q[ar, idx + 1]
    w = np.clip((flat_y - y0) / (y1 - y0 + 1e-9), -3.0, 4.0)
    z = zq[idx] + w * (zq[idx + 1] - zq[idx])
    return norm.cdf(z).reshape(N, H, C)


# ── dependence model ──────────────────────────────────────────────────────
@dataclass
class DependenceModel:
    """Gaussian factor model of the PIT vector, shape (H*C,)."""

    mean: np.ndarray            # (D,)
    load: np.ndarray            # (D, r)
    resid_var: np.ndarray       # (D,)
    shape: tuple                # (H, C)
    rank: int
    explained: float = 1.0      # share of variance kept by the factors

    @classmethod
    def fit(cls, u_cal: np.ndarray, rank: int = 6) -> "DependenceModel":
        H, C = u_cal.shape[1:]
        D = H * C
        z = _gauss(u_cal).reshape(len(u_cal), D)
        mu = z.mean(axis=0)
        zc = z - mu
        u_svd, s, vt = np.linalg.svd(zc, full_matrices=False)
        r = int(min(rank, len(s)))
        load = vt[:r].T * (s[:r] / np.sqrt(max(len(z) - 1, 1)))
        var = (load ** 2).sum(axis=1)
        resid = np.maximum((zc ** 2).mean(axis=0) - var, 1e-6)
        explained = float((s[:r] ** 2).sum()
                          / max(float((s ** 2).sum()), 1e-12))
        return cls(mean=mu, load=load, resid_var=resid,
                   shape=(H, C), rank=r, explained=explained)

    def sample_z(self, m: int, rng: np.random.Generator) -> np.ndarray:
        f = rng.standard_normal((m, self.rank)) @ self.load.T
        e = rng.standard_normal((m, self.load.shape[0])) \
            * np.sqrt(self.resid_var)
        return self.mean + f + e


# ── scenario construction ─────────────────────────────────────────────────
def sample_paths(dep: DependenceModel, q: np.ndarray, taus: np.ndarray,
                 m: int, rng: np.random.Generator) -> np.ndarray:
    """Coherent paths, (m, H, C), from a saved quantile grid."""
    H, C = dep.shape
    z = dep.sample_z(m, rng)
    u = norm.cdf(z).reshape(m, H, C)
    return inverse_map(np.repeat(q[None], m, axis=0), taus, u)


def sample_independent(q: np.ndarray, taus: np.ndarray, m: int,
                       rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(0.01, 0.99, size=(m,) + q.shape[:2])
    return inverse_map(np.repeat(q[None], m, axis=0), taus, u)


def sample_copula(q: np.ndarray, taus: np.ndarray, m: int,
                  rng: np.random.Generator,
                  corr: np.ndarray | None = None) -> np.ndarray:
    """The original construction: one tau per energy, reused at every h."""
    H, C = q.shape[:2]
    corr = np.eye(C) if corr is None else corr
    chol = np.linalg.cholesky(corr + np.eye(C) * 1e-6)
    z = rng.standard_normal((m, C)) @ chol.T
    u = np.clip(norm.cdf(z), 0.01, 0.99)
    u = np.broadcast_to(u[:, None, :], (m, H, C))
    return inverse_map(np.repeat(q[None], m, axis=0), taus, u)


# ── proper scores ─────────────────────────────────────────────────────────
def energy_score(samples: np.ndarray, y: np.ndarray,
                 scale: np.ndarray) -> float:
    """Multivariate CRPS on dimensions standardised by ``scale``.

    samples (m, H, C), y (H, C), scale (H, C).
    """
    s = (samples - y) / scale
    m = s.shape[0]
    flat = s.reshape(m, -1)
    term1 = np.linalg.norm(flat, axis=1).mean()
    diff = flat[:, None, :] - flat[None, :, :]
    term2 = np.linalg.norm(diff, axis=-1).mean()
    return float(term1 - 0.5 * term2)


def variogram_score(samples: np.ndarray, y: np.ndarray,
                    scale: np.ndarray, power: float = 0.5) -> float:
    """Variogram score, summed over the three energy blocks.

    Sensitive to lead-time dependence: two scenario sets with identical
    marginals but different temporal structure score differently.
    """
    s = samples / scale
    t = y / scale
    total = 0.0
    for c in range(s.shape[-1]):
        sc = s[..., c]                      # (m, H)
        tc = t[:, c]                        # (H,)
        gx = (np.abs(sc[:, None, :] - sc[:, :, None]) ** power).mean(axis=0)
        gt = np.abs(tc[:, None] - tc[None, :]) ** power
        total += float(((gx - gt) ** 2).sum())
    return total


def marginal_scale(q_cal: np.ndarray) -> np.ndarray:
    """Per-(H,C) spread from the 5 %/95 % grid points, for score scaling."""
    qs = np.sort(np.asarray(q_cal, dtype=float), axis=-1)
    return np.maximum((qs[..., -1] - qs[..., 0]).mean(axis=0), 1e-6)
