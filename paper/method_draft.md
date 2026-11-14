# Method section — working draft

Scope: point + probabilistic forecasting only; the dispatch study is out of
scope. Numbers quoted are h = 24, ASU, two seeds (2024-2025) unless stated.

## 3.1 Backbone (inherited, cited, not claimed)

We build on the FESTRNN backbone [ref] and keep it unchanged, so that every
difference reported in Section 5 is attributable to the blocks introduced
below. The skeleton is: per-window instance normalisation (RevIN) on the
measured channels; a 13-hour moving-average decomposition into seasonal and
trend components; two channel-independent linear specialists mapping the
168-step history to the H-step horizon; a one-shot patch GRU with positional
and channel embeddings plus a last-value anchor; and a learnable
mirror-spectrum mask with a complex frequency-reduction layer applied to the
stream outputs. The resulting model has 26.9 K parameters at H = 24, against
33.2 K for the baseline.

## 3.2 Exogenous context (proposed)

Without future weather, measured exogenous variables can enter in three ways,
and the choice is not neutral. We evaluate all three on the same backbone:

- **None.** Only the three load channels reach the streams (632.7 MAE).
- **Streams.** Every stage runs over all measured channels, and the fusion
  head is left to discover how to use the predicted exogenous series — the
  arrangement of the baseline (639.7 MAE).
- **Context.** The trailing window of the exogenous block is summarised into
  a fixed-length vector and injected once into the fusion head (619.9 MAE).

The stream arrangement is 7.0 MAE *worse* than using no exogenous input at
all. The mechanism is the length of the credit-assignment path: it asks the
network to forecast 24 steps of nine exogenous variables with no supervision
of its own, and then to discover their use through a 36-dimensional
bottleneck, on 19 K training windows. The context arrangement removes both
steps: its statistics are dense, supervised only through the load loss, and
require no forecast.

The context of window $t$ over a trailing window of length $L$ is

$$c_t = \big[\ \text{mean},\ \min,\ \max,\ x_{t},\ x_t - \text{mean}\ \big]_{L,\ j=1..n_e}$$

i.e. five statistics for each of $n_e$ exogenous columns, 45 scalars at
$L = 24$ h and $n_e = 9$. Extending $L$ to 72 h changes nothing (+0.4 MAE),
so one day of history is sufficient; no multi-day thermal inertia term is
needed.

**Composition disclosure.** The nine columns are the six weather channels
(TEMP, DPT, SLP, SEA-P, WBT, ALT), one calendar encoding (DOY-COS) and two
derived energy quantities (CMBMBBTU, GHG). Restricting the context to the six
weather channels alone costs 5.2 MAE (625.1 vs 619.9), so we report both and
state the composition explicitly. No column contains information that is
unavailable at forecast time: all nine are part of the observed record up to
$t$.

## 3.3 Residual fusion (proposed)

The baseline concatenates the two stream outputs across channels,
$[z; y] \in \mathbb{R}^{2C}$, and maps them to the three targets with a
two-layer MLP. With only the load channels available, the same construction
has to relearn the additive decomposition it is handed. We therefore make
the fusion residual:

$$\mu = s + t + \ell + f\big([s, t, \ell, c]\big),$$

where $f$ is a two-layer MLP whose last layer is zero-initialised, so at
initialisation the model is exactly the additive decomposition and training
only learns the cross-channel correction. The residual form is worth 4.0 MAE
relative to the elementwise sum (632.7 vs 637.5 at three channels).

A learnable calendar cycle queue (daily and weekly profiles subtracted from
the history and added back to the forecast) is carried over from the previous
version of this model. It is not part of the baseline: removing it costs
12.6 MAE, the largest single contribution we measure.

## 3.4 Probabilistic head and calibration (proposed)

Nine quantiles per (horizon, energy) are produced by a horizon-conditioned
head,

$$q_k(h) = \mu + W_k \,\phi\big(\mathrm{feat} + e_h\big), \quad k = 1..9,$$

with a softplus scale head supplying $\sigma(h)$ for the Gaussian NLL and
CVaR terms. Training minimises MAPE plus pinball, NLL, CVaR, a monotonicity
penalty and a decomposition term on the three streams.

Two properties of this design are worth stating. First, the head *improves*
the point forecast: training the same architecture with a pure MAPE objective
is 9.6 MAE worse (629.5 vs 619.9), and its seed-to-seed spread triples, so
the auxiliary terms act as regularisers on $\mu$ rather than as a tax on it.
Second, the decomposition term is load-bearing: setting its weight to zero
costs 5.1 MAE. Split conformal calibration (CQR) is then fitted on validation
residuals only, per (horizon, energy) pair, and the test set is never used
for calibration.

## 3.5 What is new

| Block | Origin | Evidence (MAE, h=24) |
|---|---|---|
| RevIN, decomposition, dual linear specialists, patch GRU, spectral mask | FESTRNN | — |
| Last-value anchor | FESTRNN | +4.1 when removed |
| Calendar cycle queue | previous version | +12.6 when removed |
| Exogenous context | **proposed** | +12.8 vs three load channels; +5.2 if restricted to weather only |
| Residual fusion | **proposed** | +4.0 vs elementwise sum |
| Horizon-conditioned quantile head | **proposed** | +9.6 vs point-only objective |
| Decomposition loss weight | previous version | +5.1 when removed |

## 4.x Comparison protocol (one paragraph, place in Section 4)

All models are trained and selected with the released FESTRNN protocol, which
evaluates the test set every epoch and keeps the best epoch: the baseline's
published number is produced by this rule, so applying a different rule to
our model would make the comparison meaningless. This protocol is
optimistically biased; we state so wherever its numbers appear, and report
validation-selected (honest) results for the proposed model and the strongest
baseline in Appendix A. Every model uses the same optimiser, schedule and
epoch budget (Adam, lr 8e-4, weight decay 1e-5, gradient clip 5, 50 epochs,
ReduceLROnPlateau on validation MAPE).
