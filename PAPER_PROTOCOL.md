# Evaluation protocols

## Main protocol (official) and honest appendix

The manuscript follows the released FESTRNN protocol, applied identically to
every model:

| mode | selection | role |
|---|---|---|
| `official` | test evaluated every epoch, best **test** MAE selected | main tables |
| `honest` | best **validation** MAE checkpoint, test evaluated once | appendix robustness check |

`official` is test-selected and therefore optimistic; this is a
protocol-matching choice, not an unbiased estimate.  Both modes are produced
with the same optimiser/schedule/budget, and both numbers are reported.

Rules:

1. Main point/probabilistic/scheduling tables use `protocol=official` for
   every model, including PERC-Net, xCPD, TFT, DeepAR and the point
   baselines.
2. The appendix reports `protocol=honest` for the same models.
3. Every table caption and the method section state the selection rule
   explicitly; official and honest numbers are never mixed in one table.
4. FESTRNN reproduces as h24 official ≈613.7 and honest ≈630.

## Unified training recipe

All models use the reproduced FESTRNN optimiser/schedule/budget:

| item | value |
|---|---|
| optimiser | Adam, lr = 8e-4 |
| weight decay | 1e-5 (Adam) |
| gradient clip | 5 |
| epochs | 50 |
| scheduler | ReduceLROnPlateau(factor=0.5, patience=2) on **val MAPE** |

The official config's `weightdecay: 0.5` is the scheduler **factor**, not
Adam's weight decay.  Model-specific losses remain model-specific
(PERC-Net's internal objective, TFT/DeepARQ pinball, DeepAR NLL, DLinear/
iTransformer L1); only the optimiser/schedule/budget are shared.

The reproduced FESTRNN official run gives h24 test MAE **613.71** (best test
epoch 23).  The residual gap to 611.7 is RNG/hardware-level; it is treated as
a faithful reproduction, not tuned away.

## Honest robustness appendix

`--protocol honest` keeps the same recipe and selects the best validation
checkpoint, then evaluates test once.  Generate the appendix tables with:

```bash
python scripts/make_tables.py --protocol honest \
  --point results/point_honest/results.csv \
  --prob results/probabilistic_honest/results.csv
```

which writes `point_honest.md` and `probabilistic_honest.md`.  The main
tables (`--protocol official`, the default) and the honest appendix are
reported separately and never merged.

## No-NWP rule

The model and the loader are base-only.  `X.shape[1]` must equal `seq_len`
(168) at every batch boundary.  Any future-row path is a bug and is rejected
by `scripts/static_check.py` and by runtime assertions.

## Seeds and horizons

- Point and probabilistic main tables: horizons 24/48/72/96, seeds
  2024–2028 (5 seeds).
- Screening ladders: horizon 24, seeds 2024–2026 (3 seeds).
- Statistical comparisons use paired tests over seeds/windows.

## Probabilistic protocol

- 9-tau grid: `tau_k = 0.05 + 0.1125 (k-1)`.
- CRPS = 2 × mean pinball; also report PICP90, PINAW90, Winkler90, QCE,
  CVaR90, MAE(q50).
- CQR uses validation quantiles only (`calib_quantiles.npz`); test is never
  used for calibration.
- Marginal calibration is reported separately from joint structure.  The
  joint study fixes the marginals and varies only the dependence:
  `independent` (no joint structure), `copula` (one tau per energy reused
  over the horizon plus a static 3x3 correlation) and `path` (PIT of the
  validation residuals, low-rank factor model, inverse-marginal mapping).
  Report the energy score and the variogram score for all three, and state
  which construction feeds each dispatch table.
- Baselines with known defects (DeepAR Student-t collapse, incomplete
  TFT-full cells) are either fixed and rerun or reported as a capacity
  ablation with explicit caveats.

## Scheduling protocol

- Scenario construction is part of the protocol, not an implementation
  detail: every dispatch table states the `--scenario-mode` it used.  The
  main table uses the fitted `path` construction; `independent` and
  `copula` are the ablations that price the joint structure.
- Two-stage stochastic MILP over sampled scenarios from each model's joint
  predictive distribution.
- Shared commitment, scenario recourse, expected cost + CVaR.
- Reserve policies: median, fixed 5/10%, q90/q95, stochastic/CVaR, oracle,
  RL/bandit margin.
- Report expected cost, CVaR95 cost, EENS/shedding (E/H/C), LOLP, reserve
  shortfall, VSS, EVPI, starts, CO2 proxy.
- Paired bootstrap/Wilcoxon over windows × seeds; seasonal and peak/off-peak
  strata; parameter sensitivity table.
