# PERCNet

Clean rebuild of PERC-Net for an Electric Power Systems Research (EPSR)
submission.  The final model is **`percnet_v3_wx`**; result summaries and the
figure map are in [`RESULTS.md`](RESULTS.md).  The project is **NWP-free by construction**: there is no
future-weather branch, no future-row loader path, and no `setting=nwp` switch.

## Final model

The final model for the EPSR submission is **`percnet_v3_wx`**
(`V3_VARIANTS["percnet_v3_wx"]` in `src/percnet_epsr/models/registry.py`):
the v3 backbone with the trailing observed-weather context injected once into
the residual fusion head (`use_exo_channels=False, weather_context=True`).
It is the model behind every headline table and figure; see `RESULTS.md`
for the shipped numbers and the map of result files.

## What is here

- **PERC-Net v3 (`percnet_v3_wx`)** — the final model: v3 backbone with a
  fixed-length observed-weather context (24 h trailing window) fused into the
  residual head instead of being fed as forecast streams.  Component variants
  (`noexo`, `nodeg`, `wxw`, `nocyc`, `noanc`, `ndec`, `wxh64`, …) are explicit
  entries in `registry.V3_VARIANTS`; the ablation table uses them.
- **PERC-Net v2** — seasonal/trend linear specialists + patch-GRU local stream
  + component-specialist loss + horizon-conditioned probabilistic head, with an
  optional FESTRNN-style EESM/energy-reduction frequency branch (`v2-eesm`).
- **xCPD (approximate)** — a separate, plug-and-play channel-patch graph
  spectral plugin on top of a frozen v2 backbone.  It is an *approximate*
  implementation from the public description; see `models/xcpd.py`.
- **FESTRNN protocol** — the released selection rule (`official`,
  test-selected best epoch, ~611.7 at 24 h) and a validation-selected
  variant (`honest`, ~630).  Main tables use `official` so that every model,
  including the baseline, is selected identically; the protocol's optimism
  is stated in the caption and `honest` is reported as the appendix.  See
  `PAPER_PROTOCOL.md`.
- **Probabilistic baselines** — TFT, DeepAR, DeepARQ, plus point baselines
  DLinear/iTransformer/FESTRNN and the v3 point baselines, with CQR
  calibration.
- **Scenario-coherent sampling** — marginal quantiles are not enough for
  dispatch: `scenarios.py` maps validation PIT residuals through a low-rank
  factor model so sampled paths keep lead-time decorrelation and
  cross-energy coupling, and scores them (energy score, variogram score)
  against independent and copula constructions.
- **Scheduling** — scenario-based two-stage stochastic MILP for a campus
  multi-energy hub, reserve-policy comparison, VSS/EVPI, paired statistics,
  and an optional RL/bandit reserve-margin policy.  `--scenario-mode`
  selects the joint structure, so the dispatch study can price it.

## Layout

```
configs/                 experiment configs (data/model/experiments)
src/percnet_epsr/        library code
  data.py                base-only loader, FeatureSpec, shape assertions
  metrics.py             point + probabilistic metrics
  models/                percnet_v3.py, percnet_v2.py, xcpd.py,
                         registry.py (recipes + variants), baselines/
  protocols/             official FESTRNN vs honest protocol
  training/              runner + artifact contract
  experiments/           point, probabilistic, xcpd_ablation, scheduling/
scripts/                 static_check.py, smoke_3090.py, run_all_3090.sh
data/                    ASU corrected dataset (dataset_input.csv)
docs/                    figure-1 architecture plan (code-checked)
paper/                   method section working draft
results/                 summary tables, figures, logs and per-run metadata
  *_/results.csv         the tables quoted in the manuscript
  figure_v3_composite/   EPSR composite figure (PDF/PNG + source data)
  figure_v4_analysis/    four analysis figures (PDF/PNG + source_data/)
  PERC-Net_results.xlsx  compiled workbook (T1–T7 + raw sheets)
  <run>/config.json      hyper-parameters of every run
  <run>/history.json     training curve of every run
```

`results/**/predictions.npz`, `quantiles.npz`, `calib_quantiles.npz` and
`best.pt` are excluded from this release (12 GB).  Every summary CSV,
per-run config/history, figure PDF/PNG and its source data is included, so
the reported numbers are auditable; the arrays are regenerated with
`scripts/run_all_3090.sh`.  See `RESULTS.md`.

## Compute policy

The local machine is for **zero-compute static checks only**:

```bash
python -m py_compile $(find src scripts -name '*.py')
python scripts/static_check.py
```

Do **not** run training, sampling, scheduler MILPs or shape/forward smoke
tests locally.  All functional checks run on the 3090:

```bash
python scripts/smoke_3090.py
bash scripts/run_all_3090.sh
```

## Quick start (3090)

```bash
pip install -e .            # or: export PYTHONPATH=src
# 1) reproduce FESTRNN: official 613.7 and honest ~630
python -m percnet_epsr.protocols.official_festrnn --horizon 24 --mode official
python -m percnet_epsr.protocols.official_festrnn --horizon 24 --mode honest

# 2) point table (no NWP), final model
python -m percnet_epsr.experiments.main_point \
  --models percnet_v3_wx,festrnn,dlinear,itransformer --horizons 24,48,72,96

# 3) probabilistic table + CQR, final model
python -m percnet_epsr.experiments.probabilistic \
  --models percnet_v3_wx,tft,deeparq

# 4) xCPD plugin on/off
python -m percnet_epsr.experiments.xcpd_ablation --horizon 24 --seeds 2024,2025,2026

# 5) scheduling + reserve policies + RL
python -m percnet_epsr.experiments.scheduling.two_stage --config configs/experiments/scheduling.yaml
```

## Data

`data/dataset_input.csv` is the corrected ASU campus dataset (23,807 hourly
rows, 12 columns).  Calendar (4) and rolling-weather (4) features are
generated by the loader, giving 20 model channels.  Targets are columns 0-2.

## Non-goals

- No NWP / future-weather covariates.
- No test-set selection *beyond* the released FESTRNN protocol: `official`
  selects every model identically and is disclosed as optimistic; the
  `honest` appendix is validation-selected.
- No xCPD novelty claim: it is a documented approximate component/ablation.
