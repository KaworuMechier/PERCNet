#!/usr/bin/env bash
# Run stages on the 3090.  Never run this on the local machine.
#   bash scripts/run_all_3090.sh static
#   bash scripts/run_all_3090.sh smoke
#   bash scripts/run_all_3090.sh protocol
#   bash scripts/run_all_3090.sh point_screen
#   bash scripts/run_all_3090.sh v3_screen
#   bash scripts/run_all_3090.sh v3_audit
#   bash scripts/run_all_3090.sh v3_horizons
#   bash scripts/run_all_3090.sh point
#   bash scripts/run_all_3090.sh probabilistic
#   bash scripts/run_all_3090.sh xcpd
#   bash scripts/run_all_3090.sh scheduling
#   bash scripts/run_all_3090.sh all
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
PYTHON="${PYTHON:-python}"
STAGE="${1:-all}"

run_static() { "$PYTHON" scripts/static_check.py; }
run_smoke() { "$PYTHON" scripts/smoke_3090.py; }
run_protocol() {
  # The reproduced 613.71 anchor uses the official two-model RNG sequence.
  "$PYTHON" -m percnet_epsr.protocols.official_festrnn \
    --horizon 24 --mode official --double-init
  "$PYTHON" -m percnet_epsr.protocols.official_festrnn \
    --horizon 24 --mode honest --double-init
}
run_point_screen() {
  "$PYTHON" -m percnet_epsr.experiments.main_point --screen \
    --protocol official --out results/point
}
run_v3_screen() {
  # Full-channel ablation ladder at h24: full v3, then one component off.
  # 4 variants x 2 seeds; pick one, then run the full point table.
  # Clear results/point_v3 first: a run directory must not be reused across
  # a model-code change (checkpoints load fine but mean something else).
  "$PYTHON" -m percnet_epsr.experiments.main_point \
    --models percnet_v3_wxq,percnet_v3_wxh64 \
    --horizons 24 --seeds 2024,2025 --out results/point_v3
}
run_v3_audit() {
  # Close the open questions in the method section:
  #   wxw    = context restricted to the six true weather channels
  #   ndec   = decomposition-loss weight removed
  #   nocyc  = no calendar cycle queue
  #   noanc  = no last-value anchor
  "$PYTHON" -m percnet_epsr.experiments.main_point \
    --models percnet_v3_wx,percnet_v3_wxw,percnet_v3_wx_ndec,percnet_v3_wx_nocyc,percnet_v3_wx_noanc \
    --horizons 24 --seeds 2024,2025 --out results/point_v3_audit
}
run_v3_horizons() {
  # Horizon ladder for the frozen model: does the weather context buy more
  # at 48-96 h than at 24 h?  That decides which horizons lead the paper.
  "$PYTHON" -m percnet_epsr.experiments.main_point \
    --models percnet_v3_wx --horizons 48,72,96 --seeds 2024,2025 \
    --out results/point_v3
}
run_point() {
  # Main point table.  `official` is the released FESTRNN protocol (best test
  # epoch); every model, including the baseline, is trained and selected the
  # same way, and the table states that the selection is optimistic.  The
  # validation-selected `honest` numbers are the appendix (run_honest).
  "$PYTHON" -m percnet_epsr.experiments.main_point \
    --protocol official \
    --models percnet_v3_wx,festrnn,dlinear,itransformer \
    --out results/point
}
run_baselines() {
  # Additional point baselines, same loader / recipe / protocol as the table.
  "$PYTHON" -m percnet_epsr.experiments.main_point \
    --protocol official \
    --models mamba,crossformer,film,segrnn \
    --out results/point
}
run_prob() {
  # Same protocol as the point table; CQR runs on validation quantiles only.
  # DeepAR is excluded from the `official` column set (its sampler is honest
  # only, see experiments/probabilistic.py).
  "$PYTHON" -m percnet_epsr.experiments.probabilistic \
    --protocol official \
    --models percnet_v3_wx,tft,deeparq \
    --out results/probabilistic
}
run_xcpd() {
  "$PYTHON" -m percnet_epsr.experiments.xcpd_ablation \
    --backbone percnet_v2_eesm --protocol official \
    --horizon 24 --seeds 2024,2025,2026
}
run_honest() {
  # Val-selected robustness appendix; same recipe, no test selection.
  "$PYTHON" -m percnet_epsr.experiments.main_point \
    --protocol honest \
    --models percnet_v2,percnet_v2_eesm,festrnn \
    --out results/point_honest
  "$PYTHON" -m percnet_epsr.experiments.probabilistic \
    --protocol honest --models percnet_v2,percnet_v2_eesm,tft,deeparq,deepar \
    --out results/probabilistic_honest
}
run_scenarios() {
  # Does the joint structure of the forecast matter for dispatch?  Scores the
  # three constructions and prices them with the two-stage MILP.
  # Prefer the CQR (calibrated-marginal) directories; fall back to the raw
  # ones.  Passing both would run every model twice.
  RUNS=$(ls -d results/probabilistic/*_h24_s*_cqr 2>/dev/null \
         || ls -d results/probabilistic/*_h24_s* 2>/dev/null || true)
  if [[ -z "$RUNS" ]]; then
    echo "[skip] no probabilistic run dirs"
    return
  fi
  "$PYTHON" -m percnet_epsr.experiments.scenario_value \
    --run-dirs $RUNS --out results/scenario_value \
    --scenarios 16 --max-windows 24 --rank 6
}
run_sched() {
  RUNS=$(ls -d results/probabilistic/percnet_v2_eesm_ASU_official_h24_s* 2>/dev/null || true)
  if [[ -z "$RUNS" ]]; then
    echo "[skip] no probabilistic run dirs for scheduling"
    return
  fi
  "$PYTHON" -m percnet_epsr.experiments.scheduling.rl_reserve \
    --run-dirs $RUNS --out results/scheduling/rl
  "$PYTHON" -m percnet_epsr.experiments.scheduling.two_stage \
    --run-dirs $RUNS --out results/scheduling --scenarios 8 \
    --rl-policy results/scheduling/rl/rl_policy.json
}

case "$STAGE" in
  static) run_static ;;
  smoke) run_smoke ;;
  protocol) run_protocol ;;
  point_screen) run_point_screen ;;
  v3_screen) run_v3_screen ;;
  v3_audit) run_v3_audit ;;
  v3_horizons) run_v3_horizons ;;
  point) run_point ;;
  baselines) run_baselines ;;
  probabilistic) run_prob ;;
  xcpd) run_xcpd ;;
  honest) run_honest ;;
  scenarios) run_scenarios ;;
  scheduling) run_sched ;;
  all)
    run_static
    run_smoke
    run_protocol
    run_point_screen
    run_point
    run_prob
    run_xcpd
    run_honest
    run_sched
    ;;
  *) echo "unknown stage: $STAGE"; exit 1 ;;
esac
