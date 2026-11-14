#!/usr/bin/env bash
# =============================================================================
# Paper Tables 2 and 3 — the ablation ladder, official protocol.
#
#   Table 2 (three injection paths):
#       percnet_v3_noexo   no exogenous history        (3 load channels only)
#       percnet_v3_nodeg   twelve channels end-to-end   (streams see everything)
#       percnet_v3_wxw     context restricted to the six weather channels
#       percnet_v3_wx      the proposed configuration   (control)
#
#   Table 3 (component ablation, each row switches one thing off):
#       percnet_v3_plain      elementwise stream sum instead of residual fusion
#       percnet_v3_wx_nocyc   no daily/weekly cycle queue
#       percnet_v3_wx_point   pure MAPE objective (no quantile head)
#       percnet_v3_wx_ndec    decomposition-supervision weight set to zero
#       percnet_v3_wx_noanc   no last-value anchor
#       percnet_v3_wxh64      expert hidden width 36 -> 64
#
# Usage
#   bash scripts/run_ablation_v3.sh            # GPU 1, all variants
#   bash scripts/run_ablation_v3.sh 3          # GPU 3
#   GROUP=paths bash scripts/run_ablation_v3.sh      # Table 2 only (faster)
#   GROUP=ablation bash scripts/run_ablation_v3.sh   # Table 3 only
#   SEEDS=2024,2025 bash scripts/run_ablation_v3.sh  # fewer seeds
#
#   nohup bash scripts/run_ablation_v3.sh 1 > /dev/null 2>&1 &
#
# Results append to results/point_ablation_v3/results.csv and are also written
# to results/ablation_v3.log (tee), so the log survives even without nohup
# redirection.
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.." || { echo "[ablation] cannot cd to repo root" >&2; exit 1; }
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

GPU="${1:-${GPU:-1}}"
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON="${PYTHON:-python}"
GROUP="${GROUP:-all}"
HORIZONS="${HORIZONS:-24}"
SEEDS="${SEEDS:-2024,2025,2026}"
OUT="${OUT:-results/point_ablation_v3}"
LOG="${LOG:-results/ablation_v3.log}"

PATHS="percnet_v3_wx,percnet_v3_noexo,percnet_v3_nodeg,percnet_v3_wxw"
ABLATION="percnet_v3_plain,percnet_v3_wx_nocyc,percnet_v3_wx_point,percnet_v3_wx_ndec,percnet_v3_wx_noanc,percnet_v3_wxh64"
case "$GROUP" in
  paths)    MODELS="$PATHS" ;;
  ablation) MODELS="$ABLATION" ;;
  all)      MODELS="$PATHS,$ABLATION" ;;
  *) echo "[ablation] unknown GROUP=$GROUP (use paths|ablation|all)" >&2; exit 2 ;;
esac

mkdir -p "$OUT" "$(dirname "$LOG")"

# Fail fast and visibly: import + device check before any training starts.
"$PYTHON" - <<'PY' || { echo "[ablation] pre-flight failed" >&2; exit 1; }
import sys, torch, percnet_epsr
print(f"[ablation] python {sys.version.split()[0]} | torch {torch.__version__}")
print(f"[ablation] packaged at {percnet_epsr.__file__}")
if not torch.cuda.is_available():
    raise SystemExit("[ablation] CUDA not available")
print(f"[ablation] device {torch.cuda.get_device_name(0)}")
PY

echo "[ablation] group=$GROUP device=$CUDA_VISIBLE_DEVICES horizon=$HORIZONS seeds=$SEEDS"
echo "[ablation] models=$MODELS"
echo "[ablation] out=$OUT log=$LOG"
echo "[ablation] start $(date '+%F %T')"

"$PYTHON" -m percnet_epsr.experiments.main_point \
  --protocol official \
  --models "$MODELS" \
  --horizons "$HORIZONS" \
  --seeds "$SEEDS" \
  --out "$OUT" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}

echo "[ablation] exit=$status $(date '+%F %T')"
echo "[ablation] rows in $OUT/results.csv:"
wc -l "$OUT/results.csv" 2>/dev/null || echo "  (no results file)"
exit "$status"
