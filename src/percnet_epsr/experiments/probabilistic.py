"""Probabilistic table: PERC-Net variants + CQR calibration.

The shared runner already saves test and validation quantiles for every
quantile-producing model.  This entry point trains the requested PERC-Net
variants (honest protocol) and then applies split-CQR to each run using the
validation artifact only.

Autoregressive baselines with dedicated samplers (TFT / DeepAR / DeepARQ) are
wired in ``experiments/probabilistic_ar.py``; this module does not fake their
training.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np
import torch

from percnet_epsr.data import default_spec
from percnet_epsr.metrics import TAUS, calc_all_metrics, probabilistic_metrics
from percnet_epsr.training.artifacts import (append_result, flatten_metrics,
                                             save_predictions, save_quantiles)
from percnet_epsr.training.conformal import cqr_calibrate
from percnet_epsr.training.runner import train_one
from percnet_epsr.utils import resolve_device


def _load(run_dir: str, name: str):
    path = os.path.join(run_dir, name)
    if not os.path.isfile(path):
        return None
    z = np.load(path)
    return z["quantiles"], z["y_true"], z.get("taus", TAUS)


def calibrate_run(run_dir: str, out_root: str) -> dict | None:
    test = _load(run_dir, "quantiles.npz")
    calib = _load(run_dir, "calib_quantiles.npz")
    if test is None or calib is None:
        print(f"  [skip cqr] {run_dir}: missing quantile artifacts")
        return None
    q_test, y_test, taus = test
    q_cal, y_cal, _ = calib
    q_calibrated, table = cqr_calibrate(q_cal, y_cal, q_test, taus)
    out_dir = run_dir.rstrip("/") + "_cqr"
    os.makedirs(out_dir, exist_ok=True)
    save_quantiles(out_dir, q_calibrated, y_test, taus)
    i50 = int(np.argmin(np.abs(np.asarray(taus) - 0.5)))
    save_predictions(out_dir, y_test, q_calibrated[..., i50],
                     q_calibrated[..., i50])
    with open(os.path.join(out_dir, "cqr_pairs.json"), "w") as f:
        json.dump(table, f, indent=2)

    prob = probabilistic_metrics(q_calibrated, y_test, taus)
    total, per_target = calc_all_metrics(y_test, q_calibrated[..., i50],
                                         y_test.shape[-1])
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    base_id = os.path.basename(os.path.normpath(run_dir))
    row = flatten_metrics(
        base_id + "_cqr", str(cfg.get("model", "?")) + "_cqr", "cqr",
        str(cfg.get("dataset", "ASU")), str(cfg.get("protocol", "honest")),
        int(cfg.get("horizon", 0)), int(cfg.get("seed", 0)),
        int(cfg.get("params", 0)), 0.0, total, per_target, prob)
    append_result(os.path.join(out_root, "results.csv"), row)
    print(f"  [cqr] {base_id} | CRPS {prob['CRPS']:.2f} "
          f"PICP90 {prob['PICP_90']:.3f} QCE {prob['QCE']:.3f}")
    return row


def print_table(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    print(f"\n{'model':<28}{'h':>4}{'seed':>6}{'CRPS':>11}"
          f"{'MAE':>10}{'PICP90':>9}{'QCE':>8}")
    for r in rows:
        print(f"{r['model']:<28}{r['horizon']:>4}{r['seed']:>6}"
              f"{float(r['CRPS']):>11.1f}{float(r['MAE']):>10.1f}"
              f"{float(r['PICP_90']):>9.3f}{float(r['QCE']):>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="percnet_v2,percnet_v2_eesm")
    ap.add_argument("--horizons", default="24,48,72,96")
    ap.add_argument("--seeds", default="2024,2025,2026,2027,2028")
    ap.add_argument("--out", default="results/probabilistic")
    ap.add_argument("--device", default=None)
    ap.add_argument("--protocol", choices=["honest", "official"],
                    default="official",
                    help="official = test-selected main protocol; "
                         "honest = val-selected appendix")
    ap.add_argument("--skip-train", action="store_true",
                    help="only calibrate existing run directories")
    ap.add_argument("--no-cqr", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    spec = default_spec()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    horizons = [int(x) for x in args.horizons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"probabilistic table | models={models} horizons={horizons} "
          f"seeds={seeds} protocol={args.protocol} cqr={not args.no_cqr} "
          f"device={device}")

    supported = {"percnet_v2", "percnet_v2_eesm", "percnet_v2_ema",
                 "percnet_v3", "percnet_v3_noexo", "percnet_v3_nofusion",
                 "percnet_v3_nodeg", "percnet_v3_noeesm", "percnet_v3_plain",
                 "percnet_v3_wx", "percnet_v3_wx72", "percnet_v3_wxq",
                 "percnet_v3_wxh64", "tft", "deeparq"}
    calibratable = supported | {"deepar"}
    for model in models:
        if model in supported:
            for horizon in horizons:
                for seed in seeds:
                    if not args.skip_train:
                        print(f">>> {model} | h={horizon} | seed={seed}",
                              flush=True)
                        train_one(model, horizon, seed, spec, device,
                                  out_root=args.out, protocol=args.protocol)
        elif model == "deepar":
            print("  [note] deepar: use experiments/probabilistic_ar.py "
                  "(honest only) for training; CQR still applies below",
                  flush=True)
        else:
            print(f"  [skip] {model}: no trainer wired", flush=True)

    if not args.no_cqr:
        for model in models:
            if model not in calibratable:
                continue
            for horizon in horizons:
                for seed in seeds:
                    from percnet_epsr.training.artifacts import make_run_id
                    if model == "deepar":
                        if args.protocol == "official":
                            continue
                        pattern = os.path.join(
                            "results/probabilistic_ar",
                            f"deepar_*_h{horizon}_s{seed}")
                        dirs = sorted(glob.glob(pattern))
                    else:
                        run_id = make_run_id(model, "", spec.dataset,
                                             args.protocol, horizon, seed)
                        dirs = [os.path.join(args.out, run_id)]
                    for run_dir in dirs:
                        if not os.path.isdir(run_dir):
                            continue
                        cfg_path = os.path.join(run_dir, "config.json")
                        if model == "deepar" and os.path.isfile(cfg_path):
                            with open(cfg_path) as f:
                                if json.load(f).get("protocol") != args.protocol:
                                    continue
                        calibrate_run(run_dir, args.out)
    print_table(os.path.join(args.out, "results.csv"))


if __name__ == "__main__":
    main()
