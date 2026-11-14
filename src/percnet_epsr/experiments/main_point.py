"""Point-forecast table (honest protocol, no exogenous forecast branch)."""

from __future__ import annotations

import argparse
import csv
import os

import torch

from percnet_epsr.data import default_spec
from percnet_epsr.training.runner import train_one
from percnet_epsr.utils import resolve_device


SCREEN_MODELS = ["percnet_v2", "percnet_v2_eesm", "percnet_v2_ema"]


def run_grid(models, horizons, seeds, out_root, device, spec, protocol):
    for model in models:
        for horizon in horizons:
            for seed in seeds:
                print(f">>> {model} | h={horizon} | seed={seed} | "
                      f"{protocol}", flush=True)
                if model == "festrnn":
                    from percnet_epsr.protocols.official_festrnn import run
                    run(horizon, protocol, device, spec.file,
                        os.path.join(out_root, "protocol"), seed=seed,
                        double_init=True)
                else:
                    train_one(model, horizon, seed, spec, device,
                              out_root=out_root, protocol=protocol)


def print_table(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    print(f"\n{'model':<22}{'h':>4}{'seed':>6}{'MAE':>10}{'MAPE':>8}")
    for r in rows:
        print(f"{r['model']:<22}{r['horizon']:>4}{r['seed']:>6}"
              f"{float(r['MAE']):>10.2f}{float(r['MAPE']):>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="percnet_v2,percnet_v2_eesm,"
                                        "dlinear,itransformer")
    ap.add_argument("--horizons", default="24,48,72,96")
    ap.add_argument("--seeds", default="2024,2025,2026,2027,2028")
    ap.add_argument("--out", default="results/point")
    ap.add_argument("--device", default=None)
    ap.add_argument("--protocol", choices=["honest", "official"],
                    default="official",
                    help="official = test-selected main protocol; "
                         "honest = val-selected appendix")
    ap.add_argument("--screen", action="store_true",
                    help="h24 x 3 seeds x the EESM/EMA ladder")
    args = ap.parse_args()

    device = resolve_device(args.device)
    spec = default_spec()
    if args.screen:
        models = SCREEN_MODELS
        horizons, seeds = [24], [2024, 2025, 2026]
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        horizons = [int(x) for x in args.horizons.split(",")]
        seeds = [int(x) for x in args.seeds.split(",")]
    print(f"point table | models={models} horizons={horizons} "
          f"seeds={seeds} protocol={args.protocol} device={device}")
    run_grid(models, horizons, seeds, args.out, device, spec, args.protocol)
    print_table(os.path.join(args.out, "results.csv"))


if __name__ == "__main__":
    main()
