#!/usr/bin/env python3
"""Locate the PERC-Net vs FESTRNN accuracy gap.  Inference only.

Run on the 3090 *after* the point stage has produced checkpoints:

    python scripts/diag_gap.py --protocol official --seed 2024

Nothing is trained here; the script loads the existing ``best.pt`` files and
answers three questions:

1. per-target test MAE of every checkpoint, next to FESTRNN's row;
2. how much of PERC-Net's error is added by the q50 head: the reported point
   forecast is ``quantiles[..., tau=0.5]``, but the MAPE term in
   ``compute_loss`` trains ``mu``, not the head;
3. whether the q50 correction (q50 - mu) is a constant bias: it prints the
   per-channel mean offset and the MAE after removing the *validation* mean
   offset (no test statistics are used for the correction).
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

from percnet_epsr.data import DataLoaderS, default_spec
from percnet_epsr.metrics import TAUS, calc_all_metrics
from percnet_epsr.models.registry import build_model, make_config
from percnet_epsr.training.artifacts import code_signature, make_run_id
from percnet_epsr.utils import resolve_device

NAMES = ("KW", "CHW", "HT")


def check_code(run_dir: str) -> None:
    """A checkpoint only means something for the code that produced it."""
    path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        stored = json.load(f).get("code_hash")
    now = code_signature()
    if stored is None:
        print(f"  [warn] {os.path.basename(run_dir)}: no code_hash "
              f"(run predates the check)")
    elif stored != now:
        print(f"  [warn] {os.path.basename(run_dir)}: trained on different "
              f"model code ({stored} != {now}) -- numbers below are invalid")



@torch.no_grad()
def collect(model, data, split, batch):
    """Return (y, mu, q50) in physical units for one split."""
    model.eval()
    n_tgt = int(data.spec.n_tgt)
    i50 = int(np.argmin(np.abs(TAUS - 0.5)))
    mus, q50s, ys = [], [], []
    X, Y = split
    for Xb, Yb in data.get_batches(X, Y, batch, shuffle=False):
        out = model(Xb)
        q = out.get("quantiles")
        mus.append(out["mu"].cpu())
        q50s.append((q[..., i50] if q is not None else out["mu"]).cpu())
        ys.append(Yb[:, :, :n_tgt].cpu())
    scale = data.scale[:n_tgt].cpu().numpy().astype(np.float64)
    offset = data.offset[:n_tgt].cpu().numpy().astype(np.float64)

    def phys(t):
        a = t.numpy().astype(np.float64)
        return a * scale.reshape(1, 1, -1) + offset.reshape(1, 1, -1)

    return (phys(torch.cat(ys)), phys(torch.cat(mus)),
            phys(torch.cat(q50s)))


def row(name, point, y, p):
    total, per_target = calc_all_metrics(y, p)
    cells = [f"{per_target[i]['MAE']:9.2f}" for i in range(len(NAMES))]
    print(f"{name:<24}{point:<7}{total['MAE']:9.2f}{''.join(cells)}")


def refs(paths, horizon, protocol):
    """Echo FESTRNN (and any other) rows from the result CSVs."""
    print(f"\n{'model':<24}{'point':<7}{'MAE':>9}"
          f"{'MAE_KW':>9}{'MAE_CHW':>9}{'MAE_HT':>9}   (from results.csv)")
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                if (int(r["horizon"]) != horizon
                        or r.get("protocol") != protocol):
                    continue
                cells = "".join(
                    f"{float(r.get(f'MAE_{n}', 'nan')):9.2f}" for n in NAMES)
                print(f"{r['model']:<24}{'q50':<7}{float(r['MAE']):9.2f}"
                      f"{cells}   [{os.path.basename(path)} s={r['seed']}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--protocol", choices=["official", "honest"],
                    default="official")
    ap.add_argument("--models", default="percnet_v2_eesm,percnet_v2")
    ap.add_argument("--point-root", default="results/point")
    ap.add_argument("--refs", default="results/point/results.csv,"
                                      "results/point/protocol/results.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    spec = default_spec()
    data = DataLoaderS(spec, args.horizon, device)
    print(f"device={device} h={args.horizon} seed={args.seed} "
          f"protocol={args.protocol}")

    print(f"\n{'model':<24}{'point':<7}{'MAE':>9}"
          f"{'MAE_KW':>9}{'MAE_CHW':>9}{'MAE_HT':>9}")
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        run_id = make_run_id(name, "", spec.dataset, args.protocol,
                             args.horizon, args.seed)
        ckpt = os.path.join(args.point_root, run_id, "best.pt")
        if not os.path.isfile(ckpt):
            print(f"{name:<24}(missing {ckpt})")
            continue
        cfg = make_config(name, args.horizon, spec)
        model = build_model(name, cfg).to(device)
        check_code(os.path.dirname(ckpt))
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))

        y_test, mu_test, q50_test = collect(model, data, data.test,
                                            cfg.batchsize)
        y_val, mu_val, q50_val = collect(model, data, data.valid,
                                         cfg.batchsize)

        row(name, "mu", y_test, mu_test)
        row(name, "q50", y_test, q50_test)

        # Constant q50 head bias, estimated on validation only.
        bias = (q50_val - mu_val).mean(axis=(0, 1), keepdims=True)
        row(name, "q50-b", y_test, q50_test - bias)

        print(f"{'':<24}{'':<7}q50-mu: " + ", ".join(
            f"{NAMES[i]} mean {float((q50_test - mu_test)[..., i].mean()):+8.2f} "
            f"|mean| {float(np.abs(q50_test - mu_test)[..., i].mean()):8.2f}"
            for i in range(len(NAMES))))
        corr = [float(np.corrcoef((q50_test - mu_test)[..., i].ravel(),
                                  (y_test - mu_test)[..., i].ravel())[0, 1])
                for i in range(len(NAMES))]
        print(f"{'':<24}{'':<7}corr(q50-mu, y-mu): "
              + ", ".join(f"{NAMES[i]} {corr[i]:+.3f}"
                          for i in range(len(NAMES))))

    refs([p.strip() for p in args.refs.split(",") if p.strip()],
         args.horizon, args.protocol)
    print("\nHow to read it:")
    print("  q50 much worse than mu      -> the point path is the gap; report")
    print("                                 mu or train the head with MAE too")
    print("  q50-b recovers most of it   -> the head is biased, not wrong")
    print("  mu still below FESTRNN      -> architecture/capacity/loss; run the")
    print("                                 ablations (channels, fusion, loss)")


if __name__ == "__main__":
    main()
