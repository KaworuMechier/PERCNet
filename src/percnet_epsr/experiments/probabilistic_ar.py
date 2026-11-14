"""Dedicated runner for the standard DeepAR sampling baseline.

DeepAR is not a quantile-head model: training uses teacher-forced NLL and the
predictive quantiles come from an autoregressive sample ensemble.  The shared
generic runner cannot express that, so this module implements the loop.

The sampler in ``models/baselines/deepar.py`` clamps the feedback to the
loader's normalised range and caps sigma; both are required for stable
rollouts (the Student-t full-size run collapsed without them).
"""

from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np
import torch

from percnet_epsr.data import DataLoaderS, default_spec
from percnet_epsr.metrics import TAUS, calc_all_metrics, probabilistic_metrics
from percnet_epsr.models.registry import build_model, make_config
from percnet_epsr.training.artifacts import (append_result, flatten_metrics,
                                             make_run_id, prepare_run_dir,
                                             save_checkpoint, save_predictions,
                                             save_quantiles)
from percnet_epsr.training.runner import evaluate, fix_seed
from percnet_epsr.utils import resolve_device


@torch.no_grad()
def sample_quantiles(model, data, split, batch, n_samples, seed,
                     scale):
    X, Y = split
    qs, ys = [], []
    for i, (xb, yb) in enumerate(data.get_batches(X, Y, batch,
                                                  shuffle=False)):
        s = model.sample(xb, n_samples=n_samples, chunk=batch,
                         seed=seed + i)                       # (B,H,C,S)
        s = s.cpu().numpy().astype(np.float64)
        s = s * scale.reshape(1, 1, -1, 1)
        q = np.moveaxis(np.quantile(s, TAUS, axis=-1), 0, -1)
        qs.append(q)
        ys.append(yb[:, :, :model.n_series].cpu().numpy().astype(np.float64)
                  * scale.reshape(1, 1, -1))
    return np.concatenate(qs), np.concatenate(ys)


def train_deepar(horizon, seed, spec, device, out_root,
                 n_samples=200, epochs=50, likelihood="gaussian",
                 batch=64, protocol="official"):
    select_on_test = protocol == "official"
    fix_seed(seed)
    data = DataLoaderS(spec, horizon, device)
    cfg = make_config("deepar", horizon, spec,
                      {"likelihood": likelihood, "epochs": epochs,
                       "batchsize": batch})
    model = build_model("deepar", cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    run_id = make_run_id("deepar", likelihood, spec.dataset, protocol,
                         horizon, seed)
    out_dir = os.path.join(out_root, run_id)
    prepare_run_dir(out_dir, {
        "model": "deepar", "variant": likelihood, "dataset": spec.dataset,
        "protocol": protocol, "horizon": horizon, "seed": seed,
        "spec": spec.to_dict(), "likelihood": likelihood,
        "n_samples": n_samples, "params": n_params,
    })

    scale = data.scale[:cfg.n_tgt].cpu().numpy().astype(np.float64)
    scale3 = data.scale[:cfg.n_tgt].view(1, 1, -1)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weightdecay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg.plateau_factor,
        patience=cfg.plateau_patience)
    best = {"score": float("inf"), "state": None, "epoch": -1}
    bad = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for xb, yb in data.get_batches(data.train[0], data.train[1],
                                       batch, shuffle=True):
            y = yb[:, :, :cfg.n_tgt]
            opt.zero_grad()
            out = model.teacher_forward(xb, y)
            loss = model.compute_loss(out, y, scale3)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss epoch {epoch}")
            loss.backward()
            if cfg.clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            run += float(loss.detach())
            nb += 1
        val = evaluate(model, data, data.valid, batch)
        mae = val["total"]["MAE"]
        if cfg.scheduler == "plateau":
            sched.step(val["total"]["MAPE"])
        else:
            sched.step()
        score = mae
        if select_on_test:
            te = evaluate(model, data, data.test, batch)
            score = te["total"]["MAE"]
        if score < best["score"]:
            best = {"score": score,
                    "state": copy.deepcopy(model.state_dict()),
                    "epoch": epoch}
            bad = 0
        else:
            bad += 1
        sel = "test" if select_on_test else "val"
        print(f"  deepar epoch {epoch:3d} | train {run / max(nb, 1):8.3f} "
              f"| val MAE {mae:8.2f} | best {sel} {best['score']:8.2f}",
              flush=True)
        if cfg.early_stop and bad >= 20:
            break

    model.load_state_dict(best["state"])
    save_checkpoint(out_dir, model)
    q_test, y_test = sample_quantiles(model, data, data.test, batch,
                                      n_samples, seed, scale)
    q_cal, y_cal = sample_quantiles(model, data, data.valid, batch,
                                    n_samples, seed + 999, scale)
    prob = probabilistic_metrics(q_test, y_test, TAUS)
    i50 = int(np.argmin(np.abs(TAUS - 0.5)))
    total, per_target = calc_all_metrics(y_test, q_test[..., i50],
                                         cfg.n_tgt)
    save_predictions(out_dir, y_test, q_test[..., i50], q_test[..., i50])
    save_quantiles(out_dir, q_test, y_test, TAUS)
    save_quantiles(out_dir, q_cal, y_cal, TAUS,
                   name="calib_quantiles.npz")
    row = flatten_metrics(run_id, "deepar", likelihood, spec.dataset,
                          protocol, horizon, seed, n_params,
                          time.time() - t0, total, per_target, prob)
    append_result(os.path.join(out_root, "results.csv"), row)
    print(f"  {run_id} | test MAE {total['MAE']:.2f} "
          f"CRPS {prob['CRPS']:.2f} PICP90 {prob['PICP_90']:.3f}",
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="24,48,72,96")
    ap.add_argument("--seeds", default="2024,2025,2026,2027,2028")
    ap.add_argument("--likelihood", default="gaussian",
                    choices=["gaussian", "student_t"])
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default="results/probabilistic_ar")
    ap.add_argument("--device", default=None)
    ap.add_argument("--protocol", choices=["honest", "official"],
                    default="official")
    args = ap.parse_args()
    device = resolve_device(args.device)
    spec = default_spec()
    for horizon in [int(x) for x in args.horizons.split(",")]:
        for seed in [int(x) for x in args.seeds.split(",")]:
            print(f">>> deepar | h={horizon} | seed={seed}", flush=True)
            train_deepar(horizon, seed, spec, device, args.out,
                         n_samples=args.n_samples, epochs=args.epochs,
                         likelihood=args.likelihood, protocol=args.protocol)


if __name__ == "__main__":
    main()
