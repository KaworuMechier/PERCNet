"""xCPD plugin on/off ablation on a frozen PERC-Net v2 backbone."""

from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np
import torch

from percnet_epsr.data import DataLoaderS, default_spec
from percnet_epsr.metrics import TAUS, calc_all_metrics
from percnet_epsr.models.registry import build_model, build_xcpd, make_config
from percnet_epsr.training.artifacts import (append_result, flatten_metrics,
                                             make_run_id, prepare_run_dir,
                                             save_checkpoint, save_predictions,
                                             save_quantiles)
from percnet_epsr.training.runner import evaluate, fix_seed, train_one
from percnet_epsr.utils import resolve_device


def patch_for(horizon: int) -> int:
    for p in (6, 8, 12, 4, 3):
        if horizon % p == 0:
            return p
    return 1


def load_backbone(name, horizon, seed, spec, device, point_root,
                  protocol="official"):
    run_id = make_run_id(name, "", spec.dataset, protocol, horizon, seed)
    ckpt = os.path.join(point_root, run_id, "best.pt")
    if not os.path.isfile(ckpt):
        print(f"  backbone missing ({ckpt}); training it first", flush=True)
        train_one(name, horizon, seed, spec, device, out_root=point_root,
                  protocol=protocol)
    cfg = make_config(name, horizon, spec)
    model = build_model(name, cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, ckpt


def train_plugin(model, data, cfg, device, out_dir, epochs=50, lr=8e-4,
                 wd=1e-5, patience=60, select_on_test=False):
    scale3 = data.scale[:cfg.n_tgt].view(1, 1, -1)
    opt = torch.optim.Adam(model.plugin.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2)
    best = {"score": float("inf"), "state": None, "epoch": -1}
    bad = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for xb, yb in data.get_batches(data.train[0], data.train[1],
                                       cfg.batchsize, shuffle=True):
            y = yb[:, :, :cfg.n_tgt]
            opt.zero_grad()
            out = model(xb)
            loss = model.compute_loss(out, y, scale3)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.plugin.parameters(), 5.0)
            opt.step()
            run += float(loss.detach())
            nb += 1
        val = evaluate(model, data, data.valid, cfg.batchsize)
        mae = val["total"]["MAE"]
        sched.step(val["total"]["MAPE"])
        score = mae
        if select_on_test:
            te = evaluate(model, data, data.test, cfg.batchsize)
            score = te["total"]["MAE"]
        if score < best["score"]:
            best = {"score": score, "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
            bad = 0
        else:
            bad += 1
        sel = "test" if select_on_test else "val"
        print(f"  plugin epoch {epoch:3d} | train {run / max(nb, 1):8.3f} "
              f"| val MAE {mae:8.2f} | best {sel} {best['score']:8.2f}",
              flush=True)
        if bad >= patience:
            break
    model.load_state_dict(best["state"])
    save_checkpoint(out_dir, model)
    return time.time() - t0, best["epoch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="percnet_v2_eesm")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--seeds", default="2024,2025,2026")
    ap.add_argument("--device", default=None)
    ap.add_argument("--point-root", default="results/point")
    ap.add_argument("--out", default="results/xcpd")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--protocol", choices=["honest", "official"],
                    default="official")
    args = ap.parse_args()

    device = resolve_device(args.device)
    spec = default_spec()
    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        fix_seed(seed)
        print(f">>> xCPD | backbone={args.backbone} h={args.horizon} "
              f"seed={seed}", flush=True)
        data = DataLoaderS(spec, args.horizon, device)
        backbone, ckpt = load_backbone(args.backbone, args.horizon, seed,
                                       spec, device, args.point_root,
                                       protocol=args.protocol)
        pcfg = make_config("percnet_v2_xcpd", args.horizon, spec)
        pcfg.patch_len = patch_for(args.horizon)
        wrapper = build_xcpd(pcfg, backbone).to(device)

        run_id = make_run_id("percnet_v2_xcpd", "frozen", spec.dataset,
                             args.protocol, args.horizon, seed)
        out_dir = os.path.join(args.out, run_id)
        prepare_run_dir(out_dir, {
            "model": "percnet_v2_xcpd", "variant": "frozen",
            "backbone": args.backbone, "backbone_ckpt": ckpt,
            "dataset": spec.dataset, "protocol": args.protocol,
            "horizon": args.horizon, "seed": seed,
            "patch_len": pcfg.patch_len,
            "plugin_params": sum(p.numel()
                                 for p in wrapper.plugin.parameters()),
        })

        train_time, best_epoch = train_plugin(
            wrapper, data, pcfg, device, out_dir, epochs=args.epochs,
            select_on_test=args.protocol == "official")

        test = evaluate(wrapper, data, data.test, pcfg.batchsize)
        calib = evaluate(wrapper, data, data.valid, pcfg.batchsize)
        i50 = int(np.argmin(np.abs(TAUS - 0.5)))
        save_predictions(out_dir, test["y_true"], test["y_pred"],
                         test["quantiles"][..., i50]
                         if test["quantiles"] is not None else None)
        if test["quantiles"] is not None:
            save_quantiles(out_dir, test["quantiles"], test["y_true"], TAUS)
            save_quantiles(out_dir, calib["quantiles"], calib["y_true"],
                           TAUS, name="calib_quantiles.npz")
        row = flatten_metrics(
            run_id, "percnet_v2_xcpd", "frozen", spec.dataset, args.protocol,
            args.horizon, seed,
            sum(p.numel() for p in wrapper.plugin.parameters()),
            train_time, test["total"], test["per_target"], test["prob"])
        append_result(os.path.join(args.out, "results.csv"), row)
        print(f"  xCPD h={args.horizon} s={seed} | epoch {best_epoch} | "
              f"MAE {test['total']['MAE']:.2f} "
              f"CRPS {test['prob']['CRPS']:.2f}", flush=True)


if __name__ == "__main__":
    main()
