"""Honest-protocol training runner.

Validation-selected checkpoint, test evaluated once, fixed run artifacts.
The runner covers PERC-Net v2 and point baselines; autoregressive/quantile
baselines (TFT, DeepAR, DeepARQ) use their own training loops in
``experiments/probabilistic.py``.
"""

from __future__ import annotations

import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from percnet_epsr.data import DataLoaderS, FeatureSpec
from percnet_epsr.metrics import TAUS, calc_all_metrics, probabilistic_metrics
from percnet_epsr.models.registry import build_model, make_config
from percnet_epsr.training.artifacts import (append_result, flatten_metrics,
                                             make_run_id, prepare_run_dir,
                                             save_checkpoint, save_history,
                                             save_predictions, save_quantiles)


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _physical(t: torch.Tensor, scale: np.ndarray,
              offset: np.ndarray) -> np.ndarray:
    a = t.cpu().numpy().astype(np.float64)
    return a * scale.reshape(1, 1, -1) + offset.reshape(1, 1, -1)


@torch.no_grad()
def evaluate(model, data: DataLoaderS, split, batch: int):
    model.eval()
    n_tgt = int(data.spec.n_tgt)
    scale = data.scale[:n_tgt].cpu().numpy().astype(np.float64)
    offset = data.offset[:n_tgt].cpu().numpy().astype(np.float64)
    qs, preds, trues = [], [], []
    X, Y = split
    for Xb, Yb in data.get_batches(X, Y, batch, shuffle=False):
        out = model(Xb)
        y = Yb[:, :, :n_tgt]
        if isinstance(out, dict):
            mu = out["mu"]
            q = out.get("quantiles")
            point = q[..., int(np.argmin(np.abs(TAUS - 0.5)))] \
                if q is not None else mu
        else:
            point = out
            if point.shape[1] > data.h:
                point = point[:, -data.h:, :]
            point = point[:, :, :n_tgt]
            q = None
        preds.append(point.cpu())
        trues.append(y.cpu())
        if q is not None:
            qs.append(q.cpu())

    y_true = _physical(torch.cat(trues), scale, offset)
    y_pred = _physical(torch.cat(preds), scale, offset)
    total, per_target = calc_all_metrics(y_true, y_pred, n_tgt)
    prob = None
    q_phys = None
    if qs:
        q_phys = (torch.cat(qs).cpu().numpy().astype(np.float64)
                  * scale.reshape(1, 1, -1, 1)
                  + offset.reshape(1, 1, -1, 1))
        prob = probabilistic_metrics(q_phys, y_true, TAUS)
    return {"total": total, "per_target": per_target, "prob": prob,
            "y_true": y_true, "y_pred": y_pred, "quantiles": q_phys}


def _loss(model, cfg, out, y, scale3):
    if isinstance(out, dict) and hasattr(model, "compute_loss"):
        return model.compute_loss(out, y, scale3)
    pred = out
    if pred.shape[1] > y.shape[1]:
        pred = pred[:, -y.shape[1]:, :]
    pred = pred[:, :, :y.shape[2]]
    if cfg.loss_type == "mape_raw":
        y_raw, p_raw = y * scale3, pred * scale3
        return ((p_raw - y_raw).abs() / (y_raw.abs() + 1e-2)).mean() * 100.0
    if cfg.loss_type == "swl1":
        return F.l1_loss(pred, y)
    raise ValueError(f"unsupported loss_type {cfg.loss_type!r} for this model")


def _ema_init(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _ema_update(ema, model, decay=0.999):
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point:
            ema[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
        else:
            ema[k] = v.detach().clone()


def train_one(model_name: str, horizon: int, seed: int, spec: FeatureSpec,
              device, out_root: str = "results", protocol: str = "honest",
              overrides: dict | None = None) -> dict:
    if protocol not in ("honest", "official"):
        raise ValueError(f"unknown protocol {protocol!r}")
    select_on_test = protocol == "official"
    if select_on_test:
        print("  [protocol] official: selecting the best TEST epoch "
              "(test-selected, optimistic); report the protocol explicitly "
              "and pair it with the honest val-selected appendix", flush=True)
    fix_seed(seed)
    data = DataLoaderS(spec, horizon, device)
    cfg = make_config(model_name, horizon, spec, overrides)
    model = build_model(model_name, cfg).to(device)
    if hasattr(model, "load_scale"):
        model.load_scale.copy_(data.scale[:cfg.n_free].float())
    n_params = sum(p.numel() for p in model.parameters())

    run_id = make_run_id(model_name, "", spec.dataset, protocol,
                         horizon, seed)
    out_dir = os.path.join(out_root, run_id)
    prepare_run_dir(out_dir, {
        "model": model_name, "dataset": spec.dataset, "protocol": protocol,
        "horizon": horizon, "seed": seed, "spec": spec.to_dict(),
        "model_cfg": {k: v for k, v in vars(cfg).items()
                      if isinstance(v, (int, float, str, bool, type(None)))},
        "params": n_params,
    })

    scale3 = data.scale[:cfg.n_tgt].view(1, 1, -1)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weightdecay)
    if cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.plateau_factor,
            patience=cfg.plateau_patience)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr / 10.0)

    ema = _ema_init(model) if getattr(cfg, "use_ema", False) else None
    best = {"score": float("inf"), "epoch": -1, "state": None}
    history = []
    bad = 0
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, n_batch = 0.0, 0
        for Xb, Yb in data.get_batches(data.train[0], data.train[1],
                                       cfg.batchsize, shuffle=True):
            y = Yb[:, :, :cfg.n_tgt]
            optimizer.zero_grad()
            if getattr(cfg, "teacher_forcing", False) and hasattr(
                    model, "teacher_forward"):
                out = model.teacher_forward(Xb, y)
            else:
                out = model(Xb)
            loss = _loss(model, cfg, out, y, scale3)
            loss.backward()
            if cfg.clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step()
            running += float(loss.detach())
            if n_batch == 0 and torch.cuda.is_available():
                print(f"  [mem] peak allocated "
                      f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB "
                      f"| reserved "
                      f"{torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GiB",
                      flush=True)
            n_batch += 1
        if ema is not None:
            _ema_update(ema, model)

        val = evaluate(model, data, data.valid, cfg.batchsize)
        val_mae = val["total"]["MAE"]
        val_mape = val["total"]["MAPE"]
        if cfg.scheduler == "plateau":
            scheduler.step(val_mape)
        else:
            scheduler.step()
        row = {"epoch": epoch,
               "train_loss": running / max(n_batch, 1),
               "val_mae": val_mae, "val_mape": val_mape,
               "lr": optimizer.param_groups[0]["lr"]}
        score = val_mae
        if select_on_test:
            test_epoch = evaluate(model, data, data.test, cfg.batchsize)
            score = test_epoch["total"]["MAE"]
            row["test_mae"] = score
        history.append(row)
        if score < best["score"]:
            best = {"score": score, "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
            bad = 0
        else:
            bad += 1
        if epoch % 10 == 0 or epoch == 1:
            sel = "test" if select_on_test else "val"
            print(f"  {run_id} epoch {epoch:3d} | train "
                  f"{running / max(n_batch, 1):8.3f} | val MAE "
                  f"{val_mae:8.2f} | best {sel} {best['score']:8.2f}",
                  flush=True)
        if cfg.early_stop and bad >= 20:
            print(f"  early stop at epoch {epoch}", flush=True)
            break

    # Prefer the EMA weights when they improve the selection metric.
    if ema is not None:
        raw_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(ema)
        ema_eval = evaluate(model, data,
                            data.test if select_on_test else data.valid,
                            cfg.batchsize)
        ema_score = ema_eval["total"]["MAE"]
        if ema_score < best["score"]:
            best = {"score": ema_score, "epoch": -1,
                    "state": copy.deepcopy(ema)}
            print(f"  EMA selected | score {best['score']:.2f}", flush=True)
        model.load_state_dict(raw_state)

    model.load_state_dict(best["state"])
    save_checkpoint(out_dir, model)
    test = evaluate(model, data, data.test, cfg.batchsize)
    calib = evaluate(model, data, data.valid, cfg.batchsize)

    i50 = int(np.argmin(np.abs(TAUS - 0.5)))
    save_predictions(out_dir, test["y_true"], test["y_pred"],
                     None if test["quantiles"] is None
                     else test["quantiles"][..., i50])
    if test["quantiles"] is not None:
        save_quantiles(out_dir, test["quantiles"], test["y_true"], TAUS)
        save_quantiles(out_dir, calib["quantiles"], calib["y_true"], TAUS,
                       name="calib_quantiles.npz")
    save_history(out_dir, history)

    row = flatten_metrics(
        run_id, model_name, "", spec.dataset, protocol, horizon, seed,
        n_params, time.time() - t0, test["total"], test["per_target"],
        test["prob"])
    append_result(os.path.join(out_root, "results.csv"), row)
    msg = (f"  {run_id} | test MAE {test['total']['MAE']:.2f} "
           f"MAPE {test['total']['MAPE']:.2f}%")
    if test["prob"]:
        msg += f" | CRPS {test['prob']['CRPS']:.2f}"
    print(msg, flush=True)
    return row
