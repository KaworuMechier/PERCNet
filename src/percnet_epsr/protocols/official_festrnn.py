"""FESTRNN protocol: official (test-selected) and honest (val-selected).

``official`` reproduces the released main.py: global max-abs scaling over the
full series, 80/10/10 contiguous split, MAPE loss on de-normalised values,
Adam(8e-4, wd=1e-5), ReduceLROnPlateau(factor=0.5, patience=2) on val MAPE,
50 epochs, seed 2022, and **best test MAE over epochs**.  Expected ~611.7.

``honest`` uses the same data/model/recipe but selects the checkpoint on
validation MAE and evaluates test once.  Expected ~630.

The 611.7 number is an implementation-fidelity anchor, not a ranking protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from percnet_epsr.training.artifacts import (append_result, flatten_metrics,
                                             prepare_run_dir, save_checkpoint,
                                             save_history, save_predictions)
from percnet_epsr.metrics import calc_all_metrics
from percnet_epsr.utils import resolve_device


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def mape_loss(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Official util.mape_loss: eps=1e-2, x100, raw units."""
    return ((pred - target).abs() / (target.abs() + 1e-2)).mean() * 100.0


def _mae(y, p) -> float:
    return float(np.abs(np.asarray(y) - np.asarray(p)).mean())


def _mape(y, p) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean(np.abs((y - p) / y)) * 100.0)


def _rmse(y, p) -> float:
    e = np.asarray(y, dtype=np.float64) - np.asarray(p, dtype=np.float64)
    return float(np.sqrt((e * e).mean()))


class PlateauUp:
    """ReduceLROnPlateau(factor=0.5, patience=2) without version quirks.

    Note: the official config's ``weightdecay: 0.5`` is *this factor*, not
    Adam's weight decay (which is ``args.weight_decay = 1e-5``).
    """

    def __init__(self, optimizer, factor: float = 0.5, patience: int = 2):
        self.optimizer = optimizer
        self.factor = float(factor)
        self.patience = int(patience)
        self.best = float("inf")
        self.bad = 0

    def step(self, value: float) -> None:
        if value < self.best - 1e-8:
            self.best = value
            self.bad = 0
            return
        self.bad += 1
        if self.bad >= self.patience:
            for group in self.optimizer.param_groups:
                group["lr"] *= self.factor
            self.bad = 0


class OfficialLoader:
    """Global max-abs scaling over the full series; 80/10/10 windows."""

    def __init__(self, file_name: str, horizon: int, seq_len: int = 168,
                 train: float = 0.8, valid: float = 0.1,
                 device: str | torch.device = "cpu"):
        self.P = int(seq_len)
        self.h = int(horizon)
        self.device = torch.device(device)
        raw = np.loadtxt(file_name, delimiter=",", skiprows=1)
        if raw.ndim != 2 or raw.shape[1] < 12:
            raise ValueError(f"expected >=12 columns, got {raw.shape}")
        scale = np.max(np.abs(raw), axis=0)
        self._dat_cache = raw / scale
        self.scale = torch.tensor(scale[:3], dtype=torch.float32,
                                  device=self.device)
        n = len(self._dat_cache)
        train_end = int(train * n)
        valid_end = int((train + valid) * n)
        offset = self.P + self.h - 1
        self.train = self._batchify(range(offset, train_end))
        self.valid = self._batchify(range(train_end, valid_end))
        self.test = self._batchify(range(valid_end, n))
        self.n_train = train_end - offset
        self.n_valid = valid_end - train_end
        self.n_test = n - valid_end

    def _batchify(self, idx_set):
        idx = list(idx_set)
        ends = [i - self.h + 1 for i in idx]
        starts = [e - self.P for e in ends]
        dat = self._dat_cache
        X = np.stack([dat[s:e, :] for s, e in zip(starts, ends)])
        Y = np.stack([dat[i + 1 - self.h:i + 1, :] for i in idx])
        return [torch.from_numpy(X).float(), torch.from_numpy(Y).float()]

    def get_batches(self, inputs, targets, batch_size: int,
                    shuffle: bool = True):
        index = (torch.randperm(len(inputs)) if shuffle
                 else torch.arange(len(inputs)))
        for s in range(0, len(inputs), batch_size):
            sel = index[s:s + batch_size]
            yield (inputs[sel].to(self.device),
                   targets[sel].to(self.device))


@torch.no_grad()
def evaluate(model, loader, split, batch: int):
    model.eval()
    preds, trues = [], []
    X, Y = split
    for Xb, Yb in loader.get_batches(X, Y, batch, shuffle=False):
        out = model(Xb)
        preds.append(out.cpu())
        trues.append(Yb[:, :, :3].cpu())
    pred = torch.cat(preds).numpy() * loader.scale.cpu().numpy().reshape(1, 1, 3)
    true = torch.cat(trues).numpy() * loader.scale.cpu().numpy().reshape(1, 1, 3)
    return {"MAE": _mae(true, pred), "MAPE": _mape(true, pred),
            "RMSE": _rmse(true, pred)}, pred, true


def build_model(horizon: int):
    from percnet_epsr.models.baselines.festrnn_official import Model
    cfg = SimpleNamespace(seq_len=168, pred_len=horizon, enc_in=12,
                          hidden_size=64, d_model=16, seg_len=24,
                          dropout=0.4)
    return Model(cfg)


def run(horizon: int, mode: str, device: torch.device, data_file: str,
        out_root: str, seed: int = 2022, epochs: int = 50,
        batch: int = 64, clip: float | None = None,
        double_init: bool = False) -> dict:
    fix_seed(seed)
    loader = OfficialLoader(data_file, horizon, device=device)
    # The released main.py builds the model twice (FLOPs profiling between the
    # two constructions).  The second construction therefore sees a different
    # RNG stream.  --double-init reproduces that sequence.
    if double_init:
        dummy = build_model(horizon).to(device)
        with torch.no_grad():
            _ = dummy(torch.randn(1, 168, 12, device=device))
        del dummy
    model = build_model(horizon).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4,
                                 weight_decay=1e-5)
    # Use PyTorch's exact scheduler (threshold_mode='rel', cooldown=0), which
    # is what the official trainer.py instantiates.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)

    run_name = f"festrnn_ASU_{mode}_h{horizon}_s{seed}"
    out_dir = os.path.join(out_root, run_name)
    config = {
        "model": "festrnn_official", "variant": mode, "dataset": "ASU",
        "protocol": mode, "horizon": horizon, "seed": seed,
        "data_file": data_file, "epochs": epochs, "batch": batch,
        "lr": 8e-4, "weight_decay": 1e-5, "factor": 0.5, "patience": 2,
        "loss": "mape_raw_eps1e-2", "clip": clip,
        "double_init": double_init, "params": n_params,
    }
    prepare_run_dir(out_dir, config)

    history = []
    best_test = {"MAE": float("inf"), "epoch": -1, "pred": None,
                 "true": None}
    best_val = {"MAE": float("inf"), "epoch": -1, "state": None}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running, n_batch = 0.0, 0
        for Xb, Yb in loader.get_batches(loader.train[0], loader.train[1],
                                         batch, shuffle=True):
            yb = Yb[:, :, :3]
            optimizer.zero_grad()
            pred = model(Xb)
            s3 = loader.scale.view(1, 1, 3)
            loss = mape_loss(yb * s3, pred * s3)
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            running += float(loss.detach())
            n_batch += 1

        val, _, _ = evaluate(model, loader, loader.valid, batch)
        scheduler.step(val["MAPE"])
        row = {"epoch": epoch, "train_loss": running / max(n_batch, 1),
               "val_mae": val["MAE"], "val_mape": val["MAPE"],
               "lr": optimizer.param_groups[0]["lr"]}

        if mode == "official":
            test, pred, true = evaluate(model, loader, loader.test, batch)
            row.update({"test_mae": test["MAE"], "test_mape": test["MAPE"],
                        "test_rmse": test["RMSE"]})
            if test["MAE"] < best_test["MAE"]:
                best_test = {"MAE": test["MAE"], "epoch": epoch,
                             "pred": pred.copy(), "true": true.copy(),
                             "MAPE": test["MAPE"], "RMSE": test["RMSE"]}
        else:
            if val["MAE"] < best_val["MAE"]:
                best_val = {"MAE": val["MAE"], "epoch": epoch,
                            "state": copy.deepcopy(model.state_dict())}
        history.append(row)
        print(f"  epoch {epoch:3d} | train {row['train_loss']:8.2f} | "
              f"val MAE {val['MAE']:8.2f} MAPE {val['MAPE']:5.2f}% | "
              f"lr {row['lr']:.2e}", flush=True)

    if mode == "official":
        result = {"MAE": best_test["MAE"], "MAPE": best_test.get("MAPE"),
                  "RMSE": best_test.get("RMSE"),
                  "epoch": best_test["epoch"],
                  "pred": best_test["pred"], "true": best_test["true"]}
    else:
        model.load_state_dict(best_val["state"])
        test, pred, true = evaluate(model, loader, loader.test, batch)
        result = {"MAE": test["MAE"], "MAPE": test["MAPE"],
                  "RMSE": test["RMSE"], "epoch": best_val["epoch"],
                  "pred": pred, "true": true}
        save_checkpoint(out_dir, model)

    save_history(out_dir, history)
    if result["pred"] is not None:
        save_predictions(out_dir, result["true"], result["pred"])
    total, per_target = calc_all_metrics(result["true"], result["pred"])
    row = flatten_metrics(
        run_name, "festrnn_official", mode, "ASU", mode, horizon, seed,
        n_params, time.time() - t0, total, per_target)
    append_result(os.path.join(out_root, "results.csv"), row)
    print(f"\n[{mode}] h={horizon} seed={seed} | epoch {result['epoch']} | "
          f"test MAE {result['MAE']:.2f} MAPE {result['MAPE']:.2f}% "
          f"RMSE {result['RMSE']:.1f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--mode", choices=["official", "honest"],
                    default="official")
    ap.add_argument("--device", default=None)
    ap.add_argument("--data", default="data/dataset_input.csv")
    ap.add_argument("--out", default="results/protocol")
    ap.add_argument("--seed", type=int, default=2022)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--clip", type=float, default=None,
                    help="official code clips only the first step; default off")
    ap.add_argument("--double-init", action="store_true",
                    help="reproduce the official two-model construction / "
                         "RNG stream")
    args = ap.parse_args()
    device = resolve_device(args.device)
    print(f"FESTRNN {args.mode} | h={args.horizon} seed={args.seed} "
          f"device={device}")
    run(args.horizon, args.mode, device, args.data, args.out,
        seed=args.seed, epochs=args.epochs, batch=args.batch,
        clip=args.clip, double_init=args.double_init)


if __name__ == "__main__":
    main()
