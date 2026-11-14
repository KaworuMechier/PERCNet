"""Run artifacts and the single result-table schema.

Every run directory is self-describing:

    config.json            model/variant/protocol/horizon/seed/feature spec/hash
    best.pt                selected checkpoint (state_dict)
    predictions.npz        y_true, y_mu, y_q50
    quantiles.npz          test quantiles, y_true, taus
    calib_quantiles.npz    validation quantiles for CQR
    history.json           per-epoch training/validation trace
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


POINT_KEYS = ["MAE", "MAPE", "RMSE", "CV_RMSE", "sMAPE", "NMBE", "ACCR"]
PROB_KEYS = ["CRPS", "Pinball", "PICP_90", "PINAW_90", "Winkler_90",
             "QCE", "CVaR_90"]

RESULT_HEADER = [
    "run_id", "model", "variant", "dataset", "protocol", "setting",
    "horizon", "seed", "params", "train_time_s",
    *POINT_KEYS, *PROB_KEYS,
    "MAE_KW", "MAPE_KW", "RMSE_KW", "ACCR_KW",
    "MAE_CHW", "MAPE_CHW", "RMSE_CHW", "ACCR_CHW",
    "MAE_HT", "MAPE_HT", "RMSE_HT", "ACCR_HT",
]


def load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def make_run_id(model: str, variant: str, dataset: str, protocol: str,
                horizon: int, seed: int) -> str:
    parts = [model]
    if variant:
        parts.append(variant)
    parts.extend([dataset, protocol, f"h{horizon}", f"s{seed}"])
    return "_".join(parts)


def code_signature() -> str:
    """Short hash of the model sources.

    A checkpoint is only meaningful for the code that produced it: changing
    a forward pass keeps every parameter name, so ``load_state_dict`` still
    succeeds and silently misinterprets the weights.  Storing this next to
    the run lets readers (and ``scripts/diag_gap.py``) detect that.
    """
    root = Path(__file__).resolve().parents[1] / "models"
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        if p.name == "registry.py":
            continue          # recipes, not code: model_cfg records those
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def prepare_run_dir(out_dir: str, config: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    payload = dict(config)
    payload.setdefault("code_hash", code_signature())
    payload["config_hash"] = config_hash(config)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    return payload["config_hash"]


def save_checkpoint(out_dir: str, model: torch.nn.Module) -> None:
    torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))


def save_predictions(out_dir: str, y_true, y_mu, y_q50=None) -> None:
    arrays = {"y_true": np.asarray(y_true),
              "y_mu": np.asarray(y_mu)}
    if y_q50 is not None:
        arrays["y_q50"] = np.asarray(y_q50)
    np.savez(os.path.join(out_dir, "predictions.npz"), **arrays)


def save_quantiles(out_dir: str, quantiles, y_true, taus,
                   name: str = "quantiles.npz") -> None:
    np.savez(os.path.join(out_dir, name),
             quantiles=np.asarray(quantiles),
             y_true=np.asarray(y_true),
             taus=np.asarray(taus))


def save_history(out_dir: str, history: list[dict]) -> None:
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2, default=float)


def append_result(csv_path: str, row: dict[str, Any]) -> bool:
    """Append one row; skip if the run_id already exists."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    exists = os.path.exists(csv_path)
    if exists:
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                if r.get("run_id") == str(row.get("run_id")):
                    return False
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_HEADER, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULT_HEADER})
    return True


def flatten_metrics(run_id: str, model: str, variant: str, dataset: str,
                    protocol: str, horizon: int, seed: int, params: int,
                    train_time_s: float, total: dict,
                    per_target: dict, prob: dict | None = None) -> dict:
    row = {
        "run_id": run_id, "model": model, "variant": variant,
        "dataset": dataset, "protocol": protocol, "setting": "base",
        "horizon": horizon, "seed": seed, "params": params,
        "train_time_s": round(float(train_time_s), 2),
    }
    for k in POINT_KEYS:
        row[k] = round(float(total[k]), 6)
    if prob:
        for k in PROB_KEYS:
            row[k] = round(float(prob[k]), 6)
    names = ("KW", "CHW", "HT")
    for i, name in enumerate(names):
        if i not in per_target:
            continue
        for k in ("MAE", "MAPE", "RMSE", "ACCR"):
            row[f"{k}_{name}"] = round(float(per_target[i][k]), 6)
    return row
