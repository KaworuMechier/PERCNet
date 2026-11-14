#!/usr/bin/env python3
"""Fill a comparison workbook with the results measured in this repository.

Rows whose model name matches a model we ran are overwritten (metrics,
parameters, MACs and the shared training recipe); rows for models we did not
run are left untouched.  The original workbook is never modified -- output
goes to ``<name>_filled.xlsx`` unless --out says otherwise.

    python scripts/fill_excel.py \
        --xlsx ~/Downloads/基于深度学习的综合能源负荷预测对比实验数据统计_PERC-Net增补.xlsx

Metric columns (1-indexed, sheet "工作表1"):
    5 params, 6 FLOPs, 7 time, 8 MAE, 9 MAPE, 10 RMSE, 11 ACCR,
    12-15 electricity (KW), 16-19 cooling (CHW), 20-23 heating (HT),
    24 lr, 25 epochs, 26 lr_decay, 27 decay_patience
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_flops import measure                       # noqa: E402
from percnet_epsr.data import default_spec              # noqa: E402

# workbook name (normalised) -> model name in this repository
MODEL_MAP = {
    "dlinear": "dlinear",
    "film": "film",
    "itransformer": "itransformer",
    "mamba": "mamba",
    "crossformer": "crossformer",
    "segrnn": "segrnn",
    "festrnn": "festrnn",
    "percnet": "percnet_v3_wx",
}

RECIPE = {"lr": 8e-4, "epochs": 50, "lr_decay": 0.5, "patience": 2}
METRIC_COLS = ["MAE", "MAPE", "RMSE", "ACCR",
               "MAE_KW", "MAPE_KW", "RMSE_KW", "ACCR_KW",
               "MAE_CHW", "MAPE_CHW", "RMSE_CHW", "ACCR_CHW",
               "MAE_HT", "MAPE_HT", "RMSE_HT", "ACCR_HT"]
DECIMALS = {"MAE": 2, "MAPE": 2, "RMSE": 2, "ACCR": 4}


def load_results(paths):
    """official rows, mean over seeds per (model, horizon)."""
    frames = []
    for p in paths:
        if os.path.isfile(p):
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("no result csv found")
    df = pd.concat(frames, ignore_index=True)
    df = df[df.protocol == "official"]
    df["model"] = df["model"].str.replace("_official", "", regex=False)
    keep = ["model", "horizon", "seed"] + METRIC_COLS
    df = df[keep].drop_duplicates(["model", "horizon", "seed"])
    agg = df.groupby(["model", "horizon"])[METRIC_COLS].mean()
    counts = df.groupby(["model", "horizon"]).size()
    return agg, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sheet", default="工作表1")
    ap.add_argument("--results", nargs="+", default=[
        "results/point/results.csv",
        "results/point_b/results.csv",
        "results/point/protocol/results.csv",
    ])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out = args.out or str(Path(args.xlsx).with_name(
        Path(args.xlsx).stem + "_filled.xlsx"))
    agg, counts = load_results([str(ROOT / p) if not os.path.isabs(p) else p
                                for p in args.results])
    print(f"loaded {len(agg)} (model, horizon) cells")

    spec = default_spec()
    flop_cache, param_cache = {}, {}
    for (model, horizon) in agg.index:
        if model not in MODEL_MAP.values():
            continue
        key = (model, int(horizon))
        row = measure(model, int(horizon), spec, torch_device(args.device), 3)
        flop_cache[key] = row["mmac"]
        param_cache[key] = row["params"]

    wb = load_workbook(args.xlsx)
    ws = wb[args.sheet]
    filled, skipped = [], []
    for r in range(3, ws.max_row + 1):
        name = ws.cell(r, 2).value
        h = ws.cell(r, 4).value
        if name is None or h is None:
            continue
        # "PERC-Net" -> "percnet", "Crossformer " -> "crossformer"
        key = re.sub(r"[^a-z]", "", str(name).lower())
        model = MODEL_MAP.get(key)
        if model is None or (model, int(h)) not in agg.index:
            skipped.append(f"{name} h{h}")
            continue
        vals = agg.loc[(model, int(h))]
        seed_n = int(counts.loc[(model, int(h))])
        ws.cell(r, 5).value = f"{param_cache[(model, int(h))] / 1000:.2f}K"
        ws.cell(r, 6).value = f"{flop_cache[(model, int(h))]:.3f}MMac"
        for i, col in enumerate(METRIC_COLS):
            base = col.split("_")[0]
            ws.cell(r, 8 + i).value = round(float(vals[col]),
                                            DECIMALS.get(base, 2))
        ws.cell(r, 24).value = RECIPE["lr"]
        ws.cell(r, 25).value = RECIPE["epochs"]
        ws.cell(r, 26).value = RECIPE["lr_decay"]
        ws.cell(r, 27).value = RECIPE["patience"]
        filled.append(f"{name} h{h} (n={seed_n})")

    wb.save(out)
    print(f"\nfilled {len(filled)} rows:")
    for f in filled:
        print("  ", f)
    if skipped:
        print(f"\nleft untouched ({len(skipped)}): "
              f"{', '.join(skipped[:12])}{' ...' if len(skipped) > 12 else ''}")
    print(f"\nwrote {out}")


def torch_device(name):
    import torch
    return torch.device(name)


if __name__ == "__main__":
    main()
