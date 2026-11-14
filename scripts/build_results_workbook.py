#!/usr/bin/env python3
"""Build the analysis workbook from everything under results/.

    python scripts/build_results_workbook.py --out results/PERC-Net_results.xlsx

Sheets
    README          protocol, seeds, how to read the numbers
    T1_point        point accuracy, all models, 4 horizons (official)
    T1b_per_target  per-target MAE/MAPE/RMSE/ACCR
    T2_paths        three exogenous-injection paths
    T3_ablation     component ablation
    T4_prob         CRPS / PICP / PINAW / Winkler / QCE / CVaR, with CQR
    T5_decision     critical-quantile commitment vs median
    T6_honest       validation-selected protocol, proposed vs strongest baseline
    T7_budget       parameters and MACs per model and horizon
    RAW_point / RAW_prob   every row, for traceability
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

TAUS = np.array([0.05, 0.1625, 0.275, 0.3875, 0.5,
                 0.6125, 0.725, 0.8375, 0.95])

PRETTY = {
    "percnet_v3_wx": "PERC-Net (proposed)",
    "festrnn_official": "FESTRNN",
    "festrnn": "FESTRNN",
    "crossformer": "Crossformer",
    "segrnn": "segRNN",
    "mamba": "Mamba",
    "dlinear": "DLinear",
    "itransformer": "iTransformer",
    "film": "FiLM",
    "tft": "TFT",
    "deeparq": "DeepARQ",
    "percnet_v3_noexo": "no exogenous history",
    "percnet_v3_nodeg": "12-channel streams",
    "percnet_v3_wxw": "context, weather only",
    "percnet_v3_plain": "- residual fusion",
    "percnet_v3_wx_nocyc": "- cycle queue",
    "percnet_v3_wx_point": "- quantile head",
    "percnet_v3_wx_ndec": "- decomposition loss",
    "percnet_v3_wx_noanc": "- last-value anchor",
    "percnet_v3_wxh64": "expert width 36->64",
}

MAIN_ORDER = ["percnet_v3_wx", "festrnn", "crossformer", "segrnn",
              "mamba", "dlinear", "itransformer", "film"]
METRIC_COLS = ["MAE", "MAPE", "RMSE", "ACCR"]


def _load(paths):
    frames = [pd.read_csv(p) for p in paths if os.path.isfile(p)]
    if not frames:
        raise SystemExit("no result csv found")
    df = pd.concat(frames, ignore_index=True)
    df["model"] = df["model"].str.replace("_official", "", regex=False)
    return df


def _fmt(df, model_order=None, decimals=1):
    out = df.copy()
    out.index = [PRETTY.get(m, m) for m in out.index]
    if model_order:
        keep = [PRETTY.get(m, m) for m in model_order if
                PRETTY.get(m, m) in out.index]
        rest = [i for i in out.index if i not in keep]
        out = out.loc[keep + rest]
    return out.round(decimals)


def t_point(pt):
    d = pt[pt.protocol == "official"]
    g = d.groupby(["model", "horizon"])["MAE"].agg(["mean", "std"])
    g = g.unstack("horizon")
    g.columns = [f"h{h} {s}" for s, h in g.columns]
    g = g[[f"h{h} {s}" for h in (24, 48, 72, 96) for s in ("mean", "std")]]
    extra = d.groupby("model").agg(params=("params", "first"))
    return _fmt(g.join(extra), MAIN_ORDER)


def t_per_target(pt):
    d = pt[pt.protocol == "official"]
    cols = [f"{k}_{c}" for c in ("KW", "CHW", "HT") for k in METRIC_COLS]
    g = d.groupby(["model", "horizon"])[cols].mean().round(1)
    out = g.unstack("horizon").swaplevel(axis=1)
    out.columns = [f"h{h} {m}" for h, m in out.columns]
    order = [f"h{h} {c}" for h in (24, 48, 72, 96) for c in cols]
    return out[order]


def t_paths(ab):
    order = ["percnet_v3_wx", "percnet_v3_noexo", "percnet_v3_nodeg",
             "percnet_v3_wxw"]
    d = ab[ab.model.isin(order)]
    g = d.groupby("model").agg(
        MAE=("MAE", "mean"), MAE_std=("MAE", "std"), seeds=("seed", "nunique"),
        MAE_KW=("MAE_KW", "mean"), MAE_CHW=("MAE_CHW", "mean"),
        MAE_HT=("MAE_HT", "mean"), MAPE=("MAPE", "mean"))
    g["delta_vs_full"] = (g.MAE - g.loc["percnet_v3_wx", "MAE"]).round(1)
    g = g.loc[order].round(1)
    g.index = [PRETTY.get(m, m) for m in g.index]
    return g


def t_ablation(ab):
    d = ab
    g = d.groupby("model").agg(
        MAE=("MAE", "mean"), MAE_std=("MAE", "std"), seeds=("seed", "nunique"),
        MAE_KW=("MAE_KW", "mean"), MAE_CHW=("MAE_CHW", "mean"),
        MAE_HT=("MAE_HT", "mean"))
    g["delta_vs_full"] = (g.MAE - g.loc["percnet_v3_wx", "MAE"]).round(1)
    g = g.sort_values("MAE").round(1)
    g.index = [PRETTY.get(m, m) for m in g.index]
    return g


def t_prob(pr):
    cols = ["CRPS", "Pinball", "PICP_90", "PINAW_90", "Winkler_90",
            "QCE", "CVaR_90", "MAE"]
    g = pr.groupby(["model", "horizon"])[cols].mean()
    out = g.unstack("horizon").swaplevel(axis=1)
    out.columns = [f"h{h} {m}" for h, m in out.columns]
    cols_ord = [f"h{h} {c}" for h in (24, 48, 72, 96) for c in cols]
    out = out[cols_ord]
    order = ["percnet_v3_wx", "percnet_v3_wx_cqr", "tft", "tft_cqr",
             "deeparq", "deeparq_cqr"]
    out = out.loc[[m for m in order if m in out.index]]
    out.index = [PRETTY.get(m, m).replace("_cqr", "") +
                 (" + CQR" if m.endswith("_cqr") else "") for m in out.index]
    return out.round(3)


def _q_at(q, tau):
    i = int(np.clip(np.searchsorted(TAUS, tau) - 1, 0, len(TAUS) - 2))
    w = (tau - TAUS[i]) / (TAUS[i + 1] - TAUS[i])
    return q[..., i] * (1 - w) + q[..., i + 1] * w


def t_decision():
    rows = []
    for h in (24, 48, 72, 96):
        dirs = sorted(glob.glob(
            f"results/probabilistic/percnet_v3_wx_ASU_official_h{h}_s*"))
        dirs = [d for d in dirs if not d.endswith("_cqr")]
        if not dirs:
            continue
        for tau in (0.10, 0.25, 0.75, 0.90, 0.95):
            cu = tau / (1 - tau)
            deltas = []
            for d in dirs:
                z = np.load(os.path.join(d, "quantiles.npz"))
                q, y = z["quantiles"], z["y_true"]
                q50 = _q_at(q, 0.5)
                ref = (cu * np.maximum(y - q50, 0)
                       + np.maximum(q50 - y, 0)).mean()
                crit = (cu * np.maximum(y - _q_at(q, tau), 0)
                        + np.maximum(_q_at(q, tau) - y, 0)).mean()
                deltas.append(100 * (crit - ref) / ref)
            rows.append({"horizon": h, "cu/co": round(cu, 3), "tau*": tau,
                         "delta_cost_pct": round(float(np.mean(deltas)), 1),
                         "std_pct": round(float(np.std(deltas)), 1),
                         "seeds": len(deltas)})
    return pd.DataFrame(rows)


def t_honest(pt, ho):
    rows = []
    for m in ("percnet_v3_wx", "festrnn"):
        for h in (24, 48, 72, 96):
            o = pt[(pt.model == m) & (pt.horizon == h)].MAE
            n = ho[(ho.model == m) & (ho.horizon == h)].MAE
            if len(o) == 0 or len(n) == 0:
                continue
            rows.append({
                "model": PRETTY.get(m, m), "horizon": h,
                "official_mean": round(o.mean(), 1),
                "official_std": round(o.std(), 1),
                "honest_mean": round(n.mean(), 1), "honest_std": round(n.std(), 1),
                "selection_gain": round(n.mean() - o.mean(), 1)})
    return pd.DataFrame(rows)


def t_budget():
    from measure_flops import measure
    from percnet_epsr.data import default_spec
    import torch
    spec = default_spec()
    dev = torch.device("cpu")
    rows = []
    for m in MAIN_ORDER:
        for h in (24, 48, 72, 96):
            try:
                r = measure(m, h, spec, dev, 1)
                rows.append({"model": PRETTY.get(m, m), "horizon": h,
                             "params": r["params"],
                             "MACs_MMac": round(r["mmac"], 3)})
            except Exception as exc:              # pragma: no cover
                print(f"[warn] {m} h{h}: {exc}")
    return pd.DataFrame(rows)


README_ROWS = [
    ("Dataset", "ASU Tempe campus IES, hourly, 23,807 rows; 3 loads "
                "(KW / CHWTON / HTMBTU) + 9 exogenous + 4 calendar + 4 rolling"),
    ("Split", "contiguous 80/10/10 (train / validation / test), no shuffling"),
    ("Seeds", "2024-2028 (5 seeds) for main tables; 2024-2026 (3 seeds) for "
              "the ablation ladder"),
    ("official", "each epoch evaluated on the test set, best test epoch kept "
                 "(same rule as the released baseline code; optimistically "
                 "biased, stated in every table caption)"),
    ("honest", "checkpoint selected on the validation set, test evaluated "
               "once (Appendix)"),
    ("Point metrics", "MAE / MAPE / RMSE / ACCR, macro-averaged over the "
                      "three load channels"),
    ("Probabilistic metrics", "CRPS = 2 x mean pinball over 9 quantiles "
                              "(tau = 0.05 ... 0.95); PICP90 nominal 0.90; "
                              "PINAW90 sharpness; Winkler90; QCE; CVaR90"),
    ("CQR", "split conformal calibration fitted on validation residuals only"),
    ("MACs", "one forward pass, counted on the operator graph "
             "(matmul / conv / attention / recurrence); FFT excluded"),
    ("Caveat 1", "FESTRNN's published 611.7 MAE is a single seed (2022); the "
                 "five-seed mean under the same protocol is 620.8 +/- 6.4"),
    ("Caveat 2", "the validation segment is summer (TEMP 35.1 C) while the "
                 "test segment is autumn-winter (22.0 C), so validation-based "
                 "selection is pessimistic at long horizons"),
    ("Generated by", "scripts/build_results_workbook.py"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/PERC-Net_results.xlsx")
    args = ap.parse_args()

    pt = _load(["results/point/results.csv", "results/point_b/results.csv",
                "results/point/protocol/results.csv"])
    ho = _load(["results/point_honest/results.csv",
                "results/point_honest/protocol/results.csv"])
    pr = pd.read_csv("results/probabilistic/results.csv")
    ab = pd.read_csv("results/point_ablation_v3/results.csv")

    sheets = {
        "README": pd.DataFrame(README_ROWS, columns=["item", "value"]),
        "T1_point": t_point(pt),
        "T1b_per_target": t_per_target(pt),
        "T2_paths": t_paths(ab),
        "T3_ablation": t_ablation(ab),
        "T4_prob": t_prob(pr),
        "T5_decision": t_decision(),
        "T6_honest": t_honest(pt, ho),
        "T7_budget": t_budget(),
        "RAW_point": pt.sort_values(["model", "horizon", "seed"]),
        "RAW_prob": pr.sort_values(["model", "horizon", "seed"]),
    }

    for name in ("T1_point", "T1b_per_target", "T2_paths", "T3_ablation",
                 "T4_prob"):
        sheets[name].index.name = "model"

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name, index=(name not in
                                                    ("T5_decision", "T6_honest",
                                                     "T7_budget", "README")))
    # formatting
    from openpyxl import load_workbook
    wb = load_workbook(out)
    head_fill = PatternFill("solid", fgColor="1F3864")
    head_font = Font(color="FFFFFF", bold=True, size=10)
    for ws in wb.worksheets:
        ws.freeze_panes = "B2"
        for cell in ws[1]:
            cell.fill = head_fill
            cell.font = head_font
            cell.alignment = Alignment(horizontal="center", vertical="center",
                                       wrap_text=True)
        for col in range(1, ws.max_column + 1):
            width = max((len(str(ws.cell(r, col).value or ""))
                         for r in range(1, min(ws.max_row, 60) + 1)), default=8)
            ws.column_dimensions[get_column_letter(col)].width = min(
                max(10, width * 1.1), 34)
        ws.row_dimensions[1].height = 30
    wb.save(out)
    print(f"wrote {out}")
    for name, df in sheets.items():
        print(f"  {name:16s} {df.shape[0]:4d} rows x {df.shape[1]:2d} cols")


if __name__ == "__main__":
    main()
