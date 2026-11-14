#!/usr/bin/env python3
"""Build point/probabilistic markdown tables from the result CSVs.

Main tables use protocol=official (test-selected, disclosed).  The honest
(val-selected) tables are generated alongside as the robustness appendix.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def _filter(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    if protocol == "both":
        return df
    return df[df["protocol"] == protocol]


def point_table(csv_path: str, protocol: str) -> str:
    frames = [pd.read_csv(csv_path)]
    protocol_csv = os.path.join(os.path.dirname(csv_path), "protocol",
                                "results.csv")
    if os.path.isfile(protocol_csv):
        frames.append(pd.read_csv(protocol_csv))
    df = _filter(pd.concat(frames, ignore_index=True), protocol)
    g = df.groupby(["model", "horizon"])["MAE"].agg(["mean", "std",
                                                     "count"]).reset_index()
    g["cell"] = g.apply(
        lambda r: f"{r['mean']:.1f} ± {r['std']:.1f} (n={int(r['count'])})"
        if pd.notna(r["std"]) else f"{r['mean']:.1f} (n={int(r['count'])})",
        axis=1)
    piv = g.pivot(index="model", columns="horizon", values="cell")
    return piv.to_markdown()


def prob_table(csv_path: str, protocol: str) -> str:
    frames = [pd.read_csv(csv_path)]
    ar_csv = os.path.join(os.path.dirname(csv_path), "..",
                          "probabilistic_ar", "results.csv")
    if os.path.isfile(ar_csv):
        frames.append(pd.read_csv(ar_csv))
    df = _filter(pd.concat(frames, ignore_index=True), protocol)
    g = df.groupby(["model", "horizon"])[["CRPS", "PICP_90", "QCE",
                                          "MAE"]].mean().reset_index()
    g["CRPS"] = g["CRPS"].round(1)
    g["PICP_90"] = g["PICP_90"].round(3)
    g["QCE"] = g["QCE"].round(3)
    g["MAE"] = g["MAE"].round(1)
    piv = g.pivot(index="model", columns="horizon", values="CRPS")
    return piv.to_markdown()


def _write(name: str, path: str, out_dir: str, protocol: str, fn) -> None:
    if not os.path.isfile(path):
        print(f"[skip] {path}")
        return
    text = fn(path, protocol)
    if protocol == "official":
        out = os.path.join(out_dir, f"{name}.md")
    elif protocol == "honest":
        out = os.path.join(out_dir, f"{name}_honest.md")
    else:
        out = os.path.join(out_dir, f"{name}_{protocol}.md")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", default="results/point/results.csv")
    ap.add_argument("--prob", default="results/probabilistic/results.csv")
    ap.add_argument("--out", default="paper/tables")
    ap.add_argument("--protocol", choices=["official", "honest", "both"],
                    default="official",
                    help="official = test-selected main table; honest = "
                         "val-selected appendix")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    protocols = (["official", "honest"] if args.protocol == "both"
                 else [args.protocol])
    for protocol in protocols:
        for name, path, fn in (("point", args.point, point_table),
                               ("probabilistic", args.prob, prob_table)):
            _write(name, path, args.out, protocol, fn)


if __name__ == "__main__":
    main()
