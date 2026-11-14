#!/usr/bin/env python3
"""Parameters, multiply-accumulates and forward latency per model / horizon.

MACs are counted by running one forward pass under
``torch.utils.flop_counter.FlopCounterMode``.  The counter covers every
matmul, conv, attention and recurrent step actually executed, and nothing
else: FFTs and elementwise ops are excluded by construction.  The identical
convention is applied to every model, so the column is comparable across the
table -- say so in the caption ("MACs counted on the operator graph,
excluding FFT and elementwise work").

Latency is wall-clock over repeated forward passes on the selected device
and is only meaningful on a GPU.

    python scripts/measure_flops.py                       # default table
    python scripts/measure_flops.py --models percnet_v3_wx,dlinear --horizons 24
    python scripts/measure_flops.py --out results/flops.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percnet_epsr.data import default_spec          # noqa: E402
from percnet_epsr.models.registry import build_model, make_config  # noqa: E402
from percnet_epsr.utils import resolve_device      # noqa: E402


DEFAULT_MODELS = ("percnet_v3_wx,festrnn,dlinear,itransformer,"
                  "mamba,crossformer,film,segrnn")


def _build(name: str, horizon: int, spec, device):
    if name == "festrnn":
        from percnet_epsr.protocols.official_festrnn import build_model as fb
        model = fb(horizon).to(device).eval()
        enc = 12
    else:
        model = build_model(name, make_config(name, horizon, spec))
        model = model.to(device).eval()
        enc = int(spec.enc_in)
    return model, enc


def measure(name: str, horizon: int, spec, device, iters: int):
    model, enc = _build(name, horizon, spec, device)
    x = torch.randn(1, int(spec.seq_len), enc, device=device)
    with torch.no_grad():
        try:
            with FlopCounterMode(display=False) as fc:
                model(x)
            flops = int(fc.get_total_flops())
        except Exception as exc:                    # pragma: no cover
            print(f"  [warn] {name} h{horizon}: flop counter failed ({exc})")
            flops = -1
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat = (time.time() - t0) / iters * 1000.0
    return {
        "model": name,
        "horizon": horizon,
        "params": sum(p.numel() for p in model.parameters()),
        "mmac": flops / 2e6 if flops >= 0 else float("nan"),
        "latency_ms": lat,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--horizons", default="24,48,72,96")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="optional CSV path")
    args = ap.parse_args()

    device = resolve_device(args.device)
    spec = default_spec()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    print(f"device={device}  (MACs per sample, one forward pass)")

    rows = []
    for name in models:
        for h in horizons:
            row = measure(name, h, spec, device, args.iters)
            rows.append(row)
            print(f"{row['model']:18s} h{row['horizon']:>3} "
                  f"params {row['params']:>9,}  "
                  f"MMac {row['mmac']:>9.3f}  "
                  f"fwd {row['latency_ms']:>7.2f} ms", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
