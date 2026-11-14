#!/usr/bin/env python3
"""Functional checks — run on the 3090, never locally.

Covers shapes, parameter counts, forward/backward, identity initialisation of
the xCPD plugin, the official FESTRNN model, CQR coverage, and a tiny
deterministic dispatch solve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# Allow `python scripts/smoke_3090.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percnet_epsr.data import DataLoaderS, default_spec
from percnet_epsr.metrics import TAUS, probabilistic_metrics
from percnet_epsr.models.registry import build_model, build_xcpd, make_config
from percnet_epsr.models.baselines.festrnn_official import Model as FESTRNN
from percnet_epsr.training.conformal import cqr_calibrate
from percnet_epsr.utils import resolve_device


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"[ok] {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-sched", action="store_true")
    args = ap.parse_args()
    device = resolve_device(args.device)
    print(f"device={device}")

    spec = default_spec()
    data = DataLoaderS(spec, 24, device)
    check(tuple(data.train[0].shape[1:]) == (168, 20),
          f"train window {tuple(data.train[0].shape)}")
    check(tuple(data.test[0].shape[1:]) == (168, 20),
          f"test window {tuple(data.test[0].shape)}")

    xb = data.train[0][:8].to(device)
    yb = data.train[1][:8, :, :3].to(device)
    scale3 = data.scale[:3].view(1, 1, -1)

    for name in ("percnet_v2", "percnet_v2_eesm"):
        cfg = make_config(name, 24, spec)
        model = build_model(name, cfg).to(device)
        check(not hasattr(model, "future_branch"),
              f"{name}: no future branch")
        out = model(xb)
        check(tuple(out["mu"].shape) == (8, 24, 3),
              f"{name}: mu {tuple(out['mu'].shape)}")
        check(tuple(out["quantiles"].shape) == (8, 24, 3, 9),
              f"{name}: quantiles {tuple(out['quantiles'].shape)}")
        loss = model.compute_loss(out, yb, scale3)
        check(torch.isfinite(loss),
              f"{name}: finite loss {float(loss.detach()):.3f}")
        loss.backward()
        n = sum(p.numel() for p in model.parameters())
        print(f"[info] {name}: params={n:,}")

    # Additional point baselines (same interface as DLinear/iTransformer).
    for name in ("segrnn", "crossformer", "film", "mamba"):
        cfg = make_config(name, 24, spec)
        model = build_model(name, cfg).to(device)
        out = model(xb)
        check(tuple(out.shape) == (8, 24, cfg.enc_in),
              f"{name}: out {tuple(out.shape)}")
        check(bool(torch.isfinite(out).all()), f"{name}: finite output")
        loss = torch.nn.functional.l1_loss(out[:, :, :3], yb)
        check(torch.isfinite(loss), f"{name}: finite loss")
        loss.backward()
        grad = sum(float(p.grad.abs().sum()) for p in model.parameters()
                   if p.grad is not None)
        check(grad > 0.0, f"{name}: gradient (|g|={grad:.3e})")
        print(f"[info] {name}: params="
              f"{sum(p.numel() for p in model.parameters()):,}")

    # PERC-Net v3: full channels + fusion + signed temperature response.
    # Each ablation switch must build, forward and receive gradient.
    for name in ("percnet_v3", "percnet_v3_noexo", "percnet_v3_nofusion",
                 "percnet_v3_nodeg", "percnet_v3_noeesm",
                 "percnet_v3_plain", "percnet_v3_wx", "percnet_v3_wx72",
                 "percnet_v3_wxq", "percnet_v3_wxh64",
                 "percnet_v3_wxw", "percnet_v3_wx_ndec",
                 "percnet_v3_wx_nocyc", "percnet_v3_wx_noanc",
                 "percnet_v3_point", "percnet_v3_wx_point"):
        cfg = make_config(name, 24, spec)
        model = build_model(name, cfg).to(device)
        out = model(xb)
        check(tuple(out["mu"].shape) == (8, 24, 3),
              f"{name}: mu {tuple(out['mu'].shape)}")
        if cfg.prob_head:
            check(tuple(out["quantiles"].shape) == (8, 24, 3, 9),
                  f"{name}: quantiles {tuple(out['quantiles'].shape)}")
        else:
            check("quantiles" not in out,
                  f"{name}: point-only (no quantile head)")
        loss = model.compute_loss(out, yb, scale3)
        check(torch.isfinite(loss), f"{name}: finite loss")
        loss.backward()
        grad = sum(float(p.grad.abs().sum()) for p in model.parameters()
                   if p.grad is not None)
        check(grad > 0.0, f"{name}: gradient at init (|g|={grad:.3e})")
        print(f"[info] {name}: params="
              f"{sum(p.numel() for p in model.parameters()):,}")

    v3 = build_model("percnet_v3", make_config("percnet_v3", 24, spec))
    temps = torch.linspace(0.0, 45.0, 64).unsqueeze(0)
    check(tuple(v3.degree.basis(temps).shape) == (1, 64, 6),
          "v3 degree basis (B,H,2K)")
    corr = v3.degree(temps)
    step = corr[0, 1:] - corr[0, :-1]
    check(bool((step[:, 1] >= -1e-6).all()),
          "v3 cooling response non-decreasing in T")
    check(bool((step[:, 2] <= 1e-6).all()),
          "v3 heating response non-increasing in T")
    corr.sum().backward()
    check(float(v3.degree.w_cool.grad.abs().sum()) > 0.0
          and float(v3.degree.w_heat.grad.abs().sum()) > 0.0,
          "v3 both degree bases receive gradient over 0-45 degC")
    d_in = int(v3.fusion[0].in_features)
    fused = v3.fusion(torch.randn(2, 24, d_in))
    check(float(fused.abs().max()) == 0.0,
          "v3 fusion starts as identity (zero residual)")
    check(float(v3.degree(temps).abs().max()) < 0.05,
          "v3 degree response starts near zero")
    wx = build_model("percnet_v3_wx",
                     make_config("percnet_v3_wx", 24, spec)).to(device)
    check(int(wx.fusion[0].in_features) == 3 * 3 + 5 * 9,
          f"wx fusion input {int(wx.fusion[0].in_features)} = 3*3 + 5*9")
    check(tuple(wx._weather_context(xb).shape) == (8, 45),
          "wx context (B, 5*n_exo)")

    # xCPD plugin: identity at init and 1-3K extra parameters.
    backbone = build_model("percnet_v2_eesm",
                           make_config("percnet_v2_eesm", 24, spec)).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    pcfg = make_config("percnet_v2_xcpd", 24, spec)
    pcfg.patch_len = 6
    wrapper = build_xcpd(pcfg, backbone).to(device)
    with torch.no_grad():
        base_mu = backbone(xb)["mu"]
        ref_mu = wrapper(xb)["mu"]
    diff = float((ref_mu - base_mu).abs().max())
    check(diff < 1e-6, f"xCPD identity at init (max diff {diff:.2e})")
    # Identity at init must not mean zero gradient: a double-zero init
    # (gate and output layer both zero) leaves the plugin untrainable.
    wrapper.train()
    wrapper.zero_grad()
    g_loss = wrapper.compute_loss(wrapper(xb), yb, scale3)
    g_loss.backward()
    g = sum(float(p.grad.abs().sum()) for p in wrapper.plugin.parameters()
            if p.grad is not None)
    check(g > 0.0, f"xCPD plugin gradient at init (|g|={g:.3e})")
    wrapper.zero_grad()
    n_plugin = sum(p.numel() for p in wrapper.plugin.parameters())
    check(1000 <= n_plugin <= 4000, f"xCPD extra params {n_plugin}")

    # Official FESTRNN forward.
    fcfg = SimpleNamespace(seq_len=168, pred_len=24, enc_in=12,
                           hidden_size=64, d_model=16, seg_len=24,
                           dropout=0.4)
    fm = FESTRNN(fcfg).to(device)
    fx = data.train[0][:4, :, :12].to(device)
    fo = fm(fx)
    check(tuple(fo.shape) == (4, 24, 3), f"FESTRNN out {tuple(fo.shape)}")
    print(f"[info] FESTRNN params={sum(p.numel() for p in fm.parameters()):,}")

    # CQR coverage on synthetic data.
    rng = np.random.default_rng(0)
    n, h, c = 400, 8, 3
    y_cal = rng.normal(100, 10, (n, h, c))
    q_cal = np.stack([y_cal + rng.normal(0, 3, y_cal.shape)
                      for _ in TAUS], axis=-1)
    q_cal.sort(axis=-1)
    y_test = rng.normal(100, 10, (50, h, c))
    q_test = np.stack([y_test + rng.normal(0, 3, y_test.shape)
                       for _ in TAUS], axis=-1)
    q_test.sort(axis=-1)
    q_cal2, _ = cqr_calibrate(q_cal, y_cal, q_test, TAUS)
    before = probabilistic_metrics(q_test, y_test)["PICP_90"]
    after = probabilistic_metrics(q_cal2, y_test)["PICP_90"]
    check(after >= before - 1e-9,
          f"CQR coverage {before:.3f} -> {after:.3f}")

    # Scenario-coherent sampling: PIT roundtrip, factor paths, proper scores.
    from percnet_epsr import scenarios as SCN
    nq, Hq, Cq = 200, 24, 3
    q_syn = np.sort(rng.normal(100, 10, (nq, Hq, Cq, len(TAUS)))
                    + rng.normal(0, 5, (nq, Hq, Cq, len(TAUS))), axis=-1)
    u_syn = rng.uniform(0.05, 0.95, (nq, Hq, Cq))
    y_syn = SCN.inverse_map(q_syn, TAUS, u_syn)
    err = float(np.abs(SCN.pit(q_syn, TAUS, y_syn) - u_syn).max())
    check(err < 1e-5, f"scenario PIT roundtrip (max err {err:.2e})")
    dep = SCN.DependenceModel.fit(SCN.pit(q_syn, TAUS, y_syn), rank=6)
    paths = SCN.sample_paths(dep, q_syn[0], TAUS, 16, rng)
    check(tuple(paths.shape) == (16, Hq, Cq) and bool(np.isfinite(paths).all()),
          f"scenario paths {tuple(paths.shape)}")
    sc = SCN.marginal_scale(q_syn)
    es0 = SCN.energy_score(np.repeat(y_syn[0][None], 16, 0), y_syn[0], sc)
    check(abs(es0) < 1e-9, f"oracle scenario set has energy score 0 ({es0:.1e})")
    check(SCN.energy_score(paths, y_syn[0], sc) > 0.0,
          "sampled scenario set has positive energy score")

    if not args.skip_sched:
        from percnet_epsr.experiments.scheduling.two_stage import (
            PARAMS, solve_policy)
        p = dict(PARAMS)
        H = 4
        loads = {
            "e50": np.full(H, 5000.0), "h50": np.full(H, 4000.0),
            "c50": np.full(H, 6000.0),
            "e50_total": np.full(H, 6500.0),
            "e90_total": np.full(H, 7200.0),
            "e95_total": np.full(H, 7500.0),
            "e_actual": np.full(H, 6600.0),
            "h_actual": np.full(H, 4200.0),
            "c_actual": np.full(H, 6200.0),
        }
        scen = [{"e": loads["e_actual"], "h": loads["h_actual"],
                 "c": loads["c_actual"]}]
        out = solve_policy("median", H, p, loads, scen, np.array([1.0]), 5.0)
        check(np.isfinite(out["actual_cost"]),
              f"dispatch median cost {out['actual_cost']:.1f}")

    print("\nSMOKE 3090 PASSED")


if __name__ == "__main__":
    sys.exit(main())
