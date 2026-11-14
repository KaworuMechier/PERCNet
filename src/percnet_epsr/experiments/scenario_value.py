"""Price the joint structure of the forecast: scenario scores and dispatch value.

For every probabilistic run directory and every scenario construction this
script

1. fits the dependence model on the **validation** PIT residuals,
2. samples ``m`` paths for a spread of test windows,
3. scores them with the energy score (multivariate CRPS) and the variogram
   score (lead-time dependence),
4. runs the two-stage dispatch on those same scenarios, and
5. writes per-window and per-(run, mode) tables, including VSS against the
   deterministic median policy.

The point of the table is that the marginal calibration is held fixed (the
same quantile grid feeds all three modes); only the joint structure changes.

    python -m percnet_epsr.experiments.scenario_value \
        --run-dirs results/probabilistic/percnet_v3_wx_ASU_honest_h24_s2024 \
        --out results/scenario_value --scenarios 16 --max-windows 24
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from percnet_epsr import scenarios as S
from percnet_epsr.experiments.scheduling.two_stage import (PARAMS, load_run,
                                                           solve_policy,
                                                           window_loads)


def _pair(run: dict):
    """(test quantiles, calib quantiles) and the calib observations.

    A CQR run directory holds only the calibrated test grid; its validation
    grid lives in the parent (uncalibrated) run directory.  The dependence
    model is fitted on the validation residuals either way.
    """
    if run["calib"] is None:
        base = run["dir"].rstrip("/")
        if base.endswith("_cqr"):
            path = os.path.join(base[:-4], "calib_quantiles.npz")
            if os.path.isfile(path):
                z = np.load(path)
                run["calib"] = {"q": z["quantiles"], "y": z["y_true"]}
    if run["calib"] is None:
        raise RuntimeError(f"{run['dir']}: no calib_quantiles.npz, cannot fit "
                           "the dependence model on validation residuals")
    return run["q"], run["y"], run["taus"], run["calib"]["q"], run["calib"]["y"]


def _demands(paths: np.ndarray, p: dict):
    """(m, H, C) physical paths -> list of dispatch demand dicts."""
    out = []
    for k in range(paths.shape[0]):
        out.append({"e": paths[k, :, 0],
                    "h": paths[k, :, 2] * p["mmbtu_to_kwh"],
                    "c": paths[k, :, 1] * p["ton_to_kwh"]})
    return out


def cvar(x: np.ndarray, beta: float = 0.95) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    k = max(int(np.ceil(beta * len(x))), 1)
    return float(x[k - 1:].mean())


def _solve(policy, H, p, loads, scen, probs, time_limit):
    """Solve with one longer retry; None when the MILP fails both times.

    A 16-scenario MILP occasionally hits the time limit without an incumbent
    (the CQR-widened intervals make it harder), and one such failure must not
    abort a 3-mode x N-window table.
    """
    err = None
    for factor in (1.0, 3.0):
        try:
            return solve_policy(policy, H, p, loads, scen, probs,
                                time_limit * factor)
        except RuntimeError as exc:
            err = exc
    print(f"  [warn] {policy}: {err} (window skipped)", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--out", default="results/scenario_value")
    ap.add_argument("--scenarios", type=int, default=16)
    ap.add_argument("--max-windows", type=int, default=24,
                    help="evenly spaced test windows (0 = all)")
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--modes", default="independent,copula,path")
    ap.add_argument("--policies", default="median,stochastic,oracle")
    ap.add_argument("--time-limit", type=float, default=PARAMS["time_limit"])
    ap.add_argument("--params-json", default=None,
                    help="override dispatch parameters (sensitivity study)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one dispatch parameter inline, repeatable "
                         "(e.g. --set grid_cap=12000 --set c_shed_e=40)")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    p = dict(PARAMS)
    if args.params_json:
        with open(args.params_json) as f:
            p.update(json.load(f))
    for item in args.set:
        key, _, value = item.partition("=")
        if not _:
            ap.error(f"--set expects KEY=VALUE, got {item!r}")
        p[key.strip()] = float(value)
    changed = {k: p[k] for k in sorted(p) if PARAMS.get(k) != p[k]}
    print(f"[params] overrides: {changed or 'none'}", flush=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    policies = [m.strip() for m in args.policies.split(",") if m.strip()]
    os.makedirs(args.out, exist_ok=True)
    rows, summary = [], []

    for run_dir in args.run_dirs:
        run = load_run(run_dir)
        q, y, taus, q_cal, y_cal = _pair(run)
        H, C = q.shape[1:3]
        n = len(q)
        idx = (np.arange(n) if args.max_windows <= 0 else np.unique(
            np.linspace(0, n - 1, args.max_windows).round().astype(int)))
        u_cal = S.pit(q_cal, taus, y_cal)
        dep = S.DependenceModel.fit(u_cal, rank=args.rank)
        print(f"[scen] {run['name']}: N={n} H={H} windows={len(idx)} "
              f"modes={modes} scenarios={args.scenarios} | rank {dep.rank} "
              f"keeps {dep.explained:.1%} of the PIT variance", flush=True)
        scale = S.marginal_scale(q_cal)
        corr = np.corrcoef((u_cal - 0.5).reshape(-1, C), rowvar=False)
        corr = corr if np.all(np.isfinite(corr)) else np.eye(C)

        for mode in modes:
            rng = np.random.default_rng(args.seed)
            es, vs, costs = [], [], {k: [] for k in policies}
            extra = {k: {"shed": [], "rshort": [], "commit": []}
                     for k in policies}
            failed = {k: 0 for k in policies}
            for i in idx:
                qw = q[i]
                if mode == "path":
                    paths = S.sample_paths(dep, qw, taus, args.scenarios, rng)
                elif mode == "independent":
                    paths = S.sample_independent(qw, taus, args.scenarios, rng)
                elif mode == "copula":
                    paths = S.sample_copula(qw, taus, args.scenarios, rng,
                                            corr=corr)
                else:
                    raise ValueError(f"unknown mode {mode!r}")

                es.append(S.energy_score(paths, y[i], scale))
                vs.append(S.variogram_score(paths, y[i], scale))

                loads = window_loads(run, i, p)
                scen = _demands(paths, p)
                probs = np.full(len(scen), 1.0 / len(scen))
                for policy in policies:
                    out = _solve(policy, H, p, loads, scen, probs,
                                 args.time_limit)
                    if out is None:
                        failed[policy] += 1
                        continue
                    costs[policy].append(out["actual_cost"])
                    extra[policy]["shed"].append(
                        out.get("shed_e", 0.0) + out.get("shed_h", 0.0)
                        + out.get("shed_c", 0.0))
                    extra[policy]["rshort"].append(
                        out.get("reserve_shortfall", 0.0))
                    extra[policy]["commit"].append(
                        out.get("commit_hours", float("nan")))
                rows.append({"run": run["name"], "mode": mode,
                             "window": int(i), "horizon": H,
                             "energy_score": float(es[-1]),
                             "variogram_score": float(vs[-1]),
                             "cost_median": float(costs["median"][-1])
                             if "median" in costs else float("nan")})

            row = {"run": run["name"], "horizon": H, "mode": mode,
                   "windows": len(idx), "scenarios": args.scenarios,
                   "energy_score": float(np.mean(es)),
                   "variogram_score": float(np.mean(vs))}
            for policy in policies:
                c = np.asarray(costs[policy], dtype=float)
                ok = len(c) > 0
                row[f"cost_{policy}"] = float(c.mean()) if ok else np.nan
                row[f"cvar95_{policy}"] = cvar(c) if ok else np.nan
                row[f"shed_{policy}"] = (float(np.mean(extra[policy]["shed"]))
                                         if ok else np.nan)
                row[f"rshort_{policy}"] = (
                    float(np.mean(extra[policy]["rshort"])) if ok else np.nan)
                row[f"commit_{policy}"] = (
                    float(np.mean(extra[policy]["commit"])) if ok else np.nan)
                row[f"failed_{policy}"] = failed[policy]
            if (len(costs.get("median", [])) and len(costs.get("stochastic", []))):
                row["vss"] = (float(np.mean(costs["median"]))
                              - float(np.mean(costs["stochastic"])))
            summary.append(row)
            print(f"  {mode:11s} ES {row['energy_score']:.4f} "
                  f"VS {row['variogram_score']:9.1f} "
                  f"cost {row.get('cost_stochastic', float('nan')):9.1f} "
                  f"VSS {row.get('vss', float('nan')):7.1f}", flush=True)

    with open(os.path.join(args.out, "scenario_value.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(args.out, "scenario_per_window.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.join(args.out, 'scenario_value.csv')}")


if __name__ == "__main__":
    main()
