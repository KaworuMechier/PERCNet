"""Scenario-based two-stage stochastic dispatch for the EPSR scheduling study.

Campus multi-energy hub (electricity / heat / cooling) with:

  * CHP and gas turbine with binary commitment, min/max, ramp and startup;
  * grid import, battery, boiler, electric chiller, load shedding;
  * spinning reserve from committed-unit headroom and battery discharge.

Stage 1 commits the units on the predictive distribution; stage 2 re-dispatches
on the realised loads with the commitment fixed.  The stochastic policy uses
``m`` scenarios sampled from each model's joint predictive distribution; the
deterministic policies use a reserve margin derived from q90/q95 or a fixed
percentage.

The parameter set is illustrative (campus scale) and is not site-certified.
Use the sensitivity table before quoting absolute dollar savings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from scipy.special import ndtr

from percnet_epsr import scenarios as S
from percnet_epsr.experiments.scheduling.stats import paired_bootstrap, wilcoxon


PARAMS: Dict[str, float] = {
    "chp_cap": 18000.0, "chp_min": 3000.0, "chp_ramp": 8000.0,
    "chp_heat_ratio": 1.2, "c_chp": 0.055, "c_start_chp": 120.0,
    "gt_cap": 10000.0, "gt_min": 0.0, "gt_ramp": 6000.0,
    "c_gt": 0.075, "c_start_gt": 90.0,
    "grid_cap": 36000.0, "c_grid": 0.15,
    "boiler_cap": 700000.0, "c_boiler": 0.040,
    "chiller_cap": 240000.0, "c_chiller": 0.010,
    "cop_chiller": 4.0, "ton_to_kwh": 3.517,
    "mmbtu_to_kwh": 293.071,
    "batt_energy": 12000.0, "batt_power": 2000.0, "batt_eta": 0.95,
    "batt_soc_init": 0.5, "batt_soc_end": 0.3, "c_batt": 0.005,
    "c_shed_e": 10.0, "c_shed_h": 5.0, "c_shed_c": 5.0,
    "c_rshort": 20.0, "time_limit": 20.0,
}


def _tau_index(taus, tau):
    return int(np.argmin(np.abs(np.asarray(taus) - tau)))


def _q_at(q, taus, tau):
    q = np.asarray(q, dtype=float)
    taus = np.asarray(taus, dtype=float)
    flat = q.reshape(-1, q.shape[-1])
    out = np.array([np.interp(tau, taus, row) for row in flat])
    return out.reshape(q.shape[:-1])


def load_run(run_dir: str) -> dict:
    path = os.path.join(run_dir, "quantiles.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    z = np.load(path)
    q, y, taus = z["quantiles"], z["y_true"], z["taus"]
    calib = None
    cpath = os.path.join(run_dir, "calib_quantiles.npz")
    if os.path.isfile(cpath):
        cz = np.load(cpath)
        calib = {"q": cz["quantiles"], "y": cz["y_true"]}
    name = os.path.basename(os.path.normpath(run_dir))
    m = re.search(r"_h(\d+)_s(\d+)", name)
    horizon = int(m.group(1)) if m else int(q.shape[1])
    seed = int(m.group(2)) if m else -1
    return {"dir": run_dir, "name": name, "q": q, "y": y, "taus": taus,
            "calib": calib, "horizon": horizon, "seed": seed}


def window_loads(run: dict, i: int, p: dict) -> dict:
    q = np.asarray(run["q"][i], dtype=float)
    y = np.asarray(run["y"][i], dtype=float)
    taus = np.asarray(run["taus"], dtype=float)
    kw50 = _q_at(q[:, 0, :], taus, 0.50)
    chw50 = _q_at(q[:, 1, :], taus, 0.50)
    ht50 = _q_at(q[:, 2, :], taus, 0.50)
    kw90 = _q_at(q[:, 0, :], taus, 0.90)
    chw90 = _q_at(q[:, 1, :], taus, 0.90)
    kw95 = _q_at(q[:, 0, :], taus, 0.95)
    chw95 = _q_at(q[:, 1, :], taus, 0.95)
    cool50, cool_a = chw50 * p["ton_to_kwh"], y[:, 1] * p["ton_to_kwh"]
    heat50 = ht50 * p["mmbtu_to_kwh"]
    heat_a = y[:, 2] * p["mmbtu_to_kwh"]
    e50_total = kw50 + cool50 / p["cop_chiller"]
    e90_total = kw90 + (chw90 * p["ton_to_kwh"]) / p["cop_chiller"]
    e95_total = kw95 + (chw95 * p["ton_to_kwh"]) / p["cop_chiller"]
    return {
        "e50": kw50, "h50": heat50, "c50": cool50,
        "e90_total": e90_total, "e95_total": e95_total,
        "e_actual": y[:, 0], "h_actual": heat_a, "c_actual": cool_a,
        "e50_total": e50_total,
    }


def scenario_demands(run: dict, i: int, p: dict, m: int,
                     seed: int = 2024) -> List[dict]:
    """Sample m joint scenarios from the predictive quantile function.

    A Gaussian copula is fitted to the validation residual correlation; each
    scenario draws one tau per channel and interpolates the channel quantile
    function.  With no validation artifact the identity correlation is used.
    """
    rng = np.random.default_rng(seed)
    q = np.asarray(run["q"][i], dtype=float)
    taus = np.asarray(run["taus"], dtype=float)
    corr = np.eye(3)
    calib = run.get("calib")
    if calib is not None:
        cq = np.asarray(calib["q"], dtype=float)
        cy = np.asarray(calib["y"], dtype=float)
        i50 = _tau_index(taus, 0.5)
        resid = cy - cq[..., i50]
        r = np.corrcoef(resid.reshape(-1, 3), rowvar=False)
        if np.all(np.isfinite(r)):
            corr = r
    corr = corr + np.eye(3) * 1e-6
    try:
        chol = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        chol = np.eye(3)
    out = []
    for _ in range(m):
        z = rng.standard_normal(3) @ chol.T
        u = np.clip(ndtr(z), 0.01, 0.99)
        kw = _q_at(q[:, 0, :], taus, u[0])
        chw = _q_at(q[:, 1, :], taus, u[1])
        ht = _q_at(q[:, 2, :], taus, u[2])
        out.append({"e": kw, "h": ht * p["mmbtu_to_kwh"],
                    "c": chw * p["ton_to_kwh"]})
    return out


def scenarios_for_window(run: dict, i: int, p: dict, m: int, seed: int,
                         mode: str, dep=None, corr=None) -> List[dict]:
    """Scenario paths for one window under the requested joint structure."""
    if mode == "copula":
        return scenario_demands(run, i, p, m, seed=seed)
    rng = np.random.default_rng(seed)
    q = np.asarray(run["q"][i], dtype=float)
    taus = np.asarray(run["taus"], dtype=float)
    if mode == "path":
        paths = S.sample_paths(dep, q, taus, m, rng)
    elif mode == "independent":
        paths = S.sample_independent(q, taus, m, rng)
    else:
        raise ValueError(f"unknown scenario mode {mode!r}")
    return [{"e": paths[k, :, 0],
             "h": paths[k, :, 2] * p["mmbtu_to_kwh"],
             "c": paths[k, :, 1] * p["ton_to_kwh"]} for k in range(m)]


class DispatchModel:
    def __init__(self, p: dict):
        self.p = p
        self.names: List[str] = []
        self.index: Dict[str, int] = {}
        self.lb: List[float] = []
        self.ub: List[float] = []
        self.integrality: List[int] = []
        self.cost: List[float] = []
        self.rows: List[Dict[str, float]] = []
        self.row_lb: List[float] = []
        self.row_ub: List[float] = []

    def add(self, name, lb, ub, integ=0, cost=0.0):
        self.index[name] = len(self.names)
        self.names.append(name)
        self.lb.append(float(lb))
        self.ub.append(float(ub))
        self.integrality.append(int(integ))
        self.cost.append(float(cost))

    def row(self, coeffs, lb=-np.inf, ub=np.inf):
        self.rows.append({k: float(v) for k, v in coeffs.items() if v != 0})
        self.row_lb.append(float(lb))
        self.row_ub.append(float(ub))

    def solve(self, time_limit: float):
        n = len(self.names)
        rr, cc, dd = [], [], []
        for i, co in enumerate(self.rows):
            for name, val in co.items():
                rr.append(i)
                cc.append(self.index[name])
                dd.append(val)
        A = csr_matrix((dd, (rr, cc)), shape=(len(self.rows), n))
        res = milp(
            c=np.asarray(self.cost), integrality=np.asarray(self.integrality),
            bounds=Bounds(np.asarray(self.lb), np.asarray(self.ub)),
            constraints=LinearConstraint(A, np.asarray(self.row_lb),
                                         np.asarray(self.row_ub)),
            options={"time_limit": float(time_limit), "mip_rel_gap": 1e-4})
        # A time-limited MILP can return a feasible incumbent with
        # success=False; keep it rather than discarding the window.
        return res.x if getattr(res, "x", None) is not None else None

    def get(self, x, name):
        return float(x[self.index[name]])


def _add_commitment(m, H, p, prefix=""):
    for t in range(H):
        m.add(f"{prefix}u1_{t}", 0, 1, 1)
        m.add(f"{prefix}u2_{t}", 0, 1, 1)
        m.add(f"{prefix}s1_{t}", 0, 1, 0, p["c_start_chp"])
        m.add(f"{prefix}s2_{t}", 0, 1, 0, p["c_start_gt"])


def _add_recourse(m, H, p, demand, prefix="", actual=False,
                  reserve=None):
    e, h, c = demand["e"], demand["h"], demand["c"]
    for t in range(H):
        m.add(f"{prefix}p1_{t}", 0, p["chp_cap"], 0, p["c_chp"])
        m.add(f"{prefix}p2_{t}", 0, p["gt_cap"], 0, p["c_gt"])
        m.add(f"{prefix}grid_{t}", 0, p["grid_cap"], 0, p["c_grid"])
        m.add(f"{prefix}boiler_{t}", 0, p["boiler_cap"], 0, p["c_boiler"])
        m.add(f"{prefix}chiller_{t}", 0, p["chiller_cap"], 0, p["c_chiller"])
        m.add(f"{prefix}ch_{t}", 0, p["batt_power"], 0, p["c_batt"])
        m.add(f"{prefix}dis_{t}", 0, p["batt_power"], 0, p["c_batt"])
        m.add(f"{prefix}soc_{t}", 0, p["batt_energy"], 0)
        m.add(f"{prefix}shed_e_{t}", 0, np.inf, 0, p["c_shed_e"])
        m.add(f"{prefix}shed_h_{t}", 0, np.inf, 0, p["c_shed_h"])
        m.add(f"{prefix}shed_c_{t}", 0, np.inf, 0, p["c_shed_c"])
        if not actual:
            m.add(f"{prefix}rshort_{t}", 0, np.inf, 0, p["c_rshort"])

    for t in range(H):
        p1, p2 = f"{prefix}p1_{t}", f"{prefix}p2_{t}"
        u1, u2 = f"u1_{t}", f"u2_{t}"
        m.row({p1: 1, u1: -p["chp_min"]}, 0, np.inf)
        m.row({p1: 1, u1: -p["chp_cap"]}, -np.inf, 0)
        m.row({p2: 1, u2: -p["gt_min"]}, 0, np.inf)
        m.row({p2: 1, u2: -p["gt_cap"]}, -np.inf, 0)
        if t > 0:
            m.row({p1: 1, f"{prefix}p1_{t-1}": -1}, -np.inf, p["chp_ramp"])
            m.row({f"{prefix}p1_{t-1}": 1, p1: -1}, -np.inf, p["chp_ramp"])
            m.row({p2: 1, f"{prefix}p2_{t-1}": -1}, -np.inf, p["gt_ramp"])
            m.row({f"{prefix}p2_{t-1}": 1, p2: -1}, -np.inf, p["gt_ramp"])
        m.row({p1: 1, p2: 1, f"{prefix}grid_{t}": 1,
               f"{prefix}dis_{t}": 1, f"{prefix}ch_{t}": -1,
               f"{prefix}shed_e_{t}": 1,
               f"{prefix}chiller_{t}": -1.0 / p["cop_chiller"]},
              e[t], e[t])
        m.row({p1: p["chp_heat_ratio"], f"{prefix}boiler_{t}": 1,
               f"{prefix}shed_h_{t}": 1}, h[t], h[t])
        m.row({f"{prefix}chiller_{t}": 1, f"{prefix}shed_c_{t}": 1},
              c[t], c[t])
        if t == 0:
            m.row({f"{prefix}soc_{t}": 1, f"{prefix}ch_{t}": -p["batt_eta"],
                   f"{prefix}dis_{t}": 1.0 / p["batt_eta"]},
                  p["batt_energy"] * p["batt_soc_init"],
                  p["batt_energy"] * p["batt_soc_init"])
        else:
            m.row({f"{prefix}soc_{t}": 1, f"{prefix}soc_{t-1}": -1,
                   f"{prefix}ch_{t}": -p["batt_eta"],
                   f"{prefix}dis_{t}": 1.0 / p["batt_eta"]}, 0, 0)
        if reserve is not None and not actual:
            m.row({u1: p["chp_cap"], u2: p["gt_cap"], p1: -1, p2: -1,
                   f"{prefix}dis_{t}": -1, f"{prefix}rshort_{t}": 1},
                  reserve[t] - p["batt_power"], np.inf)
    m.row({f"{prefix}soc_{H-1}": 1},
          p["batt_energy"] * p["batt_soc_end"], np.inf)


def _add_startup(m, H, p):
    for t in range(H):
        s1, s2 = f"s1_{t}", f"s2_{t}"
        u1, u2 = f"u1_{t}", f"u2_{t}"
        if t == 0:
            m.row({s1: 1, u1: -1}, 0, np.inf)
            m.row({s2: 1, u2: -1}, 0, np.inf)
        else:
            m.row({s1: 1, u1: -1, f"u1_{t-1}": 1}, 0, np.inf)
            m.row({s2: 1, u2: -1, f"u2_{t-1}": 1}, 0, np.inf)


def _build_stochastic(H, p, scenarios, probs):
    m = DispatchModel(p)
    _add_commitment(m, H, p)
    _add_startup(m, H, p)
    for k, dem in enumerate(scenarios):
        pref = f"sc{k}_"
        _add_recourse(m, H, p, dem, prefix=pref)
        # weight scenario recourse costs
        for t in range(H):
            for base in ("p1", "p2", "grid", "boiler", "chiller", "ch", "dis",
                         "shed_e", "shed_h", "shed_c"):
                name = f"{pref}{base}_{t}"
                m.cost[m.index[name]] *= probs[k]
    return m


def _build_deterministic(H, p, dem, reserve, actual=False, fix_u=None):
    m = DispatchModel(p)
    if fix_u is None:
        _add_commitment(m, H, p)
        _add_startup(m, H, p)
    else:
        for t in range(H):
            m.add(f"u1_{t}", fix_u[0][t], fix_u[0][t], 0)
            m.add(f"u2_{t}", fix_u[1][t], fix_u[1][t], 0)
    _add_recourse(m, H, p, dem, actual=actual, reserve=None if actual else reserve)
    return m


def _extract_commitment(m, x, H):
    return (np.array([m.get(x, f"u1_{t}") for t in range(H)]),
            np.array([m.get(x, f"u2_{t}") for t in range(H)]))


def _actual_cost(H, p, dem, u, time_limit):
    ma = _build_deterministic(H, p, dem, np.zeros(H), actual=True, fix_u=u)
    xa = ma.solve(time_limit)
    if xa is None:
        raise RuntimeError("actual re-dispatch failed")
    startup = (u[0][0] * p["c_start_chp"] + u[1][0] * p["c_start_gt"])
    startup += float(np.sum((np.diff(u[0]) > 0.5) * p["c_start_chp"]
                            + (np.diff(u[1]) > 0.5) * p["c_start_gt"]))
    return {
        "actual_cost": float(np.dot(ma.cost, xa)) + startup,
        "startup_cost": float(startup),
        "shed_e": sum(ma.get(xa, f"shed_e_{t}") for t in range(H)),
        "shed_h": sum(ma.get(xa, f"shed_h_{t}") for t in range(H)),
        "shed_c": sum(ma.get(xa, f"shed_c_{t}") for t in range(H)),
    }


def solve_policy(policy, H, p, loads, scenarios, probs, time_limit,
                 rl_rule=None):
    fc = {"e": loads["e50"], "h": loads["h50"], "c": loads["c50"]}
    ac = {"e": loads["e_actual"], "h": loads["h_actual"],
          "c": loads["c_actual"]}
    if policy == "median":
        reserve = np.zeros(H)
    elif policy == "fixed5":
        reserve = 0.05 * loads["e50_total"]
    elif policy == "fixed10":
        reserve = 0.10 * loads["e50_total"]
    elif policy == "q90":
        reserve = np.maximum(0, loads["e90_total"] - loads["e50_total"])
    elif policy == "q95":
        reserve = np.maximum(0, loads["e95_total"] - loads["e50_total"])
    elif policy in ("stochastic", "cvar"):
        m = _build_stochastic(H, p, scenarios, probs)
        x = m.solve(time_limit)
        if x is None:
            raise RuntimeError("stochastic MILP failed")
        u = _extract_commitment(m, x, H)
        out = _actual_cost(H, p, ac, u, time_limit)
        out.update({"policy": policy, "reserve_req": 0.0,
                    "reserve_shortfall": 0.0,
                    "commit_hours": float(u[0].sum() + u[1].sum())})
        return out
    elif policy == "oracle":
        m = _build_deterministic(H, p, ac, np.zeros(H), actual=True)
        x = m.solve(time_limit)
        if x is None:
            raise RuntimeError("oracle MILP failed")
        out = {"policy": "oracle", "actual_cost": float(np.dot(m.cost, x)),
               "startup_cost": 0.0, "reserve_req": 0.0,
               "reserve_shortfall": 0.0}
        for t in range(H):
            out["shed_e"] = out.get("shed_e", 0.0) + m.get(x, f"shed_e_{t}")
            out["shed_h"] = out.get("shed_h", 0.0) + m.get(x, f"shed_h_{t}")
            out["shed_c"] = out.get("shed_c", 0.0) + m.get(x, f"shed_c_{t}")
        return out
    elif policy == "rl":
        if rl_rule is None:
            spread = (loads["e90_total"] - loads["e50_total"]) / (
                loads["e50_total"] + 1e-6)
            rho = np.clip(0.05 + 0.5 * spread, 0.0, 0.30)
        else:
            rho = rl_rule(loads)
        reserve = rho * loads["e50_total"]
    else:
        raise ValueError(f"unknown policy {policy!r}")

    m = _build_deterministic(H, p, fc, reserve)
    x = m.solve(time_limit)
    if x is None:
        raise RuntimeError(f"forecast MILP failed for {policy}")
    u = _extract_commitment(m, x, H)
    out = _actual_cost(H, p, ac, u, time_limit)
    out.update({
        "policy": policy, "reserve_req": float(np.sum(reserve)),
        "reserve_shortfall": sum(m.get(x, f"rshort_{t}") for t in range(H)),
        "commit_hours": float(u[0].sum() + u[1].sum()),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--out", default="results/scheduling")
    ap.add_argument("--scenarios", type=int, default=8)
    ap.add_argument("--max-windows", type=int, default=0,
                    help="0 = all test windows")
    ap.add_argument("--policies", nargs="+",
                    default=["median", "fixed5", "fixed10", "q90", "q95",
                             "stochastic", "oracle", "rl"])
    ap.add_argument("--time-limit", type=float, default=PARAMS["time_limit"])
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--rl-policy", default=None,
                    help="JSON written by rl_reserve.py")
    ap.add_argument("--rl-grid",
                    default="0.00,0.05,0.10,0.15,0.20,0.25,0.30")
    ap.add_argument("--scenario-mode",
                    choices=["copula", "path", "independent"],
                    default="copula",
                    help="joint structure of the sampled scenarios; "
                         "'path' is the PIT + factor construction")
    ap.add_argument("--rank", type=int, default=6,
                    help="rank of the factor model used by --scenario-mode path")
    args = ap.parse_args()

    p = dict(PARAMS)
    if args.params_json:
        with open(args.params_json) as f:
            p.update(json.load(f))
    os.makedirs(args.out, exist_ok=True)
    rl_rule = None
    if args.rl_policy:
        from percnet_epsr.experiments.scheduling.rl_reserve import make_rule
        grid = [float(x) for x in args.rl_grid.split(",")]
        rl_rule = make_rule(args.rl_policy, grid)
    rows = []
    for run_dir in args.run_dirs:
        run = load_run(run_dir)
        H = int(run["q"].shape[1])
        n = int(run["q"].shape[0])
        idx = np.arange(n) if args.max_windows <= 0 else np.unique(
            np.linspace(0, n - 1, args.max_windows).round().astype(int))
        print(f"[sched] {run['name']}: N={n} H={H} windows={len(idx)} "
              f"policies={args.policies} scenarios={args.scenario_mode}",
              flush=True)
        dep = corr = None
        if args.scenario_mode != "copula":
            if run.get("calib") is None:
                raise RuntimeError(
                    f"{run['dir']}: no validation quantiles; the fitted "
                    "scenario modes need calib_quantiles.npz")
            taus = np.asarray(run["taus"], dtype=float)
            u_cal = S.pit(run["calib"]["q"], taus, run["calib"]["y"])
            dep = S.DependenceModel.fit(u_cal, rank=args.rank)
            corr = np.corrcoef((u_cal - 0.5).reshape(-1, u_cal.shape[-1]),
                               rowvar=False)
        for i in idx:
            loads = window_loads(run, i, p)
            scen = scenarios_for_window(run, i, p, args.scenarios,
                                        seed=2024 + i,
                                        mode=args.scenario_mode,
                                        dep=dep, corr=corr)
            probs = np.full(len(scen), 1.0 / len(scen))
            for policy in args.policies:
                out = solve_policy(policy, H, p, loads, scen, probs,
                                   args.time_limit,
                                   rl_rule=rl_rule if policy == "rl" else None)
                out.update({"run": run["name"], "window": int(i),
                            "horizon": H, "seed": run["seed"]})
                rows.append(out)
    path = os.path.join(args.out, "scheduling_per_window.csv")
    keys = ["run", "window", "horizon", "seed", "policy", "actual_cost",
            "startup_cost", "shed_e", "shed_h", "shed_c", "reserve_req",
            "reserve_shortfall"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Summary and paired comparisons against the median policy.
    import pandas as pd
    df = pd.DataFrame(rows)
    summary = df.groupby("policy")[["actual_cost", "shed_e", "shed_h",
                                    "shed_c", "reserve_req",
                                    "reserve_shortfall"]].agg(["mean", "std"])
    summary.to_csv(os.path.join(args.out, "scheduling_summary.csv"))
    stats_rows = []
    if "median" in set(df.policy):
        med = df[df.policy == "median"].set_index(["run", "window"])
        for policy in sorted(set(df.policy) - {"median"}):
            sub = df[df.policy == policy].set_index(["run", "window"])
            common = med.index.intersection(sub.index)
            if len(common) == 0:
                continue
            a = sub.loc[common, "actual_cost"].to_numpy()
            b = med.loc[common, "actual_cost"].to_numpy()
            bs = paired_bootstrap(a, b)
            wx = wilcoxon(a, b)
            stats_rows.append({"policy": policy, **bs, **wx})
    if stats_rows:
        with open(os.path.join(args.out, "scheduling_stats.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(stats_rows[0]))
            w.writeheader()
            w.writerows(stats_rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
