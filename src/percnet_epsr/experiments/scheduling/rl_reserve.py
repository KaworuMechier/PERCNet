"""Contextual-bandit reserve-margin policy.

The policy maps forecast context to a reserve margin rho in a fixed grid.
It is trained on a prefix of the test windows and evaluated on the suffix;
the environment reward is the negative actual dispatch cost from the
scenario model in ``two_stage.py``.  This is a safety layer for the
scheduling study, not a point-accuracy claim.

LinUCB is used as a stable fallback for PPO; replace ``LinUCB`` with a PPO
agent if desired.  The state is deliberately forecast-only (no realised
load) so the policy is deployable.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from percnet_epsr.experiments.scheduling.two_stage import (PARAMS,
                                                           load_run,
                                                           scenario_demands,
                                                           solve_policy,
                                                           window_loads)


def features(loads: dict) -> np.ndarray:
    q = np.asarray(loads["e50_total"], dtype=float)
    spread = np.asarray(loads["e90_total"] - loads["e50_total"], dtype=float)
    denom = float(np.mean(q) + 1e-6)
    return np.array([
        1.0,
        float(np.mean(q)) / 1e4,
        float(np.mean(spread)) / 1e4,
        float(np.std(q)) / denom,
        float(np.max(q) / denom),
        float(np.mean(spread) / denom),
    ])


class LinUCB:
    def __init__(self, n_actions: int, dim: int, alpha: float = 0.5):
        self.alpha = float(alpha)
        self.A = [np.eye(dim) for _ in range(n_actions)]
        self.b = [np.zeros(dim) for _ in range(n_actions)]

    def scores(self, x: np.ndarray) -> np.ndarray:
        out = []
        for A, b in zip(self.A, self.b):
            theta = np.linalg.solve(A, b)
            bonus = self.alpha * np.sqrt(max(x @ np.linalg.solve(A, x), 0.0))
            out.append(float(theta @ x + bonus))
        return np.array(out)

    def select(self, x: np.ndarray) -> int:
        return int(np.argmax(self.scores(x)))

    def update(self, action: int, x: np.ndarray, reward: float) -> None:
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

    def to_dict(self) -> dict:
        return {"A": [a.tolist() for a in self.A],
                "b": [b.tolist() for b in self.b]}

    @classmethod
    def from_dict(cls, payload: dict) -> "LinUCB":
        obj = cls(len(payload["A"]), len(payload["A"][0]))
        obj.A = [np.asarray(a, dtype=float) for a in payload["A"]]
        obj.b = [np.asarray(b, dtype=float) for b in payload["b"]]
        return obj


def make_rule(policy_path: str, rho_grid):
    with open(policy_path) as f:
        payload = json.load(f)
    agent = LinUCB.from_dict(payload)
    grid = np.asarray(rho_grid, dtype=float)

    def rule(loads):
        a = agent.select(features(loads))
        return float(grid[a])

    return rule


def _reward(policy, H, p, loads, scen, probs, time_limit, rho_grid):
    def rl_rule(_loads):
        return float(policy)
    out = solve_policy("rl", H, p, loads, scen, probs, time_limit,
                       rl_rule=rl_rule)
    return -float(out["actual_cost"]), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--out", default="results/scheduling/rl")
    ap.add_argument("--rho-grid", default="0.00,0.05,0.10,0.15,0.20,0.25,0.30")
    ap.add_argument("--max-train-windows", type=int, default=40)
    ap.add_argument("--scenarios", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=PARAMS["time_limit"])
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    grid = np.array([float(x) for x in args.rho_grid.split(",")])
    agent = LinUCB(len(grid), 6, alpha=args.alpha)
    os.makedirs(args.out, exist_ok=True)
    p = dict(PARAMS)

    n_seen = 0
    for run_dir in args.run_dirs:
        run = load_run(run_dir)
        H = int(run["q"].shape[1])
        n = int(run["q"].shape[0])
        n_train = min(args.max_train_windows,
                      max(1, int(0.6 * n))) if n > 1 else 1
        for i in range(n_train):
            loads = window_loads(run, i, p)
            scen = scenario_demands(run, i, p, args.scenarios, seed=2024 + i)
            probs = np.full(len(scen), 1.0 / len(scen))
            x = features(loads)
            a = agent.select(x)
            reward, _ = _reward(grid[a], H, p, loads, scen, probs,
                                args.time_limit, grid)
            agent.update(a, x, reward)
            n_seen += 1
            if n_seen % 10 == 0:
                print(f"  bandit updates {n_seen}", flush=True)

    with open(os.path.join(args.out, "rl_policy.json"), "w") as f:
        json.dump({"rho_grid": grid.tolist(), **agent.to_dict()}, f)
    print(f"wrote {os.path.join(args.out, 'rl_policy.json')} "
          f"after {n_seen} updates")


if __name__ == "__main__":
    main()
