#!/usr/bin/env python3
"""Local zero-compute checks: syntax, AST, configs, forbidden tokens.

This is the only check allowed on the local machine.  It does not import
torch, build a model, read the dataset, sample, or solve a MILP.
"""

from __future__ import annotations

import ast
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAIL = []


def fail(msg: str) -> None:
    FAIL.append(msg)
    print(f"[FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"[ok] {msg}")


def check_python() -> None:
    for path in sorted((ROOT / "src").rglob("*.py")):
        src = path.read_text()
        try:
            compile(src, str(path), "exec")
        except SyntaxError as e:
            fail(f"syntax {path}: {e}")
            continue
        try:
            ast.parse(src)
        except SyntaxError as e:
            fail(f"AST {path}: {e}")
    ok("src/*.py syntax + AST")


def check_forbidden_tokens() -> None:
    tokens = ["future" + "_branch", "use_future" + "_weather",
              "n" + "wp"]
    hits = []
    for base in ("src", "configs"):
        for path in sorted((ROOT / base).rglob("*")):
            if not path.is_file() or path.suffix not in (
                    ".py", ".yaml", ".yml", ".json"):
                continue
            text = path.read_text().lower()
            for tok in tokens:
                if tok in text:
                    hits.append(f"{path}: {tok}")
    if hits:
        for h in hits:
            fail(f"forbidden token {h}")
    else:
        ok("no forbidden exogenous-forecast tokens in src/configs")


def check_required_files() -> None:
    required = [
        "pyproject.toml", "requirements.txt", "README.md",
        "PAPER_PROTOCOL.md", "data/dataset_input.csv",
        "src/percnet_epsr/data.py", "src/percnet_epsr/metrics.py",
        "src/percnet_epsr/models/percnet_v2.py",
        "src/percnet_epsr/models/xcpd.py",
        "src/percnet_epsr/protocols/official_festrnn.py",
        "src/percnet_epsr/training/runner.py",
        "src/percnet_epsr/experiments/scheduling/two_stage.py",
        "src/percnet_epsr/experiments/scheduling/rl_reserve.py",
    ]
    for rel in required:
        if not (ROOT / rel).is_file():
            fail(f"missing {rel}")
    if not FAIL:
        ok(f"{len(required)} required files present")


def check_data_shape() -> None:
    path = ROOT / "data/dataset_input.csv"
    if not path.is_file():
        return
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        first = next(reader)
    if len(header) != 12 or len(first) != 12:
        fail(f"dataset header/data columns {len(header)}/{len(first)} != 12")
    else:
        ok("dataset has 12 numeric columns")


def check_configs() -> None:
    try:
        import yaml
    except ImportError:
        fail("pyyaml not installed (needed for config validation)")
        return
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        try:
            with open(path) as f:
                yaml.safe_load(f)
        except Exception as e:
            fail(f"yaml {path}: {e}")
    ok("yaml configs parse")


def main() -> int:
    if not (ROOT / "src").is_dir():
        print("[FAIL] run from the repository root")
        return 1
    check_python()
    check_forbidden_tokens()
    check_required_files()
    check_data_shape()
    check_configs()
    print()
    if FAIL:
        print(f"STATIC CHECK FAILED ({len(FAIL)} issues)")
        return 1
    print("STATIC CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
