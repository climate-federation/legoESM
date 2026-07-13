"""Direct test for scripts/bench/bench_fv3_sw_fb_vs_production.py (M1).

Imports the bench module and runs 2 AOT-timed steps at C12 for both cores
(every new .py gets a direct test).  x64 CPU-safe; tiny grid keeps it fast.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_bench_module():
    script = (Path(__file__).resolve().parents[2]
              / "scripts" / "bench" / "bench_fv3_sw_fb_vs_production.py")
    name = "_bench_fb_vs_prod_unit"
    spec = importlib.util.spec_from_file_location(name, script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def test_bench_runs_two_steps_c12():
    B = _load_bench_module()
    results = B.run_bench(["C12"], steps=2, warmup=1,
                          cores=("production", "fb"))
    cores = {r["core"] for r in results["rows"]}
    assert cores == {"production", "fb"}
    for r in results["rows"]:
        assert r["per_step_s"] > 0.0
        assert r["n"] == 12


def test_bench_unknown_core_raises():
    B = _load_bench_module()
    with pytest.raises(ValueError, match="unknown core"):
        B._build("bogus", 12)
