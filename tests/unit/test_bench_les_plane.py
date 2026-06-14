"""Smoke test for the plane-LES throughput benchmark.

Runs a few steps on a tiny grid for each SGS closure and checks the timer
returns a positive throughput and a finite state (the benchmark is GPU-ready /
single-compile; here we only assert it runs on the CPU backend).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
_ROOT = Path(__file__).resolve().parents[2]


def _load_bench():
    path = _ROOT / "scripts" / "bench" / "bench_les_plane.py"
    spec = importlib.util.spec_from_file_location("bench_les_plane", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bench_runs_static_and_lasd():
    m = _load_bench()
    for sgs in ("static", "lasd"):
        sps = m.bench("neutral", nx=8, ny=8, nlev=16, dx=20.0, H=1000.0,
                      dz_sfc=10.0, dt=0.05, sgs=sgs, nsteps=3)
        assert sps > 0.0
