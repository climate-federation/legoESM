"""Smoke test for the spectral-LES throughput benchmark.

Runs a few steps on a tiny grid and checks the timer returns a positive
throughput, a finite state, and the derived sim-h/wall-h figure. The benchmark
is GPU-ready / single-compile; here we only assert it runs on the CPU backend.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
_ROOT = Path(__file__).resolve().parents[2]


def _load_bench():
    path = _ROOT / "scripts" / "bench" / "bench_spectral_les.py"
    spec = importlib.util.spec_from_file_location("bench_spectral_les", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bench_one_runs_and_reports():
    m = _load_bench()
    r = m.bench_one(8, 8, 16, Lx=400.0, Ly=400.0, Lz=800.0, steps=3, dt=0.5,
                    dtype=jnp.float64, sgs_dynamic=True)
    assert r["steps_per_s"] > 0.0
    assert r["per_step_ms"] > 0.0
    assert r["sim_h_per_wall_h"] > 0.0
    assert r["finite"], "benchmark IC integrated to a non-finite state"
    assert r["ncell"] == 8 * 8 * 16
