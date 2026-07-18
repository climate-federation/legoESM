"""Smoke test for the coupled atm+slab-ocean bench (C10) — single-process.

``run_coupled_bench`` at n_ranks=1 is the serial band-equivalent (the halo
backend's single-member ring is the local wrap), so it runs without a launcher;
this pins that it completes, reports a positive throughput, and that the
one-exchange coupled-surface-energy drift is at round-off (the conservation
diagnostic the bench emits). The full serial==band-MPI parity is gated by
``tests/distributed/test_coupled_latlon_mpi.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

_BENCH = (Path(__file__).resolve().parents[2]
          / "scripts" / "bench" / "bench_coupled_latlon_scaling.py")


def _load():
    spec = importlib.util.spec_from_file_location(
        "bench_coupled_latlon_scaling", _BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bench_runs_and_conserves():
    mod = _load()
    m = mod.run_coupled_bench(
        n_lat=8, nlev=4, n_intervals=3, n_atm_substeps=2, dt=100.0,
        n_warmup=1, rank=0, n_ranks=1)
    assert m["coupled_steps_per_s"] > 0.0
    assert m["band_rows"] == 8
    assert m["n_lon"] == 16
    # The explicit sensible exchange conserves the coupled surface energy to
    # float64 round-off (single-process => no allreduce reassociation).
    assert m["energy_drift_one_exchange"] < 1e-13
