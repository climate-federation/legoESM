"""``scripts/bench/bench_halo_exchange.py --payload state`` under real MPI.

The state mode times the production batched MPAS state exchange and reports
a per-rank digest of the exchanged arrays; the GPU-direct vs host-staged
comparison rests on that digest, so it must be reproducible and must change
when the exchanged halo values change.  Run with ``mpirun -np 2``; skipped
on one rank.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "bench" / "bench_halo_exchange.py")


def _bench():
    spec = importlib.util.spec_from_file_location("bench_halo_exchange", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _size():
    try:
        from mpi4py import MPI
    except ImportError:
        pytest.skip("mpi4py not available")
    if MPI.COMM_WORLD.Get_size() < 2:
        pytest.skip("Need at least 2 MPI ranks")
    return MPI.COMM_WORLD.Get_size()


def test_state_payload_schema_and_reproducible_digest():
    n = _size()
    mod = _bench()
    a = mod._bench_voronoi_state(level=2, nlev=3, n_tracers=2,
                                 n_warmup=1, n_iters=2)
    b = mod._bench_voronoi_state(level=2, nlev=3, n_tracers=2,
                                 n_warmup=1, n_iters=2)
    assert a["payload"] == "state" and a["n_ranks"] == n
    assert len(a["per_rank_digest"]) == n
    assert a["per_rank_digest"] == b["per_rank_digest"]
    t = a["exchanges"]["state"]["time_per_exchange_ms"]
    assert t > 0.0 and t == max(a["exchanges"]["state"]["per_rank_median_ms"])


def test_state_digest_sees_the_payload():
    _size()
    mod = _bench()
    a = mod._bench_voronoi_state(level=2, nlev=3, n_tracers=2,
                                 n_warmup=0, n_iters=1)
    c = mod._bench_voronoi_state(level=2, nlev=3, n_tracers=3,
                                 n_warmup=0, n_iters=1)
    assert a["per_rank_digest"] != c["per_rank_digest"]
