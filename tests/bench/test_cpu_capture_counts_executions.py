"""A CPU capture must count EXECUTIONS, not compiled instructions.

This is the assumption the CPU collective census rests on, and getting it
wrong is not a small error: the ocean barotropic solver runs twenty
iterations inside one compiled loop, so an instrument that counted the
module instead of the executions would report one reduction per step where
there are twenty, and the step's cost would be attributed to the halo.

The test drives the REAL pipeline -- capture, then
``analyze_jax_trace_gaps.time_by_family`` -- over a program whose answer is
known by construction, and pins the distinguishing property: doubling the
loop's trip count doubles the events.  A per-invocation annotation would
return the same number both times.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp                                    # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P          # noqa: E402

_SRC = Path(__file__).resolve().parents[2] / "scripts/bench/analyze_jax_trace_gaps.py"
_spec = importlib.util.spec_from_file_location("azg", _SRC)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

AXIS = "d"
N_DEV = 4
N_STEPS = 3


def _capture(tmp_path: Path, trips: int) -> list[dict]:
    devices = jax.devices()
    if len(devices) < N_DEV:
        pytest.skip(f"needs {N_DEV} devices, have {len(devices)}")
    mesh = Mesh(np.array(devices[:N_DEV]), (AXIS,))
    perm = [(i, (i + 1) % N_DEV) for i in range(N_DEV)]

    @jax.jit
    def step(x):
        def body(xl):
            def one(_, v):
                v = jax.lax.ppermute(v, AXIS, perm)
                return jax.lax.psum(v, AXIS) / N_DEV

            return jax.lax.fori_loop(0, trips, one, xl)

        return jax.shard_map(body, mesh=mesh, in_specs=P(AXIS),
                             out_specs=P(AXIS), check_vma=False)(x)

    x = jnp.ones((N_DEV * 256,), jnp.float32)
    step(x).block_until_ready()          # compile outside the capture
    out = tmp_path / f"trace{trips}"
    jax.profiler.start_trace(str(out))
    for _ in range(N_STEPS):
        step(x).block_until_ready()
    jax.profiler.stop_trace()

    files = list(out.rglob("*.trace.json.gz"))
    assert files, f"no trace written under {out}"
    evs = []
    for f in files:
        evs += [e for e in json.load(gzip.open(f)).get("traceEvents", [])
                if e.get("ph") == "X"]
    return evs


@pytest.mark.parametrize("trips", [5, 10])
def test_executions_inside_a_compiled_loop_are_each_counted(tmp_path, trips):
    evs = _capture(tmp_path, trips)
    got = mod.time_by_family(evs, N_STEPS, devices_per_rank=N_DEV)
    want = N_STEPS * trips * N_DEV
    for family in ("permute", "all-reduce"):
        n = got["families"].get(family, {}).get("instructions", 0)
        assert n == want, (
            f"{family}: {n} events for {N_STEPS} steps x {trips} loop "
            f"iterations x {N_DEV} devices; expected {want}. A count that "
            f"does not move with the trip count is counting compiled "
            f"instructions, not executions.")


def test_the_count_moves_with_the_trip_count(tmp_path):
    """The property that distinguishes the two instruments.

    Both a per-execution and a per-invocation annotation give the 'right
    looking' number for a single trip count -- only the ratio separates
    them.
    """
    five = mod.time_by_family(_capture(tmp_path, 5), N_STEPS,
                              devices_per_rank=N_DEV)
    ten = mod.time_by_family(_capture(tmp_path, 10), N_STEPS,
                             devices_per_rank=N_DEV)
    for family in ("permute", "all-reduce"):
        a = five["families"][family]["instructions"]
        b = ten["families"][family]["instructions"]
        assert b == 2 * a, (
            f"{family}: doubling the loop trip count took the count from "
            f"{a} to {b}. If it did not double, the capture annotates per "
            f"invocation and every in-loop collective is undercounted.")
