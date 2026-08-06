"""Multi-process (srun/mpirun -n 2) tests for the cross-process geometry guard.

Run:  srun -n 2 python -m pytest tests/distributed/test_geometry_consistency_mp.py
      (or mpirun -np 2 ...)

Covers what the serial suite CANNOT. Every guard in
``legoesm.parallel.geometry_consistency`` early-returns at
``jax.process_count() <= 1``, so the ~150 single-process tests exercise the
fingerprints that DECIDE whether a collective raises -- never the collectives
themselves. That gap matters because the failure mode is not a wrong answer but
a HANG: a rank-local ``raise`` ordered before a peer enters the matching
collective deadlocks, which is strictly worse than the inconsistency being
guarded against, and no single-process test can tell the two apart.

The governing invariant (see ``geometry_consistency`` module docstring):

    Every process must reach the same collectives in the same order.

So each test below asserts not merely that a divergence is detected, but that it
is detected SYMMETRICALLY -- every rank raises, none is left blocked.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def _launcher_size() -> int:
    """World size of THIS invocation, without importing mpi4py.

    The guard uses ``jax.distributed`` / ``multihost_utils``, not mpi4jax, so
    this tier only needs the process count -- read it from whichever launcher is
    in play rather than taking on an extra dependency.

    ``SLURM_NTASKS`` is deliberately NOT used: it reports the JOB ALLOCATION, so
    a plain ``python -m pytest`` inside a 2-task allocation reads 2, tries to
    ``jax.distributed.initialize()`` alone, and dies with a coordinator
    ``ValueError`` instead of skipping. ``SLURM_STEP_NUM_TASKS`` is set by srun
    per STEP and is absent for a non-srun invocation, which is exactly the
    distinction needed.
    """
    for key in ("SLURM_STEP_NUM_TASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE",
                "MV2_COMM_WORLD_SIZE"):
        value = os.environ.get(key)
        if value and value.isdigit():
            return int(value)
    return 1


SIZE = _launcher_size()

pytestmark = pytest.mark.skipif(
    SIZE < 2, reason="needs srun -n 2 / mpirun -np 2 (guards no-op at process_count == 1)",
)

if SIZE >= 2:  # pragma: no cover - only under a multi-process launcher
    import jax

    # MUST precede any other JAX work in the process.
    jax.distributed.initialize()


def _guards():
    from legoesm.parallel.geometry_consistency import (
        assert_flags_agree,
        assert_schema_agrees,
        broadcast_checked,
    )

    return assert_schema_agrees, assert_flags_agree, broadcast_checked


def _rank() -> int:
    import jax

    return int(jax.process_index())


def test_process_count_is_really_multi():
    """Anti-vacuity for the whole module: if this collapses to 1, every test
    below passes trivially because the guards early-return."""
    import jax

    assert jax.process_count() >= 2, (
        f"launcher reported {SIZE} but jax.process_count() == {jax.process_count()}; "
        "the guards would early-return and these tests would be vacuous"
    )


def test_agreeing_geometry_passes_on_every_rank():
    """Identical inputs: no rank raises, and no rank blocks."""
    assert_schema_agrees, assert_flags_agree, broadcast_checked = _guards()

    assert_schema_agrees(("lat", "lon"), 2, context="mp-test")
    assert_flags_agree(("alpha", "beta"), (1.0, 0.0), context="mp-test")
    out = broadcast_checked(np.arange(4, dtype=np.float64), "field", context="mp-test")
    np.testing.assert_allclose(out, np.arange(4, dtype=np.float64))


def test_axis_order_divergence_raises_on_every_rank():
    """The sharp case: ``(lat,lon)`` vs ``(lon,lat)`` is invisible to a
    count-and-size payload, so this is what the name digest exists for.

    Both ranks must raise. If only one did, the other would still be inside
    ``process_allgather`` and the job would hang rather than fail.
    """
    assert_schema_agrees, _, _ = _guards()
    names = ("lat", "lon") if _rank() == 0 else ("lon", "lat")
    with pytest.raises(RuntimeError, match="SCHEMA differs"):
        assert_schema_agrees(names, 2, context="mp-test")


def test_flag_value_divergence_raises_on_every_rank():
    _, assert_flags_agree, _ = _guards()
    values = (1.0, 0.0) if _rank() == 0 else (1.0, 1.0)
    with pytest.raises(RuntimeError, match="CONFIG differs"):
        assert_flags_agree(("alpha", "beta"), values, context="mp-test")


def test_broadcast_rejects_divergent_content_instead_of_overwriting():
    """``broadcast_checked`` must REJECT a divergence, never silently replace
    every rank's array with rank 0's -- that would convert a real inconsistency
    (different wet domain, different polar mask) into silently wrong physics,
    which is the entire reason the broadcast is guarded rather than blind.
    """
    _, _, broadcast_checked = _guards()
    arr = np.arange(4, dtype=np.float64) + (0.0 if _rank() == 0 else 1.0)
    with pytest.raises(RuntimeError, match="DIVERGES"):
        broadcast_checked(arr, "field", context="mp-test")


def test_guards_still_agree_after_a_divergence_was_raised():
    """Ordering regression guard: a raised divergence must leave the collective
    stream aligned, so a SUBSEQUENT agreeing call still works on every rank.

    If a guard consumed a different number of collectives on the raising path
    than on the passing path, this is where the ranks would desynchronise.
    """
    assert_schema_agrees, _, broadcast_checked = _guards()
    out = broadcast_checked(np.full(3, 2.5, dtype=np.float64), "after", context="mp-test")
    np.testing.assert_allclose(out, 2.5)
    assert_schema_agrees(("lat", "lon"), 2, context="mp-test")
