"""Distributed (MPI) deadlock test for the COLLECTIVE correction loop (iter 88).

Run: ``mpirun -np 2 .venv/bin/python -m pytest <this file>``

The iter-86/87 owned-mask + global top-k make distributed-MPAS RANKING correct, but
the correction LOOP itself was rank-local in two ways that deadlock/diverge under
MPI (Codex iter 87): (1) the ``if not records:`` no-op gate skips the second
``compare_fn`` (model re-run, an MPI collective) on a rank owning NONE of the
globally-selected columns while peers re-run; (2) the line-search improvement test
used the rank-local bias, so ranks could choose different step fractions (different
numbers of collective ``compare_fn`` calls).  Iter 88 drives both off a global
reduction (``global_reduce=global_sum_mpi``): the gate uses the GLOBAL selected
count, the line-search uses the GLOBAL bias.

This test makes ``compare_fn`` itself COLLECTIVE (a ``global_sum_mpi`` every call),
so any per-rank divergence in the NUMBER of ``compare_fn`` calls mismatches that
collective and the job HANGS (caught by the timeout).  Rank 0 owns a flagged
column; rank 1 owns none.  With the global gate both ranks proceed in lockstep and
the run COMPLETES with an identical (global) improvement verdict.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("mpi4py")
pytest.importorskip("mpi4jax")
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig  # noqa: E402
from legoesm.parallel.reductions import global_sum_mpi  # noqa: E402
from legoesm.training.correction_loop import (  # noqa: E402
    CompareResult,
    run_correction_iteration,
)
from mpi4py import MPI  # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NPROC = COMM.Get_size()

_AREA_W = jnp.ones((2, 2))


class _Eddy(NamedTuple):
    K: jax.Array
    valid: jax.Array


class _Env(NamedTuple):
    cape: float        # mock env tag (unused — no env clustering in this test)


class _Rec(NamedTuple):
    flat_index: int
    lat_deg: float
    environment: _Env


def _diagnose(record, model_ctx):
    return _Eddy(K=jnp.array([10.0, 20.0]), valid=jnp.array([True, True]))


def _collective_compare_fn(config):
    """A COLLECTIVE compare: the global_sum_mpi makes a per-rank divergence in the
    number of compare calls deadlock.  Rank 0 owns one flagged column (its flat-0
    score is high in baseline, lowered once corrected); rank 1 owns none."""
    _ = global_sum_mpi(jnp.asarray(1.0))           # the lockstep barrier
    corrected = jnp.ndim(jnp.asarray(config.tau_equator)) > 0
    if RANK == 0:
        score = ([[1.0, 1.0], [1.0, 1.0]] if corrected else [[3.0, 1.0], [1.0, 1.0]])
        manifest = [_Rec(flat_index=0, lat_deg=10.0, environment=_Env(200.0))]
    else:
        score = [[1.0, 1.0], [1.0, 1.0]]           # rank 1: uniform, nothing flagged
        manifest = []
    return CompareResult(
        combined_score=jnp.asarray(score), manifest=manifest,
        area_weights=_AREA_W, model_ctx=None,
    )


@pytest.mark.timeout(120)
@pytest.mark.skipif(NPROC < 2, reason="needs >=2 ranks to exercise the collective gate")
def test_collective_loop_no_deadlock_with_empty_local_manifest():
    """rank 1 owns no flagged column, rank 0 does — with global_reduce the gate +
    line search are GLOBAL so BOTH ranks run compare_fn the same number of times
    (no collective mismatch), the run completes, and the improvement verdict is
    IDENTICAL on every rank."""
    result = run_correction_iteration(
        GrayRadiationConfig(),
        compare_fn=_collective_compare_fn, diagnose_fn=_diagnose,
        promotion_key="gray_tau_equator", grid_shape=(2, 2), background=7.2,
        global_reduce=global_sum_mpi,
    )
    # Reaching here at all means no rank deadlocked on the collective compare.
    improved = bool(result.bias.improved)
    all_improved = COMM.allgather(improved)
    assert all(v == all_improved[0] for v in all_improved), (
        f"ranks disagree on the GLOBAL improved verdict: {all_improved}")
    # Rank 0's flat-0 correction lowers the global bias ⇒ improved on EVERY rank.
    assert all_improved[0] is True
    # Each rank corrected only the columns it owns (rank 1 owns none).
    assert result.n_corrected == (1 if RANK == 0 else 0)
    COMM.Barrier()


@pytest.mark.timeout(120)
@pytest.mark.skipif(NPROC < 2, reason="needs >=2 ranks to show the divergence")
def test_rank_local_gate_would_diverge_without_global_reduce():
    """NON-VACUITY: with ``global_reduce=None`` (the rank-local gate) the ranks call
    ``compare_fn`` a DIFFERENT number of times — rank 0 (owns a column) re-runs in
    the line search, rank 1 (owns none) stops at the no-op gate.  A NON-collective
    counter shows the divergence WITHOUT hanging; were ``compare_fn`` collective
    (as a real distributed model is), this count mismatch is the deadlock the
    iter-88 global gate prevents."""
    calls = {"n": 0}

    def counting_compare(config):
        calls["n"] += 1                            # NOT collective → no hang here
        corrected = jnp.ndim(jnp.asarray(config.tau_equator)) > 0
        if RANK == 0:
            score = ([[1.0, 1.0], [1.0, 1.0]] if corrected else [[3.0, 1.0], [1.0, 1.0]])
            manifest = [_Rec(flat_index=0, lat_deg=10.0, environment=_Env(200.0))]
        else:
            score = [[1.0, 1.0], [1.0, 1.0]]
            manifest = []
        return CompareResult(jnp.asarray(score), manifest, _AREA_W, None)

    run_correction_iteration(
        GrayRadiationConfig(),
        compare_fn=counting_compare, diagnose_fn=_diagnose,
        promotion_key="gray_tau_equator", grid_shape=(2, 2), background=7.2,
        global_reduce=None,                        # the OLD rank-local behaviour
    )
    counts = COMM.allgather(calls["n"])
    # rank 0 re-runs (baseline + line search = 2); rank 1 no-ops at the gate (1).
    assert counts[0] == 2 and counts[1] == 1
    assert counts[0] != counts[1], "ranks would deadlock a collective compare_fn"
    COMM.Barrier()
