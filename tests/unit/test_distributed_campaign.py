"""Unit tests for the distributed-MPAS campaign hook composition
(:mod:`legoesm.training.distributed_campaign`, iter 89).

Pins (without MPI): the ``(valid_mask, manifest_reducer, global_reduce)`` triple is
built consistently from ONE partition layout, and the global ERA5 reference is
sliced to a rank's local cells in order.  The collective behaviour of the returned
reducer/reduce is covered by the distributed end-to-end test.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.training.compare_reanalysis import ColumnState, owned_cell_valid_mask
from legoesm.training.distributed_campaign import (
    distributed_campaign_hooks,
    slice_reference_to_local,
)
from legoesm.training.distributed_manifest import gather_global_worst_columns


def _layout(owned_mask, local_cells):
    return SimpleNamespace(
        owned_mask_cells=jnp.asarray(owned_mask),
        partition=SimpleNamespace(local_cells=np.asarray(local_cells)),
    )


def test_distributed_campaign_hooks_compose_from_one_layout():
    from legoesm.parallel.reductions import global_sum_mpi

    owned = [True, True, False]          # 2 owned, 1 halo
    local_cells = [10, 11, 12]           # local→global ids
    layout = _layout(owned, local_cells)

    valid_mask, manifest_reducer, global_reduce = distributed_campaign_hooks(
        layout, n_worst=3)

    # valid_mask is exactly the owned-cell mask (iter 86).
    np.testing.assert_array_equal(
        np.asarray(valid_mask), np.asarray(owned_cell_valid_mask(layout)))
    # global_reduce is the collective SUM (iter 88).
    assert global_reduce is global_sum_mpi
    # manifest_reducer is the gather (iter 87) bound to THIS layout's id map + n_worst.
    assert isinstance(manifest_reducer, partial)
    assert manifest_reducer.func is gather_global_worst_columns
    np.testing.assert_array_equal(
        manifest_reducer.keywords["local_to_global"], local_cells)
    assert manifest_reducer.keywords["n_worst"] == 3


def test_distributed_campaign_hooks_threads_base_mask():
    owned = [True, True, True, False]
    base = [True, False, True, True]     # cell 1 is land/invalid
    layout = _layout(owned, [0, 1, 2, 3])
    valid_mask, _, _ = distributed_campaign_hooks(
        layout, n_worst=2, base_valid_mask=jnp.asarray(base))
    # owned AND base: rankable only if owned AND base-valid.
    np.testing.assert_array_equal(
        np.asarray(valid_mask), np.array([True, False, True, False]))


def test_distributed_campaign_hooks_reducer_runs_via_monkeypatched_allgather(monkeypatch):
    """The composed reducer is a working manifest→manifest function: with a
    single-rank (identity) allgather it returns this rank's own worst-k unchanged."""
    def _fake_allgather(x):
        return jnp.asarray(np.asarray(x)[None, ...])     # (1,) + shape, like np=1

    monkeypatch.setattr(
        "legoesm.parallel.reductions.allgather_mpi", _fake_allgather)
    layout = _layout([True, True, True], [100, 101, 102])
    _, manifest_reducer, _ = distributed_campaign_hooks(layout, n_worst=2)
    recs = [SimpleNamespace(flat_index=2, combined_score=7.0),
            SimpleNamespace(flat_index=0, combined_score=4.0)]
    out = manifest_reducer(recs)
    assert [r.flat_index for r in out] == [2, 0]          # worst-first, both kept


def _global_ref(ncells=4, nlev=3):
    rng = np.arange(ncells * nlev, dtype=float).reshape(ncells, nlev)
    return ColumnState(
        T=jnp.asarray(280.0 + rng), q_v=jnp.asarray(1e-3 + 0.0 * rng),
        u=jnp.asarray(rng), v=jnp.zeros((ncells, nlev)),
        p_s=jnp.asarray(1.0e5 + np.arange(ncells, dtype=float)),
        sst_K=jnp.asarray(290.0 + np.arange(ncells, dtype=float)),
    )                                  # precip_mm_day / u_edge default None


def test_slice_reference_to_local_rejects_edge_field_reference():
    """A reference carrying u_edge (the model's EDGE velocity, nEdges) is REJECTED —
    a cell-id slice would corrupt that edge field (Codex iter 89)."""
    ref = _global_ref(ncells=4, nlev=3)._replace(
        u_edge=jnp.zeros((10, 3)))                    # nEdges != nCells, edge-indexed
    with pytest.raises(ValueError, match="must be a CELL-space"):
        slice_reference_to_local(ref, [0, 1])


def test_slice_reference_to_local_rejects_out_of_range_cell_id():
    """An out-of-range cell id is REJECTED LOUDLY — a JAX gather silently CLAMPS
    overflowing ids, so a too-small / wrong-mesh GLOBAL reference would corrupt the
    compare on a multi-day run with no error (iter 97)."""
    ref = _global_ref(ncells=4, nlev=3)              # cells 0..3
    with pytest.raises(ValueError, match="out of range.*4 cells"):
        slice_reference_to_local(ref, [2, 3, 4])     # 4 ∉ [0,4)
    with pytest.raises(ValueError, match="out of range"):
        slice_reference_to_local(ref, [-1, 0])       # negative id


def test_slice_reference_to_local_rejects_inconsistent_field_lengths():
    """EVERY per-cell field is gathered by the SAME ids, so they must share the
    cell-axis length — an inconsistent later field (here T has 5 cells, q_v has 4)
    is caught BEFORE the gather silently clamps it (Codex iter 97, HIGH)."""
    ref = _global_ref(ncells=4, nlev=3)._replace(T=jnp.zeros((5, 3)))   # T longer
    with pytest.raises(ValueError, match="inconsistent reference cell-axis"):
        slice_reference_to_local(ref, [0, 1])


def test_slice_reference_to_local_rejects_expected_n_cells_mismatch():
    """``expected_n_cells`` enforces EXACT mesh identity — a reference whose cell-count
    ≠ the partitioned global mesh is rejected even when every id is in range (catches a
    reference LONGER than the mesh, which the bounds check alone misses; iter 97)."""
    ref = _global_ref(ncells=4, nlev=3)
    with pytest.raises(ValueError, match="reference has 4 cells.*global mesh has 5"):
        slice_reference_to_local(ref, [0, 1], expected_n_cells=5)
    # the exact count passes through (in-range ids, matching N).
    out = slice_reference_to_local(ref, [0, 1], expected_n_cells=4)
    assert out.T.shape == (2, 3)


def test_slice_reference_to_local_gathers_cells_in_order():
    ref = _global_ref(ncells=4, nlev=3)
    local_cells = [2, 3, 1]            # owned + halo, NOT contiguous / sorted
    out = slice_reference_to_local(ref, local_cells)
    # Every per-cell field is gathered by local_cells, ORDER preserved.
    assert out.T.shape == (3, 3) and out.p_s.shape == (3,)
    np.testing.assert_array_equal(np.asarray(out.T), np.asarray(ref.T)[[2, 3, 1]])
    np.testing.assert_array_equal(np.asarray(out.p_s), np.asarray(ref.p_s)[[2, 3, 1]])
    np.testing.assert_array_equal(np.asarray(out.sst_K), np.asarray(ref.sst_K)[[2, 3, 1]])
    # None fields pass through unchanged (cell-wind reference has no precip/u_edge).
    assert out.precip_mm_day is None and out.u_edge is None
