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
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.training.compare_reanalysis import ColumnState, owned_cell_valid_mask
from legoesm.training.distributed_campaign import (
    assemble_global_campaign_result,
    assemble_global_field,
    assert_partition_covers_global,
    distributed_campaign_hooks,
    slice_reference_to_local,
)
from legoesm.training.distributed_manifest import gather_global_worst_columns


def _layout(owned_mask, local_cells, n_global=None):
    lc = np.asarray(local_cells)
    if n_global is None:
        n_global = int(lc.max()) + 1 if lc.size else 0
    return SimpleNamespace(
        owned_mask_cells=jnp.asarray(owned_mask),
        partition=SimpleNamespace(local_cells=lc, nCells_global=n_global),
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


# a single-rank stand-in for the allreduce SUM: with one rank the global owned-count
# IS this rank's local count, so the identity is the correct np=1 reduction.
def _serial_sum(x):
    return x


def test_assert_partition_covers_global_accepts_a_clean_partition():
    """Owned sets that tile the global mesh EXACTLY once pass (iter 98)."""
    # one rank owning all 4 cells (np=1 semantics) — every cell counted once.
    assert_partition_covers_global(
        _layout([True, True, True, True], [0, 1, 2, 3], n_global=4),
        global_reduce=_serial_sum) is None


def test_assert_partition_covers_global_rejects_a_gap():
    """A global cell owned by NO rank is REJECTED — it would never be ranked or
    corrected, a permanent bias the loop cannot see (iter 98)."""
    # owns 0,1,3 of a 4-cell mesh ⇒ cell 2 is a gap.
    with pytest.raises(ValueError, match="owned by NO rank.*\\[2\\]"):
        assert_partition_covers_global(
            _layout([True, True, True], [0, 1, 3], n_global=4),
            global_reduce=_serial_sum)


def test_assert_partition_covers_global_rejects_an_overlap():
    """A global cell owned by >1 rank is REJECTED — it is double-counted in the
    global top-k and double-corrected (iter 98)."""
    # local_cells repeats id 2 (as if two ranks both owned it) ⇒ overlap.
    with pytest.raises(ValueError, match="owned by >1 rank.*\\[2\\]"):
        assert_partition_covers_global(
            _layout([True, True, True, True], [0, 1, 2, 2], n_global=3),
            global_reduce=_serial_sum)


def test_assert_partition_covers_global_only_counts_owned_cells():
    """HALO (non-owned) cells do NOT count toward coverage — only owned cells tile
    the mesh; a halo copy of another rank's owned cell is not an overlap (iter 98)."""
    # owns 0,1; holds 2 as a HALO (owned=False). Cell 2 would be a gap here, but a
    # second 'rank' contributes it — emulate by summing two local count vectors.
    rank0 = _layout([True, True, False], [0, 1, 2], n_global=3)
    rank1 = _layout([False, True], [1, 2], n_global=3)   # owns 2; halo of 1

    # compose the two ranks' contributions through the same code path. zeros_like keeps
    # the extra structural-flag slots (the reduced vector is n_global+2 long).
    def _other_owns(ids):
        return lambda x: x + jnp.zeros_like(x).at[jnp.asarray(ids)].add(1)

    # rank0 owns 0,1 (cell 2 is its halo); the other rank owns 2 ⇒ clean cover.
    assert assert_partition_covers_global(
        rank0, global_reduce=_other_owns([2])) is None
    # rank1 owns only 2; the other rank owns 0,1 ⇒ clean cover.
    assert assert_partition_covers_global(
        rank1, global_reduce=_other_owns([0, 1])) is None


def test_assert_partition_covers_global_rejects_mismatched_mask_length():
    """owned_mask_cells and local_cells must be the same length (one flag per local
    cell) — a mismatch is a malformed layout (iter 98)."""
    with pytest.raises(ValueError, match="same length"):
        assert_partition_covers_global(
            _layout([True, True], [0, 1, 2], n_global=3), global_reduce=_serial_sum)


def test_assert_partition_covers_global_rejects_non_1d_mask():
    """A non-1-D owned mask is malformed — folded into the structural flag so the
    later ``local_cells[owned]`` cannot raise an IndexError BEFORE the collective on
    one rank only (Codex iter 98, collective-safety)."""
    bad = SimpleNamespace(
        owned_mask_cells=jnp.ones((3, 1), dtype=bool),       # 2-D, shape[0]==3 matches
        partition=SimpleNamespace(local_cells=np.array([0, 1, 2]), nCells_global=3))
    with pytest.raises(ValueError, match="1-D mask"):
        assert_partition_covers_global(bad, global_reduce=_serial_sum)


def test_assert_partition_covers_global_rejects_out_of_range_owned_id():
    """An owned global id ≥ nCells_global is a malformed partition — caught (and
    synchronized) rather than silently clamped by the scatter (iter 98)."""
    with pytest.raises(ValueError, match="out of range"):
        assert_partition_covers_global(
            _layout([True, True, True], [0, 1, 5], n_global=4),
            global_reduce=_serial_sum)             # owns id 5 of a 4-cell mesh


# --- assemble_global_field: the FORWARD inverse of slice_reference_to_local (iter 253).
def test_assemble_global_field_single_rank_round_trips_by_global_id():
    """One rank owning every cell: the assembled global field places each LOCAL value
    at its GLOBAL id (reordered out of local order), the exact inverse of a slice."""
    # local order is NOT global order: local cell k holds global id local_cells[k].
    layout = _layout([True, True, True, True], [2, 0, 3, 1], n_global=4)
    local_field = jnp.asarray([0.5, 1.5, 2.5, 3.5])   # value for local cell k
    g = assemble_global_field(layout, local_field, global_reduce=_serial_sum)
    # global id 0 got local-1's value 1.5; id 1 <- local-3 3.5; id 2 <- local-0 0.5; ...
    np.testing.assert_array_equal(np.asarray(g), [1.5, 3.5, 0.5, 2.5])


def test_assemble_global_field_gathers_across_ranks_and_ignores_halo():
    """Two ranks tiling the mesh: each contributes ONLY its owned cells; a rank's HALO
    copy of another rank's cell (owned=False, garbage value) must NOT leak."""
    # rank0 owns global 0,1 and holds 2 as a halo with a GARBAGE value (must be ignored).
    rank0 = _layout([True, True, False], [0, 1, 2], n_global=3)
    field0 = jnp.asarray([0.25, 0.75, 999.0])         # 999.0 is the halo (non-owned)
    # the OTHER rank owns global 2 with value 0.5 — compose its scattered contribution.
    # verify_coverage=False isolates the GATHER here (the gate is exercised separately);
    # the single closure cannot serve both the gate's int32 COUNT buffer and this float
    # VALUE buffer, and coverage IS clean (rank0 owns 0,1; the other owns 2).
    def _other(x):
        return x + jnp.zeros_like(x).at[2].add(0.5)
    g = assemble_global_field(rank0, field0, global_reduce=_other, verify_coverage=False)
    np.testing.assert_array_equal(np.asarray(g), [0.25, 0.75, 0.5])  # halo 999 gone


def test_assemble_global_field_rejects_misaligned_local_field():
    """A local_field whose length ≠ local_cells is folded into the SAME reduction's
    sentinel slot so EVERY rank raises together (no pre-collective rank-local crash)."""
    layout = _layout([True, True, True], [0, 1, 2], n_global=3)
    with pytest.raises(ValueError, match="same length as local_cells"):
        assemble_global_field(layout, jnp.asarray([0.5, 0.5]),  # 2 ≠ 3 cells
                              global_reduce=_serial_sum)


def test_assemble_global_field_rejects_out_of_range_owned_id():
    """An owned global id ≥ nCells_global would clamp in the scatter — caught + synced
    via the second sentinel slot instead."""
    layout = _layout([True, True, True], [0, 1, 5], n_global=4)   # owns id 5 of 4
    with pytest.raises(ValueError, match="out of range"):
        assemble_global_field(layout, jnp.asarray([0.1, 0.2, 0.3]),
                              global_reduce=_serial_sum)


# --- assemble_global_campaign_result: rank-local result -> global result (iter 256).
# Lightweight NamedTuple stand-ins (the helper is duck-typed: it only touches
# final_config / final_field / final_fields + _replace; the real CampaignResult is
# exercised under MPI by the e2e test).
class _Cfg(NamedTuple):
    C_K: object


class _Single(NamedTuple):
    final_config: object
    final_field: object


class _Multi(NamedTuple):
    final_config: object
    final_fields: object


def test_assemble_global_campaign_result_single_replaces_field_and_config():
    """SINGLE: BOTH final_field (read by summarize_campaign) AND
    final_config.<coef> (read by build_campaign_output_dict) become the global field."""
    layout = _layout([True, True, True, True], [2, 0, 3, 1], n_global=4)
    local = jnp.asarray([0.5, 1.5, 2.5, 3.5])         # value for local cell k
    res = _Single(final_config=_Cfg(C_K=local), final_field=local)
    out = assemble_global_campaign_result(
        res, layout, corrected_field="C_K", global_reduce=_serial_sum)
    expected = [1.5, 3.5, 0.5, 2.5]                   # placed by GLOBAL id (see above)
    np.testing.assert_array_equal(np.asarray(out.final_field), expected)
    np.testing.assert_array_equal(np.asarray(out.final_config.C_K), expected)


def test_assemble_global_campaign_result_multi_replaces_every_field():
    """MULTI: every final_fields[key] becomes its global field; config untouched
    (the multi consumers read final_fields, not the config)."""
    layout = _layout([True, True], [1, 0], n_global=2)
    res = _Multi(final_config=_Cfg(C_K=None),
                 final_fields={"clubb_lite_C_K": jnp.asarray([10.0, 20.0]),
                               "clubb_lite_Pr_t": jnp.asarray([0.3, 0.7])})
    out = assemble_global_campaign_result(res, layout, global_reduce=_serial_sum)
    # local cell 0 holds global id 1, local cell 1 holds global id 0 (reversed).
    np.testing.assert_array_equal(np.asarray(out.final_fields["clubb_lite_C_K"]), [20.0, 10.0])
    np.testing.assert_array_equal(np.asarray(out.final_fields["clubb_lite_Pr_t"]), [0.7, 0.3])


def test_assemble_global_campaign_result_single_requires_corrected_field():
    """A single CampaignResult without corrected_field is rejected (the config attr to
    replace is unknown) — only a MultiCampaignResult infers its fields."""
    layout = _layout([True], [0], n_global=1)
    res = _Single(final_config=_Cfg(C_K=jnp.asarray([1.0])), final_field=jnp.asarray([1.0]))
    with pytest.raises(ValueError, match="needs corrected_field"):
        assemble_global_campaign_result(res, layout, global_reduce=_serial_sum)


def test_assemble_global_field_verify_coverage_rejects_a_gap():
    """verify_coverage (default) re-runs the canonical coverage gate, so an UNVERIFIED
    caller still fails loudly on a gap rather than silently returning a 0 at that cell."""
    layout = _layout([True, True, True], [0, 1, 3], n_global=4)   # cell 2 is a gap
    with pytest.raises(ValueError, match="owned by NO rank"):
        assemble_global_field(layout, jnp.asarray([0.1, 0.2, 0.3]),
                              global_reduce=_serial_sum)
    # with the gate disabled the same call no longer raises (caller vouches for cover).
    g = assemble_global_field(layout, jnp.asarray([0.1, 0.2, 0.3]),
                              global_reduce=_serial_sum, verify_coverage=False)
    assert np.asarray(g)[2] == 0.0                   # the un-contributed gap cell


def test_assert_partition_covers_global_detects_cross_rank_overlap():
    """The REALISTIC overlap — two ranks each own the SAME global cell — is detected
    through the collective (emulated by a mock allreduce that adds the other rank's
    owned-count vector), proving the check sees cross-rank double-ownership (iter 98)."""
    rank0 = _layout([True, True, True], [0, 1, 2], n_global=3)   # owns 0,1,2

    def _plus_other_rank(x):                       # a second rank ALSO owns cell 2
        return x + jnp.zeros_like(x).at[jnp.asarray([2])].add(1)

    with pytest.raises(ValueError, match="owned by >1 rank.*\\[2\\]"):
        assert_partition_covers_global(rank0, global_reduce=_plus_other_rank)
