"""Unit tests for the cross-rank global top-k of per-rank worst-column manifests
(:mod:`legoesm.training.distributed_manifest`, iter 87).

Pins the PURE host-side kernels (no MPI): the global top-k selection, the padded
fixed-size candidate arrays for ``allgather``, and the per-rank owned-subset
filter that turns each rank's owned-only ranking into its share of the GLOBAL
``n_worst``.  The MPI ``allgather`` shell is covered by the distributed test.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from legoesm.training.distributed_manifest import (
    _local_candidate_arrays,
    gather_global_worst_columns,
    owned_subset_of_global_top_k,
    select_global_top_k,
)


def _rec(flat_index, score):
    """A minimal ColumnRecord duck-type (the kernels read flat_index + score)."""
    return SimpleNamespace(flat_index=flat_index, combined_score=score)


def test_select_global_top_k_picks_highest_scores():
    # 2 ranks × 2 candidates: ids 0,1 (rank A) and 3,4 (rank B).
    scores = [5.0, 3.0, 9.0, 4.0]
    ids = [0, 1, 3, 4]
    valid = [1, 1, 1, 1]
    assert select_global_top_k(scores, ids, valid, 2) == [3, 0]   # 9.0, then 5.0
    assert select_global_top_k(scores, ids, valid, 3) == [3, 0, 4]
    # n_worst exceeding the candidate count returns all valid, descending.
    assert select_global_top_k(scores, ids, valid, 10) == [3, 0, 4, 1]


def test_select_global_top_k_ignores_invalid_padding():
    # Padded slots (valid=0, score=-inf, id=-1) must never be selected.
    scores = [5.0, -np.inf, 9.0, -np.inf]
    ids = [0, -1, 3, -1]
    valid = [1, 0, 1, 0]
    assert select_global_top_k(scores, ids, valid, 4) == [3, 0]   # only the 2 valid
    # A high score flagged invalid is still excluded (validity wins over score).
    assert select_global_top_k([100.0, 9.0], [7, 3], [0, 1], 2) == [3]


def test_select_global_top_k_nan_score_never_selected():
    """A NaN combined_score (even flagged valid) must NOT be selected — a NaN sorts
    to the END ascending → the FRONT of the reversed order, so without sanitizing it
    would be wrongly picked as 'worst' (Codex iter-87 NaN hazard)."""
    scores = [np.nan, 5.0, 3.0]
    ids = [9, 0, 1]
    valid = [1, 1, 1]               # the NaN slot is flagged valid but non-finite
    assert select_global_top_k(scores, ids, valid, 3) == [0, 1]   # 9 (NaN) excluded


def test_select_global_top_k_dedups_by_global_id():
    # The SAME global id from two ranks (e.g. owned mask forgotten) consumes ONE
    # slot, keeping the higher score — so the result is the worst DISTINCT cells.
    scores = [5.0, 8.0, 3.0]
    ids = [2, 2, 7]           # id 2 appears twice
    valid = [1, 1, 1]
    assert select_global_top_k(scores, ids, valid, 2) == [2, 7]   # not [2, 2]


def test_select_global_top_k_shape_mismatch_raises():
    with pytest.raises(ValueError, match="must share shape"):
        select_global_top_k([1.0, 2.0], [0, 1, 2], [1, 1, 1], 2)


def test_local_candidate_arrays_pads_to_n_worst():
    # Rank owns 1 worst cell but n_worst=3 → 2 padded slots.
    manifest = [_rec(0, 4.0)]
    local_to_global = np.array([10, 11, 12])          # local→global ids
    scores, gids, valid = _local_candidate_arrays(manifest, local_to_global, 3)
    np.testing.assert_array_equal(valid, [1, 0, 0])
    assert scores[0] == 4.0 and not np.isfinite(scores[1])
    assert gids[0] == 10 and gids[1] == -1 and gids[2] == -1


def test_owned_subset_of_global_top_k_splits_across_ranks():
    """The global top-2 spans both ranks; each rank's owned subset is exactly its
    share, in global score order — together they are the global top-k, no overlap."""
    n_worst = 2
    # Rank A owns global ids 0,1,2; its worst owned cells: id0 (5.0), id1 (3.0).
    a_manifest = [_rec(0, 5.0), _rec(1, 3.0)]
    a_l2g = np.array([0, 1, 2])
    # Rank B owns 3,4,5; worst owned: id3 (9.0), id4 (4.0).
    b_manifest = [_rec(0, 9.0), _rec(1, 4.0)]
    b_l2g = np.array([3, 4, 5])
    # The gathered candidate arrays (what allgather would produce): A then B.
    gathered_scores = [5.0, 3.0, 9.0, 4.0]
    gathered_ids = [0, 1, 3, 4]
    gathered_valid = [1, 1, 1, 1]
    # Global top-2 = {id3 (9.0), id0 (5.0)}.
    a_owned = owned_subset_of_global_top_k(
        gathered_scores, gathered_ids, gathered_valid, a_manifest, a_l2g, n_worst)
    b_owned = owned_subset_of_global_top_k(
        gathered_scores, gathered_ids, gathered_valid, b_manifest, b_l2g, n_worst)
    # Rank A owns id0 (the 2nd-worst globally); rank B owns id3 (the worst).
    assert [r.flat_index for r in a_owned] == [0]      # id0
    assert [r.flat_index for r in b_owned] == [0]      # id3
    # No cell is claimed by both ranks; together they ARE the global top-2.
    a_ids = {int(a_l2g[r.flat_index]) for r in a_owned}
    b_ids = {int(b_l2g[r.flat_index]) for r in b_owned}
    assert a_ids.isdisjoint(b_ids)
    assert a_ids | b_ids == {0, 3}


def test_gather_global_worst_columns_single_rank_is_local_top_k(monkeypatch):
    """With one rank, allgather is the identity (leading axis size 1), so the
    reducer returns this rank's own worst-k unchanged — the documented single-rank
    semantics (local manifest already IS the global top-k)."""
    def _fake_allgather(x):
        import jax.numpy as jnp
        return jnp.asarray(np.asarray(x)[None, ...])   # (1,) + shape, like np=1

    monkeypatch.setattr(
        "legoesm.parallel.reductions.allgather_mpi", _fake_allgather)
    manifest = [_rec(2, 7.0), _rec(0, 4.0)]
    l2g = np.array([100, 101, 102])
    out = gather_global_worst_columns(manifest, l2g, 2)
    assert [r.flat_index for r in out] == [2, 0]       # both, worst-first
