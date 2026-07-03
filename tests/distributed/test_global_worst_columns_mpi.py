"""Distributed (MPI) test for the cross-rank global top-k worst-column reducer
(:func:`legoesm.training.distributed_manifest.gather_global_worst_columns`, iter 87).

Run: ``mpirun -np 2 .venv/bin/python -m pytest tests/distributed/test_global_worst_columns_mpi.py``

Each rank holds its OWNED-only worst-column manifest (rank-local indices) + a
local→global cell-id map; the reducer ``allgather``-s the per-rank top-``n_worst``
candidates and returns the records THIS rank owns that are among the GLOBAL
``n_worst``.  Verifies: (1) each rank gets exactly its share of the global top-k,
(2) no cell is claimed by two ranks (the owning rank wins), (3) the union across
ranks is precisely the global worst ``n_worst`` (not ``R × n_worst``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mpi4py")
pytest.importorskip("mpi4jax")
from legoesm.training.distributed_manifest import (  # noqa: E402
    gather_global_worst_columns,
)
from mpi4py import MPI  # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NPROC = COMM.Get_size()


def _rec(flat_index, score):
    return SimpleNamespace(flat_index=flat_index, combined_score=score)


@pytest.mark.skipif(NPROC < 2, reason="needs >=2 ranks to test cross-rank top-k")
def test_gather_global_worst_columns_splits_global_top_k_across_ranks():
    """Rank r owns global ids [r*100, r*100+1] with worst-cell scores [10-r, 1.0].
    The two globally-worst cells are rank 0's id 0 (score 10) and rank 1's id 100
    (score 9); every other rank owns none of the global top-2."""
    n_worst = 2
    # This rank's owned worst columns (sorted worst-first), rank-local flat indices.
    local_manifest = [_rec(0, 10.0 - RANK), _rec(1, 1.0)]
    # local flat index -> GLOBAL cell id (globally unique per rank; owned cells are
    # disjoint across ranks).  Slot 2 is an unused owned cell (no record this round).
    local_to_global = [RANK * 100 + 0, RANK * 100 + 1, RANK * 100 + 2]

    owned = gather_global_worst_columns(local_manifest, local_to_global, n_worst)
    owned_global_ids = [local_to_global[r.flat_index] for r in owned]

    if RANK == 0:
        assert owned_global_ids == [0]          # owns the global worst (score 10)
    elif RANK == 1:
        assert owned_global_ids == [100]        # owns the 2nd worst (score 9)
    else:
        assert owned_global_ids == []           # owns none of the global top-2

    # Cross-rank invariant: the UNION of every rank's owned subset is EXACTLY the
    # global worst n_worst (no double-count, not R*n_worst).
    all_ids = COMM.allgather(owned_global_ids)
    flat = [gid for sub in all_ids for gid in sub]
    assert sorted(flat) == [0, 100]             # the 2 globally-worst, once each
    assert len(flat) == len(set(flat))          # disjoint across ranks
    COMM.Barrier()
