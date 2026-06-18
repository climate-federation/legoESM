"""Regression test: the Voronoi halo schedule must be SYMMETRIC.

A blocking ``sendrecv`` halo deadlocks if rank A posts a message to B that B does
not post back.  The cell/edge/vertex SEND schedule must therefore exactly mirror
the RECV schedule across every pair: ``send(A->B) == recv(B<-A)`` for cells,
edges, and vertices, and the neighbour lists must be mutual (A lists B iff B
lists A).

This caught the np>=64 multi-node dycore deadlock: the edge/vertex send schedule
was built from cell-recv neighbours only, so a rank sharing only an EDGE/VERTEX
boundary (beyond the cell halo) never received the owned edge it needed and hung.
The check is pure host-side schedule math (no MPI) — it builds every rank's
schedule serially from the global mesh and cross-checks the directed counts.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.parallel.voronoi_partition import (
    build_batched_halo_schedule,
    partition_voronoi_mesh,
)


def _build_schedules(mesh, n_ranks):
    """All ranks' (cell, edge) directed counts + neighbour lists, host-side."""
    out = {}
    for r in range(n_ranks):
        part = partition_voronoi_mesh(mesh, n_ranks, r, method="geometric",
                                      halo_depth=2)
        bc = build_batched_halo_schedule(part.cell_comm, part.edge_comm)
        out[r] = dict(
            nbrs=list(bc.neighbor_ranks),
            cs=list(bc.cell_send_counts), cr=list(bc.cell_recv_counts),
            es=list(bc.edge_send_counts), er=list(bc.edge_recv_counts),
        )
    return out


def _symmetry_problems(sched, n_ranks):
    problems = []
    for A in range(n_ranks):
        sA = sched[A]
        for i, B in enumerate(sA["nbrs"]):
            sB = sched[B]
            if A not in sB["nbrs"]:
                problems.append(f"{A} lists {B} but {B} does not list {A}")
                continue
            j = sB["nbrs"].index(A)
            checks = (
                ("cell s/r", sA["cs"][i], sB["cr"][j]),
                ("cell r/s", sA["cr"][i], sB["cs"][j]),
                ("edge s/r", sA["es"][i], sB["er"][j]),
                ("edge r/s", sA["er"][i], sB["es"][j]),
            )
            for name, a, b in checks:
                if a != b:
                    problems.append(f"{name} A={A} B={B}: {a} != {b}")
    return problems


# (subdivision_level, n_ranks): include a fine-partition case (small cells/rank)
# that exercised the edge/vertex-beyond-cell-halo asymmetry, plus a non-power-of-2
# rank count.
@pytest.mark.parametrize("level,n_ranks", [(3, 16), (3, 32), (4, 48), (4, 64)])
def test_halo_schedule_is_symmetric(level, n_ranks):
    mesh = create_voronoi_mesh(subdivision_level=level)
    sched = _build_schedules(mesh, n_ranks)
    problems = _symmetry_problems(sched, n_ranks)
    assert not problems, (
        f"level={level} np={n_ranks}: {len(problems)} schedule asymmetries "
        f"(first few: {problems[:5]})"
    )
