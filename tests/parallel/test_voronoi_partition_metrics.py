"""Unit tests for Voronoi partition-quality metrics (roadmap item 6).

These metrics (edge-cut / owned-halo-ratio / cells-per-rank / message count)
are recorded in the MPAS scaling-benchmark metadata so partition quality and
halo-communication overhead are visible per run.  The per-rank extractor is
pure (no collectives) and tested here against synthetic layouts; the cross-rank
aggregation is tested with the MPI reductions stubbed to identity.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from legoesm.parallel import voronoi_mpi as vm


def _stub_layout(owned, neighbor_ranks, cell_recv, cell_send, msgs):
    bc = SimpleNamespace(
        neighbor_ranks=tuple(neighbor_ranks),
        cell_recv_counts=tuple(cell_recv),
        cell_send_counts=tuple(cell_send),
        messages_per_exchange=lambda n_dtype_groups=1: msgs,
    )
    return SimpleNamespace(
        owned_mask_cells=np.asarray(owned, dtype=bool), batched_comm=bc)


def test_metrics_owned_halo_and_schedule():
    owned = [True] * 6 + [False] * 4          # 6 owned, 4 ghost
    layout = _stub_layout(owned, [1, 2], [3, 1], [2, 2], 2)
    m = vm.voronoi_partition_metrics(layout)
    assert m["n_owned_cells"] == 6
    assert m["n_halo_cells"] == 4
    assert m["owned_halo_ratio"] == pytest.approx(4 / 6)
    assert m["n_neighbor_ranks"] == 2
    assert m["halo_recv_cells"] == 4          # 3 + 1
    assert m["owned_send_cells"] == 4         # 2 + 2
    assert m["messages_per_exchange"] == 2


def test_metrics_no_batched_schedule_falls_back_to_ghost_count():
    owned = [True] * 5 + [False] * 2
    layout = SimpleNamespace(
        owned_mask_cells=np.asarray(owned, dtype=bool), batched_comm=None)
    m = vm.voronoi_partition_metrics(layout)
    assert m["n_neighbor_ranks"] == 0
    assert m["halo_recv_cells"] == 2          # ghost-count fallback
    assert m["owned_send_cells"] == 0
    assert m["messages_per_exchange"] == 0


def test_metrics_owns_nothing_gives_inf_ratio():
    layout = _stub_layout([False] * 3, [], [], [], 0)
    m = vm.voronoi_partition_metrics(layout)
    assert m["n_owned_cells"] == 0
    assert m["n_halo_cells"] == 3
    assert m["owned_halo_ratio"] == float("inf")


def test_reduce_adds_cross_rank_aggregates(monkeypatch):
    # Stub the (non-differentiable) MPI reductions to identity so the test
    # needs no live MPI stack; on 1 rank they ARE identity anyway.
    monkeypatch.setattr(vm, "global_min_mpi", lambda x: x)
    monkeypatch.setattr(vm, "global_max_mpi", lambda x: x)
    monkeypatch.setattr(vm, "global_sum_mpi", lambda x: x)
    local = {
        "n_owned_cells": 6, "n_halo_cells": 4, "halo_recv_cells": 4,
        "n_neighbor_ranks": 2, "owned_halo_ratio": 4 / 6,
        "messages_per_exchange": 2, "owned_send_cells": 4,
    }
    out = vm.reduce_partition_metrics(local)
    assert out["cells_per_rank_min"] == 6
    assert out["cells_per_rank_max"] == 6
    assert out["edge_cut_total"] == 4
    assert out["max_neighbor_ranks"] == 2
    # Per-rank keys are preserved (dict is extended, not replaced).
    assert out["n_owned_cells"] == 6 and out["owned_send_cells"] == 4
