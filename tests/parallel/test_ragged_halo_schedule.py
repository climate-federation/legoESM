"""Direct tests for the ragged (grouped-P2P) halo schedule.

The EXECUTOR (``_ragged_halo_fill``) is GPU-only — XLA:CPU has no
ragged-all-to-all thunk — so its parity gate lives in the GPU A/B job
(bench_mpas_spmd_scaling --parity-gate with LEGOESM_MPAS_RAGGED_HALO=1).
What IS CPU-testable, and what this file pins down, is the SCHEDULE:

1. the ragged metadata is exactly a re-grouping of the SAME directed
   send/recv maps the coloured ppermute schedule moves (shared helper
   ``_build_halo_send_maps``) — row-for-row, per (src, dst) pair;
2. the ragged_all_to_all size contract (``send_sz == recv_sz.T``) and
   offset bounds hold on a real partitioned mesh;
3. staging padding scatters only to the garbage slot.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")


def _infra(n_dev=6, subdivision=3, halo_depth=2):
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.sharded_dynamics import (
        _build_voronoi_partition_infra,
    )

    mesh = create_voronoi_mesh(subdivision_level=subdivision)
    if mesh.nCells % n_dev or mesh.nEdges % n_dev:
        pytest.skip("mesh not divisible")
    cells_per = mesh.nCells // n_dev
    edges_per = mesh.nEdges // n_dev
    (_sm, _gc, _ge, _noc, _noe, max_lc, max_le, partitions,
     cell_owner) = _build_voronoi_partition_infra(
        mesh, n_dev, halo_depth=halo_depth)
    return partitions, cell_owner, n_dev, cells_per, edges_per, max_lc, \
        max_le


def test_ragged_schedule_regroups_the_same_rows():
    from legoesm.parallel.sharded_dynamics import (
        _build_halo_send_maps,
        _build_ragged_halo_schedule,
    )

    (partitions, cell_owner, n_dev, cells_per, edges_per, max_lc,
     max_le) = _infra()
    (_pairs, c_send, c_recv, e_send, e_recv) = _build_halo_send_maps(
        partitions, cell_owner, n_dev, cells_per, edges_per)
    sched = _build_ragged_halo_schedule(
        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc,
        max_le)

    for ent, send_map, recv_map, garbage in (
            ("cells", c_send, c_recv, max_lc),
            ("edges", e_send, e_recv, max_le)):
        m = sched[ent]
        send_sz = m["send_sz"]
        # contract
        assert (send_sz == m["recv_sz"].T).all()
        # every pair's block carries exactly the map's rows, in order
        for (src, dst), rows in send_map.items():
            n = len(rows)
            assert send_sz[src, dst] == n
            off = int(m["in_off"][src, dst])
            assert list(m["send_idx"][src, off:off + n]) == rows
            # the sender-chosen receiver offset lands the block where
            # the receiver's recv_pos scatters the matching positions
            oo = int(m["out_off"][src, dst])
            assert list(m["recv_pos"][dst, oo:oo + n]) == recv_map[(dst,
                                                                    src)]
        # offsets in bounds
        tot_send = send_sz.sum(axis=1)
        tot_recv = m["recv_sz"].sum(axis=1)
        assert (tot_send <= m["s_max"]).all()
        assert (tot_recv <= m["r_max"]).all()
        # padding rows scatter only to the garbage slot
        for d in range(n_dev):
            pad = m["recv_pos"][d, int(tot_recv[d]):]
            assert (pad == garbage).all(), ent


def test_ragged_and_ppermute_transfer_identical_row_sets():
    """The two schedules must be re-groupings of one another: the set of
    (src, dst, owned_row -> local_pos) transfers extracted from each is
    identical."""
    from legoesm.parallel.sharded_dynamics import (
        _build_ppermute_schedule,
        _build_ragged_halo_schedule,
    )

    (partitions, cell_owner, n_dev, cells_per, edges_per, max_lc,
     max_le) = _infra()
    pp = _build_ppermute_schedule(
        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc,
        max_le)
    rg = _build_ragged_halo_schedule(
        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc,
        max_le)

    # ppermute transfers: per round, per (u -> v) in the perm, device u
    # sends send_cell_idx[u] rows (padded; padding targets max_lc on the
    # receiver) — collect the non-garbage ones.
    pp_set = set()
    for r, perm in enumerate(pp["ppermute_perms"]):
        sc = np.asarray(pp["send_cell_idx"][r])
        rc = np.asarray(pp["recv_cell_pos"][r])
        for src, dst in perm:
            for j in range(sc.shape[1]):
                pos = int(rc[dst, j])
                if pos != max_lc:
                    pp_set.add((src, dst, int(sc[src, j]), pos))
    rg_set = set()
    m = rg["cells"]
    tot_recv = m["recv_sz"].sum(axis=1)
    for src in range(n_dev):
        for dst in range(n_dev):
            n = int(m["send_sz"][src, dst])
            if n == 0:
                continue
            io = int(m["in_off"][src, dst])
            oo = int(m["out_off"][src, dst])
            for j in range(n):
                rg_set.add((src, dst, int(m["send_idx"][src, io + j]),
                            int(m["recv_pos"][dst, oo + j])))
    assert pp_set == rg_set
    assert len(rg_set) > 0


def test_offsets_are_int32():
    """The GPU ragged-all-to-all custom call takes int32 offset/size
    vectors; int64 would fail at lowering on the cluster, not here."""
    from legoesm.parallel.sharded_dynamics import (
        _build_ragged_halo_schedule,
    )

    (partitions, cell_owner, n_dev, cells_per, edges_per, max_lc,
     max_le) = _infra()
    sched = _build_ragged_halo_schedule(
        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc,
        max_le)
    for ent in ("cells", "edges"):
        for key in ("in_off", "send_sz", "out_off", "recv_sz"):
            assert sched[ent][key].dtype == np.int32, (ent, key)


def test_resolve_ragged_halo_dispatch():
    from legoesm.parallel.sharded_dynamics import (
        _RAGGED_AUTO_MAX_NDEV,
        _resolve_ragged_halo,
    )

    assert _resolve_ragged_halo("0", 4) is False
    assert _resolve_ragged_halo("", 4) is False
    assert _resolve_ragged_halo("1", 128) is True
    assert _resolve_ragged_halo("auto", _RAGGED_AUTO_MAX_NDEV) is True
    assert _resolve_ragged_halo("auto", _RAGGED_AUTO_MAX_NDEV + 1) is False
    with pytest.raises(ValueError, match="RAGGED_HALO"):
        _resolve_ragged_halo("yes", 4)
