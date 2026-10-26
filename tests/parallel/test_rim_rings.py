"""Direct tests for ``_build_rim_rings`` — the inward (rim) complement
of the wide-halo outward ring distances.

The rim of width ``w`` is the set of OWNED cells whose radius-``w``
stencil can see a ghost value: exactly the rows an interior/rim split
must recompute after the halo fill lands. Pinned here against a
brute-force global-graph BFS on a real partitioned mesh.
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
    (_sm, _gc, _ge, _noc, _noe, max_lc, max_le, partitions,
     cell_owner) = _build_voronoi_partition_infra(
        mesh, n_dev, halo_depth=halo_depth)
    return mesh, partitions, cell_owner, max_lc, max_le


def _brute_force_rim(mesh, cell_owner, d, max_width):
    """Distance from each cell owned by ``d`` to the nearest cell NOT
    owned by ``d``, via BFS on the full global graph."""
    coc = np.asarray(mesh.cellsOnCell)
    other = np.where(np.asarray(cell_owner) != d)[0]
    dist = np.full(mesh.nCells, np.iinfo(np.int32).max, dtype=np.int64)
    dist[other] = 0
    frontier = other
    for r in range(1, max_width + 1):
        nb = coc[:, frontier].ravel()
        nb = nb[nb >= 0]
        nb = np.unique(nb)
        nb = nb[dist[nb] > r]
        dist[nb] = r
        frontier = nb
    return dist


def test_cell_rim_matches_global_bruteforce():
    from legoesm.parallel.sharded_dynamics import (
        _WIDE_RING_FAR, _build_rim_rings,
    )

    mesh, partitions, cell_owner, max_lc, max_le = _infra()
    W = 2
    cell_rim, _ = _build_rim_rings(mesh, partitions, max_lc, max_le, W)

    for d, part in enumerate(partitions):
        ref = _brute_force_rim(mesh, cell_owner, d, W)
        n_owned = part.n_owned_cells
        got = cell_rim[d]
        for i in range(n_owned):
            g = int(part.local_cells[i])
            want = ref[g] if ref[g] <= W else _WIDE_RING_FAR
            assert got[i] == want, (
                f"dev {d} owned cell {i} (global {g}): rim {got[i]} "
                f"!= brute-force {want}")
        # halo rows are the seed
        assert (got[n_owned:part.n_local_cells] == 0).all()
        # padding stays FAR
        assert (got[part.n_local_cells:] == _WIDE_RING_FAR).all()


def test_edge_rim_is_min_of_cell_rims_and_monotone_width():
    from legoesm.parallel.sharded_dynamics import (
        _WIDE_RING_FAR, _build_rim_rings,
    )

    mesh, partitions, _cell_owner, max_lc, max_le = _infra()
    c1, e1 = _build_rim_rings(mesh, partitions, max_lc, max_le, 1)
    c2, e2 = _build_rim_rings(mesh, partitions, max_lc, max_le, 2)

    # widening never removes a rim label, only adds
    lab1 = (c1 != _WIDE_RING_FAR)
    lab2 = (c2 != _WIDE_RING_FAR)
    assert (lab2 | ~lab1).all()
    assert (c2[lab1] == c1[lab1]).all()

    # edge rim: spot-check the min-of-adjacent-cells contract
    coe = np.asarray(mesh.cellsOnEdge)
    d = 0
    part = partitions[d]
    g2l = part.cell_g2l
    for j in range(0, part.n_owned_edges, 7):
        g_edge = int(part.local_edges[j])
        cells = coe[:, g_edge]
        vals = []
        for c in cells:
            if c >= 0 and g2l[c] >= 0:
                vals.append(int(c2[d, g2l[c]]))
            else:
                vals.append(int(_WIDE_RING_FAR))
        assert e2[d, j] == min(vals)


def test_rim_and_interior_partition_the_owned_block():
    """Every owned cell is exactly one of: rim (1 <= dist <= w) or
    interior (FAR). No owned cell may carry the seed label 0 — that
    would mean an owned row was treated as a ghost.

    (The rim FRACTION is a scale property: on this toy mesh a width-2
    rim covers most of a 107-cell patch; at s9@64 the same band is a
    few percent. Economics are receipted on the cluster, not here.)"""
    from legoesm.parallel.sharded_dynamics import (
        _WIDE_RING_FAR, _build_rim_rings,
    )

    mesh, partitions, _cell_owner, max_lc, max_le = _infra()
    W = 2
    cell_rim, _ = _build_rim_rings(mesh, partitions, max_lc, max_le, W)
    for d, part in enumerate(partitions):
        owned = cell_rim[d, :part.n_owned_cells]
        assert ((owned == _WIDE_RING_FAR)
                | ((owned >= 1) & (owned <= W))).all()
        assert (owned != 0).all()
