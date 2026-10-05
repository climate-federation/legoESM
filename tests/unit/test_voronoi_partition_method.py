"""Tests for Voronoi partition-method selection (``auto`` → METIS-or-RCB).

Covers the capability-aware default added so unstructured (MPAS/Voronoi) MPI
runs prefer graph partitioning (METIS) when ``pymetis`` is available — the MPAS
load-balancing lesson — while falling back to geometric RCB with no dependency.

Verifies: (1) ``resolve_partition_method`` expansion + passthrough, (2) every
dispatch site rejects an unknown method, (3) ``auto`` is byte-identical to
``geometric`` when METIS is absent (no behavior change in the default env), and
(4) the produced owner arrays are valid partitions.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import legoesm.parallel.voronoi_partition as vp
import numpy as np
import pytest
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.parallel.voronoi_partition import (
    hilbert_cell_keys,
    partition_cells_geometric,
    partition_cells_metis,
    partition_cells_sfc,
    partition_voronoi_mesh,
    reorder_voronoi_for_sharding,
    resolve_partition_method,
)

N_RANKS = 4


@pytest.fixture(scope="module")
def mesh():
    # Level-1 SCVT: 10*4^1 + 2 = 42 cells. Tiny + fast.
    return create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)


def _assert_valid_owner(owner, n_cells, n_ranks):
    owner = np.asarray(owner)
    assert owner.shape == (n_cells,)
    assert owner.dtype.kind in "iu"
    assert owner.min() >= 0 and owner.max() < n_ranks
    # Every rank must own at least one cell (no empty partition).
    assert len(np.unique(owner)) == n_ranks


# ---------------------------------------------------------------------------
# resolve_partition_method
# ---------------------------------------------------------------------------

class TestResolveMethod:
    def test_passthrough_known(self):
        assert resolve_partition_method("geometric") == "geometric"
        assert resolve_partition_method("metis") == "metis"

    def test_passthrough_unknown_unchanged(self):
        # Unknown passes through so the caller's dispatch guard can raise.
        assert resolve_partition_method("bogus") == "bogus"

    def test_auto_prefers_metis_when_available(self, monkeypatch):
        monkeypatch.setattr(vp, "_metis_available", lambda: True)
        assert resolve_partition_method("auto") == "metis"

    def test_auto_falls_back_without_metis(self, monkeypatch):
        monkeypatch.setattr(vp, "_metis_available", lambda: False)
        assert resolve_partition_method("auto") == "geometric"

    def test_metis_available_matches_importlib(self):
        import importlib.util
        assert vp._metis_available() == (importlib.util.find_spec("pymetis") is not None)


# ---------------------------------------------------------------------------
# Owner-array correctness
# ---------------------------------------------------------------------------

class TestOwnerArrays:
    def test_geometric_owner_valid(self, mesh):
        owner = partition_cells_geometric(mesh, N_RANKS)
        _assert_valid_owner(owner, mesh.nCells, N_RANKS)

    def test_geometric_deterministic(self, mesh):
        a = partition_cells_geometric(mesh, N_RANKS)
        b = partition_cells_geometric(mesh, N_RANKS)
        assert np.array_equal(a, b)

    def test_metis_owner_valid_if_available(self, mesh):
        pytest.importorskip("pymetis")
        owner = partition_cells_metis(mesh, N_RANKS)
        _assert_valid_owner(owner, mesh.nCells, N_RANKS)

    def test_sfc_owner_valid(self, mesh):
        owner = partition_cells_sfc(mesh, N_RANKS)
        _assert_valid_owner(owner, mesh.nCells, N_RANKS)

    def test_sfc_balanced_contiguous(self, mesh):
        owner = partition_cells_sfc(mesh, N_RANKS)
        counts = np.bincount(owner, minlength=N_RANKS)
        # Contiguous equal chunks -> counts differ by at most 1.
        assert counts.max() - counts.min() <= 1

    def test_sfc_deterministic(self, mesh):
        assert np.array_equal(
            partition_cells_sfc(mesh, N_RANKS), partition_cells_sfc(mesh, N_RANKS)
        )

    def test_sfc_single_rank_all_zero(self, mesh):
        owner = partition_cells_sfc(mesh, 1)
        assert owner.shape == (mesh.nCells,)
        assert np.all(owner == 0)


# ---------------------------------------------------------------------------
# Hilbert space-filling curve
# ---------------------------------------------------------------------------

class TestHilbert:
    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_hilbert_is_bijection(self, order):
        n = 1 << order
        gx, gy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        d = vp._hilbert_xy2d(order, gx.ravel(), gy.ravel())
        # Bijection [0,n)^2 -> [0,n^2): every distance hit exactly once.
        assert np.array_equal(np.sort(d), np.arange(n * n))

    def test_hilbert_unit_step_adjacency(self):
        # Consecutive Hilbert distances must be grid-adjacent (|dx|+|dy| == 1):
        # the locality property the partitioner relies on.
        order = 4
        n = 1 << order
        gx, gy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        d = vp._hilbert_xy2d(order, gx.ravel(), gy.ravel())
        inv = np.empty(n * n, dtype=np.int64)
        inv[d] = np.arange(n * n)  # inv[k] = flat index of the k-th curve point
        xs, ys = gx.ravel()[inv], gy.ravel()[inv]
        steps = np.abs(np.diff(xs)) + np.abs(np.diff(ys))
        assert np.all(steps == 1)

    def test_cell_keys_shape_and_range(self, mesh):
        keys = hilbert_cell_keys(mesh)
        n = 1 << vp._DEFAULT_HILBERT_ORDER
        assert keys.shape == (mesh.nCells,)
        assert keys.min() >= 0 and keys.max() < n * n


# ---------------------------------------------------------------------------
# Dispatch hardening — unknown method raises at EVERY site
# ---------------------------------------------------------------------------

class TestDispatchHardening:
    def test_partition_voronoi_mesh_rejects_unknown(self, mesh):
        with pytest.raises(ValueError, match="Unknown partitioning method"):
            partition_voronoi_mesh(mesh, N_RANKS, 0, method="bogus")

    def test_reorder_rejects_unknown(self, mesh):
        with pytest.raises(ValueError, match="Unknown partitioning method"):
            reorder_voronoi_for_sharding(mesh, 2, method="bogus")

    def test_reorder_single_device_still_rejects_unknown(self, mesh):
        # Validate-at-entry: the n_devices<=1 shortcut must not skip the guard.
        with pytest.raises(ValueError, match="Unknown partitioning method"):
            reorder_voronoi_for_sharding(mesh, 1, method="bogus")

    def test_partition_with_cell_owner_still_rejects_unknown(self, mesh):
        # Validate-at-entry: supplying cell_owner must not skip the guard.
        owner = np.zeros(mesh.nCells, dtype=np.int32)
        with pytest.raises(ValueError, match="Unknown partitioning method"):
            partition_voronoi_mesh(mesh, 1, 0, method="bogus", cell_owner=owner)


# ---------------------------------------------------------------------------
# auto == geometric when METIS absent (no behavior change in default env)
# ---------------------------------------------------------------------------

class TestAutoEquivalence:
    def test_partition_auto_equals_geometric_without_metis(self, mesh, monkeypatch):
        monkeypatch.setattr(vp, "_metis_available", lambda: False)
        auto = partition_voronoi_mesh(mesh, N_RANKS, 0, method="auto")
        geom = partition_voronoi_mesh(mesh, N_RANKS, 0, method="geometric")
        assert auto.n_owned_cells == geom.n_owned_cells
        assert np.array_equal(auto.local_cells, geom.local_cells)
        assert np.array_equal(auto.cell_g2l, geom.cell_g2l)

    def test_reorder_auto_runs_without_metis(self, mesh, monkeypatch):
        monkeypatch.setattr(vp, "_metis_available", lambda: False)
        reordered = reorder_voronoi_for_sharding(mesh, 2, method="auto")
        assert reordered.nCells == mesh.nCells

    def test_reorder_single_device_noop(self, mesh):
        # n_devices <= 1 returns the mesh unchanged (no partition needed).
        assert reorder_voronoi_for_sharding(mesh, 1, method="auto") is mesh


# ---------------------------------------------------------------------------
# Owned-first local indexing (existing contract — locked by a test)
# ---------------------------------------------------------------------------

class TestOwnedFirstLocalIndexing:
    def test_owned_cells_come_first(self, mesh):
        owner = partition_cells_geometric(mesh, N_RANKS)
        part = partition_voronoi_mesh(mesh, N_RANKS, 0, cell_owner=owner)
        no = part.n_owned_cells
        local = np.asarray(part.local_cells)
        # Owned block first (all owned by rank 0), halo block after (none rank 0).
        assert np.all(owner[local[:no]] == 0)
        assert np.all(owner[local[no:]] != 0)

    def test_cell_g2l_roundtrips_local_order(self, mesh):
        owner = partition_cells_geometric(mesh, N_RANKS)
        part = partition_voronoi_mesh(mesh, N_RANKS, 0, cell_owner=owner)
        local = np.asarray(part.local_cells)
        g2l = np.asarray(part.cell_g2l)
        assert np.all(g2l[local] == np.arange(len(local)))


# ---------------------------------------------------------------------------
# SFC wired into the entry points
# ---------------------------------------------------------------------------

class TestSFCWiring:
    def test_partition_voronoi_mesh_sfc(self, mesh):
        part = partition_voronoi_mesh(mesh, N_RANKS, 0, method="sfc")
        assert part.n_owned_cells > 0

    def test_reorder_sfc_is_permutation(self, mesh):
        reordered = reorder_voronoi_for_sharding(mesh, 2, method="sfc")
        assert reordered.nCells == mesh.nCells
        # Reorder is a pure permutation: the cell-latitude multiset is preserved.
        assert np.allclose(
            np.sort(np.asarray(reordered.latCell)),
            np.sort(np.asarray(mesh.latCell)),
        )

    def test_reorder_default_preserves_cells(self, mesh):
        # Default method (auto->geometric here) now SFC-orders within owners.
        reordered = reorder_voronoi_for_sharding(mesh, 2)
        assert reordered.nCells == mesh.nCells


def test_sharding_reorder_auto_is_sfc_not_metis(monkeypatch):
    """``auto`` on the SPMD reorder path must resolve to SFC.

    The global ``resolve_partition_method`` policy prefers METIS when pymetis
    is importable, which optimizes EDGE CUT. The cost that binds this path at
    high device counts is the collective-permute ROUND count (= max degree of
    the post-reorder depth-3+closure comm graph). Measured on the real layout
    those objectives move oppositely -- subdiv-8 @128: sfc 14 rounds, metis 19
    -- so auto must NOT inherit the METIS preference here.

    Asserted behaviourally, by which partitioner the reorder actually calls:
    a check on the resolver alone would prove nothing about this path.
    """
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel import voronoi_partition as vp

    # Non-vacuity: if the global policy would not have picked METIS anyway,
    # there is nothing for this test to distinguish.
    if vp.resolve_partition_method("auto") != "metis":
        pytest.skip("global auto policy is not METIS here (pymetis absent) — "
                    "this test only bites when auto would otherwise pick it")

    used = []
    for name in ("partition_cells_sfc", "partition_cells_metis",
                 "partition_cells_geometric"):
        real = getattr(vp, name)

        def _tap(m, n, _real=real, _name=name):
            used.append(_name)
            return _real(m, n)

        monkeypatch.setattr(vp, name, _tap)

    vp.reorder_voronoi_for_sharding(create_voronoi_mesh(subdivision_level=3), 4)

    assert used == ["partition_cells_sfc"], (
        f"reorder_voronoi_for_sharding(method='auto') used {used}; it must "
        f"use SFC — METIS costs +15 collective-permutes/step at 128 devices "
        f"on subdiv-8/9.")


def test_sharding_reorder_still_honours_an_explicit_method(monkeypatch):
    """The auto override must not hijack an EXPLICIT choice — a deck that
    pins metis/geometric (or a future census that prefers one) still gets it."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel import voronoi_partition as vp

    names = ("partition_cells_sfc", "partition_cells_metis",
             "partition_cells_geometric")
    # Capture the REAL functions once: re-wrapping inside the loop would tap
    # an already-tapped function and double-count.
    originals = {n: getattr(vp, n) for n in names}
    mesh = create_voronoi_mesh(subdivision_level=3)

    for method, expect in (("metis", "partition_cells_metis"),
                           ("geometric", "partition_cells_geometric"),
                           ("sfc", "partition_cells_sfc")):
        used = []
        for name in names:
            def _tap(m, n, _real=originals[name], _name=name):
                used.append(_name)
                return _real(m, n)

            monkeypatch.setattr(vp, name, _tap)
        vp.reorder_voronoi_for_sharding(mesh, 4, method=method)
        assert used == [expect], f"method={method!r} used {used}"


# ---------------------------------------------------------------------------
# Edge/vertex order inside an owner block follows the cells' Hilbert curve
# ---------------------------------------------------------------------------

def _xyz(m, kind):
    return np.stack([np.array(getattr(m, f"{a}{kind}")) for a in "xyz"], 1)


def _runs_are_sorted(owner, key):
    """True iff ``owner`` is non-decreasing and ``key`` is non-decreasing
    inside every constant-owner run."""
    if np.any(np.diff(owner) < 0):
        return False
    same = owner[1:] == owner[:-1]
    return not np.any(np.diff(key)[same] < 0)


def test_reorder_sorts_edges_and_vertices_by_hilbert_within_owner(mesh):
    """A stable sort by owner alone leaves the generator's order inside each
    block; the reorder must place edges and vertices along the same Hilbert
    curve as the cells (key = the key of the min ORIGINAL-index cell, the
    same cell that defines the entity's owner). Fails on the owner-only
    sort: the level-1 mesh's generator order is not Hilbert order."""
    from legoesm.parallel import voronoi_partition as vp
    n_dev = 2
    r = reorder_voronoi_for_sharding(mesh, n_dev, method="sfc")
    owner = vp.partition_cells_sfc(mesh, n_dev)
    key = vp.hilbert_cell_keys(mesh)
    pc = _row_perm(_xyz(r, "Cell"), _xyz(mesh, "Cell"))
    assert _runs_are_sorted(owner[pc], key[pc])

    pe = _row_perm(_xyz(r, "Edge"), _xyz(mesh, "Edge"))
    coe = np.asarray(mesh.cellsOnEdge)
    ec = np.minimum(coe[0], coe[1])[pe]
    assert _runs_are_sorted(owner[ec], key[ec]), "edges not Hilbert-ordered within owner"

    pv = _row_perm(_xyz(r, "Vertex"), _xyz(mesh, "Vertex"))
    cov = np.asarray(mesh.cellsOnVertex)
    cov = np.where(cov >= 0, cov, mesh.nCells)
    mc = cov.min(axis=0)
    # production: an all-sentinel vertex is owned by rank 0 and keyed on the
    # last cell (voronoi_partition.reorder_voronoi_for_sharding)
    v_owner = np.where(mc < mesh.nCells, owner[np.minimum(mc, mesh.nCells - 1)], 0)[pv]
    v_key = key[np.minimum(mc, mesh.nCells - 1)][pv]
    assert _runs_are_sorted(v_owner, v_key), "vertices not Hilbert-ordered within owner"


def test_reorder_owner_edge_order_keeps_generator_order_within_owner(mesh):
    """edge_order="owner" (the ocean MPAS lanes) must reproduce the
    pre-relabel layout: edges and vertices in ascending ORIGINAL index inside
    each owner block, cells still Hilbert-ordered."""
    from legoesm.parallel import voronoi_partition as vp
    r = reorder_voronoi_for_sharding(mesh, 2, method="sfc", edge_order="owner")
    owner = vp.partition_cells_sfc(mesh, 2)
    key = vp.hilbert_cell_keys(mesh)
    pc = _row_perm(_xyz(r, "Cell"), _xyz(mesh, "Cell"))
    assert _runs_are_sorted(owner[pc], key[pc])
    pe = _row_perm(_xyz(r, "Edge"), _xyz(mesh, "Edge"))
    coe = np.asarray(mesh.cellsOnEdge)
    e_owner = owner[np.minimum(coe[0], coe[1])]
    assert np.array_equal(pe, np.argsort(e_owner, kind="stable"))
    pv = _row_perm(_xyz(r, "Vertex"), _xyz(mesh, "Vertex"))
    cov = np.asarray(mesh.cellsOnVertex)
    mc = np.where(cov >= 0, cov, mesh.nCells).min(axis=0)
    v_owner = np.where(mc < mesh.nCells, owner[np.minimum(mc, mesh.nCells - 1)], 0)
    assert np.array_equal(pv, np.argsort(v_owner, kind="stable"))
    # and it differs from the default, or the switch would be inert here
    rh = reorder_voronoi_for_sharding(mesh, 2, method="sfc")
    assert not np.array_equal(np.asarray(rh.xEdge), np.asarray(r.xEdge))


def test_reorder_rejects_unknown_edge_order(mesh):
    for n in (1, 2):
        with pytest.raises(ValueError, match="edge_order"):
            reorder_voronoi_for_sharding(mesh, n, method="sfc", edge_order="bogus")


def _row_perm(a_new, a_old):
    """perm with a_new[i] == a_old[perm[i]], recovered from row-unique
    coordinates (exact, not rounded); asserts a bijection."""
    a_new, a_old = np.asarray(a_new), np.asarray(a_old)
    old = {tuple(row): i for i, row in enumerate(a_old)}
    assert len(old) == len(a_old), "coordinates are not row-unique"
    perm = np.array([old[tuple(row)] for row in a_new])
    assert np.array_equal(np.sort(perm), np.arange(len(a_old))), "not a permutation"
    return perm


def _remap(conn, inv):
    """Connectivity relabel that keeps -1 sentinels."""
    conn = np.asarray(conn)
    return np.where(conn >= 0, inv[np.maximum(conn, 0)], -1)


def test_reorder_commutes_with_trisk_operators(mesh):
    """Relabelling cells, edges and vertices must not change any TRiSK
    operator: div/grad/curl/tangential reconstruction on the reordered mesh
    are BITWISE equal to the original results mapped through the recovered
    permutations (a pure relabel runs the same float ops in the same slot
    order). Also pins every connectivity/weight table as the plain relabel of
    the original, so a table left unpermuted or a neighbour slot reordered
    (sign/weight flip) fails."""
    from legoesm.core.operators_voronoi import (
        curl_vertex,
        divergence_cell,
        gradient_edge,
        tangential_velocity,
    )
    coords = {k: _xyz(mesh, k) for k in ("Cell", "Edge", "Vertex")}
    tables = {k: np.array(getattr(mesh, k)) for k in (
        "cellsOnEdge", "verticesOnEdge", "edgesOnCell", "cellsOnCell",
        "edgesOnVertex", "cellsOnVertex", "verticesOnCell", "edgesOnEdge",
        "weightsOnEdge")}
    r = reorder_voronoi_for_sharding(mesh, 2, method="sfc")
    assert r is not mesh
    for k, v in coords.items():
        assert np.array_equal(_xyz(mesh, k), v), "reorder mutated its input"
    for k, v in tables.items():
        assert np.array_equal(np.asarray(getattr(mesh, k)), v), f"reorder mutated its input {k}"
    pc = _row_perm(_xyz(r, "Cell"), coords["Cell"])
    pe = _row_perm(_xyz(r, "Edge"), coords["Edge"])
    pv = _row_perm(_xyz(r, "Vertex"), coords["Vertex"])
    inv = {}
    for name, p in (("c", pc), ("e", pe), ("v", pv)):
        inv[name] = np.empty_like(p)
        inv[name][p] = np.arange(len(p))
    # Every table is the original's columns permuted by the source entity and
    # values relabelled by the target entity, slot order untouched.
    for tab, src, tgt in (("cellsOnEdge", pe, "c"), ("verticesOnEdge", pe, "v"),
                          ("edgesOnCell", pc, "e"), ("cellsOnCell", pc, "c"),
                          ("edgesOnVertex", pv, "e"), ("cellsOnVertex", pv, "c"),
                          ("verticesOnCell", pc, "v"), ("edgesOnEdge", pe, "e")):
        assert np.array_equal(np.asarray(getattr(r, tab)),
                              _remap(tables[tab][:, src], inv[tgt])), tab
    assert np.array_equal(np.asarray(r.weightsOnEdge), tables["weightsOnEdge"][:, pe])
    rng = np.random.default_rng(0)
    u = rng.standard_normal(mesh.nEdges)
    phi = rng.standard_normal(mesh.nCells)
    for name, f_old, f_new, p in [
        ("div", divergence_cell(u, mesh), divergence_cell(u[pe], r), pc),
        ("grad", gradient_edge(phi, mesh), gradient_edge(phi[pc], r), pe),
        ("curl", curl_vertex(u, mesh), curl_vertex(u[pe], r), pv),
        ("tangential", tangential_velocity(u, mesh), tangential_velocity(u[pe], r), pe),
    ]:
        assert np.array_equal(np.asarray(f_new), np.asarray(f_old)[p]), name


@pytest.mark.parametrize("n_dev", [3, 4, 7])
def test_reorder_block_edge_order_aligns_edge_shards_with_cell_shards(mesh, n_dev):
    """edge_order="block": every real edge in device d's contiguous edge shard
    has its smaller cell in d's cell shard, padding edges reference a cell of
    their own shard, the real edge set is unchanged, and the TRiSK connectivity
    still round-trips (edge -> cells -> edgesOnCell finds the edge)."""
    r = reorder_voronoi_for_sharding(mesh, n_dev, method="sfc", edge_order="block")
    assert r.nCells % n_dev == 0 and r.nEdges % n_dev == 0
    cp, ep = r.nCells // n_dev, r.nEdges // n_dev
    coe = np.asarray(r.cellsOnEdge)
    real = np.asarray(r.dvEdge) > 0
    shard = np.arange(r.nEdges) // ep
    assert np.array_equal(coe.min(axis=0) // cp, shard), "edge outside its shard"
    assert np.all(coe[0, ~real] == coe[1, ~real])
    assert real.sum() == mesh.nEdges
    a = np.sort(np.asarray(r.dvEdge)[real]); b = np.sort(np.asarray(mesh.dvEdge))
    np.testing.assert_array_equal(a, b)
    eoc, neoc = np.asarray(r.edgesOnCell), np.asarray(r.nEdgesOnCell)
    for e in np.flatnonzero(real)[:: max(1, real.sum() // 500)]:
        for c in coe[:, e]:
            assert e in eoc[: neoc[c], c]
    eoe = np.asarray(r.edgesOnEdge)
    assert np.all(real[eoe[eoe >= 0]]), "edgesOnEdge points at a padding edge"


def test_reorder_block_edge_order_shrinks_the_halo():
    """The point of "block": fewer local cells per device than "owner" (on a
    mesh large enough that the halo does not already cover every rank)."""
    from legoesm.parallel.sharded_dynamics import build_voronoi_partition_infra
    big = create_voronoi_mesh(subdivision_level=4, lloyd_iterations=0)
    n_dev = 16
    lc = {}
    for order in ("owner", "block"):
        r = reorder_voronoi_for_sharding(big, n_dev, method="sfc", edge_order=order)
        lc[order] = build_voronoi_partition_infra(r, n_dev, halo_depth=2)[5]
    assert lc["block"] < lc["owner"], lc
