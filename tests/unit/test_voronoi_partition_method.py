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
# High rank counts — the multi-node scaling capability (8 nodes / 32 GPUs+)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mesh_l4():
    # Level-4 SCVT: 10*4^4 + 2 = 2562 cells — big enough to partition to 64
    # ranks (>=40 cells/rank) without empties.  (L1's 42 cells can't feed 32+.)
    return create_voronoi_mesh(subdivision_level=4, lloyd_iterations=1)


class TestHighRankCount:
    """RCB (geometric) partition stays valid + tightly balanced well past 16
    ranks — the decomposition capability behind multi-node icosahedral scaling
    to 8 Derecho nodes (32 GPUs) and beyond.  There is NO rank-count cap or
    power-of-2 requirement; the only real constraint is nCells >= n_ranks."""

    @pytest.mark.parametrize("n_ranks", [16, 32, 64])
    def test_geometric_valid_and_balanced(self, mesh_l4, n_ranks):
        owner = partition_cells_geometric(mesh_l4, n_ranks)
        _assert_valid_owner(owner, mesh_l4.nCells, n_ranks)   # non-empty, covers all
        counts = np.bincount(owner, minlength=n_ranks)
        # RCB proportional bisection keeps cells/rank within one of even.
        assert counts.max() - counts.min() <= 1


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
