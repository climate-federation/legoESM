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

import numpy as np
import pytest

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.parallel.voronoi_partition import (
    partition_cells_geometric,
    partition_cells_metis,
    partition_voronoi_mesh,
    reorder_voronoi_for_sharding,
    resolve_partition_method,
)
import legoesm.parallel.voronoi_partition as vp

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
