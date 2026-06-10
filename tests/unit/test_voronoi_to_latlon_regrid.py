"""Unit tests for the SCVT/Voronoi cell → lat-lon CMIP regrid (G9).

Covers ``compute_voronoi_to_latlon_weights`` + ``apply_voronoi_to_latlon``
/ ``apply_voronoi_to_latlon_3d`` in ``legoesm.grids.regridding`` and that
the ``DiagnosticCollector`` CMIP setup no longer rejects an MPAS grid.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.grids.regridding import (
    apply_voronoi_to_latlon,
    apply_voronoi_to_latlon_3d,
    compute_voronoi_to_latlon_weights,
)
from legoesm.grids.voronoi import create_voronoi_mesh


@pytest.fixture(scope="module")
def mesh():
    # Level-3 SCVT mesh = 642 cells; cheap and enough to exercise the regrid.
    return create_voronoi_mesh(3, lloyd_iterations=10)


def test_weights_shapes_and_normalization(mesh):
    w = compute_voronoi_to_latlon_weights(
        np.asarray(mesh.latCell), np.asarray(mesh.lonCell),
        n_lon=72, n_lat=37, k=3,
    )
    n_target = 37 * 72
    assert w.idx.shape == (n_target, 3)
    assert w.w.shape == (n_target, 3)
    # IDW rows sum to 1.
    np.testing.assert_allclose(w.w.sum(axis=1), 1.0, atol=1e-12)
    # Indices are valid cell ids.
    assert w.idx.min() >= 0 and w.idx.max() < mesh.latCell.shape[0]


def test_constant_field_is_preserved(mesh):
    """A constant cell field must regrid to the same constant everywhere
    (partition of unity)."""
    w = compute_voronoi_to_latlon_weights(
        np.asarray(mesh.latCell), np.asarray(mesh.lonCell),
        n_lon=72, n_lat=37,
    )
    field = np.full(mesh.latCell.shape[0], 287.5)
    out = apply_voronoi_to_latlon(field, w)
    assert out.shape == (37, 72)
    np.testing.assert_allclose(out, 287.5, atol=1e-10)


def test_smooth_field_tracks_target(mesh):
    """A smooth analytic field (cos²lat) regrids close to its value at the
    target lat-lon points — IDW of a smooth field is near-exact."""
    w = compute_voronoi_to_latlon_weights(
        np.asarray(mesh.latCell), np.asarray(mesh.lonCell),
        n_lon=144, n_lat=73,
    )
    latc = np.asarray(mesh.latCell)
    field = np.cos(latc) ** 2          # smooth, [0, 1]
    out = apply_voronoi_to_latlon(field, w)
    # Expected at target latitudes (broadcast over lon).
    lat_t = np.deg2rad(w.lat_cent)[:, None]
    expected = np.broadcast_to(np.cos(lat_t) ** 2, out.shape)
    # Coarse 642-cell mesh → loose tolerance, but must track the structure.
    assert np.max(np.abs(out - expected)) < 0.05


def test_3d_field(mesh):
    w = compute_voronoi_to_latlon_weights(
        np.asarray(mesh.latCell), np.asarray(mesh.lonCell),
        n_lon=72, n_lat=37,
    )
    nlev = 5
    field = np.ones((mesh.latCell.shape[0], nlev)) * np.arange(1, nlev + 1)
    out = apply_voronoi_to_latlon_3d(field, w)
    assert out.shape == (37, 72, nlev)
    for lvl in range(nlev):
        np.testing.assert_allclose(out[..., lvl], lvl + 1, atol=1e-10)


def test_diagnostics_cmip_setup_accepts_mpas(mesh):
    """set_cmip_grid_info no longer raises for an MPAS grid; it builds the
    Voronoi regrid weights."""
    from legoesm.driver.diagnostics import DiagnosticCollector

    dc = DiagnosticCollector(
        nlev=5, sigma_full=np.linspace(0.1, 0.9, 5),
        dsigma=np.full(5, 0.2), experiment_id="amip",
        monthly_means=True, cmip_output=True, n_days=1,
        output_dir=None,
    )
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    assert dc._voronoi_regrid_weights is not None
    # A constant cell field regrids through the collector's 2-D path.
    field = np.full(mesh.latCell.shape[0], 300.0)
    out = dc._regrid_to_latlon_2d(field)
    assert out is not None
    np.testing.assert_allclose(out, 300.0, atol=1e-9)
