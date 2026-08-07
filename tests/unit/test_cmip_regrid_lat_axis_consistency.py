"""The CMOR output grid must be ONE grid: writer axis == regridder axis.

Regression gate for the MPAS ``rsdt`` deficit.  ``compute_voronoi_to_latlon_
weights`` / ``compute_cubedsphere_to_latlon_weights`` defaulted to a
**pole-inclusive** target ``linspace(-90, 90, n_lat)`` (spacing
``180/(n_lat-1)``) while the CMOR writer labels its files with **cell
centres** ``-90+dlat/2 .. 90-dlat/2`` (spacing ``180/n_lat``).  Every row of
every regridded MPAS/cubed-sphere CMOR field was therefore displaced
poleward by ``lat/(n_lat-1)`` degrees — up to 2.5 deg at n_lat=36.

For a field that falls off toward the poles this reads systematically LOW:
the 1979-1980 MPAS AMIP ``rsdt`` came out at 337.27 W/m^2 against a
prescribed-astronomy truth of 340.41 W/m^2 (-3.15 W/m^2, -0.94%), and
carried ~1.8 W/m^2 of the reported TOA imbalance.

The structured (lat-lon / Gaussian) lane was always correct, so the gate
covers all THREE lanes: any one of them drifting off the writer's axis is
the same defect.
"""
from __future__ import annotations

import numpy as np
import pytest
from legoesm.driver.diagnostics import (
    DiagnosticCollector,
    cmip_target_latlon,
)
from legoesm.grids.regridding import (
    compute_voronoi_to_latlon_weights,
    get_cubedsphere_to_latlon_weights,
)
from legoesm.grids.voronoi import create_voronoi_mesh

from legoesm import constants

# Deliberately not 180/181: n_lat=36 is the production AMIP CMIP grid
# (cmip_resolution_deg=5.0) and the case where the two conventions differ
# most (2.5 deg at the poles).
NLAT, NLON = 36, 72


def _collector(nlat: int = NLAT, nlon: int = NLON) -> DiagnosticCollector:
    return DiagnosticCollector(
        nlev=5, sigma_full=np.linspace(0.1, 0.9, 5),
        dsigma=np.full(5, 0.2), experiment_id="amip",
        monthly_means=True, cmip_output=True, n_days=1,
        output_dir=None, cmip_resolution_deg=180.0 / nlat,
    )


class _LatLonGrid:
    """Minimal structured-grid stand-in (radians, S->N / [0, 2pi))."""

    def __init__(self, nlat: int, nlon: int):
        self.lat = np.deg2rad(
            np.linspace(-90.0 + 90.0 / nlat, 90.0 - 90.0 / nlat, nlat))
        self.lon = np.deg2rad(
            np.linspace(180.0 / nlon, 360.0 - 180.0 / nlon, nlon))


class _CubeGrid:
    def __init__(self, n: int):
        self.n = n


@pytest.fixture(scope="module")
def mesh():
    # Level-3 SCVT mesh = 642 cells; cheap and enough to exercise the regrid.
    return create_voronoi_mesh(3, lloyd_iterations=10)


# --------------------------------------------------------------------------
# The definition itself
# --------------------------------------------------------------------------

def test_cmip_target_latlon_is_cell_centres():
    lat, lon = cmip_target_latlon(NLAT, NLON)
    assert lat.shape == (NLAT,) and lon.shape == (NLON,)
    np.testing.assert_allclose(lat[0], -87.5, atol=1e-12)
    np.testing.assert_allclose(lat[-1], 87.5, atol=1e-12)
    np.testing.assert_allclose(np.diff(lat), 5.0, atol=1e-12)
    np.testing.assert_allclose(lon[0], 2.5, atol=1e-12)
    # Cell centres never sit ON the poles.
    assert np.abs(lat).max() < 90.0


# --------------------------------------------------------------------------
# THE GATE: all three lanes sample the axis the writer labels
# --------------------------------------------------------------------------

def test_voronoi_lane_lat_axis_matches_writer(mesh):
    dc = _collector()
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    assert dc._voronoi_regrid_weights is not None
    writer_lat, _ = dc._cmip_target_latlon()
    np.testing.assert_array_equal(
        dc._voronoi_regrid_weights.lat_cent, writer_lat)


def test_cubedsphere_lane_lat_axis_matches_writer():
    dc = _collector()
    dc.set_cmip_grid_info(
        grid_type="cubed_sphere", grid=_CubeGrid(8), start_year=1979)
    assert dc._cs_regrid_weights is not None
    writer_lat, _ = dc._cmip_target_latlon()
    np.testing.assert_array_equal(
        dc._cs_regrid_weights.lat_cent, writer_lat)


def test_structured_lane_lat_axis_matches_writer():
    # Native shape != CMIP target so the regrid path is actually built.
    dc = _collector()
    dc.set_cmip_grid_info(
        grid_type="latlon", grid=_LatLonGrid(48, 96), start_year=1979)
    assert dc._structured_regrid is not None
    writer_lat, _ = dc._cmip_target_latlon()
    np.testing.assert_array_equal(
        dc._structured_regrid.lat_cent, writer_lat)


# --------------------------------------------------------------------------
# The physical consequence: the value written under label phi IS the field
# at phi.  This is what the 3.15 W/m^2 deficit actually was.
# --------------------------------------------------------------------------

def test_voronoi_regrid_lands_field_at_labelled_latitude(mesh):
    """Regrid sin(lat), a smooth field with no zonal structure.

    Row j of the output must equal sin(label_lat[j]).  Under the
    pole-inclusive bug row j held sin(-90 + j*180/35) instead.

    Stated as a DISCRIMINATION between the two candidate axes rather than a
    bare tolerance: IDW(k=3) on a 642-cell mesh smooths by 0.0056 while the
    axes differ by 0.0158 in sin, so a loose atol passes under BOTH
    hypotheses and proves nothing (it did, on the first draft of this test).
    Measured: 0.0056 against the labelled axis, 0.0196 against the
    pole-inclusive one.
    """
    dc = _collector()
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    field = np.sin(np.asarray(mesh.latCell))
    out = dc._regrid_to_latlon_2d(field)
    assert out is not None and out.shape == (NLAT, NLON)
    zonal = out.mean(axis=1)

    writer_lat, _ = dc._cmip_target_latlon()
    pole_inclusive_lat = np.linspace(-90.0, 90.0, NLAT)
    err_label = np.abs(zonal - np.sin(np.deg2rad(writer_lat))).max()
    err_pole = np.abs(zonal - np.sin(np.deg2rad(pole_inclusive_lat))).max()

    assert err_label < 0.4 * err_pole, (
        f"regridded field sits closer to the pole-inclusive axis "
        f"(err {err_pole:.5f}) than to the CMOR label axis "
        f"(err {err_label:.5f}) — the rows are displaced.")
    np.testing.assert_allclose(
        zonal, np.sin(np.deg2rad(writer_lat)), atol=0.01)


def test_structured_regrid_lands_field_at_labelled_latitude():
    dc = _collector()
    src = _LatLonGrid(48, 96)
    dc.set_cmip_grid_info(grid_type="latlon", grid=src, start_year=1979)
    field = np.broadcast_to(np.sin(src.lat)[:, None], (48, 96)).copy()
    out = dc._regrid_to_latlon_2d(field)
    assert out is not None
    writer_lat, _ = dc._cmip_target_latlon()
    np.testing.assert_allclose(
        out.mean(axis=1), np.sin(np.deg2rad(writer_lat)), atol=2e-3)


# --------------------------------------------------------------------------
# areacella must be the cell-edge bands OF THE LABELLED CENTRES
# --------------------------------------------------------------------------

def test_areacella_bands_match_the_labelled_centres():
    """Cell areas and data must describe the same grid.

    ``areacella`` was always right; it was the DATA that sat elsewhere.
    Pinning it here means a future "fix" that moves the areas instead of the
    data goes red.
    """
    lat, _ = cmip_target_latlon(NLAT, NLON)
    edges = np.deg2rad(np.linspace(-90.0, 90.0, NLAT + 1))
    band = (np.sin(edges[1:]) - np.sin(edges[:-1]))
    area_row = constants.R_earth ** 2 * band * (2.0 * np.pi / NLON)

    # Each centre lies at the midpoint (in latitude) of its own band.
    np.testing.assert_allclose(
        np.rad2deg(0.5 * (edges[1:] + edges[:-1])), lat, atol=1e-12)
    # Total area closes on the sphere.
    np.testing.assert_allclose(
        area_row.sum() * NLON, 4.0 * np.pi * constants.R_earth ** 2, rtol=1e-12)


# --------------------------------------------------------------------------
# No-op guarantee for the pole-inclusive callers (plot scripts, matrix
# runners) that pass n_lat=181 and label with the SAME linspace.
# --------------------------------------------------------------------------

def test_pole_inclusive_default_unchanged_for_voronoi(mesh):
    w = compute_voronoi_to_latlon_weights(
        np.asarray(mesh.latCell), np.asarray(mesh.lonCell),
        n_lon=360, n_lat=181)
    np.testing.assert_array_equal(w.lat_cent, np.linspace(-90.0, 90.0, 181))


def test_pole_inclusive_default_unchanged_for_cubedsphere():
    w = get_cubedsphere_to_latlon_weights(6, n_lon=360, n_lat=181)
    np.testing.assert_array_equal(w.lat_cent, np.linspace(-90.0, 90.0, 181))


def test_cubedsphere_cache_keys_on_lat_cent():
    """Two axes, same (n, n_lat, n_lon) — the cache must not conflate them."""
    custom, _ = cmip_target_latlon(181, 360)
    a = get_cubedsphere_to_latlon_weights(6, n_lon=360, n_lat=181)
    b = get_cubedsphere_to_latlon_weights(
        6, n_lon=360, n_lat=181, lat_cent=custom)
    assert a is not b
    np.testing.assert_array_equal(b.lat_cent, custom)
    np.testing.assert_array_equal(a.lat_cent, np.linspace(-90.0, 90.0, 181))


# --------------------------------------------------------------------------
# Fail loud on a mismatched axis rather than regridding onto a grid the
# caller did not ask for.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    np.linspace(-87.5, 87.5, 35),          # wrong length
    np.linspace(87.5, -87.5, 36),          # N->S
    np.zeros((2, 36)),                     # not 1-D
])
def test_bad_lat_cent_raises(mesh, bad):
    with pytest.raises(ValueError):
        compute_voronoi_to_latlon_weights(
            np.asarray(mesh.latCell), np.asarray(mesh.lonCell),
            n_lon=NLON, n_lat=NLAT, lat_cent=bad)
