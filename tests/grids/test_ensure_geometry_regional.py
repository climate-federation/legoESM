"""Regression guard for the ``ensure_geometry`` regional-grid bug.

Before the 2026-05-17 fix, ``ensure_geometry`` rebuilt a global lat-lon
``LatLonCGridGeometry`` from ``n_lat`` / ``n_lon`` alone whenever its
input was a regional ``LatLonGrid``, silently inflating ``dx`` / ``dy``
by the ratio between the global and regional extents. On the Petersen
lock-exchange channel (~64 km × 4 km) the inflation was 600x, which
weakened the pressure-gradient acceleration by 600x and silently damped
the gravity-current velocity by a factor of ~12. The fix routes the
input grid's actual 1-D lat / lon arrays into the geometry builder.

These tests pin the fix:

1. A regional ``LatLonGrid`` round-trips through ``ensure_geometry``
   without losing its ``dlat`` / ``dlon``.
2. A regional ``LatLonCGridGeometry`` reports the regional dx, not the
   global dx.
3. The global path is byte-for-byte unchanged (legacy global
   construction still works).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import (
    LatLonCGridGeometry,
    create_latlon_grid,
    create_latlon_geometry,
    create_regional_latlon_grid,
    ensure_geometry,
)


def test_ensure_geometry_preserves_regional_dlon():
    n_lat, n_lon = 4, 64
    grid, _wall = create_regional_latlon_grid(
        n_lat=n_lat, n_lon=n_lon,
        lat_south=-0.018, lat_north=+0.018,
        lon_west=0.0, lon_east=0.576,
    )
    geom = ensure_geometry(grid)

    np.testing.assert_allclose(float(geom.dlon), float(grid.dlon), rtol=1e-6)
    np.testing.assert_allclose(float(geom.dlat), float(grid.dlat), rtol=1e-6)
    # 64 km zonal over 64 interior cells -> dx ~ 1 km. Global mis-fix
    # would give dx ~ 600 km.
    R = grid.radius
    expected_dx_eq = R * float(grid.dlon) * 1.0
    np.testing.assert_allclose(expected_dx_eq, 1000.0, rtol=0.1)


def test_ensure_geometry_preserves_regional_cos_lat():
    grid, _wall = create_regional_latlon_grid(
        n_lat=4, n_lon=64,
        lat_south=-0.018, lat_north=+0.018,
        lon_west=0.0, lon_east=0.576,
    )
    geom = ensure_geometry(grid)
    # Equatorial channel: cos(lat) should be ~1 everywhere, not the
    # 0.97 that the global-reconstruction bug yielded for an n_lat=6
    # ``-pi/2 -> +pi/2`` grid.
    assert float(geom.cos_lat.max()) > 0.999
    assert float(geom.cos_lat.min()) > 0.999


def test_ensure_geometry_idempotent_on_geometry():
    """Already-converted geometry is returned unchanged."""
    geom = create_latlon_geometry(n_lat=4, n_lon=8)
    out = ensure_geometry(geom)
    assert out is geom


def test_ensure_geometry_global_path_unchanged_at_float32_precision():
    """Historical global path still produces the full-sphere grid. The
    ensure-vs-direct round-trip lands lat_1d/lon_1d through ``create_latlon_grid``'s
    float32 storage so they differ from the directly-built geometry at
    the ~3e-7 relative level (float32 epsilon); that's far tighter than
    the 600x dx blow-up the regional bug used to produce, but loose
    enough to absorb the round-trip cast."""
    n_lat, n_lon = 36, 72
    grid_global = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    geom_via_ensure = ensure_geometry(grid_global)
    geom_via_global = create_latlon_geometry(n_lat=n_lat, n_lon=n_lon)
    np.testing.assert_allclose(
        np.asarray(geom_via_ensure.dx_T),
        np.asarray(geom_via_global.dx_T),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(geom_via_ensure.area_T),
        np.asarray(geom_via_global.area_T),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        float(geom_via_ensure.dlon),
        float(geom_via_global.dlon),
        rtol=1e-5,
    )


def test_create_latlon_geometry_rejects_bad_lat_1d_shape():
    with pytest.raises(ValueError, match="lat_1d"):
        create_latlon_geometry(n_lat=4, n_lon=8, lat_1d=jnp.zeros(3))


def test_create_latlon_geometry_rejects_bad_lon_1d_shape():
    with pytest.raises(ValueError, match="lon_1d"):
        create_latlon_geometry(n_lat=4, n_lon=8, lon_1d=jnp.zeros(5))


def test_regional_dx_matches_great_circle_calc():
    """End-to-end sanity: dx_T at the equator should match the great-
    circle ``R · cos(lat) · dlon`` directly, for the regional channel."""
    grid, _wall = create_regional_latlon_grid(
        n_lat=4, n_lon=64,
        lat_south=-0.018, lat_north=+0.018,
        lon_west=0.0, lon_east=0.576,
    )
    geom = ensure_geometry(grid)
    R = grid.radius
    dlon = float(geom.dlon)
    # n_lat+2 rows (2 wall cells), interior centre at j=3 (n_lat//2+1)
    j_eq = (geom.n_lat) // 2
    expected = R * dlon * float(geom.cos_lat[j_eq])
    np.testing.assert_allclose(
        float(geom.dx_T[j_eq, 1]), expected, rtol=1e-5,
    )
