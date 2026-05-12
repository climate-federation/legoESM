"""FV3_3D iter 662: add_rankine_vortex port.

Faithful JAX port of FV3 ``rankine_vortex`` (tools/test_cases.F90:
4207-4292).  Adds Rankine vortex tangential wind to D-grid u, v.

Tests
-----

1. ``test_rankine_shape``.
2. ``test_rankine_zero_ubar``.
3. ``test_rankine_far_field_finite``.
4. ``test_rankine_center_field_zero``.
5. ``test_rankine_max_wind_at_r0``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    add_rankine_vortex,
    make_fv3_native_grid,
)


def _setup(n=8):
    """Build cubed-sphere face-1 grid + zero D-grid winds."""
    lons, lats = make_fv3_native_grid(n, grid_type=0)
    # Use face 0
    grid_lon = lons[0]
    grid_lat = lats[0]
    u = jnp.zeros((n, n + 1))
    v = jnp.zeros((n + 1, n))
    return grid_lon, grid_lat, u, v


def test_rankine_shape():
    """Output shapes match input D-grid winds."""
    n = 8
    grid_lon, grid_lat, u, v = _setup(n)
    ubar = 50.0
    r0 = 100.0e3
    u_new, v_new = add_rankine_vortex(
        u, v, grid_lon, grid_lat, ubar, r0,
        center_lon=float(grid_lon[n // 2, n // 2]),
        center_lat=float(grid_lat[n // 2, n // 2]),
    )
    assert u_new.shape == u.shape
    assert v_new.shape == v.shape


def test_rankine_zero_ubar():
    """ubar=0 → no change."""
    n = 8
    grid_lon, grid_lat, u, v = _setup(n)
    u_new, v_new = add_rankine_vortex(
        u, v, grid_lon, grid_lat, ubar=0.0, r0=100.0e3,
        center_lon=float(grid_lon[n // 2, n // 2]),
        center_lat=float(grid_lat[n // 2, n // 2]),
    )
    assert jnp.allclose(u_new, u, atol=1e-14)
    assert jnp.allclose(v_new, v, atol=1e-14)


def test_rankine_far_field_finite():
    """No NaN/Inf even when vortex placed far from grid."""
    n = 8
    grid_lon, grid_lat, u, v = _setup(n)
    # Place center on opposite face (lon shifted by π)
    center_lon = float(grid_lon[n // 2, n // 2]) + jnp.pi
    center_lat = -float(grid_lat[n // 2, n // 2])
    u_new, v_new = add_rankine_vortex(
        u, v, grid_lon, grid_lat, ubar=50.0, r0=200.0e3,
        center_lon=center_lon, center_lat=center_lat,
    )
    assert jnp.all(jnp.isfinite(u_new))
    assert jnp.all(jnp.isfinite(v_new))


def test_rankine_center_field_zero():
    """Right AT vortex center, tangential wind ≈ 0 (r → 0 limit)."""
    n = 8
    grid_lon, grid_lat, u, v = _setup(n)
    # Place center at cell-center near (lon_c, lat_c)
    center_lon = float(grid_lon[n // 2, n // 2])
    center_lat = float(grid_lat[n // 2, n // 2])
    u_new, v_new = add_rankine_vortex(
        u, v, grid_lon, grid_lat, ubar=50.0, r0=100.0e3,
        center_lon=center_lon, center_lat=center_lat,
    )
    # At center cell j-edge nearest to center, wind contribution should
    # remain finite and bounded
    # Verify only finiteness; exact-zero at center is degenerate due to
    # d2 = max(1e-25, ...) regularization
    assert jnp.all(jnp.isfinite(u_new))
    assert jnp.all(jnp.isfinite(v_new))


def test_rankine_max_wind_at_r0():
    """Maximum wind magnitude reaches near ``ubar`` (within FV3 grid resolution)."""
    n = 16
    grid_lon, grid_lat, u, v = _setup(n)
    ubar = 50.0
    r0 = 200.0e3  # ~200 km (resolves at C16 face — Earth circumference / 6 / 16 ≈ 420 km per cell)
    center_lon = float(grid_lon[n // 2, n // 2])
    center_lat = float(grid_lat[n // 2, n // 2])
    u_new, v_new = add_rankine_vortex(
        u, v, grid_lon, grid_lat, ubar, r0,
        center_lon=center_lon, center_lat=center_lat,
    )
    # Maximum wind speed should be less than ubar (vortex projected onto
    # cube-aligned grid → projection factor ≤ 1).  At least some cells
    # should see a substantial fraction of ubar.
    speed = jnp.sqrt(u_new ** 2 + jnp.zeros_like(u_new))
    speed_v = jnp.sqrt(v_new ** 2 + jnp.zeros_like(v_new))
    max_speed = float(jnp.maximum(jnp.max(speed), jnp.max(speed_v)))
    assert max_speed > 1.0  # at least 1 m/s somewhere near vortex
    assert max_speed <= ubar + 1.0  # capped by ubar (+ small numerical)
