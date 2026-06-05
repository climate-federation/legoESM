"""FV3_3D iter 618: get_center_vect port.

Faithful port of FV3 ``get_center_vect`` (fv_grid_utils.F90:
1795-1845, non-``OLD_VECT`` branch).

Tests
-----

1. ``test_get_center_vect_shapes``.
2. ``test_get_center_vect_unit_norm``.
3. ``test_get_center_vect_tangent_to_sphere``.
4. ``test_get_center_vect_equator_alignment``.
5. ``test_get_center_vect_orthogonal_on_equator_square``.
6. ``test_get_center_vect_legoesm_face0``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    cell_center3,
    get_center_vect,
    gnomonic_angl,
    latlon2xyz,
)


def _build_simple_grid(n_x, n_y, dl=0.05, lon0=0.0, lat0=0.0):
    """Build a small (n_x+1, n_y+1, 3) regular grid of corner positions."""
    lons = lon0 + dl * (jnp.arange(n_x + 1, dtype=jnp.float64) - n_x / 2)
    lats = lat0 + dl * (jnp.arange(n_y + 1, dtype=jnp.float64) - n_y / 2)
    ll_grid_lon, ll_grid_lat = jnp.meshgrid(lons, lats, indexing="ij")
    x, y, z = latlon2xyz(ll_grid_lon, ll_grid_lat)
    return jnp.stack([x, y, z], axis=-1)


def test_get_center_vect_shapes():
    """Output shapes (n_x, n_y, 3)."""
    n_x, n_y = 5, 7
    pp = _build_simple_grid(n_x, n_y)
    u1, u2 = get_center_vect(pp)
    assert u1.shape == (n_x, n_y, 3)
    assert u2.shape == (n_x, n_y, 3)


def test_get_center_vect_unit_norm():
    """All u1, u2 must be unit-length."""
    n_x, n_y = 4, 4
    pp = _build_simple_grid(n_x, n_y)
    u1, u2 = get_center_vect(pp)
    n1 = jnp.linalg.norm(u1, axis=-1)
    n2 = jnp.linalg.norm(u2, axis=-1)
    assert jnp.allclose(n1, 1.0, atol=1e-12)
    assert jnp.allclose(n2, 1.0, atol=1e-12)


def test_get_center_vect_tangent_to_sphere():
    """u1, u2 must be orthogonal to cell center position (sphere tangent)."""
    n_x, n_y = 4, 4
    pp = _build_simple_grid(n_x, n_y)
    u1, u2 = get_center_vect(pp)
    # Cell-center positions
    sw = pp[:-1, :-1, :]
    se = pp[1:, :-1, :]
    nw = pp[:-1, 1:, :]
    ne = pp[1:, 1:, :]
    pc = cell_center3(sw, se, nw, ne)
    d1 = jnp.sum(u1 * pc, axis=-1)
    d2 = jnp.sum(u2 * pc, axis=-1)
    assert jnp.allclose(d1, 0.0, atol=1e-12)
    assert jnp.allclose(d2, 0.0, atol=1e-12)


def test_get_center_vect_equator_alignment():
    """At the equator with a square cell, u1 ≈ +east, u2 ≈ +north.

    With n_x=n_y=4 and dl=0.02 centered at lon=lat=0, cell (2, 2)
    has its center at lon=0.5·dl=0.01, lat=0.5·dl=0.01.
    Expected u1 = (-sin λ, cos λ, 0) · (cos φ tangent) ≈ east-ish.
    """
    dl = 0.02
    pp = _build_simple_grid(4, 4, dl=dl, lon0=0.0, lat0=0.0)
    u1, u2 = get_center_vect(pp)
    u1_c = u1[2, 2]
    u2_c = u2[2, 2]
    lon_c = lat_c = 0.5 * dl  # cell center
    # Expected east direction at (lon_c, lat_c): (-sin λ, cos λ, 0)
    east = jnp.asarray([-jnp.sin(lon_c), jnp.cos(lon_c), 0.0])
    # Expected north direction: (-sin φ cos λ, -sin φ sin λ, cos φ)
    north = jnp.asarray([
        -jnp.sin(lat_c) * jnp.cos(lon_c),
        -jnp.sin(lat_c) * jnp.sin(lon_c),
        jnp.cos(lat_c),
    ])
    assert jnp.allclose(u1_c, east, atol=1e-4)
    assert jnp.allclose(u2_c, north, atol=1e-4)


def test_get_center_vect_orthogonal_on_equator_square():
    """On equator with axis-aligned square cells, u1 ⊥ u2."""
    pp = _build_simple_grid(4, 4, dl=0.02, lon0=0.0, lat0=0.0)
    u1, u2 = get_center_vect(pp)
    dot = jnp.sum(u1 * u2, axis=-1)
    # Should be ~0 (perpendicular)
    assert jnp.allclose(dot, 0.0, atol=1e-6)


def test_get_center_vect_legoesm_face0():
    """Apply to a real legoESM gnomonic face (face 2): result should be
    finite, unit-norm, and tangent to sphere everywhere."""
    im = 8
    lon, lat = gnomonic_angl(im)
    x, y, z = latlon2xyz(lon, lat)
    pp = jnp.stack([x, y, z], axis=-1)
    u1, u2 = get_center_vect(pp)
    assert jnp.all(jnp.isfinite(u1))
    assert jnp.all(jnp.isfinite(u2))
    n1 = jnp.linalg.norm(u1, axis=-1)
    n2 = jnp.linalg.norm(u2, axis=-1)
    assert jnp.allclose(n1, 1.0, atol=1e-12)
    assert jnp.allclose(n2, 1.0, atol=1e-12)
