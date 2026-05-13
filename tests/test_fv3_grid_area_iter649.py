"""FV3_3D iter 649: grid_area_fv3 port.

Faithful JAX port of FV3 ``grid_area`` (tools/fv_grid_tools.F90:
2512-2620, spherical-excess branch).  2D vectorized
cell-area computation via iter-614 get_area.

Tests
-----

1. ``test_grid_area_shape``.
2. ``test_grid_area_positive``.
3. ``test_grid_area_matches_legoesm``.
4. ``test_grid_area_scales_with_radius``.
5. ``test_grid_area_6_face_total``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    create_cubed_sphere,
    grid_area_fv3,
    make_fv3_native_grid,
)


def test_grid_area_shape():
    """Output shape (..., n_x, n_y) from corners (..., n_x+1, n_y+1)."""
    n = 10
    lons, lats = make_fv3_native_grid(n, grid_type=0)
    # lons, lats shape (6, n+1, n+1)
    area = grid_area_fv3(lons, lats, radius=1.0)
    assert area.shape == (6, n, n)


def test_grid_area_positive():
    """All cell areas > 0."""
    n = 8
    lons, lats = make_fv3_native_grid(n, grid_type=0)
    area = grid_area_fv3(lons, lats, radius=1.0)
    assert jnp.all(area > 0)


def test_grid_area_min_max_ratio():
    """FV3 gnomonic_ed area max/min ratio < 2.5 (dx, dy each √2 →
    area = dx·dy ratio ≈ 2)."""
    n = 16
    lons, lats = make_fv3_native_grid(n, grid_type=0)
    area = grid_area_fv3(lons, lats, radius=1.0)
    a_min = float(jnp.min(area))
    a_max = float(jnp.max(area))
    ratio = a_max / a_min
    assert ratio < 2.5, f"area max/min ratio = {ratio}, expected < 2.5"


def test_grid_area_scales_with_radius():
    """Doubling radius → area scales by 4."""
    n = 6
    lons, lats = make_fv3_native_grid(n, grid_type=0)
    a1 = grid_area_fv3(lons, lats, radius=1.0)
    a2 = grid_area_fv3(lons, lats, radius=2.0)
    assert jnp.allclose(a2, 4.0 * a1, atol=1e-10)


def test_grid_area_6_face_total():
    """Total area over all 6 faces ≈ 4π·R² (full sphere)."""
    n = 24
    lons, lats = make_fv3_native_grid(n, grid_type=0)
    R = 1.0
    area = grid_area_fv3(lons, lats, radius=R)
    total = float(jnp.sum(area))
    expected = 4.0 * float(jnp.pi) * R * R
    rel_err = abs(total - expected) / expected
    assert rel_err < 1e-9, (
        f"total = {total}, expected = {expected}, rel_err = {rel_err}"
    )
