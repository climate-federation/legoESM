"""FV3_3D iter 589: cube_transform (revised Schmidt) variant.

Tests
-----

1. ``test_cube_transform_runs`` — basic invocation.
2. ``test_cube_transform_differs_from_schmidt`` — π-shift produces
   different output from direct_transform at same params.
3. ``test_create_cubed_sphere_do_cube_transform`` — full grid creation
   with do_cube_transform=True produces valid grid.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    create_cubed_sphere,
    cube_transform,
    schmidt_transform,
)


def test_cube_transform_runs():
    """cube_transform produces finite output."""
    lon = jnp.asarray([0.0, jnp.pi / 4, jnp.pi / 2, jnp.pi])
    lat = jnp.asarray([0.0, jnp.pi / 8, jnp.pi / 4, -jnp.pi / 4])
    lon_new, lat_new = cube_transform(
        lon, lat,
        stretch_fac=2.0,
        target_lon=0.0,
        target_lat=0.0,
    )
    assert jnp.all(jnp.isfinite(lon_new))
    assert jnp.all(jnp.isfinite(lat_new))


def test_cube_transform_differs_from_schmidt():
    """The π-shift makes cube_transform output different from
    schmidt_transform at same params."""
    rng = np.random.default_rng(seed=589)
    lon = jnp.asarray(rng.uniform(0, 2 * jnp.pi, size=20))
    lat = jnp.asarray(rng.uniform(-jnp.pi / 2, jnp.pi / 2, size=20))
    lon_s, lat_s = schmidt_transform(
        lon, lat, 2.0, 0.5, 0.3,
    )
    lon_c, lat_c = cube_transform(
        lon, lat, 2.0, 0.5, 0.3,
    )
    # They should differ (π-shift is non-trivial).
    max_diff = float(jnp.max(jnp.abs(lon_s - lon_c) + jnp.abs(lat_s - lat_c)))
    assert max_diff > 1e-3, (
        f"cube_transform should differ from schmidt_transform; "
        f"got max diff={max_diff:.3e}"
    )


def test_create_cubed_sphere_do_cube_transform():
    """create_cubed_sphere with do_cube_transform=True produces valid grid."""
    n = 8
    grid = create_cubed_sphere(
        n,
        stretch_fac=2.0,
        target_lon=0.0,
        target_lat=0.0,
        do_cube_transform=True,
    )
    assert jnp.all(jnp.isfinite(grid.lat))
    assert jnp.all(jnp.isfinite(grid.lon))
    assert jnp.all(jnp.isfinite(grid.area))
    expected_area = 4.0 * jnp.pi * grid.radius ** 2
    actual_area = float(jnp.sum(grid.area))
    rel_err = abs(actual_area - expected_area) / expected_area
    assert rel_err < 0.05, (
        f"total area should be ~4π·R²; got rel_err={rel_err:.3f}"
    )


def test_do_cube_transform_false_matches_schmidt():
    """do_cube_transform=False with stretch should use schmidt_transform."""
    n = 8
    g_schmidt = create_cubed_sphere(
        n, stretch_fac=2.0, target_lon=0.0, target_lat=0.0,
    )
    g_explicit = create_cubed_sphere(
        n, stretch_fac=2.0, target_lon=0.0, target_lat=0.0,
        do_cube_transform=False,
    )
    diff = float(jnp.abs(g_schmidt.lat - g_explicit.lat).max())
    assert diff < 1e-10, (
        f"do_cube_transform=False should match default; diff={diff}"
    )
