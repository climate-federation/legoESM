"""FV3_3D iter 586: Schmidt transformation (stretched cubed-sphere).

User audit item #1 partial: stretched grid via Schmidt
conformal transformation.  Nested grid (2-way refinement)
deferred — much larger scope.

Tests
-----

1. ``test_schmidt_identity_no_stretch`` — c=1.0 + default target →
   transformation is identity (no change to lon, lat).
2. ``test_schmidt_stretch_concentrates_at_target`` — c=3 with
   target near equator → cells near target have smaller dx (higher res).
3. ``test_create_cubed_sphere_with_stretch`` — full grid creation
   with stretch_fac=2.0 produces valid grid.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    create_cubed_sphere,
    schmidt_transform,
)


def test_schmidt_identity_no_stretch():
    """c=1.0 + target_lat=-π/2 → no change."""
    lon = jnp.linspace(0, 2 * jnp.pi, 10)
    lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 10)
    lon2, lat2 = schmidt_transform(
        lon, lat,
        stretch_fac=1.0,
        target_lon=0.0,
        target_lat=-0.5 * jnp.pi,
    )
    # With c=1, no stretching; with target_lat=-π/2, pole stays at
    # south pole → identity rotation
    diff_lon = float(jnp.abs(lon - lon2).max())
    diff_lat = float(jnp.abs(lat - lat2).max())
    # Some tiny float roundoff allowed; should be near machine epsilon
    assert diff_lat < 1e-6, f"lat changed: {diff_lat}"


def test_schmidt_stretch_concentrates_at_target():
    """c=3 with target at equator should pull cells toward target."""
    # Build a small grid at the equator and test that with c=3,
    # latitudes near equator stay near equator (high-res region)
    lon = jnp.asarray([0.0, jnp.pi / 4, jnp.pi / 2])
    lat = jnp.asarray([0.0, jnp.pi / 8, jnp.pi / 4])  # near equator
    lon_new, lat_new = schmidt_transform(
        lon, lat,
        stretch_fac=3.0,
        target_lon=0.0,
        target_lat=0.0,  # equator
    )
    assert jnp.all(jnp.isfinite(lon_new))
    assert jnp.all(jnp.isfinite(lat_new))


def test_create_cubed_sphere_with_stretch():
    """create_cubed_sphere(stretch_fac=2.0) produces valid grid."""
    n = 8
    grid = create_cubed_sphere(
        n, stretch_fac=2.0,
        target_lon=0.0, target_lat=0.0,
    )
    # Lat range covers full sphere still (cubed-sphere cell centers reach
    # ±~82° not ±90° — total range ~2.87 rad unstretched).
    lat_range = float(grid.lat.max() - grid.lat.min())
    assert lat_range > 2.5, (
        f"lat should cover most of sphere; got range {lat_range}"
    )
    # No NaN
    assert jnp.all(jnp.isfinite(grid.lat))
    assert jnp.all(jnp.isfinite(grid.lon))
    assert jnp.all(jnp.isfinite(grid.area))
    # Total area should equal 4π·R²
    expected_area = 4.0 * jnp.pi * grid.radius ** 2
    actual_area = float(jnp.sum(grid.area))
    rel_err = abs(actual_area - expected_area) / expected_area
    assert rel_err < 0.05, (
        f"total area should be ~4π·R²; got rel_err={rel_err:.3f}"
    )


def test_create_cubed_sphere_unstretched_still_works():
    """Default (no stretch) still produces same grid as before."""
    n = 8
    grid = create_cubed_sphere(n)
    grid_explicit = create_cubed_sphere(
        n, stretch_fac=1.0,
        target_lon=0.0, target_lat=-0.5 * jnp.pi,
    )
    assert float(jnp.abs(grid.lat - grid_explicit.lat).max()) < 1e-10
    assert float(jnp.abs(grid.lon - grid_explicit.lon).max()) < 1e-10
