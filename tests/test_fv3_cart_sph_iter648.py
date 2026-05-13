"""FV3_3D iter 648: cartesian_to_spherical_fv3 + spherical_to_cartesian_fv3.

Faithful JAX ports of FV3 radius-aware coord conversions
(tools/fv_grid_tools.F90:2373-2402).

Tests
-----

1. ``test_cart_to_sph_radius_preserved``.
2. ``test_cart_to_sph_pole_branch``.
3. ``test_sph_to_cart_unit_sphere``.
4. ``test_round_trip_random``.
5. ``test_sph_to_cart_scales_with_r``.
6. ``test_cart_to_sph_lon_range_atan2``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    cartesian_to_spherical_fv3,
    spherical_to_cartesian_fv3,
)


def test_cart_to_sph_radius_preserved():
    """r = sqrt(x²+y²+z²)."""
    x, y, z = 3.0, 4.0, 12.0  # 3-4-12 → 13
    _, _, r = cartesian_to_spherical_fv3(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(z),
    )
    assert abs(float(r) - 13.0) < 1e-12


def test_cart_to_sph_pole_branch():
    """At pole (|x|+|y|<eps): lon = 0."""
    lon, _, _ = cartesian_to_spherical_fv3(
        jnp.asarray(1e-12), jnp.asarray(1e-12), jnp.asarray(1.0),
    )
    assert abs(float(lon)) < 1e-14


def test_sph_to_cart_unit_sphere():
    """At r=1, equator: (0, 0, 1) → (lon=0, lat=0)·1 = (1, 0, 0)."""
    x, y, z = spherical_to_cartesian_fv3(
        jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1.0),
    )
    assert abs(float(x) - 1.0) < 1e-12
    assert abs(float(y)) < 1e-12
    assert abs(float(z)) < 1e-12


def test_round_trip_random():
    """Random (lon, lat, r) → cart → sph round-trip."""
    rng = np.random.default_rng(seed=648)
    for _ in range(5):
        lon_in = float(rng.uniform(-jnp.pi + 0.1, jnp.pi - 0.1))
        lat_in = float(rng.uniform(-jnp.pi / 2 + 0.2, jnp.pi / 2 - 0.2))
        r_in = float(rng.uniform(1.0, 1000.0))
        x, y, z = spherical_to_cartesian_fv3(
            jnp.asarray(lon_in), jnp.asarray(lat_in), jnp.asarray(r_in),
        )
        lon, lat, r = cartesian_to_spherical_fv3(x, y, z)
        assert abs(float(lon) - lon_in) < 1e-12
        assert abs(float(lat) - lat_in) < 1e-12
        assert abs(float(r) - r_in) / r_in < 1e-12


def test_sph_to_cart_scales_with_r():
    """Scaling r doubles each (x, y, z)."""
    lon, lat = 0.5, 0.3
    x1, y1, z1 = spherical_to_cartesian_fv3(
        jnp.asarray(lon), jnp.asarray(lat), jnp.asarray(1.0),
    )
    x2, y2, z2 = spherical_to_cartesian_fv3(
        jnp.asarray(lon), jnp.asarray(lat), jnp.asarray(2.0),
    )
    assert abs(float(x2) - 2 * float(x1)) < 1e-12
    assert abs(float(y2) - 2 * float(y1)) < 1e-12
    assert abs(float(z2) - 2 * float(z1)) < 1e-12


def test_cart_to_sph_lon_range_atan2():
    """FV3 returns lon in [-π, π] (atan2 range), NOT [0, 2π)."""
    # Point on -y axis: lon should be -π/2 (atan2(-1, 0) = -π/2)
    lon, _, _ = cartesian_to_spherical_fv3(
        jnp.asarray(0.0), jnp.asarray(-1.0), jnp.asarray(0.0),
    )
    assert abs(float(lon) - (-jnp.pi / 2)) < 1e-12
