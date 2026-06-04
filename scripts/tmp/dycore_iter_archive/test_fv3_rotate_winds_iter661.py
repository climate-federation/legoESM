"""FV3_3D iter 661: rotate_winds_sphere_cube port.

Faithful JAX port of FV3 ``rotate_winds`` (tools/test_cases.F90:
8183-8226).  Rotate winds between sphere (lat/lon) frame and
cube (i, j) frame at a point.

Tests
-----

1. ``test_rotate_zero_winds``.
2. ``test_rotate_roundtrip``.
3. ``test_rotate_finite``.
4. ``test_rotate_invalid_direction``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import rotate_winds_sphere_cube


def _make_neighbors(lon_t, lat_t, dlon=0.05, dlat=0.05):
    """Build 4 neighbor points around central t1."""
    lon1, lat1 = lon_t - dlon, lat_t          # west
    lon3, lat3 = lon_t + dlon, lat_t          # east
    lon2, lat2 = lon_t, lat_t - dlat          # south
    lon4, lat4 = lon_t, lat_t + dlat          # north
    return (
        jnp.asarray(lon1), jnp.asarray(lat1),
        jnp.asarray(lon2), jnp.asarray(lat2),
        jnp.asarray(lon3), jnp.asarray(lat3),
        jnp.asarray(lon4), jnp.asarray(lat4),
    )


def test_rotate_zero_winds():
    """Zero winds map to zero in both directions."""
    lon_t, lat_t = 0.5, 0.3
    nbrs = _make_neighbors(lon_t, lat_t)
    for dir_ in (1, 2):
        newu, newv = rotate_winds_sphere_cube(
            jnp.asarray(0.0), jnp.asarray(0.0),
            *nbrs,
            jnp.asarray(lon_t), jnp.asarray(lat_t),
            direction=dir_,
        )
        assert abs(float(newu)) < 1e-14
        assert abs(float(newv)) < 1e-14


def test_rotate_roundtrip():
    """Sphere→cube→sphere recovers original winds."""
    rng = np.random.default_rng(seed=661)
    for _ in range(5):
        lon_t = float(rng.uniform(0.3, 2 * jnp.pi - 0.3))
        lat_t = float(rng.uniform(-1.0, 1.0))
        nbrs = _make_neighbors(lon_t, lat_t)
        u0 = float(rng.uniform(-10, 10))
        v0 = float(rng.uniform(-10, 10))
        # Sphere → cube
        uc, vc = rotate_winds_sphere_cube(
            jnp.asarray(u0), jnp.asarray(v0),
            *nbrs,
            jnp.asarray(lon_t), jnp.asarray(lat_t),
            direction=1,
        )
        # Cube → sphere (inverse)
        us, vs = rotate_winds_sphere_cube(
            uc, vc,
            *nbrs,
            jnp.asarray(lon_t), jnp.asarray(lat_t),
            direction=2,
        )
        assert abs(float(us) - u0) < 1e-10
        assert abs(float(vs) - v0) < 1e-10


def test_rotate_finite():
    """No NaN/Inf on random inputs."""
    rng = np.random.default_rng(seed=662)
    lon_t = float(rng.uniform(0.5, 2 * jnp.pi - 0.5))
    lat_t = float(rng.uniform(-0.8, 0.8))
    nbrs = _make_neighbors(lon_t, lat_t)
    u, v = rotate_winds_sphere_cube(
        jnp.asarray(5.0), jnp.asarray(-3.0),
        *nbrs,
        jnp.asarray(lon_t), jnp.asarray(lat_t),
        direction=1,
    )
    assert jnp.isfinite(u)
    assert jnp.isfinite(v)


def test_rotate_invalid_direction():
    """direction != 1 or 2 raises ValueError."""
    lon_t, lat_t = 0.5, 0.3
    nbrs = _make_neighbors(lon_t, lat_t)
    with pytest.raises(ValueError):
        rotate_winds_sphere_cube(
            jnp.asarray(1.0), jnp.asarray(1.0),
            *nbrs,
            jnp.asarray(lon_t), jnp.asarray(lat_t),
            direction=3,
        )
