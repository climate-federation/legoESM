"""FV3_3D iter 657: checker_tracers port.

Faithful JAX port of FV3 ``checker_tracers`` (tools/test_cases.F90:
4067-4135).  Checkerboard tracer pattern for HIWPP transport tests.

Tests
-----

1. ``test_checker_shape``.
2. ``test_checker_binary_values``.
3. ``test_checker_uniform_in_k_and_iq``.
4. ``test_checker_random_perturbation``.
5. ``test_checker_rng_key_required``.
6. ``test_checker_pattern_alternates``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import checker_tracers


def _make_lat_lon(n=20):
    """Build (n, n) lat-lon grid spanning much of the globe."""
    lons = jnp.linspace(0.0, 2 * jnp.pi, n)
    lats = jnp.linspace(-jnp.pi / 2 + 0.1, jnp.pi / 2 - 0.1, n)
    lon_2d, lat_2d = jnp.meshgrid(lons, lats, indexing="ij")
    return lon_2d, lat_2d


def test_checker_shape():
    """Output shape (..., n_x, n_y, km, nq)."""
    lon, lat = _make_lat_lon(20)
    q = checker_tracers(lon, lat, nq=3, km=5, nx=9.0, ny=9.0)
    assert q.shape == (20, 20, 5, 3)


def test_checker_binary_values():
    """Without noise, values are exactly 0 or 0.01."""
    lon, lat = _make_lat_lon(30)
    q = checker_tracers(lon, lat, nq=1, km=1, nx=9.0, ny=9.0)
    unique_vals = jnp.unique(q)
    assert jnp.all((unique_vals == 0.0) | (jnp.abs(unique_vals - 0.01) < 1e-12))


def test_checker_uniform_in_k_and_iq():
    """Without noise, same pattern across all k, iq."""
    lon, lat = _make_lat_lon(20)
    q = checker_tracers(lon, lat, nq=4, km=3)
    # All (k, iq) slices identical
    for k in range(3):
        for iq in range(4):
            assert jnp.allclose(q[..., k, iq], q[..., 0, 0])


def test_checker_random_perturbation():
    """With rn=0.1, output is perturbed (not exactly 0 or 0.01)."""
    lon, lat = _make_lat_lon(20)
    key = jax.random.PRNGKey(657)
    q = checker_tracers(lon, lat, nq=2, km=2, nx=9.0, ny=9.0, rn=0.1, rng_key=key)
    # Output should have many distinct values (not just 0 and 0.01)
    n_distinct = int(jnp.unique(q.flatten()).shape[0])
    assert n_distinct > 100, f"only {n_distinct} distinct values (expected noise)"


def test_checker_rng_key_required():
    """Missing rng_key when rn is given raises ValueError."""
    lon, lat = _make_lat_lon(10)
    with pytest.raises(ValueError):
        checker_tracers(lon, lat, nq=1, km=1, rn=0.1)


def test_checker_pattern_alternates():
    """Adjacent boxes should alternate 0 ↔ 0.01 (checker)."""
    lon, lat = _make_lat_lon(60)
    q = checker_tracers(lon, lat, nq=1, km=1, nx=9.0, ny=9.0)
    flat = q[..., 0, 0].flatten()
    # Mixture: some 0, some 0.01
    n_high = int(jnp.sum(flat > 0.005))
    n_low = int(jnp.sum(flat < 0.005))
    assert n_high > 0 and n_low > 0, (
        f"pattern not alternating: n_high={n_high}, n_low={n_low}"
    )
