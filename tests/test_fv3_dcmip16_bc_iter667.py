"""FV3_3D iter 667: dcmip16_bc_temperature + dcmip16_bc_pressure ports.

Faithful JAX ports of FV3 DCMIP16 baroclinic-instability test
T and p profiles (tools/test_cases.F90:6774-6805).

Tests
-----

1. ``test_bc_temperature_surface_equator``.
2. ``test_bc_pressure_surface_p0``.
3. ``test_bc_pressure_decreases_with_z``.
4. ``test_bc_temperature_shape``.
5. ``test_bc_finite``.
6. ``test_bc_temperature_realistic_range``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    dcmip16_bc_pressure,
    dcmip16_bc_temperature,
)


def test_bc_temperature_surface_equator():
    """At z=0, equator: T ≈ 290 K (between Tp=240 and Te=310)."""
    T = float(dcmip16_bc_temperature(jnp.asarray(0.0), jnp.asarray(0.0)))
    assert 250.0 < T < 320.0, f"surface equator T = {T}"


def test_bc_pressure_surface_p0():
    """At z=0, p = p0 = 100000 Pa."""
    p = float(dcmip16_bc_pressure(jnp.asarray(0.0), jnp.asarray(0.0)))
    assert abs(p - 1.0e5) < 1e-6


def test_bc_pressure_decreases_with_z():
    """p monotonically decreasing with z."""
    z = jnp.linspace(0.0, 30000.0, 30)
    lat = jnp.zeros_like(z)
    p = dcmip16_bc_pressure(z, lat)
    diffs = p[1:] - p[:-1]
    assert jnp.all(diffs < 0)


def test_bc_temperature_shape():
    """T shape matches input (z, lat)."""
    z = jnp.linspace(0.0, 20000.0, 20)
    lat = jnp.linspace(-1.5, 1.5, 20)
    T = dcmip16_bc_temperature(z, lat)
    assert T.shape == z.shape


def test_bc_finite():
    """No NaN/Inf in T or p for tropospheric range."""
    rng = np.random.default_rng(seed=667)
    z = jnp.asarray(rng.uniform(0, 25000, size=50))
    lat = jnp.asarray(rng.uniform(-jnp.pi / 2 + 0.1, jnp.pi / 2 - 0.1, size=50))
    T = dcmip16_bc_temperature(z, lat)
    p = dcmip16_bc_pressure(z, lat)
    assert jnp.all(jnp.isfinite(T))
    assert jnp.all(jnp.isfinite(p))
    assert jnp.all(T > 0)
    assert jnp.all(p > 0)


def test_bc_temperature_realistic_range():
    """Temperature stays in realistic atmospheric range across globe."""
    z = jnp.full((10,), 5000.0)
    lat = jnp.linspace(-1.5, 1.5, 10)
    T = dcmip16_bc_temperature(z, lat)
    assert jnp.all(T > 150.0)
    assert jnp.all(T < 350.0)
