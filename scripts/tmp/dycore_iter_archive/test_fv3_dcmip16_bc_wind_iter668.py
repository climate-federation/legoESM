"""FV3_3D iter 668: dcmip16_bc_uwind + dcmip16_bc_sphum ports.

Faithful JAX ports of FV3 DCMIP16 BC zonal wind + humidity
(tools/test_cases.F90:6807-6852).

Tests
-----

1. ``test_bc_uwind_surface_equator_zero``.
2. ``test_bc_uwind_finite``.
3. ``test_bc_uwind_jet_peak``.
4. ``test_bc_sphum_surface_max``.
5. ``test_bc_sphum_stratosphere_qt``.
6. ``test_bc_sphum_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    dcmip16_bc_sphum,
    dcmip16_bc_temperature,
    dcmip16_bc_uwind,
)


def test_bc_uwind_surface_equator_zero():
    """At surface equator, zonal wind is 0 (no shear)."""
    T = dcmip16_bc_temperature(jnp.asarray(0.0), jnp.asarray(0.0))
    u = dcmip16_bc_uwind(jnp.asarray(0.0), T, jnp.asarray(0.0))
    assert abs(float(u)) < 1e-10


def test_bc_uwind_finite():
    """No NaN/Inf at troposphere range."""
    rng = np.random.default_rng(seed=668)
    z = jnp.asarray(rng.uniform(0, 12000, size=30))
    lat = jnp.asarray(rng.uniform(-1.4, 1.4, size=30))
    T = dcmip16_bc_temperature(z, lat)
    u = dcmip16_bc_uwind(z, T, lat)
    assert jnp.all(jnp.isfinite(u))


def test_bc_uwind_jet_peak():
    """Peak BC jet is in midlatitudes, positive."""
    z = jnp.full((10,), 10000.0)
    lat = jnp.linspace(0.1, 1.5, 10)
    T = dcmip16_bc_temperature(z, lat)
    u = dcmip16_bc_uwind(z, T, lat)
    # Should be positive in NH (eastward) and have a max
    assert jnp.any(u > 5.0), f"max u in midlatitudes = {float(jnp.max(u))}"


def test_bc_sphum_surface_max():
    """At surface equator, q ≈ q0 = 0.018."""
    # eta = p/ps = 1 → exp(0) = 1; lat=0 → exp(0) = 1
    p = jnp.asarray(1.0e5)
    ps = jnp.asarray(1.0e5)
    lat = jnp.asarray(0.0)
    q = dcmip16_bc_sphum(p, ps, lat)
    assert abs(float(q) - 0.018) < 1e-10


def test_bc_sphum_stratosphere_qt():
    """In stratosphere (p < ptrop), q = qt = 1e-12."""
    p = jnp.asarray(5000.0)        # below ptrop=10000
    ps = jnp.asarray(1.0e5)
    lat = jnp.asarray(0.0)
    q = dcmip16_bc_sphum(p, ps, lat)
    assert abs(float(q) - 1.0e-12) < 1e-20


def test_bc_sphum_finite():
    """q finite for any (p, ps, lat) in reasonable range."""
    rng = np.random.default_rng(seed=669)
    p = jnp.asarray(rng.uniform(5000, 1.0e5, size=50))
    ps = jnp.full((50,), 1.0e5)
    lat = jnp.asarray(rng.uniform(-1.4, 1.4, size=50))
    q = dcmip16_bc_sphum(p, ps, lat)
    assert jnp.all(jnp.isfinite(q))
    assert jnp.all(q > 0)
