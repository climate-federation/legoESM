"""FV3_3D iter 819: mean_wind_layer_fv3.

ū = Σ u_mid·Δz·mask / Σ Δz·mask  over [z_bot, z_top].

Tests
-----

1. ``test_mean_uniform``: u=const, v=const → ū=u, v̄=v.
2. ``test_mean_linear``: linear u → ū = (u_bot + u_top)/2.
3. ``test_mean_zero_v``: v=0 everywhere → v̄=0.
4. ``test_mean_compose_iter817``: chain to mean over EIL.
5. ``test_mean_outside_layer_floored``: tight bounds outside layer.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    effective_inflow_layer_fv3,
    mean_wind_layer_fv3,
)


def test_mean_uniform():
    """u, v constant → mean = const."""
    z = jnp.linspace(0.0, 10_000.0, 21)
    u = jnp.full((21,), 15.0)
    v = jnp.full((21,), 3.0)
    u_mean, v_mean = mean_wind_layer_fv3(u, v, z, 1000.0, 9000.0)
    np.testing.assert_allclose(np.asarray(u_mean), 15.0, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(v_mean), 3.0, rtol=1e-10)


def test_mean_linear():
    """u = 0.005·z → mean over [2000, 8000] ≈ 0.005·5000 = 25."""
    z = jnp.linspace(0.0, 10_000.0, 41)
    u = 0.005 * z
    v = jnp.zeros_like(z)
    u_mean, _ = mean_wind_layer_fv3(u, v, z, 2000.0, 8000.0)
    # Midpoint of linear function over [2000, 8000] = 0.005·5000 = 25
    np.testing.assert_allclose(np.asarray(u_mean), 25.0, rtol=0.05)


def test_mean_zero_v():
    """v ≡ 0 → v̄ = 0."""
    z = jnp.linspace(0.0, 10_000.0, 21)
    u = jnp.linspace(0.0, 50.0, 21)
    v = jnp.zeros_like(z)
    _, v_mean = mean_wind_layer_fv3(u, v, z, 1000.0, 5000.0)
    np.testing.assert_allclose(np.asarray(v_mean), 0.0, atol=1e-15)


def test_mean_compose_iter817():
    """EIL bounds from iter-817 → mean wind over EIL."""
    z = jnp.array([100.0, 500.0, 1500.0, 3000.0, 6000.0, 10_000.0])
    cape = jnp.array([50.0, 80.0, 500.0, 1000.0, 800.0, 50.0])
    cin = jnp.array([300.0, 280.0, 100.0, 50.0, 80.0, 400.0])
    u = jnp.array([5.0, 8.0, 12.0, 18.0, 25.0, 35.0])
    v = jnp.array([0.0, 1.0, 3.0, 5.0, 7.0, 10.0])
    z_bot, z_top = effective_inflow_layer_fv3(cape, cin, z)
    u_mean, v_mean = mean_wind_layer_fv3(u, v, z, z_bot, z_top)
    # EIL bounds (1500, 6000); midpoints 2250, 4500
    # Plausible range
    assert 12.0 < float(u_mean) < 25.0
    assert 3.0 < float(v_mean) < 7.0


def test_mean_outside_layer_floored():
    """Bounds entirely below column → weights = 0 → output ≈ 0 via floor."""
    z = jnp.linspace(1000.0, 5000.0, 5)
    u = jnp.array([10.0, 12.0, 14.0, 16.0, 18.0])
    v = jnp.zeros_like(z)
    # Layer [0, 500] — entirely below column
    u_mean, v_mean = mean_wind_layer_fv3(u, v, z, 0.0, 500.0)
    assert jnp.isfinite(u_mean)
    assert jnp.isfinite(v_mean)
    # weights = 0 → 0/floor ≈ 0
    np.testing.assert_allclose(np.asarray(u_mean), 0.0, atol=1e-6)
