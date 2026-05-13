"""FV3_3D iter 818: effective_bulk_shear_fv3.

Δu = u(z_top) − u(z_bot); Δv = v(z_top) − v(z_bot)  via jnp.interp.

Tests
-----

1. ``test_ebs_linear_u``: linearly-increasing u → Δu = (Δz)·slope.
2. ``test_ebs_zero_v``: v=0 everywhere → Δv = 0.
3. ``test_ebs_nan_propagates``: NaN bounds → NaN output.
4. ``test_ebs_edge_clamps``: bounds outside column range clamp.
5. ``test_ebs_compose_iter817``: chain (CAPE, |CIN|, z) → (z_bot, z_top) → EBWD.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    effective_bulk_shear_fv3,
    effective_inflow_layer_fv3,
)


def test_ebs_linear_u():
    """u(z) = 0.01·z → Δu = 0.01·(z_top − z_bot)."""
    z = jnp.linspace(0.0, 10_000.0, 21)
    u = 0.01 * z
    v = jnp.zeros_like(z)
    du, dv = effective_bulk_shear_fv3(u, v, z, 1000.0, 5000.0)
    np.testing.assert_allclose(np.asarray(du), 0.01 * 4000.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(dv), 0.0, atol=1e-15)


def test_ebs_zero_v():
    """v=0 → Δv=0 regardless of u."""
    z = jnp.linspace(0.0, 10_000.0, 21)
    u = jnp.linspace(0.0, 50.0, 21)
    v = jnp.zeros_like(z)
    du, dv = effective_bulk_shear_fv3(u, v, z, 2000.0, 8000.0)
    np.testing.assert_allclose(np.asarray(dv), 0.0, atol=1e-15)


def test_ebs_nan_propagates():
    """NaN z_bot or z_top → NaN output."""
    z = jnp.linspace(0.0, 10_000.0, 11)
    u = jnp.linspace(0.0, 30.0, 11)
    v = jnp.linspace(-5.0, 15.0, 11)
    du, dv = effective_bulk_shear_fv3(u, v, z, jnp.nan, 5000.0)
    assert jnp.isnan(du)
    assert jnp.isnan(dv)


def test_ebs_edge_clamps():
    """Bounds outside column clamp to edge values."""
    z = jnp.linspace(1000.0, 5000.0, 5)
    u = jnp.array([10.0, 12.0, 14.0, 16.0, 18.0])
    v = jnp.zeros_like(z)
    # z_bot below column min, z_top above column max → both clamp
    du, dv = effective_bulk_shear_fv3(u, v, z, 0.0, 10_000.0)
    # u(0.0) → u[0] = 10; u(10000) → u[-1] = 18 → Δu = 8
    np.testing.assert_allclose(np.asarray(du), 8.0, rtol=1e-12)


def test_ebs_compose_iter817():
    """Full chain: (CAPE, |CIN|, z) → (z_bot, z_top) → EBWD."""
    z = jnp.array([100.0, 500.0, 1500.0, 3000.0, 6000.0, 10_000.0])
    cape = jnp.array([50.0, 80.0, 500.0, 1000.0, 800.0, 50.0])
    cin = jnp.array([300.0, 280.0, 100.0, 50.0, 80.0, 400.0])
    u = jnp.array([5.0, 8.0, 12.0, 18.0, 25.0, 35.0])
    v = jnp.array([0.0, 1.0, 3.0, 5.0, 7.0, 10.0])
    z_bot, z_top = effective_inflow_layer_fv3(cape, cin, z)
    du, dv = effective_bulk_shear_fv3(u, v, z, z_bot, z_top)
    # EIL = z[2..4] = (1500, 6000); Δu = 25 − 12 = 13; Δv = 7 − 3 = 4
    np.testing.assert_allclose(np.asarray(du), 13.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(dv), 4.0, rtol=1e-12)
