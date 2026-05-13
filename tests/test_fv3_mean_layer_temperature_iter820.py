"""FV3_3D iter 820: mean_layer_temperature_fv3.

T̄ = Σ T_mid·Δz·mask / Σ Δz·mask  over [z_bot, z_top].

Scalar analog of iter-819 ``mean_wind_layer_fv3``.

Tests
-----

1. ``test_T_uniform``: T=const → T̄=const.
2. ``test_T_linear``: linear T(z) → midpoint analytic.
3. ``test_T_layer_bounds``: bounds outside column floored.
4. ``test_T_compose_eil``: chain with iter-817 EIL → mean-T over EIL.
5. ``test_T_finite_typical``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    effective_inflow_layer_fv3,
    mean_layer_temperature_fv3,
)


def test_T_uniform():
    """T constant → T̄=T."""
    z = jnp.linspace(0.0, 10_000.0, 21)
    t = jnp.full((21,), 288.0)
    t_mean = mean_layer_temperature_fv3(t, z, 1000.0, 5000.0)
    np.testing.assert_allclose(np.asarray(t_mean), 288.0, rtol=1e-10)


def test_T_linear():
    """T(z) = 288 - 6.5e-3·z (ICAO lapse). Mean over [0, 10000] = 288 - 6.5e-3·5000 = 255.5."""
    z = jnp.linspace(0.0, 10_000.0, 41)
    t = 288.0 - 6.5e-3 * z
    t_mean = mean_layer_temperature_fv3(t, z, 0.0, 10_000.0)
    np.testing.assert_allclose(np.asarray(t_mean), 255.5, rtol=0.01)


def test_T_layer_bounds():
    """Bounds entirely below column → 0 (via floor)."""
    z = jnp.linspace(1000.0, 5000.0, 5)
    t = jnp.array([285.0, 280.0, 275.0, 270.0, 265.0])
    t_mean = mean_layer_temperature_fv3(t, z, 0.0, 500.0)
    assert jnp.isfinite(t_mean)
    # No midpoints in [0, 500], output = 0/floor ≈ 0
    np.testing.assert_allclose(np.asarray(t_mean), 0.0, atol=1e-6)


def test_T_compose_eil():
    """EIL → mean-T over EIL."""
    z = jnp.array([100.0, 500.0, 1500.0, 3000.0, 6000.0, 10_000.0])
    cape = jnp.array([50.0, 80.0, 500.0, 1000.0, 800.0, 50.0])
    cin = jnp.array([300.0, 280.0, 100.0, 50.0, 80.0, 400.0])
    t = jnp.array([295.0, 290.0, 280.0, 265.0, 245.0, 220.0])
    z_bot, z_top = effective_inflow_layer_fv3(cape, cin, z)
    t_mean = mean_layer_temperature_fv3(t, z, z_bot, z_top)
    # EIL (1500, 6000) mid-T plausible range
    assert 240.0 < float(t_mean) < 285.0


def test_T_finite_typical():
    """3-D-like sounding, sensible T̄."""
    z = jnp.linspace(0.0, 15_000.0, 31)
    t = 288.0 - 6.5e-3 * z
    t = jnp.maximum(t, 217.0)  # tropopause flat
    t_mean = mean_layer_temperature_fv3(t, z, 1000.0, 9000.0)
    assert jnp.isfinite(t_mean)
    assert 210.0 < float(t_mean) < 290.0
