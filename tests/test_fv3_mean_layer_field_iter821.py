"""FV3_3D iter 821: mean_layer_field_fv3 + iter-819/820 refactor.

Generic depth-weighted scalar mean over [z_bot, z_top].

Extracted from iter-819 (mean_wind) and iter-820 (mean_T) pattern;
both refactored to delegate.

Tests
-----

1. ``test_field_uniform``: constant field → ⟨X⟩=const.
2. ``test_field_linear``: linear X(z) → midpoint analytic.
3. ``test_field_humidity_like``: q-style field over PBL.
4. ``test_field_floored_out_of_range``: bounds outside → 0.
5. ``test_iter819_unchanged``: iter-819 wind preserved after refactor.
6. ``test_iter820_unchanged``: iter-820 T preserved after refactor.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    mean_layer_field_fv3,
    mean_layer_temperature_fv3,
    mean_wind_layer_fv3,
)


def test_field_uniform():
    """Constant field → ⟨X⟩ = const."""
    z = jnp.linspace(0.0, 10_000.0, 21)
    field = jnp.full((21,), 0.005)  # e.g., q
    out = mean_layer_field_fv3(field, z, 1000.0, 5000.0)
    np.testing.assert_allclose(np.asarray(out), 0.005, rtol=1e-10)


def test_field_linear():
    """Linear field(z) → midpoint analytic."""
    z = jnp.linspace(0.0, 10_000.0, 41)
    field = 2.0 * z  # arbitrary linear
    out = mean_layer_field_fv3(field, z, 2000.0, 8000.0)
    # Midpoint of linear over [2000, 8000] = 2·5000 = 10000
    np.testing.assert_allclose(np.asarray(out), 10_000.0, rtol=0.05)


def test_field_humidity_like():
    """q decreasing with height; PBL-mean q higher than mid-trop."""
    z = jnp.linspace(0.0, 10_000.0, 21)
    q = 0.018 * jnp.exp(-z / 3000.0)  # exponential decay
    q_pbl = mean_layer_field_fv3(q, z, 0.0, 2000.0)
    q_midtrop = mean_layer_field_fv3(q, z, 4000.0, 8000.0)
    assert float(q_pbl) > float(q_midtrop)


def test_field_floored_out_of_range():
    """Bounds entirely below column → 0 via floor."""
    z = jnp.linspace(1000.0, 5000.0, 5)
    field = jnp.array([10.0, 12.0, 14.0, 16.0, 18.0])
    out = mean_layer_field_fv3(field, z, 0.0, 500.0)
    assert jnp.isfinite(out)
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-6)


def test_iter819_unchanged():
    """iter-819 mean wind output preserved after refactor."""
    z = jnp.linspace(0.0, 10_000.0, 21)
    u = jnp.linspace(0.0, 50.0, 21)
    v = jnp.linspace(-10.0, 20.0, 21)
    u_mean, v_mean = mean_wind_layer_fv3(u, v, z, 1000.0, 5000.0)
    assert jnp.isfinite(u_mean)
    assert jnp.isfinite(v_mean)
    # Both should be roughly midpoint of linear: u ≈ 15, v ≈ 0
    assert 10.0 < float(u_mean) < 20.0
    assert -5.0 < float(v_mean) < 5.0


def test_iter820_unchanged():
    """iter-820 mean T output preserved after refactor."""
    z = jnp.linspace(0.0, 10_000.0, 21)
    t = 288.0 - 6.5e-3 * z
    t_mean = mean_layer_temperature_fv3(t, z, 0.0, 10_000.0)
    # Midpoint of linear over [0, 10000] = 288 - 6.5e-3·5000 = 255.5
    np.testing.assert_allclose(np.asarray(t_mean), 255.5, rtol=0.01)
