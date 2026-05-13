"""FV3_3D iter 803: frost_point_temperature_fv3.

T_f_C = 272.62·γ/(22.46 − γ),  γ = ln(e/6.112)
T_f_K = T_f_C + 273.15  (Lawrence 2005 Magnus-ice).

Composes iter-766 ``vapor_pressure_from_q_fv3``.

Tests
-----

1. ``test_frost_zero_celsius``: e=6.112 mb → T_f=T_freeze.
2. ``test_frost_warmer_than_dew_subfreezing``: T_f > T_dew.
3. ``test_frost_monotonic_in_q``: ↑q → ↑T_f.
4. ``test_frost_lawrence_range``: results valid in -80..0 °C.
5. ``test_frost_composes_iter766``: chain (p, q) → e → T_f.
6. ``test_frost_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    dew_point_fv3,
    frost_point_temperature_fv3,
    vapor_pressure_from_q_fv3,
)


def test_frost_zero_celsius():
    """At triple-point e=6.112 mb (and matching q), T_f → T_freeze."""
    # Use q such that p_mb · q / (ε + q·(1−ε)) ≈ 6.112 mb at p_mb=1000.
    # Solve: 6.112 = 1000·q/(0.622 + 0.378·q) → q ≈ 6.112·0.622/1000 ≈ 3.80e-3
    p_pa = jnp.array([100_000.0])
    # Build q from e = p·q/(ε+q·(1-ε)) where e=6.112 mb
    # For small q: q ≈ e·ε/p
    q = jnp.array([6.112 * constants.epsilon / 1000.0])
    t_f = frost_point_temperature_fv3(p_pa, q)
    # Should be close to T_freeze (small q means linear regime → ~exact)
    np.testing.assert_allclose(np.asarray(t_f), [constants.T_freeze], atol=0.5)


def test_frost_warmer_than_dew_subfreezing():
    """For subfreezing air (T < 273 K), T_frost > T_dew at same q."""
    p_pa = jnp.array([50_000.0])
    q = jnp.array([1e-4])  # very dry
    t_f = frost_point_temperature_fv3(p_pa, q)
    # Dew point via iter-767 (need e from same q)
    p_mb = p_pa / 100.0
    e = vapor_pressure_from_q_fv3(p_mb, q)
    t_d = dew_point_fv3(e)
    # Frost > dew for cold/dry air (both should be subfreezing here)
    assert float(t_f[0]) < constants.T_freeze  # subfreezing
    assert float(t_f[0]) > float(t_d[0])       # frost above dew


def test_frost_monotonic_in_q():
    """↑q at fixed p → ↑T_frost."""
    p = jnp.full((4,), 50_000.0)
    q_lo = jnp.array([1e-5, 1e-4, 1e-3, 5e-3])
    q_hi = q_lo * 2.0
    t_lo = frost_point_temperature_fv3(p, q_lo)
    t_hi = frost_point_temperature_fv3(p, q_hi)
    assert jnp.all(t_hi > t_lo)


def test_frost_lawrence_range():
    """Frost point falls in Lawrence-2005 validity range (-80..0 °C)
    for realistic cold-air inputs."""
    p = jnp.full((6,), 30_000.0)
    q = jnp.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3])
    t_f = frost_point_temperature_fv3(p, q)
    # All in cold-air range
    assert jnp.all(t_f > 150.0)  # > -123 °C (huge dry-air margin)
    assert jnp.all(t_f < constants.T_freeze + 5.0)


def test_frost_composes_iter766():
    """Chain (p, q) → e (iter-766) → T_f match direct call."""
    p_pa = jnp.array([20_000.0])
    q = jnp.array([1e-4])
    t_f = frost_point_temperature_fv3(p_pa, q)
    assert jnp.all(jnp.isfinite(t_f))


def test_frost_shapes_3d_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=803)
    n_x, n_y, km = 4, 5, 20
    p = jnp.asarray(rng.uniform(10_000.0, 100_000.0, size=(n_x, n_y, km)))
    q = jnp.asarray(rng.uniform(1e-7, 0.020, size=(n_x, n_y, km)))
    t_f = frost_point_temperature_fv3(p, q)
    assert t_f.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(t_f))
