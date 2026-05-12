"""FV3_3D iter 768: relative_humidity_fv3 port.

RH [%] = 100 · e(p,q) / e_sat(T)

Composes iter-766 ``vapor_pressure_from_q_fv3`` with shared
``thermo.saturation_vapor_pressure``.

Tests
-----

1. ``test_rh_saturated_q_eq_qsat``: q = q_sat → RH ≈ 100%.
2. ``test_rh_dry_q_zero``: q → 0 → RH → 0%.
3. ``test_rh_monotonic_in_q``: ∂RH/∂q > 0.
4. ``test_rh_decreases_with_T``: warmer T at same q → lower RH
   (e_sat increases).
5. ``test_rh_shapes_3d``.
6. ``test_rh_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import thermo
from legoesm.grids.cubed_sphere import relative_humidity_fv3


def test_rh_saturated_q_eq_qsat():
    """q ≈ q_sat → RH ≈ 100%.  Use saturation specific humidity directly."""
    T = jnp.array([280.0, 290.0, 300.0])
    p = jnp.array([90_000.0, 95_000.0, 101_325.0])
    q_sat = thermo.saturation_specific_humidity(T, p)
    rh = relative_humidity_fv3(T, p, q_sat)
    # Vapor-pressure RH vs q-ratio RH differ slightly due to
    # (1-q)/(1-q_sat) ≈ 1 approximation.  Tolerance 1%.
    np.testing.assert_allclose(np.asarray(rh), [100.0, 100.0, 100.0], atol=1.0)


def test_rh_dry_q_zero():
    """q → 0 → RH → 0%."""
    T = jnp.full((3,), 290.0)
    p = jnp.full((3,), 100_000.0)
    q = jnp.array([0.0, 1e-10, 1e-8])
    rh = relative_humidity_fv3(T, p, q)
    assert jnp.all(rh < 1.0)  # < 1% RH


def test_rh_monotonic_in_q():
    """∂RH/∂q > 0 at fixed (T, p)."""
    T = jnp.full((4,), 290.0)
    p = jnp.full((4,), 100_000.0)
    q_lo = jnp.array([0.001, 0.003, 0.005, 0.010])
    q_hi = q_lo + 0.001
    rh_lo = relative_humidity_fv3(T, p, q_lo)
    rh_hi = relative_humidity_fv3(T, p, q_hi)
    assert jnp.all(rh_hi > rh_lo)


def test_rh_decreases_with_T():
    """At fixed (p, q), RH decreases as T increases (e_sat ↑).

    Clausius-Clapeyron: warming air at fixed moisture lowers RH.
    """
    T_cold = jnp.full((3,), 280.0)
    T_warm = jnp.full((3,), 300.0)
    p = jnp.full((3,), 100_000.0)
    q = jnp.full((3,), 0.005)
    rh_cold = relative_humidity_fv3(T_cold, p, q)
    rh_warm = relative_humidity_fv3(T_warm, p, q)
    assert jnp.all(rh_warm < rh_cold)


def test_rh_shapes_3d():
    """3-D shapes preserved."""
    rng = np.random.default_rng(seed=768)
    n_x, n_y, km = 4, 5, 20
    T = jnp.asarray(rng.uniform(240.0, 300.0, size=(n_x, n_y, km)))
    p = jnp.asarray(rng.uniform(10_000.0, 100_000.0, size=(n_x, n_y, km)))
    q = jnp.asarray(rng.uniform(0.001, 0.020, size=(n_x, n_y, km)))
    rh = relative_humidity_fv3(T, p, q)
    assert rh.shape == (n_x, n_y, km)


def test_rh_finite():
    """No NaN/Inf for realistic atmospheric range."""
    rng = np.random.default_rng(seed=769)
    T = jnp.asarray(rng.uniform(220.0, 305.0, size=(8, 30)))
    p = jnp.asarray(rng.uniform(5_000.0, 105_000.0, size=(8, 30)))
    q = jnp.asarray(rng.uniform(1e-6, 0.025, size=(8, 30)))
    rh = relative_humidity_fv3(T, p, q)
    assert jnp.all(jnp.isfinite(rh))
    assert jnp.all(rh >= 0.0)
