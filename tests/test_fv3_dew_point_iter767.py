"""FV3_3D iter 767: dew_point_fv3 (Bolton 1980 eq. 11).

T_d (°C) = 243.5 · γ / (17.67 − γ),  γ = ln(e/6.112)
T_d (K) = T_d (°C) + 273.15

Composes with iter-766 ``vapor_pressure_from_q_fv3``.

Tests
-----

1. ``test_tdew_saturation_at_freezing``: e=6.112 mb → T_d=T_freeze.
2. ``test_tdew_monotonic_in_e``: ∂T_d/∂e > 0.
3. ``test_tdew_bounded_below_T``: realistic e < e_sat(T) → T_d<T.
4. ``test_tdew_compose_iter766``: (p, q) → e → T_d chain works.
5. ``test_tdew_shapes_3d``.
6. ``test_tdew_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    dew_point_fv3,
    vapor_pressure_from_q_fv3,
)


def test_tdew_saturation_at_freezing():
    """e = 6.112 mb is the saturation vapor pressure at 0°C.

    By construction of Bolton eq. 11, T_d(6.112 mb) = 273.15 K exactly.
    """
    e = jnp.array([6.112])
    t_dew = dew_point_fv3(e)
    np.testing.assert_allclose(np.asarray(t_dew), [constants.T_freeze], atol=1e-10)


def test_tdew_monotonic_in_e():
    """∂T_d/∂e > 0."""
    e_lo = jnp.array([2.0, 5.0, 10.0, 30.0])
    e_hi = e_lo + 0.5
    t_lo = dew_point_fv3(e_lo)
    t_hi = dew_point_fv3(e_hi)
    assert jnp.all(t_hi > t_lo)


def test_tdew_bounded_below_T():
    """For unsaturated air, T_d < T.  Use realistic surface conditions."""
    p_mb = jnp.array([1000.0, 1000.0, 1000.0])
    q = jnp.array([0.001, 0.005, 0.010])  # increasingly moist
    e = vapor_pressure_from_q_fv3(p_mb, q)
    t_dew = dew_point_fv3(e)
    T_env = jnp.full((3,), 295.0)
    assert jnp.all(t_dew < T_env)
    # Wetter parcel → higher T_d
    assert t_dew[1] > t_dew[0]
    assert t_dew[2] > t_dew[1]


def test_tdew_compose_iter766():
    """(p, q) → e → T_d chain.  Realistic warm-moist surface parcel."""
    p_mb = jnp.array([1000.0])
    q = jnp.array([0.012])
    e = vapor_pressure_from_q_fv3(p_mb, q)
    t_dew = dew_point_fv3(e)
    # T_d typically 285-295 K for q~12 g/kg at sea level
    assert 282.0 < float(t_dew[0]) < 296.0


def test_tdew_shapes_3d():
    """3-D shapes preserved."""
    rng = np.random.default_rng(seed=767)
    n_x, n_y, km = 4, 5, 20
    p_mb = jnp.asarray(rng.uniform(100.0, 1000.0, size=(n_x, n_y, km)))
    q = jnp.asarray(rng.uniform(0.001, 0.020, size=(n_x, n_y, km)))
    e = vapor_pressure_from_q_fv3(p_mb, q)
    t_dew = dew_point_fv3(e)
    assert t_dew.shape == (n_x, n_y, km)


def test_tdew_finite():
    """No NaN/Inf for realistic atmospheric range."""
    rng = np.random.default_rng(seed=768)
    p_mb = jnp.asarray(rng.uniform(50.0, 1050.0, size=(8, 30)))
    q = jnp.asarray(rng.uniform(1e-6, 0.025, size=(8, 30)))
    e = vapor_pressure_from_q_fv3(p_mb, q)
    t_dew = dew_point_fv3(e)
    assert jnp.all(jnp.isfinite(t_dew))
