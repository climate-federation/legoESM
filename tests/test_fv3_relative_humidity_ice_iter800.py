"""FV3_3D iter 800: relative_humidity_ice_fv3.

RH_ice [%] = 100 · q / q_sat_ice(T, p).

Composes canonical thermo.saturation_mixing_ratio_ice.

Tests
-----

1. ``test_rh_ice_zero_q``: q=0 → RH_ice=0.
2. ``test_rh_ice_supersaturation``: q > q_sat_ice → RH_ice > 100.
3. ``test_rh_ice_below_freezing_higher_than_liquid``: T<T_freeze
   gives RH_ice > RH_liquid at same q.
4. ``test_rh_ice_monotonic_q``: ↑q → ↑RH_ice.
5. ``test_rh_ice_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import thermo
from legoesm.grids.cubed_sphere import (
    relative_humidity_fv3,
    relative_humidity_ice_fv3,
)


def test_rh_ice_zero_q():
    """q=0 → RH_ice=0."""
    t = jnp.array([240.0, 250.0, 260.0])
    p = jnp.full((3,), 50_000.0)
    q = jnp.zeros((3,))
    rh_ice = relative_humidity_ice_fv3(t, p, q)
    np.testing.assert_allclose(np.asarray(rh_ice), jnp.zeros((3,)), atol=1e-15)


def test_rh_ice_supersaturation():
    """q = 1.5·q_sat_ice → RH_ice ≈ 150%."""
    t = jnp.array([220.0])
    p = jnp.array([30_000.0])
    q_sat_ice = thermo.saturation_mixing_ratio_ice(t, p)
    q = 1.5 * q_sat_ice
    rh_ice = relative_humidity_ice_fv3(t, p, q)
    np.testing.assert_allclose(np.asarray(rh_ice), [150.0], rtol=1e-12)


def test_rh_ice_below_freezing_higher_than_liquid():
    """Below freezing: RH_ice > RH_liquid at same q (e_sat_liq > e_sat_ice)."""
    t = jnp.array([250.0])  # below freezing
    p = jnp.array([70_000.0])
    q = jnp.array([0.001])
    rh_ice = relative_humidity_ice_fv3(t, p, q)
    rh_liq = relative_humidity_fv3(t, p, q)
    assert float(rh_ice[0]) > float(rh_liq[0])


def test_rh_ice_monotonic_q():
    """↑q → ↑RH_ice at fixed (T, p)."""
    t = jnp.full((4,), 240.0)
    p = jnp.full((4,), 50_000.0)
    q_lo = jnp.array([1e-5, 5e-5, 1e-4, 5e-4])
    q_hi = q_lo * 2.0
    rh_lo = relative_humidity_ice_fv3(t, p, q_lo)
    rh_hi = relative_humidity_ice_fv3(t, p, q_hi)
    assert jnp.all(rh_hi > rh_lo)


def test_rh_ice_shapes_3d_finite():
    """3-D shapes preserved, finite, non-negative for q ≥ 0."""
    rng = np.random.default_rng(seed=800)
    n_x, n_y, km = 4, 5, 20
    t = jnp.asarray(rng.uniform(200.0, 270.0, size=(n_x, n_y, km)))
    p = jnp.asarray(rng.uniform(10_000.0, 80_000.0, size=(n_x, n_y, km)))
    q = jnp.asarray(rng.uniform(0.0, 0.005, size=(n_x, n_y, km)))
    rh_ice = relative_humidity_ice_fv3(t, p, q)
    assert rh_ice.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(rh_ice))
    assert jnp.all(rh_ice >= 0.0)
