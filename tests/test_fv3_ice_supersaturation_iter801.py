"""FV3_3D iter 801: ice_supersaturation_fv3 (RH_ice > thresh).

Composes iter-800 ``relative_humidity_ice_fv3``.

Tests
-----

1. ``test_iss_below_threshold``: RH_ice = 100% → False (no ISS).
2. ``test_iss_above_threshold``: RH_ice = 150% → True with default 140.
3. ``test_iss_custom_threshold``: 160% threshold rejects 150%.
4. ``test_iss_zero_q``: q=0 → False everywhere.
5. ``test_iss_composes_iter800``: mask alignment with RH_ice.
6. ``test_iss_shapes_3d_bool``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import thermo
from legoesm.grids.cubed_sphere import (
    ice_supersaturation_fv3,
    relative_humidity_ice_fv3,
)


def test_iss_below_threshold():
    """RH_ice = 100% (q=q_sat_ice) → ISS False with default 140%."""
    t = jnp.array([240.0])
    p = jnp.array([30_000.0])
    q = thermo.saturation_mixing_ratio_ice(t, p)
    iss = ice_supersaturation_fv3(t, p, q)
    assert bool(iss[0]) is False


def test_iss_above_threshold():
    """RH_ice = 150% (q = 1.5·q_sat_ice) → ISS True at default 140."""
    t = jnp.array([220.0])
    p = jnp.array([20_000.0])
    q = 1.5 * thermo.saturation_mixing_ratio_ice(t, p)
    iss = ice_supersaturation_fv3(t, p, q)
    assert bool(iss[0]) is True


def test_iss_custom_threshold():
    """RH_ice = 150% with thresh=160 → False."""
    t = jnp.array([220.0])
    p = jnp.array([20_000.0])
    q = 1.5 * thermo.saturation_mixing_ratio_ice(t, p)
    iss = ice_supersaturation_fv3(t, p, q, rh_thresh=160.0)
    assert bool(iss[0]) is False


def test_iss_zero_q():
    """q=0 → ISS False everywhere."""
    t = jnp.array([220.0, 240.0, 260.0])
    p = jnp.full((3,), 30_000.0)
    q = jnp.zeros((3,))
    iss = ice_supersaturation_fv3(t, p, q)
    assert jnp.all(~iss)


def test_iss_composes_iter800():
    """Mask aligns with RH_ice > thresh."""
    rng = np.random.default_rng(seed=801)
    t = jnp.asarray(rng.uniform(200.0, 270.0, size=(20,)))
    p = jnp.asarray(rng.uniform(10_000.0, 50_000.0, size=(20,)))
    q = jnp.asarray(rng.uniform(0.0, 0.005, size=(20,)))
    rh_ice = relative_humidity_ice_fv3(t, p, q)
    iss = ice_supersaturation_fv3(t, p, q, rh_thresh=120.0)
    expected = rh_ice > 120.0
    np.testing.assert_array_equal(np.asarray(iss), np.asarray(expected))


def test_iss_shapes_3d_bool():
    """3-D shapes preserved, dtype bool."""
    rng = np.random.default_rng(seed=802)
    n_x, n_y, km = 4, 5, 20
    t = jnp.asarray(rng.uniform(200.0, 270.0, size=(n_x, n_y, km)))
    p = jnp.asarray(rng.uniform(10_000.0, 50_000.0, size=(n_x, n_y, km)))
    q = jnp.asarray(rng.uniform(0.0, 0.005, size=(n_x, n_y, km)))
    iss = ice_supersaturation_fv3(t, p, q)
    assert iss.shape == (n_x, n_y, km)
    assert iss.dtype == jnp.bool_
