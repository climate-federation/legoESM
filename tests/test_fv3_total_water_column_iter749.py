"""FV3_3D iter 749: total_water_column_fv3 port.

TWC = sum_species column_integral(q_species).

Tests
-----

1. ``test_twc_single_sphum_matches_iter746``.
2. ``test_twc_multi_species_sum``.
3. ``test_twc_no_tracers_raises``.
4. ``test_twc_shapes_3d``.
5. ``test_twc_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    column_integral_delp_fv3,
    total_water_column_fv3,
)


def test_twc_single_sphum_matches_iter746():
    """Single q_sphum → TWC = column_integral(q_sphum)."""
    km = 10
    q = jnp.full((km,), 0.01)
    delp = jnp.full((km,), 1.0e4)
    twc = total_water_column_fv3(delp, q_sphum=q)
    expected = column_integral_delp_fv3(q, delp)
    assert abs(float(twc) - float(expected)) < 1e-12


def test_twc_multi_species_sum():
    """q_v=0.01, q_l=0.005, q_i=0.002 → TWC = (0.017)·p_s/g."""
    km = 5
    delp = jnp.full((km,), 2.0e4)   # p_s = 1e5
    q_v = jnp.full((km,), 0.01)
    q_l = jnp.full((km,), 0.005)
    q_i = jnp.full((km,), 0.002)
    twc = total_water_column_fv3(
        delp, q_sphum=q_v, q_liq_wat=q_l, q_ice_wat=q_i,
    )
    expected = 0.017 * 1.0e5 / constants.g
    assert abs(float(twc) - expected) / expected < 1e-10


def test_twc_no_tracers_raises():
    """No tracers → raises."""
    km = 5
    delp = jnp.full((km,), 1.0e4)
    with pytest.raises(ValueError):
        total_water_column_fv3(delp)


def test_twc_shapes_3d():
    """3-D → 2-D output."""
    rng = np.random.default_rng(seed=749)
    n_x, n_y, km = 4, 5, 20
    q = jnp.asarray(rng.uniform(0.0, 0.02, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    twc = total_water_column_fv3(delp, q_sphum=q)
    assert twc.shape == (n_x, n_y)


def test_twc_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=750)
    q_v = jnp.asarray(rng.uniform(0.0, 0.025, size=(4, 4, 30)))
    q_l = jnp.asarray(rng.uniform(0.0, 0.005, size=(4, 4, 30)))
    q_i = jnp.asarray(rng.uniform(0.0, 0.002, size=(4, 4, 30)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(4, 4, 30)))
    twc = total_water_column_fv3(
        delp, q_sphum=q_v, q_liq_wat=q_l, q_ice_wat=q_i,
    )
    assert jnp.all(jnp.isfinite(twc))
