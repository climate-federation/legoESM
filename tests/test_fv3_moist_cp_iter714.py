"""FV3_3D iter 714: moist_cp_fv3 port.

Faithful JAX port of FV3 ``moist_cp`` (general nwat>=3 branch)
(model/fv_mapz.F90:3656-3733).  Companion to iter-713 moist_cv.

Tests
-----

1. ``test_moist_cp_dry_returns_c_pd``.
2. ``test_moist_cp_pure_vapor``.
3. ``test_moist_cp_with_condensate``.
4. ``test_moist_cp_cp_minus_cv_equals_R_dry``.
5. ``test_moist_cp_q_con_correct``.
6. ``test_moist_cp_requires_input``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import moist_cp_fv3, moist_cv_fv3


def test_moist_cp_dry_returns_c_pd():
    """All tracers zero → cpm = c_pd."""
    q_sphum = jnp.zeros((5,))
    cpm, q_con = moist_cp_fv3(q_sphum=q_sphum)
    assert jnp.all(jnp.abs(cpm - constants.c_pd) < 1e-10)
    assert jnp.all(q_con == 0.0)


def test_moist_cp_pure_vapor():
    """q_sphum=1, rest=0 → cpm = c_pv."""
    q_sphum = jnp.ones((5,))
    cpm, q_con = moist_cp_fv3(q_sphum=q_sphum)
    assert jnp.all(jnp.abs(cpm - constants.c_pv) < 1e-10)
    assert jnp.all(q_con == 0.0)


def test_moist_cp_with_condensate():
    """q_v=0.01, q_l=0.005, q_i=0.002 → analytical cpm."""
    q_sphum = jnp.array([0.01])
    q_liq = jnp.array([0.005])
    q_ice = jnp.array([0.002])
    cpm, q_con = moist_cp_fv3(
        q_sphum=q_sphum, q_liq_wat=q_liq, q_ice_wat=q_ice,
    )
    expected = (
        constants.c_pd * (1 - 0.01 - 0.007)
        + constants.c_pv * 0.01
        + constants.c_pw * 0.005
        + constants.c_pi * 0.002
    )
    assert abs(float(cpm[0]) - expected) < 1e-8
    assert abs(float(q_con[0]) - 0.007) < 1e-12


def test_moist_cp_cp_minus_cv_equals_R_dry():
    """Dry case: cpm − cvm = c_pd − (c_pd−R_d) = R_d."""
    q_sphum = jnp.zeros((5,))
    cpm, _ = moist_cp_fv3(q_sphum=q_sphum)
    cvm, _ = moist_cv_fv3(q_sphum=q_sphum)
    assert jnp.all(jnp.abs((cpm - cvm) - constants.R_d) < 1e-10)


def test_moist_cp_q_con_correct():
    """q_con = sum of all condensate species."""
    q_sphum = jnp.array([0.01])
    q_liq = jnp.array([0.001])
    q_rain = jnp.array([0.002])
    q_ice = jnp.array([0.0005])
    q_snow = jnp.array([0.0003])
    q_grp = jnp.array([0.0001])
    _, q_con = moist_cp_fv3(
        q_sphum=q_sphum, q_liq_wat=q_liq, q_rainwat=q_rain,
        q_ice_wat=q_ice, q_snowwat=q_snow, q_graupel=q_grp,
    )
    expected = 0.001 + 0.002 + 0.0005 + 0.0003 + 0.0001
    assert abs(float(q_con[0]) - expected) < 1e-12


def test_moist_cp_requires_input():
    """No tracers → raises."""
    with pytest.raises(ValueError):
        moist_cp_fv3()
