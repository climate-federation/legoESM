"""FV3_3D iter 713: moist_cv_fv3 port + nh_total_energy_fv3 PPME upgrade.

Faithful JAX port of FV3 ``moist_cv`` (model/fv_mapz.F90:3579-3654)
+ opt-in path in iter-693 ``nh_total_energy_fv3``.

Tests
-----

1. ``test_moist_cv_dry_returns_cv_air``.
2. ``test_moist_cv_pure_vapor``.
3. ``test_moist_cv_with_condensate``.
4. ``test_moist_cv_q_con_correct``.
5. ``test_moist_cv_requires_input``.
6. ``test_nh_te_moist_cv_path_finite``.
7. ``test_nh_te_default_unchanged_iter693``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import moist_cv_fv3, nh_total_energy_fv3


def test_moist_cv_dry_returns_cv_air():
    """All tracers zero → cvm = cv_air."""
    n = 10
    q_sphum = jnp.zeros((n,))
    cvm, q_con = moist_cv_fv3(q_sphum=q_sphum)
    cv_air = constants.c_pd - constants.R_d
    assert jnp.all(jnp.abs(cvm - cv_air) < 1e-10)
    assert jnp.all(q_con == 0.0)


def test_moist_cv_pure_vapor():
    """q_sphum=1, rest=0 → cvm = cv_vap."""
    n = 5
    q_sphum = jnp.ones((n,))
    cvm, q_con = moist_cv_fv3(q_sphum=q_sphum)
    cv_vap = constants.c_pv - constants.R_v
    assert jnp.all(jnp.abs(cvm - cv_vap) < 1e-10)
    assert jnp.all(q_con == 0.0)


def test_moist_cv_with_condensate():
    """q_v=0.01, q_l=0.005, q_i=0.002.
    Expected:
        cv_air * (1 - 0.01 - 0.007)
        + cv_vap * 0.01
        + c_pw * 0.005
        + c_pi * 0.002
    """
    q_sphum = jnp.array([0.01])
    q_liq = jnp.array([0.005])
    q_ice = jnp.array([0.002])
    cvm, q_con = moist_cv_fv3(
        q_sphum=q_sphum, q_liq_wat=q_liq, q_ice_wat=q_ice,
    )
    cv_air = constants.c_pd - constants.R_d
    cv_vap = constants.c_pv - constants.R_v
    expected_cvm = (
        cv_air * (1 - 0.01 - 0.007)
        + cv_vap * 0.01
        + constants.c_pw * 0.005
        + constants.c_pi * 0.002
    )
    assert abs(float(cvm[0]) - expected_cvm) < 1e-8
    assert abs(float(q_con[0]) - 0.007) < 1e-12


def test_moist_cv_q_con_correct():
    """q_con = liq + rain + ice + snow + graupel."""
    q_sphum = jnp.array([0.01])
    q_liq = jnp.array([0.001])
    q_rain = jnp.array([0.002])
    q_ice = jnp.array([0.0005])
    q_snow = jnp.array([0.0003])
    q_grp = jnp.array([0.0001])
    _, q_con = moist_cv_fv3(
        q_sphum=q_sphum, q_liq_wat=q_liq, q_rainwat=q_rain,
        q_ice_wat=q_ice, q_snowwat=q_snow, q_graupel=q_grp,
    )
    expected = 0.001 + 0.002 + 0.0005 + 0.0003 + 0.0001
    assert abs(float(q_con[0]) - expected) < 1e-12


def test_moist_cv_requires_input():
    """No tracers → raises."""
    with pytest.raises(ValueError):
        moist_cv_fv3()


def test_nh_te_moist_cv_path_finite():
    """iter-693 nh_total_energy_fv3 use_moist_cv=True branch finite."""
    rng = np.random.default_rng(seed=713)
    km = 30
    pt = jnp.asarray(rng.uniform(220, 300, size=(4, 4, km)))
    delp = jnp.full((4, 4, km), 3000.0)
    delz = jnp.full((4, 4, km), -300.0)
    hs = jnp.zeros((4, 4))
    ua = jnp.asarray(rng.normal(scale=10.0, size=(4, 4, km)))
    va = jnp.asarray(rng.normal(scale=10.0, size=(4, 4, km)))
    w = jnp.zeros((4, 4, km))
    q_sphum = jnp.asarray(rng.uniform(1e-6, 0.02, size=(4, 4, km)))
    q_liq = jnp.asarray(rng.uniform(0.0, 5e-4, size=(4, 4, km)))
    q_ice = jnp.asarray(rng.uniform(0.0, 5e-4, size=(4, 4, km)))
    te = nh_total_energy_fv3(
        ua, va, w, pt, delp, delz, hs,
        q_sphum=q_sphum, q_liq_wat=q_liq, q_ice_wat=q_ice,
        moist_phys=True, use_moist_cv=True,
    )
    assert jnp.all(jnp.isfinite(te))


def test_nh_te_default_unchanged_iter693():
    """Default use_moist_cv=False keeps iter-693 dry-cv path."""
    rng = np.random.default_rng(seed=714)
    km = 10
    pt = jnp.full((km,), 280.0)
    delp = jnp.full((km,), 5000.0)
    delz = jnp.full((km,), -500.0)
    hs = jnp.asarray(0.0)
    ua = jnp.zeros((km,))
    va = jnp.zeros((km,))
    w = jnp.zeros((km,))
    q_sphum = jnp.full((km,), 0.01)
    te_default = nh_total_energy_fv3(
        ua, va, w, pt, delp, delz, hs,
        q_sphum=q_sphum, moist_phys=True,
    )
    te_explicit = nh_total_energy_fv3(
        ua, va, w, pt, delp, delz, hs,
        q_sphum=q_sphum, moist_phys=True, use_moist_cv=False,
    )
    assert abs(float(te_default) - float(te_explicit)) < 1e-10
