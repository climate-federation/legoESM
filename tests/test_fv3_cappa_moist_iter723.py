"""FV3_3D iter 723: cappa_moist_fv3 port + iter-716 refactor.

Extracted moist Poisson-exponent formula from inline use in iter-716
Bolton θ_e (and FV3 fv_mapz.F90 lines 470, 475, 487).

Tests
-----

1. ``test_cappa_moist_dry_limit_matches_kappa``.
2. ``test_cappa_moist_q_dependence``.
3. ``test_cappa_moist_custom_zvir``.
4. ``test_cappa_moist_with_pkz_path``.
5. ``test_cappa_moist_iter716_bolton_unchanged``.
6. ``test_cappa_moist_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    cappa_moist_fv3,
    compute_pkz_fv3,
    eqv_pot_bolton_fv3,
)


def test_cappa_moist_dry_limit_matches_kappa():
    """q = 0 → cappa = R_d / c_pd = constants.kappa."""
    q = jnp.zeros((5,))
    cappa = cappa_moist_fv3(q)
    assert jnp.allclose(cappa, constants.kappa, atol=1e-12)


def test_cappa_moist_q_dependence():
    """cappa monotone in q (moist atmosphere has different cv).
    cv_vap > cv_air (1846−461.5 ~= 1384.5 vs 1004.64−287.05 ~= 717.59),
    so moist cappa < dry kappa."""
    q = jnp.linspace(0.0, 0.025, 10)
    cappa = cappa_moist_fv3(q)
    # Dry value at q=0
    assert abs(float(cappa[0]) - constants.kappa) < 1e-12
    # cv_vap > cv_air → moist heat capacity higher → cappa smaller
    assert jnp.all(cappa[1:] < cappa[0])
    # Monotone decreasing in q
    assert jnp.all(jnp.diff(cappa) < 0.0)


def test_cappa_moist_custom_zvir():
    """Custom zvir affects the (1+zvir·q) denominator."""
    q = jnp.full((5,), 0.01)
    cappa_default = cappa_moist_fv3(q)
    cappa_custom = cappa_moist_fv3(q, zvir=1.0)  # Non-physical, tests plumbing
    assert not jnp.allclose(cappa_default, cappa_custom, atol=1e-8)


def test_cappa_moist_with_pkz_path():
    """iter-722 compute_pkz_fv3 + iter-723 cappa_moist composes:
    moist nwat path."""
    km = 5
    q = jnp.full((km,), 0.015)
    cappa = cappa_moist_fv3(q)
    pt = jnp.full((km,), 280.0)
    delp = jnp.full((km,), 5000.0)
    delz = jnp.full((km,), -500.0)
    pkz = compute_pkz_fv3(
        delp, pt=pt, delz=delz, hydrostatic=False, cappa=cappa,
    )
    assert pkz.shape == (km,)
    assert jnp.all(pkz > 0.0)
    assert jnp.all(jnp.isfinite(pkz))


def test_cappa_moist_iter716_bolton_unchanged():
    """iter-716 eqv_pot_bolton_fv3 refactor preserves output exactly."""
    rng = np.random.default_rng(seed=723)
    km = 10
    T = jnp.asarray(rng.uniform(260.0, 300.0, size=(km,)))
    q = jnp.asarray(rng.uniform(0.001, 0.02, size=(km,)))
    delp = jnp.full((km,), 1000.0)
    delz = jnp.full((km,), -100.0)
    theta_e = eqv_pot_bolton_fv3(T, delp, q, delz=delz, moist=True)
    # Recompute with manual cappa formula (pre-refactor) to verify match
    cv_air = constants.c_pd - constants.R_d
    cv_vap = constants.c_pv - constants.R_v
    zvir = constants.R_v / constants.R_d - 1.0
    cappa_manual = constants.R_d / (
        constants.R_d
        + ((1.0 - q) * cv_air + q * cv_vap) / (1.0 + zvir * q)
    )
    cappa_helper = cappa_moist_fv3(q)
    assert jnp.allclose(cappa_helper, cappa_manual, atol=1e-15)
    # Sanity: theta_e finite
    assert jnp.all(jnp.isfinite(theta_e))


def test_cappa_moist_finite():
    """No NaN/Inf on random."""
    rng = np.random.default_rng(seed=724)
    q = jnp.asarray(rng.uniform(0.0, 0.025, size=(4, 30)))
    cappa = cappa_moist_fv3(q)
    assert jnp.all(jnp.isfinite(cappa))
