"""FV3_3D iter 730: dz_from_delz_or_hydrostatic_fv3 helper + 5-iter refactor.

Extracted common dz reconstruction pattern from iters 686/687/688/
689/707 (supercell suite).

Tests
-----

1. ``test_dz_non_hydro_negates_delz``.
2. ``test_dz_hydrostatic_isothermal_known_value``.
3. ``test_dz_hydrostatic_virtual_T_factor``.
4. ``test_dz_iter686_through_707_unchanged``.
5. ``test_dz_missing_args_raise``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    bunkers_vector_fv3,
    compute_brn_fv3,
    dz_from_delz_or_hydrostatic_fv3,
    helicity_relative_caps_fv3,
    helicity_relative_fv3,
    updraft_helicity_fv3,
)


def test_dz_non_hydro_negates_delz():
    """Non-hydrostatic: dz = -delz."""
    delz = jnp.full((5,), -500.0)
    dz = dz_from_delz_or_hydrostatic_fv3(delz=delz, hydrostatic=False)
    assert jnp.allclose(dz, 500.0, atol=1e-12)


def test_dz_hydrostatic_isothermal_known_value():
    """Hydrostatic isothermal:
        dz = (R_d/g) · T · (1+zvir·q) · Δpeln
    Dry q=0: dz = (R_d/g) · T · ln(p_bot/p_top)
    """
    pt = jnp.full((1,), 280.0)
    q = jnp.zeros((1,))
    pe = jnp.array([9.0e4, 1.0e5])
    peln = jnp.log(pe)
    dz = dz_from_delz_or_hydrostatic_fv3(
        pt=pt, q=q, peln=peln, hydrostatic=True,
    )
    expected = constants.R_d / constants.g * 280.0 * jnp.log(1.0e5 / 9.0e4)
    assert abs(float(dz[0]) - expected) / expected < 1e-10


def test_dz_hydrostatic_virtual_T_factor():
    """Moist q > 0 → dz factor of (1+zvir·q) compared to dry."""
    pt = jnp.full((1,), 290.0)
    q_moist = jnp.full((1,), 0.020)
    q_dry = jnp.zeros((1,))
    pe = jnp.array([9.0e4, 1.0e5])
    peln = jnp.log(pe)
    dz_moist = dz_from_delz_or_hydrostatic_fv3(
        pt=pt, q=q_moist, peln=peln, hydrostatic=True,
    )
    dz_dry = dz_from_delz_or_hydrostatic_fv3(
        pt=pt, q=q_dry, peln=peln, hydrostatic=True,
    )
    zvir = constants.R_v / constants.R_d - 1.0
    expected_ratio = 1.0 + zvir * 0.020
    actual_ratio = float(dz_moist[0]) / float(dz_dry[0])
    assert abs(actual_ratio - expected_ratio) < 1e-12


def test_dz_iter686_through_707_unchanged():
    """All 5 supercell-suite functions still produce finite output
    after refactor (regression sanity)."""
    rng = np.random.default_rng(seed=730)
    km = 20
    delz = jnp.full((km,), -500.0)
    ua = jnp.asarray(rng.normal(scale=10.0, size=(km,)))
    va = jnp.asarray(rng.normal(scale=10.0, size=(km,)))
    vort = jnp.asarray(rng.normal(scale=0.01, size=(km,)))
    w = jnp.asarray(rng.uniform(0.0, 5.0, size=(km,)))
    delp = jnp.full((km,), 1000.0)
    cape = jnp.asarray(2000.0)
    uc = jnp.asarray(5.0)
    vc = jnp.asarray(-2.0)
    assert jnp.isfinite(updraft_helicity_fv3(vort, w, delz=delz))
    assert jnp.isfinite(helicity_relative_fv3(ua, va, delz=delz))
    brn, shear = compute_brn_fv3(ua, va, delp, delz, cape)
    assert jnp.isfinite(brn) and jnp.isfinite(shear)
    uc_b, vc_b = bunkers_vector_fv3(ua, va, delz=delz)
    assert jnp.isfinite(uc_b) and jnp.isfinite(vc_b)
    assert jnp.isfinite(helicity_relative_caps_fv3(ua, va, uc, vc, delz=delz))


def test_dz_missing_args_raise():
    """Missing delz (non-hydro) or pt/q/peln (hydrostatic) raises."""
    with pytest.raises(ValueError):
        dz_from_delz_or_hydrostatic_fv3(hydrostatic=False)
    with pytest.raises(ValueError):
        dz_from_delz_or_hydrostatic_fv3(hydrostatic=True)
