"""Direct tests for the package DCMIP-2025 IC primitives module (audit item 9).

The reference-atmosphere profiles, per-case PARAMS, and squall-line sounding
were moved out of the test tree into ``legoesm.atmosphere.dynamics.gcm.dcmip2025_ic``
so the production spectral-NH initializers no longer import ``from tests...``.
These tests pin the physical properties of the moved primitives and verify the
test-tree re-exports are the SAME objects (single source of truth, no drift).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm import dcmip2025_ic as ic


def test_isothermal_theta_increases_upward():
    """For T=const, theta = T*(p0/p)^kappa increases monotonically upward."""
    theta_fn = ic.isothermal_theta_ref(T0=250.0)
    z = jnp.linspace(0.0, 30000.0, 60)
    theta = np.asarray(theta_fn(z))
    assert theta[0] == pytest_approx(250.0)  # theta = T at surface (p=p_ref)
    assert np.all(np.diff(theta) > 0.0)


def test_piecewise_lapse_temperature_structure():
    """Tropospheric T decreases (lapse<0), stratospheric T increases (lapse>0)
    across the tropopause; theta is positive and increases upward (stable)."""
    theta_fn = ic.piecewise_lapse_theta_ref(
        T_s=300.0, lapse_tropo=-5.0e-3, lapse_strato=5.0e-3, z_tropopause=20000.0,
    )
    z = jnp.linspace(0.0, 35000.0, 80)
    theta = np.asarray(theta_fn(z))
    assert np.all(theta > 0.0)
    # Stable stratification overall (theta increases upward).
    assert np.all(np.diff(theta) > 0.0)


def test_squall_sounding_hydrostatic_and_tropopause():
    """Squall sounding: T decreases to the tropopause then is isothermal;
    pressure decreases monotonically upward (hydrostatic)."""
    z = jnp.linspace(0.0, ic.TC3_PARAMS["H"], 100)
    T, theta, p = ic.squall_line_sounding(z, ic.TC3_PARAMS)
    T = np.asarray(T); p = np.asarray(p)
    assert T[0] == pytest_approx(ic.TC3_PARAMS["T_s"])
    assert np.all(np.diff(p) < 0.0)               # pressure drops upward
    # Above the tropopause the profile is isothermal at T_tropopause.
    z_tr = ic.TC3_PARAMS["z_tropopause"]
    strat = np.asarray(z) > z_tr + 1000.0
    assert np.allclose(T[strat], ic.TC3_PARAMS["T_tropopause"], atol=1e-6)


def test_params_dicts_have_expected_keys():
    assert "u0" in ic.TC1_PARAMS and "mountain_height" in ic.TC1_PARAMS
    assert "small_earth_factor" in ic.TC2_PARAMS and "T0" in ic.TC2_PARAMS
    assert "n_bubbles" in ic.TC3_PARAMS and "RH_low" in ic.TC3_PARAMS


def test_test_tree_reexports_are_same_objects():
    """The test-tree modules re-export the package primitives (single source of
    truth): the PARAMS dicts and squall helpers must be the SAME objects."""
    from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025 import (
        common as tcommon,
        test_case_1 as t1,
        test_case_2 as t2,
        test_case_3 as t3,
    )
    assert t1.TC1_PARAMS is ic.TC1_PARAMS
    assert t2.TC2_PARAMS is ic.TC2_PARAMS
    assert t3.TC3_PARAMS is ic.TC3_PARAMS
    assert tcommon.isothermal_theta_ref is ic.isothermal_theta_ref
    assert tcommon.piecewise_lapse_theta_ref is ic.piecewise_lapse_theta_ref
    assert t3._squall_line_sounding is ic.squall_line_sounding
    assert t3._squall_line_theta_fn is ic.squall_line_theta_fn


def pytest_approx(value, rel=1e-9, abs=1e-6):
    import pytest

    return pytest.approx(value, rel=rel, abs=abs)
