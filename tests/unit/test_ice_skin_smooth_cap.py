"""RULE 5 tests for the smooth melt cap on the prognostic ice skin.

The reference (the previous port, Semtner zero-layer) caps the skin with a hard
``min(T_new, T_freeze)``.  The default replaces it with
``smooth_min(T_new, T_freeze, sharpness=1/width)`` at
``ice_skin_melt_cap_width_K = 0.01`` K; width 0 recovers the hard cap.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm import constants
from legoesm.driver.config import ExperimentConfig
from legoesm.forcing.surface_utils import prognostic_ice_skin_temperature as skin

jax.config.update("jax_enable_x64", True)

_WIDTH = ExperimentConfig._field_defaults["ice_skin_melt_cap_width_K"]

# (T_skin K, F_net_down W/m2, sic, dt s, h_ice m, stored gap soft-hard K).
# Gaps measured at width 0.01 K, x64 CPU, 2026-10-24.
_CASES = [
    (constants.T_freeze, 200.0, 1.0, 600.0, 2.0, -2.1211120611042134e-05),
    (constants.T_freeze - 0.05, 50.0, 1.0, 600.0, 2.0, -2.967735573520258e-04),
    (constants.T_freeze - 0.5, 300.0, 0.8, 600.0, 2.0, 0.0),
    (constants.T_freeze - 1.0, 0.0, 1.0, 600.0, 2.0, 0.0),
    (260.0, -30.0, 1.0, 600.0, 2.0, 0.0),
    (constants.T_freeze, 600.0, 0.5, 3600.0, 0.5, 0.0),
    (271.0, 100.0, 0.0, 600.0, 2.0, 0.0),
]


def _run(case, width):
    t, f, s, dt, h = case[:5]
    return float(skin(jnp.array([t]), jnp.array([f]), jnp.array([s]), dt, h,
                      melt_cap_width_K=width)[0])


def test_default_width_is_the_approved_value():
    assert _WIDTH == 0.01


@pytest.mark.parametrize("case", _CASES)
def test_hard_limit_matches_hard_clip(case):
    t, f, s, dt, h = case[:5]
    hard = _run(case, 0.0)
    # Recompute the uncapped backward-Euler value with the cap pushed far
    # above any reachable temperature, then apply the reference clip.
    uncapped = float(skin(jnp.array([t]), jnp.array([f]), jnp.array([s]), dt,
                          h, T_melt_surface_K=1.0e4, melt_cap_width_K=0.0)[0])
    # No case reaches the 185 K floor, so only the melt clip applies.
    ref = (min(uncapped, constants.T_freeze) if s > 0.0
           else constants.T_freeze_ocean)
    np.testing.assert_allclose(hard, ref, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("case", _CASES)
def test_default_width_gap_per_case(case):
    gap = _run(case, _WIDTH) - _run(case, 0.0)
    np.testing.assert_allclose(gap, case[5], rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("case", _CASES[:2])
def test_gradient_through_cap_at_default(case):
    t, f, s, dt, h = case[:5]

    def g(T, F):
        return skin(T, F, jnp.array([s]), dt, h, melt_cap_width_K=_WIDTH)

    T0, F0 = jnp.array([t]), jnp.array([f])
    check_grads(g, (T0, F0), order=1, modes=["rev"])
    dT = jax.grad(lambda T: g(T, F0).sum())(T0)
    assert np.isfinite(dT).all() and float(jnp.abs(dT).sum()) > 0.0
