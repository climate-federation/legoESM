"""Optional per-column canopy band albedo (blended with the wet soil bands by
canopy cover) and per-column leaf emissivity on the two-leaf land tile.

Unset, the land step must match the pre-change code bit for bit; that is checked
across checkouts by ``scripts/validate/land_step_golden_dump.py``, since within
one checkout both sides would run the same code."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.land.canopy import radiative_transfer as rt
from legoesm.land.multilayer_land import (
    MultiLayerLandConfig, init_multilayer_land_state,
    step_multilayer_land_with_diagnostics)
from legoesm.land.soil_albedo import rewet_soil_bands
from legoesm.land.surface_scheme import TwoLeafCanopyConfig

from tests.land.unit.test_canopy_soil_albedo_wetness import (
    NCOL, THETA_TOP, _forcing, _params)

jax.config.update("jax_enable_x64", True)

LAI = jnp.asarray([3.0, 0.5, 0.0])
CI = jnp.asarray([0.8, 0.9, 1.0])
CAN_VIS = jnp.asarray([0.05, 0.07, 0.06])
CAN_NIR = jnp.asarray([0.30, 0.25, 0.28])


def _veg_params(**kw):
    return _params()._replace(LAI=LAI, CI=CI, **kw)


def _step(lp, snow_albedo_feedback=True):
    cfg = MultiLayerLandConfig(snow_albedo_feedback=snow_albedo_feedback,
                               surface_scheme=TwoLeafCanopyConfig())
    state = init_multilayer_land_state(NCOL, cfg, T_init=300.0)
    state = state._replace(
        theta_soil=state.theta_soil.at[:, 0].set(jnp.asarray(THETA_TOP)),
        snow_depth=jnp.zeros(NCOL), snow_age=jnp.zeros(NCOL))
    return step_multilayer_land_with_diagnostics(
        state, _forcing(), cfg, 1.0, 600.0, lat=jnp.full(NCOL, 0.4), land_params=lp)


def test_blend_matches_cover_weighted_formula():
    theta = jnp.full(NCOL, 0.1)
    soil = rewet_soil_bands(_veg_params(), theta)
    out = rewet_soil_bands(_veg_params(ALB_VIS_CAN=CAN_VIS, ALB_NIR_CAN=CAN_NIR), theta)
    cov = 1.0 - np.exp(-0.5 * np.asarray(CI) * np.asarray(LAI))   # independent G=0.5
    np.testing.assert_allclose(
        out.ALB_VIS, cov * CAN_VIS + (1 - cov) * soil.ALB_VIS, rtol=1e-13)
    np.testing.assert_allclose(
        out.ALB_NIR, cov * CAN_NIR + (1 - cov) * soil.ALB_NIR, rtol=1e-13)
    # no leaves: the soil (or glacier) band survives exactly
    assert out.ALB_VIS[2] == soil.ALB_VIS[2] and out.ALB_NIR[2] == soil.ALB_NIR[2]


def test_per_column_epsf_equal_to_scalar_is_bit_identical():
    eps = TwoLeafCanopyConfig().epsf
    base = _step(_veg_params())
    col = _step(_veg_params(epsf=jnp.full(NCOL, eps)))
    for a, b in zip(jax.tree.leaves(base), jax.tree.leaves(col)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_blend_and_epsf_reach_the_land_step():
    base = _step(_veg_params())[1]
    blended = _step(_veg_params(ALB_VIS_CAN=CAN_VIS, ALB_NIR_CAN=CAN_NIR))[1]
    darker = _step(_veg_params(epsf=jnp.full(NCOL, 0.94)))[1]
    assert abs(float(blended.albedo[0] - base.albedo[0])) > 1e-3
    assert float(blended.albedo[2]) == float(base.albedo[2])      # LAI = 0 column
    assert abs(float(darker.lw_up[0] - base.lw_up[0])) > 1e-3


def test_gradients_through_canopy_albedo_and_epsf():
    theta = jnp.full(NCOL, 0.1)

    def alb(cv, cn):
        lp = rewet_soil_bands(_veg_params(ALB_VIS_CAN=cv, ALB_NIR_CAN=cn), theta)
        return jnp.sum(rt.broadband_albedo(lp.ALB_VIS, lp.ALB_NIR))
    check_grads(alb, (CAN_VIS, CAN_NIR), order=2, modes=("rev",))

    def lw(e):
        return jnp.sum(_step(_veg_params(epsf=e))[1].lw_up)
    g = jax.grad(lw)(jnp.full(NCOL, 0.97))
    assert np.all(np.isfinite(np.asarray(g))) and abs(float(g[0])) > 0.0


def test_inconsistent_canopy_fields_raise():
    theta = jnp.full(NCOL, 0.1)
    with pytest.raises(ValueError, match="both set or both None"):
        rewet_soil_bands(_veg_params(ALB_VIS_CAN=CAN_VIS), theta)
    no_bounds = _veg_params(ALB_VIS_CAN=CAN_VIS, ALB_NIR_CAN=CAN_NIR)._replace(
        ALB_VIS_DRY=None, ALB_VIS_SAT=None, ALB_NIR_DRY=None, ALB_NIR_SAT=None)
    with pytest.raises(ValueError, match="need the soil-colour bounds"):
        rewet_soil_bands(no_bounds, theta)
