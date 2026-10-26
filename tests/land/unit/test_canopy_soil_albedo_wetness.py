"""The two-leaf land tile's soil band albedos follow the LIVE top-layer soil
water (CTSM SurfaceAlbedoMod: ``inc = max(0.11 - 0.40*h2osoi_vol, 0)``,
``alb = min(albsat + inc, albdry)``), for absorption AND for the albedo
exported to radiation.  Before, the bands were frozen at build time at a
top-layer water of 0.2, so a dry desert reflected like damp soil."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.boundary_data import fill_land_param_gaps
from legoesm.land.boundary_data.gap_fill import bare_canopy_params
from legoesm.land.canopy import radiative_transfer as rt
from legoesm.land.multilayer_land import (
    MultiLayerLandConfig, init_multilayer_land_state,
    step_multilayer_land_with_diagnostics)
from legoesm.land.soil_albedo import SOIL_COLOR_ALBEDO
from legoesm.land.surface_scheme import TwoLeafCanopyConfig

jax.config.update("jax_enable_x64", True)

NCOL = 3
SW = 300.0
# CLM soil-colour class 1 (brightest): dry VIS/NIR, saturated VIS/NIR
DRY_VIS, DRY_NIR, SAT_VIS, SAT_NIR = SOIL_COLOR_ALBEDO[0]
GLAC_VIS, GLAC_NIR = 0.8178, 0.6178
# columns: dry desert soil, wet soil, glacier
THETA_TOP = (0.03, 0.35, 0.03)


def _ctsm(dry, sat, theta):
    """Independent transcription of the CTSM relation (not the model's code)."""
    return min(sat + max(0.11 - 0.40 * theta, 0.0), dry)


def _forcing():
    n = NCOL
    return AtmToSurface(
        sw_down=jnp.full(n, SW), lw_down=jnp.full(n, 350.0),
        precip_total=jnp.zeros(n), precip_snow=jnp.zeros(n),
        T_lowest=jnp.full(n, 300.0), q_lowest=jnp.full(n, 0.005),
        u_lowest=jnp.full(n, 5.0), v_lowest=jnp.full(n, 2.0),
        p_lowest=jnp.full(n, 95000.0), p_surface=jnp.full(n, 100000.0),
        rho_lowest=jnp.full(n, 1.2), cos_zenith=jnp.full(n, 0.5),
        co2_ppmv=jnp.full(n, 400.0), has_radiation=jnp.ones(n),
        has_precipitation=jnp.ones(n))


def _params():
    # Build-time bands deliberately at the OLD frozen wetness (0.2): the step
    # must not use them.
    vis_dry = jnp.asarray([DRY_VIS, DRY_VIS, GLAC_VIS])
    vis_sat = jnp.asarray([SAT_VIS, SAT_VIS, GLAC_VIS])
    nir_dry = jnp.asarray([DRY_NIR, DRY_NIR, GLAC_NIR])
    nir_sat = jnp.asarray([SAT_NIR, SAT_NIR, GLAC_NIR])
    return bare_canopy_params(NCOL)._replace(
        LAI=jnp.asarray([0.0, 0.0, 0.0]),
        ALB_VIS=jnp.asarray([_ctsm(DRY_VIS, SAT_VIS, 0.2)] * 2 + [GLAC_VIS]),
        ALB_NIR=jnp.asarray([_ctsm(DRY_NIR, SAT_NIR, 0.2)] * 2 + [GLAC_NIR]),
        ALB_VIS_DRY=vis_dry, ALB_VIS_SAT=vis_sat,
        ALB_NIR_DRY=nir_dry, ALB_NIR_SAT=nir_sat)


def _run(snow_albedo_feedback):
    cfg = MultiLayerLandConfig(snow_albedo_feedback=snow_albedo_feedback,
                               surface_scheme=TwoLeafCanopyConfig())
    state = init_multilayer_land_state(NCOL, cfg, T_init=300.0)
    theta = state.theta_soil.at[:, 0].set(jnp.asarray(THETA_TOP))
    state = state._replace(theta_soil=theta, snow_depth=jnp.zeros(NCOL),
                           snow_age=jnp.zeros(NCOL))
    state_new, resp, _, sfc = step_multilayer_land_with_diagnostics(
        state, _forcing(), cfg, 1.0, 600.0, lat=jnp.full(NCOL, 0.4),
        land_params=_params())
    return state, state_new, resp, sfc


def _expected_alpha(theta_top):
    out = []
    for c, th in enumerate(np.asarray(theta_top)):
        if c == 2:
            out.append(float(rt.broadband_albedo(GLAC_VIS, GLAC_NIR)))
        else:
            out.append(float(rt.broadband_albedo(
                _ctsm(DRY_VIS, SAT_VIS, th), _ctsm(DRY_NIR, SAT_NIR, th))))
    return np.asarray(out)


def test_exported_albedo_follows_live_soil_water():
    for feedback in (True, False):
        state, state_new, resp, sfc = _run(feedback)
        alpha = np.asarray(resp.albedo)
        # dry desert soil is brighter than wet soil of the same colour class
        assert alpha[0] > alpha[1] + 0.05, (feedback, alpha)
        # glacier: dry == sat, so the ice albedo survives at any wetness
        np.testing.assert_allclose(
            alpha[2], rt.broadband_albedo(GLAC_VIS, GLAC_NIR), rtol=1e-12)
        # export = CTSM relation at the END-of-step top-layer water, with or
        # without snow layering (snow-free here)
        np.testing.assert_allclose(
            alpha, _expected_alpha(state_new.theta_soil[:, 0]), rtol=1e-10)


def test_absorbed_uses_the_same_wet_soil_albedo():
    state, _, _, sfc = _run(True)
    absorbed = np.asarray(sfc.sw_net)
    np.testing.assert_allclose(
        absorbed, (1.0 - _expected_alpha(state.theta_soil[:, 0])) * SW,
        rtol=1e-9, atol=1e-9)


def test_albedo_differentiable_in_soil_water():
    from legoesm.land.soil_albedo import wet_soil_albedo
    g = jax.grad(lambda th: wet_soil_albedo(DRY_VIS, SAT_VIS, th))(0.1)
    np.testing.assert_allclose(g, -0.40, rtol=1e-12)


def test_gap_fill_matches_bounded_params():
    """Params carrying the bounds gap-fill against a matching bare fallback
    (a structure mismatch would raise), and filled columns keep dry == sat."""
    lp = _params()
    gsd = type("G", (), {"pft_frac": np.asarray(
        [[1.0] + [0.0] * 16, [np.nan] * 17, [1.0] + [0.0] * 16])})()
    out = fill_land_param_gaps(lp, gsd)
    assert out.ALB_VIS_DRY is not None
    np.testing.assert_allclose(out.ALB_VIS_DRY[1], out.ALB_VIS_SAT[1])
    np.testing.assert_allclose(out.ALB_VIS_DRY[0], DRY_VIS)


@pytest.mark.parametrize("feedback", [True, False])
def test_handoff_export_is_what_the_next_step_absorbs(feedback):
    """Exported albedo (end-of-step water) == what the NEXT step absorbs with."""
    cfg = MultiLayerLandConfig(snow_albedo_feedback=feedback,
                               surface_scheme=TwoLeafCanopyConfig())
    state, state1, resp1, _ = _run(feedback)
    _, _, _, sfc2 = step_multilayer_land_with_diagnostics(
        state1, _forcing(), cfg, 1.0, 600.0, lat=jnp.full(NCOL, 0.4),
        land_params=_params())
    np.testing.assert_allclose(np.asarray(sfc2.sw_net),
                               (1.0 - np.asarray(resp1.albedo)) * SW,
                               rtol=1e-9, atol=1e-9)
    # and the water did change within step 1, so this is not the start value
    assert np.max(np.abs(state1.theta_soil[:, 0] - state.theta_soil[:, 0])) > 1e-4


def _bare_gsd():
    """2-column surfdata: column 0 bare soil (colour class 1), column 1 glacier."""
    from legoesm.land.global_surface_data import (
        GlobalSurfaceData, GlobalSurfaceDataConfig)
    from legoesm.land.surface_params import N_PFT_CLM5
    ncol, npft = 2, N_PFT_CLM5
    pft = np.zeros((1, ncol, npft))
    pft[0, :, 0] = 1.0
    z4 = np.zeros((ncol, 4))
    gl = np.asarray([0.0, 1.0])[None, :]
    return GlobalSurfaceData(
        sand_frac=jnp.asarray(np.full((ncol, 4), 0.9)),
        clay_frac=jnp.asarray(np.full((ncol, 4), 0.05)),
        organic=jnp.asarray(z4), bulk_density=jnp.asarray(z4),
        soil_color=jnp.asarray(np.array([1, 1])),
        cell_area=jnp.ones(ncol), years=jnp.asarray([2000.0]),
        f_land=jnp.asarray(1.0 - gl), f_lake=jnp.zeros((1, ncol)),
        f_glacier=jnp.asarray(gl), pft_frac=jnp.asarray(pft),
        months=jnp.arange(12.0),
        lai_monthly=jnp.zeros((12, ncol, npft)), sai_monthly=jnp.zeros((12, ncol, npft)),
        htop_monthly=jnp.zeros((12, ncol, npft)), hbot_monthly=jnp.zeros((12, ncol, npft)),
        config=GlobalSurfaceDataConfig())


def test_production_builders_carry_live_wetness_bounds():
    """The params the production lane builds (init-time builder AND per-step
    updater) carry the bounds, so the land step's rewet is not a no-op there."""
    from legoesm.land.boundary_data import (
        make_step_land_params_updater, surface_data_to_land_params)
    from legoesm.land.boundary_data._internals import GLACIER_ALB_VIS
    from legoesm.land.canopy import CanopyConfig
    from legoesm.land.soil_albedo import rewet_soil_bands
    gsd = _bare_gsd()
    lp_init = surface_data_to_land_params(gsd, CanopyConfig(), 15.0, jnp.full(2, 0.2))
    lp_step, _ = make_step_land_params_updater(gsd, CanopyConfig())(
        jnp.full(2, 0.2), jnp.asarray(15.0), jnp.asarray(2000.0))
    for lp in (lp_init, lp_step):
        dry = rewet_soil_bands(lp, jnp.full(2, 0.03))
        np.testing.assert_allclose(dry.ALB_VIS[0], _ctsm(DRY_VIS, SAT_VIS, 0.03), rtol=1e-12)
        np.testing.assert_allclose(dry.ALB_NIR[0], _ctsm(DRY_NIR, SAT_NIR, 0.03), rtol=1e-12)
        np.testing.assert_allclose(dry.ALB_VIS[1], GLACIER_ALB_VIS, rtol=1e-12)
        # build-time bands unchanged: still the 0.2 value
        np.testing.assert_allclose(lp.ALB_VIS[0], _ctsm(DRY_VIS, SAT_VIS, 0.2), rtol=1e-12)


def test_partial_bounds_raise():
    from legoesm.land.soil_albedo import rewet_soil_bands
    with pytest.raises(ValueError, match="all set or all None"):
        rewet_soil_bands(_params()._replace(ALB_NIR_SAT=None), jnp.full(NCOL, 0.1))


def test_snow_layered_on_the_rewetted_bands():
    """With snow on the ground the export layers snow on the END-of-step wet
    soil bands, not on the build-time ones."""
    from legoesm.land.multilayer_land import compute_land_albedo
    from legoesm.land.soil_albedo import rewet_soil_bands
    cfg = MultiLayerLandConfig(snow_albedo_feedback=True,
                               surface_scheme=TwoLeafCanopyConfig())
    state = init_multilayer_land_state(NCOL, cfg, T_init=265.0)
    state = state._replace(
        theta_soil=state.theta_soil.at[:, 0].set(jnp.asarray(THETA_TOP)),
        snow_depth=jnp.full(NCOL, 5.0), snow_age=jnp.full(NCOL, 86400.0))
    lat = jnp.full(NCOL, 0.4)
    state_new, resp, _, _ = step_multilayer_land_with_diagnostics(
        state, _forcing(), cfg, 1.0, 600.0, lat=lat, land_params=_params())
    lp_end = rewet_soil_bands(_params(), state_new.theta_soil[:, 0])
    band = lambda a: compute_land_albedo(lat, state_new.snow_depth,
                                         state_new.snow_age, cfg.land_albedo,
                                         base_albedo=a)
    want = np.asarray(rt.broadband_albedo(band(lp_end.ALB_VIS), band(lp_end.ALB_NIR)))
    alpha = np.asarray(resp.albedo)
    # snow brightens the soil columns, so this is not the snow-free value
    assert np.all(alpha[:2] > _expected_alpha(state_new.theta_soil[:, 0])[:2])
    np.testing.assert_allclose(alpha, want, rtol=1e-10)
