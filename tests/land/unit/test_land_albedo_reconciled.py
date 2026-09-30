"""The two-leaf land tile absorbs sunlight with the SAME albedo it exports to
the atmosphere.  Before the reconciliation the canopy absorbed through the
snow-free soil-colour band albedos while the atmosphere reflected the exported
snow-aged albedo on the same cell, and nothing compared the two."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land import multilayer_land as ml
from legoesm.land.boundary_data.gap_fill import bare_canopy_params
from legoesm.land.canopy import radiative_transfer as rt
from legoesm.land.multilayer_land import (
    MultiLayerLandConfig, init_multilayer_land_state,
    step_multilayer_land_with_diagnostics)
from legoesm.land.surface_scheme import TwoLeafCanopyConfig

jax.config.update("jax_enable_x64", True)

NCOL = 4
SW = 300.0


def _forcing():
    n = NCOL
    return AtmToSurface(
        sw_down=jnp.full(n, SW), lw_down=jnp.full(n, 250.0),
        precip_total=jnp.zeros(n), precip_snow=jnp.zeros(n),
        T_lowest=jnp.full(n, 268.0), q_lowest=jnp.full(n, 0.002),
        u_lowest=jnp.full(n, 5.0), v_lowest=jnp.full(n, 2.0),
        p_lowest=jnp.full(n, 95000.0), p_surface=jnp.full(n, 100000.0),
        rho_lowest=jnp.full(n, 1.2), cos_zenith=jnp.full(n, 0.5),
        co2_ppmv=jnp.full(n, 400.0), has_radiation=jnp.ones(n),
        has_precipitation=jnp.ones(n))


def _run(monkeypatch=None):
    cfg = MultiLayerLandConfig(snow_albedo_feedback=True,
                               surface_scheme=TwoLeafCanopyConfig())
    state = init_multilayer_land_state(NCOL, cfg, T_init=265.0)
    # snow-free soil / fresh thin snow / deep aged snow / snow-free GLACIER
    state = state._replace(
        snow_depth=jnp.asarray([0.0, 5.0, 200.0, 0.0]),
        snow_age=jnp.asarray([0.0, 2.0 * 86400.0, 60.0 * 86400.0, 0.0]))
    lp = bare_canopy_params(NCOL)._replace(LAI=jnp.full(NCOL, 0.5),
                                           ALB_VIS=jnp.asarray([0.1, 0.1, 0.1, 0.8178]),
                                           ALB_NIR=jnp.asarray([0.2, 0.2, 0.2, 0.6178]))
    state_new, resp, _, sfc = step_multilayer_land_with_diagnostics(
        state, _forcing(), cfg, 1.0, 600.0, lat=jnp.full(NCOL, 1.2),
        land_params=lp)
    return resp, sfc, state, cfg, state_new, lp


def _bands_alpha(st, lp, cfg):
    from legoesm.land.multilayer_land import compute_land_albedo
    lat = jnp.full(NCOL, 1.2)
    band = lambda a: compute_land_albedo(lat, st.snow_depth, st.snow_age,
                                         cfg.land_albedo, base_albedo=a)
    return np.asarray(rt.broadband_albedo(band(lp.ALB_VIS), band(lp.ALB_NIR)))


def _check(resp, sfc, state, cfg, state_new, lp):
    absorbed = np.asarray(sfc.sw_net)          # canopy + ground, summed
    # exact: absorption used the RT's own broadband reflectance of the
    # snow-layered bands on the PRE-step snow (soil-reflection terms cancel)
    alpha_pre = _bands_alpha(state, lp, cfg)
    np.testing.assert_allclose(absorbed, (1.0 - alpha_pre) * SW, rtol=1e-9, atol=1e-9)
    # the export is the same construction on the POST-step snow, for the next
    # radiation call
    alpha = np.asarray(resp.albedo)
    np.testing.assert_allclose(alpha, _bands_alpha(state_new, lp, cfg), rtol=1e-12)
    return alpha, state, cfg


def test_absorbed_matches_exported_albedo():
    alpha, state, cfg = _check(*_run())
    # both snow columns are brighter than the snow-free one
    assert alpha[1] > alpha[0] and alpha[2] > alpha[0]
    # snow-free columns export the canopy's OWN bands folded to broadband (the
    # calibrated glacier bands survive, no config scalar, no dry-soil
    # brightening), and absorb exactly what the unreconciled canopy absorbed
    # with those bands: (1 - broadband) * SW
    np.testing.assert_allclose(alpha[0], rt.broadband_albedo(0.1, 0.2), rtol=1e-12)
    np.testing.assert_allclose(alpha[3], rt.broadband_albedo(0.8178, 0.6178), rtol=1e-12)


def test_planted_mismatch_is_caught(monkeypatch):
    orig = ml.compute_two_leaf_canopy_fluxes

    def canopy_with_soil_bands(**kw):
        lp = kw["canopy_params"]
        kw["canopy_params"] = lp._replace(ALB_VIS=jnp.full(NCOL, 0.1),
                                          ALB_NIR=jnp.full(NCOL, 0.2))
        return orig(**kw)

    monkeypatch.setattr(ml, "compute_two_leaf_canopy_fluxes", canopy_with_soil_bands)
    with pytest.raises(AssertionError):
        _check(*_run())


def test_ground_absorption_nonnegative_under_bright_snow():
    n = 3
    one = jnp.ones(n)
    out = rt.canopy_shortwave_rt(
        PAR_dir=200.0 * one, PAR_diff=50.0 * one, NIR_dir=200.0 * one,
        NIR_diff=50.0 * one, UV=10.0 * one, SZA=jnp.full(n, 0.5),
        LAI=jnp.asarray([0.5, 2.0, 4.0]), CI=one, ALB_VIS=0.8 * one,
        ALB_NIR=0.7 * one, Vcmax25_C3_leaf=50.0 * one,
        Vcmax25_C4_leaf=jnp.zeros(n), kn=0.3 * one)
    soil = np.asarray(out.ASW_Soil)
    assert np.all(soil >= -1e-9), soil
    # components stay non-negative and the column total is exact
    for a in (out.ASW_Sun, out.ASW_Sh):
        assert np.all(np.asarray(a) >= -1e-9)
    # the column total is the band-weighted absorbed fraction of what was
    # handed in (PAR 250, NIR 250, UV 10 W/m2 here)
    total = np.asarray(out.ASW_Sun + out.ASW_Sh + out.ASW_Soil)
    want = (1 - 0.8) * 250.0 + (1 - 0.7) * 250.0 + (1 - rt._RHO_UV) * 10.0
    np.testing.assert_allclose(total, want, rtol=1e-9)


def test_snow_age_activation_default_is_bats_and_production_pins_it():
    """Default is the BATS 5000 K temperature-dependent clock (the calendar
    clock left every snow cell at 0.5207); production still sets it explicitly."""
    from pathlib import Path

    import yaml

    from legoesm.surface_albedo import LandAlbedoConfig
    assert LandAlbedoConfig().snow_age_activation_K == 5000.0
    deck = Path(__file__).resolve().parents[3] / "config/amip/amip_production.yaml"
    assert yaml.safe_load(deck.read_text())["snow_age_activation_K"] == 5000.0
