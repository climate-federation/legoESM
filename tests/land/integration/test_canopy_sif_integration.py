"""Integration tests: SIF wiring for both canopy paths.

Guards that the optional SIF diagnostic is populated on ``SurfaceFluxOutput``
(and threaded through ``step_multilayer_land_with_diagnostics``) for BOTH:
  1. the two-leaf (two-big-leaf) canopy — ``TwoLeafCanopyConfig(sif=...)``
  2. the SimpleSEB big-leaf coupled Farquhar — ``StomataConfig(enabled, sif=...)``
and that it is None when no ``SIFConfig`` is attached (opt-in, zero-cost default).

Physical gate: SIF is 0 in the dark and positive under midday light for both.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy import CanopyLandParams, SIFConfig
from legoesm.land.carbon.carbon_cycle import init_carbon_state
from legoesm.land.carbon.config import CarbonConfig
from legoesm.land.stomata import StomataConfig
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    step_multilayer_land_with_diagnostics,
)
from legoesm.land.surface_scheme import SimpleSEBConfig, TwoLeafCanopyConfig

NCOL = 2


def _forcing(sw_down: float, cosz: float) -> AtmToSurface:
    return AtmToSurface(
        T_lowest=jnp.full(NCOL, 295.0), q_lowest=jnp.full(NCOL, 0.010),
        u_lowest=jnp.full(NCOL, 2.5), v_lowest=jnp.full(NCOL, 0.5),
        p_lowest=jnp.full(NCOL, 98000.0), p_surface=jnp.full(NCOL, 101325.0),
        rho_lowest=jnp.full(NCOL, 1.18), sw_down=jnp.full(NCOL, sw_down),
        lw_down=jnp.full(NCOL, 350.0), cos_zenith=jnp.full(NCOL, cosz),
        precip_total=jnp.zeros(NCOL), precip_snow=jnp.zeros(NCOL),
        co2_ppmv=jnp.full(NCOL, 420.0), has_radiation=True, has_precipitation=False,
    )


_CANOPY_PARAMS = CanopyLandParams(
    LAI=jnp.full(NCOL, 3.0), hc=jnp.full(NCOL, 12.0), fC4=jnp.zeros(NCOL),
    FNonVeg=jnp.zeros(NCOL), CI=jnp.full(NCOL, 0.7), kn=jnp.full(NCOL, 0.3),
    Vcmax25_C3_leaf=jnp.full(NCOL, 60.0), Vcmax25_C4_leaf=jnp.full(NCOL, 40.0),
    m_C3=jnp.full(NCOL, 9.0), m_C4=jnp.full(NCOL, 4.0),
    b0_C3=jnp.full(NCOL, 0.01), b0_C4=jnp.full(NCOL, 0.04),
    alf=jnp.full(NCOL, 0.3), TgC=jnp.full(NCOL, 20.0),
    ALB_VIS=jnp.full(NCOL, 0.1), ALB_NIR=jnp.full(NCOL, 0.2),
    emissivity=jnp.full(NCOL, 0.97), rz0m=jnp.full(NCOL, 0.055), rd=jnp.full(NCOL, 0.67),
)


def _two_leaf_sif(sw_down, cosz, sif_cfg):
    cfg = MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(max_iters=25, sif=sif_cfg))
    state = init_multilayer_land_state(NCOL, cfg, T_init=290.0, TgC_init=20.0)
    _, _, _, sfc = step_multilayer_land_with_diagnostics(
        state, _forcing(sw_down, cosz), cfg, U_min=1.0, dt=3600.0,
        lat=jnp.zeros(NCOL), doy=180.0, land_params=_CANOPY_PARAMS)
    return sfc.sif


def _simple_seb_sif(sw_down, cosz, sif_cfg):
    cfg = MultiLayerLandConfig(
        surface_scheme=SimpleSEBConfig(),
        stomata=StomataConfig(enabled=True, sif=sif_cfg),
        carbon=CarbonConfig(scheme="differland"))
    state = init_multilayer_land_state(NCOL, cfg, T_init=290.0)
    cstate = init_carbon_state((NCOL,), cfg.carbon)
    _, _, _, sfc = step_multilayer_land_with_diagnostics(
        state, _forcing(sw_down, cosz), cfg, U_min=1.0, dt=3600.0,
        lat=jnp.zeros(NCOL), carbon_state=cstate, doy=180.0)
    return sfc.sif


def test_two_leaf_sif_daytime_positive_night_zero():
    day = _two_leaf_sif(900.0, 0.9, SIFConfig())
    night = _two_leaf_sif(0.0, 0.0, SIFConfig())
    assert day is not None and night is not None
    assert jnp.all(jnp.isfinite(day)) and float(day[0]) > 0.0
    assert float(night[0]) == 0.0
    assert 0.5 < float(day[0]) < 60.0        # plausible emitted photon flux


def test_simple_seb_sif_daytime_positive_night_zero():
    day = _simple_seb_sif(900.0, 0.9, SIFConfig())
    night = _simple_seb_sif(0.0, 0.0, SIFConfig())
    assert day is not None and night is not None
    assert jnp.all(jnp.isfinite(day)) and float(day[0]) > 0.0
    assert float(night[0]) == 0.0
    assert 0.5 < float(day[0]) < 60.0


def test_two_leaf_sif_use_ta_for_photosynthesis():
    # HIGH-fix regression: when use_ta_for_photosynthesis=True the Farquhar An
    # uses Ta, so SIF's Gamma* must too — the path must still give finite,
    # positive, plausible midday SIF (not NaN / not wildly off).
    cfg = MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(
            max_iters=25, use_ta_for_photosynthesis=True, sif=SIFConfig()))
    state = init_multilayer_land_state(NCOL, cfg, T_init=290.0, TgC_init=20.0)
    _, _, _, sfc = step_multilayer_land_with_diagnostics(
        state, _forcing(900.0, 0.9), cfg, U_min=1.0, dt=3600.0,
        lat=jnp.zeros(NCOL), doy=180.0, land_params=_CANOPY_PARAMS)
    assert sfc.sif is not None and jnp.all(jnp.isfinite(sfc.sif))
    assert 0.5 < float(sfc.sif[0]) < 60.0


def test_sif_none_when_not_configured():
    # Opt-in: no SIFConfig -> sif field stays None (zero-cost default) on both paths.
    assert _two_leaf_sif(900.0, 0.9, None) is None
    assert _simple_seb_sif(900.0, 0.9, None) is None


def test_escape_probability_scales_sif():
    full = _two_leaf_sif(900.0, 0.9, SIFConfig(escape_probability=1.0))
    half = _two_leaf_sif(900.0, 0.9, SIFConfig(escape_probability=0.5))
    assert float(half[0]) == float(full[0]) * 0.5
