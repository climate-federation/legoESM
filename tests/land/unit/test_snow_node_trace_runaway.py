"""Regression: the snow-node runaway (2026-10).  A thin pack under strong
ground heating used to be treated as whole-cell cover held at T_freeze, so the
canopy kept pushing heat into the ground and the soil under the switch-on arm
ran tens of kelvin above the switch-off arm (offline replay of an AMIP day:
top soil 446 K vs 335 K).  With the snow-cover fraction the switch-on column
tracks the switch-off one."""
import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (init_multilayer_land_state,
                                          step_multilayer_land)
from legoesm.land.soil_grid import SoilGridConfig
from legoesm.land.soil_thermal import SoilThermalConfig
from legoesm.land.surface_scheme import TwoLeafCanopyConfig


def _warm_sunny_forcing(ncol):
    o = jnp.ones(ncol)
    p_s = 1.0e5 * o
    T_air = 290.0
    return AtmToSurface(
        sw_down=700.0 * o, lw_down=340.0 * o, precip_total=0.0 * o,
        precip_snow=0.0 * o, T_lowest=T_air * o, q_lowest=0.006 * o,
        u_lowest=3.0 * o, v_lowest=0.0 * o, p_lowest=0.99 * p_s, p_surface=p_s,
        rho_lowest=p_s / (constants.R_d * T_air), cos_zenith=0.8 * o,
        co2_ppmv=412.0 * o, has_radiation=o, has_precipitation=o)


def _top_soil_after(on, swe, n_steps):
    ncol = swe.shape[0]
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0),
        thermal=SoilThermalConfig(snow_insulation=on),
        surface_scheme=TwoLeafCanopyConfig())
    st = init_multilayer_land_state(ncol, cfg, T_init=280.0, theta_init=0.25)
    st = st._replace(snow_depth=swe)
    if on:
        st = st._replace(T_snow=jnp.full(ncol, constants.T_freeze))
    f = _warm_sunny_forcing(ncol)
    step = jax.jit(lambda s: step_multilayer_land(s, f, cfg, 1.0, 1800.0,
                                                  lat=jnp.full(ncol, 0.8))[0])
    for _ in range(n_steps):
        st = step(st)
    return np.asarray(st.T_soil[:, 0])


def test_thin_pack_does_not_overheat_the_soil():
    """Packs of 1 and 0.01 kg/m2, 12 h of strong sun and warm air.  Before the
    fix the switch-on top soil ended ~100 K above the switch-off one."""
    swe = jnp.array([1.0, 0.01])
    T_off = _top_soil_after(False, swe, 24)
    T_on = _top_soil_after(True, swe, 24)
    assert np.all(np.isfinite(T_on))
    np.testing.assert_array_less(np.abs(T_on - T_off), 3.0)
