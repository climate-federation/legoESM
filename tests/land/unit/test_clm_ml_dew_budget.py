"""Multilayer canopy (CLM-ML) land column water budget over a humid night.

The land + atmosphere water budget closes with the dew limit active: the
change in soil + pond + snow water equals minus the reported surface water
flux minus runoff, to round-off, every step.

KNOWN DEFECT (documented here, measured 2026-11-11): CLM-ML's internal
canopy-water store (``h2ocan_profile``) is counted by no land budget.  Rain
intercepted into it also reaches the soil in full, and leaf dew and canopy-air
condensation also reach the soil through the net latent flux.  Wet-leaf
evaporation drawn from it is also charged to the soil.  It is a shadow of
water the soil already holds, so land + atmosphere close without it.  Over this
night it gains about 0.1 kg/m2 (the condensation on leafy layers).  The last
test pins that the store holds water this budget never counts; making the
store a real, budgeted store changes it.

Run under ``JAX_ENABLE_X64=1``.  CLM-ML is not jittable, so this is slow.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy.config import CLMMLCanopyConfig
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (init_multilayer_land_state,
                                          step_multilayer_land_with_diagnostics)
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.thermo import saturation_specific_humidity

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.slow

_P = 95000.0
_DT = 1800.0
_NIGHT_TA = np.linspace(291.0, 286.0, 8)    # cooling humid night [K]


def _forcing(Ta, u, sw=0.0, cz=0.0):
    q = 0.97 * float(saturation_specific_humidity(jnp.asarray(Ta), jnp.asarray(_P)))
    f = lambda v: jnp.full(1, v)  # noqa: E731
    return AtmToSurface(
        sw_down=f(sw), lw_down=f(320.0), precip_total=f(0.0), precip_snow=f(0.0),
        T_lowest=f(Ta), q_lowest=f(q), u_lowest=f(u), v_lowest=f(0.0), p_lowest=f(_P),
        p_surface=f(100000.0), rho_lowest=f(1.15), cos_zenith=f(cz), co2_ppmv=f(400.0),
        has_radiation=f(1.0), has_precipitation=f(1.0))


def _canopy_store(state):
    """Sum of the canopy-water store over the canopy layers [kg m-2].  Only
    layers 1..ncan: the padded layers above hold a huge fill value."""
    m = state.canopy_state.mlcanopy
    p = int(np.argmax(np.asarray(m.ncan_canopy)))
    n = int(np.asarray(m.ncan_canopy)[p])
    return float(np.sum(np.asarray(m.h2ocan_profile)[p, 1:n + 1]))


@pytest.fixture(scope="module")
def night():
    """A warm day step, then 8 steps of a cooling humid night.  Per step: the
    land + atmosphere water residual [kg m-2], and the canopy store after it."""
    cfg = MultiLayerLandConfig(soil_grid=SoilGridConfig(n_layers=8, total_depth=3.0),
                               surface_scheme=CLMMLCanopyConfig())
    dz = np.asarray(make_soil_grid(cfg.soil_grid).dz)
    st = init_multilayer_land_state(1, cfg, T_init=290.0, theta_init=0.3)

    def ground_water(s):
        return float((np.sum(dz * np.asarray(s.theta_soil)[0]) + float(s.surface_water[0]))
                     * constants.rho_water + float(s.snow_depth[0]))

    res, store = [], []
    for f in [_forcing(295.0, 0.5, 400.0, 0.7)] + [_forcing(float(T), 0.5) for T in _NIGHT_TA]:
        s2, r, _, _ = step_multilayer_land_with_diagnostics(
            st, f, cfg, 1.0, _DT, lat=jnp.zeros(1), doy=180.0)
        evap = float(r.surface_mass_flux[0]) * _DT                 # +up [kg m-2]
        runoff = float(s2.runoff_surface[0] + s2.runoff_subsurface[0]) * _DT
        res.append(ground_water(s2) - ground_water(st) + evap + runoff)
        store.append(_canopy_store(s2))
        st = s2
    return np.asarray(res), np.asarray(store)


def test_land_and_atmosphere_water_closes_with_dew(night):
    res, _ = night
    assert np.all(np.isfinite(res))
    assert np.abs(res).max() < 1e-12, np.abs(res).max()   # measured 1.1e-13


def test_canopy_store_holds_water_no_budget_counts(night):
    """KNOWN DEFECT pin: the store gains the night's dew although the budget
    above closes without it."""
    _, store = night
    gain = store[-1] - store[1]
    assert gain > 0.05, gain                               # measured ~0.104
