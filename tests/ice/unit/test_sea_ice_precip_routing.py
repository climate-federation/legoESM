"""Precipitation falling on the ice fraction must reach the ocean or the pack.

Each test runs the extended ice thermodynamics twice, with and without the
precipitation, and closes the per-grid-cell water (and, for snow, latent heat)
budget on the DIFFERENCE, so every other term cancels exactly.

Sign conventions: ``freshwater_to_ocean`` is +ve INTO the ocean;
``ocean_heat_extraction`` is +ve when the ocean LOSES heat.
"""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.sea_ice import _thermo_v2

SHAPE = (1, 1)
DT = 3600.0


def _forcing(precip_total, precip_snow, has_precip=1.0):
    full = lambda v: jnp.full(SHAPE, v)  # noqa: E731
    return AtmToSurface(
        sw_down=full(0.0), lw_down=full(250.0),
        precip_total=full(precip_total), precip_snow=full(precip_snow),
        T_lowest=full(268.0), q_lowest=full(2e-3),
        u_lowest=full(3.0), v_lowest=full(0.0),
        p_lowest=full(9.5e4), p_surface=full(1e5),
        rho_lowest=full(1.25), cos_zenith=full(0.0),
        co2_ppmv=full(400.0),
        has_radiation=full(0.0), has_precipitation=full(has_precip),
    )


def _run(cfg, forcing, h_snow0=0.0, h0=2.0, conc0=0.9):
    full = lambda v: jnp.full(SHAPE, v)  # noqa: E731
    return _thermo_v2(
        full(h0), full(265.0), full(conc0), full(h_snow0), full(5.0),
        full(0.0), full(0.0), forcing,
        full(constants.T_freeze_ocean), cfg, 1.0, DT,
    )


def test_snow_off_snowfall_on_ice_reaches_ocean_with_melt_heat():
    cfg = SeaIceConfig()
    assert not cfg.snow.enabled
    snow = 1.0e-4  # kg/m^2/s, all solid
    wet = _run(cfg, _forcing(snow, snow))
    dry = _run(cfg, _forcing(0.0, 0.0))
    conc = float(wet["conc"][0, 0])
    assert conc == pytest.approx(float(dry["conc"][0, 0]), rel=1e-14)
    d_fw = float(wet["freshwater_to_ocean"][0, 0] - dry["freshwater_to_ocean"][0, 0])
    d_q = float(wet["ocean_heat_extraction"][0, 0]
                - dry["ocean_heat_extraction"][0, 0])
    assert d_fw == pytest.approx(snow * conc, rel=1e-10)
    # The ocean melts the snow: it loses L_f per kg delivered.
    assert d_q == pytest.approx(snow * conc * cfg.L_f, rel=1e-10)


def test_snow_off_no_phantom_snow_without_precipitation_flag():
    cfg = SeaIceConfig()
    off = _run(cfg, _forcing(1.0e-4, 1.0e-4, has_precip=0.0))
    dry = _run(cfg, _forcing(0.0, 0.0))
    assert float(off["freshwater_to_ocean"][0, 0]) == pytest.approx(
        float(dry["freshwater_to_ocean"][0, 0]), abs=1e-18)
