"""The coupler's ocean/surface insolation honours the atmosphere's seasonal seam (iter 461).

iter 449 made the ATMOSPHERE insolation season-aligned (``config.insolation_start_doy``), but
``CoupledESMDriver`` computed its OWN ocean/surface ``cos_zenith`` (the surface energy budget)
from a bare ``day_to_calendar(day)`` — JANUARY-based — so a coupled run with the offset set had
the atmosphere and the ocean surface seeing DIFFERENT seasons (a physically inconsistent sun).
This locks that the coupler now routes through ``_atm._calendar_for_radiation`` so BOTH align.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def _coupled_driver(insolation_start_doy):
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    from legoesm.driver.coupled_config import CoupledConfig
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    atm = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=4, nlev=4),
        dycore=DycoreConfig(energy_fixer=False, dt=600.0, model_type="hydrostatic",
                            discretization="latlon_cgrid"),
        radiation="gray", convection="none", turbulence="none", days=1,
        insolation_start_doy=insolation_start_doy)
    drv = CoupledESMDriver(atm, CoupledConfig())     # slab ocean (default)
    drv.setup()
    return _as_after_one_segment(drv)


def _as_after_one_segment(drv):
    """Stash the surface fields an advanced atmosphere segment leaves in ``_carry_aux``:
    the coupler refuses to build surface forcing before one exists."""
    z = jnp.zeros_like(drv._atm.state.p_s.data)
    drv._atm._carry_aux = {"held_sw_net_sfc": z, "held_lw_net_sfc": z, "seg_precip": z}
    return drv


def _cos_zen_ref(lat, doy):
    """The coupler's daily-mean cos(zenith) = clip(Q_daily/S_0) at a given day-of-year."""
    from legoesm.atmosphere.physics.radiation.solar import daily_mean_insolation

    from legoesm import constants
    return jnp.clip(daily_mean_insolation(lat, float(doy), S_0=constants.S_0)
                    / constants.S_0, 0.0, 1.0)


def test_coupler_insolation_uses_the_atm_seasonal_offset():
    """With insolation_start_doy=244 (Sep 1), the coupler's surface cos_zenith matches the
    doy-244 insolation, NOT the January (doy 1) value the bare day_to_calendar gave."""
    drv = _coupled_driver(244)
    cos_zen = jnp.asarray(drv._build_atm_forcing(0.0).cos_zenith)
    lat = drv._atm._grid_lat
    # the coupler insolation is the OFFSET season (Sep 1, doy 244), to numerical tol
    np.testing.assert_allclose(np.asarray(cos_zen), np.asarray(_cos_zen_ref(lat, 244.0)),
                               rtol=1e-6, atol=1e-6)
    # and it is NOT the January value (the pre-fix bug)
    assert not np.allclose(np.asarray(cos_zen), np.asarray(_cos_zen_ref(lat, 1.0)),
                           rtol=1e-3, atol=1e-3)


def test_coupler_insolation_default_is_january_byte_identical():
    """Default (insolation_start_doy=None) keeps the legacy JANUARY-based coupler insolation —
    so the fix changes nothing unless the operator opts in (production unchanged)."""
    drv = _coupled_driver(None)
    cos_zen = jnp.asarray(drv._build_atm_forcing(0.0).cos_zenith)
    lat = drv._atm._grid_lat
    np.testing.assert_allclose(np.asarray(cos_zen), np.asarray(_cos_zen_ref(lat, 1.0)),
                               rtol=1e-6, atol=1e-6)


def _earth_system_driver(insolation_start_doy, tmp_path):
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )
    from legoesm.driver.earth_system_driver import EarthSystemDriver

    cfg = ExperimentConfig(
        grid=GridConfig(resolution=8, nlev=4),
        dycore=DycoreConfig(energy_fixer=False, dt=600.0),
        output=OutputConfig(diag_days=1), days=1, dataset="analytical", precision="fp64",
        insolation_start_doy=insolation_start_doy)
    drv = EarthSystemDriver(cfg, output_dir=tmp_path)
    drv.setup()
    return _as_after_one_segment(drv)


def test_earth_system_driver_insolation_uses_the_atm_seasonal_offset(tmp_path):
    """The sibling EarthSystemDriver (iter 462) has the SAME fix as CoupledESMDriver (461):
    insolation_start_doy=244 makes its surface cos_zenith the doy-244 season, not January."""
    drv = _earth_system_driver(244, tmp_path)
    cos_zen = jnp.asarray(drv._build_atm_forcing(0.0).cos_zenith)
    lat = drv._atm._grid_lat
    np.testing.assert_allclose(np.asarray(cos_zen), np.asarray(_cos_zen_ref(lat, 244.0)),
                               rtol=1e-6, atol=1e-6)
    assert not np.allclose(np.asarray(cos_zen), np.asarray(_cos_zen_ref(lat, 1.0)),
                           rtol=1e-3, atol=1e-3)


def test_coupled_land_co2_is_the_radiation_co2(tmp_path):
    """Both coupled drivers hand the land canopy the CO2 radiation uses (review F30),
    not the fixed config value; without a GHG override they fall back to it."""
    for drv in (_coupled_driver(None), _earth_system_driver(None, tmp_path)):
        fixed = float(drv._atm.config.co2_ppmv)
        assert drv._atm.radiation_ghg_vmr(0.0) is None
        np.testing.assert_allclose(
            np.asarray(drv._build_atm_forcing(0.0).co2_ppmv), fixed)
        drv._atm.radiation_ghg_vmr = lambda day: {"co2": jnp.asarray(337.0e-6)}
        np.testing.assert_allclose(
            np.asarray(drv._build_atm_forcing(0.0).co2_ppmv), 337.0, rtol=1e-12)
        assert fixed != 337.0
