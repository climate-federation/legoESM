"""The coupler's ocean/surface insolation honours the atmosphere's seasonal seam (iter 461).

iter 449 made the ATMOSPHERE insolation season-aligned (``config.insolation_start_doy``), but
``CoupledESMDriver`` computed its OWN ocean/surface ``cos_zenith`` (the surface energy budget)
from a bare ``day_to_calendar(day)`` — JANUARY-based — so a coupled run with the offset set had
the atmosphere and the ocean surface seeing DIFFERENT seasons (a physically inconsistent sun).
This locks that the coupler now routes through ``_atm._calendar_for_radiation`` so BOTH align.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _coupled_driver(insolation_start_doy, *, diurnal=False, orbit_options=None,
                    surface_radiation=False, precision=None, output_dir=None):
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    from legoesm.driver.coupled_config import CoupledConfig
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    atm = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=4, nlev=4),
        dycore=DycoreConfig(energy_fixer=False, dt=600.0, model_type="hydrostatic",
                            discretization="latlon_cgrid"),
        radiation="gray", convection="none", turbulence="none", days=1,
        insolation_start_doy=insolation_start_doy,
        diurnal_cycle=diurnal, **(orbit_options or {}),
        **({} if precision is None else {"precision": precision}))
    drv = CoupledESMDriver(atm, CoupledConfig(couple_surface_radiation=surface_radiation),
                           output_dir=output_dir)
    drv.setup()
    return _as_after_one_segment(drv)


def _as_after_one_segment(drv):
    """Stash the surface fields an advanced atmosphere segment leaves in ``_carry_aux``:
    the coupler refuses to build surface forcing before one exists."""
    z = jnp.zeros_like(drv._atm.state.p_s.data)
    drv._atm._carry_aux = {"held_sw_net_sfc": z, "held_lw_net_sfc": z, "seg_precip": z}
    if drv.atm_config.diurnal_cycle:
        drv._atm._carry_aux.update(seg_sw_net_sfc=z, seg_lw_net_sfc=z)
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


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("orbit_options", [
    pytest.param({}, id="circular"),
    pytest.param({"orbital_insolation": True}, id="present_day"),
    pytest.param(
        {"orbital_insolation": True, "orbit_eccentricity": 0.2, "orbit_solver": "kepler"},
        id="configured_kepler",
    ),
])
@pytest.mark.parametrize("diurnal", [False, True])
def test_segment_hands_surface_weighted_sun_or_legacy_daily_mean(
        monkeypatch, tmp_path, owned, orbit_options, diurnal):
    """Exercise both real segment callers up to their first surface handoff.

    Reference the diurnal integral with one-second midpoint samples;
    the legacy daily-mean branch must still be bit-for-bit identical.
    """
    from legoesm.atmosphere.physics.radiation.solar import (
        cos_zenith_angle,
        daily_mean_insolation,
        earth_orbit,
        earth_sun_distance_factor,
    )
    from legoesm.coupler import grid_remap
    from legoesm.driver.orbit import experiment_orbit

    drv = _coupled_driver(244, diurnal=diurnal, orbit_options=orbit_options,
                          output_dir=tmp_path)
    # Check orbit resolution independently before using it in the reference.
    expected_orbit = earth_orbit() if orbit_options else None
    if "orbit_solver" in orbit_options:
        expected_orbit = expected_orbit._replace(eccentricity=0.2, solver="kepler")
    orbit = experiment_orbit(drv.atm_config)
    assert orbit == expected_orbit
    handed = []

    class SurfaceHandoffReachedError(Exception):
        pass

    def capture_forcing(forcing, remapper):
        handed.append(forcing)
        raise SurfaceHandoffReachedError

    # Stop at the real handoff, before unrelated ocean/land integration.
    monkeypatch.setattr(grid_remap, "remap_surface_fields", capture_forcing)
    dt = float(drv.coupled_cfg.coupling_dt)
    lat, lon = drv._atm._grid_lat, drv._atm._grid_lon
    for end_hour in (0.25, 7.0, 13.0, 19.0):
        end_day = end_hour / 24.0
        with pytest.raises(SurfaceHandoffReachedError):
            if owned:
                drv._owned_segment(end_day, dt)
            else:
                drv._segment_hook(drv._atm, end_day, dt)
        actual = np.asarray(handed[-1].cos_zenith)
        if diurnal:
            days = end_day - (np.arange(int(dt)) + 0.5) / 86400.0
            doy, seconds = drv._atm._calendar_for_radiation(days)
            raw = jax.vmap(lambda d, s: cos_zenith_angle(
                lat, lon, d, s / 3600.0, orbit=orbit))(doy, seconds)
            mu = np.maximum(np.asarray(raw), 0.0)
            energy = mu.sum(axis=0)
            expected = (mu * mu).sum(axis=0) / np.where(energy > 0.0, energy, 1.0)
            assert np.any(energy == 0.0) and np.any(energy > 0.0)
            np.testing.assert_array_equal(actual[energy == 0.0], 0.0)
            # Bound of five-minute piecewise-linear interpolation of the cosine,
            # (2*pi*5/1440)**2 / 8; the largest difference found on a global
            # scan of latitudes, local times and seasons is 4.0e-5.
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=6e-5)
        else:
            doy, _ = drv._atm._calendar_for_radiation(end_day)
            eccf = earth_sun_distance_factor(float(doy), orbit) if orbit is not None else 1.0
            s0 = drv.atm_config.S_0
            expected = jnp.clip(daily_mean_insolation(
                lat, float(doy), S_0=s0, orbit=orbit) / (eccf * s0), 0.0, 1.0)
            np.testing.assert_array_equal(actual, np.asarray(expected))


@pytest.mark.parametrize("precision", [None, "fp64"], ids=["default", "fp64"])
@pytest.mark.parametrize("peak_cosine", [0.06, 0.0225], ids=["peak_0.06", "peak_0.0225"])
@pytest.mark.parametrize("terminator", ["sunrise", "sunset"])
def test_terminator_land_shortwave_budget(tmp_path, record_property, terminator,
                                          peak_cosine, precision):
    """A one-hour segment that contains only a few minutes of daylight.

    The sunlit part ends (sunrise) or starts (sunset) at ``peak_cosine``. Near
    the horizon the cosine is almost linear in time, so the insolation-weighted
    cosine is 2/3 of the peak: 0.04 and 0.015, the second 1.5 times the
    canopy's night cutoff of 0.01 on any grid. The segment midpoint is dark,
    so a midpoint cosine hands the canopy a night sun together with positive
    shortwave. Fluxes in W/m².
    """
    from legoesm.land.canopy.radiative_transfer import (
        broadband_albedo,
        canopy_shortwave_rt,
        split_sw_components,
    )

    drv = _coupled_driver(80, diurnal=True, surface_radiation=True, precision=precision,
                          output_dir=tmp_path)
    lat, lon = np.broadcast_arrays(drv._atm._grid_lat, drv._atm._grid_lon)
    cell = np.unravel_index(np.argmin(np.abs(lat)), lat.shape)
    # Sunrise and sunset of the host formula in this column, to the second.
    from legoesm.atmosphere.physics.radiation.solar import cos_zenith_angle
    seconds = np.arange(86400.0)
    doy, sod = drv._atm._calendar_for_radiation(seconds / 86400.0)
    cosine = np.asarray(jax.vmap(lambda d, h: cos_zenith_angle(
        lat[cell], lon[cell], d, h / 3600.0))(doy, sod))
    sun = cosine > 0.0
    rise = np.flatnonzero(~np.roll(sun, 1) & sun)[0]
    fall = np.flatnonzero(np.roll(sun, 1) & ~sun)[0]
    # The sunlit part of the segment reaches ``peak_cosine`` at its sunny end.
    if terminator == "sunrise":
        end_seconds = seconds[rise + np.flatnonzero(cosine[rise:] >= peak_cosine)[0]]
    else:
        end_seconds = seconds[np.flatnonzero(cosine[:fall] >= peak_cosine)[-1]] + 3600.0
    end_day = end_seconds / 86400.0
    dt = 3600.0

    # Prescribe an independent atmosphere budget for the selected land column.
    # The dynamic feedback uses the same land albedo for its net-to-gross inversion.
    vis, nir = 0.10, 0.30
    albedo = broadband_albedo(vis, nir)
    down = 10.0
    net = down * (1.0 - albedo)
    grid = jnp.zeros_like(drv._atm.state.p_s.data)
    drv._last_sfc_response = SimpleNamespace(albedo=grid + albedo, T_sfc=grid + 285.0,
                                             lw_up=grid)
    drv._atm._carry_aux["seg_sw_net_sfc"] = grid.at[cell].set(net)
    forcing = drv._build_atm_forcing(end_day, segment_seconds=dt)
    handed, mu = forcing.sw_down[cell], forcing.cos_zenith[cell]
    exact = np.asarray(handed).dtype == np.float64
    np.testing.assert_allclose(handed, down, rtol=1e-12 if exact else 1e-6)
    np.testing.assert_allclose(mu, 2.0 / 3.0 * peak_cosine, rtol=1e-2)
    # The canopy treats a cosine at or below 0.01 as night.
    assert mu > 0.01

    # Independent land side: sum the real sunlit, shaded and soil absorption.
    bands = split_sw_components(handed, mu)
    rt = canopy_shortwave_rt(*bands, jnp.degrees(jnp.arccos(mu)), LAI=3.0, CI=0.8,
                             ALB_VIS=vis, ALB_NIR=nir, Vcmax25_C3_leaf=50.0,
                             Vcmax25_C4_leaf=0.0, kn=0.5)
    absorbed = rt.ASW_Sun + rt.ASW_Sh + rt.ASW_Soil
    for name, value in (("cosine", mu), ("net_w_m2", net), ("absorbed_w_m2", absorbed),
                        ("residual_j_m2", (net - absorbed) * dt)):
        record_property(name, float(value))
    # Roundoff only: the land partition must account for the prescribed net SW.
    np.testing.assert_allclose(absorbed, net, rtol=1e-12 if exact else 1e-6)


@pytest.mark.parametrize("segment_seconds, end_day", [
    pytest.param(1000.0, 0.3, id="not_a_multiple_of_300_s"),
    pytest.param(3600.0, 365.02, id="across_the_year_boundary"),
])
def test_weighted_sun_for_other_segment_lengths_and_the_year_boundary(
        tmp_path, segment_seconds, end_day):
    """Segment lengths that are not a multiple of the 300 s node spacing, and a
    segment that runs across the end of the 365-day calendar year."""
    from legoesm.atmosphere.physics.radiation.solar import cos_zenith_angle
    from legoesm.driver.orbit import experiment_orbit

    drv = _coupled_driver(None, diurnal=True, output_dir=tmp_path)
    lat, lon = drv._atm._grid_lat, drv._atm._grid_lon
    actual = np.asarray(drv._build_atm_forcing(end_day, segment_seconds=segment_seconds).cos_zenith)
    days = end_day - (np.arange(int(segment_seconds)) + 0.5) / 86400.0
    doy, seconds = drv._atm._calendar_for_radiation(days)
    if end_day > 365.0:
        assert doy[0] < 2.0 and doy[-1] > 365.0     # the window holds both sides of the wrap
    orbit = experiment_orbit(drv.atm_config)
    mu = np.maximum(np.asarray(jax.vmap(lambda d, s: cos_zenith_angle(
        lat, lon, d, s / 3600.0, orbit=orbit))(doy, seconds)), 0.0)
    energy = mu.sum(axis=0)
    expected = (mu * mu).sum(axis=0) / np.where(energy > 0.0, energy, 1.0)
    assert np.any(energy == 0.0) and np.any(energy > 0.0)
    np.testing.assert_array_equal(actual[energy == 0.0], 0.0)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=6e-5)


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
