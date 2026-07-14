"""Smoke test for the AMIP experiment with RRTMG radiation.

Verifies that the AMIP pipeline with RRTMGP correlated-k radiation runs
for a few steps without blowup, using synthetic forcing data and a small
grid (C4/L5) to keep CI fast.

Also tests radiation-option selection plumbing and the radiation cadence
(update_interval) mechanism.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.forcing.amip import AMIPForcingConfig, load_amip_forcing, get_forcing_at_time
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel as PrimitiveEquationModel,
    CDGridPrimitiveEquationConfig as PrimitiveEquationConfig,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig, RRTMGPConfig, RadiationConfig,
)
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import rrtmgp_radiation
from legoesm.atmosphere.physics.radiation.solar import daily_mean_insolation
from legoesm.atmosphere.physics.convection.config import SBMConfig
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm import constants
from legoesm.forcing.time_utils import day_to_calendar


def _make_synthetic_forcing(path, n_lat=18, n_lon=36):
    """Create minimal synthetic AMIP forcing."""
    lat = np.linspace(-85, 85, n_lat)
    lon = np.linspace(5, 355, n_lon)
    times = np.array([np.datetime64("2000-01-01") + np.timedelta64(i * 30, "D")
                       for i in range(4)])
    sst = np.full((4, n_lat, n_lon), 293.0, dtype=np.float32)
    sst += lat[None, :, None] * 0.2
    sic = np.zeros((4, n_lat, n_lon), dtype=np.float32)
    sic[:, :2, :] = 0.8
    sic[:, -2:, :] = 0.6

    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), sst),
         "sic": (("time", "lat", "lon"), sic)},
        coords={"time": times, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)


@pytest.fixture(scope="module")
def rrtmg_setup(tmp_path_factory):
    """Set up a minimal AMIP experiment with RRTMG radiation."""
    N, NLEV = 4, 5
    DT = 600.0

    forcing_path = str(tmp_path_factory.mktemp("amip_rrtmg") / "forcing.nc")
    _make_synthetic_forcing(forcing_path)

    grid = create_cubed_sphere(N)
    sigma = create_sigma_coordinate(NLEV)

    config = AMIPForcingConfig(
        path=forcing_path, sst_var="sst", sic_var="sic",
        sst_offset=0.0, sic_scale=1.0,
    )
    forcing = load_amip_forcing(config, grid)

    dycore_config = PrimitiveEquationConfig(
        hyperdiff_coeff=1e15,
        hyperdiff_ps_coeff=1e15,
        use_conservation_fixer=True,
        fix_mass=True,
    )
    model = PrimitiveEquationModel(grid, sigma, dycore_config)
    state = held_suarez_init(grid, sigma, T_init=280.0)

    # Initialize moisture
    p_full = state.p_s.data[..., None] * sigma.sigma_full
    q_sat = saturation_mixing_ratio(state.T.data, p_full)
    q_v = 0.6 * q_sat * sigma.sigma_full ** 2
    q_v = jnp.minimum(q_v, q_sat)

    return {
        "grid": grid, "sigma": sigma, "model": model, "state": state,
        "q_v": q_v, "forcing": forcing, "config": config, "dt": DT, "N": N,
    }


class TestRRTMGRadiationDirect:
    """Test RRTMG radiation backend directly on column data."""

    def test_rrtmg_produces_finite_output(self, rrtmg_setup):
        """RRTMG should produce finite fluxes and heating rates."""
        grid = rrtmg_setup["grid"]
        sigma = rrtmg_setup["sigma"]
        state = rrtmg_setup["state"]
        q_v = rrtmg_setup["q_v"]

        nlev = sigma.n_levels
        ncol = 6 * grid.n * grid.n
        p_full = (state.p_s.data[..., None] * sigma.sigma_full).reshape(ncol, nlev)
        p_half = (state.p_s.data[..., None] * sigma.sigma_half).reshape(ncol, nlev + 1)
        T_col = state.T.data.reshape(ncol, nlev)
        q_v_col = q_v.reshape(ncol, nlev)
        T_sfc = state.T.data[..., -1].reshape(ncol)

        # Effective daily-mean cos zenith at equinox
        cos_zenith = jnp.clip(
            daily_mean_insolation(grid.lat.reshape(ncol), 80.0, 1360.0) / 1360.0,
            0.0, 1.0,
        )

        rrtmg_cfg = RRTMGPConfig(
            co2_ppmv=415.0, ch4_ppbv=1900.0, n2o_ppbv=332.0,
            sfc_emissivity=0.98, sfc_albedo=0.06, S_0=1360.0,
        )

        rad_out = rrtmgp_radiation(
            T=T_col, p_full=p_full, p_half=p_half,
            sfc_temperature=T_sfc, q_v=q_v_col,
            cos_zenith=cos_zenith, config=rrtmg_cfg,
        )

        assert jnp.all(jnp.isfinite(rad_out.heating_rate)), "RRTMG heating rate not finite"
        assert jnp.all(jnp.isfinite(rad_out.lw_flux_up)), "RRTMG LW flux up not finite"
        assert jnp.all(jnp.isfinite(rad_out.sw_flux_down)), "RRTMG SW flux down not finite"
        assert rad_out.heating_rate.shape == (ncol, nlev)
        assert rad_out.lw_flux_up.shape == (ncol, nlev + 1)

    def test_rrtmg_lw_flux_positive(self, rrtmg_setup):
        """Upward LW flux should be positive (emitting upward)."""
        grid = rrtmg_setup["grid"]
        sigma = rrtmg_setup["sigma"]
        state = rrtmg_setup["state"]
        q_v = rrtmg_setup["q_v"]

        nlev = sigma.n_levels
        ncol = 6 * grid.n * grid.n
        p_full = (state.p_s.data[..., None] * sigma.sigma_full).reshape(ncol, nlev)
        p_half = (state.p_s.data[..., None] * sigma.sigma_half).reshape(ncol, nlev + 1)
        T_col = state.T.data.reshape(ncol, nlev)
        q_v_col = q_v.reshape(ncol, nlev)
        T_sfc = state.T.data[..., -1].reshape(ncol)
        cos_zenith = 0.5 * jnp.ones(ncol)

        rrtmg_cfg = RRTMGPConfig(sfc_emissivity=0.98, sfc_albedo=0.06)
        rad_out = rrtmgp_radiation(
            T=T_col, p_full=p_full, p_half=p_half,
            sfc_temperature=T_sfc, q_v=q_v_col,
            cos_zenith=cos_zenith, config=rrtmg_cfg,
        )

        assert float(jnp.min(rad_out.lw_flux_up)) >= 0.0, "LW flux up should be non-negative"

    def test_rrtmg_albedo_override(self, rrtmg_setup):
        """Albedo override should affect SW fluxes."""
        grid = rrtmg_setup["grid"]
        sigma = rrtmg_setup["sigma"]
        state = rrtmg_setup["state"]
        q_v = rrtmg_setup["q_v"]

        nlev = sigma.n_levels
        ncol = 6 * grid.n * grid.n
        p_full = (state.p_s.data[..., None] * sigma.sigma_full).reshape(ncol, nlev)
        p_half = (state.p_s.data[..., None] * sigma.sigma_half).reshape(ncol, nlev + 1)
        T_col = state.T.data.reshape(ncol, nlev)
        q_v_col = q_v.reshape(ncol, nlev)
        T_sfc = state.T.data[..., -1].reshape(ncol)
        cos_zenith = 0.5 * jnp.ones(ncol)

        rrtmg_cfg = RRTMGPConfig(sfc_albedo=0.06)

        # Low albedo
        out_low = rrtmgp_radiation(
            T=T_col, p_full=p_full, p_half=p_half,
            sfc_temperature=T_sfc, q_v=q_v_col,
            cos_zenith=cos_zenith, config=rrtmg_cfg,
            sfc_albedo_override=0.06,
        )
        # High albedo (ice)
        out_high = rrtmgp_radiation(
            T=T_col, p_full=p_full, p_half=p_half,
            sfc_temperature=T_sfc, q_v=q_v_col,
            cos_zenith=cos_zenith, config=rrtmg_cfg,
            sfc_albedo_override=0.65,
        )

        # Higher albedo should reflect more SW upward at TOA
        sw_up_low = float(jnp.mean(out_low.sw_flux_up[:, 0]))
        sw_up_high = float(jnp.mean(out_high.sw_flux_up[:, 0]))
        assert sw_up_high > sw_up_low, (
            f"Higher albedo should reflect more SW: low={sw_up_low:.1f}, high={sw_up_high:.1f}"
        )

    def test_rrtmg_aerosol_coupling_reduces_sw_surface_flux(self, rrtmg_setup):
        """Adding aerosol optical depth should reduce downwelling SW at surface."""
        grid = rrtmg_setup["grid"]
        sigma = rrtmg_setup["sigma"]
        state = rrtmg_setup["state"]
        q_v = rrtmg_setup["q_v"]

        nlev = sigma.n_levels
        ncol = 6 * grid.n * grid.n
        p_full = (state.p_s.data[..., None] * sigma.sigma_full).reshape(ncol, nlev)
        p_half = (state.p_s.data[..., None] * sigma.sigma_half).reshape(ncol, nlev + 1)
        T_col = state.T.data.reshape(ncol, nlev)
        q_v_col = q_v.reshape(ncol, nlev)
        T_sfc = state.T.data[..., -1].reshape(ncol)
        cos_zenith = 0.5 * jnp.ones(ncol)
        cfg = RRTMGPConfig(sfc_albedo=0.06, S_0=1360.0)

        out_clear = rrtmgp_radiation(
            T=T_col, p_full=p_full, p_half=p_half,
            sfc_temperature=T_sfc, q_v=q_v_col,
            cos_zenith=cos_zenith, config=cfg,
        )
        aerosol_od = 0.25 * jnp.ones((ncol, nlev))
        out_hazy = rrtmgp_radiation(
            T=T_col, p_full=p_full, p_half=p_half,
            sfc_temperature=T_sfc, q_v=q_v_col,
            cos_zenith=cos_zenith, config=cfg,
            aerosol_optical_depth=aerosol_od,
        )
        sw_down_clear = float(jnp.mean(out_clear.sw_flux_down[:, -1]))
        sw_down_hazy = float(jnp.mean(out_hazy.sw_flux_down[:, -1]))
        assert sw_down_hazy < sw_down_clear

    def test_rrtmg_spectral_solar_weights_are_consumed(self, rrtmg_setup):
        """Custom solar spectral weights should alter SW fluxes."""
        grid = rrtmg_setup["grid"]
        sigma = rrtmg_setup["sigma"]
        state = rrtmg_setup["state"]
        q_v = rrtmg_setup["q_v"]

        nlev = sigma.n_levels
        ncol = 6 * grid.n * grid.n
        p_full = (state.p_s.data[..., None] * sigma.sigma_full).reshape(ncol, nlev)
        p_half = (state.p_s.data[..., None] * sigma.sigma_half).reshape(ncol, nlev + 1)
        T_col = state.T.data.reshape(ncol, nlev)
        q_v_col = q_v.reshape(ncol, nlev)
        T_sfc = state.T.data[..., -1].reshape(ncol)
        cos_zenith = 0.5 * jnp.ones(ncol)
        cfg = RRTMGPConfig(sfc_albedo=0.06, S_0=1360.0)

        out_default = rrtmgp_radiation(
            T=T_col, p_full=p_full, p_half=p_half,
            sfc_temperature=T_sfc, q_v=q_v_col,
            cos_zenith=cos_zenith, config=cfg,
        )
        # Strongly front-load energy to low g-points to force a measurable change.
        ngpt = 112
        weights = jnp.linspace(2.0, 0.2, ngpt)
        weights = weights / jnp.sum(weights)
        out_custom = rrtmgp_radiation(
            T=T_col, p_full=p_full, p_half=p_half,
            sfc_temperature=T_sfc, q_v=q_v_col,
            cos_zenith=cos_zenith, config=cfg,
            solar_spectral_fraction=weights,
        )
        diff = float(jnp.mean(jnp.abs(out_default.sw_flux_down - out_custom.sw_flux_down)))
        assert diff > 1.0e-8


class TestAMIPWithRRTMG:
    """Integration test: AMIP pipeline with RRTMG radiation."""

    def test_amip_rrtmg_3_steps(self, rrtmg_setup):
        """Run 3 AMIP steps with RRTMG radiation without blowup."""
        grid = rrtmg_setup["grid"]
        sigma = rrtmg_setup["sigma"]
        model = rrtmg_setup["model"]
        state = rrtmg_setup["state"]
        q_v = rrtmg_setup["q_v"]
        forcing = rrtmg_setup["forcing"]
        DT = rrtmg_setup["dt"]

        rrtmg_cfg = RRTMGPConfig(
            co2_ppmv=415.0, ch4_ppbv=1900.0, n2o_ppbv=332.0,
            sfc_emissivity=0.98, sfc_albedo=0.06, S_0=1360.0,
        )
        sbm_cfg = SBMConfig(tau_c=7200.0, rh_ref=0.7)

        sigma_full = sigma.sigma_full
        sigma_half = sigma.sigma_half
        dsigma = sigma.dsigma
        T_ice = 271.35
        albedo_ice = 0.65
        albedo_ocean = 0.06

        for step in range(3):
            day = step * DT / 86400.0
            day_of_year, _ = day_to_calendar(day)
            sst, sic = get_forcing_at_time(forcing, day)

            # Dynamics
            state = model.step_with_physics(state, DT)

            nlev = sigma_full.shape[0]
            ncol = 6 * grid.n * grid.n
            T_sfc = sic * T_ice + (1.0 - sic) * sst
            albedo = sic * albedo_ice + (1.0 - sic) * albedo_ocean
            p_full = state.p_s.data[..., None] * sigma_full
            p_half = state.p_s.data[..., None] * sigma_half

            insol = daily_mean_insolation(grid.lat.reshape(ncol), day_of_year, 1360.0)
            cos_zenith = jnp.clip(insol / 1360.0, 0.0, 1.0)

            rad_out = rrtmgp_radiation(
                T=state.T.data.reshape(ncol, nlev),
                p_full=p_full.reshape(ncol, nlev),
                p_half=p_half.reshape(ncol, nlev + 1),
                sfc_temperature=T_sfc.reshape(ncol),
                q_v=q_v.reshape(ncol, nlev),
                cos_zenith=cos_zenith,
                config=rrtmg_cfg,
                sfc_albedo_override=float(jnp.mean(albedo)),
            )
            dT_rad = rad_out.heating_rate.reshape(state.T.data.shape)
            new_T = state.T.data + DT * dT_rad

            # Convection
            conv_out = sbm_convection(
                T=state.T.data.reshape(ncol, nlev),
                q_v=q_v.reshape(ncol, nlev),
                p_full=p_full.reshape(ncol, nlev),
                p_half=p_half.reshape(ncol, nlev + 1),
                dt=DT, config=sbm_cfg,
            )
            new_T = new_T + DT * conv_out.dT_dt.reshape(state.T.data.shape)
            q_v = q_v + DT * conv_out.dq_v_dt.reshape(q_v.shape)

            # Large-scale condensation
            q_sat = saturation_mixing_ratio(new_T, p_full)
            excess = jnp.maximum(q_v - q_sat, 0.0)
            q_v = q_v - excess
            new_T = new_T + constants.L_v * excess / constants.c_pd
            state = state._replace(T=state.T.replace(data=new_T))
            q_v = jnp.maximum(q_v, 0.0)

        # Verify everything is finite
        assert jnp.all(jnp.isfinite(state.T.data)), "Temperature not finite after RRTMG"
        assert jnp.all(jnp.isfinite(state.u.data)), "Winds not finite after RRTMG"
        assert jnp.all(q_v >= 0), "Negative humidity after RRTMG"

        T_min = float(jnp.min(state.T.data))
        T_max = float(jnp.max(state.T.data))
        assert T_min > 100, f"T_min={T_min} too cold"
        assert T_max < 500, f"T_max={T_max} too hot"


class TestRadiationOptionSelection:
    """Test that radiation scheme selection works correctly."""

    def test_gray_and_rrtmg_produce_different_output(self, rrtmg_setup):
        """Gray and RRTMG should produce meaningfully different heating rates."""
        grid = rrtmg_setup["grid"]
        sigma = rrtmg_setup["sigma"]
        state = rrtmg_setup["state"]
        q_v = rrtmg_setup["q_v"]

        nlev = sigma.n_levels
        ncol = 6 * grid.n * grid.n
        p_full = (state.p_s.data[..., None] * sigma.sigma_full).reshape(ncol, nlev)
        p_half = (state.p_s.data[..., None] * sigma.sigma_half).reshape(ncol, nlev + 1)
        T_col = state.T.data.reshape(ncol, nlev)
        q_v_col = q_v.reshape(ncol, nlev)
        T_sfc = state.T.data[..., -1].reshape(ncol)
        lat_col = grid.lat.reshape(ncol)

        insol = daily_mean_insolation(lat_col, 80.0, 1360.0)

        # Gray radiation
        gray_cfg = GrayRadiationConfig(
            tau_equator=7.2, tau_pole=1.8, S_0=1360.0,
            sfc_albedo=0.06, perpetual_equinox=False,
        )
        gray_out = gray_radiation(
            T=T_col, p_full=p_full, p_half=p_half,
            sfc_temperature=T_sfc, lat=lat_col,
            q_v=q_v_col, insolation=insol, config=gray_cfg,
        )

        # RRTMG radiation
        cos_zenith = jnp.clip(insol / 1360.0, 0.0, 1.0)
        rrtmg_cfg = RRTMGPConfig(sfc_albedo=0.06, S_0=1360.0)
        rrtmg_out = rrtmgp_radiation(
            T=T_col, p_full=p_full, p_half=p_half,
            sfc_temperature=T_sfc, q_v=q_v_col,
            cos_zenith=cos_zenith, config=rrtmg_cfg,
        )

        # Both should be finite
        assert jnp.all(jnp.isfinite(gray_out.heating_rate))
        assert jnp.all(jnp.isfinite(rrtmg_out.heating_rate))

        # They should produce different results (not identical)
        diff = float(jnp.max(jnp.abs(gray_out.heating_rate - rrtmg_out.heating_rate)))
        assert diff > 1e-8, f"Gray and RRTMG should differ, max diff = {diff}"

    def test_radiation_config_scheme_selection(self):
        """RadiationConfig correctly selects gray vs rrtmgp."""
        cfg_gray = RadiationConfig(scheme="gray")
        assert cfg_gray.scheme == "gray"

        cfg_rrtmg = RadiationConfig(scheme="rrtmgp")
        assert cfg_rrtmg.scheme == "rrtmgp"

    def test_amip_config_radiation_field(self):
        """AMIPExperimentConfig has radiation field."""
        from legoesm.forcing.amip_config import AMIPExperimentConfig
        cfg = AMIPExperimentConfig(radiation="rrtmg", rad_update_steps=3)
        assert cfg.radiation == "rrtmg"
        assert cfg.rad_update_steps == 3

    def test_amip_config_gas_concentrations(self):
        """AMIPExperimentConfig exposes gas concentrations."""
        from legoesm.forcing.amip_config import AMIPExperimentConfig
        cfg = AMIPExperimentConfig(co2_ppmv=560.0)
        assert cfg.co2_ppmv == 560.0
        assert cfg.ch4_ppbv == 1900.0  # default
