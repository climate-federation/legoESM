"""Category 2: Multi-Layer Land -- Soil Thermal & Hydraulic Physics.

Tests the multi-layer soil model including thermal diffusion,
Richards equation, and root water uptake.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.land.multilayer_land import step_multilayer_land, init_multilayer_land_state
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.state import MultiLayerLandState
from legoesm.land.soil_thermal import (
    solve_soil_thermal, compute_heat_capacity, compute_thermal_conductivity,
    SoilThermalConfig,
)
from legoesm.land.soil_hydraulics import (
    theta_from_psi, psi_from_theta, hydraulic_conductivity,
    SoilHydraulicsConfig,
)
from legoesm.land.richards import solve_richards, RichardsConfig, RichardsOutput
from legoesm.land.soil_grid import make_soil_grid, SoilGridConfig, SoilGrid
from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm import constants


NCOL = 8
CONFIG = MultiLayerLandConfig()


def make_forcing(
    ncol=NCOL,
    sw_down=300.0, lw_down=300.0,
    T_lowest=280.0, q_lowest=0.008,
    u_lowest=5.0, v_lowest=2.0,
    p_lowest=95000.0, p_surface=1e5,
    rho_lowest=1.2, cos_zenith=0.5,
    co2_ppmv=415.0, precip_total=0.0,
    precip_snow=0.0,
) -> AtmToSurface:
    f = jnp.float64
    s = (ncol,)
    return AtmToSurface(
        sw_down=jnp.full(s, sw_down, f),
        lw_down=jnp.full(s, lw_down, f),
        precip_total=jnp.full(s, precip_total, f),
        precip_snow=jnp.full(s, precip_snow, f),
        T_lowest=jnp.full(s, T_lowest, f),
        q_lowest=jnp.full(s, q_lowest, f),
        u_lowest=jnp.full(s, u_lowest, f),
        v_lowest=jnp.full(s, v_lowest, f),
        p_lowest=jnp.full(s, p_lowest, f),
        p_surface=jnp.full(s, p_surface, f),
        rho_lowest=jnp.full(s, rho_lowest, f),
        cos_zenith=jnp.full(s, cos_zenith, f),
        co2_ppmv=jnp.full(s, co2_ppmv, f),
        has_radiation=jnp.full(s, 1.0, f),
        has_precipitation=jnp.full(s, 1.0, f),
    )


# ===================================================================
# 2a  Smoke test -- step_multilayer_land runs
# ===================================================================


class Test2a_SmokeMultilayer:
    def test_returns_finite(self):
        state = init_multilayer_land_state(NCOL, CONFIG, T_init=280.0)
        forcing = make_forcing()
        new_state, response, _ = step_multilayer_land(
            state, forcing, CONFIG, U_min=1.0, dt=1800.0,
        )
        for name in MultiLayerLandState._fields:
            arr = getattr(new_state, name)
            if arr is None:            # feature-gated fields (ponding, elevation bands)
                continue
            assert jnp.all(jnp.isfinite(arr)), f"MultiLayerLandState.{name} has non-finite values"
        for name in TileResponse._fields:
            arr = getattr(response, name)
            if arr is None:
                continue
            assert jnp.all(jnp.isfinite(arr)), f"TileResponse.{name} has non-finite values"

    def test_T_soil_bounded(self):
        state = init_multilayer_land_state(NCOL, CONFIG, T_init=280.0)
        forcing = make_forcing()
        new_state, _, _ = step_multilayer_land(state, forcing, CONFIG, U_min=1.0, dt=1800.0)
        assert jnp.all(new_state.T_soil > 150.0)
        assert jnp.all(new_state.T_soil < 400.0)

    def test_theta_bounded(self):
        state = init_multilayer_land_state(NCOL, CONFIG, T_init=280.0)
        forcing = make_forcing()
        new_state, _, _ = step_multilayer_land(state, forcing, CONFIG, U_min=1.0, dt=1800.0)
        assert jnp.all(new_state.theta_soil >= CONFIG.hydraulics.theta_r - 1e-10)
        assert jnp.all(new_state.theta_soil <= CONFIG.hydraulics.theta_sat + 1e-10)


# ===================================================================
# 2b  Soil thermal diffusion -- uniform T steady state
# ===================================================================


class Test2b_ThermalSteadyState:
    def test_uniform_T_no_change(self):
        """Uniform T with G=0, Q_geo=0 should not change."""
        grid = make_soil_grid(CONFIG.soil_grid)
        ncol, nlayers = 4, grid.n_layers
        T_soil = jnp.full((ncol, nlayers), 280.0)
        theta = jnp.full((ncol, nlayers), 0.2)
        G_surface = jnp.zeros(ncol)
        thermal_cfg = SoilThermalConfig(Q_geothermal=0.0)
        T_new = solve_soil_thermal(T_soil, theta, grid, CONFIG.hydraulics, thermal_cfg, G_surface, dt=3600.0)
        assert jnp.all(jnp.abs(T_new - T_soil) < 1e-10)


# ===================================================================
# 2c  Heat penetration
# ===================================================================


class Test2c_HeatPenetration:
    def test_surface_heating(self):
        """Strong G_surface heats top layer; bottom barely changes."""
        grid = make_soil_grid(CONFIG.soil_grid)
        ncol, nlayers = 4, grid.n_layers
        T_soil = jnp.full((ncol, nlayers), 280.0)
        theta = jnp.full((ncol, nlayers), 0.2)
        G_surface = jnp.full(ncol, 100.0)
        thermal_cfg = SoilThermalConfig(Q_geothermal=0.0)
        T_new = solve_soil_thermal(T_soil, theta, grid, CONFIG.hydraulics, thermal_cfg, G_surface, dt=3600.0)
        dT_top = T_new[:, 0] - T_soil[:, 0]
        assert jnp.all(dT_top > 0)
        dT_bot = jnp.abs(T_new[:, -1] - T_soil[:, -1])
        assert jnp.all(dT_bot < 0.1 * jnp.abs(dT_top))


# ===================================================================
# 2d  Soil thermal energy conservation
# ===================================================================


class Test2d_ThermalEnergyConservation:
    def test_energy_conserved(self):
        grid = make_soil_grid(CONFIG.soil_grid)
        ncol, nlayers = 4, grid.n_layers
        T_soil = jnp.full((ncol, nlayers), 280.0)
        theta = jnp.full((ncol, nlayers), 0.2)
        G_surface = jnp.full(ncol, 50.0)
        thermal_cfg = CONFIG.thermal
        dt = 3600.0
        C_eff = compute_heat_capacity(theta, CONFIG.hydraulics, thermal_cfg)
        dz = grid.dz
        E_old = jnp.sum(C_eff * T_soil * dz[None, :], axis=1)
        T_new = solve_soil_thermal(T_soil, theta, grid, CONFIG.hydraulics, thermal_cfg, G_surface, dt)
        E_new = jnp.sum(C_eff * T_new * dz[None, :], axis=1)
        E_input = (G_surface + thermal_cfg.Q_geothermal) * dt
        rel_err = jnp.abs(E_new - E_old - E_input) / jnp.maximum(jnp.abs(E_input), 1e-12)
        assert jnp.all(rel_err < 0.01)


# ===================================================================
# 2e  Heat capacity and conductivity
# ===================================================================


class Test2e_ThermalProperties:
    def test_heat_capacity_wet_gt_dry(self):
        C_dry = compute_heat_capacity(jnp.array([0.0]), CONFIG.hydraulics, CONFIG.thermal)
        C_wet = compute_heat_capacity(jnp.array([CONFIG.hydraulics.theta_sat]), CONFIG.hydraulics, CONFIG.thermal)
        assert float(C_wet[0]) > float(C_dry[0])

    def test_heat_capacity_range(self):
        theta = jnp.array([0.0, 0.1, 0.2, 0.3, CONFIG.hydraulics.theta_sat])
        C = compute_heat_capacity(theta, CONFIG.hydraulics, CONFIG.thermal)
        assert jnp.all(C > 1e5)
        assert jnp.all(C < 5e6)

    def test_conductivity_wet_gt_dry(self):
        k_dry = compute_thermal_conductivity(jnp.array([CONFIG.hydraulics.theta_r + 0.01]), CONFIG.hydraulics, CONFIG.thermal)
        k_wet = compute_thermal_conductivity(jnp.array([CONFIG.hydraulics.theta_sat]), CONFIG.hydraulics, CONFIG.thermal)
        assert float(k_wet[0]) > float(k_dry[0])

    def test_conductivity_range(self):
        theta = jnp.array([CONFIG.hydraulics.theta_r + 0.01, 0.2, CONFIG.hydraulics.theta_sat])
        k = compute_thermal_conductivity(theta, CONFIG.hydraulics, CONFIG.thermal)
        assert jnp.all(k > 0.1)
        assert jnp.all(k < 5.0)


# ===================================================================
# 2f  Richards equation -- steady state
# ===================================================================


class Test2f_RichardsSteadyState:
    def test_no_change_with_zero_flux(self):
        grid = make_soil_grid(CONFIG.soil_grid)
        ncol, nlayers = 4, grid.n_layers
        hydro = CONFIG.hydraulics
        theta_init = 0.5 * (hydro.theta_r + hydro.theta_sat)
        theta = jnp.full((ncol, nlayers), theta_init)
        psi = psi_from_theta(theta, hydro)
        flux_top = jnp.zeros(ncol)
        sink = jnp.zeros((ncol, nlayers))
        out = solve_richards(psi, theta, grid, hydro, RichardsConfig(bottom_bc="zero_flux"), flux_top, sink, dt=1800.0)
        assert jnp.max(jnp.abs(out.psi_new - psi)) < 0.01


# ===================================================================
# 2g  Richards equation -- infiltration
# ===================================================================


class Test2g_Infiltration:
    def test_moderate_infiltration_low_runoff(self):
        """Moderate flux_top well below capacity => small or zero runoff."""
        grid = make_soil_grid(CONFIG.soil_grid)
        ncol, nlayers = 4, grid.n_layers
        hydro = CONFIG.hydraulics
        # Use moderately wet soil so top layer has room to absorb
        theta_init = 0.5 * (hydro.theta_r + hydro.theta_sat)
        theta = jnp.full((ncol, nlayers), theta_init)
        psi = psi_from_theta(theta, hydro)
        # Very small flux
        flux_top = jnp.full(ncol, 0.01 * hydro.K_sat)
        sink = jnp.zeros((ncol, nlayers))
        out = solve_richards(psi, theta, grid, hydro, CONFIG.richards, flux_top, sink, dt=1800.0)
        # Should have very little runoff (most water infiltrates)
        assert jnp.all(out.runoff_surface < 0.01), (
            f"Runoff too high: {float(jnp.max(out.runoff_surface))}"
        )

    def test_excessive_infiltration_runoff(self):
        """Flux_top >> K_sat => surface runoff occurs."""
        grid = make_soil_grid(CONFIG.soil_grid)
        ncol, nlayers = 4, grid.n_layers
        hydro = CONFIG.hydraulics
        theta_init = 0.5 * (hydro.theta_r + hydro.theta_sat)
        theta = jnp.full((ncol, nlayers), theta_init)
        psi = psi_from_theta(theta, hydro)
        flux_top = jnp.full(ncol, 100.0 * hydro.K_sat)
        sink = jnp.zeros((ncol, nlayers))
        out = solve_richards(psi, theta, grid, hydro, CONFIG.richards, flux_top, sink, dt=1800.0)
        assert jnp.all(out.runoff_surface > 0.0)


# ===================================================================
# 2h  Richards equation -- free drainage
# ===================================================================


class Test2h_FreeDrainage:
    def test_free_drainage_drains(self):
        grid = make_soil_grid(CONFIG.soil_grid)
        ncol, nlayers = 4, grid.n_layers
        hydro = CONFIG.hydraulics
        theta_init = hydro.theta_sat * 0.9
        theta = jnp.full((ncol, nlayers), theta_init)
        psi = psi_from_theta(theta, hydro)
        flux_top = jnp.zeros(ncol)
        sink = jnp.zeros((ncol, nlayers))
        out = solve_richards(psi, theta, grid, hydro, RichardsConfig(bottom_bc="free_drainage"), flux_top, sink, dt=1800.0)
        assert jnp.all(out.runoff_subsurface > 0.0)

    def test_zero_flux_no_drain(self):
        grid = make_soil_grid(CONFIG.soil_grid)
        ncol, nlayers = 4, grid.n_layers
        hydro = CONFIG.hydraulics
        theta_init = hydro.theta_sat * 0.9
        theta = jnp.full((ncol, nlayers), theta_init)
        psi = psi_from_theta(theta, hydro)
        flux_top = jnp.zeros(ncol)
        sink = jnp.zeros((ncol, nlayers))
        out = solve_richards(psi, theta, grid, hydro, RichardsConfig(bottom_bc="zero_flux"), flux_top, sink, dt=1800.0)
        assert jnp.allclose(out.runoff_subsurface, 0.0, atol=1e-10)


# ===================================================================
# 2i  Richards equation -- water conservation
# ===================================================================


class Test2i_WaterConservation:
    def test_water_budget(self):
        grid = make_soil_grid(CONFIG.soil_grid)
        ncol, nlayers = 4, grid.n_layers
        hydro = CONFIG.hydraulics
        dt = 1800.0
        theta_init = 0.5 * (hydro.theta_r + hydro.theta_sat)
        theta = jnp.full((ncol, nlayers), theta_init)
        psi = psi_from_theta(theta, hydro)
        flux_top = jnp.full(ncol, 0.5 * hydro.K_sat)
        sink = jnp.zeros((ncol, nlayers))
        out = solve_richards(psi, theta, grid, hydro, CONFIG.richards, flux_top, sink, dt)
        dz = grid.dz
        rho_w = 1000.0
        W_old = jnp.sum(theta * dz[None, :], axis=1) * rho_w
        W_new = jnp.sum(out.theta_new * dz[None, :], axis=1) * rho_w
        dW = W_new - W_old
        inputs = flux_top * dt * rho_w
        outputs = (out.runoff_surface + out.runoff_subsurface) * dt
        residual = jnp.abs(dW - (inputs - outputs))
        budget = jnp.maximum(jnp.abs(inputs), 1e-12)
        rel_err = residual / budget
        assert jnp.all(rel_err < 0.05)


# ===================================================================
# 2j  Root water uptake
# ===================================================================


class Test2j_RootWaterUptake:
    def test_root_zone_dries(self):
        grid = make_soil_grid(CONFIG.soil_grid)
        ncol, nlayers = 4, grid.n_layers
        hydro = CONFIG.hydraulics
        theta_init = 0.8 * hydro.theta_sat
        theta = jnp.full((ncol, nlayers), theta_init)
        psi = psi_from_theta(theta, hydro)
        flux_top = jnp.zeros(ncol)
        sink = jnp.zeros((ncol, nlayers))
        sink = sink.at[:, 0].set(1e-5)
        sink = sink.at[:, 1].set(5e-6)
        out = solve_richards(psi, theta, grid, hydro, RichardsConfig(bottom_bc="zero_flux"), flux_top, sink, dt=1800.0)
        assert jnp.all(out.theta_new[:, 0] < theta_init)
        dtheta_top = jnp.abs(out.theta_new[:, 0] - theta_init)
        dtheta_bot = jnp.abs(out.theta_new[:, -1] - theta_init)
        assert jnp.all(dtheta_bot < dtheta_top)
