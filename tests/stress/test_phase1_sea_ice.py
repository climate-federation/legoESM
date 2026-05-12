"""Phase 1A: Sea ice component stress tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.ice.sea_ice import step_sea_ice
from legoesm.ice.state import SeaIceState, DynamicSeaIceState, init_dynamic_ice_state
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.dynamics import evp_solver, free_drift_velocity
from legoesm.ice.transport import advect_ice_tracers
from legoesm.ice.itd import (
    category_bounds,
    upper_bounds,
    linear_remap,
    aggregate_state,
    distribute_to_categories,
)
from legoesm.coupler.coupling_fields import AtmToSurface
from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import create_cubed_sphere


def _make_forcing(shape, T_lowest=250.0, sw_down=50.0, lw_down=200.0):
    return AtmToSurface(
        sw_down=jnp.full(shape, sw_down),
        lw_down=jnp.full(shape, lw_down),
        precip_total=jnp.zeros(shape),
        precip_snow=jnp.zeros(shape),
        T_lowest=jnp.full(shape, T_lowest),
        q_lowest=jnp.full(shape, 0.001),
        u_lowest=jnp.full(shape, 5.0),
        v_lowest=jnp.zeros(shape),
        p_lowest=jnp.full(shape, 95000.0),
        p_surface=jnp.full(shape, 101325.0),
        rho_lowest=jnp.full(shape, 1.15),
        cos_zenith=jnp.full(shape, 0.5),
        co2_ppmv=jnp.full(shape, 415.0),
        has_radiation=jnp.ones(shape),
        has_precipitation=jnp.zeros(shape),
    )


class TestSeaIceComponent:
    """Comprehensive stress tests for the sea ice component."""

    # ------------------------------------------------------------------
    # 1A.1  Thermodynamic growth under cold forcing
    # ------------------------------------------------------------------
    def test_thermo_growth(self):
        """Cold forcing over freezing ocean should grow existing ice."""
        n = 4
        shape = (6, n, n)
        config = SeaIceConfig(dynamics="none")
        # Extremely cold air, no solar radiation, low LW => strong freezing.
        forcing = _make_forcing(shape, T_lowest=230.0, sw_down=0.0, lw_down=100.0)
        # Ocean at freezing point so ocean heat flux is zero
        ocean_sst = jnp.full(shape, config.T_freeze_ocean)
        ocean_u = jnp.zeros(shape)
        ocean_v = jnp.zeros(shape)

        # Start with moderate ice; under extreme cold it will grow.
        h_init = 1.0
        conc_init = 0.8
        state = SeaIceState(
            h_ice=Field(jnp.full(shape, h_init), name="h_ice"),
            T_ice=Field(jnp.full(shape, 260.0), name="T_ice"),
            concentration=Field(jnp.full(shape, conc_init), name="concentration"),
        )

        dt = 3600.0
        for _ in range(100):
            state, _ = step_sea_ice(
                state, forcing, ocean_sst, ocean_u, ocean_v,
                config, U_min=1.0, dt=dt,
            )

        h = state.h_ice.data
        conc = state.concentration.data
        T = state.T_ice.data

        assert jnp.all(jnp.isfinite(h)), "h_ice contains non-finite values"
        assert jnp.all(jnp.isfinite(conc)), "concentration contains non-finite values"
        assert jnp.all(jnp.isfinite(T)), "T_ice contains non-finite values"

        # Ice should have grown from initial thickness
        assert jnp.all(h > h_init), (
            f"Ice should have grown: min h = {jnp.min(h).item():.4f}"
        )
        # Concentration should have increased
        assert jnp.all(conc > conc_init), (
            f"Concentration should have increased: min conc = {jnp.min(conc).item():.4f}"
        )
        # Temperature should be in [T_ice_min, T_freeze_ocean]
        assert jnp.all(T >= config.T_ice_min), (
            f"T_ice below minimum: min T = {jnp.min(T).item():.2f}"
        )
        assert jnp.all(T <= config.T_freeze_ocean), (
            f"T_ice above freezing: max T = {jnp.max(T).item():.2f}"
        )

    # ------------------------------------------------------------------
    # 1A.2  Thermodynamic melt under warm forcing
    # ------------------------------------------------------------------
    def test_thermo_melt(self):
        """Warm forcing should melt ice and decrease concentration."""
        n = 4
        shape = (6, n, n)
        config = SeaIceConfig(dynamics="none")
        forcing = _make_forcing(shape, T_lowest=280.0, sw_down=300.0, lw_down=350.0)
        ocean_sst = jnp.full(shape, 274.0)
        ocean_u = jnp.zeros(shape)
        ocean_v = jnp.zeros(shape)

        state = SeaIceState(
            h_ice=Field(jnp.full(shape, 1.0), name="h_ice"),
            T_ice=Field(jnp.full(shape, 268.0), name="T_ice"),
            concentration=Field(jnp.full(shape, 0.9), name="concentration"),
        )

        dt = 3600.0
        for _ in range(200):
            state, _ = step_sea_ice(
                state, forcing, ocean_sst, ocean_u, ocean_v,
                config, U_min=1.0, dt=dt,
            )

        h = state.h_ice.data
        conc = state.concentration.data

        assert jnp.all(jnp.isfinite(h)), "h_ice contains non-finite values"
        assert jnp.all(h >= 0.0), (
            f"Negative ice thickness: min h = {jnp.min(h).item():.6f}"
        )
        # Ice should have decreased from 1.0 m
        assert jnp.all(h < 1.0), (
            f"Ice should have melted: max h = {jnp.max(h).item():.4f}"
        )
        # Concentration should have decreased from 0.9
        assert jnp.all(conc < 0.9), (
            f"Concentration should have decreased: max conc = {jnp.max(conc).item():.4f}"
        )

    # ------------------------------------------------------------------
    # 1A.3  Energy conservation (approximate check)
    # ------------------------------------------------------------------
    def test_energy_conservation(self):
        """Single step: energy quantities should be finite and non-zero,
        and the sign of dE should be consistent with ice thickness change."""
        n = 4
        shape = (6, n, n)
        config = SeaIceConfig(dynamics="none")
        forcing = _make_forcing(shape, T_lowest=260.0, sw_down=100.0, lw_down=250.0)
        ocean_sst = jnp.full(shape, config.T_freeze_ocean)
        ocean_u = jnp.zeros(shape)
        ocean_v = jnp.zeros(shape)

        h0 = 0.5
        T0 = 268.0
        conc0 = 0.5
        state = SeaIceState(
            h_ice=Field(jnp.full(shape, h0), name="h_ice"),
            T_ice=Field(jnp.full(shape, T0), name="T_ice"),
            concentration=Field(jnp.full(shape, conc0), name="concentration"),
        )

        dt = 3600.0
        rho_ice = config.rho_ice
        c_ice = config.c_ice
        L_f = config.L_f

        # Ice enthalpy: E = rho_ice * conc * h * (c_ice * T + L_f)
        E_before = rho_ice * conc0 * h0 * (c_ice * T0 + L_f)

        new_state, _ = step_sea_ice(
            state, forcing, ocean_sst, ocean_u, ocean_v,
            config, U_min=1.0, dt=dt,
        )

        h1 = new_state.h_ice.data
        T1 = new_state.T_ice.data
        conc1 = new_state.concentration.data

        E_after = rho_ice * conc1 * h1 * (c_ice * T1 + L_f)
        dE = E_after - E_before
        dV = h1 * conc1 - h0 * conc0  # volume change

        assert jnp.all(jnp.isfinite(E_before)), "E_before not finite"
        assert jnp.all(jnp.isfinite(E_after)), "E_after not finite"
        assert jnp.all(jnp.isfinite(dE)), "dE not finite"
        assert jnp.any(dE != 0.0), "Energy change should be non-zero"
        # Energy change and volume change should have the same sign:
        # more ice => more stored energy, less ice => less stored energy.
        mean_dE = jnp.mean(dE).item()
        mean_dV = jnp.mean(dV).item()
        assert mean_dE * mean_dV >= 0.0, (
            f"dE and dV signs should agree: mean dE = {mean_dE:.2f}, mean dV = {mean_dV:.6f}"
        )

    # ------------------------------------------------------------------
    # 1A.4  EVP dynamics produce bounded, finite velocities
    # ------------------------------------------------------------------
    def test_evp_dynamics(self):
        """EVP solver should produce finite, bounded ice velocities and stresses."""
        n = 8
        shape = (6, n, n)
        grid = create_cubed_sphere(n)

        h_ice = jnp.full(shape, 1.5)
        concentration = jnp.full(shape, 0.95)
        u_ice = jnp.zeros(shape)
        v_ice = jnp.zeros(shape)
        sigma_11 = jnp.zeros(shape)
        sigma_22 = jnp.zeros(shape)
        sigma_12 = jnp.zeros(shape)

        wind_u = jnp.full(shape, 5.0)
        wind_v = jnp.full(shape, 2.0)
        ocean_u = jnp.full(shape, 0.1)
        ocean_v = jnp.zeros(shape)

        dt = 3600.0
        N_evp = 60

        u_out, v_out, s11_out, s22_out, s12_out = evp_solver(
            u_ice, v_ice, sigma_11, sigma_22, sigma_12,
            h_ice, concentration,
            wind_u, wind_v,
            ocean_u, ocean_v,
            grid, dt,
            N_evp=N_evp,
        )

        # All outputs should be finite
        assert jnp.all(jnp.isfinite(u_out)), "u_ice has non-finite values"
        assert jnp.all(jnp.isfinite(v_out)), "v_ice has non-finite values"
        assert jnp.all(jnp.isfinite(s11_out)), "sigma_11 has non-finite values"
        assert jnp.all(jnp.isfinite(s22_out)), "sigma_22 has non-finite values"
        assert jnp.all(jnp.isfinite(s12_out)), "sigma_12 has non-finite values"

        # Ice velocity should be bounded: less than 2 m/s for these forcing conditions
        speed = jnp.sqrt(u_out**2 + v_out**2)
        assert jnp.all(speed < 2.0), (
            f"Ice speed too large: max = {jnp.max(speed).item():.4f} m/s"
        )

    # ------------------------------------------------------------------
    # 1A.5  ITD linear remap conserves volume
    # ------------------------------------------------------------------
    def test_itd_volume_conservation(self):
        """Linear remapping should conserve total ice volume across categories."""
        n_cat = 5
        shape = (6, 4, 4)

        # Distribute a uniform slab into 5 categories
        h_single = jnp.full(shape, 1.0)
        T_single = jnp.full(shape, 268.0)
        conc_single = jnp.full(shape, 0.8)

        h_mc, T_mc, conc_mc = distribute_to_categories(
            h_single, T_single, conc_single, n_cat,
        )

        # Perturb thickness in first category (simulate growth)
        h_new = h_mc.at[..., 0].add(0.05)
        a_new = conc_mc  # concentrations unchanged before remap

        # Volume before remap
        vol_before = jnp.sum(h_new * a_new)

        # Apply linear remapping
        h_remap, a_remap, T_remap = linear_remap(
            h_mc, conc_mc, h_new, a_new, n_cat, T_new=T_mc,
        )

        # Volume after remap
        vol_after = jnp.sum(h_remap * a_remap)

        rel_err = jnp.abs(vol_after - vol_before) / jnp.maximum(jnp.abs(vol_before), 1e-20)
        assert rel_err < 1e-10, (
            f"Volume not conserved: relative error = {rel_err.item():.2e}"
        )

    # ------------------------------------------------------------------
    # 1A.6  Transport conserves volume (approximate)
    # ------------------------------------------------------------------
    def test_transport_volume_conservation(self):
        """Ice advection on a small cubed-sphere grid should approximately
        conserve area-weighted volume."""
        n = 8
        shape = (6, n, n)
        grid = create_cubed_sphere(n)

        h_ice = jnp.full(shape, 1.0)
        concentration = jnp.full(shape, 0.8)
        T_ice = jnp.full(shape, 268.0)
        u_ice = jnp.full(shape, 0.1)
        v_ice = jnp.zeros(shape)

        # Area-weighted volume before
        vol_before = jnp.sum(h_ice * concentration * grid.area)

        dt = 3600.0
        h, conc, T = h_ice, concentration, T_ice
        for _ in range(10):
            h, conc, T = advect_ice_tracers(h, conc, T, u_ice, v_ice, grid, dt)

        # Area-weighted volume after
        vol_after = jnp.sum(h * conc * grid.area)

        rel_change = jnp.abs(vol_after - vol_before) / jnp.maximum(jnp.abs(vol_before), 1e-20)
        assert rel_change < 1e-5, (
            f"Volume change too large: relative = {rel_change.item():.2e}"
        )

    # ------------------------------------------------------------------
    # 1A.7  Free-drift velocity bounds
    # ------------------------------------------------------------------
    def test_free_drift_bounds(self):
        """Free-drift velocity should have physically reasonable magnitude.

        The formula is: u_ice = drag_ocean * u_ocean + drag_atm * (rho_air/rho_ice) * u_wind.
        Both drag coefficients are O(1e-3), so ice speed is much smaller than
        either wind or ocean current.
        """
        shape = (6, 4, 4)
        drag_ocean = 5.5e-3
        drag_atm = 1.3e-3
        rho_air = constants.rho_air
        rho_ice = constants.rho_ice

        # Case 1: nonzero wind, small ocean current
        wind_u = jnp.full(shape, 10.0)
        wind_v = jnp.zeros(shape)
        ocean_u = jnp.full(shape, 0.1)
        ocean_v = jnp.zeros(shape)

        u_ice, v_ice = free_drift_velocity(
            ocean_u, ocean_v, wind_u, wind_v,
        )
        speed = jnp.sqrt(u_ice**2 + v_ice**2)

        # Ice speed should be positive (both forces push in +x)
        assert jnp.all(speed > 0.0), "Ice speed should be nonzero"

        # Ice speed should be much less than wind speed (drag << 1)
        wind_speed = jnp.sqrt(wind_u**2 + wind_v**2)
        assert jnp.all(speed < wind_speed), (
            f"Ice speed should be less than wind speed: "
            f"max ice = {jnp.max(speed).item():.4f}"
        )

        # Check expected magnitude:
        # drag_ocean * 0.1 + drag_atm * (rho_air/rho_ice) * 10
        expected = drag_ocean * 0.1 + drag_atm * (rho_air / rho_ice) * 10.0
        assert jnp.allclose(speed, expected, rtol=1e-10), (
            f"Ice speed mismatch: got {jnp.mean(speed).item():.6e}, expected {expected:.6e}"
        )

        # Case 2: zero wind => ice speed = drag_ocean * ocean_current
        u_ice_calm, v_ice_calm = free_drift_velocity(
            ocean_u, ocean_v,
            jnp.zeros(shape), jnp.zeros(shape),
        )
        speed_calm = jnp.sqrt(u_ice_calm**2 + v_ice_calm**2)
        expected_calm = drag_ocean * 0.1
        assert jnp.allclose(speed_calm, expected_calm, atol=1e-10), (
            f"Zero-wind ice speed should be drag_ocean * ocean_speed: "
            f"got {jnp.mean(speed_calm).item():.6e}, expected {expected_calm:.6e}"
        )
