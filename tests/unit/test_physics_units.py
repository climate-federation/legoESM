"""Category 9: Unit Consistency & Dimensional Analysis.

Verifies tendency units, radiation flux-heating consistency, latent heat
coupling, pressure units, surface flux magnitude bounds, and physical
constants cross-checks.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.core.field import Field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state_and_run_physics():
    """Run combined physics and return tendencies, state, grid, sigma."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig

    grid = create_cubed_sphere(8)
    sigma = create_sigma_coordinate(10)
    state = held_suarez_init(grid, sigma)
    state = state._replace(
        u=Field(data=jnp.ones((6, 8, 8, 10)) * 10.0,
                name="u", dims=("face", "x", "y", "level"), units="m/s"),
        v=Field(data=jnp.ones((6, 8, 8, 10)) * 3.0,
                name="v", dims=("face", "x", "y", "level"), units="m/s"),
    )
    tracers = {
        "q_v": Field(1e-3 * jnp.ones((6, 8, 8, 10)), name="q_v",
                     dims=("face", "x", "y", "level"), units="kg/kg"),
    }
    state = state._replace(tracers=tracers)

    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray"),
        convection=ConvectionConfig(scheme="sbm"),
        turbulence=TurbulenceConfig(scheme="smagorinsky"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
    )
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    physics_fn.set_time(80.0, 43200.0)
    tend, _ = physics_fn(state, grid, sigma)
    return tend, state, grid, sigma


# ============================================================================
# 9a  Tendency units: applying tendencies for dt produces reasonable changes
# ============================================================================

def test_temperature_tendency_units():
    """dT_dt [K/s] * dt [s] -> DeltaT [K]. |DeltaT| < 10 K for dt=300s."""
    tend, _, _, _ = _make_state_and_run_physics()
    dt = 300.0
    delta_T = tend.dT_dt.data * dt
    max_dT = float(jnp.max(jnp.abs(delta_T)))
    assert max_dT < 10.0, (
        f"|DeltaT| = {max_dT:.2f} K for dt=300s (likely unit error)"
    )


def test_wind_tendency_units():
    """du_dt [m/s^2] * dt [s] -> Deltau [m/s]. |Deltau| < 30 m/s for dt=300s."""
    tend, _, _, _ = _make_state_and_run_physics()
    dt = 300.0
    delta_u = tend.du_dt.data * dt
    max_du = float(jnp.max(jnp.abs(delta_u)))
    assert max_du < 30.0, (
        f"|Deltau| = {max_du:.2f} m/s for dt=300s (likely unit error)"
    )


# ============================================================================
# 9b  Radiation flux -> heating rate consistency
# ============================================================================

def test_radiation_flux_heating_rate_consistency():
    """Heating rate from flux divergence matches the scheme's reported value."""
    from legoesm.atmosphere.physics.radiation.gray import gray_radiation
    from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
    from legoesm.atmosphere.physics.radiation.solar import perpetual_equinox_insolation
    from legoesm.thermo import saturation_mixing_ratio

    config = GrayRadiationConfig()
    nlev = 20
    p_s = 1.0e5
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_half = (sigma_half * p_s)[None, :]
    p_full = (sigma_full * p_s)[None, :]
    T = (300.0 * jnp.clip(sigma_full, 0.01, None) ** 0.19)[None, :]
    T = jnp.maximum(T, 200.0)
    T_sfc = jnp.array([300.0])
    lat = jnp.array([0.3])
    q_v = 0.8 * saturation_mixing_ratio(T, p_full)
    insol = perpetual_equinox_insolation(lat, config.S_0)

    out = gray_radiation(T, p_full, p_half, T_sfc, lat, q_v, insol, config)

    F_net_up = (out.lw_flux_up + out.sw_flux_up) - (out.lw_flux_down + out.sw_flux_down)
    dF = F_net_up[0, 1:] - F_net_up[0, :-1]
    dp = p_half[0, 1:] - p_half[0, :-1]
    dp = jnp.clip(dp, 1.0, None)
    hr_recomputed = (constants.g / constants.c_pd) * dF / dp

    hr_reported = out.heating_rate[0]
    rel_err = jnp.abs(hr_recomputed - hr_reported) / jnp.maximum(jnp.abs(hr_reported), 1e-10)
    active = jnp.abs(hr_reported) > 1e-10
    if jnp.any(active):
        max_err = float(jnp.max(rel_err[active]))
        assert max_err < 0.01, (
            f"Flux->heating rate rel_err = {max_err:.4f} (expected < 1%)"
        )


# ============================================================================
# 9c  Latent heat consistency: column-integrated MSE for SBM
# ============================================================================

def test_sbm_column_mse_conservation():
    """SBM: column-integrated MSE tendency should be small (< 10 W/m^2)."""
    from legoesm.atmosphere.physics.convection.sbm import sbm_convection
    from legoesm.atmosphere.physics.convection.config import SBMConfig
    from legoesm.thermo import saturation_mixing_ratio

    nlev = 20; ncol = 4; p_s = 1.0e5
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (ncol, nlev + 1))
    p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (ncol, nlev))
    T = 300.0 * jnp.clip(sigma_full, 0.01, None) ** 0.19
    T = jnp.maximum(T, 200.0)
    T = jnp.broadcast_to(T[None, :], (ncol, nlev))
    q_v = 0.9 * saturation_mixing_ratio(T, p_full)

    out = sbm_convection(T, q_v, p_full, p_half, 300.0, config=SBMConfig())

    dp = p_half[:, 1:] - p_half[:, :-1]
    mse_tend = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
    col_mse = jnp.sum(mse_tend * dp / constants.g, axis=1)
    max_imbalance = float(jnp.max(jnp.abs(col_mse)))
    assert max_imbalance < 10.0, (
        f"SBM column MSE imbalance = {max_imbalance:.2f} W/m^2 (expected < 10)"
    )


# ============================================================================
# 9d  Pressure units
# ============================================================================

def test_pressure_in_pascals():
    """Pressure fields should be in Pa (not hPa)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

    grid = create_cubed_sphere(8)
    sigma = create_sigma_coordinate(10)
    state = held_suarez_init(grid, sigma)

    p_s = state.p_s.data
    assert float(jnp.min(p_s)) > 5000.0, "p_s looks like hPa not Pa"
    assert float(jnp.max(p_s)) < 2e5, "p_s unreasonably high"


# ============================================================================
# 9f  Constants cross-check
# ============================================================================

class TestConstantsCrossCheck:

    def test_gravity(self):
        assert abs(constants.g - 9.81) / 9.81 < 0.01

    def test_specific_heat(self):
        assert abs(constants.c_pd - 1004.0) / 1004.0 < 0.01

    def test_latent_heat_vaporization(self):
        assert abs(constants.L_v - 2.501e6) / 2.501e6 < 0.01

    def test_gas_constant_vapor(self):
        assert abs(constants.R_v - 461.0) / 461.0 < 0.01

    def test_stefan_boltzmann(self):
        assert abs(constants.sigma_sb - 5.67e-8) / 5.67e-8 < 0.01

    def test_earth_radius(self):
        assert abs(constants.R_earth - 6.371e6) / 6.371e6 < 0.01

    def test_poisson_constant(self):
        assert abs(constants.kappa - 0.286) < 0.005

    def test_epsilon(self):
        assert abs(constants.epsilon - 0.622) < 0.005

    def test_freezing_point(self):
        assert abs(constants.T_freeze - 273.15) < 0.01
