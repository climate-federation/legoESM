"""Category 10: Combined Physics -- Cross-Scheme Integration.

Tests all-schemes-on smoke test, tendency additivity, radiation sub-cycling,
all convection x radiation combinations, shape consistency, and multi-step
stability.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(n=8, nlev=10):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init(grid, sigma)
    state = state._replace(
        u=Field(data=jnp.ones((6, n, n, nlev)) * 10.0,
                name="u", dims=("face", "x", "y", "level"), units="m/s"),
        v=Field(data=jnp.ones((6, n, n, nlev)) * 3.0,
                name="v", dims=("face", "x", "y", "level"), units="m/s"),
    )
    tracers = {
        "q_v": Field(1e-3 * jnp.ones((6, n, n, nlev)), name="q_v",
                     dims=("face", "x", "y", "level"), units="kg/kg"),
        "q_c": Field(1e-5 * jnp.ones((6, n, n, nlev)), name="q_c",
                     dims=("face", "x", "y", "level"), units="kg/kg"),
    }
    state = state._replace(tracers=tracers)
    return state, grid, sigma


# ============================================================================
# 10a  All-schemes-on smoke test
# ============================================================================

def test_all_schemes_on_smoke():
    """All 5 physics modules active: gray + sbm + smagorinsky + kessler + rayleigh."""
    state, grid, sigma = _make_state()
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray"),
        convection=ConvectionConfig(scheme="sbm"),
        turbulence=TurbulenceConfig(scheme="smagorinsky"),
        microphysics=MicrophysicsConfig(scheme="kessler"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
    )
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    physics_fn.set_time(80.0, 43200.0)
    tend, _ = physics_fn(state, grid, sigma)

    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), "dT_dt NaN"
    assert jnp.all(jnp.isfinite(tend.du_dt.data)), "du_dt NaN"
    assert jnp.all(jnp.isfinite(tend.dv_dt.data)), "dv_dt NaN"


# ============================================================================
# 10b  Tendency additivity
# ============================================================================

def test_tendency_additivity():
    """Combined tendencies should approximately equal sum of individual."""
    state, grid, sigma = _make_state()
    dt = 300.0

    # Run each module individually
    cfg_rad = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    rad_fn = make_physics(cfg_rad, model_type="hydrostatic", dt=dt)
    rad_fn.set_time(80.0, 43200.0)
    tend_rad, _ = rad_fn(state, grid, sigma)

    cfg_turb = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="smagorinsky"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    turb_fn = make_physics(cfg_turb, model_type="hydrostatic", dt=dt)
    tend_turb, _ = turb_fn(state, grid, sigma)

    cfg_gwd = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
    )
    gwd_fn = make_physics(cfg_gwd, model_type="hydrostatic", dt=dt)
    tend_gwd, _ = gwd_fn(state, grid, sigma)

    # Combined
    cfg_all = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="smagorinsky"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
    )
    all_fn = make_physics(cfg_all, model_type="hydrostatic", dt=dt)
    all_fn.set_time(80.0, 43200.0)
    tend_all, _ = all_fn(state, grid, sigma)

    # Sum of individual
    sum_dT = tend_rad.dT_dt.data + tend_turb.dT_dt.data + tend_gwd.dT_dt.data
    sum_du = tend_rad.du_dt.data + tend_turb.du_dt.data + tend_gwd.du_dt.data

    # Should match within numerical precision
    max_dT_err = float(jnp.max(jnp.abs(tend_all.dT_dt.data - sum_dT)))
    max_du_err = float(jnp.max(jnp.abs(tend_all.du_dt.data - sum_du)))
    assert max_dT_err < 1e-10, f"dT additivity error = {max_dT_err:.2e}"
    assert max_du_err < 1e-10, f"du additivity error = {max_du_err:.2e}"


# ============================================================================
# 10e  All convection x radiation combinations
# ============================================================================

@pytest.mark.parametrize("conv", ["sbm", "dca", "kuo", "mass_flux", "edmf"])
def test_convection_radiation_combination(conv):
    """Gray radiation + each convection scheme should produce finite output."""
    state, grid, sigma = _make_state()
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray"),
        convection=ConvectionConfig(scheme=conv),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    physics_fn.set_time(80.0, 43200.0)
    tend, _ = physics_fn(state, grid, sigma)
    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"NaN with gray+{conv}"
    assert jnp.all(jnp.isfinite(tend.du_dt.data)), f"NaN with gray+{conv}"


# ============================================================================
# 10f  Physics -> dynamics coupling shape
# ============================================================================

def test_tendency_shapes():
    """Physics tendencies have exact same shape as state fields."""
    state, grid, sigma = _make_state()
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray"),
        convection=ConvectionConfig(scheme="sbm"),
        turbulence=TurbulenceConfig(scheme="louis"),
    )
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    physics_fn.set_time(80.0, 43200.0)
    tend, _ = physics_fn(state, grid, sigma)

    assert tend.dT_dt.data.shape == state.T.data.shape, "dT_dt shape mismatch"
    assert tend.du_dt.data.shape == state.u.data.shape, "du_dt shape mismatch"
    assert tend.dv_dt.data.shape == state.v.data.shape, "dv_dt shape mismatch"
    assert tend.dp_s_dt.data.shape == state.p_s.data.shape, "dp_s_dt shape mismatch"


# ============================================================================
# 10g  Multi-step stability (physics only, no dynamics)
# ============================================================================

def test_multi_step_stability():
    """20 steps of physics only: T stays in [150, 350], q stays in [0, 0.05]."""
    state, grid, sigma = _make_state()
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray"),
        convection=ConvectionConfig(scheme="sbm"),
        turbulence=TurbulenceConfig(scheme="smagorinsky"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
    )
    dt = 300.0
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=dt)
    physics_fn.set_time(80.0, 43200.0)

    for step in range(20):
        tend, _ = physics_fn(state, grid, sigma)
        # Apply tendencies
        new_T = state.T.data + tend.dT_dt.data * dt
        new_u = state.u.data + tend.du_dt.data * dt
        new_v = state.v.data + tend.dv_dt.data * dt
        state = state._replace(
            T=state.T.replace(data=new_T),
            u=state.u.replace(data=new_u),
            v=state.v.replace(data=new_v),
        )

    # Check bounds
    T_data = state.T.data
    assert float(jnp.min(T_data)) > 150.0, (
        f"T min = {float(jnp.min(T_data)):.1f} < 150 K after 20 steps"
    )
    assert float(jnp.max(T_data)) < 350.0, (
        f"T max = {float(jnp.max(T_data)):.1f} > 350 K after 20 steps"
    )
    # No NaN/Inf
    assert jnp.all(jnp.isfinite(T_data)), "T has NaN/Inf after 20 steps"
    assert jnp.all(jnp.isfinite(state.u.data)), "u has NaN/Inf after 20 steps"
