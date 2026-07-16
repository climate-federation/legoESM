"""Category 1: Atmosphere Physics -- Smoke Tests (Every Scheme Runs).

Verifies that every atmosphere physics scheme can be constructed and called
without errors, and produces finite outputs with correct shapes.

Each scheme is tested individually through its integration bridge (the same
pathway used by ModelDriver). Realistic initial conditions come from
held_suarez_init.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_hydrostatic_setup(n=8, nlev=10):
    """Create minimal cubed-sphere hydrostatic state + grid + sigma."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init(grid, sigma)
    # Add realistic wind so turbulence / GWD have something to work on
    state = state._replace(
        u=Field(data=jnp.ones((6, n, n, nlev)) * 10.0,
                name="u", dims=("face", "x", "y", "level"), units="m/s"),
        v=Field(data=jnp.ones((6, n, n, nlev)) * 3.0,
                name="v", dims=("face", "x", "y", "level"), units="m/s"),
    )
    # Add tracers for microphysics / convection
    tracers = {
        "q_v": Field(1e-3 * jnp.ones((6, n, n, nlev)), name="q_v",
                     dims=("face", "x", "y", "level"), units="kg/kg"),
        "q_c": Field(1e-5 * jnp.ones((6, n, n, nlev)), name="q_c",
                     dims=("face", "x", "y", "level"), units="kg/kg"),
        "q_r": Field(1e-6 * jnp.ones((6, n, n, nlev)), name="q_r",
                     dims=("face", "x", "y", "level"), units="kg/kg"),
    }
    state = state._replace(tracers=tracers)
    return state, grid, sigma


def check_tendencies_finite(tend, name):
    """Assert all tendency fields are finite."""
    for fname in tend._fields:
        val = getattr(tend, fname)
        if val is None:
            continue
        if isinstance(val, dict):
            for k, v in val.items():
                data = v.data if hasattr(v, "data") else v
                assert jnp.all(jnp.isfinite(data)), f"{name}.{fname}[{k}] has NaN/Inf"
        elif hasattr(val, "data"):
            assert jnp.all(jnp.isfinite(val.data)), f"{name}.{fname} has NaN/Inf"
        elif isinstance(val, jnp.ndarray):
            assert jnp.all(jnp.isfinite(val)), f"{name}.{fname} has NaN/Inf"


def check_shape(arr, expected, name):
    """Assert array has expected shape."""
    data = arr.data if hasattr(arr, "data") else arr
    assert data.shape == expected, f"{name}: shape {data.shape} != {expected}"


# ============================================================================
# 1a  Radiation schemes
# ============================================================================

@pytest.mark.parametrize("scheme", ["gray"])
def test_radiation_smoke(scheme):
    """Each radiation scheme produces finite tendencies with correct shape."""
    from legoesm.atmosphere.physics.radiation.integration import make_radiation_physics
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig

    state, grid, sigma = make_hydrostatic_setup()
    config = RadiationConfig(scheme=scheme)
    rad_fn = make_radiation_physics(config, model_type="hydrostatic")
    rad_fn.set_time(80.0, 43200.0)
    tend = rad_fn(state, grid, sigma)
    check_tendencies_finite(tend, f"radiation({scheme})")
    # Shape check: (6, n, n, nlev)
    check_shape(tend.dT_dt, (6, 8, 8, 10), f"radiation({scheme}).dT_dt")
    # Non-zero heating
    assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0, f"{scheme}: zero heating"


# ============================================================================
# 1b  Convection schemes
# ============================================================================

@pytest.mark.parametrize("scheme", ["sbm", "dca", "kuo", "mass_flux", "edmf"])
def test_convection_smoke(scheme):
    """Each convection scheme produces finite tendencies."""
    from legoesm.atmosphere.physics.convection.integration import make_convection_physics
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig

    state, grid, sigma = make_hydrostatic_setup()
    config = ConvectionConfig(scheme=scheme)
    conv_fn = make_convection_physics(config, model_type="hydrostatic", dt=300.0)
    tend, prog = conv_fn(state, grid, sigma)
    check_tendencies_finite(tend, f"convection({scheme})")
    check_shape(tend.dT_dt, (6, 8, 8, 10), f"convection({scheme}).dT_dt")


# ============================================================================
# 1c  Microphysics schemes
# ============================================================================

@pytest.mark.parametrize("scheme", ["kessler", "sundqvist", "seifert_beheng",
                                     "morrison", "thompson"])
def test_microphysics_smoke(scheme):
    """Each microphysics scheme produces finite tendencies."""
    from legoesm.atmosphere.physics.microphysics.integration import make_microphysics_physics
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig

    state, grid, sigma = make_hydrostatic_setup()
    # Put some cloud water in lowest 5 levels to trigger microphysics
    q_c_data = jnp.zeros((6, 8, 8, 10))
    q_c_data = q_c_data.at[..., -5:].set(1e-4)
    state = state._replace(
        tracers={
            **state.tracers,
            "q_c": Field(q_c_data, name="q_c",
                         dims=("face", "x", "y", "level"), units="kg/kg"),
        }
    )
    config = MicrophysicsConfig(scheme=scheme)
    micro_fn = make_microphysics_physics(config, model_type="hydrostatic", dt=300.0)
    tend = micro_fn(state, grid, sigma)
    check_tendencies_finite(tend, f"microphysics({scheme})")
    check_shape(tend.dT_dt, (6, 8, 8, 10), f"microphysics({scheme}).dT_dt")


# ============================================================================
# 1d  Turbulence schemes
# ============================================================================

@pytest.mark.parametrize("scheme", ["smagorinsky", "louis", "tke", "clubb_lite",
                                     "holtslag_boville", "ysu", "edmf"])
def test_turbulence_smoke(scheme):
    """Each turbulence scheme produces finite tendencies."""
    from legoesm.atmosphere.physics.turbulence.integration import make_turbulence_physics
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

    state, grid, sigma = make_hydrostatic_setup()
    config = TurbulenceConfig(scheme=scheme)
    turb_fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)
    tend, prog = turb_fn(state, grid, sigma)
    check_tendencies_finite(tend, f"turbulence({scheme})")
    check_shape(tend.dT_dt, (6, 8, 8, 10), f"turbulence({scheme}).dT_dt")
    # Non-zero wind tendency (surface friction)
    assert float(jnp.max(jnp.abs(tend.du_dt.data))) > 0.0, f"{scheme}: zero du_dt"


# ============================================================================
# 1e  Gravity Wave Drag schemes
# ============================================================================

@pytest.mark.parametrize("scheme", ["rayleigh", "lindzen", "mcfarlane",
                                     "hines", "prognostic_spectral"])
def test_gwd_smoke(scheme):
    """Each GWD scheme produces finite tendencies."""
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import make_gwd_physics
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig

    state, grid, sigma = make_hydrostatic_setup()
    config = GravityWaveDragConfig(scheme=scheme)
    gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
    tend, prog = gwd_fn(state, grid, sigma)
    check_tendencies_finite(tend, f"gwd({scheme})")
    check_shape(tend.dT_dt, (6, 8, 8, 10), f"gwd({scheme}).dT_dt")


# ============================================================================
# 1f  Combined physics via make_physics
# ============================================================================

def test_combined_all_defaults():
    """Default PhysicsConfig produces finite tendencies."""
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics

    state, grid, sigma = make_hydrostatic_setup()
    cfg = PhysicsConfig()
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    physics_fn.set_time(80.0, 43200.0)
    tend, phys_state = physics_fn(state, grid, sigma)
    check_tendencies_finite(tend, "combined_defaults")


def test_combined_full_stack():
    """All five physics modules active simultaneously produce finite output."""
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig

    state, grid, sigma = make_hydrostatic_setup()
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray"),
        convection=ConvectionConfig(scheme="sbm"),
        turbulence=TurbulenceConfig(scheme="louis"),
        microphysics=MicrophysicsConfig(scheme="kessler"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
    )
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    physics_fn.set_time(80.0, 43200.0)
    tend, phys_state = physics_fn(state, grid, sigma)
    check_tendencies_finite(tend, "combined_full_stack")
    assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0
    assert float(jnp.max(jnp.abs(tend.du_dt.data))) > 0.0
