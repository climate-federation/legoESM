"""Physics smoke tests: verify every scheme runs and produces finite output.

Category 1: Every scheme constructs and runs without errors, producing
finite outputs with correct shapes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState


def make_hydrostatic_setup(n=8, nlev=10):
    """Create minimal cubed-sphere hydrostatic state + grid + sigma."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.physics.held_suarez import held_suarez_init

    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init(grid, sigma)
    # Add tracers for microphysics/convection
    tracers = {
        "q_v": Field(1e-3 * jnp.ones((6, n, n, nlev)), name="q_v"),
        "q_c": Field(1e-5 * jnp.ones((6, n, n, nlev)), name="q_c"),
        "q_r": Field(1e-6 * jnp.ones((6, n, n, nlev)), name="q_r"),
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


# ============================================================================
# 1a  Radiation schemes
# ============================================================================

@pytest.mark.parametrize("scheme", ["gray"])
def test_radiation_smoke(scheme):
    from legoesm.atmosphere.physics.radiation.integration import make_radiation_physics
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig

    state, grid, sigma = make_hydrostatic_setup()
    config = RadiationConfig(scheme=scheme)
    rad_fn = make_radiation_physics(config, model_type="hydrostatic")
    rad_fn.set_time(80.0, 43200.0)
    tend = rad_fn(state, grid, sigma)
    check_tendencies_finite(tend, f"radiation({scheme})")


# ============================================================================
# 1b  Convection schemes
# ============================================================================

@pytest.mark.parametrize("scheme", ["sbm", "dca", "kuo"])
def test_convection_smoke(scheme):
    from legoesm.atmosphere.physics.convection.integration import make_convection_physics
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig

    state, grid, sigma = make_hydrostatic_setup()
    config = ConvectionConfig(scheme=scheme)
    conv_fn = make_convection_physics(config, model_type="hydrostatic", dt=300.0)
    tend, _ = conv_fn(state, grid, sigma)
    check_tendencies_finite(tend, f"convection({scheme})")


# ============================================================================
# 1c  Turbulence schemes
# ============================================================================

@pytest.mark.parametrize("scheme", ["smagorinsky", "louis"])
def test_turbulence_smoke(scheme):
    from legoesm.atmosphere.physics.turbulence.integration import make_turbulence_physics
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

    state, grid, sigma = make_hydrostatic_setup()
    config = TurbulenceConfig(scheme=scheme)
    turb_fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)
    tend, _ = turb_fn(state, grid, sigma)
    check_tendencies_finite(tend, f"turbulence({scheme})")


# ============================================================================
# 1d  Microphysics schemes
# ============================================================================

@pytest.mark.parametrize("scheme", ["kessler", "sundqvist"])
def test_microphysics_smoke(scheme):
    from legoesm.atmosphere.physics.microphysics.integration import make_microphysics_physics
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig

    state, grid, sigma = make_hydrostatic_setup()
    config = MicrophysicsConfig(scheme=scheme)
    micro_fn = make_microphysics_physics(config, model_type="hydrostatic", dt=300.0)
    tend = micro_fn(state, grid, sigma)
    check_tendencies_finite(tend, f"microphysics({scheme})")


# ============================================================================
# 1e  Combined physics
# ============================================================================

def test_combined_physics_smoke():
    from legoesm.atmosphere.physics.combined import make_physics, PhysicsConfig
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig

    state, grid, sigma = make_hydrostatic_setup()
    config = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray"),
        convection=ConvectionConfig(scheme="sbm"),
        turbulence=TurbulenceConfig(scheme="smagorinsky"),
        microphysics=MicrophysicsConfig(scheme="none"),
    )
    physics_fn = make_physics(config, model_type="hydrostatic", dt=300.0)
    physics_fn.set_time(80.0, 43200.0)
    tend, _ = physics_fn(state, grid, sigma)
    check_tendencies_finite(tend, "combined_physics")
