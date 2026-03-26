"""Category 7: Ocean Physics -- Smoke Tests & Physical Consistency.

Tests vertical mixing, bottom drag, and ocean convection for finite
outputs, physical sign conventions, and stability responses.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init import rest_state_ocean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ocean_state(n=8, nlev=10):
    """Create a small ocean state for testing."""
    grid = create_cubed_sphere(n)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    state = rest_state_ocean(
        grid, z_coord, T_surface=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0,
    )
    # Add small currents
    state = state._replace(
        u=Field(data=jnp.ones_like(state.u.data) * 0.1,
                name="u", dims=state.u.dims, units="m/s"),
        v=Field(data=jnp.ones_like(state.v.data) * 0.05,
                name="v", dims=state.v.dims, units="m/s"),
    )
    return state, grid, z_coord


# ============================================================================
# 7a  Vertical mixing smoke tests
# ============================================================================

@pytest.mark.parametrize("scheme", ["constant", "richardson", "kpp"])
def test_vertical_mixing_smoke(scheme):
    """Each vertical mixing scheme produces finite outputs."""
    from legoesm.ocean.physics.vertical_mixing.integration import make_vertical_mixing_physics
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig

    state, grid, z_coord = _make_ocean_state()
    config = VerticalMixingConfig(scheme=scheme)
    phys_fn = make_vertical_mixing_physics(config)
    tend = phys_fn(state, grid, z_coord)

    assert jnp.all(jnp.isfinite(tend.du_dt.data)), f"{scheme}: du_dt has NaN/Inf"
    assert jnp.all(jnp.isfinite(tend.dv_dt.data)), f"{scheme}: dv_dt has NaN/Inf"
    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{scheme}: dT_dt has NaN/Inf"
    assert jnp.all(jnp.isfinite(tend.dS_dt.data)), f"{scheme}: dS_dt has NaN/Inf"


# ============================================================================
# 7g  Bottom drag smoke tests and sign checks
# ============================================================================

@pytest.mark.parametrize("scheme", ["linear", "quadratic"])
def test_bottom_drag_smoke(scheme):
    """Each bottom drag scheme produces finite outputs."""
    from legoesm.ocean.physics.bottom_drag.integration import make_bottom_drag_physics
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig

    state, grid, z_coord = _make_ocean_state()
    config = BottomDragConfig(scheme=scheme)
    phys_fn = make_bottom_drag_physics(config)
    tend = phys_fn(state, grid, z_coord)

    assert jnp.all(jnp.isfinite(tend.du_dt.data)), f"{scheme}: du_dt NaN/Inf"
    assert jnp.all(jnp.isfinite(tend.dv_dt.data)), f"{scheme}: dv_dt NaN/Inf"


@pytest.mark.parametrize("scheme", ["linear", "quadratic"])
def test_bottom_drag_opposes_flow(scheme):
    """Bottom drag opposes flow: du_dt * u <= 0 at bottom level."""
    from legoesm.ocean.physics.bottom_drag.integration import make_bottom_drag_physics
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig

    state, grid, z_coord = _make_ocean_state()
    config = BottomDragConfig(scheme=scheme)
    phys_fn = make_bottom_drag_physics(config)
    tend = phys_fn(state, grid, z_coord)

    u_bot = state.u.data[..., -1]
    du_bot = tend.du_dt.data[..., -1]
    product = u_bot * du_bot
    # Drag should oppose flow where u is nonzero
    active = jnp.abs(u_bot) > 1e-6
    if jnp.any(active):
        frac = float(jnp.mean((product[active] <= 1e-10).astype(jnp.float64)))
        assert frac > 0.9, f"{scheme}: only {frac:.0%} bottom drag opposes flow"


@pytest.mark.parametrize("scheme", ["linear", "quadratic"])
def test_bottom_drag_zero_at_rest(scheme):
    """At rest (u=v=0), bottom drag should be zero."""
    from legoesm.ocean.physics.bottom_drag.integration import make_bottom_drag_physics
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig

    state, grid, z_coord = _make_ocean_state()
    state = state._replace(
        u=Field(data=jnp.zeros_like(state.u.data),
                name="u", dims=state.u.dims, units="m/s"),
        v=Field(data=jnp.zeros_like(state.v.data),
                name="v", dims=state.v.dims, units="m/s"),
    )
    config = BottomDragConfig(scheme=scheme)
    phys_fn = make_bottom_drag_physics(config)
    tend = phys_fn(state, grid, z_coord)

    max_du = float(jnp.max(jnp.abs(tend.du_dt.data)))
    max_dv = float(jnp.max(jnp.abs(tend.dv_dt.data)))
    assert max_du < 1e-15, f"{scheme}: du_dt = {max_du:.2e} at rest"
    assert max_dv < 1e-15, f"{scheme}: dv_dt = {max_dv:.2e} at rest"


# ============================================================================
# 7h  Ocean convection smoke tests
# ============================================================================

@pytest.mark.parametrize("scheme", ["enhanced_diffusion", "plume"])
def test_ocean_convection_smoke(scheme):
    """Each ocean convection scheme produces finite outputs."""
    from legoesm.ocean.physics.convection.integration import (
        make_convection_physics as make_ocean_convection,
    )
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig

    state, grid, z_coord = _make_ocean_state()
    config = OceanConvectionConfig(scheme=scheme)
    phys_fn = make_ocean_convection(config)
    tend = phys_fn(state, grid, z_coord)

    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{scheme}: dT_dt NaN/Inf"
    assert jnp.all(jnp.isfinite(tend.dS_dt.data)), f"{scheme}: dS_dt NaN/Inf"
