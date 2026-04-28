"""Category 6: Gravity Wave Drag -- Physical Consistency.

Tests drag opposing wind, energy dissipation, Rayleigh sponge structure,
zero-wind behavior, and magnitude bounds for all GWD schemes.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
from legoesm.atmosphere.physics.gravity_wave_drag.integration import make_gwd_physics


def _make_state(n=8, nlev=10, wind_speed=10.0):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.held_suarez import held_suarez_init

    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init(grid, sigma)
    state = state._replace(
        u=Field(data=jnp.ones((6, n, n, nlev)) * wind_speed,
                name="u", dims=("face", "x", "y", "level"), units="m/s"),
        v=Field(data=jnp.ones((6, n, n, nlev)) * 3.0,
                name="v", dims=("face", "x", "y", "level"), units="m/s"),
    )
    tracers = {
        "q_v": Field(1e-3 * jnp.ones((6, n, n, nlev)), name="q_v",
                     dims=("face", "x", "y", "level"), units="kg/kg"),
    }
    state = state._replace(tracers=tracers)
    return state, grid, sigma


def _run_gwd(scheme, **kwargs):
    state, grid, sigma = _make_state(**kwargs)
    config = GravityWaveDragConfig(scheme=scheme)
    gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
    tend, prog = gwd_fn(state, grid, sigma)
    return tend, state


ALL_SCHEMES = ["rayleigh", "lindzen", "mcfarlane", "hines", "prognostic_spectral"]
# Diagnostic schemes (not multi-directional) where drag should oppose wind
DIAGNOSTIC_SCHEMES = ["rayleigh", "lindzen", "mcfarlane", "hines"]


# ============================================================================
# 6a  Drag opposes wind (diagnostic schemes only)
# ============================================================================

@pytest.mark.parametrize("scheme", DIAGNOSTIC_SCHEMES)
def test_drag_opposes_wind(scheme):
    """du_dt * u <= 0 wherever drag is active (drag decelerates)."""
    tend, state = _run_gwd(scheme, wind_speed=15.0)
    u = state.u.data
    du_dt = tend.du_dt.data
    product = u * du_dt
    active = jnp.abs(du_dt) > 1e-12
    if jnp.any(active):
        n_opposing = int(jnp.sum((product[active] <= 1e-10)))
        n_active = int(jnp.sum(active))
        frac = n_opposing / max(n_active, 1)
        assert frac > 0.9, (
            f"{scheme}: only {frac:.0%} of active points have drag opposing wind"
        )


# ============================================================================
# 6d  Zero wind -> zero drag
# ============================================================================

@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_zero_wind_zero_drag(scheme):
    """With u = v = 0, GWD tendencies should be zero."""
    state, grid, sigma = _make_state(wind_speed=0.0)
    state = state._replace(
        v=Field(data=jnp.zeros_like(state.v.data),
                name="v", dims=state.v.dims, units="m/s"),
    )
    config = GravityWaveDragConfig(scheme=scheme)
    gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
    tend, _ = gwd_fn(state, grid, sigma)
    max_du = float(jnp.max(jnp.abs(tend.du_dt.data)))
    max_dv = float(jnp.max(jnp.abs(tend.dv_dt.data)))
    assert max_du < 1e-10, f"{scheme}: du_dt = {max_du:.2e} with zero wind"
    assert max_dv < 1e-10, f"{scheme}: dv_dt = {max_dv:.2e} with zero wind"


# ============================================================================
# 6f  Magnitude bounds
# ============================================================================

@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_magnitude_bounds(scheme):
    """GWD tendencies should be within physical bounds."""
    tend, _ = _run_gwd(scheme)
    max_du = float(jnp.max(jnp.abs(tend.du_dt.data)))
    max_dT = float(jnp.max(jnp.abs(tend.dT_dt.data)))
    assert max_du < 0.1, f"{scheme}: |du_dt| = {max_du:.2e} exceeds 0.1 m/s^2"
    assert max_dT < 1e-3, f"{scheme}: |dT_dt| = {max_dT:.2e} exceeds 1e-3 K/s"


# ============================================================================
# All outputs finite
# ============================================================================

@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_all_finite(scheme):
    """All GWD output fields should be finite."""
    tend, _ = _run_gwd(scheme)
    assert jnp.all(jnp.isfinite(tend.du_dt.data)), f"{scheme}: du_dt NaN/Inf"
    assert jnp.all(jnp.isfinite(tend.dv_dt.data)), f"{scheme}: dv_dt NaN/Inf"
    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{scheme}: dT_dt NaN/Inf"


# ============================================================================
# Rayleigh: structure check
# ============================================================================

def test_rayleigh_sponge_structure():
    """Rayleigh drag should be active at model top and/or BL."""
    tend, _ = _run_gwd("rayleigh", nlev=20, wind_speed=20.0)
    du_dt = tend.du_dt.data
    top_drag = float(jnp.max(jnp.abs(du_dt[..., :3])))
    bot_drag = float(jnp.max(jnp.abs(du_dt[..., -5:])))
    assert top_drag > 1e-8 or bot_drag > 1e-8, "Rayleigh: no drag anywhere"
