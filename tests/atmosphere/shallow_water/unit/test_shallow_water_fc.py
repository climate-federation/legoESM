"""Tests for FC-Gram shallow water models."""

import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fc import (
    FCShallowWaterModel,
    FCShallowWaterConfig,
)
from legoesm.atmosphere.dynamics.shallow_water_fc_cgrid import (
    FCCGShallowWaterModel,
    FCCGShallowWaterConfig,
)


def _rest_state(grid, h0=1000.0):
    """Create a rest state for testing."""
    n = grid.n
    dims = ("face", "x", "y")
    return ShallowWaterState(
        h=Field(data=jnp.full((6, n, n), h0, dtype=jnp.float32),
                name="h", dims=dims, units="m"),
        u=Field(data=jnp.zeros((6, n, n), dtype=jnp.float32),
                name="u", dims=dims, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n), dtype=jnp.float32),
                name="v", dims=dims, units="m/s"),
        h_s=Field(data=jnp.zeros((6, n, n), dtype=jnp.float32),
                  name="h_s", dims=dims, units="m"),
    )


@pytest.fixture
def grid():
    return create_cubed_sphere(12)


def test_fc_sw_tendencies_shape(grid):
    """FC SW tendencies have correct shapes."""
    model = FCShallowWaterModel(grid)
    state = _rest_state(grid)
    tend = model.tendencies(state)
    assert tend.dh_dt.data.shape == (6, 12, 12)
    assert tend.du_dt.data.shape == (6, 12, 12)
    assert tend.dv_dt.data.shape == (6, 12, 12)


def test_fc_sw_rest_state_stable(grid):
    """FC SW at rest should have near-zero tendencies."""
    model = FCShallowWaterModel(grid)
    state = _rest_state(grid)
    tend = model.tendencies(state)
    assert float(jnp.max(jnp.abs(tend.dh_dt.data))) < 1.0
    assert float(jnp.max(jnp.abs(tend.du_dt.data))) < 1.0
    assert float(jnp.max(jnp.abs(tend.dv_dt.data))) < 1.0


def test_fc_sw_step_runs(grid):
    """FC SW step executes without error."""
    config = FCShallowWaterConfig(use_conservation_fixer=False)
    model = FCShallowWaterModel(grid, config=config)
    state = _rest_state(grid)
    state_new = model.step(state, 60.0)
    assert jnp.all(jnp.isfinite(state_new.h.data))


def test_fc_cgrid_sw_tendencies_shape(grid):
    """FC + div-damping SW tendencies have correct shapes."""
    config = FCCGShallowWaterConfig(div_damp_2=1e5)
    model = FCCGShallowWaterModel(grid, config=config)
    state = _rest_state(grid)
    tend = model.tendencies(state)
    assert tend.dh_dt.data.shape == (6, 12, 12)


def test_fc_cgrid_sw_step_runs(grid):
    """FC + div-damping SW step executes without error."""
    config = FCCGShallowWaterConfig(
        div_damp_2=1e5,
        use_conservation_fixer=False,
    )
    model = FCCGShallowWaterModel(grid, config=config)
    state = _rest_state(grid)
    state_new = model.step(state, 60.0)
    assert jnp.all(jnp.isfinite(state_new.h.data))


def test_fc_sw_mass_conserved(grid):
    """FC SW conserves mass (with conservation fixer)."""
    model = FCShallowWaterModel(grid)
    state = _rest_state(grid)
    mass_before = float(jnp.sum(state.h.data * grid.area))

    state_new = model.step(state, 60.0)
    mass_after = float(jnp.sum(state_new.h.data * grid.area))

    rel_err = abs(mass_after - mass_before) / abs(mass_before)
    # FC spectral divergence is not perfectly telescoping; conservation
    # relies on zero_mean_tendency correction, slightly looser than FV
    assert rel_err < 1e-4, f"Mass conservation error: {rel_err}"
