"""Tests for FC-Gram hydrostatic primitive equation models."""

import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate

# Skip this module if FC-Gram primitive equations are not available
try:
    from legoesm.atmosphere.dynamics.primitive_eq_fc import (
        FCPrimitiveEquationModel,
        FCPrimitiveEquationConfig,
        fc_hydrostatic_tendencies,
    )
    from legoesm.atmosphere.dynamics.primitive_eq_fc_cgrid import (
        FCCGPrimitiveEquationModel,
        FCCGPrimitiveEquationConfig,
    )
    from legoesm.core.operators_fc import build_fc_config
except ImportError:
    pytest.skip("FC-Gram primitive equation models not available", allow_module_level=True)


def _rest_state_pe(grid, sigma_coord, T0=300.0, p_s0=1e5):
    """Create an isothermal rest state for PE testing."""
    n = grid.n
    nlev = sigma_coord.n_levels
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    return HydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float32),
                name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float32),
                name="v", dims=dims_3d, units="m/s"),
        T=Field(data=jnp.full((6, n, n, nlev), T0, dtype=jnp.float32),
                name="T", dims=dims_3d, units="K"),
        p_s=Field(data=jnp.full((6, n, n), p_s0, dtype=jnp.float32),
                  name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=jnp.zeros((6, n, n), dtype=jnp.float32),
                   name="phis", dims=dims_2d, units="m^2/s^2"),
    )


@pytest.fixture
def grid():
    return create_cubed_sphere(12)


@pytest.fixture
def sigma_coord():
    return create_sigma_coordinate(5)


def test_fc_pe_tendencies_shape(grid, sigma_coord):
    """FC PE tendencies have correct shapes."""
    fc_config = build_fc_config(d=2, C=4, degree=5)
    state = _rest_state_pe(grid, sigma_coord)
    tend = fc_hydrostatic_tendencies(
        state, grid, sigma_coord, fc_config,
    )
    assert tend.du_dt.data.shape == (6, 12, 12, 5)
    assert tend.dT_dt.data.shape == (6, 12, 12, 5)
    assert tend.dp_s_dt.data.shape == (6, 12, 12)


def test_fc_pe_rest_state_stable(grid, sigma_coord):
    """FC PE at rest should have near-zero tendencies."""
    fc_config = build_fc_config(d=2, C=4, degree=5)
    state = _rest_state_pe(grid, sigma_coord)
    tend = fc_hydrostatic_tendencies(
        state, grid, sigma_coord, fc_config,
    )
    assert float(jnp.max(jnp.abs(tend.dp_s_dt.data))) < 10.0


def test_fc_pe_step_runs(grid, sigma_coord):
    """FC PE model step executes without error."""
    config = FCPrimitiveEquationConfig(use_conservation_fixer=False)
    model = FCPrimitiveEquationModel(grid, sigma_coord, config=config)
    state = _rest_state_pe(grid, sigma_coord)
    state_new = model.step(state, 60.0)
    assert jnp.all(jnp.isfinite(state_new.p_s.data))
    assert jnp.all(jnp.isfinite(state_new.T.data))


def test_fc_pe_mass_conservation(grid, sigma_coord):
    """FC PE conserves surface pressure global integral."""
    model = FCPrimitiveEquationModel(grid, sigma_coord)
    state = _rest_state_pe(grid, sigma_coord)
    mass_before = float(jnp.sum(state.p_s.data * grid.area))
    state_new = model.step(state, 60.0)
    mass_after = float(jnp.sum(state_new.p_s.data * grid.area))
    rel_err = abs(mass_after - mass_before) / abs(mass_before)
    assert rel_err < 1e-4, f"Mass conservation error: {rel_err}"


def test_fc_cgrid_pe_tendencies_shape(grid, sigma_coord):
    """FC + div-damping PE tendencies have correct shapes."""
    config = FCCGPrimitiveEquationConfig(div_damp_2=1e5)
    model = FCCGPrimitiveEquationModel(grid, sigma_coord, config=config)
    state = _rest_state_pe(grid, sigma_coord)
    tend = model.tendencies(state)
    assert tend.du_dt.data.shape == (6, 12, 12, 5)


def test_fc_cgrid_pe_step_runs(grid, sigma_coord):
    """FC + div-damping PE step executes without error."""
    config = FCCGPrimitiveEquationConfig(
        div_damp_2=1e5,
        use_conservation_fixer=False,
    )
    model = FCCGPrimitiveEquationModel(grid, sigma_coord, config=config)
    state = _rest_state_pe(grid, sigma_coord)
    state_new = model.step(state, 60.0)
    assert jnp.all(jnp.isfinite(state_new.p_s.data))
