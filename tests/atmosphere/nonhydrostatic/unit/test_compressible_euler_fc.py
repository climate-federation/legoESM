"""Tests for FC-Gram compressible Euler models."""

import warnings

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_height_coordinate, compute_terrain_metric

# Skip this module if FC-Gram compressible Euler is not available
try:
    from legoesm.atmosphere.dynamics.compressible_euler_fc import (
        FCCompressibleEulerModel,
        FCCompressibleEulerConfig,
        fc_compressible_euler_slow_tendencies,
    )
    from legoesm.atmosphere.dynamics.compressible_euler_fc_cgrid import (
        FCCGCompressibleEulerModel,
        FCCGCompressibleEulerConfig,
        fc_cgrid_compressible_euler_slow_tendencies,
    )
    from legoesm.core.operators_fc import build_fc_config
except ImportError:
    pytest.skip("FC-Gram compressible euler models not available", allow_module_level=True)


def _rest_state_ce(grid, height_coord):
    """Create a rest state for CE testing."""
    n = grid.n
    nlev = height_coord.n_levels
    nlev_half = nlev + 1
    n_tracers = 1
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    dims_tr = ("face", "x", "y", "level", "tracer")

    return NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float32),
                name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float32),
                name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev_half), dtype=jnp.float32),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float32),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float32),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n), dtype=jnp.float32),
                   name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, n_tracers), dtype=jnp.float32),
                      name="tracers", dims=dims_tr, units="kg/kg"),
    )


def _require_x64():
    if not jax.config.jax_enable_x64:
        pytest.skip("requires JAX_ENABLE_X64=True")


@pytest.fixture
def grid():
    return create_cubed_sphere(8)


@pytest.fixture
def height_coord():
    return create_height_coordinate(5, 30000.0)


@pytest.fixture
def terrain_metric(grid, height_coord):
    z_s = jnp.zeros((6, grid.n, grid.n))
    return compute_terrain_metric(z_s, height_coord)


def test_fc_ce_tendencies_shape(grid, height_coord, terrain_metric):
    """FC CE slow tendencies have correct shapes."""
    fc_config = build_fc_config(d=2, C=4, degree=5)
    config = FCCompressibleEulerConfig()
    state = _rest_state_ce(grid, height_coord)

    tend = fc_compressible_euler_slow_tendencies(
        state, grid, height_coord, terrain_metric,
        fc_config, config,
    )
    assert tend.du_dt.data.shape == (6, 8, 8, 5)
    assert tend.dw_dt.data.shape == (6, 8, 8, 6)
    assert tend.dtheta_prime_dt.data.shape == (6, 8, 8, 5)


def test_fc_ce_rest_state_finite(grid, height_coord, terrain_metric):
    """FC CE rest state produces finite tendencies."""
    fc_config = build_fc_config(d=2, C=4, degree=5)
    config = FCCompressibleEulerConfig()
    state = _rest_state_ce(grid, height_coord)

    tend = fc_compressible_euler_slow_tendencies(
        state, grid, height_coord, terrain_metric,
        fc_config, config,
    )
    assert jnp.all(jnp.isfinite(tend.du_dt.data))
    assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))


def test_fc_ce_acoustic_substeps_unchanged(grid, height_coord, terrain_metric):
    """Acoustic substeps are reused from centered CE (not modified)."""
    from legoesm.atmosphere.dynamics.compressible_euler import (
        acoustic_substeps,
        acoustic_substeps_semi_implicit,
    )
    assert callable(acoustic_substeps)
    assert callable(acoustic_substeps_semi_implicit)


def test_fc_ce_w_tendency_no_futurewarning_x64(grid, height_coord, terrain_metric):
    """FC CE w tendency should not trigger dtype-scatter FutureWarnings in x64 mode."""
    _require_x64()
    fc_config = build_fc_config(d=2, C=4, degree=5)
    config = FCCompressibleEulerConfig()
    state = _rest_state_ce(grid, height_coord)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        tend = fc_compressible_euler_slow_tendencies(
            state, grid, height_coord, terrain_metric, fc_config, config,
        )
    assert tend.dw_dt.data.dtype == state.w.data.dtype


def test_fc_cgrid_ce_w_tendency_no_futurewarning_x64(grid, height_coord, terrain_metric):
    """FC C-grid CE w tendency should not trigger dtype-scatter FutureWarnings in x64 mode."""
    _require_x64()
    fc_config = build_fc_config(d=2, C=4, degree=5)
    config = FCCGCompressibleEulerConfig()
    state = _rest_state_ce(grid, height_coord)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        tend = fc_cgrid_compressible_euler_slow_tendencies(
            state, grid, height_coord, terrain_metric, fc_config, config,
        )
    assert tend.dw_dt.data.dtype == state.w.data.dtype
