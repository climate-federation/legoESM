"""Construction + JIT smoke + config rejection tests for the plane NH dycore."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.dynamics.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    make_flat_plane_terrain_metric,
    make_rest_state,
    validate_plane_config,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import (
    TerrainMetric,
    create_height_coordinate,
)


jax.config.update("jax_enable_x64", True)


def _minimal_config(**overrides):
    """``CompressibleEulerConfig`` with everything PR2b disallows turned off."""
    base = dict(
        sponge_coeff=0.0,
        hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0,
        hyperdiff_w_coeff=0.0,
        semi_implicit_acoustic=False,
        use_coriolis=False,
    )
    base.update(overrides)
    return CompressibleEulerConfig(**base)


def _setup(ny=4, nx=4, nlev=6, H=20.0e3, **cfg_kwargs):
    grid = create_plane_grid(
        nx=nx, ny=ny, nlev=nlev, dx=1.0e3, dy=1.0e3, dtype=jnp.float64
    )
    height_coord = create_height_coordinate(grid.nlev, H=H)
    terrain = make_flat_plane_terrain_metric(grid, height_coord)
    config = _minimal_config(**cfg_kwargs)
    model = PlaneCompressibleEulerModel(grid, height_coord, terrain, config)
    state = make_rest_state(grid, height_coord, dtype=jnp.float64)
    return model, state


# --------------------------------------------------------------------- #
# Construction                                                          #
# --------------------------------------------------------------------- #


def test_construct_with_flat_terrain():
    model, state = _setup()
    assert model.grid.ny == 4
    assert state.u.data.shape == (4, 4, 6)
    assert state.w.data.shape == (4, 4, 7)


def test_construct_rejects_non_plane_grid():
    from legoesm.grids.latlon import create_latlon_grid

    latlon_grid = create_latlon_grid(n_lat=8)
    height_coord = create_height_coordinate(4, H=20.0e3)
    # Fake a TerrainMetric that ``_assert_flat_terrain`` would accept;
    # the type check happens first regardless.
    fake_terrain = TerrainMetric(
        z_s=jnp.zeros((8, 16)),
        jacobian=jnp.ones((8, 16)),
        z_full_3d=jnp.zeros((8, 16, 4)),
        z_half_3d=jnp.zeros((8, 16, 5)),
    )
    with pytest.raises(TypeError, match="expects a PlaneGrid"):
        PlaneCompressibleEulerModel(
            latlon_grid, height_coord, fake_terrain, _minimal_config(),
        )


def test_construct_rejects_non_flat_terrain():
    grid = create_plane_grid(nx=4, ny=4, nlev=4, dx=1.0e3, dy=1.0e3,
                              dtype=jnp.float64)
    height_coord = create_height_coordinate(grid.nlev, H=20.0e3)
    terrain = make_flat_plane_terrain_metric(grid, height_coord)
    bad = TerrainMetric(
        z_s=terrain.z_s,
        jacobian=terrain.jacobian + 0.5,
        z_full_3d=terrain.z_full_3d,
        z_half_3d=terrain.z_half_3d,
    )
    with pytest.raises(ValueError, match="Jacobian == 1"):
        PlaneCompressibleEulerModel(grid, height_coord, bad, _minimal_config())


# --------------------------------------------------------------------- #
# Config rejection                                                      #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"semi_implicit_acoustic": True}, "forward-backward acoustic"),
        ({"hyperdiff_coeff": 1.0e-6}, "hyperdiff_coeff"),
        ({"hyperdiff_rho_coeff": 1.0e-6}, "hyperdiff_rho_coeff"),
        ({"hyperdiff_w_coeff": 1.0e-6}, "hyperdiff_w_coeff"),
    ],
)
def test_validate_plane_config_rejects_unsupported_flags(overrides, match):
    cfg = _minimal_config(**overrides)
    with pytest.raises(NotImplementedError, match=match):
        validate_plane_config(cfg)


def test_validate_plane_config_accepts_sponge_coeff_positive():
    """PR2d wires the Rayleigh sponge; the rejection that PR2b carried
    on ``sponge_coeff > 0`` is now lifted."""
    cfg = _minimal_config(sponge_coeff=0.05)
    validate_plane_config(cfg)  # no raise


def test_step_rejects_nonzero_tracer_axis():
    model, state = _setup()
    nonzero_tracers = state.tracers.replace(
        data=jnp.zeros(state.tracers.data.shape[:-1] + (2,)),
    )
    bad_state = state._replace(tracers=nonzero_tracers)
    with pytest.raises(NotImplementedError, match="n_tracers == 0"):
        model.step(bad_state, dt=1.0)


# --------------------------------------------------------------------- #
# One-step JIT smoke                                                    #
# --------------------------------------------------------------------- #


def test_one_step_runs_under_jit_and_returns_plane_state():
    model, state = _setup()
    next_state = model.step(state, dt=1.0)
    # Type preserved through the jitted step.
    from legoesm.core.state import PlaneNonHydrostaticState
    assert isinstance(next_state, PlaneNonHydrostaticState)
    # Shapes preserved.
    assert next_state.u.data.shape == state.u.data.shape
    assert next_state.w.data.shape == state.w.data.shape
    assert next_state.theta_prime.data.shape == state.theta_prime.data.shape


def test_step_compiles_once_for_repeated_dt():
    model, state = _setup()
    # First call traces; second call hits the cache.
    s1 = model.step(state, dt=1.0)
    s2 = model.step(s1, dt=1.0)
    assert jnp.all(jnp.isfinite(s2.u.data))
    assert jnp.all(jnp.isfinite(s2.w.data))
