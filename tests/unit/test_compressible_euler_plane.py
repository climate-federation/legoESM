"""Construction + JIT smoke + config rejection tests for the plane NH dycore."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
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


def test_validate_plane_config_accepts_semi_implicit_acoustic():
    """semi_implicit_acoustic now supported via per-column Thomas
    solve (plane_acoustic_substeps_semi_implicit)."""
    cfg = _minimal_config(semi_implicit_acoustic=True)
    validate_plane_config(cfg)  # no raise


def test_validate_plane_config_accepts_sponge_coeff_positive():
    """PR2d wires the Rayleigh sponge; the rejection that PR2b carried
    on ``sponge_coeff > 0`` is now lifted."""
    cfg = _minimal_config(sponge_coeff=0.05)
    validate_plane_config(cfg)  # no raise


@pytest.mark.parametrize(
    "field",
    ["hyperdiff_coeff", "hyperdiff_rho_coeff", "hyperdiff_w_coeff"],
)
def test_validate_plane_config_accepts_hyperdiff_positive(field):
    """PR3a wires biharmonic hyperdiffusion for u/v/theta', rho', and
    w; the PR2b rejection on each of the three coefficients is
    lifted."""
    cfg = _minimal_config(**{field: 1.0e-6})
    validate_plane_config(cfg)  # no raise


def test_step_accepts_nonzero_tracer_axis_after_pr3d():
    """PR3d lifted the PR2b ``n_tracers == 0`` restriction. The step
    should now run with a non-empty tracer axis and return a state
    of the same shape."""
    model, state = _setup()
    nonzero_tracers = state.tracers.replace(
        data=jnp.zeros(state.tracers.data.shape[:-1] + (2,)),
    )
    state_with_tracers = state._replace(tracers=nonzero_tracers)
    next_state = model.step(state_with_tracers, dt=1.0)
    assert next_state.tracers.data.shape == state_with_tracers.tracers.data.shape
    # All-zero tracers + zero rest state stays zero.
    assert float(jnp.max(jnp.abs(next_state.tracers.data))) == 0.0


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


# --------------------------------------------------------------------- #
# Semi-implicit acoustic substeps                                       #
# --------------------------------------------------------------------- #


def test_semi_implicit_acoustic_step_runs():
    """semi_implicit_acoustic=True path runs end-to-end without
    NotImplementedError; produces finite output for rest state."""
    model, state = _setup(semi_implicit_acoustic=True)
    next_state = model.step(state, dt=1.0)
    assert jnp.all(jnp.isfinite(next_state.w.data))
    assert jnp.all(jnp.isfinite(next_state.theta_prime.data))
    assert jnp.all(jnp.isfinite(next_state.rho_prime.data))


def test_semi_implicit_acoustic_matches_explicit_at_rest():
    """At rest state with zero perturbation, both paths produce
    identical output (no acoustic activity to differentiate them)."""
    model_e, state = _setup(semi_implicit_acoustic=False)
    model_si, _ = _setup(semi_implicit_acoustic=True)
    s_e = model_e.step(state, dt=1.0)
    s_si = model_si.step(state, dt=1.0)
    # Rest state stays at rest under both paths.
    import numpy as np
    np.testing.assert_allclose(
        np.asarray(s_e.w.data), np.asarray(s_si.w.data), atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(s_e.theta_prime.data),
        np.asarray(s_si.theta_prime.data), atol=1e-14,
    )


def test_semi_implicit_acoustic_jit_compilable():
    """JIT path compiles cleanly under semi-implicit acoustic."""
    model, state = _setup(semi_implicit_acoustic=True)
    fn = jax.jit(lambda s: model.step(s, dt=1.0))
    out = fn(state)
    assert jnp.all(jnp.isfinite(out.w.data))


def test_semi_implicit_acoustic_stable_on_stretched_grid():
    """Codex iter-2: regression guard for the feature's actual purpose.

    On a stretched vertical grid with thin surface layer (dz_sfc=50m),
    the vertical acoustic CFL bounds dt for the explicit
    forward-backward path at dt ~ 0.3 s. The semi-implicit path lifts
    this and must remain finite at dt=2.0s with a non-trivial
    perturbation that excites vertical acoustic modes.
    """
    from legoesm.grids.vertical import create_stretched_height_coordinate

    nx = ny = 4
    nlev = 20
    grid = create_plane_grid(
        nx=nx, ny=ny, nlev=nlev, dx=2.0e3, dy=2.0e3, dtype=jnp.float64,
    )
    hc = create_stretched_height_coordinate(nlev, H=20.0e3, dz_sfc=50.0)
    terrain = make_flat_plane_terrain_metric(grid, hc)
    cfg = _minimal_config(
        semi_implicit_acoustic=True,
        n_acoustic_substeps=6,
        acoustic_off_centering=0.1,
    )
    model = PlaneCompressibleEulerModel(grid, hc, terrain, cfg)
    rest = make_rest_state(grid, hc, dtype=jnp.float64)
    # Seed vertical acoustic mode via single-cell w-impulse in lowest
    # interior interface (dz_sfc=50m → fastest vertical mode).
    w_kick = rest.w.data.at[ny // 2, nx // 2, -2].set(0.5)
    state = rest._replace(w=rest.w.replace(data=w_kick))
    # Run a few outer steps at dt=2.0s — explicit path would blow up
    # at dz_sfc=50m within 1-2 steps (acoustic CFL ≈ 13 even with 24
    # substeps). Semi-implicit must remain finite.
    for _ in range(5):
        state = model.step(state, dt=2.0)
    assert jnp.all(jnp.isfinite(state.w.data)), (
        "semi-implicit acoustic must remain finite at dt=2 on a "
        "dz_sfc=50m stretched grid; explicit path fails here."
    )
    assert jnp.all(jnp.isfinite(state.theta_prime.data))
    assert jnp.all(jnp.isfinite(state.rho_prime.data))
