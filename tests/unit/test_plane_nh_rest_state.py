"""Rest-state preservation tests for ``PlaneCompressibleEulerModel``.

Under the minimal-dry PR2b config (no physics, no sponge, no
diffusion, no Coriolis, no mass fixer) a hydrostatically balanced
rest state must stay at rest. Tolerances are derived per step from
the float64 epsilon and the number of acoustic-substep operations so
the test is not flaky across JAX versions.
"""

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
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


jax.config.update("jax_enable_x64", True)


def _setup(use_coriolis=False, fix_mass=False):
    grid = create_plane_grid(
        nx=4, ny=4, nlev=6, dx=1.0e3, dy=1.0e3, dtype=jnp.float64
    )
    height_coord = create_height_coordinate(grid.nlev, H=20.0e3)
    terrain = make_flat_plane_terrain_metric(grid, height_coord)
    config = CompressibleEulerConfig(
        sponge_coeff=0.0,
        hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0,
        hyperdiff_w_coeff=0.0,
        semi_implicit_acoustic=False,
        use_coriolis=use_coriolis,
        fix_mass=fix_mass,
    )
    model = PlaneCompressibleEulerModel(grid, height_coord, terrain, config)
    state = make_rest_state(grid, height_coord, dtype=jnp.float64)
    return model, state


def _max_abs(state):
    return float(jnp.max(jnp.array([
        jnp.max(jnp.abs(state.u.data)),
        jnp.max(jnp.abs(state.v.data)),
        jnp.max(jnp.abs(state.w.data)),
        jnp.max(jnp.abs(state.theta_prime.data)),
        jnp.max(jnp.abs(state.rho_prime.data)),
    ])))


def test_one_step_at_rest_is_exactly_at_rest():
    """No floating-point ops should be triggered on a pure-zero state;
    every tendency contains at least one factor of a velocity or
    perturbation, so every product is exactly zero."""
    model, state = _setup()
    next_state = model.step(state, dt=1.0)
    assert _max_abs(next_state) == 0.0


def test_one_step_at_rest_with_coriolis():
    """f * v = 0 and f * u = 0 when u = v = 0, so Coriolis cannot break
    the rest state."""
    model, state = _setup(use_coriolis=True)
    next_state = model.step(state, dt=1.0)
    assert _max_abs(next_state) == 0.0


def test_one_step_at_rest_with_mass_fixer():
    """The mass fixer is a no-op when the state already matches the
    target mass; it must not perturb the rest state."""
    model, state = _setup(fix_mass=True)
    next_state = model.step(state, dt=1.0)
    assert _max_abs(next_state) == 0.0


def test_many_steps_at_rest_stays_at_rest():
    """Looping 50 steps keeps the state at rest. Tolerance allows for
    eventual round-off accumulation if any introduced (none expected
    for the algebra written; this guards against future regressions)."""
    model, state = _setup()
    for _ in range(50):
        state = model.step(state, dt=1.0)
    # Allow up to 50 ULPs per field — way below any physical signal.
    eps_floor = 50 * jnp.finfo(jnp.float64).eps
    assert _max_abs(state) < eps_floor
