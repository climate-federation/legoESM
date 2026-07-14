"""Pytree round-trip + tree_map sanity for ``PlaneNonHydrostaticState``
and ``PlaneNonHydrostaticTendencies``.

These cover the cross-cutting change of adding new state classes to
``src/legoesm/core/state.py`` so the existing tree-based time
integration (SSP-RK3, physics coupling) accepts them without
modification.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    make_rest_state,
)
from legoesm.core.state import (
    PlaneNonHydrostaticState,
    PlaneNonHydrostaticTendencies,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


jax.config.update("jax_enable_x64", True)


def _grid_and_height():
    grid = create_plane_grid(
        nx=4, ny=4, nlev=4, dx=1.0e3, dy=1.0e3, dtype=jnp.float64
    )
    height_coord = create_height_coordinate(grid.nlev, H=20.0e3)
    return grid, height_coord


def test_state_tree_flatten_round_trip():
    grid, height_coord = _grid_and_height()
    state = make_rest_state(grid, height_coord, dtype=jnp.float64)

    leaves, treedef = jax.tree_util.tree_flatten(state)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, PlaneNonHydrostaticState)
    for orig_leaf, new_leaf in zip(leaves, jax.tree_util.tree_leaves(rebuilt)):
        assert jnp.array_equal(orig_leaf, new_leaf)


def test_state_tree_map_identity_preserves_shapes():
    grid, height_coord = _grid_and_height()
    state = make_rest_state(grid, height_coord, dtype=jnp.float64)

    doubled = jax.tree_util.tree_map(lambda x: 2.0 * x, state)
    assert isinstance(doubled, PlaneNonHydrostaticState)
    for orig_leaf, new_leaf in zip(
        jax.tree_util.tree_leaves(state),
        jax.tree_util.tree_leaves(doubled),
    ):
        assert orig_leaf.shape == new_leaf.shape


def test_tendencies_match_state_pytree_structure():
    """SSP-RK3 averaging assumes state and tendency pytrees share
    structure so :func:`jax.tree_util.tree_map` aligns leaves."""
    grid, height_coord = _grid_and_height()
    state = make_rest_state(grid, height_coord, dtype=jnp.float64)

    zero_tend = PlaneNonHydrostaticTendencies(
        du_dt=state.u.replace(data=jnp.zeros_like(state.u.data)),
        dv_dt=state.v.replace(data=jnp.zeros_like(state.v.data)),
        dw_dt=state.w.replace(data=jnp.zeros_like(state.w.data)),
        dtheta_prime_dt=state.theta_prime.replace(
            data=jnp.zeros_like(state.theta_prime.data)),
        drho_prime_dt=state.rho_prime.replace(
            data=jnp.zeros_like(state.rho_prime.data)),
        dphis_dt=state.phis.replace(data=jnp.zeros_like(state.phis.data)),
        dtracers_dt=state.tracers.replace(
            data=jnp.zeros_like(state.tracers.data)),
    )

    state_leaves = jax.tree_util.tree_leaves(state)
    tend_leaves = jax.tree_util.tree_leaves(zero_tend)
    assert len(state_leaves) == len(tend_leaves)
    for s_leaf, t_leaf in zip(state_leaves, tend_leaves):
        assert s_leaf.shape == t_leaf.shape


def test_state_supports_n_tracers_zero():
    grid, height_coord = _grid_and_height()
    state = make_rest_state(grid, height_coord, dtype=jnp.float64)
    assert state.tracers.data.shape[-1] == 0


def test_state_under_jit_smoke():
    grid, height_coord = _grid_and_height()
    state = make_rest_state(grid, height_coord, dtype=jnp.float64)

    @jax.jit
    def total_rho(s: PlaneNonHydrostaticState) -> jax.Array:
        return s.rho_prime.data.sum()

    out = total_rho(state)
    assert float(out) == pytest.approx(0.0)
