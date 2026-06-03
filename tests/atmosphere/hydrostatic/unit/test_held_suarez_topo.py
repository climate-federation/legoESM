"""Sanity tests for ``legoesm.atmosphere.idealized.held_suarez_topo``."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.idealized.held_suarez_topo import (
    held_suarez_topo_forcing,
    held_suarez_topo_init,
    held_suarez_topo_init_latlon,
    held_suarez_topo_init_mpas,
    held_suarez_topo_init_spectral,
)
from legoesm.atmosphere.held_suarez import held_suarez_forcing
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import standard_hybrid_levels
from legoesm.grids.voronoi import create_voronoi_mesh


@pytest.fixture(scope="module")
def sigma():
    return standard_hybrid_levels(8)


def test_topo_forcing_is_alias():
    """The topography variant reuses the canonical HS forcing exactly."""
    assert held_suarez_topo_forcing is held_suarez_forcing


def test_topo_init_cubed_sphere(sigma):
    grid = create_cubed_sphere(8)
    state = held_suarez_topo_init(grid, sigma, h_0=2000.0)

    assert state.T.data.shape == (6, 8, 8, 8)
    assert state.phis.data.shape == (6, 8, 8)
    assert bool(jnp.all(jnp.isfinite(state.T.data)))
    assert bool(jnp.all(jnp.isfinite(state.p_s.data)))

    # Mountain has non-zero peak, so phis must vary.
    phis_max = float(jnp.max(state.phis.data))
    phis_min = float(jnp.min(state.phis.data))
    assert phis_max > 0.0
    assert phis_min == 0.0  # cosine-bell is exactly zero outside R_m

    # Surface pressure must be hydrostatically lowered over the mountain
    p_max = float(jnp.max(state.p_s.data))
    p_min = float(jnp.min(state.p_s.data))
    assert p_max == pytest.approx(constants.p_ref, rel=1e-6)
    assert p_min < constants.p_ref


def test_topo_init_latlon(sigma):
    grid = create_latlon_grid(24, 48)
    state = held_suarez_topo_init_latlon(grid, sigma, h_0=2000.0)
    assert state.T.data.shape == (24, 48, 8)
    assert bool(jnp.all(jnp.isfinite(state.p_s.data)))
    assert float(jnp.max(state.phis.data)) > 0.0


def test_topo_init_mpas(sigma):
    mesh = create_voronoi_mesh(4)
    state = held_suarez_topo_init_mpas(mesh, sigma, h_0=2000.0)
    assert state.T.data.shape == (mesh.nCells, 8)
    assert bool(jnp.all(jnp.isfinite(state.T.data)))


def test_topo_init_spectral(sigma):
    grid = create_gaussian_grid(21)
    state = held_suarez_topo_init_spectral(grid, sigma, h_0=2000.0)
    assert state.T_hat.data.shape == (grid.n_sh, 8)
    # phis should be non-zero in spectral coefficients (some power away
    # from zero mode)
    phis_amp = float(jnp.max(jnp.abs(state.phis_hat.data)))
    assert phis_amp > 0.0
