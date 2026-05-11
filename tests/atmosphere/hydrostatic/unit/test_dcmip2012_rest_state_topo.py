"""Sanity tests for the DCMIP 2012 §2-0-0 rest-state-with-topography IC."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import standard_hybrid_levels
from legoesm.grids.voronoi import create_voronoi_mesh

from tests.test_cases.dcmip2012.rest_state_topography import (
    rest_state_topography_init,
    rest_state_topography_init_latlon,
    rest_state_topography_init_mpas,
    rest_state_topography_init_spectral,
)


@pytest.fixture(scope="module")
def sigma():
    return standard_hybrid_levels(8)


def _check_rest_state_invariants(state, *, h_0=2000.0, T_0=300.0,
                                  is_spectral=False):
    """Common sanity asserts for any rest-state-topography IC."""
    if is_spectral:
        # Spectral state stores complex coefficients; check finiteness
        # and rest-condition via vor=div=0.
        assert float(jnp.max(jnp.abs(state.vor_hat.data))) < 1e-10
        assert float(jnp.max(jnp.abs(state.div_hat.data))) < 1e-10
        return

    # Grid-space states: u=v=0 exactly, T = T_0 everywhere.
    assert float(jnp.max(jnp.abs(state.u.data))) == 0.0
    if state.v is not None:
        assert float(jnp.max(jnp.abs(state.v.data))) == 0.0
    assert float(jnp.max(jnp.abs(state.T.data - T_0))) < 1e-6

    # Surface geopotential: 0 ≤ phis ≤ g·h_0
    assert float(jnp.min(state.phis.data)) >= 0.0
    assert float(jnp.max(state.phis.data)) <= constants.g * h_0 * 1.001

    # Surface pressure: hydrostatically reduced from p_ref by mountain
    # height — p_s_min = p_ref·exp(-g·h_0/(R_d·T_0))
    p_min_expected = constants.p_ref * jnp.exp(
        -constants.g * h_0 / (constants.R_d * T_0))
    assert float(jnp.min(state.p_s.data)) >= float(p_min_expected) * 0.99
    assert float(jnp.max(state.p_s.data)) <= constants.p_ref + 1e-6


def test_rest_state_cubed_sphere(sigma):
    grid = create_cubed_sphere(8)
    state = rest_state_topography_init(grid, sigma, h_0=2000.0)
    _check_rest_state_invariants(state)


def test_rest_state_latlon(sigma):
    grid = create_latlon_grid(24, 48)
    state = rest_state_topography_init_latlon(grid, sigma, h_0=2000.0)
    _check_rest_state_invariants(state)


def test_rest_state_mpas(sigma):
    mesh = create_voronoi_mesh(4)
    state = rest_state_topography_init_mpas(mesh, sigma, h_0=2000.0)
    # MPAS state has u at edges only, no v
    assert float(jnp.max(jnp.abs(state.u.data))) == 0.0
    assert float(jnp.max(jnp.abs(state.T.data - 300.0))) < 1e-6
    assert float(jnp.min(state.phis.data)) >= 0.0


def test_rest_state_spectral(sigma):
    grid = create_gaussian_grid(21)
    state = rest_state_topography_init_spectral(grid, sigma, h_0=2000.0)
    _check_rest_state_invariants(state, is_spectral=True)
