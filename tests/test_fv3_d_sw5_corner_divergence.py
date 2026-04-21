"""Tests for `fv3_d_sw5_corner_divergence` (Fortran d_sw5 port, iter-759).

Iter-759 is the first source-code step of the d_sw5 port chain.  It
delivers the standalone corner-divergence algorithm from
`sw_core.F90:1641-1719`.  These tests verify shape, sign, and basic
structural properties without yet wiring into production.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.fv3_d_sw5_corner_divergence import (
    fv3_d_sw5_corner_divergence, _cell_centre_winds)


@pytest.fixture
def cdgrid():
    n = 8
    grid = create_cubed_sphere(n)
    return create_cubed_sphere_cdgrid(grid)


def test_corner_divergence_shape(cdgrid):
    """Output shape must be (6, n-1, n-1) — INTERIOR B-grid corners
    (iter-759 scope).  Full (6, n+1, n+1) boundary-aware corners
    deferred to iter-760+."""
    n = cdgrid.base.n
    u_d = jnp.zeros((6, n, n + 1))
    v_d = jnp.zeros((6, n + 1, n))
    delpc = fv3_d_sw5_corner_divergence(u_d, v_d, cdgrid)
    assert delpc.shape == (6, n - 1, n - 1)


def test_corner_divergence_zero_field(cdgrid):
    """Zero D-grid winds → zero divergence at corners."""
    n = cdgrid.base.n
    u_d = jnp.zeros((6, n, n + 1))
    v_d = jnp.zeros((6, n + 1, n))
    delpc = fv3_d_sw5_corner_divergence(u_d, v_d, cdgrid)
    assert bool(jnp.all(delpc == 0.0))


def test_cell_centre_winds_shape(cdgrid):
    """Helper _cell_centre_winds returns correct A-grid shape."""
    n = cdgrid.base.n
    u_d = jnp.asarray(np.random.default_rng(0).normal(size=(6, n, n + 1)))
    v_d = jnp.asarray(np.random.default_rng(1).normal(size=(6, n + 1, n)))
    u_cc, v_cc = _cell_centre_winds(u_d, v_d)
    assert u_cc.shape == (6, n, n)
    assert v_cc.shape == (6, n, n)


def test_cell_centre_winds_averaging(cdgrid):
    """_cell_centre_winds computes 0.5*(u_d[south] + u_d[north])."""
    n = cdgrid.base.n
    u_d = jnp.full((6, n, n + 1), 2.0)
    v_d = jnp.full((6, n + 1, n), 3.0)
    u_cc, v_cc = _cell_centre_winds(u_d, v_d)
    # Averaging a constant gives the same constant.
    assert bool(jnp.allclose(u_cc, 2.0))
    assert bool(jnp.allclose(v_cc, 3.0))


def test_corner_divergence_linear_in_winds(cdgrid):
    """Divergence is linear in (u_d, v_d): scaling by k scales output by k."""
    n = cdgrid.base.n
    rng = np.random.default_rng(42)
    u_d = jnp.asarray(rng.normal(size=(6, n, n + 1)))
    v_d = jnp.asarray(rng.normal(size=(6, n + 1, n)))
    delpc1 = fv3_d_sw5_corner_divergence(u_d, v_d, cdgrid)
    delpc2 = fv3_d_sw5_corner_divergence(2.0 * u_d, 2.0 * v_d, cdgrid)
    # Linearity: 2x input → 2x output (at machine precision).
    assert bool(jnp.allclose(delpc2, 2.0 * delpc1, atol=1e-12))


def test_corner_divergence_units_1_per_second(cdgrid):
    """Divergence has units [1/s]; for O(1) winds over ~6371 km radius,
    magnitude should be O(1e-7) at typical cell size C8 (radius/n).
    Not a strict bound — just a sanity check on order of magnitude."""
    n = cdgrid.base.n
    rng = np.random.default_rng(1)
    # Winds of O(1) m/s.
    u_d = jnp.asarray(rng.normal(size=(6, n, n + 1)))
    v_d = jnp.asarray(rng.normal(size=(6, n + 1, n)))
    delpc = fv3_d_sw5_corner_divergence(u_d, v_d, cdgrid)
    magnitude = float(jnp.max(jnp.abs(delpc)))
    # For C8 grid, cell size ~6371 km / 8 ~ 800 km.
    # Divergence ~ du/dx ~ 1 m/s / 800 km ~ 1.25e-6 /s.
    # Random noise can give larger; check we're within O(1e-3).
    assert 1e-10 < magnitude < 1e-3, \
        f"Corner divergence magnitude {magnitude:.2e} suspicious"
