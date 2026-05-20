"""Regression: stress_divergence_cgrid must handle 3-D stresses on
non-tripolar regular and channel lat-lon grids.

Pre-fix the non-tripolar branch built ``area_u_dual`` / ``area_v_dual``
as 1-D ``(n_lat,)`` arrays and then tried to divide a 3-D
``(n_lat, n_lon+1, n_lev)`` tendency by ``area_u_dual[..., None]`` —
shape ``(n_lat, 1)`` — which raises ``Incompatible shapes for
broadcasting``. The Eady-uniform and ACC-channel ocean-matrix runs
on ``latlon_channel`` reproduce the failure.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    stress_divergence_cgrid,
)


def _run_full_pipeline(grid, n_lev=None):
    """Drive strain_rate_cgrid → stress_divergence_cgrid end-to-end."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        strain_rate_cgrid,
    )
    if n_lev is None:
        u_shape = (grid.n_lat, grid.n_lon + 1)
        v_shape = (grid.n_lat + 1, grid.n_lon)
    else:
        u_shape = (grid.n_lat, grid.n_lon + 1, n_lev)
        v_shape = (grid.n_lat + 1, grid.n_lon, n_lev)
    rng = jax.random.PRNGKey(0)
    ku, kv = jax.random.split(rng)
    u = jax.random.normal(ku, u_shape)
    v = jax.random.normal(kv, v_shape)
    D_T, D_S = strain_rate_cgrid(u, v, grid)
    tend_u, tend_v = stress_divergence_cgrid(
        D_T, D_S, grid, normalize=True,
    )
    return tend_u, tend_v


def test_stress_divergence_cgrid_3d_normalized_regular_latlon():
    """3-D stresses must not raise on the regular non-tripolar branch.

    Pre-fix this triggered ``Incompatible shapes for broadcasting:
    shapes=[(n_lat, n_lon+1, n_lev), (n_lat, 1)]`` from dividing by a
    1-D ``area_u_dual``.
    """
    grid = create_latlon_grid(n_lat=16, n_lon=32)
    n_lev = 4
    tend_u, tend_v = _run_full_pipeline(grid, n_lev=n_lev)
    assert tend_u.shape == (grid.n_lat, grid.n_lon + 1, n_lev)
    assert tend_v.shape == (grid.n_lat + 1, grid.n_lon, n_lev)
    assert jnp.isfinite(tend_u).all()
    assert jnp.isfinite(tend_v).all()


def test_stress_divergence_cgrid_2d_still_works():
    """Pre-existing 2-D path must not regress."""
    grid = create_latlon_grid(n_lat=16, n_lon=32)
    tend_u, tend_v = _run_full_pipeline(grid, n_lev=None)
    assert tend_u.shape == (grid.n_lat, grid.n_lon + 1)
    assert tend_v.shape == (grid.n_lat + 1, grid.n_lon)
    assert jnp.isfinite(tend_u).all()
    assert jnp.isfinite(tend_v).all()
