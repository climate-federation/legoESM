"""Sigma-coordinate omega fed to WRF Tiedtke (vote V20a):
``omega_k = sigma_k (dp_s/dt + v_k.grad p_s) + p_s sigmadot_k`` from
``div(p_s v)``, versus the constant-p_s form Kain-Fritsch still uses."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.convection.integration import (
    diagnose_omega_columns_hydrostatic as omega_fn,
)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate

NLEV = 6


def _setup(shear, ps_wave):
    grid = create_latlon_grid(n_lat=16, n_lon=32)
    sigma = create_sigma_coordinate(NLEV, sigma_top=0.05)
    lat = jnp.asarray(grid.grid_lat)[..., None]
    lon = jnp.asarray(grid.grid_lon)
    prof = 1.0 + shear * jnp.linspace(1.0, -1.0, NLEV)       # top -> surface
    u = 10.0 * jnp.cos(lat) * prof                            # solid body
    v = jnp.zeros_like(u)
    p_s = 1.0e5 + ps_wave * jnp.sin(2.0 * lon) * jnp.cos(jnp.asarray(grid.grid_lat))
    return grid, sigma, u, v, p_s


def _omega(full, *a):
    grid, sigma, u, v, p_s = a
    n = int(np.prod(p_s.shape))
    return np.asarray(omega_fn(u, v, p_s, grid, sigma, n, NLEV,
                               full_continuity=full))


def test_uniform_surface_pressure_reduces_to_constant_ps_form():
    a = _setup(shear=0.8, ps_wave=0.0)
    full, const = _omega(True, *a), _omega(False, *a)
    scale = np.max(np.abs(const)) + 1e-30
    assert np.max(np.abs(full - const)) <= 1e-12 * max(scale, 1.0)


def test_pure_advection_of_surface_pressure_has_no_vertical_motion():
    """Height-uniform wind over varying p_s: dp_s/dt = -v.grad p_s, so
    Dp/Dt = 0 on every level (exact identity of the discrete formula)."""
    a = _setup(shear=0.0, ps_wave=2000.0)
    full = _omega(True, *a)
    grid, sigma, u, v, p_s = a
    # round-off scale: the mass-flux differences the discrete divergence
    # cancels, p_s |u| / (radius dlon) (vote V22)
    flux_scale = 1.0e5 * 10.0 / (grid.radius * (2.0 * np.pi / 32))
    assert np.max(np.abs(full)) <= 1e-9 * flux_scale


def test_sheared_flow_over_varying_ps_moves_air_vertically():
    """With shear the v.grad p_s terms no longer cancel: omega is non-zero,
    while the constant-p_s form (div v = 0 for solid-body flow) gives 0."""
    a = _setup(shear=0.8, ps_wave=2000.0)
    full, const = _omega(True, *a), _omega(False, *a)
    assert np.max(np.abs(const)) <= 1e-12
    assert np.max(np.abs(full)) > 1e-3     # Pa/s; order sigma v.grad p_s
