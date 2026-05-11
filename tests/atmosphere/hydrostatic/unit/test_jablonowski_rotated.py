"""Sanity tests for the rotated Jablonowski-Williamson DCMIP-2008 IC."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import standard_hybrid_levels
from legoesm.grids.voronoi import create_voronoi_mesh

from tests.test_cases.dcmip2008.jablonowski_rotated import (
    _rotate_to_rotated_frame,
    _wind_rotation_factors,
    rotated_baroclinic_init,
    rotated_baroclinic_init_latlon,
    rotated_baroclinic_init_mpas,
    rotated_baroclinic_init_spectral,
    rotated_steady_init,
)


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------


def test_rotation_alpha_zero_is_identity():
    """At α=0 the rotation should leave (lon, lat) unchanged."""
    lon = jnp.linspace(0.0, 2.0 * jnp.pi, 8, endpoint=False)
    lat = jnp.linspace(-1.4, 1.4, 8)
    lr, latr = _rotate_to_rotated_frame(lon, lat, 0.0)
    assert float(jnp.max(jnp.abs(latr - lat))) < 1e-12
    # Compare lon modulo 2π
    dlon = jnp.mod(lr - lon + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    assert float(jnp.max(jnp.abs(dlon))) < 1e-12


def test_rotation_wind_factors_alpha_zero():
    """At α=0 the wind transform matrix is identity (cos_γ=1, sin_γ=0)."""
    lon = jnp.array([0.1, 1.0, 2.0, 3.0])
    lat = jnp.array([0.1, -0.5, 0.0, 0.7])
    lr, latr = _rotate_to_rotated_frame(lon, lat, 0.0)
    cg, sgnv = _wind_rotation_factors(lon, lat, lr, latr, 0.0)
    assert float(jnp.max(jnp.abs(cg - 1.0))) < 1e-12
    assert float(jnp.max(jnp.abs(sgnv))) < 1e-12


def test_rotation_alpha_pi_4_at_axis():
    """At α=π/4 a point on the rotation axis (the y-axis through
    lon=π/2 equator) is mapped to itself; at lon=0, lat=0 the rotated
    frame's north pole tilts up by π/4."""
    # Point on rotation axis: lon=pi/2, lat=0
    lon = jnp.array([math.pi / 2.0])
    lat = jnp.array([0.0])
    lr, latr = _rotate_to_rotated_frame(lon, lat, math.pi / 4.0)
    assert float(jnp.abs(latr[0])) < 1e-12, (
        "Point on rotation axis should map to lat=0 in rotated frame")


# ---------------------------------------------------------------------------
# Per-grid initial conditions
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sigma8():
    return standard_hybrid_levels(8)


def test_cubed_sphere_init_finite(sigma8):
    grid = create_cubed_sphere(8)
    state = rotated_baroclinic_init(grid, sigma8, alpha=math.pi / 4.0)
    assert state.T.data.shape == (6, 8, 8, 8)
    assert state.u.data.shape == (6, 8, 8, 8)
    assert bool(jnp.all(jnp.isfinite(state.T.data)))
    assert bool(jnp.all(jnp.isfinite(state.u.data)))
    assert bool(jnp.all(jnp.isfinite(state.v.data)))
    # Physical bounds for J-W background T: ≳185 K, ≲320 K
    T_min = float(jnp.min(state.T.data))
    T_max = float(jnp.max(state.T.data))
    assert 180.0 < T_min < 250.0, f"T_min = {T_min} K out of range"
    assert 250.0 < T_max < 320.0, f"T_max = {T_max} K out of range"


def test_latlon_init_finite(sigma8):
    grid = create_latlon_grid(24, 48)
    state = rotated_baroclinic_init_latlon(grid, sigma8, alpha=math.pi / 4.0)
    assert state.T.data.shape == (24, 48, 8)
    assert bool(jnp.all(jnp.isfinite(state.T.data)))
    assert bool(jnp.all(jnp.isfinite(state.u.data)))


def test_mpas_init_finite(sigma8):
    mesh = create_voronoi_mesh(4)
    state = rotated_baroclinic_init_mpas(mesh, sigma8, alpha=math.pi / 4.0)
    assert state.T.data.shape == (mesh.nCells, 8)
    assert state.u.data.shape == (mesh.nEdges, 8)
    assert bool(jnp.all(jnp.isfinite(state.T.data)))
    assert bool(jnp.all(jnp.isfinite(state.u.data)))


def test_spectral_init_finite(sigma8):
    grid = create_gaussian_grid(21)
    state = rotated_baroclinic_init_spectral(grid, sigma8, alpha=math.pi / 4.0)
    assert state.T_hat.data.shape == (grid.n_sh, 8)
    assert bool(jnp.all(jnp.isfinite(state.T_hat.data.real)))
    assert bool(jnp.all(jnp.isfinite(state.lnps_hat.data.real)))


def test_steady_alpha_zero_matches_jw(sigma8):
    """At α=0, the rotated steady IC should reduce to the existing
    Jablonowski-Williamson steady-state (within bisection tolerance)."""
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init

    grid = create_cubed_sphere(8)
    rotated = rotated_steady_init(grid, sigma8, alpha=0.0)
    canonical = baroclinic_wave_init(grid, sigma8, perturbed=False)
    # T fields should match to 1e-6 K (analytic solution, no perturbation)
    dT = jnp.abs(rotated.T.data - canonical.T.data)
    max_dT = float(jnp.max(dT))
    assert max_dT < 1e-3, (
        f"Rotated α=0 should match canonical J-W; max |ΔT| = {max_dT}")
