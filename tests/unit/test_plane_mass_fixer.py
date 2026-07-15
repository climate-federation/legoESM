"""Tests for ``compute_dry_mass_plane`` and
``fix_mass_nonhydrostatic_plane`` in
``src/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py``.

Covers:

- Mass of the rest state equals the analytic integral of ``rho_ref``.
- Adding a uniform ``+epsilon`` to ``rho_prime`` increases the
  integral by exactly ``epsilon * sum(J * area * dz)``.
- The mass fixer recovers a stored target exactly from an
  artificially mass-biased state (rel < 1e-12 in x64).
- The fixer preserves every spatial gradient of ``rho_prime``
  (correction is spatially uniform).
- Non-flat terrain is rejected at the dycore boundary.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    _assert_flat_terrain,
    compute_dry_mass_plane,
    fix_mass_nonhydrostatic_plane,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate, TerrainMetric


jax.config.update("jax_enable_x64", True)


def _setup(ny=4, nx=5, nlev=6, H=20.0e3):
    grid = create_plane_grid(
        nx=nx, ny=ny, nlev=nlev, dx=1.0e3, dy=2.0e3, dtype=jnp.float64
    )
    height_coord = create_height_coordinate(grid.nlev, H=H)
    terrain = make_flat_plane_terrain_metric(grid, height_coord)
    state = make_rest_state(grid, height_coord, dtype=jnp.float64)
    return grid, height_coord, terrain, state


def test_rest_state_mass_matches_reference_profile_integral():
    grid, height_coord, terrain, state = _setup()
    mass = compute_dry_mass_plane(state, grid, height_coord, terrain)
    expected = jnp.sum(height_coord.rho_ref * height_coord.dz) * grid.total_area
    assert float(mass) == pytest.approx(float(expected), rel=1.0e-14)


def test_mass_responds_linearly_to_uniform_rho_prime_shift():
    grid, height_coord, terrain, state = _setup()
    mass_0 = compute_dry_mass_plane(state, grid, height_coord, terrain)
    epsilon = 1.0e-3
    shifted = state._replace(
        rho_prime=state.rho_prime.replace(
            data=state.rho_prime.data + epsilon,
        )
    )
    mass_1 = compute_dry_mass_plane(shifted, grid, height_coord, terrain)
    weighted_vol = (
        jnp.sum(terrain.jacobian[:, :, None] * grid.area_T[:, :, None]
                * height_coord.dz)
    )
    delta_expected = epsilon * weighted_vol
    # Subtraction of two ~8e11 totals loses ~12 digits of precision to
    # cancellation; the difference matches the analytic value to ~1e-12
    # relative under x64, which is the appropriate tolerance for an
    # accumulated reduction.
    assert float(mass_1 - mass_0) == pytest.approx(
        float(delta_expected), rel=1.0e-11
    )


def test_fix_mass_restores_exact_target():
    grid, height_coord, terrain, state = _setup()
    target = compute_dry_mass_plane(state, grid, height_coord, terrain)
    # Add a non-uniform bias so the correction is non-trivial.
    rng = np.random.default_rng(0)
    bias = jnp.asarray(rng.standard_normal(state.rho_prime.data.shape)) * 1.0e-4
    biased = state._replace(
        rho_prime=state.rho_prime.replace(data=state.rho_prime.data + bias)
    )
    corrected = fix_mass_nonhydrostatic_plane(
        biased, target, grid, height_coord, terrain
    )
    final = compute_dry_mass_plane(corrected, grid, height_coord, terrain)
    assert float(jnp.abs(final - target) / jnp.abs(target)) < 1.0e-12


def test_fix_mass_preserves_spatial_gradients_of_rho_prime():
    """Uniform additive correction must not change any difference
    between two cells of ``rho_prime``."""
    grid, height_coord, terrain, state = _setup()
    rng = np.random.default_rng(1)
    bias = jnp.asarray(rng.standard_normal(state.rho_prime.data.shape)) * 1.0e-4
    biased = state._replace(
        rho_prime=state.rho_prime.replace(data=state.rho_prime.data + bias)
    )
    target = compute_dry_mass_plane(state, grid, height_coord, terrain)
    corrected = fix_mass_nonhydrostatic_plane(
        biased, target, grid, height_coord, terrain
    )
    delta = corrected.rho_prime.data - biased.rho_prime.data
    # The correction must be a single scalar broadcast to the shape.
    assert float(jnp.max(delta) - jnp.min(delta)) < 1.0e-15


def test_fix_mass_is_idempotent_when_already_at_target():
    grid, height_coord, terrain, state = _setup()
    target = compute_dry_mass_plane(state, grid, height_coord, terrain)
    corrected = fix_mass_nonhydrostatic_plane(
        state, target, grid, height_coord, terrain
    )
    assert jnp.allclose(corrected.rho_prime.data, state.rho_prime.data)


def test_flat_terrain_assert_passes_on_flat():
    grid, height_coord, terrain, _ = _setup()
    _assert_flat_terrain(terrain)


def test_flat_terrain_assert_rejects_non_flat_jacobian():
    grid, height_coord, terrain, _ = _setup()
    bad = TerrainMetric(
        z_s=terrain.z_s,
        jacobian=terrain.jacobian + 0.5,  # J != 1
        z_full_3d=terrain.z_full_3d,
        z_half_3d=terrain.z_half_3d,
    )
    with pytest.raises(ValueError, match="Jacobian == 1"):
        _assert_flat_terrain(bad)


def test_flat_terrain_assert_rejects_non_zero_z_s():
    grid, height_coord, terrain, _ = _setup()
    bad = TerrainMetric(
        z_s=terrain.z_s + 100.0,
        jacobian=terrain.jacobian,
        z_full_3d=terrain.z_full_3d,
        z_half_3d=terrain.z_half_3d,
    )
    with pytest.raises(ValueError, match="z_s == 0"):
        _assert_flat_terrain(bad)


def test_compute_dry_mass_is_differentiable():
    grid, height_coord, terrain, state = _setup()

    def loss(rho_prime_data):
        s = state._replace(
            rho_prime=state.rho_prime.replace(data=rho_prime_data)
        )
        return compute_dry_mass_plane(s, grid, height_coord, terrain)

    g_fn = jax.grad(loss)
    grad = g_fn(state.rho_prime.data)
    # dM/drho_prime[j,i,k] = J[j,i] * area_T[j,i] * dz[k]
    expected = (
        terrain.jacobian[:, :, None] * grid.area_T[:, :, None]
        * height_coord.dz
    )
    assert jnp.allclose(grad, expected, rtol=1.0e-14)
