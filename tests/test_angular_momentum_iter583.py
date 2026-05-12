"""FV3_3D iter 583: atmospheric angular momentum diagnostic test.

Faithful port of FV3 ``compute_aam`` (fv_dynamics.F90:1264).

Tests
-----

1. ``test_aam_rest_is_solid_body`` — at zero u, AAM = R²·Ω·∫ρ·cos²(lat)·dV.
2. ``test_aam_changes_with_zonal_wind`` — adding positive uniform u
   increases AAM (correct sign).
3. ``test_aam_finite_and_positive_at_rest``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.diagnostics import compute_atmospheric_angular_momentum
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_height_coordinate


def test_aam_finite_and_positive_at_rest():
    """At rest (u=0), AAM = R²·Ω·sum(rho·dz·cos²(lat)·area)."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    hc = create_height_coordinate(nlev, z_top)
    u_center = jnp.zeros((6, n, n, nlev))
    rho_full = jnp.broadcast_to(
        jnp.asarray(hc.rho_ref)[None, None, None, :],
        (6, n, n, nlev),
    )
    col, total = compute_atmospheric_angular_momentum(
        u_center, rho_full, grid, hc,
    )
    assert col.shape == (6, n, n)
    assert np.isfinite(total)
    assert total > 0.0  # Earth rotation contributes positive AAM


def test_aam_rest_is_solid_body():
    """At u=0 with constant rho: AAM should equal closed-form for solid
    body rotation: R²·Ω·M·<cos²(lat)>·∫dV."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    hc = create_height_coordinate(nlev, z_top)
    u_center = jnp.zeros((6, n, n, nlev))
    rho_const = 1.0  # uniform rho
    rho_full = jnp.full((6, n, n, nlev), rho_const)
    _, total = compute_atmospheric_angular_momentum(
        u_center, rho_full, grid, hc,
    )
    # Closed form: ρ * Ω * R² * ∫cos²(lat)·dV
    cos_lat = np.asarray(jnp.cos(grid.lat))
    area = np.asarray(grid.area)
    dz_total = float(jnp.sum(jnp.asarray(hc.dz)))
    expected = (
        rho_const * constants.Omega * grid.radius ** 2
        * np.sum(cos_lat ** 2 * area) * dz_total
    )
    rel_err = abs(total - expected) / abs(expected)
    assert rel_err < 1e-5, (
        f"AAM at rest should match closed form, got "
        f"{total:.4e} vs expected {expected:.4e} "
        f"(rel_err={rel_err:.2e})"
    )


def test_aam_changes_with_zonal_wind():
    """Adding +1 m/s uniform u_center should increase AAM (positive)."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    hc = create_height_coordinate(nlev, z_top)
    rho_full = jnp.full((6, n, n, nlev), 1.0)
    _, total_rest = compute_atmospheric_angular_momentum(
        jnp.zeros((6, n, n, nlev)), rho_full, grid, hc,
    )
    _, total_pos = compute_atmospheric_angular_momentum(
        jnp.ones((6, n, n, nlev)), rho_full, grid, hc,
    )
    delta = total_pos - total_rest
    # delta should equal R·∫cos(lat)·dm·u = R·sum(cos·area·dz·rho)·1
    cos_lat = np.asarray(jnp.cos(grid.lat))
    area = np.asarray(grid.area)
    dz_total = float(jnp.sum(jnp.asarray(hc.dz)))
    expected_delta = grid.radius * np.sum(cos_lat * area) * dz_total
    rel_err = abs(delta - expected_delta) / abs(expected_delta)
    assert rel_err < 1e-5, (
        f"AAM delta for +1 m/s u: {delta:.4e} vs expected "
        f"{expected_delta:.4e}"
    )
