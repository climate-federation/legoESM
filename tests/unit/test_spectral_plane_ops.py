"""Spectral plane operator tests.

Pins the round-trip identity, the exact differential operators (vs
analytic plane waves), the 2/3 dealias mask shape + coverage, and
the discrete adjoint identity between PG and divergence in spectral
space.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.spectral_plane_ops import (
    apply_dealias,
    biharmonic_spec,
    divergence_spec,
    grad_x_spec,
    grad_y_spec,
    laplacian_spec,
    make_spectral_axis,
    to_phys,
    to_spec,
    two_thirds_dealias_mask,
)
from legoesm.grids.plane import create_plane_grid

jax.config.update("jax_enable_x64", True)


def _grid(nx=16, ny=12, nlev=4, dx=1000.0, dy=1000.0):
    return create_plane_grid(
        nx=nx, ny=ny, nlev=nlev, dx=dx, dy=dy, dtype=jnp.float64,
    )


def test_round_trip_to_spec_to_phys_identity():
    """``irfft2(rfft2(φ)) == φ`` to machine epsilon for any real input."""
    grid = _grid()
    rng = np.random.default_rng(0)
    phi = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    phi_hat = to_spec(phi)
    phi_back = to_phys(phi_hat, grid.nx, grid.ny)
    np.testing.assert_allclose(
        np.asarray(phi_back), np.asarray(phi), rtol=0.0, atol=1.0e-12,
    )
    assert phi_hat.shape == (grid.ny, grid.nx // 2 + 1, grid.nlev)
    assert jnp.iscomplexobj(phi_hat)


def test_grad_x_spec_on_plane_wave_matches_analytic():
    """``∂/∂x exp(i·k_x·x) = i·k_x·exp(i·k_x·x)``. Spectral grad must
    reproduce the analytic eigenvalue exactly."""
    grid = _grid(nx=32, ny=24)
    axis = make_spectral_axis(grid)
    # Build a single-mode plane wave: cos(k0·x) where k0 picks the
    # 3rd Fourier mode.
    m = 3
    k0 = 2.0 * jnp.pi * m / grid.Lx
    x = jnp.arange(grid.nx, dtype=jnp.float64) * (grid.Lx / grid.nx)
    phi = jnp.broadcast_to(
        jnp.cos(k0 * x)[None, :, None],
        (grid.ny, grid.nx, grid.nlev),
    )
    phi_hat = to_spec(phi)
    dphi_dx_hat = grad_x_spec(phi_hat, axis)
    dphi_dx = to_phys(dphi_dx_hat, grid.nx, grid.ny)
    expected = -k0 * jnp.broadcast_to(
        jnp.sin(k0 * x)[None, :, None],
        (grid.ny, grid.nx, grid.nlev),
    )
    np.testing.assert_allclose(
        np.asarray(dphi_dx), np.asarray(expected),
        rtol=1.0e-10, atol=1.0e-10,
    )


def test_grad_y_spec_on_plane_wave_matches_analytic():
    grid = _grid(nx=24, ny=32)
    axis = make_spectral_axis(grid)
    m = 4
    k0 = 2.0 * jnp.pi * m / grid.Ly
    y = jnp.arange(grid.ny, dtype=jnp.float64) * (grid.Ly / grid.ny)
    phi = jnp.broadcast_to(
        jnp.cos(k0 * y)[:, None, None],
        (grid.ny, grid.nx, grid.nlev),
    )
    phi_hat = to_spec(phi)
    dphi_dy_hat = grad_y_spec(phi_hat, axis)
    dphi_dy = to_phys(dphi_dy_hat, grid.nx, grid.ny)
    expected = -k0 * jnp.broadcast_to(
        jnp.sin(k0 * y)[:, None, None],
        (grid.ny, grid.nx, grid.nlev),
    )
    np.testing.assert_allclose(
        np.asarray(dphi_dy), np.asarray(expected),
        rtol=1.0e-10, atol=1.0e-10,
    )


def test_laplacian_spec_on_plane_wave_matches_analytic():
    grid = _grid(nx=16, ny=16)
    axis = make_spectral_axis(grid)
    mx, my = 2, 3
    kx = 2.0 * jnp.pi * mx / grid.Lx
    ky = 2.0 * jnp.pi * my / grid.Ly
    x = jnp.arange(grid.nx, dtype=jnp.float64) * (grid.Lx / grid.nx)
    y = jnp.arange(grid.ny, dtype=jnp.float64) * (grid.Ly / grid.ny)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    phi = jnp.cos(kx * X + ky * Y)
    phi_hat = to_spec(phi)
    lap_hat = laplacian_spec(phi_hat, axis)
    lap = to_phys(lap_hat, grid.nx, grid.ny)
    expected = -(kx ** 2 + ky ** 2) * phi
    np.testing.assert_allclose(
        np.asarray(lap), np.asarray(expected),
        rtol=1.0e-10, atol=1.0e-10,
    )


def test_pg_div_adjoint_identity_exact_in_spectral():
    """Parseval gives ``sum(φ · div(u, v)) = -sum(u · ∂φ/∂x) -
    sum(v · ∂φ/∂y)`` to round-off for any periodic real field. Pins
    the energy-consistent PG/divergence pair in spectral space."""
    grid = _grid(nx=16, ny=12)
    axis = make_spectral_axis(grid, dealias=False)
    rng = np.random.default_rng(7)
    phi = jnp.asarray(rng.standard_normal((grid.ny, grid.nx)))
    u = jnp.asarray(rng.standard_normal((grid.ny, grid.nx)))
    v = jnp.asarray(rng.standard_normal((grid.ny, grid.nx)))

    phi_hat = to_spec(phi)
    u_hat = to_spec(u)
    v_hat = to_spec(v)

    div_uv = to_phys(divergence_spec(u_hat, v_hat, axis), grid.nx, grid.ny)
    grad_x_phi = to_phys(grad_x_spec(phi_hat, axis), grid.nx, grid.ny)
    grad_y_phi = to_phys(grad_y_spec(phi_hat, axis), grid.nx, grid.ny)

    lhs = float(jnp.sum(phi * div_uv))
    rhs = float(-jnp.sum(u * grad_x_phi) - jnp.sum(v * grad_y_phi))
    np.testing.assert_allclose(lhs, rhs, rtol=1.0e-10, atol=1.0e-10)


def test_two_thirds_dealias_mask_shape_and_cutoff():
    mask = two_thirds_dealias_mask(nx=12, ny=9)
    nx_r = 12 // 2 + 1
    assert mask.shape == (9, nx_r)
    # nx_cut = 12 // 3 = 4 → kx indices 0..4 kept (5 modes).
    # ny_cut = 9 // 3 = 3 → ky signed indices |n| <= 3 kept.
    assert int(jnp.sum(mask[0, :])) == 5   # kx kept on the k=0 ky row
    # ky modes kept: indices where |signed_idx| <= 3. Under fftfreq
    # for ny=9: signed = [0,1,2,3,4,-4,-3,-2,-1]; |.| <= 3 picks 7.
    assert int(jnp.sum(mask[:, 0])) == 7


def test_dealias_idempotent():
    """Applying the dealias mask twice equals applying it once."""
    grid = _grid()
    axis = make_spectral_axis(grid, dealias=True)
    rng = np.random.default_rng(2)
    phi = jnp.asarray(
        rng.standard_normal((grid.ny, grid.nx, grid.nlev)),
    )
    phi_hat = to_spec(phi)
    once = apply_dealias(phi_hat, axis)
    twice = apply_dealias(once, axis)
    np.testing.assert_allclose(
        np.asarray(twice), np.asarray(once), rtol=0.0, atol=0.0,
    )


def test_biharmonic_spec_on_plane_wave():
    """``∇⁴ exp(i(kx·x + ky·y)) = (kx² + ky²)² · exp(...)``."""
    grid = _grid(nx=16, ny=16)
    axis = make_spectral_axis(grid, dealias=False)
    mx, my = 2, 1
    kx = 2.0 * jnp.pi * mx / grid.Lx
    ky = 2.0 * jnp.pi * my / grid.Ly
    x = jnp.arange(grid.nx, dtype=jnp.float64) * (grid.Lx / grid.nx)
    y = jnp.arange(grid.ny, dtype=jnp.float64) * (grid.Ly / grid.ny)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    phi = jnp.cos(kx * X + ky * Y)
    phi_hat = to_spec(phi)
    bih_hat = biharmonic_spec(phi_hat, axis)
    bih = to_phys(bih_hat, grid.nx, grid.ny)
    expected = (kx ** 2 + ky ** 2) ** 2 * phi
    np.testing.assert_allclose(
        np.asarray(bih), np.asarray(expected),
        rtol=1.0e-10, atol=1.0e-10,
    )


def test_spectral_ops_support_jax_grad():
    """Pseudo-spectral operators must be differentiable end-to-end."""
    grid = _grid(nx=8, ny=8)
    axis = make_spectral_axis(grid, dealias=False)

    def loss_fn(phi):
        phi_hat = to_spec(phi)
        lap_hat = laplacian_spec(phi_hat, axis)
        lap = to_phys(lap_hat, grid.nx, grid.ny)
        return jnp.sum(lap ** 2)

    rng = np.random.default_rng(3)
    phi = jnp.asarray(rng.standard_normal((grid.ny, grid.nx)))
    grad = jax.grad(loss_fn)(phi)
    assert grad.shape == phi.shape
    assert bool(jnp.all(jnp.isfinite(grad)))
