"""Eddy / turbulence diagnostics for the Silvestri et al. 2024 reproduction.

Covers the D1-D6 diagnostics added to ``ocean/diagnostics.py``: relative
vorticity at cell centres, KE/enstrophy integrals, eddy decomposition + EKE/
TKE/eddy-APE, w'b' eddy buoyancy flux, zonal power/co-spectra, zonal mean.
Each is checked against an analytic or hand-computed reference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants

jax.config.update("jax_enable_x64", True)

import legoesm.ocean.diagnostics as D


def _grid(n_lat=16, n_lon=32):
    from legoesm.grids.latlon import create_latlon_grid
    return create_latlon_grid(n_lat=n_lat, n_lon=n_lon, radius=constants.R_earth)


def test_velocity_to_cell_centre_shapes_and_uniform():
    grid = _grid()
    u = jnp.full((grid.n_lat, grid.n_lon + 1, 3), 0.4)
    v = jnp.full((grid.n_lat + 1, grid.n_lon, 3), -0.2)
    u_h, v_h = D.velocity_to_cell_centre(u, v)
    assert u_h.shape == (grid.n_lat, grid.n_lon, 3)
    assert v_h.shape == (grid.n_lat, grid.n_lon, 3)
    assert jnp.allclose(u_h, 0.4) and jnp.allclose(v_h, -0.2)


def test_relative_vorticity_zonal_shear():
    """A meridional shear in u (∂u/∂y < 0 going north) gives ζ = −∂u/∂y > 0,
    spatially uniform in lon (no zonal structure). Checks D1 sign, lon-uniformity,
    and order of magnitude (exact value carries the spherical metric, so don't
    over-specify it)."""
    grid = _grid()
    R = constants.R_earth
    y = R * np.asarray(grid.lat)                   # metres
    alpha = 1.0e-6
    u = jnp.asarray(np.broadcast_to((alpha * y)[:, None],
                                    (grid.n_lat, grid.n_lon + 1)).copy())
    v = jnp.zeros((grid.n_lat + 1, grid.n_lon))
    zeta = D.relative_vorticity_cell_centre(u, v, grid)
    assert zeta.shape == (grid.n_lat, grid.n_lon)
    interior = np.asarray(zeta)[4:-4, :]
    # u increases northward → ζ = -∂u/∂y < 0, uniform in longitude, O(alpha).
    assert np.all(interior < 0.0)
    assert np.allclose(interior, interior[:, :1], atol=1e-20)   # no zonal structure
    assert 0.1 * alpha < np.mean(np.abs(interior)) < 3.0 * alpha


def test_domain_kinetic_energy_uniform():
    grid = _grid()
    area = grid.area
    u_h = jnp.full((grid.n_lat, grid.n_lon), 3.0)
    v_h = jnp.full((grid.n_lat, grid.n_lon), 4.0)
    ke = float(D.domain_kinetic_energy(u_h, v_h, area))
    # ½(9+16)=12.5 per cell × total area.
    assert np.isclose(ke, 12.5 * float(jnp.sum(area)))


def test_enstrophy_nonneg_and_scales():
    grid = _grid()
    zeta = jnp.full((grid.n_lat, grid.n_lon), 2.0e-5)
    Z = float(D.domain_enstrophy(zeta, grid.area))
    assert np.isclose(Z, 0.5 * (2.0e-5) ** 2 * float(jnp.sum(grid.area)))


def test_remove_zonal_mean():
    f = jnp.asarray(np.arange(6.0).reshape(2, 3))  # rows [0,1,2],[3,4,5]; zonal means 1,4
    eddy = D.remove_zonal_mean(f)
    assert np.allclose(np.asarray(eddy), [[-1, 0, 1], [-1, 0, 1]])
    assert np.allclose(np.asarray(jnp.mean(eddy, axis=1)), 0.0)


def test_eddy_vs_total_kinetic_energy():
    """A purely zonal-mean flow (no zonal structure) has ZERO EKE; adding a
    zonal wave injects EKE < TKE."""
    grid = _grid()
    area = grid.area
    # Zonal-mean-only flow: u depends on lat only.
    u_mean = jnp.asarray(np.broadcast_to(np.linspace(0.1, 0.5, grid.n_lat)[:, None],
                                         (grid.n_lat, grid.n_lon)).copy())
    v0 = jnp.zeros((grid.n_lat, grid.n_lon))
    tke_mean = float(D.domain_kinetic_energy(u_mean, v0, area))
    # Zonal-mean-only flow → EKE is ~roundoff relative to TKE (area weights ~1e11).
    assert float(D.eddy_kinetic_energy(u_mean, v0, area)) < 1e-12 * tke_mean
    # Add a zonal wave.
    lon = np.asarray(grid.lon)[None, :]
    u_wave = u_mean + 0.2 * jnp.asarray(np.cos(3 * lon) * np.ones((grid.n_lat, 1)))
    eke = float(D.eddy_kinetic_energy(u_wave, v0, area))
    tke = float(D.domain_kinetic_energy(u_wave, v0, area))
    assert 0.0 < eke < tke


def test_eddy_ape_positive_and_zero_for_zonal():
    grid = _grid()
    area = grid.area
    N2 = 1.0e-5
    b_zonal = jnp.asarray(np.broadcast_to(np.linspace(-1e-3, 1e-3, grid.n_lat)[:, None],
                                          (grid.n_lat, grid.n_lon)).copy())
    # Zonal-mean buoyancy → eddy APE ~roundoff (area weights ~1e11); use a small
    # absolute floor scaled to the weights rather than a hard 0.
    assert float(D.eddy_available_potential_energy(b_zonal, N2, area)) < 1e-12 * float(jnp.sum(area))
    lon = np.asarray(grid.lon)[None, :]
    b_eddy = b_zonal + 1e-4 * jnp.asarray(np.sin(2 * lon) * np.ones((grid.n_lat, 1)))
    ape = float(D.eddy_available_potential_energy(b_eddy, N2, area))
    assert ape > 0.0


def test_vertical_eddy_buoyancy_flux():
    """w'b' picks up correlated eddy w and b; uncorrelated/zonal-mean parts drop."""
    grid = _grid()
    lon = np.asarray(grid.lon)[None, :] * np.ones((grid.n_lat, 1))
    w = jnp.asarray(0.01 * np.cos(2 * lon))          # zero zonal mean
    b = jnp.asarray(1e-3 * np.cos(2 * lon))          # in phase → positive flux
    wb = D.vertical_eddy_buoyancy_flux(w, b)
    assert float(jnp.mean(wb)) > 0.0
    # Out of phase → negative mean flux.
    b2 = jnp.asarray(1e-3 * np.sin(2 * lon))
    assert abs(float(jnp.mean(D.vertical_eddy_buoyancy_flux(w, b2)))) < 1e-6


def test_zonal_power_spectrum_single_wave_and_parseval():
    """A pure cos(m·x) eddy peaks at bin m, and the spectrum CONSERVES variance:
    Σ_k P[k] == zonal variance of the field (½ for unit-amplitude cos)."""
    grid = _grid(n_lat=8, n_lon=64)
    m = 5
    lon_idx = np.arange(grid.n_lon)[None, :]
    field = jnp.asarray(np.cos(2 * np.pi * m * lon_idx / grid.n_lon)
                        * np.ones((grid.n_lat, 1)))
    dx = 2 * np.pi / grid.n_lon
    k, P = D.zonal_power_spectrum(field, dx)
    P = np.asarray(P)
    assert int(np.argmax(P)) == m, (int(np.argmax(P)), m)
    P_sorted = np.sort(P)[::-1]
    assert P_sorted[0] > 100 * P_sorted[1]
    # Parseval: Σ P == variance of cos(m x) = 1/2.
    var = float(np.var(np.asarray(field)[0]))
    assert np.isclose(float(P.sum()), var, rtol=1e-10), (float(P.sum()), var)
    assert np.isclose(var, 0.5, atol=1e-12)


def test_zonal_cospectrum_conserves_covariance():
    """Σ_k co[k] == zonal covariance of a',b' — asserted against the FUNCTION
    output (not a NumPy tautology). Mixed wavenumbers + a phase offset so the
    interior factor-of-2 actually matters."""
    grid = _grid(n_lat=4, n_lon=32)
    n = grid.n_lon
    x = np.arange(n)[None, :] * np.ones((grid.n_lat, 1)) * (2 * np.pi / n)
    a = jnp.asarray(np.cos(3 * x) + 0.5 * np.cos(7 * x))
    b = jnp.asarray(2.0 * np.cos(3 * x + 0.4) - np.cos(7 * x))
    dx = 2 * np.pi / n
    _, co = D.zonal_cospectrum(a, b, dx)
    cov = float(jnp.mean(D.remove_zonal_mean(a) * D.remove_zonal_mean(b)))
    assert np.isclose(float(jnp.sum(co)), cov, rtol=1e-10), (float(jnp.sum(co)), cov)

    # And the power spectrum likewise conserves the auto-covariance (variance).
    _, Pa = D.zonal_power_spectrum(a, dx)
    var_a = float(jnp.mean(D.remove_zonal_mean(a) ** 2))
    assert np.isclose(float(jnp.sum(Pa)), var_a, rtol=1e-10)


def test_zonal_mean_shape():
    grid = _grid()
    f = jnp.asarray(np.random.default_rng(0).standard_normal((grid.n_lat, grid.n_lon, 4)))
    zm = D.zonal_mean(f)
    assert zm.shape == (grid.n_lat, 4)
    assert np.allclose(np.asarray(zm), np.asarray(f).mean(axis=1))
