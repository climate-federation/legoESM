"""GM/Redi shared-density hoist (scaling review 2026-06-13 lever #3).

When ``implicit_K33=True`` (the Veros/ACC production recipe) the lat-lon
ocean step calls BOTH ``gm_redi_tracer_tendency_latlon`` and
``compute_isoneutral_K33_latlon`` per step with the SAME
``(T, S, eta, H_bathy)`` — each was independently running the expensive
2-iteration EOS coupling + Jacobian.  ``gm_redi_density_and_jacobian``
computes them once; the ``density_jacobian`` argument threads the result
into both.

Pins: passing the precomputed ``(rho, jacobian)`` is BIT-IDENTICAL to the
inline path (the hoist is a pure dedup, never a numerics change), for both
the tracer tendency and the K_33 diagonal, across slope schemes.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star, compute_ocean_jacobian
from legoesm.ocean.eos import make_eos_fn
from legoesm.ocean.dynamics.ocean_tendency_common import (
    iterate_eos_and_pressure_anomaly,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import neumann_fill_cgrid
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    gm_redi_density_and_jacobian,
    gm_redi_tracer_tendency_latlon,
    compute_isoneutral_K33_latlon,
)


def _legacy_density_jacobian(T, S, eta, H_bathy, grid, z_coord, mask):
    """The EXACT pre-hoist inline sequence (independent oracle): jacobian
    then 2-iteration EOS coupling.  Decouples the parity tests from the
    helper so a drift in gm_redi_density_and_jacobian away from this
    historical sequence is CAUGHT rather than masked (codex 2026-06-13)."""
    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
    eos_fn = make_eos_fn("wright", None)
    fill_fn = lambda field: neumann_fill_cgrid(field, mask)
    rho, _rho_prime, _p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask, fill_fn, eos_fn, z_coord.dz_ref,
        constants.rho_ocean, constants.g, n_iter=2,
    )
    return rho, jacobian


def _setup(n_lat=10, n_lon=20, nlev=6, H_max=4000.0, slope=1.0e-4):
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(
        n_levels=nlev, H_max=H_max, dz_surface=100.0, dz_deep=1500.0,
    )
    mask = jnp.ones((n_lat, n_lon))
    u_mask = jnp.ones((n_lat, n_lon + 1))
    v_mask = jnp.ones((n_lat + 1, n_lon))
    eta = jnp.zeros((n_lat, n_lon))
    H_bathy = jnp.full((n_lat, n_lon), H_max)

    # Stable stratification + meridional tilt (so slopes/K33 are nonzero).
    rho_z = jnp.linspace(constants.rho_ocean, 1027.0, nlev)
    H_total = float(jnp.sum(z_coord.dz_ref))
    drho_dz = 2.0 / H_total
    drho_dy = slope * drho_dz
    lat_idx = jnp.arange(n_lat, dtype=jnp.float64)
    dy_ref = float(grid.dy[0])
    rho = (rho_z[None, None, :]
           + drho_dy * lat_idx[:, None, None] * dy_ref
           + jnp.zeros((n_lat, n_lon, nlev)))
    T = (constants.rho_ocean - rho) / 0.2
    S = jnp.full((n_lat, n_lon, nlev), 35.0, dtype=jnp.float64)
    return grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, T, S


def test_density_and_jacobian_matches_legacy_oracle():
    """The helper's (rho, jacobian) are BIT-IDENTICAL to the exact pre-hoist
    inline sequence (independent oracle) — proves the hoist did not perturb
    the historical compute_ocean_jacobian -> make_eos_fn ->
    iterate_eos_and_pressure_anomaly(n_iter=2) numerics."""
    grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, T, S = _setup()
    rho, jac = gm_redi_density_and_jacobian(
        T, S, eta, H_bathy, grid, z_coord,
        eos="wright", mask=mask,
        rho_0=constants.rho_ocean, g=constants.g,
    )
    rho_legacy, jac_legacy = _legacy_density_jacobian(
        T, S, eta, H_bathy, grid, z_coord, mask)
    assert np.array_equal(np.asarray(rho), np.asarray(rho_legacy)), \
        "helper rho drifted from the legacy inline sequence"
    assert np.array_equal(np.asarray(jac), np.asarray(jac_legacy)), \
        "helper jacobian drifted from the legacy inline sequence"
    # density strictly increases with depth (stable strat) on this column
    assert float(rho[0, 0, -1]) > float(rho[0, 0, 0])


@pytest.mark.parametrize("slope_scheme", ["triads", "centered"])
def test_tracer_tendency_hoist_bit_identical(slope_scheme):
    grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, T, S = _setup()
    cfg = GMRediConfig(kappa_GM=1000.0, kappa_Redi=1000.0, S_max=5.0e-3,
                       slope_scheme=slope_scheme)
    kw = dict(eos="wright", mask=mask, u_mask=u_mask, v_mask=v_mask,
              rho_0=constants.rho_ocean, g=constants.g)

    dT0, dS0 = gm_redi_tracer_tendency_latlon(
        T, S, eta, H_bathy, grid, z_coord, cfg, **kw)
    # Feed the LEGACY oracle tuple (not the helper) so this also catches a
    # helper drift: None-branch (helper) must equal the legacy-fed branch.
    dj = _legacy_density_jacobian(T, S, eta, H_bathy, grid, z_coord, mask)
    dT1, dS1 = gm_redi_tracer_tendency_latlon(
        T, S, eta, H_bathy, grid, z_coord, cfg, density_jacobian=dj, **kw)

    assert np.array_equal(np.asarray(dT0), np.asarray(dT1)), "dT not bit-identical"
    assert np.array_equal(np.asarray(dS0), np.asarray(dS1)), "dS not bit-identical"


def test_k33_hoist_bit_identical():
    grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, T, S = _setup()
    cfg = GMRediConfig(kappa_GM=1000.0, kappa_Redi=1000.0, S_max=5.0e-3,
                       implicit_K33=True)
    kw = dict(eos="wright", mask=mask,
              rho_0=constants.rho_ocean, g=constants.g)

    k0 = compute_isoneutral_K33_latlon(
        T, S, eta, H_bathy, grid, z_coord, cfg, **kw)
    # Legacy oracle tuple (not the helper) — also catches helper drift.
    dj = _legacy_density_jacobian(T, S, eta, H_bathy, grid, z_coord, mask)
    k1 = compute_isoneutral_K33_latlon(
        T, S, eta, H_bathy, grid, z_coord, cfg, density_jacobian=dj, **kw)

    assert np.array_equal(np.asarray(k0), np.asarray(k1)), "K33 not bit-identical"
    assert jnp.all(k1 >= 0.0), "K33 must be non-negative"
