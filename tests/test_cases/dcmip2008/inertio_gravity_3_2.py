"""DCMIP 2008 §3-2: inertio-gravity wave on a rotating planet.

Same isothermal-with-θ-perturbation setup as :mod:`gravity_wave_3_1`,
but on the standard rotating Earth (Ω = 7.292·10⁻⁵ rad/s).  The Coriolis
parameter is non-zero everywhere outside the equator, so the wave packet
acquires inertial character (longer-period oscillation envelopes).

DCMIP 2008 §3-2 uses a slightly larger horizontal scale (R_p = a) and a
slightly smaller perturbation amplitude than §3-1; we mirror that here.
The thin-perturbation centre stays at the equator since the inertial
contribution is most distinct against a non-zero f.

Reference
---------
DCMIP 2008 Test Case Document, §3-2 (Skamarock-Klemp gravity-wave
test, rotating-planet variant).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.idealized.topography import great_circle_distance

from tests.test_cases.dcmip2008._shared import (
    P0_DEFAULT,
    T0_DEFAULT,
    U0_DEFAULT,
    height_isothermal,
    hydrostatic_pressure,
    isothermal_state_cube,
    isothermal_state_latlon,
    isothermal_state_mpas,
    isothermal_state_spectral,
)


# ---------------------------------------------------------------------------
# Perturbation parameters (§3-2 defaults; smaller amplitude, larger scale)
# ---------------------------------------------------------------------------

DELTA_THETA = 0.5    # Perturbation amplitude [K]
L_Z = 20000.0        # Vertical wavelength [m]
LON_C = 0.0
LAT_C = 0.0


def _theta_perturbation_factors(lon, lat, z, radius, *, R_p_frac=1.0):
    """Δθ(λ, φ, z) for §3-2 — broader horizontal envelope (R_p = a)."""
    R_p = R_p_frac * radius
    d = great_circle_distance(lon, lat, LON_C, LAT_C, radius)
    horizontal = R_p ** 2 / (d ** 2 + R_p ** 2)
    vertical = jnp.sin(jnp.pi * z / L_Z)
    return DELTA_THETA * horizontal[..., None] * vertical


def _delta_T_from_theta(delta_theta, p, p_0=P0_DEFAULT):
    return delta_theta * (p / p_0) ** constants.kappa


# ---------------------------------------------------------------------------
# Per-grid initialisations (mirror gravity_wave_3_1 with §3-2 perturbation)
# ---------------------------------------------------------------------------


def inertio_gravity_init(
    grid,
    sigma_coord,
    *,
    T_0: float = T0_DEFAULT,
    u_0: float = U0_DEFAULT,
):
    base = isothermal_state_cube(grid, sigma_coord, T_0=T_0, u_0=u_0)
    p = hydrostatic_pressure(sigma_coord, base.p_s.data)
    z = height_isothermal(sigma_coord, base.p_s.data, T_0)
    delta_theta = _theta_perturbation_factors(
        grid.lon, grid.lat, z, float(grid.radius),
    )
    delta_T = _delta_T_from_theta(delta_theta, p)
    return base._replace(T=base.T.replace(data=base.T.data + delta_T))


def inertio_gravity_init_latlon(
    grid,
    sigma_coord,
    *,
    T_0: float = T0_DEFAULT,
    u_0: float = U0_DEFAULT,
):
    base = isothermal_state_latlon(grid, sigma_coord, T_0=T_0, u_0=u_0)
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    lat_2d = jnp.broadcast_to(jnp.asarray(grid.lat)[:, None], (n_lat, n_lon))
    lon_2d = jnp.broadcast_to(jnp.asarray(grid.lon)[None, :], (n_lat, n_lon))
    p = hydrostatic_pressure(sigma_coord, base.p_s.data)
    z = height_isothermal(sigma_coord, base.p_s.data, T_0)
    delta_theta = _theta_perturbation_factors(
        lon_2d, lat_2d, z, float(grid.radius),
    )
    delta_T = _delta_T_from_theta(delta_theta, p)
    return base._replace(T=base.T.replace(data=base.T.data + delta_T))


def inertio_gravity_init_mpas(
    mesh,
    sigma_coord,
    *,
    T_0: float = T0_DEFAULT,
    u_0: float = U0_DEFAULT,
):
    base = isothermal_state_mpas(mesh, sigma_coord, T_0=T_0, u_0=u_0)
    p = hydrostatic_pressure(sigma_coord, base.p_s.data)
    z = height_isothermal(sigma_coord, base.p_s.data, T_0)
    delta_theta = _theta_perturbation_factors(
        mesh.lonCell, mesh.latCell, z, float(mesh.radius),
    )
    delta_T = _delta_T_from_theta(delta_theta, p)
    return base._replace(T=base.T.replace(data=base.T.data + delta_T))


def inertio_gravity_init_spectral(
    grid,
    sigma_coord,
    *,
    T_0: float = T0_DEFAULT,
    u_0: float = U0_DEFAULT,
):
    from legoesm.grids.gaussian import sh_analysis_3d

    n_lat = grid.n_lat
    n_lon = grid.n_lon
    base = isothermal_state_spectral(grid, sigma_coord, T_0=T_0, u_0=u_0)
    lat_2d = jnp.broadcast_to(grid.lat[:, None], (n_lat, n_lon))
    lon_2d = jnp.broadcast_to(grid.lon2d, (n_lat, n_lon))
    p_s_grid = jnp.full((n_lat, n_lon), P0_DEFAULT, dtype=jnp.float64)
    p = hydrostatic_pressure(sigma_coord, p_s_grid)
    z = height_isothermal(sigma_coord, p_s_grid, T_0)
    delta_theta = _theta_perturbation_factors(
        lon_2d, lat_2d, z, float(grid.radius),
    )
    delta_T = _delta_T_from_theta(delta_theta, p).astype(jnp.float64)
    delta_T_hat = sh_analysis_3d(grid, delta_T)
    return base._replace(T_hat=base.T_hat.replace(
        data=base.T_hat.data + delta_T_hat,
    ))
