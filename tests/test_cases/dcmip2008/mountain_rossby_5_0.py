"""DCMIP 2008 mountain-induced Rossby wave train (reference test 5-0-0).

Reference
---------
Jablonowski, C., P. H. Lauritzen, R. D. Nair and M. A. Taylor (2008):
"Idealized test cases for the dynamical cores of Atmospheric General
Circulation Models: A proposal for the NCAR ASP 2008 summer colloquium",
29 May 2008, Sect. 1.5 "Mountain-induced Rossby wave train", Eqs. (78)-(83).

Initial conditions (every formula and case parameter from the reference;
thermodynamic constants and Omega as noted below)
------------------------------------------------------------------
- u = u0 cos(phi), v = 0, u0 = 20 m/s, vertically uniform (Eqs. 78-79).
- T = T0 = 288 K, isothermal (text below Eq. 79); N^2 = g^2 / (c_p T0)
  (Eq. 80).
- Phi_s = g h0 exp(-(r/d)^2), h0 = 2000 m, d = 1500 km, r the great-circle
  distance to (lambda_c, phi_c) = (pi/2, pi/6), i.e. 90E / 30N
  (Eqs. 81-82): the shared :func:`gaussian_mountain`.
- p_s = p_sp exp[-(a N^2 u0 / (2 g^2 kappa)) (u0/a + 2 Omega)(sin^2 phi - 1)
  - N^2 Phi_s / (g^2 kappa)], p_sp = 930 hPa (South-Pole p_s),
  kappa = R_d/c_p (Eq. 83).  With Eq. (80) and kappa = R_d/c_p the
  terrain factor is exactly exp(-Phi_s / (R_d T0)), so the zonal factor
  :func:`eq83_zonal_surface_pressure` is passed as the ``p_s_init`` field
  of the shared isothermal builders in :mod:`._shared`, which apply that
  terrain factor.
- The reference quotes a horizontal-mean p_s of ~1001.456 hPa.

Constants are ``legoesm.constants`` (R_d = 287.05, c_pd = 1004.64)
rather than the reference's Appendix G (R_d = 287.04, Omega =
7.29211e-5); ``a`` and ``Omega`` are the grid's own (``constants.Omega``
where the grid carries none; the FV3 duo cells carry FV3_OMEGA).

Reference run setup vs the matrix row (reported, not changed here): the
reference shows the train over 15 days and recommends integrating to day
30, at ~1 deg on CAM L26 (top 2.194 hPa, Appendix C); the matrix row runs
10 days (2 with ``--quick``) on CAM6 L32.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.idealized.topography import gaussian_mountain

from tests.test_cases.dcmip2008._shared import (
    U0_DEFAULT,
    grid_rotation_rate,
    isothermal_state_cube,
    isothermal_state_latlon,
    isothermal_state_mpas,
    isothermal_state_spectral,
)


# --- DCMIP 2008 Sect. 1.5 parameters ---
T0_MR = 288.0                 # T0 [K], text below Eq. (79)
H0 = 2000.0                   # h0 [m], text below Eq. (81)
SIGMA_M = 1.5e6               # half width d [m], text below Eq. (81)
LON_C = jnp.pi / 2.0          # lambda_c = pi/2 (90E), text below Eq. (82)
LAT_C = jnp.pi / 6.0          # phi_c = pi/6 (30N), text below Eq. (82)
P_SP = 93000.0                # South-Pole p_s p_sp = 930 hPa, below Eq. (83)


def _phis(lon, lat, radius):
    """Eq. (81): Phi_s = g h0 exp(-(r/d)^2), r from Eq. (82)."""
    z_s = gaussian_mountain(
        lon, lat, radius,
        h_0=H0, lon_c=LON_C, lat_c=LAT_C, sigma=SIGMA_M,
    )
    return constants.g * z_s


def eq83_zonal_surface_pressure(lat, radius, *, omega: float,
                                T_0: float = T0_MR, u_0: float = U0_DEFAULT):
    """Eq. (83) without its terrain term [Pa]:
    p_sp exp[-(a N^2 u0 / (2 g^2 kappa)) (u0/a + 2 Omega)(sin^2 phi - 1)],
    N^2 = g^2 / (c_p T0) (Eq. 80).  ``omega`` is the run's rotation rate
    (the balance holds for the Omega the dycore integrates with)."""
    n2 = constants.g ** 2 / (constants.c_pd * T_0)
    coef = radius * n2 * u_0 / (2.0 * constants.g ** 2 * constants.kappa)
    return P_SP * jnp.exp(-coef * (u_0 / radius + 2.0 * omega)
                          * (jnp.sin(lat) ** 2 - 1.0))


def mountain_rossby_init(grid, sigma_coord, *, T_0: float = T0_MR, u_0: float = U0_DEFAULT):
    R = float(grid.radius)
    phis = _phis(grid.lon, grid.lat, R)
    return isothermal_state_cube(
        grid, sigma_coord, T_0=T_0, u_0=u_0, phis=phis,
        p_s_init=eq83_zonal_surface_pressure(grid.lat, R, omega=grid_rotation_rate(grid),
                                             T_0=T_0, u_0=u_0),
    )


def mountain_rossby_init_latlon(grid, sigma_coord, *, T_0: float = T0_MR, u_0: float = U0_DEFAULT):
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    R = float(grid.radius)
    lat_2d = jnp.broadcast_to(jnp.asarray(grid.lat)[:, None], (n_lat, n_lon))
    lon_2d = jnp.broadcast_to(jnp.asarray(grid.lon)[None, :], (n_lat, n_lon))
    phis = _phis(lon_2d, lat_2d, R)
    return isothermal_state_latlon(
        grid, sigma_coord, T_0=T_0, u_0=u_0, phis=phis,
        p_s_init=eq83_zonal_surface_pressure(lat_2d, R, omega=grid_rotation_rate(grid),
                                             T_0=T_0, u_0=u_0),
    )


def mountain_rossby_init_mpas(mesh, sigma_coord, *, T_0: float = T0_MR, u_0: float = U0_DEFAULT):
    R = float(mesh.radius)
    phis = _phis(mesh.lonCell, mesh.latCell, R)
    return isothermal_state_mpas(
        mesh, sigma_coord, T_0=T_0, u_0=u_0, phis=phis,
        p_s_init=eq83_zonal_surface_pressure(mesh.latCell, R, omega=grid_rotation_rate(mesh),
                                             T_0=T_0, u_0=u_0),
    )


def mountain_rossby_init_spectral(grid, sigma_coord, *, T_0: float = T0_MR, u_0: float = U0_DEFAULT):
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    R = float(grid.radius)
    lat_2d = jnp.broadcast_to(grid.lat[:, None], (n_lat, n_lon))
    lon_2d = jnp.broadcast_to(grid.lon2d, (n_lat, n_lon))
    phis = _phis(lon_2d, lat_2d, R)
    return isothermal_state_spectral(
        grid, sigma_coord, T_0=T_0, u_0=u_0, phis=phis,
        p_s_init=eq83_zonal_surface_pressure(lat_2d, R, omega=grid_rotation_rate(grid),
                                             T_0=T_0, u_0=u_0),
    )
