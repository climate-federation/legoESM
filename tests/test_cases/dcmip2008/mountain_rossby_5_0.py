"""DCMIP 2008 §5-0: mountain-induced Rossby wave.

A solid-body zonal flow over an isolated Gaussian mountain triggers a
stationary Rossby-wave train in a hydrostatic isothermal background
atmosphere.  This is the 3D primitive-equation analogue of Williamson
Test 5: the same kind of wave train should develop in the upper
troposphere over the days following day 0.

Initial conditions
------------------
- T(λ, φ, η) = T_0 = 288 K  (isothermal — slightly cooler than §3-1
  to match the DCMIP §5-0 spec)
- u(λ, φ, η) = u_0 cos(φ),  u_0 = 20 m/s
- v = 0
- φ_s(λ, φ) = g · h_0 · exp(-(d/σ_m)²)
  - h_0 = 2000 m, σ_m = 1500 km
  - d = great-circle distance from (λ_c, φ_c) = (3π/2, π/6)  (270 °E, 30 °N)
- p_s(λ, φ) = p_0 · exp(-φ_s / (R_d T_0))  (hydrostatic adjustment)

Reference
---------
DCMIP 2008 Test Case Document §5-0 (mountain-induced Rossby wave on
the rotating sphere) — see also Williamson 1992 Test 5 for the SW
analogue.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.idealized.topography import gaussian_mountain

from tests.test_cases.dcmip2008._shared import (
    P0_DEFAULT,
    U0_DEFAULT,
    isothermal_state_cube,
    isothermal_state_latlon,
    isothermal_state_mpas,
    isothermal_state_spectral,
)


# DCMIP §5-0 defaults
T0_MR = 288.0    # Background T [K]
H0 = 2000.0      # Mountain peak [m]
SIGMA_M = 1.5e6  # Gaussian e-folding width [m]
LON_C = 3.0 * jnp.pi / 2.0   # 270 °E
LAT_C = jnp.pi / 6.0          # 30 °N


def _phis(lon, lat, radius):
    z_s = gaussian_mountain(
        lon, lat, radius,
        h_0=H0, lon_c=LON_C, lat_c=LAT_C, sigma=SIGMA_M,
    )
    return constants.g * z_s


def mountain_rossby_init(grid, sigma_coord, *, T_0: float = T0_MR, u_0: float = U0_DEFAULT):
    phis = _phis(grid.lon, grid.lat, float(grid.radius))
    return isothermal_state_cube(
        grid, sigma_coord, T_0=T_0, u_0=u_0, phis=phis,
    )


def mountain_rossby_init_latlon(grid, sigma_coord, *, T_0: float = T0_MR, u_0: float = U0_DEFAULT):
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    lat_2d = jnp.broadcast_to(jnp.asarray(grid.lat)[:, None], (n_lat, n_lon))
    lon_2d = jnp.broadcast_to(jnp.asarray(grid.lon)[None, :], (n_lat, n_lon))
    phis = _phis(lon_2d, lat_2d, float(grid.radius))
    return isothermal_state_latlon(
        grid, sigma_coord, T_0=T_0, u_0=u_0, phis=phis,
    )


def mountain_rossby_init_mpas(mesh, sigma_coord, *, T_0: float = T0_MR, u_0: float = U0_DEFAULT):
    phis = _phis(mesh.lonCell, mesh.latCell, float(mesh.radius))
    return isothermal_state_mpas(
        mesh, sigma_coord, T_0=T_0, u_0=u_0, phis=phis,
    )


def mountain_rossby_init_spectral(grid, sigma_coord, *, T_0: float = T0_MR, u_0: float = U0_DEFAULT):
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    lat_2d = jnp.broadcast_to(grid.lat[:, None], (n_lat, n_lon))
    lon_2d = jnp.broadcast_to(grid.lon2d, (n_lat, n_lon))
    phis = _phis(lon_2d, lat_2d, float(grid.radius))
    return isothermal_state_spectral(
        grid, sigma_coord, T_0=T_0, u_0=u_0, phis=phis,
    )
