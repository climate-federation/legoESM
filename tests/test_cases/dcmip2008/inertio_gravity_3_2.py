"""DCMIP 2008 inertio-gravity wave on the rotating Earth (test 6-3-0).

The matrix row keeps its historical name ``inertio_gravity_3_2``; in the
reference document this case is test **6-3-0** (Table 1; there is no
"3-2").

Reference
---------
Jablonowski, C., P. H. Lauritzen, R. D. Nair and M. A. Taylor (2008):
"Idealized test cases for the dynamical cores of Atmospheric General
Circulation Models: A proposal for the NCAR ASP 2008 summer colloquium",
29 May 2008, Sect. 1.6 "Gravity waves with and without the Earth's
rotation", variant [6-3-0].

Initial conditions ([6-3-0] parameters, text below Eq. 112)
------------------------------------------------------------
- Earth's rotation (the grid's own Omega), u = v = 0 (u0 = 0, Eq. 86
  with u0 = 0), Phi_s = 0, p_s = p_eq = p0 = 1000 hPa (Eq. 87 with u0 = 0).
- Isothermal background T0 = 300 K, N^2 = g^2 / (c_p T0) (Eq. 90); the
  isothermal height z(p) = (R_d T0 / g) ln(p_s / p) (Eq. 92).
- Theta = Theta_bar + Delta_Theta s(lambda, phi) sin(2 pi z / L_z)
  (Eqs. 95-96), Delta_Theta = 10 K, L_z = 20 km; T = Theta (p/p0)^kappa
  (Eq. 94), so the perturbation adds Delta_Theta s sin(2 pi z/L_z)
  (p/p0)^kappa to the isothermal T.
- s = 0.5 (1 + cos(pi r / R)) for r < R, 0 otherwise, R = a/3 (Eq. 97),
  r the great-circle distance (Eq. 98) to (lambda_c, phi_c) = (pi, pi/4).

Reference run setup vs the matrix row (reported, not changed here): the
reference runs 15 days on 20 levels (Delta z ~ 500 m) under a 10 km lid
(z_top = L_z / 2, p_top ~ 320.2 hPa isothermal, Sect. 1.6.1); the matrix
row runs 3 days on CAM6 L32 (lid ~3.6 hPa), so the sin(2 pi z / L_z)
perturbation continues above 10 km, where the reference has no levels.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.idealized.topography import great_circle_distance

from tests.test_cases.dcmip2008._shared import (
    P0_DEFAULT,
    T0_DEFAULT,
    height_isothermal,
    hydrostatic_pressure,
    isothermal_state_cube,
    isothermal_state_latlon,
    isothermal_state_mpas,
    isothermal_state_spectral,
)


# --- DCMIP 2008 Sect. 1.6, variant [6-3-0] ---
DELTA_THETA = 10.0           # Delta_Theta [K], text below Eq. (98)
L_Z = 20000.0                # vertical wavelength L_z [m], below Eq. (98)
LON_C = jnp.pi               # lambda_c, [6-3-0] parameters (below Eq. 112)
LAT_C = jnp.pi / 4.0         # phi_c = pi/4
U0_IG = 0.0                  # u0 = 0 m/s for [6-3-0]


def _theta_perturbation_factors(lon, lat, z, radius):
    """Delta_Theta s(lambda, phi) sin(2 pi z / L_z) (Eqs. 95-98)."""
    R = radius / 3.0
    r = great_circle_distance(lon, lat, LON_C, LAT_C, radius)
    s = jnp.where(r < R, 0.5 * (1.0 + jnp.cos(jnp.pi * r / R)), 0.0)
    return DELTA_THETA * s[..., None] * jnp.sin(2.0 * jnp.pi * z / L_Z)


def _delta_T_from_theta(delta_theta, p, p_0=P0_DEFAULT):
    """Eq. (94): T' = Theta' (p / p0)^kappa."""
    return delta_theta * (p / p_0) ** constants.kappa


# ---------------------------------------------------------------------------
# Per-grid initialisations
# ---------------------------------------------------------------------------


def inertio_gravity_init(grid, sigma_coord, *, T_0: float = T0_DEFAULT):
    base = isothermal_state_cube(grid, sigma_coord, T_0=T_0, u_0=U0_IG)
    p = hydrostatic_pressure(sigma_coord, base.p_s.data)
    z = height_isothermal(sigma_coord, base.p_s.data, T_0)
    delta_theta = _theta_perturbation_factors(
        grid.lon, grid.lat, z, float(grid.radius),
    )
    delta_T = _delta_T_from_theta(delta_theta, p)
    return base._replace(T=base.T.replace(data=base.T.data + delta_T))


def inertio_gravity_init_latlon(grid, sigma_coord, *, T_0: float = T0_DEFAULT):
    base = isothermal_state_latlon(grid, sigma_coord, T_0=T_0, u_0=U0_IG)
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


def inertio_gravity_init_mpas(mesh, sigma_coord, *, T_0: float = T0_DEFAULT):
    base = isothermal_state_mpas(mesh, sigma_coord, T_0=T_0, u_0=U0_IG)
    p = hydrostatic_pressure(sigma_coord, base.p_s.data)
    z = height_isothermal(sigma_coord, base.p_s.data, T_0)
    delta_theta = _theta_perturbation_factors(
        mesh.lonCell, mesh.latCell, z, float(mesh.radius),
    )
    delta_T = _delta_T_from_theta(delta_theta, p)
    return base._replace(T=base.T.replace(data=base.T.data + delta_T))


def inertio_gravity_init_spectral(grid, sigma_coord, *, T_0: float = T0_DEFAULT):
    from legoesm.grids.gaussian import sh_analysis_3d

    n_lat = grid.n_lat
    n_lon = grid.n_lon
    base = isothermal_state_spectral(grid, sigma_coord, T_0=T_0, u_0=U0_IG)
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
