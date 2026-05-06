"""DCMIP 2008 §3-1: gravity wave on a non-rotating sphere.

A localised potential-temperature perturbation in an otherwise
isothermal hydrostatic background atmosphere triggers an inertia-free
gravity-wave packet.  The canonical setup uses a small-radius sphere
(scaling factor X=125) and turns rotation off — the matrix-script
runner constructs the grid with ``omega=0`` so the Coriolis term
vanishes everywhere (see ``run_baroclinic`` in
``scripts/run_atmosphere_test_matrix.py``).  We keep the standard
Earth radius for now; the small-planet (X=125) variant is left to a
follow-up that wires :mod:`legoesm.atmosphere.idealized.small_planet`
into the runner.

Initial conditions (Skamarock-Klemp 2008 / DCMIP 2008 §3-1)
-----------------------------------------------------------
- T(λ, φ, η) = T_0 = 300 K              (isothermal)
- u(λ, φ, η) = u_0 cos(φ),  u_0 = 20 m/s
- v = 0
- p_s = p_0 = 10⁵ Pa,  φ_s = 0
- Δθ(λ, φ, z) = δθ · sin(π z / L_z) · R_p² / (d² + R_p²)
  - δθ = 1 K, L_z = 10 km, R_p = a/3
  - d = great-circle distance from (λ_c, φ_c) = (0, 0)
- ΔT(λ, φ, z) = Δθ · (p/p_0)^κ   (Exner conversion)

References
----------
Skamarock, W. C., & Klemp, J. B. (2008). A time-split nonhydrostatic
atmospheric model for weather research and forecasting applications.
*J. Comput. Phys.*, 227(7), 3465-3485 — gravity-wave dispersion target.
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
# Perturbation parameters (DCMIP 2008 §3-1 defaults)
# ---------------------------------------------------------------------------

DELTA_THETA = 1.0      # Perturbation amplitude [K]
L_Z = 10000.0          # Vertical wavelength [m]
LON_C = 0.0            # Centre longitude [rad]
LAT_C = 0.0            # Centre latitude (equator) [rad]


def _theta_perturbation_factors(lon, lat, z, radius, *, R_p_frac=1.0 / 3.0):
    """Compute Δθ(λ, φ, z) on grid-space arrays."""
    R_p = R_p_frac * radius
    d = great_circle_distance(lon, lat, LON_C, LAT_C, radius)
    horizontal = R_p ** 2 / (d ** 2 + R_p ** 2)
    vertical = jnp.sin(jnp.pi * z / L_Z)
    return DELTA_THETA * horizontal[..., None] * vertical


def _delta_T_from_theta(delta_theta, p, p_0=P0_DEFAULT):
    """Convert Δθ to ΔT using the Exner factor (p/p_0)^κ."""
    return delta_theta * (p / p_0) ** constants.kappa


# ---------------------------------------------------------------------------
# Per-grid initialisations
# ---------------------------------------------------------------------------


def gravity_wave_init(
    grid,
    sigma_coord,
    *,
    T_0: float = T0_DEFAULT,
    u_0: float = U0_DEFAULT,
):
    """DCMIP 2008 §3-1 on a cubed-sphere grid."""
    base = isothermal_state_cube(grid, sigma_coord, T_0=T_0, u_0=u_0)
    p = hydrostatic_pressure(sigma_coord, base.p_s.data)
    z = height_isothermal(sigma_coord, base.p_s.data, T_0)
    delta_theta = _theta_perturbation_factors(
        grid.lon, grid.lat, z, float(grid.radius),
    )
    delta_T = _delta_T_from_theta(delta_theta, p)
    return base._replace(T=base.T.replace(data=base.T.data + delta_T))


def gravity_wave_init_latlon(
    grid,
    sigma_coord,
    *,
    T_0: float = T0_DEFAULT,
    u_0: float = U0_DEFAULT,
):
    """DCMIP 2008 §3-1 on a lat-lon grid."""
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


def gravity_wave_init_mpas(
    mesh,
    sigma_coord,
    *,
    T_0: float = T0_DEFAULT,
    u_0: float = U0_DEFAULT,
):
    """DCMIP 2008 §3-1 on an MPAS Voronoi mesh (cells only carry T)."""
    base = isothermal_state_mpas(mesh, sigma_coord, T_0=T_0, u_0=u_0)
    p = hydrostatic_pressure(sigma_coord, base.p_s.data)
    z = height_isothermal(sigma_coord, base.p_s.data, T_0)
    delta_theta = _theta_perturbation_factors(
        mesh.lonCell, mesh.latCell, z, float(mesh.radius),
    )
    delta_T = _delta_T_from_theta(delta_theta, p)
    return base._replace(T=base.T.replace(data=base.T.data + delta_T))


def gravity_wave_init_spectral(
    grid,
    sigma_coord,
    *,
    T_0: float = T0_DEFAULT,
    u_0: float = U0_DEFAULT,
):
    """DCMIP 2008 §3-1 in spectral (Gaussian-grid) space."""
    from legoesm.grids.gaussian import sh_analysis_3d

    n_lat = grid.n_lat
    n_lon = grid.n_lon

    # Build the perturbation in grid space first, then transform.
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
