"""DCMIP 2012 test 3-1: non-orographic gravity waves on a small planet.

Ullrich, P. A., Jablonowski, C., Kent, J., Lauritzen, P. H., Nair, R. D.,
Taylor, M. A. (2012): Dynamical Core Model Intercomparison Project
(DCMIP) Test Case Document, v1.7 (14 Jan 2013), Sect. 3 (Eqs. 86-111,
Table XIV) and App. F.3.

Planet of radius ``a = a_ref / X`` (X = 125), NOT rotating (Omega = 0),
background zonal flow ``u = u0 cos(lat)`` (u0 = 20 m/s) in gradient-wind
(here cyclostrophic) balance with a constant Brunt-Vaisala frequency
(N = 0.01 1/s) temperature field, and a potential-temperature bell

    Theta' = dTheta s(lon, lat) sin(2 pi z / Lz),   s = d^2 / (d^2 + r^2)

with d = 5 km, dTheta = 1 K, Lz = 20 km, centred at (2 pi/3, 0); ``r`` is
the great-circle distance on the small planet (Eq. 106).  The builders
here are the PRESSURE-based form for hydrostatic dycores: on the model's
own full-level pressures ``p_k = A_k p0 + B_k p_s`` the background T is
Eq. (92), the height Eq. (96) and the perturbation Eqs. (107), (109).
``p_s`` is Eq. (93), ``T_s`` Eq. (89).  ``p0 = p_eq = 1000 hPa``.

Thermodynamic constants are the model's shared ``legoesm.constants``
(``g``, ``R_d``, ``c_pd``, ``kappa``), so the state is balanced in the
dycore's own thermodynamics (vote 2026-11-26, author A / codex A).  The
lid pressure is therefore 273.831410 hPa; the document's 273.759 hPa is
the same Eq. (232) with its Table III R_d / c_p.

The grid / mesh arguments are STATIC geometry: radius and rotation are
validated on the host (numpy), so a builder may run under ``jax.jit`` with
the grid closed over, but not with grid arrays passed as traced arguments.

With Omega = 0 the balance (Eqs. 89, 93) is independent of the radius:
the cyclostrophic term ``u^2 tan(lat) / a`` and the meridional pressure
gradient ``(1/a) d/dlat`` both scale with ``1/a``.  ``radius`` enters
only through ``r`` (Eq. 106).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.idealized.topography import great_circle_distance
from legoesm.grids.vertical import (
    DCMIP2012_GW31_N as N_BV,
    DCMIP2012_GW31_PEQ as P_EQ,
    DCMIP2012_GW31_TEQ as T_EQ,
    HybridSigmaPressureCoordinate,
    dcmip2012_gw31_isothermal_g,
    pressure_from_hybrid,
)

from tests.test_cases.dcmip2008._shared import (
    isothermal_state_cube,
    isothermal_state_latlon,
    isothermal_state_mpas,
    isothermal_state_spectral,
)

# --- Table XIV (DCMIP 2012 test 3-x) ---------------------------------------
X_SCALE = 125.0                 # reduced-planet factor X
OMEGA = 0.0                     # [1/s] non-rotating
U0 = 20.0                       # [m/s]
P0 = P_EQ                       # [Pa] reference pressure of Theta (Eq. 97)
D_WIDTH = 5000.0                # [m] width d of the Theta' bell
LON_C = 2.0 * math.pi / 3.0     # [rad] 120 E
LAT_C = 0.0                     # [rad]
DELTA_THETA = 1.0               # [K]
L_Z = 20000.0                   # [m]
DURATION_S = 3600.0             # [s] simulation length (Sect. 3, "Grid
                                # spacings, simulation time and output")


def small_planet_radius() -> float:
    """``a = a_ref / X`` [m] with the shared ``constants.R_earth`` (vote
    2026-11-26: one radius on all four grids, author A / codex A)."""
    from legoesm.atmosphere.idealized.small_planet import scaled_radius
    return scaled_radius(X_SCALE)


def surface_temperature(lat):
    """Eq. (89) with Omega = 0: ``T_s = G + (T_eq - G) exp(-u0^2 N^2 /
    (4 g^2) (cos 2 lat - 1))`` [K]."""
    G = dcmip2012_gw31_isothermal_g()
    return G + (T_EQ - G) * jnp.exp(
        -(U0 * U0 * N_BV ** 2) / (4.0 * constants.g ** 2)
        * (jnp.cos(2.0 * lat) - 1.0))


def surface_pressure(lat):
    """Eq. (93) with Omega = 0: ``p_s = p_eq exp(u0^2/(4 G R_d) (cos 2 lat
    - 1)) (T_s/T_eq)^(1/kappa)`` [Pa]."""
    G = dcmip2012_gw31_isothermal_g()
    return P_EQ * jnp.exp(U0 * U0 / (4.0 * G * constants.R_d)
                          * (jnp.cos(2.0 * lat) - 1.0)) * (
        surface_temperature(lat) / T_EQ) ** (1.0 / constants.kappa)


def _column_ratio(p, T_s, p_s):
    """``T_s/G ((p/p_s)^kappa - 1) + 1`` -- the common factor of Eqs.
    (92), (96), (98)."""
    G = dcmip2012_gw31_isothermal_g()
    return T_s[..., None] / G * (
        (p / p_s[..., None]) ** constants.kappa - 1.0) + 1.0


def background_temperature(p, T_s, p_s):
    """Eq. (92): T_b at pressure ``p`` (..., nlev) [K]."""
    return (T_s[..., None] * (p / p_s[..., None]) ** constants.kappa
            / _column_ratio(p, T_s, p_s))


def height_from_pressure(p, T_s, p_s):
    """Eq. (96): z at pressure ``p`` (..., nlev) [m]."""
    return -constants.g / N_BV ** 2 * jnp.log(_column_ratio(p, T_s, p_s))


def shape_function(lon, lat, radius: float):
    """Eqs. (105)-(106): ``s = d^2 / (d^2 + r^2)``, r the great-circle
    distance [m] on the sphere of ``radius`` from (LON_C, LAT_C)."""
    r = great_circle_distance(lon, lat, LON_C, LAT_C, radius)
    return D_WIDTH ** 2 / (D_WIDTH ** 2 + r ** 2)


def theta_perturbation(lon, lat, p, T_s, p_s, radius: float):
    """Eq. (107): Theta' at pressure ``p`` (..., nlev) [K]."""
    z = height_from_pressure(p, T_s, p_s)
    return (DELTA_THETA * shape_function(lon, lat, radius)[..., None]
            * jnp.sin(2.0 * jnp.pi * z / L_Z))


def columns(lon, lat, coord: HybridSigmaPressureCoordinate, radius: float):
    """``(p_s (...), T (..., nlev))`` of test 3-1 on ``coord``'s own full
    levels: T = Eq. (92) + Eq. (109) at ``p_k = A_k p0 + B_k p_s``."""
    if not isinstance(coord, HybridSigmaPressureCoordinate):
        raise TypeError("DCMIP 2012 3-1 is built on the App. F.3 hybrid "
                        f"levels; got {type(coord).__name__}")
    lat = jnp.asarray(lat, dtype=jnp.float64)
    lon = jnp.asarray(lon, dtype=jnp.float64)
    T_s = surface_temperature(lat)
    p_s = surface_pressure(lat)
    p = pressure_from_hybrid(coord, p_s, full=True)
    T = (background_temperature(p, T_s, p_s)
         + theta_perturbation(lon, lat, p, T_s, p_s, radius)
         * (p / P0) ** constants.kappa)
    return p_s, T


def _check_grid(grid) -> None:
    """Refuse a grid that is not the non-rotating a/X planet.  The rotation
    rate is the grid's ``omega``; a grid without one (the MPAS mesh) must
    carry Coriolis fields (``fEdge``/``fVertex``) that are identically 0."""
    radius = float(grid.radius)
    if not math.isclose(radius, small_planet_radius(), rel_tol=1e-12):
        raise ValueError(f"DCMIP 2012 3-1 runs on the a/X = "
                         f"{small_planet_radius():.6g} m planet; grid radius "
                         f"is {radius:.6g} m")
    omega = getattr(grid, "omega", None)
    if omega is not None:
        rotating = float(omega) != OMEGA
    else:
        # host-side (numpy) so a builder traced under jit still validates
        if any(getattr(grid, k, None) is None for k in ("fEdge", "fVertex")):
            raise ValueError("DCMIP 2012 3-1: a grid without omega must "
                             "carry both fEdge and fVertex; cannot verify "
                             "Omega = 0")
        rotating = any(bool(np.any(np.asarray(getattr(grid, k)) != 0.0))
                       for k in ("fEdge", "fVertex"))
    if rotating:
        raise ValueError("DCMIP 2012 3-1 is non-rotating; the grid's "
                         "Coriolis parameter is not zero")


# ---------------------------------------------------------------------------
# Per-grid builders (winds/packaging: the shared isothermal-state builders,
# whose T is then replaced; their p_s = p_s_init when phis is flat)
# ---------------------------------------------------------------------------


def gravity_wave_init(grid, coord):
    """Cubed sphere (``grid.lon``/``grid.lat`` (6, n, n))."""
    _check_grid(grid)
    p_s, T = columns(grid.lon, grid.lat, coord, float(grid.radius))
    base = isothermal_state_cube(grid, coord, T_0=T_EQ, u_0=U0, p_s_init=p_s)
    return base._replace(T=base.T.replace(data=T))


def gravity_wave_init_latlon(grid, coord):
    """Lat-lon grid."""
    _check_grid(grid)
    lat = jnp.broadcast_to(jnp.asarray(grid.lat)[:, None],
                           (grid.n_lat, grid.n_lon))
    lon = jnp.broadcast_to(jnp.asarray(grid.lon)[None, :],
                           (grid.n_lat, grid.n_lon))
    p_s, T = columns(lon, lat, coord, float(grid.radius))
    base = isothermal_state_latlon(grid, coord, T_0=T_EQ, u_0=U0,
                                   p_s_init=p_s)
    return base._replace(T=base.T.replace(data=T))


def gravity_wave_init_mpas(mesh, coord):
    """MPAS Voronoi mesh (T at cells, normal wind at edges)."""
    _check_grid(mesh)
    p_s, T = columns(mesh.lonCell, mesh.latCell, coord, float(mesh.radius))
    base = isothermal_state_mpas(mesh, coord, T_0=T_EQ, u_0=U0, p_s_init=p_s)
    return base._replace(T=base.T.replace(data=T))


def gravity_wave_init_spectral(grid, coord):
    """Spectral (Gaussian-grid) state."""
    from legoesm.grids.gaussian import sh_analysis_3d
    _check_grid(grid)
    lat = jnp.broadcast_to(grid.lat[:, None], (grid.n_lat, grid.n_lon))
    lon = jnp.broadcast_to(grid.lon2d, (grid.n_lat, grid.n_lon))
    p_s, T = columns(lon, lat, coord, float(grid.radius))
    base = isothermal_state_spectral(grid, coord, T_0=T_EQ, u_0=U0,
                                     p_s_init=p_s)
    return base._replace(T_hat=base.T_hat.replace(
        data=sh_analysis_3d(grid, T.astype(jnp.float64))))
