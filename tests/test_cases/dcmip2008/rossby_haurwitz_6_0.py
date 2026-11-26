"""DCMIP 2008 3D Rossby-Haurwitz wave (reference test 4-0-0, Sect. 1.4).

The matrix row keeps its historical name ``rossby_haurwitz_6_0``; in the
reference document this case is test **4-0-0** (its 6-x-0 family is the
gravity waves).

Reference
---------
Jablonowski, C., P. H. Lauritzen, R. D. Nair and M. A. Taylor (2008):
"Idealized test cases for the dynamical cores of Atmospheric General
Circulation Models: A proposal for the NCAR ASP 2008 summer colloquium",
29 May 2008, Sect. 1.4 "3D Rossby-Haurwitz wave", Eqs. (60)-(74).

Initial conditions (every formula and constant from the reference)
------------------------------------------------------------------
- Wave number n = 4; M = K = u0 / (n a) with u0 = 50 m/s, i.e.
  M = K ~ 1.962e-6 1/s (text below Eq. 60).  NOTE: this is NOT the
  shallow-water Williamson (1992) case-6 amplitude (omega = K =
  7.848e-6 1/s, ~100 m/s winds); the 3D test's peak wind is ~25 m/s.
  ``a`` is the grid's own radius, so the duo (GFS radius) and the other
  grids each get the reference's u0.
- Winds, vertically uniform (Eqs. 62-63) == the Williamson case-6 wind
  formulas with (omega, K) -> (M, K):
  :func:`legoesm.core.williamson_sw_analytic.rossby_haurwitz_4_winds`.
- Phi'(lambda, phi) = a^2 [A + B cos(n lambda) + C cos(2 n lambda)]
  (Eqs. 71-74) == the Williamson case-6 geopotential with gh0 = 0 and
  (omega, K) -> (M, K):
  :func:`legoesm.core.williamson_sw_analytic.rossby_haurwitz_4_geopotential`.
  (Eq. 72's last term ``-n^2 K^2/2 cos^{2(n-1)}`` is that function's
  ``K^2/4 cos^{2n} * (-2 n^2 cos^{-2})``.)
- p_s = p_ref (1 + Gamma Phi' / (g T0)) ** (g / (Gamma R_d))  (Eq. 70),
  p_ref = 955 hPa, T0 = 288 K, Gamma = 0.0065 K/m, Phi_s = 0.
- T(p) = T0 (p / p_ref) ** (Gamma R_d / g)  (Eq. 66), evaluated at the
  model's OWN full-level pressures p = A p0 + B p_s (Eq. 68) or the sigma
  equivalent (Eq. 67).
- Eq. 70 and Eq. 66 are the constant-lapse-rate columns of
  :func:`tests.test_cases.dcmip2012.rest_state_topography.lapse_rate_columns`
  with ``p_0 = p_ref`` and surface at the equivalent height (Eq. 65)
  ``z_s = -Phi'/g``; that helper is reused, not re-derived.

Balance: with T a function of p alone, the geopotential on every pressure
surface is ``Phi' + f(p)``, so the horizontal pressure-gradient force is
``-grad Phi'`` at every level -- the nonlinear-balance geopotential of the
barotropic winds; the initial divergence tendency vanishes analytically.

Constants are ``legoesm.constants`` (R_d = 287.05, Omega = 7.292e-5),
which differ slightly from the reference's Appendix G (R_d = 287.04,
Omega = 7.29211e-5): the area-mean p_s is 1000.3748 hPa here vs the
reference's quoted ~1000.377 hPa.

Reference run setup vs the matrix row (reported, not changed here): the
reference integrates 30 days at ~1 deg on the CAM L26 hybrid levels
(model top 2.194 hPa, Appendix C); the matrix row runs 14 days (2 with
``--quick``) on CAM6 L32.  Expected behaviour: westward propagation with
near-steady shape and amplitude.  Eq. (61) gives the BAROTROPIC
(nondivergent) benchmark: ~-15.0 deg/day with these constants (the
reference's prose quotes ~-15.2 deg/day, period ~24 days); the reference
notes it is only approximate for 3D hydrostatic flow.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.state import (
    HydrostaticState,
    MPASHydrostaticState,
)
from legoesm.core.williamson_sw_analytic import (
    RH4_WAVENUMBER,
    rossby_haurwitz_4_geopotential,
    rossby_haurwitz_4_winds,
)
from legoesm.grids.vertical import (
    HybridSigmaPressureCoordinate,
    SigmaCoordinate,
)
from tests.test_cases.dcmip2012.rest_state_topography import (
    lapse_rate_columns,
)


# --- DCMIP 2008 Sect. 1.4 parameters ---
RH_U0 = 50.0          # u0 [m/s], text below Eq. (60)
RH_T0 = 288.0         # T0 [K], text above Eq. (65)
RH_GAMMA = 0.0065     # lapse rate Gamma [K/m], text above Eq. (65)
RH_P_REF = 95500.0    # p_ref = 955 hPa [Pa], text below Eq. (65)


def rh_wave_frequency(radius: float) -> float:
    """``M = K = u0 / (n a)`` [1/s] (DCMIP 2008, text below Eq. 60)."""
    return RH_U0 / (RH4_WAVENUMBER * radius)


def _w6_winds_geo(lon, lat, radius):
    """``(u_east, v_north)`` of DCMIP 2008 Eqs. (62)-(63), M = K = u0/(n a).

    (Name kept for its callers; these are the Williamson case-6 FORMULAS at
    the DCMIP 2008 amplitude, not the case-6 amplitude.)"""
    mk = rh_wave_frequency(radius)
    return rossby_haurwitz_4_winds(lon, lat, radius=radius,
                                   omega_wave=mk, k_wave=mk, xp=jnp)


def rh_balanced_columns(lon, lat, radius, sigma_coord):
    """``(p_s, T)`` of DCMIP 2008 Eqs. (70)-(74) and (66).

    ``p_s`` has the broadcast shape of ``lon`` and ``lat``; ``T`` adds a
    trailing level axis on the coordinate's own full levels.  Phi_s = 0.
    """
    mk = rh_wave_frequency(radius)
    phi_prime = rossby_haurwitz_4_geopotential(
        lon, lat, radius=radius, omega=constants.Omega, gh0=0.0,
        omega_wave=mk, k_wave=mk, xp=jnp)
    return lapse_rate_columns(-phi_prime / constants.g, sigma_coord,
                              T_init=RH_T0, gamma=RH_GAMMA, p_0=RH_P_REF)


# ---------------------------------------------------------------------------
# Per-grid initialisations
# ---------------------------------------------------------------------------


def rossby_haurwitz_init(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
) -> HydrostaticState:
    """DCMIP 2008 4-0-0 on cubed-sphere."""
    from legoesm.grids.cubed_sphere import rotate_winds_geo_to_grid

    n = grid.n
    nlev = sigma_coord.n_levels
    R = float(grid.radius)
    shape_3d = (6, n, n, nlev)
    shape_2d = (6, n, n)

    u_east, v_north = _w6_winds_geo(grid.lon, grid.lat, R)
    u_grid_2d, v_grid_2d = rotate_winds_geo_to_grid(
        u_east, v_north, grid.angle,
    )
    u_3d = jnp.broadcast_to(u_grid_2d[..., None], shape_3d)
    v_3d = jnp.broadcast_to(v_grid_2d[..., None], shape_3d)

    p_s, T_3d = rh_balanced_columns(grid.lon, grid.lat, R, sigma_coord)
    phis = jnp.zeros(shape_2d)

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    return HydrostaticState(
        u=Field(data=u_3d, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_3d, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=phis, name="phis", dims=dims_2d, units="m^2/s^2"),
    )


def rossby_haurwitz_init_latlon(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
) -> HydrostaticState:
    """DCMIP 2008 4-0-0 on lat-lon."""
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = sigma_coord.n_levels
    R = float(grid.radius)

    lat_2d = jnp.asarray(grid.lat2d)
    lon_2d = jnp.asarray(grid.lon2d)
    u_east, v_north = _w6_winds_geo(lon_2d, lat_2d, R)
    u_3d = jnp.broadcast_to(u_east[..., None], (n_lat, n_lon, nlev))
    v_3d = jnp.broadcast_to(v_north[..., None], (n_lat, n_lon, nlev))

    p_s, T_3d = rh_balanced_columns(lon_2d, lat_2d, R, sigma_coord)
    phis = jnp.zeros((n_lat, n_lon))

    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    return HydrostaticState(
        u=Field(data=u_3d, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_3d, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=phis, name="phis", dims=dims_2d, units="m^2/s^2"),
    )


def rossby_haurwitz_init_mpas(
    mesh,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
) -> MPASHydrostaticState:
    """DCMIP 2008 4-0-0 on MPAS Voronoi mesh."""
    nCells = mesh.nCells
    nEdges = mesh.nEdges
    nlev = sigma_coord.n_levels
    R = float(mesh.radius)

    # Edge winds: project (u_east, v_north) at edges onto the edge normal.
    u_east, v_north = _w6_winds_geo(mesh.lonEdge, mesh.latEdge, R)
    u_normal = (u_east * jnp.cos(mesh.angleEdge)
                + v_north * jnp.sin(mesh.angleEdge))
    u_data = jnp.broadcast_to(u_normal[:, None], (nEdges, nlev))

    p_s, T_data = rh_balanced_columns(mesh.lonCell, mesh.latCell, R,
                                      sigma_coord)
    phis = jnp.zeros((nCells,))

    return MPASHydrostaticState(
        u=Field(data=u_data, name="u", dims=("nEdges", "level"), units="m/s"),
        T=Field(data=T_data, name="T", dims=("nCells", "level"), units="K"),
        p_s=Field(data=p_s, name="p_s", dims=("nCells",), units="Pa"),
        phis=Field(data=phis, name="phis", dims=("nCells",), units="m^2/s^2"),
    )


def rossby_haurwitz_init_spectral(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
):
    """DCMIP 2008 4-0-0 in spectral (Gaussian-grid) space."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralHydrostaticState
    from legoesm.grids.gaussian import (
        sh_analysis,
        sh_analysis_3d,
        sh_analysis_dmu_3d,
        sh_analysis_oc2_3d,
    )

    n_lat = grid.n_lat
    n_lon = grid.n_lon
    n_sh = grid.n_sh
    nlev = sigma_coord.n_levels
    R = float(grid.radius)

    lat_2d = jnp.broadcast_to(grid.lat[:, None], (n_lat, n_lon))
    lon_2d = jnp.broadcast_to(grid.lon2d, (n_lat, n_lon))
    u_east, v_north = _w6_winds_geo(lon_2d, lat_2d, R)

    u_grid = jnp.broadcast_to(u_east[..., None], (n_lat, n_lon, nlev)
                              ).astype(jnp.float64)
    v_grid = jnp.broadcast_to(v_north[..., None], (n_lat, n_lon, nlev)
                              ).astype(jnp.float64)

    a = R
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a
    cos_lat_3d = grid.cos_lat[:, None, None]
    u_cos = u_grid * cos_lat_3d
    v_cos = v_grid * cos_lat_3d

    vor_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, v_cos)
        + one_over_a * sh_analysis_dmu_3d(grid, u_cos)
    )
    div_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, v_cos)
    )

    p_s_grid, T_grid = rh_balanced_columns(lon_2d, lat_2d, R, sigma_coord)
    T_hat = sh_analysis_3d(grid, jnp.asarray(T_grid, dtype=jnp.float64))
    lnps_hat = sh_analysis(
        grid, jnp.log(jnp.asarray(p_s_grid, dtype=jnp.float64)))
    phis_hat = jnp.zeros(n_sh, dtype=jnp.complex128)

    dims_3d = ("spectral", "level")
    dims_2d = ("spectral",)
    return SpectralHydrostaticState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        T_hat=Field(data=T_hat, name="T_hat", dims=dims_3d, units="K"),
        lnps_hat=Field(data=lnps_hat, name="lnps_hat", dims=dims_2d, units=""),
        phis_hat=Field(
            data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2",
        ),
    )
