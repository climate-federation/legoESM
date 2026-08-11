"""DCMIP 2008 §6-0: small-amplitude Rossby-Haurwitz wave (3D extension).

A 3D extension of the shallow-water Williamson Test 6 (Rossby-Haurwitz
wave-4 mode).  The horizontal velocity field follows the analytic W6
formulas at every vertical level (barotropic / depth-independent), with
an isothermal hydrostatic background and uniform surface pressure.  The
wave should propagate eastward as a Rossby wave with negligible
amplitude growth.

Initial conditions
------------------
- T(λ, φ, η) = T_0 = 300 K   (isothermal)
- u(λ, φ, η) = u_W6(λ, φ),  v(λ, φ, η) = v_W6(λ, φ)
  where u_W6, v_W6 are the analytic R-H wave-4 surface winds (Williamson
  1992 Test 6).
- p_s = p_0,  φ_s = 0   (no topography)

Reference
---------
DCMIP 2008 Test Case Document §6-0 (3D Rossby-Haurwitz wave); see also
Williamson 1992 Test 6 for the SW analogue.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.state import (
    HydrostaticState,
    MPASHydrostaticState,
)
from legoesm.core.williamson_sw_analytic import rossby_haurwitz_4_winds
from legoesm.grids.vertical import (
    HybridSigmaPressureCoordinate,
    SigmaCoordinate,
)


_T0_RH = 300.0


def _w6_winds_geo(lon, lat, radius):
    """Williamson 1992 Eq. 146-147 surface winds.

    Thin adapter over the one shared analytic definition in
    :mod:`legoesm.core.williamson_sw_analytic` (this file used to carry
    its own copy)."""
    return rossby_haurwitz_4_winds(lon, lat, radius=radius, xp=jnp)


# ---------------------------------------------------------------------------
# Per-grid initialisations
# ---------------------------------------------------------------------------


def rossby_haurwitz_init(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    T_0: float = _T0_RH,
) -> HydrostaticState:
    """DCMIP §6-0 on cubed-sphere."""
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

    T_3d = jnp.full(shape_3d, T_0)
    p_s = jnp.full(shape_2d, constants.p_ref)
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
    *,
    T_0: float = _T0_RH,
) -> HydrostaticState:
    """DCMIP §6-0 on lat-lon."""
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = sigma_coord.n_levels
    R = float(grid.radius)

    lat_2d = jnp.asarray(grid.lat2d)
    lon_2d = jnp.asarray(grid.lon2d)
    u_east, v_north = _w6_winds_geo(lon_2d, lat_2d, R)
    u_3d = jnp.broadcast_to(u_east[..., None], (n_lat, n_lon, nlev))
    v_3d = jnp.broadcast_to(v_north[..., None], (n_lat, n_lon, nlev))

    T_3d = jnp.full((n_lat, n_lon, nlev), T_0)
    p_s = jnp.full((n_lat, n_lon), constants.p_ref)
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
    *,
    T_0: float = _T0_RH,
) -> MPASHydrostaticState:
    """DCMIP §6-0 on MPAS Voronoi mesh."""
    nCells = mesh.nCells
    nEdges = mesh.nEdges
    nlev = sigma_coord.n_levels
    R = float(mesh.radius)

    # Edge winds: project (u_east, v_north) at edges onto the edge normal.
    u_east, v_north = _w6_winds_geo(mesh.lonEdge, mesh.latEdge, R)
    u_normal = (u_east * jnp.cos(mesh.angleEdge)
                + v_north * jnp.sin(mesh.angleEdge))
    u_data = jnp.broadcast_to(u_normal[:, None], (nEdges, nlev))

    T_data = jnp.full((nCells, nlev), T_0)
    p_s = jnp.full((nCells,), constants.p_ref)
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
    *,
    T_0: float = _T0_RH,
):
    """DCMIP §6-0 in spectral (Gaussian-grid) space."""
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

    T_grid = jnp.full((n_lat, n_lon, nlev), T_0, dtype=jnp.float64)
    T_hat = sh_analysis_3d(grid, T_grid)
    lnps_grid = jnp.full(
        (n_lat, n_lon), jnp.log(constants.p_ref), dtype=jnp.float64,
    )
    lnps_hat = sh_analysis(grid, lnps_grid)
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
