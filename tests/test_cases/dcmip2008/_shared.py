"""Shared helpers for DCMIP-2008 dry 3D test cases.

The four DCMIP-2008 §3-x / §5-0 / §6-0 tests (gravity wave on
non-rotating Earth, inertio-gravity wave on rotating planet,
mountain-induced Rossby wave, small-amplitude Rossby-Haurwitz)
all start from an *isothermal hydrostatic background with zonal flow*
that this module constructs once for every grid type.  Each test
module then adds its own perturbation (θ' bubble, mountain, R-H
wave structure) on top of the returned base state.

By factoring the shared piece here we avoid duplicating ~150 LOC
of grid-specific state assembly across four IC modules.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.state import (
    HydrostaticState,
    MPASHydrostaticState,
)
from legoesm.grids.vertical import (
    HybridSigmaPressureCoordinate,
    SigmaCoordinate,
)


# ---------------------------------------------------------------------------
# Common defaults (DCMIP 2008 conventions)
# ---------------------------------------------------------------------------

T0_DEFAULT = 300.0           # Isothermal background T [K]
U0_DEFAULT = 20.0            # Solid-body zonal flow amplitude [m/s]
P0_DEFAULT = constants.p_ref  # Reference surface pressure [Pa]


def hydrostatic_pressure(
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    p_s: jnp.ndarray,
) -> jnp.ndarray:
    """Pressure at full levels from sigma or hybrid coordinates [Pa].

    Returns shape ``(*p_s.shape, nlev)``.
    """
    if isinstance(sigma_coord, HybridSigmaPressureCoordinate):
        from legoesm.grids.vertical import pressure_from_hybrid
        return pressure_from_hybrid(sigma_coord, p_s)
    sigma_full = jnp.asarray(sigma_coord.sigma_full)
    return p_s[..., None] * sigma_full


def height_isothermal(
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    p_s: jnp.ndarray,
    T_0: float,
) -> jnp.ndarray:
    """Approximate full-level height z from p via isothermal scale height.

    z(p) = (R_d T_0 / g) · ln(p_s / p)

    For hybrid coords this uses the local sigma = p / p_s.
    """
    p = hydrostatic_pressure(sigma_coord, p_s)
    H = constants.R_d * T_0 / constants.g
    return H * jnp.log(p_s[..., None] / jnp.maximum(p, 1.0))


# ---------------------------------------------------------------------------
# Cubed-sphere isothermal state with optional topography
# ---------------------------------------------------------------------------


def isothermal_state_cube(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    T_0: float = T0_DEFAULT,
    u_0: float = U0_DEFAULT,
    p_s_init: float = P0_DEFAULT,
    phis: jnp.ndarray | None = None,
) -> HydrostaticState:
    """Isothermal hydrostatic state with solid-body zonal flow on cube."""
    from legoesm.grids.cubed_sphere import rotate_winds_geo_to_grid

    n = grid.n
    nlev = sigma_coord.n_levels
    shape_3d = (6, n, n, nlev)
    shape_2d = (6, n, n)

    if phis is None:
        phis = jnp.zeros(shape_2d)

    p_s = p_s_init * jnp.exp(-phis / (constants.R_d * T_0))

    # Solid-body rotation: u_east = u_0 cos(lat), v_north = 0
    u_east_2d = u_0 * jnp.cos(grid.lat)        # (6, n, n)
    v_north_2d = jnp.zeros_like(u_east_2d)
    u_grid_2d, v_grid_2d = rotate_winds_geo_to_grid(
        u_east_2d, v_north_2d, grid.angle,
    )
    u_3d = jnp.broadcast_to(u_grid_2d[..., None], shape_3d)
    v_3d = jnp.broadcast_to(v_grid_2d[..., None], shape_3d)

    T_3d = jnp.full(shape_3d, T_0)

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    return HydrostaticState(
        u=Field(data=u_3d, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_3d, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=phis, name="phis", dims=dims_2d, units="m^2/s^2"),
    )


# ---------------------------------------------------------------------------
# Lat-lon
# ---------------------------------------------------------------------------


def isothermal_state_latlon(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    T_0: float = T0_DEFAULT,
    u_0: float = U0_DEFAULT,
    p_s_init: float = P0_DEFAULT,
    phis: jnp.ndarray | None = None,
) -> HydrostaticState:
    """Isothermal hydrostatic state on a lat-lon grid."""
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = sigma_coord.n_levels
    shape_3d = (n_lat, n_lon, nlev)
    shape_2d = (n_lat, n_lon)

    if phis is None:
        phis = jnp.zeros(shape_2d)

    p_s = p_s_init * jnp.exp(-phis / (constants.R_d * T_0))

    lat_2d = jnp.asarray(grid.lat)[:, None]
    u_2d = jnp.broadcast_to(u_0 * jnp.cos(lat_2d), shape_2d)
    u_3d = jnp.broadcast_to(u_2d[..., None], shape_3d)
    v_3d = jnp.zeros(shape_3d)
    T_3d = jnp.full(shape_3d, T_0)

    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    return HydrostaticState(
        u=Field(data=u_3d, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_3d, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=phis, name="phis", dims=dims_2d, units="m^2/s^2"),
    )


# ---------------------------------------------------------------------------
# MPAS Voronoi
# ---------------------------------------------------------------------------


def isothermal_state_mpas(
    mesh,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    T_0: float = T0_DEFAULT,
    u_0: float = U0_DEFAULT,
    p_s_init: float = P0_DEFAULT,
    phis: jnp.ndarray | None = None,
) -> MPASHydrostaticState:
    """Isothermal hydrostatic state on an MPAS Voronoi mesh."""
    nCells = mesh.nCells
    nEdges = mesh.nEdges
    nlev = sigma_coord.n_levels

    if phis is None:
        phis = jnp.zeros((nCells,))

    p_s = p_s_init * jnp.exp(-phis / (constants.R_d * T_0))
    T_data = jnp.full((nCells, nlev), T_0)

    # u_normal at edges: project (u_east, 0) onto edge normal direction
    # u_normal = u_east * cos(angleEdge)
    u_edge = u_0 * jnp.cos(mesh.latEdge) * jnp.cos(mesh.angleEdge)
    u_data = jnp.broadcast_to(u_edge[:, None], (nEdges, nlev))

    return MPASHydrostaticState(
        u=Field(data=u_data, name="u", dims=("nEdges", "level"), units="m/s"),
        T=Field(data=T_data, name="T", dims=("nCells", "level"), units="K"),
        p_s=Field(data=p_s, name="p_s", dims=("nCells",), units="Pa"),
        phis=Field(data=phis, name="phis", dims=("nCells",), units="m^2/s^2"),
    )


# ---------------------------------------------------------------------------
# Spectral (Gaussian)
# ---------------------------------------------------------------------------


def isothermal_state_spectral(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    T_0: float = T0_DEFAULT,
    u_0: float = U0_DEFAULT,
    p_s_init: float = P0_DEFAULT,
    phis: jnp.ndarray | None = None,
):
    """Isothermal hydrostatic state in spectral (Gaussian-grid) space.

    ``phis`` may be either a grid-space ``(n_lat, n_lon)`` array or
    ``None`` (flat terrain).  The grid-space surface fields are
    transformed to spherical-harmonic coefficients before being
    packaged into a :class:`SpectralHydrostaticState`.
    """
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

    if phis is None:
        phis_grid = jnp.zeros((n_lat, n_lon), dtype=jnp.float64)
    else:
        phis_grid = jnp.asarray(phis, dtype=jnp.float64)

    p_s_grid = p_s_init * jnp.exp(-phis_grid / (constants.R_d * T_0))

    lat_2d = jnp.broadcast_to(grid.lat[:, None], (n_lat, n_lon))
    u_grid = jnp.broadcast_to(
        (u_0 * jnp.cos(lat_2d))[..., None],
        (n_lat, n_lon, nlev),
    ).astype(jnp.float64)
    v_grid = jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64)
    T_grid = jnp.full((n_lat, n_lon, nlev), T_0, dtype=jnp.float64)

    a = grid.radius
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

    T_hat = sh_analysis_3d(grid, T_grid)
    lnps_hat = sh_analysis(grid, jnp.log(p_s_grid))
    phis_hat = sh_analysis(grid, phis_grid)

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


# ---------------------------------------------------------------------------
# Helpers for adding scalar perturbations
# ---------------------------------------------------------------------------


def add_temperature_perturbation_grid(
    state: HydrostaticState,
    delta_T: jnp.ndarray,
) -> HydrostaticState:
    """Return a new state whose temperature field is ``T + delta_T``.

    For grid-space (cubed-sphere / lat-lon) flavours.  ``delta_T`` must
    broadcast against ``state.T.data``.
    """
    return state._replace(
        T=state.T.replace(data=state.T.data + delta_T),
    )
