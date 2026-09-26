"""DCMIP 2012 §2-0-0: atmosphere at rest with steep topography.

A purely hydrostatic, isothermal atmosphere at rest is initialised over a
ridged cosine-bell mountain. With a perfectly consistent
pressure-gradient-force discretisation on a terrain-following vertical
coordinate the state should remain at rest indefinitely; spurious wind
generation diagnoses PGF errors.

Initial conditions (Ullrich et al. 2012, Eq. for §2-0-0)
-------------------------------------------------------
- u = v = 0
- T(λ, φ, z) = T_0 = 300 K  (isothermal)
- φ_s(λ, φ) = g · z_s(λ, φ)  with z_s from
  :func:`tests.test_cases._terrain_helpers.dcmip_2_0_0_mountain`
- p_s(λ, φ) = p_0 · exp(-φ_s / (R_d · T_0))  (hydrostatic balance)
- p(level k) = p_s · sigma_full[k] for sigma coords; for hybrid
  coordinates the relation
  :math:`p(η) = A(η) · p_{ref} + B(η) · p_s` is used (handled by the
  dycore; here we only set p_s consistently).

Reference
---------
Ullrich, P. A., Jablonowski, C., Kent, J., Lauritzen, P. H., Nair, R.,
& Taylor, M. A. (2012). DCMIP test case document. *Geosci. Model Dev.*,
projects supplement.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.grids.vertical import (
    HybridSigmaPressureCoordinate,
    SigmaCoordinate,
)

from legoesm.atmosphere.idealized.topography import dcmip_2_0_0_mountain


# ---------------------------------------------------------------------------
# Test constants (DCMIP 2012 §2-0-0)
# ---------------------------------------------------------------------------

T_0 = 300.0           # Isothermal background temperature [K]
P_0 = constants.p_ref  # Reference pressure [Pa]
H_0 = 2000.0          # Mountain peak height [m]


# ---------------------------------------------------------------------------
# Cubed-sphere
# ---------------------------------------------------------------------------


def rest_state_topography_init(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    h_0: float = H_0,
    T_init: float = T_0,
) -> HydrostaticState:
    """Initialise rest state with DCMIP §2-0-0 mountain on cubed-sphere."""
    n = grid.n
    nlev = sigma_coord.n_levels
    R = float(grid.radius)

    z_s = dcmip_2_0_0_mountain(grid.lon, grid.lat, R, h_0=h_0)
    phis = constants.g * z_s

    p_s = P_0 * jnp.exp(-phis / (constants.R_d * T_init))
    T_data = jnp.full((6, n, n, nlev), T_init)

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    zeros_3d = jnp.zeros((6, n, n, nlev))

    return HydrostaticState(
        u=Field(data=zeros_3d, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=zeros_3d, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_data, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=phis, name="phis", dims=dims_2d, units="m^2/s^2"),
    )


# ---------------------------------------------------------------------------
# Lat-lon
# ---------------------------------------------------------------------------


def rest_state_topography_init_latlon(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    h_0: float = H_0,
    T_init: float = T_0,
) -> HydrostaticState:
    """Initialise rest state with DCMIP §2-0-0 mountain on lat-lon."""
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = sigma_coord.n_levels
    R = float(grid.radius)

    lat_2d = jnp.asarray(grid.lat)[:, None]
    lon_2d = jnp.asarray(grid.lon)[None, :]
    lat_b = jnp.broadcast_to(lat_2d, (n_lat, n_lon))
    lon_b = jnp.broadcast_to(lon_2d, (n_lat, n_lon))

    z_s = dcmip_2_0_0_mountain(lon_b, lat_b, R, h_0=h_0)
    phis = constants.g * z_s

    p_s = P_0 * jnp.exp(-phis / (constants.R_d * T_init))
    T_data = jnp.full((n_lat, n_lon, nlev), T_init)

    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    zeros_3d = jnp.zeros((n_lat, n_lon, nlev))

    return HydrostaticState(
        u=Field(data=zeros_3d, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=zeros_3d, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_data, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=phis, name="phis", dims=dims_2d, units="m^2/s^2"),
    )


# ---------------------------------------------------------------------------
# Stratified (constant-lapse-rate) variant of the same rest state
# ---------------------------------------------------------------------------

GAMMA = 0.0065        # Constant lapse rate [K/m] (US Standard troposphere)


def lapse_rate_state_topography_init_latlon(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    h_0: float = H_0,
    T_init: float = T_0,
    gamma: float = GAMMA,
) -> HydrostaticState:
    """Rest state over the same mountain, but STRATIFIED (#1029).

    The DCMIP 2-0-0 state above is isothermal, so ``grad T = 0`` identically
    and the temperature-gradient part of the pressure-gradient force,
    ``-R_d * integral of grad(T_v) d ln p``, is never exercised: the balance it
    certifies is only the one between ``-grad(Phi)`` and the surface-pressure
    correction.  A terrain-following discretisation can pass that and still be
    inconsistent in the temperature term, which is the part that switches on
    the moment the atmosphere is stratified -- i.e. in every real run.

    This variant uses the exact constant-lapse-rate atmosphere

        T(z) = T_init - gamma * z ,
        p(z) = P_0 * (1 - gamma * z / T_init) ** (g / (R_d * gamma)) ,

    whose temperature is a function of PRESSURE alone,

        T(p) = T_init * (p / P_0) ** (R_d * gamma / g) ,

    so it is still an EXACT hydrostatic state at rest, and the correct answer
    is still "stays at rest indefinitely".  But because the model's levels are
    terrain-following, ``p`` -- and therefore ``T`` -- varies horizontally
    along a coordinate surface wherever the terrain slopes, so the temperature
    integral now carries a real signal that must cancel discretely.

    ``T`` is evaluated on the model's OWN full-level pressures (via
    ``pressure_from_hybrid`` / ``pressure_from_sigma``) rather than on a
    re-derived ladder, so the initial state is hydrostatically consistent with
    the coordinate the dycore actually integrates.
    """
    from legoesm.grids.vertical import (
        pressure_from_hybrid, pressure_from_sigma,
    )

    if gamma <= 0.0:
        raise ValueError(
            "gamma must be > 0 (a zero lapse rate reproduces the isothermal "
            f"rest state, which is what this variant exists to escape); got {gamma!r}")

    n_lat = grid.n_lat
    n_lon = grid.n_lon
    R = float(grid.radius)

    lat_b = jnp.broadcast_to(jnp.asarray(grid.lat)[:, None], (n_lat, n_lon))
    lon_b = jnp.broadcast_to(jnp.asarray(grid.lon)[None, :], (n_lat, n_lon))

    z_s = dcmip_2_0_0_mountain(lon_b, lat_b, R, h_0=h_0)
    phis = constants.g * z_s

    # Surface pressure of the constant-lapse-rate atmosphere at terrain height.
    exponent = constants.g / (constants.R_d * gamma)
    p_s = P_0 * (1.0 - gamma * z_s / T_init) ** exponent

    if isinstance(sigma_coord, HybridSigmaPressureCoordinate):
        p_full = pressure_from_hybrid(sigma_coord, p_s, full=True)
    else:
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)

    # T(p) for the same atmosphere -- the inverse of p(z) above.
    T_data = T_init * (p_full / P_0) ** (constants.R_d * gamma / constants.g)

    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    zeros_3d = jnp.zeros((n_lat, n_lon, sigma_coord.n_levels))

    return HydrostaticState(
        u=Field(data=zeros_3d, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=zeros_3d, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_data, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=phis, name="phis", dims=dims_2d, units="m^2/s^2"),
    )


# ---------------------------------------------------------------------------
# MPAS Voronoi
# ---------------------------------------------------------------------------


def rest_state_topography_init_mpas(
    mesh,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    h_0: float = H_0,
    T_init: float = T_0,
):
    """Initialise rest state with DCMIP §2-0-0 mountain on MPAS Voronoi."""
    from legoesm.core.state import MPASHydrostaticState

    nCells = mesh.nCells
    nEdges = mesh.nEdges
    nlev = sigma_coord.n_levels
    R = float(mesh.radius)

    z_s = dcmip_2_0_0_mountain(mesh.lonCell, mesh.latCell, R, h_0=h_0)
    phis = constants.g * z_s

    p_s = P_0 * jnp.exp(-phis / (constants.R_d * T_init))
    T_data = jnp.full((nCells, nlev), T_init)
    u_data = jnp.zeros((nEdges, nlev))

    return MPASHydrostaticState(
        u=Field(data=u_data, name="u", dims=("nEdges", "level"), units="m/s"),
        T=Field(data=T_data, name="T", dims=("nCells", "level"), units="K"),
        p_s=Field(data=p_s, name="p_s", dims=("nCells",), units="Pa"),
        phis=Field(data=phis, name="phis", dims=("nCells",), units="m^2/s^2"),
    )


# ---------------------------------------------------------------------------
# Spectral (Gaussian)
# ---------------------------------------------------------------------------


def rest_state_topography_init_spectral(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    h_0: float = H_0,
    T_init: float = T_0,
):
    """Initialise rest state with DCMIP §2-0-0 mountain in spectral space."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralHydrostaticState
    from legoesm.grids.gaussian import sh_analysis, sh_analysis_3d

    n_lat = grid.n_lat
    n_lon = grid.n_lon
    n_sh = grid.n_sh
    nlev = sigma_coord.n_levels
    R = float(grid.radius)

    lat_2d = jnp.broadcast_to(grid.lat[:, None], (n_lat, n_lon))
    lon_2d = jnp.broadcast_to(grid.lon2d, (n_lat, n_lon))

    z_s = dcmip_2_0_0_mountain(lon_2d, lat_2d, R, h_0=h_0)
    phis_grid = constants.g * z_s
    p_s_grid = P_0 * jnp.exp(-phis_grid / (constants.R_d * T_init))

    T_grid = jnp.full((n_lat, n_lon, nlev), T_init, dtype=jnp.float64)

    T_hat = sh_analysis_3d(grid, T_grid)
    lnps_hat = sh_analysis(grid, jnp.log(p_s_grid))
    phis_hat = sh_analysis(grid, phis_grid)

    vor_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    div_hat = jnp.zeros_like(vor_hat)

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
