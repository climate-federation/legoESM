"""DCMIP 2012 §2-0-0: atmosphere at rest with steep topography.

A hydrostatic atmosphere at rest is initialised over a ridged cosine-bell
mountain.  The DCMIP 2012 specification is a CONSTANT LAPSE RATE atmosphere,
T = T_0 - gamma z with T_0 = 300 K, gamma = 0.0065 K/m (reference Fortran
``test2_steady_state_mountain``): see :func:`lapse_rate_columns` and the
``lapse_rate_state_topography_init*`` builders, which the atmosphere matrix's
``rest_state_topo`` row uses.  NOTE: they sample that profile on each
model's NATIVE pressure levels; the reference's 12 km model top (about
206 hPa) is not reproduced, so the profile extends to the native top.  The ``rest_state_topography_init*`` builders
below are an ISOTHERMAL simplification (kept for regression tests that probe
that balance specifically; isothermal is the one profile an exactly
balanced isothermal-reference PGF cannot fail). With a perfectly consistent
pressure-gradient-force discretisation on a terrain-following vertical
coordinate the state should remain at rest indefinitely; spurious wind
generation diagnoses PGF errors.

Isothermal simplification (rest_state_topography_init*)
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


def lapse_rate_columns(z_s, sigma_coord, *, T_init: float = T_0,
                       gamma: float = GAMMA, p_0: float = P_0):
    """``(p_s, T)`` of the constant-lapse-rate atmosphere at rest over terrain
    height ``z_s`` (any shape ``(...,)``), on the model's OWN full levels:

        p_s = p_0 (1 - gamma z_s / T_init) ** (g / (R_d gamma))
        T(p) = T_init (p / p_0) ** (R_d gamma / g)          shape (..., nlev)

    This is the DCMIP 2012 test 2-0-0 atmosphere (T = T_0 - gamma z, T_0 =
    300 K, gamma = 0.0065 K/m; reference ``test2_steady_state_mountain``,
    ``dcmip_initial_conditions_test_1_2_3_v5.f90``) with ``p_0 = P_0``.
    ``p_0`` is the pressure at ``z = 0``; the DCMIP 2008 Rossby-Haurwitz
    wave reuses these columns with ``p_0 = p_ref = 955 hPa`` (its Eqs.
    66 and 70, with ``z_s`` its equivalent surface height ``-Phi'/g``).
    """
    from legoesm.grids.vertical import (
        pressure_from_hybrid, pressure_from_sigma,
    )

    if gamma <= 0.0:
        raise ValueError(
            "gamma must be > 0 (a zero lapse rate reproduces the isothermal "
            f"rest state, which is what this variant exists to escape); got {gamma!r}")
    # Surface pressure of the constant-lapse-rate atmosphere at terrain height.
    exponent = constants.g / (constants.R_d * gamma)
    p_s = p_0 * (1.0 - gamma * z_s / T_init) ** exponent

    if isinstance(sigma_coord, HybridSigmaPressureCoordinate):
        p_full = pressure_from_hybrid(sigma_coord, p_s, full=True)
    else:
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)

    # T(p) for the same atmosphere -- the inverse of p(z) above.
    T_data = T_init * (p_full / p_0) ** (constants.R_d * gamma / constants.g)
    return p_s, T_data


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
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    R = float(grid.radius)

    lat_b = jnp.broadcast_to(jnp.asarray(grid.lat)[:, None], (n_lat, n_lon))
    lon_b = jnp.broadcast_to(jnp.asarray(grid.lon)[None, :], (n_lat, n_lon))

    z_s = dcmip_2_0_0_mountain(lon_b, lat_b, R, h_0=h_0)
    phis = constants.g * z_s
    p_s, T_data = lapse_rate_columns(z_s, sigma_coord, T_init=T_init, gamma=gamma)

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


# ---------------------------------------------------------------------------
# DCMIP 2012 2-0-0 as specified (constant lapse rate) on the other grids
# ---------------------------------------------------------------------------
# The ``rest_state_topography_init*`` builders above are ISOTHERMAL; the DCMIP
# 2012 2-0-0 specification is the constant-lapse-rate atmosphere of
# ``lapse_rate_columns``.  These are the spec-faithful builders the atmosphere
# matrix's ``rest_state_topo`` row uses on every grid.


def lapse_rate_state_topography_init(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    h_0: float = H_0,
    T_init: float = T_0,
    gamma: float = GAMMA,
) -> HydrostaticState:
    """DCMIP 2012 2-0-0 (constant lapse rate) on a cubed-sphere grid."""
    n = grid.n
    nlev = sigma_coord.n_levels
    z_s = dcmip_2_0_0_mountain(grid.lon, grid.lat, float(grid.radius), h_0=h_0)
    phis = constants.g * z_s
    p_s, T_data = lapse_rate_columns(z_s, sigma_coord, T_init=T_init, gamma=gamma)
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


def lapse_rate_state_topography_init_mpas(
    mesh,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    h_0: float = H_0,
    T_init: float = T_0,
    gamma: float = GAMMA,
):
    """DCMIP 2012 2-0-0 (constant lapse rate) on an MPAS Voronoi mesh."""
    from legoesm.core.state import MPASHydrostaticState

    z_s = dcmip_2_0_0_mountain(mesh.lonCell, mesh.latCell, float(mesh.radius), h_0=h_0)
    phis = constants.g * z_s
    p_s, T_data = lapse_rate_columns(z_s, sigma_coord, T_init=T_init, gamma=gamma)
    u_data = jnp.zeros((mesh.nEdges, sigma_coord.n_levels))
    return MPASHydrostaticState(
        u=Field(data=u_data, name="u", dims=("nEdges", "level"), units="m/s"),
        T=Field(data=T_data, name="T", dims=("nCells", "level"), units="K"),
        p_s=Field(data=p_s, name="p_s", dims=("nCells",), units="Pa"),
        phis=Field(data=phis, name="phis", dims=("nCells",), units="m^2/s^2"),
    )


def lapse_rate_state_topography_init_spectral(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    h_0: float = H_0,
    T_init: float = T_0,
    gamma: float = GAMMA,
    dealiasing_fraction: float,
):
    """DCMIP 2012 2-0-0 (constant lapse rate) in spectral space.

    ``dealiasing_fraction`` is the model's ``SpectralPEConfig`` value: the
    orography is projected onto the band the model keeps
    (``n <= floor(fraction * n_max)``, ``band_limited_orography``) BEFORE the
    columns are built, as the spectral NH TC1/TC2 inits do.  The model
    truncates ``lnps`` to that band after every step, so on an unprojected
    IC the first step changed p_s by up to ~2.2 kPa at the mountain at T21
    (projection, not dynamics), and ``phis`` kept the cut terrain detail
    that ``lnps`` no longer balanced.  The column build is nonlinear in
    ``z_s``, so lnps keeps a small off-band part: projecting it changes p_s
    by up to 1.07 Pa at T21.  ``<= 0`` keeps the raw terrain.
    """
    from legoesm.atmosphere.dynamics.gcm.spectral_nh import band_limited_orography
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralHydrostaticState
    from legoesm.grids.gaussian import sh_analysis, sh_analysis_3d

    n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, sigma_coord.n_levels
    lat_2d = jnp.broadcast_to(grid.lat[:, None], (n_lat, n_lon))
    lon_2d = jnp.broadcast_to(grid.lon2d, (n_lat, n_lon))
    z_s = jnp.asarray(dcmip_2_0_0_mountain(lon_2d, lat_2d, float(grid.radius),
                                           h_0=h_0), dtype=jnp.float64)
    z_s = band_limited_orography(grid, z_s, dealiasing_fraction)
    phis_grid = constants.g * z_s
    p_s_grid, T_grid = lapse_rate_columns(z_s, sigma_coord, T_init=T_init, gamma=gamma)

    T_hat = sh_analysis_3d(grid, jnp.asarray(T_grid, dtype=jnp.float64))
    lnps_hat = sh_analysis(grid, jnp.log(p_s_grid))
    phis_hat = sh_analysis(grid, phis_grid)
    vor_hat = jnp.zeros((grid.n_sh, nlev), dtype=jnp.complex128)
    dims_3d = ("spectral", "level")
    dims_2d = ("spectral",)
    return SpectralHydrostaticState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=jnp.zeros_like(vor_hat), name="div_hat", dims=dims_3d,
                      units="1/s"),
        T_hat=Field(data=T_hat, name="T_hat", dims=dims_3d, units="K"),
        lnps_hat=Field(data=lnps_hat, name="lnps_hat", dims=dims_2d, units=""),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d,
                       units="m^2/s^2"),
    )
