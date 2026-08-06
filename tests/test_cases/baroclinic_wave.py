"""Baroclinic wave test case for all grid types (DCMIP 2016).

Implements the Jablonowski-Williamson baroclinic instability test, the
gold-standard benchmark for hydrostatic dynamical cores.

The test has two components:

1. **Steady-state preservation**: Initialize with an analytically balanced
   state (gradient-wind + hydrostatic balance). Verify the model maintains
   it over ~30 days. Surface pressure perturbations indicate discretization
   error in the balance.

2. **Baroclinic wave growth**: Add a localized perturbation to trigger
   baroclinic instability. Wave growth begins ~day 4, explosive
   cyclogenesis ~day 8-9. Verify realistic wave structure at 850 hPa.

Initial conditions are defined analytically in height (z) coordinates,
then mapped to sigma levels by inverting p(z) via bisection.

Supported grids: cubed-sphere, lat-lon, MPAS Voronoi, spectral (Gaussian).

Expected results (C48, 26 levels, 10 days with perturbation):
- Subtropical jet ~35 m/s at ~250 hPa
- Wave packet develops in Northern Hemisphere midlatitudes
- 850 hPa relative vorticity shows frontal features (~1e-4 s^-1)
- Surface pressure minimum drops to ~960 hPa by day 9

References
----------
- Jablonowski, C., & Williamson, D. L. (2006). A baroclinic instability
  test case for atmospheric model dynamical cores. QJRMS, 132, 2943-2975.
- Ullrich, P. A., et al. (2016). DCMIP2016 Test Case Document.
  https://github.com/ClimateGlobalChange/DCMIP2016
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.grids.vertical import SigmaCoordinate
from legoesm import constants


# ==============================================================================
# Test case parameters (DCMIP 2016, dry, shallow atmosphere, X=1)
# ==============================================================================

# Temperature parameters
T0E = 310.0           # Equatorial surface temperature [K]
T0P = 240.0           # Polar surface temperature [K]
B_PARAM = 2.0         # Jet half-width parameter (controls vertical extent)
K_PARAM = 3.0         # Jet width parameter (controls meridional width)
LAPSE = 0.005         # Background lapse rate [K/m]

# Perturbation parameters (exponential type, as in DCMIP 2016)
PERT_UP = 1.0         # Perturbation amplitude [m/s]
PERT_EXPR = 0.1       # Perturbation radius [Earth radii]
PERT_LON = math.pi / 9.0       # Perturbation center longitude [rad] (~20 deg)
PERT_LAT = 2.0 * math.pi / 9.0  # Perturbation center latitude [rad] (~40 deg N)
PERT_Z = 15000.0      # Perturbation height cap [m]

# Reference surface pressure
P0 = 1.0e5            # [Pa]

# Bisection parameters
Z_MAX = 50000.0       # Maximum height for bisection [m]
N_BISECT = 60         # Number of bisection iterations (~1e-14 m precision)


# ==============================================================================
# Derived constants
# ==============================================================================

_T0 = 0.5 * (T0E + T0P)              # Mean temperature [K]
_constA = 1.0 / LAPSE                 # = 200.0
_constB = (_T0 - T0P) / (_T0 * T0P)  # Meridional T gradient parameter
_constC = 0.5 * (K_PARAM + 2.0) * (T0E - T0P) / (T0E * T0P)  # Jet T parameter


def _scale_height() -> float:
    """Compute the reference scale height H = R_d * T0 / g."""
    return float(constants.R_d) * _T0 / float(constants.g)


# ==============================================================================
# Analytic solution functions (JAX-native, vectorized)
# ==============================================================================

def evaluate_pressure_temperature(
    z: jnp.ndarray,
    lat: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute pressure and temperature at given height and latitude.

    These are the analytic steady-state fields that satisfy both
    hydrostatic balance and gradient-wind balance.

    Parameters
    ----------
    z : jnp.ndarray
        Height [m]. Any shape.
    lat : jnp.ndarray
        Latitude [rad]. Must broadcast with z.

    Returns
    -------
    p : jnp.ndarray
        Pressure [Pa].
    T : jnp.ndarray
        Temperature [K].
    """
    g = float(constants.g)
    Rd = float(constants.R_d)
    H = _scale_height()

    scaledZ = z / (B_PARAM * H)

    # Temperature factors (Eq. in DCMIP2016 test case document)
    tau1 = (_constA * (LAPSE / _T0) * jnp.exp(LAPSE * z / _T0)
            + _constB * (1.0 - 2.0 * scaledZ**2) * jnp.exp(-scaledZ**2))
    tau2 = _constC * (1.0 - 2.0 * scaledZ**2) * jnp.exp(-scaledZ**2)

    # Latitude-dependent term
    cos_lat = jnp.cos(lat)
    inttermT = (cos_lat**K_PARAM
                - (K_PARAM / (K_PARAM + 2.0)) * cos_lat**(K_PARAM + 2.0))

    # Temperature
    T = 1.0 / (tau1 - tau2 * inttermT)

    # Integrated hydrostatic terms for pressure
    inttau1 = (_constA * (jnp.exp(LAPSE * z / _T0) - 1.0)
               + _constB * z * jnp.exp(-scaledZ**2))
    inttau2 = _constC * z * jnp.exp(-scaledZ**2)

    # Pressure
    p = P0 * jnp.exp(-(g / Rd) * (inttau1 - inttau2 * inttermT))

    return p, T


def find_z_for_pressure(
    p_target: jnp.ndarray,
    lat: jnp.ndarray,
    n_iter: int = N_BISECT,
) -> jnp.ndarray:
    """Find height z where pressure equals p_target via bisection.

    Since pressure decreases monotonically with height, bisection
    converges reliably. With 60 iterations, precision is ~Z_MAX/2^60
    ~ 4e-14 m.

    Parameters
    ----------
    p_target : jnp.ndarray
        Target pressure [Pa]. Any shape.
    lat : jnp.ndarray
        Latitude [rad]. Must broadcast with p_target.
    n_iter : int
        Number of bisection iterations.

    Returns
    -------
    z : jnp.ndarray
        Height [m] where p(z, lat) = p_target.
    """
    z_lo = jnp.zeros_like(p_target)
    z_hi = jnp.full_like(p_target, Z_MAX)

    for _ in range(n_iter):
        z_mid = 0.5 * (z_lo + z_hi)
        p_mid, _ = evaluate_pressure_temperature(z_mid, lat)
        # Pressure decreases with height: if p_mid < p_target, z is too high
        too_high = p_mid < p_target
        z_hi = jnp.where(too_high, z_mid, z_hi)
        z_lo = jnp.where(too_high, z_lo, z_mid)

    return 0.5 * (z_lo + z_hi)


def compute_zonal_wind(
    z: jnp.ndarray,
    lat: jnp.ndarray,
    T: jnp.ndarray,
) -> jnp.ndarray:
    """Compute zonal wind from gradient-wind balance.

    Solves the quadratic gradient-wind equation for the steady-state
    zonal flow (shallow atmosphere):

        u^2 / (a cos phi) + 2 Omega u = bigU

    where bigU encodes the pressure gradient.

    Parameters
    ----------
    z : jnp.ndarray
        Height [m].
    lat : jnp.ndarray
        Latitude [rad].
    T : jnp.ndarray
        Temperature [K] (needed for the pressure gradient term).

    Returns
    -------
    u : jnp.ndarray
        Zonal wind [m/s].
    """
    a = float(constants.R_earth)
    omega = float(constants.Omega)
    g = float(constants.g)
    H = _scale_height()

    scaledZ = z / (B_PARAM * H)
    cos_lat = jnp.cos(lat)

    # Integrated tau2 for wind
    inttau2 = _constC * z * jnp.exp(-scaledZ**2)

    # Latitude-dependent term for wind (different from temperature term!)
    inttermU = cos_lat**(K_PARAM - 1.0) - cos_lat**(K_PARAM + 1.0)

    # Gradient-wind balance: solve u^2/(a*cos(lat)) + 2*omega*u = bigU
    bigU = (g / a) * K_PARAM * inttau2 * inttermU * T

    rcoslat = a * cos_lat
    omegarcoslat = omega * rcoslat

    # Quadratic formula: u = -b + sqrt(b^2 + c)
    # where b = omega*a*cos(lat), c = a*cos(lat)*bigU
    u = -omegarcoslat + jnp.sqrt(jnp.maximum(
        0.0, omegarcoslat**2 + rcoslat * bigU
    ))

    return u


def exponential_perturbation(
    lat: jnp.ndarray,
    lon: jnp.ndarray,
    z: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the exponential wind perturbation.

    A localized perturbation centered at (PERT_LON, PERT_LAT) in the
    Northern Hemisphere midlatitudes, tapering to zero above PERT_Z.

    Parameters
    ----------
    lat : jnp.ndarray
        Latitude [rad].
    lon : jnp.ndarray
        Longitude [rad].
    z : jnp.ndarray
        Height [m].

    Returns
    -------
    u_pert : jnp.ndarray
        Perturbation zonal wind [m/s].
    """
    # Great circle distance (in units of perturbation radius)
    cos_angle = (jnp.sin(PERT_LAT) * jnp.sin(lat)
                 + jnp.cos(PERT_LAT) * jnp.cos(lat) * jnp.cos(lon - PERT_LON))
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    r_gc = jnp.arccos(cos_angle) / PERT_EXPR

    # Vertical taper (cubic hermite: 1 at z=0, 0 at z=PERT_Z)
    zfrac = z / PERT_Z
    taper = jnp.where(
        z < PERT_Z,
        1.0 - 3.0 * zfrac**2 + 2.0 * zfrac**3,
        0.0,
    )

    # Gaussian in great-circle distance, zero outside r_gc=1
    u_pert = jnp.where(
        r_gc < 1.0,
        PERT_UP * taper * jnp.exp(-r_gc**2),
        0.0,
    )

    return u_pert


# ==============================================================================
# Cubed-sphere initialization
# ==============================================================================

def baroclinic_wave_init(
    grid,
    sigma_coord: SigmaCoordinate,
    perturbed: bool = True,
    moist: bool = False,
    rh_init: float = 0.7,
) -> HydrostaticState:
    """Create initial conditions for the baroclinic wave test on a cubed-sphere grid.

    Computes the analytic balanced state on the model's sigma levels
    by inverting the height-based formulation via bisection at each
    grid point and level.

    Parameters
    ----------
    grid : CubedSphereGrid
        Horizontal grid.
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    perturbed : bool
        If True, add the exponential perturbation to trigger instability.
        If False, return only the steady-state (balanced) initial conditions.
    moist : bool
        If True, attach ``q_v``/``q_c``/``q_r`` tracers for a *moist*
        baroclinic wave, using the same RH-tapered profile as
        :func:`baroclinic_wave_init_mpas` / ``_spectral``
        (``rh_init · q_sat(T,p) · σ²``, capped at saturation; ``q_c=q_r=0``)
        so the balanced moist state is identical across grids.  Dry mass /
        temperature / wind are unchanged (moisture added passively at t=0).
    rh_init : float
        Surface relative humidity for the ``q_v`` taper when ``moist=True``.

    Returns
    -------
    HydrostaticState
        Initial state for the baroclinic wave test.
    """
    from legoesm.grids.cubed_sphere import rotate_winds_geo_to_grid

    n = grid.n
    nlev = sigma_coord.n_levels

    # Grid coordinates (already JAX arrays)
    lat = grid.lat                # (6, n, n)
    lon = grid.lon                # (6, n, n)
    sigma_full = sigma_coord.sigma_full  # (nlev,)

    # Surface pressure: constant everywhere (no topography)
    p_s = jnp.full((6, n, n), P0)

    # Allocate 3D fields
    u_3d = jnp.zeros((6, n, n, nlev))
    T_3d = jnp.zeros((6, n, n, nlev))

    # Compute initial conditions level by level
    for k in range(nlev):
        # Target pressure at this sigma level
        p_target = jnp.full((6, n, n), float(sigma_full[k]) * P0)

        # Find height where p(z, lat) = p_target
        z_k = find_z_for_pressure(p_target, lat)

        # Compute temperature at this height
        _, T_k = evaluate_pressure_temperature(z_k, lat)

        # Compute zonal wind from gradient-wind balance
        u_k = compute_zonal_wind(z_k, lat, T_k)

        # Add perturbation if requested
        if perturbed:
            u_k = u_k + exponential_perturbation(lat, lon, z_k)

        u_3d = u_3d.at[:, :, :, k].set(u_k)
        T_3d = T_3d.at[:, :, :, k].set(T_k)

    # u_3d is the geographic EASTWARD wind; v_geo = 0 (no northward wind).
    # The model stores GRID-ALIGNED velocity components, so we must rotate
    # from geographic (east, north) to local cubed-sphere (x, y) coordinates.
    v_north = jnp.zeros((6, n, n, nlev))

    # Rotate at each level using the grid angle
    u_grid = jnp.zeros_like(u_3d)
    v_grid = jnp.zeros_like(u_3d)
    for k in range(nlev):
        u_k, v_k = rotate_winds_geo_to_grid(
            u_3d[..., k], v_north[..., k], grid.angle
        )
        u_grid = u_grid.at[..., k].set(u_k)
        v_grid = v_grid.at[..., k].set(v_k)

    # Build state
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    tracers = None
    if moist:
        # Same RH-tapered q_v init as baroclinic_wave_init_mpas / _spectral:
        # q_v = rh · q_sat(T, p) · σ², capped at saturation; q_c=q_r=0.
        # Sigma coordinate ⇒ p_full = p_s · σ_full.
        from legoesm.thermo import saturation_mixing_ratio

        p_full = p_s[..., None] * sigma_full[None, None, None, :]  # (6,n,n,nlev)
        q_sat = saturation_mixing_ratio(T_3d, p_full)
        q_v_data = jnp.minimum(
            rh_init * q_sat * sigma_full[None, None, None, :] ** 2, q_sat)
        q_zero = jnp.zeros((6, n, n, nlev))
        tracers = {
            "q_v": Field(data=q_v_data, name="q_v", dims=dims_3d, units="kg/kg"),
            "q_c": Field(data=q_zero, name="q_c", dims=dims_3d, units="kg/kg"),
            "q_r": Field(data=q_zero, name="q_r", dims=dims_3d, units="kg/kg"),
        }

    return HydrostaticState(
        u=Field(data=u_grid, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_grid, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=jnp.zeros((6, n, n)),
                   name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=tracers,
    )


# ==============================================================================
# Lat-lon initialization
# ==============================================================================

def baroclinic_wave_init_latlon(
    grid,
    sigma_coord: SigmaCoordinate,
    perturbed: bool = True,
    moist: bool = False,
    rh_init: float = 0.7,
) -> HydrostaticState:
    """Create baroclinic wave initial conditions for a lat-lon grid.

    Same physics as :func:`baroclinic_wave_init` but for lat-lon grids
    where ``grid.lat`` is 1-D ``(n_lat,)`` and ``grid.lon`` is 1-D
    ``(n_lon,)``.  No wind rotation is needed (grid axes = geographic).

    ``moist=True`` attaches the same RH-tapered ``q_v``/``q_c``/``q_r``
    tracers as :func:`baroclinic_wave_init` (``rh_init · q_sat(T,p) · σ²``,
    capped at saturation; ``q_c=q_r=0``) for a moist baroclinic wave.
    """
    nlev = sigma_coord.n_levels
    # Build 2D coordinate arrays from 1D lat/lon
    lat_1d = jnp.asarray(grid.lat)   # (n_lat,)
    lon_1d = jnp.asarray(grid.lon)   # (n_lon,)
    lat = lat_1d[:, None]            # (n_lat, 1) -- broadcasts with (n_lat, n_lon)
    lon = lon_1d[None, :]            # (1, n_lon)
    n_lat = lat_1d.size
    n_lon = lon_1d.size

    sigma_full = sigma_coord.sigma_full

    p_s = jnp.full((n_lat, n_lon), P0)
    u_3d = jnp.zeros((n_lat, n_lon, nlev))
    v_3d = jnp.zeros((n_lat, n_lon, nlev))
    T_3d = jnp.zeros((n_lat, n_lon, nlev))

    for k in range(nlev):
        p_target = jnp.full((n_lat, n_lon), float(sigma_full[k]) * P0)
        z_k = find_z_for_pressure(p_target, lat)
        _, T_k = evaluate_pressure_temperature(z_k, lat)
        u_k = compute_zonal_wind(z_k, lat, T_k)
        if perturbed:
            u_k = u_k + exponential_perturbation(lat, lon, z_k)
        u_3d = u_3d.at[:, :, k].set(u_k)
        T_3d = T_3d.at[:, :, k].set(T_k)

    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    tracers = None
    if moist:
        # Same RH-tapered q_v init as baroclinic_wave_init / _mpas / _spectral:
        # q_v = rh · q_sat(T, p) · σ², capped at saturation; q_c=q_r=0.
        from legoesm.thermo import saturation_mixing_ratio

        p_full = p_s[..., None] * sigma_full[None, None, :]  # (n_lat,n_lon,nlev)
        q_sat = saturation_mixing_ratio(T_3d, p_full)
        q_v_data = jnp.minimum(
            rh_init * q_sat * sigma_full[None, None, :] ** 2, q_sat)
        q_zero = jnp.zeros((n_lat, n_lon, nlev))
        tracers = {
            "q_v": Field(data=q_v_data, name="q_v", dims=dims_3d, units="kg/kg"),
            "q_c": Field(data=q_zero, name="q_c", dims=dims_3d, units="kg/kg"),
            "q_r": Field(data=q_zero, name="q_r", dims=dims_3d, units="kg/kg"),
        }

    return HydrostaticState(
        u=Field(data=u_3d, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_3d, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=jnp.zeros((n_lat, n_lon)),
                   name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=tracers,
    )


# ==============================================================================
# MPAS Voronoi mesh initialization
# ==============================================================================

def baroclinic_wave_init_mpas(
    mesh,
    sigma_coord,
    perturbed: bool = True,
    moist: bool = False,
    rh_init: float = 0.7,
):
    """Initialize Jablonowski-Williamson baroclinic wave on MPAS mesh.

    Uses the same DCMIP 2016 analytic solution as the cubed-sphere and
    spectral initializations, ensuring identical balanced states across
    all grid types.  The height-based formulation is inverted to sigma
    levels via bisection.

    Parameters
    ----------
    mesh : VoronoiMesh
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    perturbed : bool
        If True, add the exponential perturbation to trigger instability.
    moist : bool
        If True, attach ``q_v``/``q_c``/``q_r`` tracers for a *moist*
        baroclinic wave (the dycore then advects them and a Kessler
        ``physics_fn`` condenses/precipitates).  ``q_v`` follows the same
        RH-tapered profile used by ``ModelDriver`` for all grids
        (``rh_init · q_sat(T,p) · σ²``, capped at saturation); ``q_c`` and
        ``q_r`` start at zero.  Dry mass / temperature / wind are unchanged
        (the moisture is added passively at t=0).
    rh_init : float
        Surface relative humidity used for the ``q_v`` taper when
        ``moist=True``.

    Returns
    -------
    MPASHydrostaticState
    """
    from legoesm.core.state import MPASHydrostaticState

    nCells = mesh.nCells
    nEdges = mesh.nEdges
    nlev = sigma_coord.n_levels
    sigma_full = sigma_coord.sigma_full  # (nlev,)

    lat_c = mesh.latCell   # (nCells,)
    lat_e = mesh.latEdge   # (nEdges,)
    lon_e = mesh.lonEdge   # (nEdges,)
    cos_angle = jnp.cos(mesh.angleEdge)  # (nEdges,)

    # Surface pressure: constant (no topography)
    p_s_data = jnp.full((nCells,), P0)

    # Compute T at cell centers and u at edges, level by level
    T_data = jnp.zeros((nCells, nlev))
    u_data = jnp.zeros((nEdges, nlev))

    for k in range(nlev):
        sig_k = float(sigma_full[k])

        # --- Temperature at cell centers ---
        p_target_c = jnp.full((nCells,), sig_k * P0)
        z_c = find_z_for_pressure(p_target_c, lat_c)
        _, T_k = evaluate_pressure_temperature(z_c, lat_c)
        T_data = T_data.at[:, k].set(T_k)

        # --- Zonal wind at edge midpoints ---
        p_target_e = jnp.full((nEdges,), sig_k * P0)
        z_e = find_z_for_pressure(p_target_e, lat_e)
        _, T_e = evaluate_pressure_temperature(z_e, lat_e)
        u_zonal = compute_zonal_wind(z_e, lat_e, T_e)

        # Add perturbation (eastward direction, projected to edge normal)
        if perturbed:
            u_zonal = u_zonal + exponential_perturbation(lat_e, lon_e, z_e)

        # Project zonal wind to edge normal: u_n = u_east * cos(angle)
        # (v_north = 0 for this test case)
        u_data = u_data.at[:, k].set(u_zonal * cos_angle)

    # Surface geopotential: flat
    phis_data = jnp.zeros((nCells,))

    tracers = None
    if moist:
        # Shared RH-tapered q_v init (identical formula to ModelDriver's
        # all-grid moist initialization): q_v = rh · q_sat(T, p) · σ²,
        # capped at saturation; q_c = q_r = 0.  Sigma coordinate ⇒
        # p_full = p_s · σ_full.
        from legoesm.thermo import saturation_mixing_ratio

        p_full = p_s_data[:, None] * sigma_full[None, :]  # (nCells, nlev)
        q_sat = saturation_mixing_ratio(T_data, p_full)
        q_v_data = jnp.minimum(rh_init * q_sat * sigma_full[None, :] ** 2, q_sat)
        q_zero = jnp.zeros((nCells, nlev))
        tracers = {
            "q_v": Field(data=q_v_data, name="q_v",
                         dims=("nCells", "level"), units="kg/kg"),
            "q_c": Field(data=q_zero, name="q_c",
                         dims=("nCells", "level"), units="kg/kg"),
            "q_r": Field(data=q_zero, name="q_r",
                         dims=("nCells", "level"), units="kg/kg"),
        }

    return MPASHydrostaticState(
        u=Field(data=u_data, name="u", dims=("nEdges", "level"), units="m/s"),
        T=Field(data=T_data, name="T", dims=("nCells", "level"), units="K"),
        p_s=Field(data=p_s_data, name="p_s", dims=("nCells",), units="Pa"),
        phis=Field(data=phis_data, name="phis", dims=("nCells",), units="m^2/s^2"),
        tracers=tracers,
    )


# ==============================================================================
# Spectral (Gaussian grid) initialization
# ==============================================================================

def baroclinic_wave_init_spectral(
    grid,
    sigma_coord: SigmaCoordinate,
    perturbed: bool = True,
    moist: bool = False,
    rh_init: float = 0.7,
):
    """Initialize Jablonowski-Williamson baroclinic wave in spectral space.

    Evaluates the analytic JW06 balanced state on Gaussian grid points,
    then transforms u,v -> vorticity/divergence via spectral analysis.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid with SH transform matrices.
    sigma_coord : SigmaCoordinate
        Vertical sigma coordinate.
    perturbed : bool
        If True, add the exponential perturbation to trigger instability.
    moist : bool
        If True, attach grid-space ``q_v``/``q_c``/``q_r`` tracers for a moist
        baroclinic wave (passive at t=0; ``make_kessler_forcing_spectral``
        condenses/precipitates). ``q_v`` uses the same RH-tapered formula as the
        MPAS / cubed-sphere moist init: ``q_v = rh · q_sat(T, p) · σ²`` capped at
        saturation; ``q_c = q_r = 0``.
    rh_init : float
        Surface relative humidity for the ``q_v`` taper when ``moist=True``.

    Returns
    -------
    SpectralHydrostaticState
        Initial state for the baroclinic wave test.
    """
    import numpy as np
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralHydrostaticState
    from legoesm.grids.gaussian import (
        sh_analysis,
        sh_analysis_3d,
        sh_analysis_oc2_3d,
        sh_analysis_dmu_3d,
    )

    nlev = sigma_coord.n_levels
    n_sh = grid.n_sh

    # Grid coordinates as numpy for the analytic solution
    lat_np = np.array(grid.lat)        # (n_lat,)
    lon_np = np.array(grid.lon2d[0])   # (n_lon,) -- all rows same longitude
    sigma_full = np.array(sigma_coord.sigma_full)  # (nlev,)

    n_lat = grid.n_lat
    n_lon = grid.n_lon

    # Create 2D lat/lon for each level
    lat_2d = np.broadcast_to(lat_np[:, None], (n_lat, n_lon))
    lon_2d = np.array(grid.lon2d)

    # Allocate 3D fields (n_lat, n_lon, nlev)
    u_3d = np.zeros((n_lat, n_lon, nlev))
    v_3d = np.zeros((n_lat, n_lon, nlev))
    T_3d = np.zeros((n_lat, n_lon, nlev))

    # Compute initial conditions level by level
    for k in range(nlev):
        # Target pressure at this sigma level
        p_target = np.full((n_lat, n_lon), sigma_full[k] * P0)

        # Find height where p(z, lat) = p_target
        z_k = find_z_for_pressure(p_target, lat_2d)

        # Compute temperature at this height
        _, T_k = evaluate_pressure_temperature(z_k, lat_2d)

        # Compute zonal wind from gradient-wind balance
        u_k = compute_zonal_wind(z_k, lat_2d, T_k)

        # Add perturbation if requested
        if perturbed:
            u_k = u_k + exponential_perturbation(lat_2d, lon_2d, z_k)

        u_3d[:, :, k] = u_k
        T_3d[:, :, k] = T_k

    # Convert to JAX float64
    u_jax = jnp.array(u_3d, dtype=jnp.float64)
    v_jax = jnp.array(v_3d, dtype=jnp.float64)
    T_jax = jnp.array(T_3d, dtype=jnp.float64)

    # --- Transform u,v to spectral vorticity/divergence ---
    a = grid.radius
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a

    cos_lat_3d = grid.cos_lat[:, None, None]
    u_cos = u_jax * cos_lat_3d
    v_cos = v_jax * cos_lat_3d

    # vor_hat = (im/a) * SH{v*cos/cos^2} + (1/a) * SH_dmu{u*cos}
    vor_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, v_cos)
        + one_over_a * sh_analysis_dmu_3d(grid, u_cos)
    )
    # div_hat = (im/a) * SH{u*cos/cos^2} - (1/a) * SH_dmu{v*cos}
    div_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, v_cos)
    )

    # Temperature to spectral
    T_hat = sh_analysis_3d(grid, T_jax)

    # Uniform surface pressure (no topography for JW06)
    lnps_grid = jnp.full(
        (n_lat, n_lon), jnp.log(P0), dtype=jnp.float64,
    )
    lnps_hat = sh_analysis(grid, lnps_grid)

    # No topography
    phis_hat = jnp.zeros(n_sh, dtype=jnp.complex128)

    dims_3d = ("spectral", "level")
    dims_2d = ("spectral",)

    # Moist tracers: same RH-tapered q_v init as baroclinic_wave_init_mpas (and
    # the cubed-sphere / lat-lon moist inits) so the balanced state is identical
    # across grids. q_v = rh · q_sat(T, p) · σ², capped at saturation; q_c=q_r=0.
    # Grid-space, passive at t=0; make_kessler_forcing_spectral drives the
    # condensation/precipitation. p is hydrostatic on the uniform p_s=P0 column.
    tracers = None
    if moist:
        from legoesm.thermo import saturation_mixing_ratio

        p_full = jnp.asarray(sigma_full, dtype=jnp.float64)[None, None, :] * P0
        q_sat = saturation_mixing_ratio(T_jax, p_full)
        sigma2 = jnp.asarray(sigma_full, dtype=jnp.float64)[None, None, :] ** 2
        q_v_data = jnp.minimum(rh_init * q_sat * sigma2, q_sat)
        q_zero = jnp.zeros_like(q_v_data)
        tdims = ("lat", "lon", "level")
        tracers = {
            "q_v": Field(data=q_v_data, name="q_v", dims=tdims, units="kg/kg"),
            "q_c": Field(data=q_zero, name="q_c", dims=tdims, units="kg/kg"),
            "q_r": Field(data=q_zero, name="q_r", dims=tdims, units="kg/kg"),
        }

    return SpectralHydrostaticState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        T_hat=Field(data=T_hat, name="T_hat", dims=dims_3d, units="K"),
        lnps_hat=Field(data=lnps_hat, name="lnps_hat", dims=dims_2d, units=""),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
        tracers=tracers,
    )


def build_sharded_baroclinic_wave_state_mpas(
    mesh,
    sigma_coord,
    dev_config,
    perturbed: bool = True,
    moist: bool = False,
    rh_init: float = 0.7,
):
    """Partition-local MPAS baroclinic-wave state (#1100 MPAS twin).

    The multi-device invariant of ``build_sharded_held_suarez_state_atm_latlon``
    for the voronoi lane: every state leaf is created with
    ``jax.make_array_from_callback``, whose callback runs only for the
    cell/edge ranges owned by THIS process's addressable devices — no
    per-process global state build, no global ``device_put``.  The mesh
    itself remains global on every process (its partition-local
    construction through the SFC machinery is the OPEN remainder of
    #1100 — connectivity, not analytic IC).

    Value-identical per shard to
    ``shard_pytree(baroclinic_wave_init_mpas(mesh, sigma, ...), dev_config)``:
    every field is ELEMENTWISE per cell/edge (fixed-iteration bisection,
    analytic wind/temperature/perturbation, saturation taper), so a row
    slice of the global computation equals the same computation on the row
    slice up to shape-specialized codegen.  The REQUIRED contract is a
    few-ULP match (XLA does not guarantee exact bits across differently
    shaped compilations); on the pinned CPU stack the match is measured
    EXACT — both gated by
    ``tests/parallel/test_mpas_partitionlocal_build.py``.  "No global
    build" applies to the O(nCells*nlev) STATE leaves; O(nEntities) 1-D
    temporaries (``cos(angleEdge)``) are still evaluated in full, like the
    global mesh itself.

    Requires ``reorder_voronoi_for_sharding`` padding (nCells and nEdges
    divisible by the device count) — the same precondition
    ``shard_pytree`` has.  ``dev_config.face_sharding is None`` (single
    device) falls back to the global builder unchanged.
    """
    import jax

    from legoesm.core.state import MPASHydrostaticState

    if dev_config.face_sharding is None:
        return baroclinic_wave_init_mpas(
            mesh, sigma_coord,
            perturbed=perturbed, moist=moist, rh_init=rh_init)

    nCells = mesh.nCells
    nEdges = mesh.nEdges
    nlev = sigma_coord.n_levels
    sigma_full = sigma_coord.sigma_full  # (nlev,)

    lat_c = mesh.latCell    # (nCells,)
    lat_e = mesh.latEdge    # (nEdges,)
    lon_e = mesh.lonEdge    # (nEdges,)
    cos_angle = jnp.cos(mesh.angleEdge)  # (nEdges,)

    shard = dev_config.face_sharding     # P("device") on axis 0

    def _levels_block(lat_blk, fn):
        """Stack per-level elementwise results for an entity slice.

        EXACT per-element expressions of baroclinic_wave_init_mpas's level
        loop (zeros + .at[:, k].set) so slices stay bit-identical."""
        n_loc = lat_blk.shape[0]
        out = jnp.zeros((n_loc, nlev))
        for k in range(nlev):
            out = out.at[:, k].set(fn(float(sigma_full[k]), lat_blk))
        return out

    def _T_cb(idx):
        lat_blk = lat_c[idx[0]]

        def _T_level(sig_k, latb):
            p_t = jnp.full(latb.shape, sig_k * P0)
            z = find_z_for_pressure(p_t, latb)
            _, T_k = evaluate_pressure_temperature(z, latb)
            return T_k

        return _levels_block(lat_blk, _T_level)

    def _u_cb(idx):
        lat_blk = lat_e[idx[0]]
        lon_blk = lon_e[idx[0]]
        cos_blk = cos_angle[idx[0]]

        def _u_level(sig_k, latb):
            p_t = jnp.full(latb.shape, sig_k * P0)
            z = find_z_for_pressure(p_t, latb)
            _, T_e = evaluate_pressure_temperature(z, latb)
            u_zonal = compute_zonal_wind(z, latb, T_e)
            if perturbed:
                u_zonal = u_zonal + exponential_perturbation(latb, lon_blk, z)
            return u_zonal * cos_blk

        return _levels_block(lat_blk, _u_level)

    def _ps_cb(idx):
        return jnp.full(lat_c[idx[0]].shape, P0)

    def _phis_cb(idx):
        return jnp.zeros(lat_c[idx[0]].shape)

    def _make(gshape, cb):
        return jax.make_array_from_callback(gshape, shard, cb)

    T_arr = _make((nCells, nlev), _T_cb)
    u_arr = _make((nEdges, nlev), _u_cb)
    ps_arr = _make((nCells,), _ps_cb)
    phis_arr = _make((nCells,), _phis_cb)

    tracers = None
    if moist:
        from legoesm.thermo import saturation_mixing_ratio

        def _qv_cb(idx):
            # Same expression as the global builder, on the cell slice:
            # q_v = min(rh · q_sat(T, p_s·σ) · σ², q_sat).  Recomputes the
            # T block for this slice (elementwise -> bit-identical).
            T_blk = _T_cb(idx)
            ps_blk = _ps_cb(idx)
            p_full = ps_blk[:, None] * sigma_full[None, :]
            q_sat = saturation_mixing_ratio(T_blk, p_full)
            return jnp.minimum(
                rh_init * q_sat * sigma_full[None, :] ** 2, q_sat)

        def _qzero_cb(idx):
            return jnp.zeros((lat_c[idx[0]].shape[0], nlev))

        tracers = {
            "q_v": Field(data=_make((nCells, nlev), _qv_cb), name="q_v",
                         dims=("nCells", "level"), units="kg/kg"),
            "q_c": Field(data=_make((nCells, nlev), _qzero_cb), name="q_c",
                         dims=("nCells", "level"), units="kg/kg"),
            "q_r": Field(data=_make((nCells, nlev), _qzero_cb), name="q_r",
                         dims=("nCells", "level"), units="kg/kg"),
        }

    return MPASHydrostaticState(
        u=Field(data=u_arr, name="u", dims=("nEdges", "level"), units="m/s"),
        T=Field(data=T_arr, name="T", dims=("nCells", "level"), units="K"),
        p_s=Field(data=ps_arr, name="p_s", dims=("nCells",), units="Pa"),
        phis=Field(data=phis_arr, name="phis", dims=("nCells",),
                   units="m^2/s^2"),
        tracers=tracers,
    )
