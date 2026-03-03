"""Baroclinic wave test case for dynamical core validation (DCMIP 2016).

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
from legoesm.grids.cubed_sphere import CubedSphereGrid, rotate_winds_geo_to_grid
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
    ≈ 4e-14 m.

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
# Initialization
# ==============================================================================

def baroclinic_wave_init(
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate,
    perturbed: bool = True,
) -> HydrostaticState:
    """Create initial conditions for the baroclinic wave test.

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

    Returns
    -------
    HydrostaticState
        Initial state for the baroclinic wave test.
    """
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
    # This is essential: on faces where the grid x-axis is not aligned with
    # east, the geographic u_east projects onto both u_grid and v_grid.
    v_north = jnp.zeros((6, n, n, nlev))

    # Rotate at each level using the grid angle (angle between x-axis and east)
    # rotate_winds_geo_to_grid expects (6,n,n) arrays, so we loop over levels
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

    return HydrostaticState(
        u=Field(data=u_grid, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_grid, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=jnp.zeros((6, n, n)),
                   name="phis", dims=dims_2d, units="m^2/s^2"),
    )
