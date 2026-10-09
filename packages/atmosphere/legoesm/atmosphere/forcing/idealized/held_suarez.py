"""Held-Suarez forcing and initialization for all grid types.

The Held-Suarez (1994) benchmark provides a minimal physics package
for testing atmospheric dynamical cores. It replaces full radiation
and boundary-layer parameterizations with:

1. **Newtonian relaxation** of temperature toward a prescribed
   radiative-convective equilibrium profile T_eq(sigma, phi).
2. **Rayleigh friction** (linear drag) on winds in the boundary layer.

Supported grids: cubed-sphere, lat-lon, MPAS Voronoi, spectral (Gaussian).

References
----------
- Held, I. M., & Suarez, M. J. (1994). A Proposal for the
  Intercomparison of the Dynamical Cores of Atmospheric General
  Circulation Models. Bull. Amer. Meteor. Soc., 75(10), 1825-1830.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralHydrostaticState,
    spectral_pe_to_grid,
)
from legoesm.core.field import Field
from legoesm.core.precision import get_policy
from legoesm.core.state import (
    HydrostaticState,
    HydrostaticTendencies,
    MPASHydrostaticState,
    MPASHydrostaticTendencies,
)
from legoesm.grids.gaussian import (
    sh_analysis_3d,
    sh_analysis_oc2_3d,
    sh_analysis_dmu_3d,
)
from legoesm.grids.vertical import (
    SigmaCoordinate,
    HybridSigmaPressureCoordinate,
    pressure_from_sigma,
    pressure_from_hybrid,
)
from legoesm import constants


# ==============================================================================
# Held-Suarez parameters (Table 1 of Held & Suarez 1994)
# ==============================================================================

# Temperature relaxation timescales
K_A = 1.0 / (40.0 * 86400.0)   # 1/40 day in [1/s] (free atmosphere)
K_S = 1.0 / (4.0 * 86400.0)    # 1/4 day in [1/s] (surface)

# Rayleigh friction timescale
K_F = 1.0 / (1.0 * 86400.0)    # 1 day in [1/s]

# Boundary layer threshold
SIGMA_B = 0.7

# Equilibrium temperature parameters
DELTA_T_Y = 60.0    # [K] meridional temperature gradient
DELTA_THETA_Z = 10.0  # [K] vertical potential temperature gradient
T_MIN = 200.0        # [K] minimum equilibrium temperature

# Reference pressure (alias for constants.p_ref kept for local readability)
P_0 = constants.p_ref


# ==============================================================================
# Shared physics
# ==============================================================================

def held_suarez_equilibrium_temperature(
    lat: jax.Array,
    p: jax.Array,
    delta_T_y: float = DELTA_T_Y,
    delta_theta_z: float = DELTA_THETA_Z,
    T_min: float = T_MIN,
    p_ref: float = P_0,
) -> jax.Array:
    """Compute the Held-Suarez equilibrium temperature T_eq.

    T_eq = max(T_min,
               (315 - delta_T_y sin^2(phi)
                    - delta_theta_z ln(p/p_ref) cos^2(phi)) * (p/p_ref)^kappa)

    Parameters
    ----------
    lat : jax.Array
        Latitude in radians. Can be any shape that broadcasts with p.
    p : jax.Array
        Pressure [Pa]. Same broadcastable shape.
    delta_T_y, delta_theta_z, T_min, p_ref : float
        Optional overrides of the module-level Held-Suarez parameters.
        Default to ``DELTA_T_Y``, ``DELTA_THETA_Z``, ``T_MIN``, ``P_0``
        respectively.  Passing them as traced JAX scalars makes them
        reachable by ``jax.grad`` for parameter estimation.

    Returns
    -------
    jax.Array : Equilibrium temperature [K].
    """
    kappa = constants.kappa
    sin_lat = jnp.sin(lat)
    cos_lat = jnp.cos(lat)

    # Pressure ratio
    p_ratio = p / p_ref

    # Equilibrium temperature (before min-capping)
    T_eq = (
        (315.0 - delta_T_y * sin_lat**2
         - delta_theta_z * jnp.log(p_ratio) * cos_lat**2)
        * p_ratio**kappa
    )

    # Cap at minimum temperature
    T_eq = jnp.maximum(T_eq, T_min)

    return T_eq


def held_suarez_sigma_factor(sigma: jax.Array, sigma_b: float = SIGMA_B) -> jax.Array:
    """Boundary-layer weight ``max(0, (σ − σ_b)/(1 − σ_b))`` used by k_T and k_v."""
    return jnp.maximum(0.0, (sigma - sigma_b) / (1.0 - sigma_b))


def held_suarez_temperature_tendency(
    T: jax.Array,
    lat: jax.Array,
    p: jax.Array,
    sigma: jax.Array,
    *,
    k_a: float = K_A,
    k_s: float = K_S,
    sigma_b: float = SIGMA_B,
    delta_T_y: float = DELTA_T_Y,
    delta_theta_z: float = DELTA_THETA_Z,
    T_min: float = T_MIN,
    p_ref: float = P_0,
) -> jax.Array:
    """Held-Suarez Newtonian relaxation ``−k_T(σ, φ)·(T − T_eq)`` [K/s].

    ``k_T = k_a + (k_s − k_a)·max(0, (σ−σ_b)/(1−σ_b))·cos⁴φ``.  ``lat``
    [rad], ``p`` [Pa] and ``sigma`` must broadcast against ``T``.  The single
    implementation behind every grid's forcing (and the driver's HS lane).
    """
    T_eq = held_suarez_equilibrium_temperature(
        lat, p, delta_T_y=delta_T_y, delta_theta_z=delta_theta_z,
        T_min=T_min, p_ref=p_ref,
    )
    k_T = k_a + (k_s - k_a) * held_suarez_sigma_factor(sigma, sigma_b) * jnp.cos(lat) ** 4
    return -k_T * (T - T_eq)


# ==============================================================================
# Cubed-sphere forcing and initialization
# ==============================================================================

def held_suarez_forcing(
    state: HydrostaticState,
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    k_a: float = K_A,
    k_s: float = K_S,
    k_f: float = K_F,
    sigma_b: float = SIGMA_B,
    delta_T_y: float = DELTA_T_Y,
    delta_theta_z: float = DELTA_THETA_Z,
    T_min: float = T_MIN,
    p_ref: float = P_0,
) -> HydrostaticTendencies:
    """Compute Held-Suarez physics tendencies on a cubed-sphere grid.

    Parameters
    ----------
    state : HydrostaticState
        Current model state.
    grid : CubedSphereGrid
        Horizontal grid (provides latitude).
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
        Vertical coordinate.
    k_a, k_s, k_f, sigma_b, delta_T_y, delta_theta_z, T_min, p_ref : float
        Held-Suarez tunable parameters with the module-level defaults
        from Held & Suarez 1994 Table 1.  Each can be passed as a
        traced JAX scalar so ``jax.grad`` reaches them for parameter
        estimation (issue #249-style AD coverage).

    Returns
    -------
    HydrostaticTendencies : Physics forcing tendencies.
    """
    u = state.u.data       # (6, n, n, nlev)
    v = state.v.data
    T = state.T.data
    p_s = state.p_s.data   # (6, n, n)

    lat = grid.lat  # (6, n, n)
    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    # --- Pressure at full levels (for T_eq computation) ---
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)  # (6,n,n,nlev)
        sigma_eff = p_full / jnp.maximum(p_s[..., None], 1.0)
    else:
        sigma_full = sigma_coord.sigma_full  # (nlev,)
        p_full = pressure_from_sigma(sigma_full, p_s)  # (6,n,n,nlev)
        sigma_eff = jnp.broadcast_to(sigma_full[None, None, None, :], T.shape)

    # --- Newtonian relaxation (lat (6,n,n) -> (6,n,n,nlev)) ---
    dT_dt_phys = held_suarez_temperature_tendency(
        T, lat[..., None], p_full, sigma_eff,
        k_a=k_a, k_s=k_s, sigma_b=sigma_b, delta_T_y=delta_T_y,
        delta_theta_z=delta_theta_z, T_min=T_min, p_ref=p_ref,
    )

    # --- Rayleigh friction coefficient k_v(sigma) ---
    k_v = k_f * held_suarez_sigma_factor(sigma_eff, sigma_b)

    # --- Rayleigh friction: Q_u = -k_v*u, Q_v = -k_v*v ---
    du_dt_phys = -k_v * u
    dv_dt_phys = -k_v * v

    # --- Build tendency pytree ---
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    return HydrostaticTendencies(
        du_dt=Field(data=du_dt_phys, name="du_dt_phys", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt_phys, name="dv_dt_phys", dims=dims_3d, units="m/s^2"),
        dT_dt=Field(data=dT_dt_phys, name="dT_dt_phys", dims=dims_3d, units="K/s"),
        dp_s_dt=Field(
            data=jnp.zeros_like(p_s), name="dp_s_dt_phys", dims=dims_2d, units="Pa/s"
        ),
        dphis_dt=Field(
            data=jnp.zeros_like(p_s), name="dphis_dt_phys", dims=dims_2d, units="m^2/s^3"
        ),
    )


def held_suarez_init(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    T_init: float = 300.0,
    p_s_init: float = constants.p_ref,
    perturbation_amplitude: float = 1.0,
    seed: int = 42,
    phis: jnp.ndarray | None = None,
) -> HydrostaticState:
    """Create initial conditions for the Held-Suarez test on a cubed-sphere grid.

    Isothermal atmosphere at rest with a small random temperature
    perturbation at the lowest level to break symmetry and trigger
    baroclinic instability.

    Parameters
    ----------
    grid : CubedSphereGrid
        Horizontal grid.
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
        Vertical coordinate.
    T_init : float
        Initial temperature [K].
    p_s_init : float
        Initial surface pressure [Pa].
    perturbation_amplitude : float
        Amplitude of temperature perturbation [K].
    seed : int
        Random seed for perturbation.
    phis : jnp.ndarray or None
        Surface geopotential [m^2/s^2], shape (6, n, n). If None, flat
        terrain is used (phis=0). When provided, surface pressure is
        reduced hydrostatically: p_s = p_s_init * exp(-phis / (R_d * T_init)).

    Returns
    -------
    HydrostaticState : Initial state.
    """
    _dtype = get_policy().storage

    n = grid.n
    nlev = sigma_coord.n_levels
    shape_3d = (6, n, n, nlev)
    shape_2d = (6, n, n)
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    # Surface geopotential
    if phis is None:
        phis_data = jnp.zeros(shape_2d, dtype=_dtype)
    else:
        phis_data = jnp.asarray(phis, dtype=_dtype)

    # Surface pressure (hydrostatic adjustment for topography)
    p_s_data = (p_s_init * jnp.exp(-phis_data / (constants.R_d * T_init))).astype(_dtype)

    # Uniform temperature
    T_data = jnp.ones(shape_3d, dtype=_dtype) * T_init

    # Add small perturbation at lowest level to break symmetry
    key = jax.random.PRNGKey(seed)
    perturbation = jax.random.normal(key, shape_2d, dtype=_dtype) * jnp.asarray(
        perturbation_amplitude, dtype=_dtype
    )
    T_data = T_data.at[..., -1].add(perturbation)

    return HydrostaticState(
        u=Field(data=jnp.zeros(shape_3d, dtype=_dtype), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros(shape_3d, dtype=_dtype), name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_data, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s_data, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=phis_data, name="phis", dims=dims_2d, units="m^2/s^2"),
    )


# ==============================================================================
# Lat-lon forcing and initialization
# ==============================================================================

def held_suarez_forcing_latlon(
    state: HydrostaticState,
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    k_a: float = K_A,
    k_s: float = K_S,
    k_f: float = K_F,
    sigma_b: float = SIGMA_B,
    delta_T_y: float = DELTA_T_Y,
    delta_theta_z: float = DELTA_THETA_Z,
    T_min: float = T_MIN,
    p_ref: float = P_0,
) -> HydrostaticTendencies:
    """Compute Held-Suarez physics tendencies on a lat-lon grid.

    Parameters
    ----------
    state : HydrostaticState
        Current model state.
    grid : LatLonGrid
        Horizontal grid (provides latitude).
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
        Vertical coordinate.
    k_a, k_s, k_f, sigma_b, delta_T_y, delta_theta_z, T_min, p_ref : float
        Held-Suarez parameters, as in :func:`held_suarez_forcing`.

    Returns
    -------
    HydrostaticTendencies : Physics forcing tendencies.
    """
    u = state.u.data       # (n_lat, n_lon, nlev)
    v = state.v.data
    T = state.T.data
    p_s = state.p_s.data   # (n_lat, n_lon)

    lat = grid.lat  # (n_lat,)
    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    # Pressure at full levels
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)  # (n_lat, n_lon, nlev)
        sigma_eff = p_full / jnp.maximum(p_s[..., None], 1.0)
    else:
        sigma_full = sigma_coord.sigma_full  # (nlev,)
        p_full = pressure_from_sigma(sigma_full, p_s)  # (n_lat, n_lon, nlev)
        sigma_eff = jnp.broadcast_to(sigma_full[None, None, :], T.shape)

    # Newtonian relaxation (lat (n_lat,) -> (n_lat, 1, 1))
    dT_dt_phys = held_suarez_temperature_tendency(
        T, lat[:, None, None], p_full, sigma_eff,
        k_a=k_a, k_s=k_s, sigma_b=sigma_b, delta_T_y=delta_T_y,
        delta_theta_z=delta_theta_z, T_min=T_min, p_ref=p_ref,
    )

    # Rayleigh friction coefficient k_v(sigma)
    k_v = k_f * held_suarez_sigma_factor(sigma_eff, sigma_b)

    du_dt_phys = -k_v * u
    dv_dt_phys = -k_v * v

    # Build tendency pytree
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    return HydrostaticTendencies(
        du_dt=Field(data=du_dt_phys, name="du_dt_phys", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt_phys, name="dv_dt_phys", dims=dims_3d, units="m/s^2"),
        dT_dt=Field(data=dT_dt_phys, name="dT_dt_phys", dims=dims_3d, units="K/s"),
        dp_s_dt=Field(
            data=jnp.zeros_like(p_s), name="dp_s_dt_phys", dims=dims_2d, units="Pa/s"
        ),
        dphis_dt=Field(
            data=jnp.zeros_like(p_s), name="dphis_dt_phys", dims=dims_2d, units="m^2/s^3"
        ),
    )


def held_suarez_init_latlon(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    T_init: float = 300.0,
    p_s_init: float = constants.p_ref,
    perturbation_amplitude: float = 1.0,
    seed: int = 42,
    phis: jnp.ndarray | None = None,
) -> HydrostaticState:
    """Create initial conditions for the Held-Suarez test on a lat-lon grid.

    Isothermal atmosphere at rest with a small random temperature
    perturbation at the lowest level to break symmetry.

    Parameters
    ----------
    grid : LatLonGrid
        Horizontal grid.
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
        Vertical coordinate.
    T_init : float
        Initial temperature [K].
    p_s_init : float
        Initial surface pressure [Pa].
    perturbation_amplitude : float
        Amplitude of temperature perturbation [K].
    seed : int
        Random seed for perturbation.
    phis : jnp.ndarray or None
        Surface geopotential [m^2/s^2], shape (n_lat, n_lon). If None,
        flat terrain is used. When provided, surface pressure is reduced
        hydrostatically: p_s = p_s_init * exp(-phis / (R_d * T_init)).

    Returns
    -------
    HydrostaticState : Initial state.
    """
    _dtype = get_policy().storage

    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = sigma_coord.n_levels
    shape_3d = (n_lat, n_lon, nlev)
    shape_2d = (n_lat, n_lon)
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    # Surface geopotential
    if phis is None:
        phis_data = jnp.zeros(shape_2d, dtype=_dtype)
    else:
        phis_data = jnp.asarray(phis, dtype=_dtype)

    # Surface pressure (hydrostatic adjustment for topography)
    p_s_data = (p_s_init * jnp.exp(-phis_data / (constants.R_d * T_init))).astype(_dtype)

    # Uniform temperature
    T_data = jnp.ones(shape_3d, dtype=_dtype) * T_init

    # Small perturbation at lowest level
    key = jax.random.PRNGKey(seed)
    perturbation = jax.random.normal(key, shape_2d, dtype=_dtype) * jnp.asarray(
        perturbation_amplitude, dtype=_dtype
    )
    T_data = T_data.at[:, :, -1].add(perturbation)

    return HydrostaticState(
        u=Field(data=jnp.zeros(shape_3d, dtype=_dtype), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros(shape_3d, dtype=_dtype), name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_data, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s_data, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=phis_data, name="phis", dims=dims_2d, units="m^2/s^2"),
    )


# ==============================================================================
# MPAS Voronoi mesh forcing and initialization
# ==============================================================================

def held_suarez_forcing_mpas(
    state,
    mesh,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    phys_state=None,
    forcing=None,
    k_a: float = K_A,
    k_s: float = K_S,
    k_f: float = K_F,
    sigma_b: float = SIGMA_B,
    delta_T_y: float = DELTA_T_Y,
    delta_theta_z: float = DELTA_THETA_Z,
    T_min: float = T_MIN,
    p_ref: float = P_0,
):
    """Compute Held-Suarez physics tendencies on an MPAS mesh.

    Parameters
    ----------
    state : MPASHydrostaticState
        u (nEdges, nlev), T (nCells, nlev), p_s (nCells,).
    mesh : VoronoiMesh
        Provides latCell, latEdge.
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
        Vertical coordinate.
    phys_state, forcing : optional
        Accepted to satisfy the MPAS hydrostatic dycore's operator-split
        ``physics_fn(state, mesh, sigma_coord, *, phys_state, forcing)``
        calling convention (``primitive_eq_mpas.py``).  Held-Suarez is a
        stateless Newtonian relaxation with no prognostic physics carry and
        no external forcing, so both are ignored; the bare-tendencies return
        (not a ``(tendencies, phys_state_out)`` tuple) leaves the dycore's
        ``phys_state`` carry untouched.
    k_a, k_s, k_f, sigma_b, delta_T_y, delta_theta_z, T_min, p_ref : float
        Held-Suarez parameters, as in :func:`held_suarez_forcing`.

    Returns
    -------
    MPASHydrostaticTendencies
    """
    u = state.u.data       # (nEdges, nlev)
    T = state.T.data       # (nCells, nlev)
    p_s = state.p_s.data   # (nCells,)

    lat_cell = mesh.latCell  # (nCells,)

    # Pressure at cell centers
    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)  # (nCells, nlev)
        sigma_eff = p_full / jnp.maximum(p_s[:, None], 1.0)
    else:
        sigma_full = sigma_coord.sigma_full  # (nlev,)
        p_full = pressure_from_sigma(sigma_full, p_s)    # (nCells, nlev)
        sigma_eff = jnp.broadcast_to(sigma_full[None, :], T.shape)

    # Newtonian relaxation at cell centers (nCells, nlev)
    dT_dt = held_suarez_temperature_tendency(
        T, lat_cell[:, None], p_full, sigma_eff,
        k_a=k_a, k_s=k_s, sigma_b=sigma_b, delta_T_y=delta_T_y,
        delta_theta_z=delta_theta_z, T_min=T_min, p_ref=p_ref,
    )

    # Rayleigh friction coefficient k_v(sigma) at edge locations
    c0 = mesh.cellsOnEdge[0]  # (nEdges,)
    c1 = mesh.cellsOnEdge[1]  # (nEdges,)
    if _hybrid:
        p_edge = 0.5 * (p_full[c0] + p_full[c1])        # (nEdges, nlev)
        ps_edge = 0.5 * (p_s[c0] + p_s[c1])             # (nEdges,)
        sigma_edge = p_edge / ps_edge[:, None]
    else:
        sigma_edge = jnp.broadcast_to(sigma_full[None, :], u.shape)

    k_v = k_f * held_suarez_sigma_factor(sigma_edge, sigma_b)  # (nEdges, nlev)

    # Rayleigh friction on normal velocity
    du_dt = -k_v * u  # (nEdges, nlev)

    # Construct tendencies
    dims_edge = ("nEdges", "level")
    dims_cell = ("nCells", "level")
    dims_cell_2d = ("nCells",)

    return MPASHydrostaticTendencies(
        du_dt=Field(data=du_dt, name="du_dt_phys", dims=dims_edge, units="m/s^2"),
        dT_dt=Field(data=dT_dt, name="dT_dt_phys", dims=dims_cell, units="K/s"),
        dp_s_dt=Field(
            data=jnp.zeros_like(p_s), name="dp_s_dt_phys",
            dims=dims_cell_2d, units="Pa/s",
        ),
        dphis_dt=Field(
            data=jnp.zeros_like(p_s), name="dphis_dt_phys",
            dims=dims_cell_2d, units="m^2/s^3",
        ),
    )


# NB: Held-Suarez is NOT tagged column-local.  Its Rayleigh friction is
# column-local in the sigma branch, but the HYBRID-coordinate branch computes
# the edge sigma from adjacent CELL pressures (p_full[c0]/p_full[c1],
# p_s[c0]/p_s[c1]) — for an owned edge whose neighbor cell is a halo, that needs
# fresh halo p_s.  Skipping the pre-physics halo exchange would use stale halo
# values for owned-edge wind (codex review).  So we keep the exchange for HS.


def held_suarez_init_mpas(
    mesh,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    T_init: float = 300.0,
    p_s_init: float = constants.p_ref,
    perturbation_amplitude: float = 1.0,
    seed: int = 42,
    phis: jnp.ndarray | None = None,
):
    """Create isothermal rest-state initial conditions for Held-Suarez on MPAS.

    Parameters
    ----------
    mesh : VoronoiMesh
        MPAS Voronoi mesh.
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
        Vertical coordinate.
    T_init : float
        Initial temperature [K].
    p_s_init : float
        Initial surface pressure [Pa].
    perturbation_amplitude : float
        Amplitude of random T perturbation at lowest level [K].
    seed : int
        Random seed.
    phis : jnp.ndarray or None
        Surface geopotential [m^2/s^2], shape (nCells,). If None, flat
        terrain is used. When provided, surface pressure is reduced
        hydrostatically: p_s = p_s_init * exp(-phis / (R_d * T_init)).

    Returns
    -------
    MPASHydrostaticState
    """
    _dtype = get_policy().storage

    nCells = mesh.nCells
    nEdges = mesh.nEdges
    nlev = sigma_coord.n_levels

    # Surface geopotential. Thread the precision-policy storage dtype through
    # EVERY field (matching held_suarez_init / _latlon) so a mixed-precision
    # policy does not leave the MPAS HS state's dtypes disagreeing with the
    # policy storage (which reproduces the float64-config-through-float32-state
    # scan-carry mismatch).
    if phis is None:
        phis_data = jnp.zeros((nCells,), dtype=_dtype)
    else:
        phis_data = jnp.asarray(phis, dtype=_dtype)

    # Surface pressure (hydrostatic adjustment for topography)
    p_s_data = (
        p_s_init * jnp.exp(-phis_data / (constants.R_d * T_init))
    ).astype(_dtype)

    # Temperature: uniform with small perturbation at lowest level
    T_data = jnp.full((nCells, nlev), T_init, dtype=_dtype)
    key = jax.random.PRNGKey(seed)
    perturbation = jax.random.normal(key, (nCells,), dtype=_dtype) * jnp.asarray(
        perturbation_amplitude, dtype=_dtype
    )
    T_data = T_data.at[:, -1].add(perturbation)

    # Velocity: at rest
    u_data = jnp.zeros((nEdges, nlev), dtype=_dtype)

    return MPASHydrostaticState(
        u=Field(data=u_data, name="u", dims=("nEdges", "level"), units="m/s"),
        T=Field(data=T_data, name="T", dims=("nCells", "level"), units="K"),
        p_s=Field(data=p_s_data, name="p_s", dims=("nCells",), units="Pa"),
        phis=Field(data=phis_data, name="phis", dims=("nCells",), units="m^2/s^2"),
    )


# ==============================================================================
# Spectral (Gaussian grid) forcing
# ==============================================================================

def held_suarez_forcing_spectral(
    state,
    grid,
    sigma_coord,
    *,
    k_a: float = K_A,
    k_s: float = K_S,
    k_f: float = K_F,
    sigma_b: float = SIGMA_B,
    delta_T_y: float = DELTA_T_Y,
    delta_theta_z: float = DELTA_THETA_Z,
    T_min: float = T_MIN,
    p_ref: float = P_0,
):
    """Compute Held-Suarez physics tendencies for the spectral PE.

    Transforms the spectral state to grid space, computes Newtonian
    relaxation and Rayleigh friction on the Gaussian grid, then
    transforms the wind tendencies back to spectral vorticity/divergence
    tendencies.

    Parameters
    ----------
    state : SpectralHydrostaticState
        Current spectral model state.
    grid : GaussianGrid
        Gaussian grid with SH transform matrices.
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    k_a, k_s, k_f, sigma_b, delta_T_y, delta_theta_z, T_min, p_ref : float
        Held-Suarez parameters, as in :func:`held_suarez_forcing`.

    Returns
    -------
    SpectralHydrostaticState
        Physics tendencies in spectral space (same pytree structure).
    """
    # --- 1. Transform state to grid space ---
    fields = spectral_pe_to_grid(state, grid, sigma_coord)
    u = fields['u']         # (n_lat, n_lon, nlev)
    v = fields['v']
    T = fields['T']
    p_s = fields['p_s']     # (n_lat, n_lon)

    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)
    lat = grid.lat  # (n_lat,)

    # --- 2. Pressure at full levels ---
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)  # (n_lat, n_lon, nlev)
        sigma_eff = p_full / jnp.maximum(p_s[..., None], 1.0)
    else:
        sigma_full = sigma_coord.sigma_full  # (nlev,)
        p_full = p_s[..., None] * sigma_full  # (n_lat, n_lon, nlev)
        sigma_eff = jnp.broadcast_to(sigma_full[None, None, :], T.shape)

    # --- 3-5. Newtonian relaxation (n_lat, n_lon, nlev) ---
    dT_dt_phys = held_suarez_temperature_tendency(
        T, lat[:, None, None], p_full, sigma_eff,
        k_a=k_a, k_s=k_s, sigma_b=sigma_b, delta_T_y=delta_T_y,
        delta_theta_z=delta_theta_z, T_min=T_min, p_ref=p_ref,
    )

    # --- 6. Rayleigh friction coefficient k_v(sigma) ---
    k_v = k_f * held_suarez_sigma_factor(sigma_eff, sigma_b)

    # --- 7. Rayleigh friction: du/dt = -k_v*u, dv/dt = -k_v*v ---
    du_dt_phys = -k_v * u
    dv_dt_phys = -k_v * v

    # --- 8. Convert wind tendencies to spectral vor/div tendencies ---
    a = grid.radius
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a

    # Multiply by cos(lat) for the spectral transform
    cos_lat_3d = grid.cos_lat[:, None, None]  # (n_lat, 1, 1)
    du_cos = du_dt_phys * cos_lat_3d
    dv_cos = dv_dt_phys * cos_lat_3d

    # curl(du,dv) -> dvor_hat (vorticity tendency)
    dvor_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, dv_cos)
        + one_over_a * sh_analysis_dmu_3d(grid, du_cos)
    )

    # div(du,dv) -> ddiv_hat (divergence tendency)
    ddiv_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, du_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, dv_cos)
    )

    # --- 9. Transform T tendency to spectral ---
    dT_hat = sh_analysis_3d(grid, dT_dt_phys)

    # --- 10. No surface pressure tendency from HS forcing ---
    dlnps_hat = jnp.zeros_like(state.lnps_hat.data)

    # Build physics tendency pytree
    return SpectralHydrostaticState(
        vor_hat=state.vor_hat.replace(data=dvor_hat),
        div_hat=state.div_hat.replace(data=ddiv_hat),
        T_hat=state.T_hat.replace(data=dT_hat),
        lnps_hat=state.lnps_hat.replace(data=dlnps_hat),
        phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
    )
