"""Held-Suarez forcing for dynamical core validation.

The Held-Suarez (1994) benchmark provides a minimal physics package
for testing atmospheric dynamical cores. It replaces full radiation
and boundary-layer parameterizations with:

1. **Newtonian relaxation** of temperature toward a prescribed
   radiative-convective equilibrium profile T_eq(σ, φ).
2. **Rayleigh friction** (linear drag) on winds in the boundary layer.

The expected climate response includes:
- Subtropical jets ~30 m/s at ~200 hPa and ~30° latitude
- Hadley cell circulation
- Midlatitude baroclinic eddies
- Roughly realistic surface temperature distribution

References
----------
- Held, I. M., & Suarez, M. J. (1994). A Proposal for the
  Intercomparison of the Dynamical Cores of Atmospheric General
  Circulation Models. Bull. Amer. Meteor. Soc., 75(10), 1825-1830.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, HydrostaticTendencies
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import SigmaCoordinate, pressure_from_sigma
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

# Reference pressure
P_0 = 1.0e5  # [Pa]


def held_suarez_equilibrium_temperature(
    lat: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute the Held-Suarez equilibrium temperature T_eq.

    T_eq = max(200, (315 - ΔT_y sin²φ - Δθ_z ln(p/p₀) cos²φ) · (p/p₀)^κ)

    Parameters
    ----------
    lat : jax.Array
        Latitude in radians. Can be any shape that broadcasts with p.
    p : jax.Array
        Pressure [Pa]. Same broadcastable shape.

    Returns
    -------
    jax.Array : Equilibrium temperature [K].
    """
    kappa = constants.kappa
    sin_lat = jnp.sin(lat)
    cos_lat = jnp.cos(lat)

    # Pressure ratio
    p_ratio = p / P_0

    # Equilibrium temperature (before min-capping)
    T_eq = (
        (315.0 - DELTA_T_Y * sin_lat**2
         - DELTA_THETA_Z * jnp.log(p_ratio) * cos_lat**2)
        * p_ratio**kappa
    )

    # Cap at minimum temperature
    T_eq = jnp.maximum(T_eq, T_MIN)

    return T_eq


def held_suarez_forcing(
    state: HydrostaticState,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate,
) -> HydrostaticTendencies:
    """Compute Held-Suarez physics tendencies.

    Parameters
    ----------
    state : HydrostaticState
        Current model state.
    grid : CubedSphereGrid
        Horizontal grid (provides latitude).
    sigma_coord : SigmaCoordinate
        Vertical coordinate.

    Returns
    -------
    HydrostaticTendencies : Physics forcing tendencies.
    """
    u = state.u.data       # (6, n, n, nlev)
    v = state.v.data
    T = state.T.data
    p_s = state.p_s.data   # (6, n, n)

    sigma_full = sigma_coord.sigma_full  # (nlev,)
    lat = grid.lat  # (6, n, n)

    # --- Pressure at full levels (for T_eq computation) ---
    p_full = pressure_from_sigma(sigma_full, p_s)  # (6,n,n,nlev)

    # --- Equilibrium temperature ---
    # lat shape (6,n,n) -> broadcast to (6,n,n,nlev)
    T_eq = held_suarez_equilibrium_temperature(
        lat[..., None], p_full
    )  # (6,n,n,nlev)

    # --- Temperature relaxation coefficient k_T(σ, φ) ---
    # k_T = k_a + (k_s - k_a) · max(0, (σ-σ_b)/(1-σ_b)) · cos⁴(φ)
    sigma_factor = jnp.maximum(
        0.0, (sigma_full[None, None, None, :] - SIGMA_B) / (1.0 - SIGMA_B)
    )  # (1,1,1,nlev) -> broadcasts
    cos_lat_4 = jnp.cos(lat)**4  # (6,n,n)

    k_T = K_A + (K_S - K_A) * sigma_factor * cos_lat_4[..., None]

    # --- Newtonian relaxation: Q_T = -k_T · (T - T_eq) ---
    dT_dt_phys = -k_T * (T - T_eq)

    # --- Rayleigh friction coefficient k_v(σ) ---
    # k_v = k_f · max(0, (σ-σ_b)/(1-σ_b))
    k_v = K_F * jnp.maximum(
        0.0, (sigma_full[None, None, None, :] - SIGMA_B) / (1.0 - SIGMA_B)
    )  # (1,1,1,nlev)

    # --- Rayleigh friction: Q_u = -k_v·u, Q_v = -k_v·v ---
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
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate,
    T_init: float = 300.0,
    p_s_init: float = 1.0e5,
    perturbation_amplitude: float = 1.0,
    seed: int = 42,
) -> HydrostaticState:
    """Create initial conditions for the Held-Suarez test.

    Isothermal atmosphere at rest with a small random temperature
    perturbation at the lowest level to break symmetry and trigger
    baroclinic instability.

    Parameters
    ----------
    grid : CubedSphereGrid
        Horizontal grid.
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    T_init : float
        Initial temperature [K].
    p_s_init : float
        Initial surface pressure [Pa].
    perturbation_amplitude : float
        Amplitude of temperature perturbation [K].
    seed : int
        Random seed for perturbation.

    Returns
    -------
    HydrostaticState : Initial state.
    """
    n = grid.n
    nlev = sigma_coord.n_levels
    shape_3d = (6, n, n, nlev)
    shape_2d = (6, n, n)
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    # Uniform temperature
    T_data = jnp.ones(shape_3d) * T_init

    # Add small perturbation at lowest level to break symmetry
    key = jax.random.PRNGKey(seed)
    perturbation = jax.random.normal(key, shape_2d) * perturbation_amplitude
    T_data = T_data.at[..., -1].add(perturbation)

    return HydrostaticState(
        u=Field(data=jnp.zeros(shape_3d), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros(shape_3d), name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_data, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=jnp.ones(shape_2d) * p_s_init, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=dims_2d, units="m^2/s^2"),
    )
