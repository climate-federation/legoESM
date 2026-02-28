"""Kessler warm-rain microphysics for legoESM.

A simple one-moment warm-rain scheme that tracks three moisture species:
- q_vapor (tracers[..., 0]): water vapor mixing ratio [kg/kg]
- q_cloud (tracers[..., 1]): cloud water mixing ratio [kg/kg]
- q_rain  (tracers[..., 2]): rain water mixing ratio [kg/kg]

Processes:
1. Saturation adjustment: condensation/evaporation to maintain q_sat
2. Autoconversion: cloud water -> rain when q_cloud > threshold
3. Accretion: cloud water collected by falling rain
4. Evaporation: rain evaporates in subsaturated air
5. Rain sedimentation: terminal velocity fallout
6. Latent heating: feedback to potential temperature

All operations use smooth (differentiable) approximations for
compatibility with jax.grad.

References
----------
- Kessler (1969): On the Distribution and Continuity of Water
  Substance in Atmospheric Circulations.
- Klemp & Wilhelmson (1978): The Simulation of Three-Dimensional
  Convective Storm Dynamics.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState, NonHydrostaticTendencies
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm import constants


class KesslerConfig(NamedTuple):
    """Configuration for Kessler microphysics."""
    autoconversion_threshold: float = 1.0e-3   # q_c threshold [kg/kg]
    autoconversion_rate: float = 1.0e-3         # Rate [1/s]
    accretion_coeff: float = 2.2                # Collection coefficient
    evaporation_coeff: float = 1.0              # Evaporation coefficient
    rain_fall_speed: float = 5.0                # Terminal velocity [m/s]
    saturation_sharpness: float = 100.0         # Smooth switch sharpness


# ==============================================================================
# Thermodynamic utilities
# ==============================================================================

def saturation_mixing_ratio(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute saturation mixing ratio using Tetens formula.

    e_sat = 611.2 * exp(17.67 * (T - 273.15) / (T - 29.65))
    q_sat = epsilon * e_sat / (p - e_sat)

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Saturation mixing ratio [kg/kg].
    """
    T_c = T - constants.T_freeze  # Celsius
    e_sat = 611.2 * jnp.exp(17.67 * T_c / (T_c + 243.5))
    # Clip to avoid division by zero
    denom = jnp.clip(p - e_sat, 1.0, None)
    return constants.epsilon * e_sat / denom


def temperature_from_theta(
    theta: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Recover temperature from potential temperature and pressure.

    T = theta * (p / p_0)^kappa

    Parameters
    ----------
    theta : jax.Array
        Potential temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Temperature [K].
    """
    return theta * (p / constants.p_ref) ** constants.kappa


def pressure_from_eos(
    rho: jax.Array,
    theta: jax.Array,
) -> jax.Array:
    """Compute pressure from density and potential temperature.

    p = p_0 * (R_d * rho * theta / p_0)^(c_p / c_v)

    Parameters
    ----------
    rho : jax.Array
        Density [kg/m^3].
    theta : jax.Array
        Potential temperature [K].

    Returns
    -------
    jax.Array
        Pressure [Pa].
    """
    R_d = constants.R_d
    c_p = constants.c_pd
    c_v = constants.c_vd
    p_0 = constants.p_ref

    return p_0 * (R_d * rho * theta / p_0) ** (c_p / c_v)


# ==============================================================================
# Kessler microphysics tendencies
# ==============================================================================

def kessler_tendencies(
    state: NonHydrostaticState,
    grid: CubedSphereGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    config: KesslerConfig = KesslerConfig(),
) -> NonHydrostaticTendencies:
    """Compute Kessler microphysics tendencies.

    Expects tracers[..., 0] = q_vapor, [..1] = q_cloud, [..2] = q_rain.

    Parameters
    ----------
    state : NonHydrostaticState
    grid : CubedSphereGrid
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    config : KesslerConfig

    Returns
    -------
    NonHydrostaticTendencies
        Tendencies from microphysics. Only dtheta_prime_dt and
        dtracers_dt are nonzero.
    """
    tracers = state.tracers.data  # (6, n, n, nlev, 3)
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data

    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    theta_total = theta_0 + theta_p
    rho_total = rho_0 + rho_p

    q_v = tracers[..., 0]  # vapor
    q_c = tracers[..., 1]  # cloud
    q_r = tracers[..., 2]  # rain

    # Pressure and temperature from equation of state
    p = pressure_from_eos(rho_total, theta_total)
    T = temperature_from_theta(theta_total, p)

    # Saturation mixing ratio
    q_sat = saturation_mixing_ratio(T, p)

    # --- 1. Saturation adjustment ---
    # Condensation/evaporation to bring q_v toward q_sat
    # Using smooth sigmoid instead of hard threshold
    excess = q_v - q_sat
    sharpness = config.saturation_sharpness
    cond_frac = jax.nn.sigmoid(sharpness * excess)
    # Condensation rate: positive when supersaturated
    condensation = cond_frac * excess  # instantaneous adjustment

    dq_v_sat = -condensation
    dq_c_sat = condensation

    # Latent heating: dtheta/dt = L_v * condensation / (c_p * exner)
    exner = (p / constants.p_ref) ** constants.kappa
    dtheta_latent = constants.L_v * condensation / (constants.c_pd * exner)

    # --- 2. Autoconversion: cloud -> rain ---
    q_c_excess = jnp.clip(q_c + dq_c_sat - config.autoconversion_threshold, 0.0, None)
    autoconv = config.autoconversion_rate * q_c_excess

    # --- 3. Accretion: cloud collected by rain ---
    accretion = config.accretion_coeff * q_c * jnp.clip(q_r, 0.0, None) ** 0.875

    # --- 4. Evaporation of rain ---
    subsaturation = jnp.clip(q_sat - q_v, 0.0, None) / jnp.clip(q_sat, 1e-10, None)
    evaporation = config.evaporation_coeff * subsaturation * jnp.clip(q_r, 0.0, None) ** 0.525

    # --- 5. Rain sedimentation (tendency from vertical flux divergence) ---
    # Terminal velocity: V_t = V_0 * (rho_0/rho)^0.5
    rho_sfc = rho_0[-1]
    V_t = config.rain_fall_speed * jnp.sqrt(rho_sfc / jnp.clip(rho_total, 0.1, None))

    # Vertical flux divergence: -d(V_t * q_r * rho) / (rho * dz)
    # Simple upwind: flux at interface = V_t * q_r from above
    q_r_total = jnp.clip(q_r, 0.0, None)
    flux = V_t * q_r_total * rho_total  # (6,n,n,nlev)

    # Flux at interfaces (from above to below)
    dz = height_coord.dz
    J = terrain_metric.jacobian
    dz_phys = dz * J[..., None]

    # Sedimentation tendency
    flux_in = jnp.concatenate([jnp.zeros((*flux.shape[:3], 1)), flux[..., :-1]], axis=-1)
    sed_tend = (flux_in - flux) / (rho_total * jnp.clip(dz_phys, 1.0, None))

    # --- Combine tracer tendencies ---
    dq_v_dt = dq_v_sat + evaporation
    dq_c_dt = dq_c_sat - autoconv - accretion
    dq_r_dt = autoconv + accretion - evaporation + sed_tend

    dtracers = jnp.stack([dq_v_dt, dq_c_dt, dq_r_dt], axis=-1)

    # --- Build tendency ---
    shape_3d = theta_p.shape
    shape_w = state.w.data.shape
    shape_2d = state.phis.data.shape

    zero_3d = jnp.zeros(shape_3d)
    zero_w = jnp.zeros(shape_w)
    zero_2d = jnp.zeros(shape_2d)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    dims_tr = ("face", "x", "y", "level", "tracer")

    return NonHydrostaticTendencies(
        du_dt=Field(data=zero_3d, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=zero_3d, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dw_dt=Field(data=zero_w, name="dw_dt", dims=dims_w, units="m/s^2"),
        dtheta_prime_dt=Field(
            data=dtheta_latent, name="dtheta_prime_dt", dims=dims_3d, units="K/s",
        ),
        drho_prime_dt=Field(
            data=zero_3d, name="drho_prime_dt", dims=dims_3d, units="kg/m^3/s",
        ),
        dphis_dt=Field(
            data=zero_2d, name="dphis_dt", dims=dims_2d, units="m^2/s^3",
        ),
        dtracers_dt=Field(
            data=dtracers, name="dtracers_dt", dims=dims_tr, units="1/s",
        ),
    )
