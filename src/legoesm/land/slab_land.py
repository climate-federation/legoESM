"""Slab thermal + bucket hydrology land model.

Energy balance:
    C_soil * d_soil * dT/dt = SW_net + LW_net - SH - LH

Bucket hydrology:
    dW/dt = precip - E,   W in [0, W_max]

Snow:
    d(snow)/dt = precip_snow - melt
    Snow cover fraction and albedo feedback (Task 10).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.land.config import LandConfig
from legoesm.land.state import LandState
from legoesm.surface_albedo import land_albedo as compute_land_albedo


def step_land(
    state: LandState,
    forcing: AtmToSurface,
    config: LandConfig,
    U_min: float,
    dt: float,
    lat: jnp.ndarray | None = None,
) -> tuple[LandState, TileResponse]:
    """Step the slab land model forward by dt seconds.

    Parameters
    ----------
    state : LandState
        Current land state.
    forcing : AtmToSurface
        Atmospheric forcing fields.
    config : LandConfig
        Land model parameters.
    U_min : float
        Minimum wind speed floor [m/s].
    dt : float
        Time step [s].
    lat : jnp.ndarray or None
        Latitude in radians, same shape as T_soil. Required when
        snow_albedo_feedback is True.

    Returns
    -------
    (LandState, TileResponse)
        Updated state and surface response for tile blending.
    """
    T_soil = state.T_soil.data
    W = state.W_bucket.data
    snow = state.snow_depth.data
    snow_age = state.snow_age.data

    # --- Snow budget ---
    # Accumulation from snowfall
    snow_accum = forcing.precip_snow * dt  # kg/m2

    # Melt: proportional to T above freezing
    melt_rate = config.snow_melt_rate * jnp.maximum(
        T_soil - config.T_snow_melt, 0.0
    )  # kg/m2/s
    snow_melt = jnp.minimum(melt_rate * dt, snow + snow_accum)

    snow_new = jnp.maximum(snow + snow_accum - snow_melt, 0.0)

    # Snow age: reset when fresh snowfall, otherwise age
    is_snowing = forcing.precip_snow > 1e-10
    snow_age_new = jnp.where(is_snowing, 0.0, snow_age + dt)
    # If all snow has melted, reset age to zero
    snow_age_new = jnp.where(snow_new > 0.0, snow_age_new, 0.0)

    # --- Surface albedo ---
    if config.snow_albedo_feedback and lat is not None:
        alpha = compute_land_albedo(
            lat, snow_new, snow_age_new, config.land_albedo,
        )
    else:
        alpha = jnp.broadcast_to(jnp.array(config.albedo_land), T_soil.shape)

    # Smooth wind speed floor
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
    )

    # Moisture availability: smooth ramp from beta_min to 1
    w_frac = jnp.clip(W / config.W_max, 0.0, 1.0)
    beta = config.beta_min + (1.0 - config.beta_min) * w_frac

    # Surface saturation humidity
    q_sat_sfc = saturation_mixing_ratio(T_soil, forcing.p_surface)
    q_sfc = beta * q_sat_sfc

    # Bulk fluxes
    rho = forcing.rho_lowest

    if config.bulk_scheme in ("most", "coare3", "large_yeager"):
        from legoesm.coupler.bulk_flux import compute_most_fluxes
        tau_x, tau_y, shflx, lhflx, _ = compute_most_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_soil, q_sfc, rho,
            z_ref=config.z_ref,
            z0_init=config.z0_land,
            scheme=config.bulk_scheme,
            n_iter=config.bulk_n_iter,
        )
    else:
        Cd = config.Cd_land
        Ch = config.Ch_land
        tau_x = -rho * Cd * wind_speed * forcing.u_lowest
        tau_y = -rho * Cd * wind_speed * forcing.v_lowest
        shflx = rho * constants.c_pd * Ch * wind_speed * (T_soil - forcing.T_lowest)
        lhflx = rho * constants.L_v * Ch * wind_speed * (q_sfc - forcing.q_lowest)

    # Radiation
    sw_net = (1.0 - alpha) * forcing.sw_down
    lw_down_abs = config.emissivity_land * forcing.lw_down
    lw_up = config.emissivity_land * constants.sigma_sb * T_soil ** 4
    lw_net = lw_down_abs - lw_up

    # Energy balance: dT/dt
    heat_cap = config.C_soil * config.d_soil
    dT_dt = (sw_net + lw_net - shflx - lhflx) / heat_cap
    T_soil_new = T_soil + dt * dT_dt

    # Bucket hydrology
    evap_rate = lhflx / constants.L_v  # kg/m2/s (positive = upward)
    dW_dt = forcing.precip_total - evap_rate
    W_new = W + dt * dW_dt
    W_new = jnp.clip(W_new, 0.0, config.W_max)

    new_state = LandState(
        T_soil=state.T_soil.replace(data=T_soil_new),
        W_bucket=state.W_bucket.replace(data=W_new),
        snow_depth=state.snow_depth.replace(data=snow_new),
        snow_age=state.snow_age.replace(data=snow_age_new),
    )

    # Recompute upward LW with updated temperature for consistency
    lw_up_new = config.emissivity_land * constants.sigma_sb * T_soil_new ** 4

    response = TileResponse(
        T_surface=T_soil_new,
        albedo=alpha,
        emissivity=jnp.broadcast_to(jnp.array(config.emissivity_land), T_soil.shape),
        z0=jnp.broadcast_to(jnp.array(config.z0_land), T_soil.shape),
        q_surface=q_sfc,
        shflx=shflx,
        lhflx=lhflx,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up_new,
        u_ocean_sfc=jnp.zeros_like(T_soil),
        v_ocean_sfc=jnp.zeros_like(T_soil),
        co2_flux=jnp.zeros_like(T_soil),
    )

    return new_state, response
