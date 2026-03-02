"""Slab thermal + bucket hydrology land model.

Energy balance:
    C_soil * d_soil * dT/dt = SW_net + LW_net - SH - LH

Bucket hydrology:
    dW/dt = precip - E,   W in [0, W_max]
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.land.config import LandConfig
from legoesm.land.state import LandState


def step_land(
    state: LandState,
    forcing: AtmToSurface,
    config: LandConfig,
    U_min: float,
    dt: float,
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

    Returns
    -------
    (LandState, TileResponse)
        Updated state and surface response for tile blending.
    """
    T_soil = state.T_soil.data
    W = state.W_bucket.data

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
    Cd = config.Cd_land
    Ch = config.Ch_land

    tau_x = -rho * Cd * wind_speed * forcing.u_lowest
    tau_y = -rho * Cd * wind_speed * forcing.v_lowest
    shflx = rho * constants.c_pd * Ch * wind_speed * (T_soil - forcing.T_lowest)
    lhflx = rho * constants.L_v * Ch * wind_speed * (q_sfc - forcing.q_lowest)

    # Radiation
    sw_net = (1.0 - config.albedo_land) * forcing.sw_down
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
    )

    # Recompute upward LW with updated temperature for consistency
    lw_up_new = config.emissivity_land * constants.sigma_sb * T_soil_new ** 4

    response = TileResponse(
        T_surface=T_soil_new,
        albedo=jnp.broadcast_to(jnp.array(config.albedo_land), T_soil.shape),
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
