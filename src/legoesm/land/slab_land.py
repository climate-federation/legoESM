"""Slab thermal + bucket hydrology land model.

Energy balance:
    C_soil * d_soil * dT/dt = SW_net + LW_net - SH - LH - L_f * melt_rate

Bucket hydrology:
    dW/dt = precip_rain + melt - E,   W in [0, W_max]

Snow:
    d(snow)/dt = precip_snow - melt
    Energy-limited melt: melt = min(snow, max(0, Q_net * dt / L_f))
    Melt energy is subtracted from the surface energy budget.
    Meltwater enters the soil water bucket.
    Snow cover fraction and albedo feedback (Task 10).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.coupler.bulk_flux import simple_bulk_fluxes
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.coupler.surface_energy import surface_radiation_fluxes
from legoesm.land.carbon.config import CarbonState
from legoesm.land.carbon.carbon_cycle import step_carbon
from legoesm.land.config import LandConfig
from legoesm.land.snow_budget import update_snow
from legoesm.land.state import LandState
from legoesm.land.stomata_utils import compute_effective_beta
from legoesm.surface_albedo import land_albedo as compute_land_albedo


def step_land(
    state: LandState,
    forcing: AtmToSurface,
    config: LandConfig,
    U_min: float,
    dt: float,
    lat: jnp.ndarray | None = None,
    carbon_state: CarbonState | None = None,
    doy: float = 0.0,
) -> tuple[LandState, TileResponse, CarbonState | None]:
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
    (LandState, TileResponse, CarbonState | None)
        Updated state, surface response, and updated carbon state.
    """
    T_soil = state.T_soil.data
    W = state.W_bucket.data
    snow = state.snow_depth.data
    snow_age = state.snow_age.data

    # --- Surface albedo (from current snow state) ---
    if config.snow_albedo_feedback and lat is not None:
        alpha = compute_land_albedo(
            lat, snow, snow_age, config.land_albedo,
        )
    else:
        alpha = jnp.broadcast_to(jnp.array(config.albedo_land), T_soil.shape)

    # Smooth wind speed floor
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
    )

    # Moisture availability: smooth ramp from beta_min to 1
    w_frac = jnp.clip(W / config.W_max, 0.0, 1.0)
    beta_soil = config.beta_min + (1.0 - config.beta_min) * w_frac

    # --- Stomatal conductance (if enabled) ---
    beta, gpp_farq = compute_effective_beta(
        T_soil, forcing, beta_soil, config, carbon_state, dt,
    )

    # Stomatal reduction factor: ratio of effective beta to soil-only beta.
    # This captures the stomatal limitation independent of soil moisture,
    # so it can be applied to updated soil moisture later.
    stomatal_ratio = beta / jnp.maximum(beta_soil, 1e-10)

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
        tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_soil, q_sfc, rho, wind_speed,
            config.Cd_land, config.Ch_land,
        )

    # Radiation
    sw_net, lw_net, lw_up = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_soil, alpha,
        config.emissivity_land,
    )

    # --- Net surface energy flux (positive = energy into soil) ---
    Q_net = sw_net + lw_net - shflx - lhflx

    # --- Snow budget (energy-limited melt) ---
    # Q_net drives the melt rate: M = max(0, Q_net * dt / L_f)
    snow_new, snow_age_new, snow_melt = update_snow(
        snow, snow_age, T_soil, forcing.precip_snow, dt,
        Q_net=Q_net,
        snow_melt_rate=config.snow_melt_rate,
        T_snow_melt=config.T_snow_melt,
    )

    # --- Energy balance: dT/dt ---
    # The melt consumes latent heat of fusion, reducing the energy
    # available for warming the soil slab.
    heat_cap = config.C_soil * config.d_soil
    melt_energy = snow_melt * constants.L_f / dt  # W/m2 consumed by melt
    dT_dt = (Q_net - melt_energy) / heat_cap
    T_soil_new = T_soil + dt * dT_dt

    # --- Bucket hydrology ---
    # Rain and snow meltwater enter the bucket; snow goes to snowpack.
    evap_rate = lhflx / constants.L_v  # kg/m2/s (positive = upward)
    precip_rain = forcing.precip_total - forcing.precip_snow
    melt_rate = snow_melt / dt  # kg/m2/s entering liquid budget
    dW_dt = precip_rain + melt_rate - evap_rate
    W_new = W + dt * dW_dt
    W_new = jnp.clip(W_new, 0.0, config.W_max)

    new_state = LandState(
        T_soil=state.T_soil.replace(data=T_soil_new),
        W_bucket=state.W_bucket.replace(data=W_new),
        snow_depth=state.snow_depth.replace(data=snow_new),
        snow_age=state.snow_age.replace(data=snow_age_new),
    )

    # Recompute upward LW with updated temperature for consistency
    _, _, lw_up_new = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_soil_new, alpha,
        config.emissivity_land,
    )

    # Recompute q_surface from updated T and moisture for consistency.
    # Apply the stomatal reduction factor so that q_surface reflects both
    # soil moisture availability AND stomatal conductance limitation.
    w_frac_new = jnp.clip(W_new / config.W_max, 0.0, 1.0)
    beta_soil_new = config.beta_min + (1.0 - config.beta_min) * w_frac_new
    beta_new = stomatal_ratio * beta_soil_new
    q_sfc_new = beta_new * saturation_mixing_ratio(T_soil_new, forcing.p_surface)

    # --- Carbon cycle ---
    if config.carbon.scheme != "none":
        lat_arr = lat if lat is not None else jnp.zeros_like(T_soil)
        # Recompute Farquhar GPP with updated T and moisture so that
        # photosynthesis and respiration use consistent end-of-step state.
        _, gpp_farq_new = compute_effective_beta(
            T_soil_new, forcing, beta_soil_new, config, carbon_state, dt,
        )
        carbon_state_new, co2_flux = step_carbon(
            carbon_state, forcing.sw_down, T_soil_new, forcing.co2_ppmv,
            beta_soil_new, lat_arr, doy, forcing.precip_total, config.carbon, dt,
            gpp_override=gpp_farq_new,
        )
    else:
        carbon_state_new = carbon_state
        co2_flux = jnp.zeros_like(T_soil)

    response = TileResponse(
        T_surface=T_soil_new,
        albedo=alpha,
        emissivity=jnp.broadcast_to(jnp.array(config.emissivity_land), T_soil.shape),
        z0=jnp.broadcast_to(jnp.array(config.z0_land), T_soil.shape),
        q_surface=q_sfc_new,
        shflx=shflx,
        lhflx=lhflx,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up_new,
        u_ocean_sfc=jnp.zeros_like(T_soil),
        v_ocean_sfc=jnp.zeros_like(T_soil),
        co2_flux=co2_flux,
    )

    return new_state, response, carbon_state_new
