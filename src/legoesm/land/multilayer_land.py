"""Multi-layer soil land model (Task 8).

Energy balance at the surface drives ground heat flux into a multi-layer
soil thermal model. Precipitation minus evaporation drives Richards
equation for unsaturated flow. Thermal and hydraulic properties depend
on soil moisture (Johansen 1975, Van Genuchten 1980).

Physics sequence each time step:
1. Compute bulk surface fluxes (same as slab land)
2. Surface energy balance → ground heat flux G
3. Infiltration flux = (precip - evap) converted to m/s
4. Richards equation → updated psi, theta, runoff
5. Soil thermal diffusion → updated T_soil
6. Return TileResponse for coupler blending
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.land.carbon.config import CarbonState
from legoesm.land.carbon.carbon_cycle import step_carbon
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.state import MultiLayerLandState
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.richards import solve_richards
from legoesm.land.soil_thermal import solve_soil_thermal
from legoesm.surface_albedo import land_albedo as compute_land_albedo


def step_multilayer_land(
    state: MultiLayerLandState,
    forcing: AtmToSurface,
    config: MultiLayerLandConfig,
    U_min: float,
    dt: float,
    lat: jnp.ndarray | None = None,
    carbon_state: CarbonState | None = None,
    doy: float = 0.0,
) -> tuple[MultiLayerLandState, TileResponse, CarbonState | None]:
    """Step the multi-layer land model forward by dt seconds.

    Parameters
    ----------
    state : MultiLayerLandState
        Current land state with multi-layer soil T, psi, theta.
    forcing : AtmToSurface
        Atmospheric forcing fields.
    config : MultiLayerLandConfig
        Land model parameters including sub-configs.
    U_min : float
        Minimum wind speed floor [m/s].
    dt : float
        Time step [s].
    lat : jnp.ndarray or None
        Latitude in radians, shape (ncol,). Required when
        snow_albedo_feedback is True.

    Returns
    -------
    (MultiLayerLandState, TileResponse, CarbonState | None)
        Updated state, surface response, and updated carbon state.
    """
    T_soil = state.T_soil       # (ncol, n_layers)
    psi = state.psi_soil        # (ncol, n_layers)
    theta = state.theta_soil    # (ncol, n_layers)
    snow = state.snow_depth     # (ncol,)
    snow_age = state.snow_age   # (ncol,)

    # Build soil grid from config
    grid = make_soil_grid(config.soil_grid)

    # Surface temperature = top soil layer
    T_surface = T_soil[:, 0]
    ncol = T_surface.shape[0]

    # --- Smooth wind speed floor ---
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
    )

    # --- Moisture availability from top-layer water content ---
    theta_top = theta[:, 0]
    theta_sat = config.hydraulics.theta_sat
    theta_r = config.hydraulics.theta_r
    w_frac = jnp.clip(
        (theta_top - theta_r) / (theta_sat - theta_r + 1e-10), 0.0, 1.0
    )
    beta = config.beta_min + (1.0 - config.beta_min) * w_frac

    # --- Surface saturation humidity ---
    q_sat_sfc = saturation_mixing_ratio(T_surface, forcing.p_surface)
    q_sfc = beta * q_sat_sfc

    # --- Bulk fluxes ---
    rho = forcing.rho_lowest

    if config.bulk_scheme in ("most", "coare3", "large_yeager"):
        from legoesm.coupler.bulk_flux import compute_most_fluxes
        tau_x, tau_y, shflx, lhflx, _ = compute_most_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_surface, q_sfc, rho,
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
        shflx = rho * constants.c_pd * Ch * wind_speed * (
            T_surface - forcing.T_lowest
        )
        lhflx = rho * constants.L_v * Ch * wind_speed * (
            q_sfc - forcing.q_lowest
        )

    # --- Snow budget ---
    snow_accum = forcing.precip_snow * dt
    melt_rate = config.snow_melt_rate * jnp.maximum(
        T_surface - config.T_snow_melt, 0.0
    )
    snow_melt = jnp.minimum(melt_rate * dt, snow + snow_accum)
    snow_new = jnp.maximum(snow + snow_accum - snow_melt, 0.0)

    is_snowing = forcing.precip_snow > 1e-10
    snow_age_new = jnp.where(is_snowing, 0.0, snow_age + dt)
    snow_age_new = jnp.where(snow_new > 0.0, snow_age_new, 0.0)

    # --- Surface albedo ---
    if config.snow_albedo_feedback and lat is not None:
        alpha = compute_land_albedo(
            lat, snow_new, snow_age_new, config.land_albedo,
        )
    else:
        alpha = jnp.broadcast_to(jnp.array(config.albedo_land), T_surface.shape)

    # --- Radiation ---
    sw_net = (1.0 - alpha) * forcing.sw_down
    lw_down_abs = config.emissivity_land * forcing.lw_down
    lw_up = config.emissivity_land * constants.sigma_sb * T_surface ** 4
    lw_net = lw_down_abs - lw_up

    # --- Ground heat flux (residual of surface energy balance) ---
    # G = SW_net + LW_net - SH - LH  (positive into soil)
    G_surface = sw_net + lw_net - shflx - lhflx

    # --- Infiltration flux for Richards equation ---
    # Convert precip (kg/m2/s) and evap (kg/m2/s) to water depth rate (m/s)
    rho_w = 1000.0
    evap_rate = lhflx / constants.L_v  # kg/m2/s, positive up
    flux_top = (forcing.precip_total - evap_rate) / rho_w  # m/s, positive down

    # --- Richards equation: update soil moisture ---
    # Sink term (root uptake) set to zero for now
    sink = jnp.zeros_like(theta)

    richards_out = solve_richards(
        psi, theta, grid,
        config.hydraulics, config.richards,
        flux_top, sink, dt,
    )

    # --- Soil thermal diffusion: update soil temperature ---
    T_soil_new = solve_soil_thermal(
        T_soil, richards_out.theta_new, grid,
        config.hydraulics, config.thermal,
        G_surface, dt,
    )

    # --- Build new state ---
    new_state = MultiLayerLandState(
        T_soil=T_soil_new,
        psi_soil=richards_out.psi_new,
        theta_soil=richards_out.theta_new,
        runoff_surface=richards_out.runoff_surface,
        runoff_subsurface=richards_out.runoff_subsurface,
        snow_depth=snow_new,
        snow_age=snow_age_new,
    )

    # --- Build TileResponse ---
    T_surface_new = T_soil_new[:, 0]
    lw_up_new = config.emissivity_land * constants.sigma_sb * T_surface_new ** 4

    # Recompute q_surface with updated temperature
    theta_top_new = richards_out.theta_new[:, 0]
    w_frac_new = jnp.clip(
        (theta_top_new - theta_r) / (theta_sat - theta_r + 1e-10), 0.0, 1.0
    )
    beta_new = config.beta_min + (1.0 - config.beta_min) * w_frac_new
    q_sfc_new = beta_new * saturation_mixing_ratio(T_surface_new, forcing.p_surface)

    # --- Carbon cycle ---
    if config.carbon.scheme != "none":
        lat_arr = lat if lat is not None else jnp.zeros(ncol)
        carbon_state_new, co2_flux = step_carbon(
            carbon_state, forcing.sw_down, T_surface_new, forcing.co2_ppmv,
            beta, lat_arr, doy, forcing.precip_total, config.carbon, dt,
        )
    else:
        carbon_state_new = carbon_state
        co2_flux = jnp.zeros(ncol)

    response = TileResponse(
        T_surface=T_surface_new,
        albedo=alpha,
        emissivity=jnp.broadcast_to(
            jnp.array(config.emissivity_land), T_surface.shape
        ),
        z0=jnp.broadcast_to(jnp.array(config.z0_land), T_surface.shape),
        q_surface=q_sfc_new,
        shflx=shflx,
        lhflx=lhflx,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up_new,
        u_ocean_sfc=jnp.zeros(ncol),
        v_ocean_sfc=jnp.zeros(ncol),
        co2_flux=co2_flux,
    )

    return new_state, response, carbon_state_new


def init_multilayer_land_state(
    ncol: int,
    config: MultiLayerLandConfig,
    T_init: float = 280.0,
    theta_init: float | None = None,
) -> MultiLayerLandState:
    """Create initial multi-layer land state.

    Parameters
    ----------
    ncol : int
        Number of columns.
    config : MultiLayerLandConfig
        Land model configuration.
    T_init : float
        Initial uniform soil temperature [K].
    theta_init : float or None
        Initial uniform volumetric water content [m3/m3].
        If None, uses 0.5 * theta_sat.

    Returns
    -------
    MultiLayerLandState
    """
    from legoesm.land.soil_hydraulics import psi_from_theta

    grid = make_soil_grid(config.soil_grid)
    nlayers = grid.n_layers

    if theta_init is None:
        theta_init = 0.5 * config.hydraulics.theta_sat

    T_soil = jnp.full((ncol, nlayers), T_init)
    theta_soil = jnp.full((ncol, nlayers), theta_init)
    psi_soil = psi_from_theta(theta_soil, config.hydraulics)

    return MultiLayerLandState(
        T_soil=T_soil,
        psi_soil=psi_soil,
        theta_soil=theta_soil,
        runoff_surface=jnp.zeros(ncol),
        runoff_subsurface=jnp.zeros(ncol),
        snow_depth=jnp.zeros(ncol),
        snow_age=jnp.zeros(ncol),
    )
