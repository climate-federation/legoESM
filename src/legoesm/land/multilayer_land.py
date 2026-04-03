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
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice
from legoesm.coupler.bulk_flux import simple_bulk_fluxes
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.coupler.surface_energy import surface_radiation_fluxes
from legoesm.land.carbon.config import CarbonState
from legoesm.land.carbon.carbon_cycle import step_carbon
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.snow_budget import update_snow
from legoesm.land.state import MultiLayerLandState
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.stomata_utils import compute_effective_beta
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
    beta_soil = config.beta_min + (1.0 - config.beta_min) * w_frac

    # --- Stomatal conductance (if enabled) ---
    beta, gpp_farq = compute_effective_beta(
        T_surface, forcing, beta_soil, config, carbon_state, dt,
    )

    # Stomatal reduction factor: ratio of effective beta to soil-only beta.
    # This captures the stomatal limitation independent of soil moisture,
    # so it can be applied to updated soil moisture later (same as slab land).
    stomatal_ratio = beta / jnp.maximum(beta_soil, 1e-10)

    # --- Surface saturation humidity: use ice saturation over snow ---
    q_sat_liq = saturation_mixing_ratio(T_surface, forcing.p_surface)
    q_sat_ice = saturation_mixing_ratio_ice(T_surface, forcing.p_surface)
    has_snow = snow > 1e-6  # kg/m2 threshold
    q_sat_sfc = jnp.where(has_snow, q_sat_ice, q_sat_liq)
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
        tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_surface, q_sfc, rho, wind_speed,
            config.Cd_land, config.Ch_land,
        )

    # --- Surface albedo (from current snow state) ---
    if config.snow_albedo_feedback and lat is not None:
        alpha = compute_land_albedo(
            lat, snow, snow_age, config.land_albedo,
        )
    else:
        alpha = jnp.broadcast_to(jnp.array(config.albedo_land), T_surface.shape)

    # --- Radiation ---
    sw_net, lw_net, lw_up = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_surface, alpha,
        config.emissivity_land,
    )

    # --- Ground heat flux (residual of surface energy balance) ---
    # G = SW_net + LW_net - SH - LH  (positive into soil)
    G_surface = sw_net + lw_net - shflx - lhflx

    # --- Snow budget (energy-limited melt) ---
    # G_surface drives the melt: M = max(0, G * dt / L_f)
    snow_new, snow_age_new, snow_melt = update_snow(
        snow, snow_age, T_surface, forcing.precip_snow, dt,
        Q_net=G_surface,
        snow_melt_rate=config.snow_melt_rate,
        T_snow_melt=config.T_snow_melt,
    )

    # Subtract melt energy from ground heat flux before soil thermal solve.
    # Melting snow consumes L_f per kg, reducing the energy entering the soil.
    melt_energy = snow_melt * constants.L_f / dt  # W/m2 consumed by melt
    G_surface = G_surface - melt_energy

    # --- Infiltration flux for Richards equation ---
    # Convert precip (kg/m2/s) and evap (kg/m2/s) to water depth rate (m/s)
    rho_w = constants.rho_water
    evap_rate_demand = lhflx / constants.L_v  # kg/m2/s, positive up

    # --- Water-limit evaporation ---
    # Total extractable water: integrate (theta - theta_r) * dz * rho_w
    dz = grid.dz  # (n_layers,) layer thicknesses [m]
    extractable_water = jnp.sum(
        jnp.maximum(theta - theta_r, 0.0) * dz[None, :], axis=-1
    ) * rho_w  # (ncol,) kg/m2
    precip_rain = forcing.precip_total - forcing.precip_snow
    melt_rate = snow_melt / dt  # kg/m2/s meltwater entering liquid budget
    max_evap = jnp.maximum(extractable_water / dt + precip_rain + melt_rate, 0.0)
    evap_rate = jnp.minimum(evap_rate_demand, max_evap)

    # Excess latent heat (demand that couldn't be met) redirected to soil warming
    evap_excess_energy = (evap_rate_demand - evap_rate) * constants.L_v  # W/m2
    lhflx_actual = evap_rate * constants.L_v

    # --- Root water uptake sink term ---
    # Exponential root distribution: root_frac(z) ~ exp(-z / root_depth)
    z_centers = grid.z_node  # (n_layers,) depth below surface [m]
    root_frac = jnp.exp(-z_centers / config.root_depth)
    root_frac = root_frac / jnp.sum(root_frac)  # normalize to 1

    # Soil moisture stress: beta(theta) = clip((theta - theta_wp)/(theta_fc - theta_wp), 0, 1)
    beta_root = jnp.clip(
        (theta - config.theta_wp) / (config.theta_fc - config.theta_wp + 1e-10),
        0.0, 1.0,
    )

    # Partition evaporation into bare-soil and root-mediated transpiration
    # to avoid double-counting (surface flux_top subtracts bare-soil evap,
    # Richards sink removes root-mediated transpiration).
    f_veg = jnp.clip(
        jnp.sum(root_frac[None, :] * beta_root, axis=-1), 0.0, 1.0,
    )  # (ncol,) vegetation cover proxy
    evap_bare = evap_rate * (1.0 - f_veg)      # bare-soil evaporation
    evap_transp = evap_rate * f_veg             # transpiration (root-mediated)

    # Infiltration: rain + snow meltwater enter the soil; snow goes to snowpack.
    # Only bare-soil evap subtracted (transpiration handled by sink).
    flux_top = (precip_rain + melt_rate - evap_bare) / rho_w  # m/s, positive down

    # Root sink: distribute transpiration across layers weighted by moisture-
    # available root density. The total vertically-integrated sink must equal
    # E_pot_transp (the actual transpiration from the energy balance) to close
    # the water budget. beta_root weights the distribution but must NOT reduce
    # the total — the surface flux already embedded moisture stress via f_veg.
    E_pot_transp = jnp.maximum(evap_transp, 0.0) / rho_w  # m/s
    weight = root_frac[None, :] * beta_root  # (ncol, n_layers)
    weight_sum = jnp.sum(weight, axis=-1, keepdims=True)  # (ncol, 1)
    # Safe normalization: when all layers are dry, E_pot_transp ≈ 0 anyway
    weight_norm = weight / jnp.maximum(weight_sum, 1e-20)
    sink = weight_norm * E_pot_transp[:, None] / dz[None, :]

    # --- Richards equation: update soil moisture ---
    richards_out = solve_richards(
        psi, theta, grid,
        config.hydraulics, config.richards,
        flux_top, sink, dt,
    )

    # --- Evaporation water budget closure ---
    # Ensure top-layer theta reflects actual evaporative loss not captured
    # by infiltration flux alone (e.g., when evap > precip and soil is dry).
    theta_corrected = jnp.clip(
        richards_out.theta_new,
        config.hydraulics.theta_r,
        config.hydraulics.theta_sat,
    )
    richards_out = richards_out._replace(theta_new=theta_corrected)

    # --- Soil thermal diffusion: update soil temperature ---
    # Excess energy from water-limited evaporation warms the soil
    G_surface = G_surface + evap_excess_energy
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

    # Post-step albedo: reflects updated snow for the next atmosphere step
    if config.snow_albedo_feedback and lat is not None:
        alpha_new = compute_land_albedo(
            lat, snow_new, snow_age_new, config.land_albedo,
        )
    else:
        alpha_new = alpha

    _, _, lw_up_new = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_surface_new, alpha_new,
        config.emissivity_land,
    )

    # Recompute q_surface with updated temperature.
    # Apply stomatal_ratio so q_surface reflects both soil moisture
    # availability AND stomatal limitation (same as slab land).
    theta_top_new = richards_out.theta_new[:, 0]
    w_frac_new = jnp.clip(
        (theta_top_new - theta_r) / (theta_sat - theta_r + 1e-10), 0.0, 1.0
    )
    beta_soil_new = config.beta_min + (1.0 - config.beta_min) * w_frac_new
    beta_new = stomatal_ratio * beta_soil_new
    q_sat_liq_new = saturation_mixing_ratio(T_surface_new, forcing.p_surface)
    q_sat_ice_new = saturation_mixing_ratio_ice(T_surface_new, forcing.p_surface)
    has_snow_new = snow_new > 1e-6
    q_sat_sfc_new = jnp.where(has_snow_new, q_sat_ice_new, q_sat_liq_new)
    q_sfc_new = beta_new * q_sat_sfc_new

    # --- Carbon cycle ---
    if config.carbon.scheme != "none":
        lat_arr = lat if lat is not None else jnp.zeros(ncol)
        # Recompute Farquhar GPP with updated T and moisture so that
        # photosynthesis and respiration use consistent end-of-step state.
        _, gpp_farq_new = compute_effective_beta(
            T_surface_new, forcing, beta_soil_new, config, carbon_state, dt,
        )
        carbon_state_new, co2_flux = step_carbon(
            carbon_state, forcing.sw_down, T_surface_new, forcing.co2_ppmv,
            beta_soil_new, lat_arr, doy, forcing.precip_total, config.carbon, dt,
            gpp_override=gpp_farq_new,
        )
    else:
        carbon_state_new = carbon_state
        co2_flux = jnp.zeros(ncol)

    response = TileResponse(
        T_surface=T_surface_new,
        albedo=alpha_new,
        emissivity=jnp.broadcast_to(
            jnp.array(config.emissivity_land), T_surface.shape
        ),
        z0=jnp.broadcast_to(jnp.array(config.z0_land), T_surface.shape),
        q_surface=q_sfc_new,
        shflx=shflx,
        lhflx=lhflx_actual,
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
