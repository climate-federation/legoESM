"""Multi-layer soil land model.

Energy balance at the surface drives ground heat flux into a multi-layer
soil thermal model. Precipitation minus evaporation drives a Richards
equation solver for unsaturated flow. Thermal and hydraulic properties
depend on soil moisture (Johansen 1975, Van Genuchten 1980).

**Approximations and limitations:**

* The Richards solver uses a fixed number of Picard iterations (default
  10) with no early-termination convergence check.
* Snow-covered latent exchange uses sublimation energetics (L_s) and
  draws from the snowpack, not the soil moisture reservoir.
* Transpiration moisture stress integrates over the root zone (not just
  the top layer), but root distribution is a simple exponential profile.
* Turbulent fluxes in TileResponse are step-averaged (computed from
  beginning-of-step state); state fields reflect end-of-step.

Physics sequence each time step:

1. Compute bulk surface fluxes (same as slab land)
2. Surface energy balance → ground heat flux G
3. Snow sublimation/deposition from snowpack
4. Infiltration flux = (precip + melt - bare-soil evap) to m/s
5. Richards equation → updated psi, theta, runoff
6. Soil thermal diffusion → updated T_soil
7. Return TileResponse for coupler blending
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice
from legoesm.coupler.bulk_flux import simple_bulk_fluxes, compute_most_fluxes
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.land.soil_hydraulics import psi_from_theta
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


def _get(lp, name: str, fallback):
    """Read from spatial LandSurfaceParams if available, else config scalar."""
    return getattr(lp, name) if lp is not None else fallback


def step_multilayer_land(
    state: MultiLayerLandState,
    forcing: AtmToSurface,
    config: MultiLayerLandConfig,
    U_min: float,
    dt: float,
    lat: jnp.ndarray | None = None,
    carbon_state: CarbonState | None = None,
    doy: float = 0.0,
    land_params=None,
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
    lp = land_params
    T_soil = state.T_soil       # (ncol, n_layers)
    psi = state.psi_soil        # (ncol, n_layers)
    theta = state.theta_soil    # (ncol, n_layers)
    snow = state.snow_depth     # (ncol,)
    snow_age = state.snow_age   # (ncol,)

    # Build soil grid from config
    grid = make_soil_grid(config.soil_grid)

    # Spatially-varying surface parameters (or config scalar fallbacks)
    albedo_land = _get(lp, "albedo_veg", config.albedo_land)
    emissivity = _get(lp, "emissivity", config.emissivity_land)
    z0 = _get(lp, "z0", config.z0_land)

    # Spatially-varying root zone params
    root_depth = _get(lp, "root_depth", config.root_depth)
    theta_wp = _get(lp, "theta_wp", config.theta_wp)
    theta_fc = _get(lp, "theta_fc", config.theta_fc)

    # Surface temperature = top soil layer
    T_surface = T_soil[:, 0]
    ncol = T_surface.shape[0]

    # --- Smooth wind speed floor ---
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
    )

    # --- Root distribution and per-layer moisture stress ---
    # Compute these early so beta_soil reflects the full root zone,
    # not just the top layer.
    theta_sat = config.hydraulics.theta_sat
    theta_r = config.hydraulics.theta_r
    z_centers = grid.z_node  # (n_layers,) depth below surface [m]
    if lp is not None:
        root_frac = jnp.exp(-z_centers[None, :] / root_depth[:, None])
        root_frac = root_frac / jnp.sum(root_frac, axis=-1, keepdims=True)
    else:
        root_frac = jnp.exp(-z_centers / root_depth)
        root_frac = root_frac / jnp.sum(root_frac)

    # Wilting-point / field-capacity range guard.  Using ``+ 1e-10``
    # only protects against exact equality; a misconfigured cell with
    # ``theta_fc <= theta_wp`` still produced exploding ``beta_root``
    # values because the denominator goes near-zero on the same scale
    # as theta itself (~0.1).  Floor the range at 1e-3 m³/m³ (~1 % of
    # theta_sat) so even pathological PFT lookup tables produce sane
    # ``beta_root ∈ [0, 1]``.  Audit finding #6.
    if lp is not None:
        denom = jnp.maximum(
            theta_fc[:, None] - theta_wp[:, None], 1e-3,
        )
        beta_root = jnp.clip(
            (theta - theta_wp[:, None]) / denom,
            0.0, 1.0,
        )
    else:
        denom = jnp.maximum(theta_fc - theta_wp, 1e-3)
        beta_root = jnp.clip(
            (theta - theta_wp) / denom,
            0.0, 1.0,
        )

    # --- Moisture availability from root-zone water content ---
    # Root-zone weighted beta: integrates moisture stress across layers
    # weighted by root density, so a dry top with wet deeper layers
    # still permits transpiration.  Numpy broadcasting handles both
    # ``root_frac`` shapes (``(nlayers,)`` when ``lp is None``,
    # ``(ncol, nlayers)`` when present); ``root_frac[None, :] * beta_root``
    # produces the same result as ``root_frac * beta_root`` for the 1D
    # case so a separate branch is unnecessary.
    w_frac_rz = jnp.clip(
        jnp.sum(root_frac * beta_root, axis=-1), 0.0, 1.0,
    )  # (ncol,)
    beta_soil = config.beta_min + (1.0 - config.beta_min) * w_frac_rz

    # --- Stomatal conductance (if enabled) ---
    beta, gpp_farq = compute_effective_beta(
        T_surface, forcing, beta_soil, config, carbon_state, dt,
        land_params=lp,
    )

    # Stomatal reduction factor: ratio of effective beta to soil-only beta.
    # This captures the stomatal limitation independent of soil moisture,
    # so it can be applied to updated soil moisture later (same as slab land).
    stomatal_ratio = beta / jnp.maximum(beta_soil, 1e-10)

    # --- Surface saturation humidity: use ice saturation over snow ---
    q_sat_liq = saturation_mixing_ratio(T_surface, forcing.p_surface)
    q_sat_ice = saturation_mixing_ratio_ice(T_surface, forcing.p_surface)
    # Treat a column as snow-covered when:
    #   (a) Existing snowpack > 1e-6 kg/m² (always snow regardless of
    #       fresh accumulation OR melt), OR
    #   (b) Fresh snowfall is happening AND the surface is below
    #       freezing (so the new snow will survive — won't melt
    #       immediately during this step).
    # Rule (b) prevents the "warm-surface snowfall" anti-pattern that
    # the iter-67 first-pass fix introduced: a snow-free warm column
    # receiving precip_snow would have been routed as L_s
    # sublimation over an ice qsat surface for the whole turbulent
    # step even though the snow melts away in seconds.  By gating
    # on T_surface < T_freeze we only switch to snow phase when the
    # snow can survive.  Existing snow always uses snow phase
    # regardless of surface temperature (snow_budget handles melt
    # energy correctly).  Iter-68 audit fix.
    fresh_snow_mass = forcing.precip_snow * dt
    has_existing_snow = snow > 1e-6
    has_surviving_fresh_snow = (fresh_snow_mass > 1e-6) & (T_surface < constants.T_freeze)
    has_snow = has_existing_snow | has_surviving_fresh_snow
    q_sat_sfc = jnp.where(has_snow, q_sat_ice, q_sat_liq)
    # Over snow, moisture is freely available from the snowpack (beta=1)
    beta_effective = jnp.where(has_snow, 1.0, beta)
    q_sfc = beta_effective * q_sat_sfc

    # Phase-appropriate latent heat: sublimation over snow, vaporisation over bare soil
    L_eff = jnp.where(has_snow, constants.L_s, constants.L_v)

    # --- Bulk fluxes ---
    rho = forcing.rho_lowest

    if config.bulk_scheme in ("most", "coare3", "large_yeager"):
        tau_x, tau_y, shflx, lhflx, _ = compute_most_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_surface, q_sfc, rho,
            z_ref=config.z_ref,
            z0_init=z0,
            scheme=config.bulk_scheme,
            n_iter=config.bulk_n_iter,
            L_latent=L_eff,
        )
    else:
        tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_surface, q_sfc, rho, wind_speed,
            config.Cd_land, config.Ch_land,
            L_latent=L_eff,
        )

    # --- Surface albedo (snow-mass dependent) ---
    # Use the SAME effective snow mass as the bulk-flux phase decision
    # (iter-68 fix): existing snow always counts; fresh snow counts
    # only when T_surface < T_freeze (it survives the step).  Without
    # this consistency, SW absorption would lag the LH/SH phase
    # transition by one step on every fresh-snow event.  Iter-71 fix.
    snow_effective = jnp.where(
        has_existing_snow | has_surviving_fresh_snow,
        snow + jnp.where(has_surviving_fresh_snow, fresh_snow_mass, 0.0),
        snow,
    )
    if config.snow_albedo_feedback and lat is not None:
        alpha = compute_land_albedo(
            lat, snow_effective, snow_age, config.land_albedo,
        )
    else:
        alpha = jnp.full(T_surface.shape, albedo_land, dtype=T_surface.dtype)

    # --- Radiation ---
    sw_net, lw_net, lw_up = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_surface, alpha,
        emissivity,
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

    # --- Latent mass exchange ---
    # Convert lhflx to mass flux using phase-appropriate latent heat.
    rho_w = constants.rho_water
    evap_rate_demand = lhflx / L_eff  # kg/m2/s, positive up

    # --- Snow sublimation / deposition ---
    # Over snow: latent exchange removes/adds mass from/to the snowpack.
    snow_after_melt = snow_new
    max_sublim = jnp.maximum(snow_after_melt / dt, 0.0)
    sublim_demand = jnp.where(has_snow, evap_rate_demand, 0.0)
    sublim_actual = jnp.minimum(sublim_demand, max_sublim)
    sublim_actual = jnp.where(sublim_demand < 0.0, sublim_demand, sublim_actual)
    snow_new = jnp.maximum(snow_new - sublim_actual * dt, 0.0)

    # --- Water-limit soil evaporation (bare soil only) ---
    dz = grid.dz  # (n_layers,) layer thicknesses [m]
    extractable_water = jnp.sum(
        jnp.maximum(theta - theta_r, 0.0) * dz[None, :], axis=-1
    ) * rho_w  # (ncol,) kg/m2
    precip_rain = forcing.precip_total - forcing.precip_snow
    melt_rate = snow_melt / dt  # kg/m2/s meltwater entering liquid budget
    soil_evap_demand = jnp.where(has_snow, 0.0, evap_rate_demand)
    max_soil_evap = jnp.maximum(extractable_water / dt + precip_rain + melt_rate, 0.0)
    soil_evap = jnp.minimum(soil_evap_demand, max_soil_evap)

    # Total actual mass flux and excess energy
    evap_rate = jnp.where(has_snow, sublim_actual, soil_evap)
    evap_excess_energy = (evap_rate_demand - evap_rate) * L_eff  # W/m2
    lhflx_actual = evap_rate * L_eff

    # --- Root water uptake sink term ---
    # root_frac and beta_root were computed earlier (before beta_soil).
    # Partition evaporation into bare-soil and root-mediated transpiration
    # to avoid double-counting (surface flux_top subtracts bare-soil evap,
    # Richards sink removes root-mediated transpiration).
    # ``f_veg`` and ``weight_sum`` reduce the same ``root_frac * beta_root``
    # product; compute the column reduction once and reuse it.
    # ``root_frac`` is ``(n_layers,)`` when ``lp is None`` and
    # ``(ncol, n_layers)`` when ``lp`` is present; both broadcast
    # cleanly against ``beta_root`` of shape ``(ncol, n_layers)`` without
    # an explicit unsqueeze.  An earlier ``root_frac[None, :]`` worked
    # only for the 1D case and silently produced a ``(ncol, ncol, n_layers)``
    # weight in the 2D case (audit finding 2026-05-12 #4).
    weight = root_frac * beta_root  # (ncol, n_layers)
    _weight_sum_raw = jnp.sum(weight, axis=-1)  # (ncol,)
    f_veg = jnp.clip(_weight_sum_raw, 0.0, 1.0)  # vegetation cover proxy
    # Split surface flux between bare-soil and transpiration.  For
    # ``evap_rate >= 0`` (evaporation upward): bare/veg partition by
    # ``1 - f_veg`` / ``f_veg``.  For ``evap_rate < 0`` (dew /
    # deposition downward): route ALL of the negative flux to
    # ``flux_top`` (treated as bare-soil input) so the column water
    # budget closes.  Routing the vegetated fraction to a
    # ``transpiration`` sink would later be clipped at zero (line
    # below), losing the dew mass — flagged by codex iter-22 #3.
    is_dew = evap_rate < 0.0
    evap_bare = jnp.where(is_dew, evap_rate, evap_rate * (1.0 - f_veg))
    evap_transp = jnp.where(is_dew, 0.0, evap_rate * f_veg)

    # Infiltration: rain + snow meltwater enter the soil; snow goes to snowpack.
    # Only bare-soil evap subtracted (transpiration handled by sink).
    flux_top = (precip_rain + melt_rate - evap_bare) / rho_w  # m/s, positive down

    # Root sink: distribute transpiration across layers weighted by moisture-
    # available root density. The total vertically-integrated sink must equal
    # E_pot_transp (the actual transpiration from the energy balance) to close
    # the water budget. beta_root weights the distribution but must NOT reduce
    # the total — the surface flux already embedded moisture stress via f_veg.
    E_pot_transp = jnp.maximum(evap_transp, 0.0) / rho_w  # m/s
    weight_sum = _weight_sum_raw[..., None]  # (ncol, 1)
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
        emissivity,
    )

    # Recompute q_surface with updated temperature and root-zone moisture.
    # Apply stomatal_ratio so q_surface reflects both soil moisture
    # availability AND stomatal limitation (same as slab land).
    # Use the same ``jnp.maximum(theta_fc - theta_wp, 1e-3)`` floor as
    # the pre-step computation above (lines 142-150) so degenerate PFT
    # lookup-table cells (theta_fc ≈ theta_wp) cannot blow up the
    # post-step ``beta_root_new``.  The previous ``+ 1e-10`` floor was
    # too small relative to the typical theta scale (~0.1), so a
    # pathological PFT cell would produce O(1e7) beta_root_new values
    # — propagating into ``q_sfc_new`` reported back to the atmosphere.
    # Iter-65 audit fix.
    theta_new = richards_out.theta_new
    if lp is not None:
        denom_new = jnp.maximum(
            theta_fc[:, None] - theta_wp[:, None], 1e-3,
        )
        beta_root_new = jnp.clip(
            (theta_new - theta_wp[:, None]) / denom_new,
            0.0, 1.0,
        )
    else:
        denom_new = jnp.maximum(theta_fc - theta_wp, 1e-3)
        beta_root_new = jnp.clip(
            (theta_new - theta_wp) / denom_new,
            0.0, 1.0,
        )
    # See comment above ``w_frac_rz``: broadcasting handles both
    # ``root_frac`` shapes uniformly, no per-branch reduction needed.
    w_frac_rz_new = jnp.clip(
        jnp.sum(root_frac * beta_root_new, axis=-1), 0.0, 1.0,
    )
    beta_soil_new = config.beta_min + (1.0 - config.beta_min) * w_frac_rz_new
    beta_new = stomatal_ratio * beta_soil_new
    q_sat_liq_new = saturation_mixing_ratio(T_surface_new, forcing.p_surface)
    q_sat_ice_new = saturation_mixing_ratio_ice(T_surface_new, forcing.p_surface)
    has_snow_new = snow_new > 1e-6
    q_sat_sfc_new = jnp.where(has_snow_new, q_sat_ice_new, q_sat_liq_new)
    # Over snow, moisture is freely available (beta=1)
    beta_effective_new = jnp.where(has_snow_new, 1.0, beta_new)
    q_sfc_new = beta_effective_new * q_sat_sfc_new

    # --- Carbon cycle ---
    if config.carbon.scheme != "none":
        lat_arr = lat if lat is not None else jnp.zeros(ncol)
        # Recompute Farquhar GPP with updated T and moisture so that
        # photosynthesis and respiration use consistent end-of-step state.
        _, gpp_farq_new = compute_effective_beta(
            T_surface_new, forcing, beta_soil_new, config, carbon_state, dt,
            land_params=lp,
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
        emissivity=jnp.full(T_surface.shape, emissivity, dtype=T_surface.dtype),
        z0=jnp.full(T_surface.shape, z0, dtype=T_surface.dtype),
        q_surface=q_sfc_new,
        shflx=shflx,
        lhflx=lhflx_actual,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up_new,
        u_ocean_sfc=jnp.zeros(ncol),
        v_ocean_sfc=jnp.zeros(ncol),
        co2_flux=co2_flux,
        # Multilayer land: total freshwater to ocean is surface +
        # subsurface runoff.  Both already kg/m²/s.
        freshwater_flux=(
            richards_out.runoff_surface + richards_out.runoff_subsurface
        ),
        # Land does not extract heat directly from the ocean.
        ocean_heat_extraction=jnp.zeros(ncol),
        # Land does not exert stress on the ocean.
        ocean_stress_x=jnp.zeros(ncol),
        ocean_stress_y=jnp.zeros(ncol),
        # Phase-aware moisture mass flux (evap_rate already accounts
        # for L_eff switch and water-limit in the columnar Richards
        # solve).  Audit F3.
        surface_mass_flux=evap_rate,
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
