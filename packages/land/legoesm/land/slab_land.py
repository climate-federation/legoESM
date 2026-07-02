"""Slab thermal + bucket hydrology land model.

Energy balance:
    C_soil * d_soil * dT/dt = SW_net + LW_net - SH - LH - L_f * melt_rate

Bucket hydrology:
    dW/dt = precip_rain + melt - E_bare,   W in [0, W_max]

Snow:
    d(snow)/dt = precip_snow - melt - sublimation + deposition
    Energy-limited melt: melt = min(snow, max(0, Q_net * dt / L_f))
    Melt energy subtracted from the surface energy budget.
    Meltwater enters the soil water bucket.
    Sublimation/deposition uses L_s (not L_v) and draws from/adds to
    the snowpack — not the liquid soil bucket.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice
from legoesm.core.bulk_flux import simple_bulk_fluxes, compute_most_fluxes
from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm.core.surface_energy import surface_radiation_fluxes
from legoesm.land.carbon.config import CarbonState
from legoesm.land.carbon.carbon_cycle import step_carbon
from legoesm.land.bucket_hydrology import partition_bucket_runoff
from legoesm.land.config import LandConfig
from legoesm.land.snow_budget import update_snow
from legoesm.land.state import LandState
from legoesm.land.stomata_utils import compute_effective_beta
from legoesm.surface_albedo import land_albedo as compute_land_albedo
from legoesm.land.surface_params import read_spatial_param as _get


def step_land(
    state: LandState,
    forcing: AtmToSurface,
    config: LandConfig,
    U_min: float,
    dt: float,
    lat: jnp.ndarray | None = None,
    carbon_state: CarbonState | None = None,
    doy: float = 0.0,
    land_params=None,
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
    lp = land_params
    T_soil = state.T_soil.data
    W = state.W_bucket.data
    snow = state.snow_depth.data
    snow_age = state.snow_age.data

    # Spatially-varying surface parameters (or config scalar fallbacks)
    albedo_land = _get(lp, "albedo_veg", config.albedo_land)
    emissivity = _get(lp, "emissivity", config.emissivity_land)
    z0 = _get(lp, "z0", config.z0_land)
    W_max = _get(lp, "W_max", config.W_max)
    C_soil = _get(lp, "C_soil", config.C_soil)
    d_soil = _get(lp, "d_soil", config.d_soil)
    # Bucket-hydrology scalars are LandConfig fields, NOT part of the
    # LandSurfaceParams spatial container (which is locked to the 12 PARAM_NAMES /
    # PARAM_BOUNDS entries).  Read them straight from config — never via the spatial
    # helper (a passed LandSurfaceParams has no such attribute).
    K_infiltration = config.K_infiltration
    infil_suction_boost = config.infil_suction_boost

    # Account for fresh snowfall that will survive this step when the
    # surface is below freezing.  Used both by the albedo block here
    # AND by the later ``has_snow`` dispatch — they must agree.  Codex
    # iter-44 #1: previously the albedo block used only the pre-step
    # ``snow`` while later latent fluxes treated ``precip_snow*dt`` as
    # snow-covered, so a snow-free cell receiving fresh snow absorbed
    # bare-land SW for one timestep.
    fresh_snow_mass = forcing.precip_snow * dt
    fresh_snow_surviving = jnp.where(
        T_soil < constants.T_freeze, fresh_snow_mass, 0.0,
    )
    snow_for_albedo = snow + fresh_snow_surviving

    # --- Surface albedo (from current snow state + surviving fresh snow) ---
    if config.snow_albedo_feedback and lat is not None:
        # Snow-free base = per-cell map albedo (CLM PFT) when land_params supplied,
        # else the latitude-band default; snow albedo blends on top either way.
        _base = None if lp is None else jnp.broadcast_to(albedo_land, T_soil.shape)
        alpha = compute_land_albedo(
            lat, snow_for_albedo, snow_age, config.land_albedo, base_albedo=_base,
        )
    else:
        alpha = jnp.full(T_soil.shape, albedo_land, dtype=T_soil.dtype)

    # Smooth wind speed floor
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
    )

    # Moisture availability: smooth ramp from beta_min to 1
    w_frac = jnp.clip(W / W_max, 0.0, 1.0)
    beta_soil = config.beta_min + (1.0 - config.beta_min) * w_frac

    # --- Stomatal conductance (if enabled) ---
    beta, _ = compute_effective_beta(
        T_soil, forcing, beta_soil, config, carbon_state, dt,
        land_params=lp,
    )

    # Stomatal reduction factor: ratio of effective beta to soil-only beta.
    # This captures the stomatal limitation independent of soil moisture,
    # so it can be applied to updated soil moisture later.
    stomatal_ratio = beta / jnp.maximum(beta_soil, 1e-10)

    # Surface saturation humidity: use ice saturation over snow-covered ground.
    # ``has_snow`` includes fresh snowfall when the surface is below
    # freezing (so the snow survives the step) — same rule as
    # multilayer_land.py iter-68 fix.  Without this, a warm-surface
    # column receiving precip_snow would have routed L_v vapour with
    # a liquid q_sat for the whole step even though the surface is
    # snow-covered.
    q_sat_liq = saturation_mixing_ratio(T_soil, forcing.p_surface)
    q_sat_ice = saturation_mixing_ratio_ice(T_soil, forcing.p_surface)
    # ``fresh_snow_mass`` already computed for the albedo block above.
    has_existing_snow = snow > 1e-6
    has_surviving_fresh_snow = (fresh_snow_mass > 1e-6) & (T_soil < constants.T_freeze)
    has_snow = has_existing_snow | has_surviving_fresh_snow
    q_sat_sfc = jnp.where(has_snow, q_sat_ice, q_sat_liq)
    # Over snow, moisture is freely available from the snowpack (beta=1);
    # water-limiting is applied later via snow mass.  Over bare soil,
    # beta reflects bucket moisture and stomatal limitation.
    beta_effective = jnp.where(has_snow, 1.0, beta)
    q_sfc = beta_effective * q_sat_sfc

    # Phase-appropriate latent heat: sublimation (L_s) over snow, vaporisation (L_v) over bare soil
    L_eff = jnp.where(has_snow, constants.L_s, constants.L_v)

    # Bulk fluxes
    rho = forcing.rho_lowest

    _valid_bulk = ("constant", "most", "coare3", "large_yeager")
    if config.bulk_scheme not in _valid_bulk:
        raise ValueError(
            f"Unknown bulk_scheme {config.bulk_scheme!r}; expected one of {_valid_bulk}."
        )
    if config.bulk_scheme in ("most", "coare3", "large_yeager"):
        tau_x, tau_y, shflx, lhflx, _ = compute_most_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_soil, q_sfc, rho,
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
            T_soil, q_sfc, rho, wind_speed,
            config.Cd_land, config.Ch_land,
            L_latent=L_eff,
        )

    # Radiation
    sw_net, lw_net, _ = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_soil, alpha,
        emissivity,
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
    heat_cap = C_soil * d_soil
    melt_energy = snow_melt * constants.L_f / dt  # W/m2 consumed by melt
    dT_dt = (Q_net - melt_energy) / heat_cap
    T_soil_new = T_soil + dt * dT_dt

    # --- Latent mass exchange ---
    # Convert lhflx to mass flux using the phase-appropriate latent heat.
    # lhflx already embeds L_eff (passed to bulk flux), so dividing by
    # L_eff recovers the correct mass flux for either phase.
    evap_rate = lhflx / L_eff  # kg/m2/s (positive = upward)
    precip_rain = forcing.precip_total - forcing.precip_snow
    melt_rate = snow_melt / dt  # kg/m2/s entering liquid budget

    # --- Snow sublimation / deposition ---
    # Over snow: latent exchange removes/adds mass from/to the snowpack.
    # Sublimation (evap_rate>0) limited by available snow after melt.
    # Deposition (evap_rate<0) always accepted (adds to snow).
    snow_after_melt = snow_new  # snow already updated by update_snow
    max_sublim = jnp.maximum(snow_after_melt / dt, 0.0)  # kg/m2/s
    sublim_demand = jnp.where(has_snow, evap_rate, 0.0)
    sublim_actual = jnp.minimum(sublim_demand, max_sublim)
    # Deposition: negative sublim_demand adds to snow (no limit)
    sublim_actual = jnp.where(sublim_demand < 0.0, sublim_demand, sublim_actual)
    snow_new = snow_new - sublim_actual * dt  # sublimation removes, deposition adds
    snow_new = jnp.maximum(snow_new, 0.0)

    # --- Bucket hydrology: Green-Ampt-style infiltration excess + saturation excess
    # Over bare soil: evaporation removes from the bucket.  Over snow: the bucket is
    # not the latent source (snow sublimation handles it), so soil_evap is the bare-
    # soil demand only.  partition_bucket_runoff caps infiltration (Hortonian runoff
    # for the rejected rain) and spills the overfilled bucket (saturation-excess
    # runoff).  limit_evaporation=True throttles soil_evap to the available water and
    # returns soil_evap_actual, which is then used for the latent heat flux below so
    # energy and water stay consistent: precip + melt == dW/dt + E + runoff exactly.
    soil_evap = jnp.where(has_snow, 0.0, evap_rate)
    P_input = precip_rain + melt_rate
    W_new, soil_evap_actual, runoff, _runoff_inf, _runoff_sat = partition_bucket_runoff(
        W, P_input, soil_evap, dt, W_max,
        K_infiltration, infil_suction_boost,
        infiltration_excess=config.infiltration_excess,
    )

    # Total actual mass flux and excess energy
    evap_rate_actual = jnp.where(has_snow, sublim_actual, soil_evap_actual)
    evap_excess_energy = (evap_rate - evap_rate_actual) * L_eff  # W/m2
    # Actual lhflx consistent with water-limited evaporation/sublimation
    lhflx_actual = evap_rate_actual * L_eff

    # Correct soil temperature: energy that couldn't drive evaporation heats soil
    T_soil_new = T_soil_new + dt * evap_excess_energy / heat_cap

    new_state = LandState(
        T_soil=state.T_soil.replace(data=T_soil_new),
        W_bucket=state.W_bucket.replace(data=W_new),
        snow_depth=state.snow_depth.replace(data=snow_new),
        snow_age=state.snow_age.replace(data=snow_age_new),
        runoff=runoff,
    )

    # Post-step albedo: reflects updated snow state for the next atmosphere step
    if config.snow_albedo_feedback and lat is not None:
        # Mirror the pre-step block: snow-free base = per-cell map albedo (CLM
        # PFT) when land_params supplied, else the latitude-band default. Without
        # base_albedo the post-step value silently reverts to the vegetation
        # default over snow-free cells, creating a pre/post discontinuity in the
        # albedo reported to the atmosphere (corrupts the coupled SW balance).
        _base_new = None if lp is None else jnp.broadcast_to(
            albedo_land, T_soil_new.shape
        )
        alpha_new = compute_land_albedo(
            lat, snow_new, snow_age_new, config.land_albedo,
            base_albedo=_base_new,
        )
    else:
        alpha_new = alpha

    # Recompute upward LW with updated temperature and post-step albedo
    _, _, lw_up_new = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_soil_new, alpha_new,
        emissivity,
    )

    # Recompute q_surface from updated T and moisture for consistency.
    # Apply the stomatal reduction factor so that q_surface reflects both
    # soil moisture availability AND stomatal conductance limitation.
    w_frac_new = jnp.clip(W_new / W_max, 0.0, 1.0)
    beta_soil_new = config.beta_min + (1.0 - config.beta_min) * w_frac_new
    beta_new = stomatal_ratio * beta_soil_new
    q_sat_liq_new = saturation_mixing_ratio(T_soil_new, forcing.p_surface)
    q_sat_ice_new = saturation_mixing_ratio_ice(T_soil_new, forcing.p_surface)
    has_snow_new = snow_new > 1e-6
    q_sat_sfc_new = jnp.where(has_snow_new, q_sat_ice_new, q_sat_liq_new)
    # Over snow, moisture is freely available (beta=1)
    beta_effective_new = jnp.where(has_snow_new, 1.0, beta_new)
    q_sfc_new = beta_effective_new * q_sat_sfc_new

    # --- Carbon cycle ---
    if config.carbon.scheme != "none":
        lat_arr = lat if lat is not None else jnp.zeros_like(T_soil)
        # Recompute Farquhar GPP with updated T and moisture so that
        # photosynthesis and respiration use consistent end-of-step state.
        _, gpp_farq_new = compute_effective_beta(
            T_soil_new, forcing, beta_soil_new, config, carbon_state, dt,
            land_params=lp,
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
        T_sfc=T_soil_new,
        albedo=alpha_new,
        emissivity=jnp.full(T_soil.shape, emissivity, dtype=T_soil.dtype),
        z0=jnp.full(T_soil.shape, z0, dtype=T_soil.dtype),
        q_surface=q_sfc_new,
        shflx=shflx,
        lhflx=lhflx_actual,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up_new,
        u_ocean_sfc=jnp.zeros_like(T_soil),
        v_ocean_sfc=jnp.zeros_like(T_soil),
        co2_flux=co2_flux,
        # Slab land: freshwater leaving the column to the ocean is the
        # bucket overflow ``runoff`` (kg/m²/s).  This closes the water
        # budget through the coupler if a downstream consumer wires it.
        freshwater_flux=runoff,
        # Land does not extract heat directly from the ocean.
        ocean_heat_extraction=jnp.zeros_like(T_soil),
        # Land does not exert stress on the ocean.
        ocean_stress_x=jnp.zeros_like(T_soil),
        ocean_stress_y=jnp.zeros_like(T_soil),
        # Phase-aware moisture mass flux: lhflx_actual was computed
        # using L_eff (L_s if snow-covered, L_v otherwise) so dividing
        # by L_eff recovers the correct mass.
        surface_mass_flux=lhflx_actual / L_eff,
        # Land tile does not exchange salt with the ocean directly.
        salt_flux=jnp.zeros_like(T_soil),
    )

    # Carry-dtype stability (mirrors multilayer_land, commit 49e9fa41e): under
    # JAX_ENABLE_X64 float64 forcing / land_params / lat promote the slab
    # updates (T_soil, W_bucket, snow_*, runoff) and the carbon pools to
    # float64 while the carried leaves keep the storage dtype (float32 in the
    # SOTA runs).  lax.scan REQUIRES carry input/output dtypes to match PER
    # LEAF (SurfaceState — land + carbon — is a scan carry; see
    # coupler.init_surface_state), so cast every returned leaf back to the
    # corresponding INPUT leaf's dtype.  Compute stays at working precision;
    # only the STORED carry is coerced.  The response (fluxes to the
    # atmosphere) is NOT cast.  is_leaf treats None as a leaf so an optional
    # field (state.runoff=None on legacy callers) is paired safely, not
    # descended.
    def _pin_to_input_dtype(new, ref):
        if hasattr(ref, "dtype") and hasattr(new, "astype"):
            return new.astype(ref.dtype)
        return new

    new_state = jax.tree.map(_pin_to_input_dtype, new_state, state,
                             is_leaf=lambda x: x is None)
    if carbon_state is not None and carbon_state_new is not None:
        carbon_state_new = jax.tree.map(
            _pin_to_input_dtype, carbon_state_new, carbon_state,
            is_leaf=lambda x: x is None)

    return new_state, response, carbon_state_new
