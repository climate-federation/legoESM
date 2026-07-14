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

Phase 3b: surface scheme dispatch.  ``LandConfig.surface_scheme`` may be
either ``SimpleSEBConfig`` (default — bulk flux on slab T as the skin
temperature) or ``TwoLeafCanopyConfig`` (DifferBESS-style two-leaf
canopy Newton + Picard with an explicit single-layer thermal callback).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm.core.surface_energy import surface_radiation_fluxes
from legoesm.land.bucket_hydrology import partition_bucket_runoff
from legoesm.land.carbon.carbon_cycle import step_carbon
from legoesm.land.carbon.config import CarbonState
from legoesm.land.config import LandConfig
from legoesm.land.snow_budget import update_snow
from legoesm.land.state import LandState
from legoesm.land.stomata_utils import compute_effective_beta
from legoesm.land.surface_scheme import (
    TwoLeafCanopyConfig,
    compute_simple_seb_fluxes,
    compute_two_leaf_canopy_fluxes,
)
from legoesm.land.surface_scheme.two_leaf_canopy import (
    advance_TgC_ema,
    compute_prognostic_lai,
)
from legoesm.surface_albedo import land_albedo as compute_land_albedo
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice

from legoesm import constants


def _get(lp, name: str, fallback):
    """Read a per-column field from ``lp`` if present, else ``fallback``.

    Safe ``getattr(lp, name, fallback)`` form: ``lp`` may be ``LandSurfaceParams``
    (SimpleSEB; full field set) or ``CanopyLandParams`` (TwoLeafCanopy; disjoint
    set) — a missing field falls back to the default rather than raising.
    """
    if lp is None:
        return fallback
    return getattr(lp, name, fallback)


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

    Dispatches between ``SimpleSEBConfig`` (default — bulk flux on the
    slab T) and ``TwoLeafCanopyConfig`` (two-leaf canopy with explicit
    single-layer thermal Picard callback) via
    ``isinstance(config.surface_scheme, TwoLeafCanopyConfig)``.
    """
    if isinstance(config.surface_scheme, TwoLeafCanopyConfig):
        return _step_land_canopy(
            state, forcing, config, U_min, dt,
            lat=lat, carbon_state=carbon_state, doy=doy,
            land_params=land_params)

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
    # Green-Ampt infiltration params are CONFIG-ONLY (no per-cell field on
    # LandSurfaceParams): reading them through ``_get(lp, ...)`` raised
    # AttributeError for every provider-driven run.  Promote to per-cell params
    # only when a provider actually supplies them.
    K_infiltration = config.K_infiltration
    infil_suction_boost = config.infil_suction_boost

    # Moisture availability: smooth ramp from beta_min to 1
    w_frac = jnp.clip(W / W_max, 0.0, 1.0)
    beta_soil = config.beta_min + (1.0 - config.beta_min) * w_frac

    # Shared SimpleSEB surface closure.  Keep the slab-specific post-flux
    # pipeline below, but do not reimplement bulk fluxes / snow-phase humidity /
    # albedo here: the multilayer SimpleSEB path uses this same helper, and it is
    # where the land MOST exchange cap, condensation floor, and albedo fixes live.
    surface_out = compute_simple_seb_fluxes(
        T_surface=T_soil,
        snow=snow,
        snow_age=snow_age,
        beta_soil=beta_soil,
        forcing=forcing,
        land_config=config,
        U_min=U_min,
        lat=lat,
        carbon_state=carbon_state,
        dt=dt,
        land_params=lp,
        albedo_land=albedo_land,
        emissivity=emissivity,
        z0=z0,
    )
    tau_x = surface_out.tau_x
    tau_y = surface_out.tau_y
    shflx = surface_out.shflx
    lhflx = surface_out.lhflx
    alpha = surface_out.albedo
    Q_net = surface_out.G_soil
    stomatal_ratio = surface_out.stomatal_ratio

    # Match compute_simple_seb_fluxes' snow-phase gate for water/energy
    # partitioning after the flux demand has been computed.
    fresh_snow_mass = forcing.precip_snow * dt
    has_existing_snow = snow > 1e-6
    has_surviving_fresh_snow = (fresh_snow_mass > 1e-6) & (T_soil < constants.T_freeze)
    has_snow = has_existing_snow | has_surviving_fresh_snow

    # Phase-appropriate latent heat: sublimation (L_s) over snow, vaporisation (L_v) over bare soil
    L_eff = jnp.where(has_snow, constants.L_s, constants.L_v)

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
    # Runoff scheme dispatch (hardened: unknown -> ValueError on the static
    # config string).  "bucket" (default) is byte-identical; "topmodel" adds
    # the SIMTOP sub-grid saturated fraction + topographic baseflow.  Direct
    # attribute read (LandConfig always carries the field) so a non-LandConfig
    # caller fails LOUDLY rather than silently defaulting to bucket.
    _runoff_scheme = config.runoff_scheme
    if _runoff_scheme == "topmodel":
        from legoesm.land.topmodel_runoff import partition_topmodel_runoff
        W_new, soil_evap_actual, runoff, _runoff_inf, _runoff_sat = partition_topmodel_runoff(
            W, P_input, soil_evap, dt, W_max,
            K_infiltration, infil_suction_boost, config.topmodel,
            infiltration_excess=config.infiltration_excess,
        )
    elif _runoff_scheme == "bucket":
        W_new, soil_evap_actual, runoff, _runoff_inf, _runoff_sat = partition_bucket_runoff(
            W, P_input, soil_evap, dt, W_max,
            K_infiltration, infil_suction_boost,
            infiltration_excess=config.infiltration_excess,
        )
    else:
        raise ValueError(
            f"Unknown land runoff_scheme {_runoff_scheme!r}; "
            "expected one of: 'bucket', 'topmodel'."
        )

    # Total actual mass flux and excess energy
    evap_rate_actual = jnp.where(has_snow, sublim_actual, soil_evap_actual)
    evap_excess_energy = (evap_rate - evap_rate_actual) * L_eff  # W/m2
    # Actual lhflx consistent with water-limited evaporation/sublimation
    lhflx_actual = evap_rate_actual * L_eff

    # Correct soil temperature: energy that couldn't drive evaporation heats soil
    T_soil_new = T_soil_new + dt * evap_excess_energy / heat_cap

    # Advance the optional 30-day TgC EMA on the SimpleSEB path too —
    # if the user later switches surface_scheme to TwoLeafCanopy mid-run
    # the accumulated value is already warm.
    if state.TgC is not None:
        TgC_new = advance_TgC_ema(state.TgC, forcing.T_lowest, dt)
    else:
        TgC_new = None

    new_state = LandState(
        T_soil=state.T_soil.replace(data=T_soil_new),
        W_bucket=state.W_bucket.replace(data=W_new),
        snow_depth=state.snow_depth.replace(data=snow_new),
        snow_age=state.snow_age.replace(data=snow_age_new),
        runoff=runoff,
        TgC=TgC_new,
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
        _, gpp_farq_new, _ = compute_effective_beta(
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
        # Radiometric surface T for the coupler's LW blend: for the no-canopy slab it
        # is the emitting skin temperature (same T that produced ``lw_up_new``).
        T_rad=T_soil_new,
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


# ---------------------------------------------------------------------------
# Slab + Two-leaf canopy surface scheme
# ---------------------------------------------------------------------------

def _step_land_canopy(
    state: LandState,
    forcing: AtmToSurface,
    config: LandConfig,
    U_min: float,
    dt: float,
    lat: jnp.ndarray | None,
    carbon_state: CarbonState | None,
    doy: float,
    land_params,
) -> tuple[LandState, TileResponse, CarbonState | None]:
    """Slab land step with the two-leaf canopy surface scheme.

    The slab model has a single-layer explicit soil thermal update
    (no Richards, no multi-layer thermal solver), so the canopy Picard
    loop receives a closure that computes
    ``Ts_new = Ts_old + dt * G / (C_soil * d_soil)`` for each tentative
    G.  Root-zone moisture stress is the bucket fraction
    ``W / W_max`` (clipped), since there is no per-layer ``theta``
    profile to integrate over.

    The slab cubed-sphere shape ``(6, n, n)`` is flattened to a
    pseudo-columnar ``(6*n*n,)`` axis for the duration of the canopy
    closure (which uses ``jax.vmap`` over the leading axis) and
    reshaped back on return.  1D slab states pass through unchanged.

    Latent-flux cold-start guarding differs from the SimpleSEB slab path
    (``step_land``): the two-leaf canopy bounds spurious condensation via its
    own ``le_cap_mode`` latent-energy cap in ``two_leaf_canopy.py``, so the
    ``LAND_CONDENSATION_FLOOR_W`` floor used on the bulk SimpleSEB path is not
    applied here.
    """
    lp = land_params
    T_soil = state.T_soil.data
    W = state.W_bucket.data
    snow = state.snow_depth.data
    snow_age = state.snow_age.data

    original_shape = T_soil.shape
    is_flat = (T_soil.ndim == 1)

    # --- Spatial parameters (for slab post-flux pipeline) ---
    W_max  = _get(lp, "W_max", config.W_max)
    C_soil = _get(lp, "C_soil", config.C_soil)
    d_soil = _get(lp, "d_soil", config.d_soil)

    # --- Flatten state for the canopy vmap ---
    def _flat(x):
        return x if is_flat else x.reshape(-1)

    T_soil_flat   = _flat(T_soil)
    W_flat        = _flat(W)
    snow_flat     = _flat(snow)
    snow_age_flat = _flat(snow_age)

    # Forcing fields are typically the same shape as T_soil.  Flatten them.
    forcing_flat = jax.tree.map(
        lambda x: _flat(x) if hasattr(x, "shape") and x.ndim >= 1 else x,
        forcing,
    )

    # --- Bucket-derived soil moisture stress (slab analogue of root-zone β) ---
    w_frac = jnp.clip(W_flat / W_max, 0.0, 1.0)
    # ``w_frac_rz`` here is the unclamped bucket fraction — the canopy
    # uses it as fStress_soil / fStress_vcmax directly.  No beta_min
    # floor applied (the canopy module has its own thresholds).
    w_frac_rz = w_frac

    # --- Wind speed and direction (flattened) ---
    wind_speed = jnp.sqrt(
        forcing_flat.u_lowest ** 2
        + forcing_flat.v_lowest ** 2
        + U_min ** 2)
    wind_dir_x = forcing_flat.u_lowest / jnp.maximum(wind_speed, 1e-6)
    wind_dir_y = forcing_flat.v_lowest / jnp.maximum(wind_speed, 1e-6)

    # --- Slab thermal callback for the canopy Picard loop ---
    # Captures the start-of-step T_soil and slab heat capacity so each
    # Picard iteration can compute a fresh tentative end-of-step T_soil
    # from the candidate G value.
    heat_cap_total = C_soil * d_soil  # J / m^2 / K

    def _slab_thermal_cb(G, dt_):
        return T_soil_flat + dt_ * G / heat_cap_total

    # State-carried TgC EMA (flattened) takes precedence over any
    # ``CanopyLandParams.TgC`` override (which the canopy uses if
    # ``TgC_override`` is None).
    TgC_override = _flat(state.TgC) if state.TgC is not None else None

    # Phase 6 / Stage 2b: prognostic LAI feedback.  ``compute_prognostic_lai``
    # returns ``C_fol / LCMA`` if differland carbon + use_prognostic_lai
    # are both active (the flag defaults to False; enable explicitly on
    # the canopy surface scheme to opt in).  For slab on cubed-sphere
    # shapes, the carbon_state is (6, n, n) matching the slab state; we
    # flatten it for the canopy vmap and unflatten is not needed because
    # LAI_override is only read inside the canopy compute function.
    _lai_full = compute_prognostic_lai(carbon_state, config, config.surface_scheme)
    LAI_override = _flat(_lai_full) if _lai_full is not None else None

    # --- Canopy surface flux closure ---
    surface_out = compute_two_leaf_canopy_fluxes(
        T_soil_top=T_soil_flat,
        forcing=forcing_flat,
        canopy_config=config.surface_scheme,
        land_config=config,
        canopy_params=lp,
        w_frac_rz=w_frac_rz,
        wind_speed=wind_speed,
        wind_dir_x=wind_dir_x,
        wind_dir_y=wind_dir_y,
        soil_thermal_fn=_slab_thermal_cb,
        dt=dt,
        TgC_override=TgC_override,
        LAI_override=LAI_override,
    )

    # --- Slab post-flux: snow, bucket, T_soil dT/dt (flattened) ---
    has_snow = snow_flat > 1e-6
    G_surface = surface_out.G_soil  # canopy-converged ground heat flux

    snow_new, snow_age_new, snow_melt = update_snow(
        snow_flat, snow_age_flat, T_soil_flat, _flat(forcing.precip_snow), dt,
        Q_net=G_surface,
        snow_melt_rate=config.snow_melt_rate,
        T_snow_melt=config.T_snow_melt,
    )
    melt_energy = snow_melt * constants.L_f / dt

    # Slab energy balance: net energy into soil = G - melt_energy.
    # NOTE: the canopy already accounts for SW/LW/SH/LE in G; melt is the
    # only additional sink at this layer.
    Q_net_slab = G_surface - melt_energy
    T_soil_new = T_soil_flat + dt * Q_net_slab / heat_cap_total

    # --- Latent mass partition ---
    L_eff = jnp.where(has_snow, constants.L_s, constants.L_v)
    evap_rate_demand = surface_out.lhflx / L_eff

    snow_after_melt = snow_new
    max_sublim = jnp.maximum(snow_after_melt / dt, 0.0)
    sublim_demand = jnp.where(has_snow, evap_rate_demand, 0.0)
    sublim_actual = jnp.minimum(sublim_demand, max_sublim)
    sublim_actual = jnp.where(sublim_demand < 0.0, sublim_demand, sublim_actual)
    snow_new = jnp.maximum(snow_new - sublim_actual * dt, 0.0)

    precip_rain = _flat(forcing.precip_total) - _flat(forcing.precip_snow)
    melt_rate = snow_melt / dt

    soil_evap_demand = jnp.where(has_snow, 0.0, evap_rate_demand)
    P_input = precip_rain + melt_rate
    _runoff_scheme = config.runoff_scheme
    if _runoff_scheme == "topmodel":
        from legoesm.land.topmodel_runoff import partition_topmodel_runoff
        W_new, soil_evap_actual, runoff, _runoff_inf, _runoff_sat = partition_topmodel_runoff(
            W_flat, P_input, soil_evap_demand, dt, W_max,
            config.K_infiltration, config.infil_suction_boost, config.topmodel,
            infiltration_excess=config.infiltration_excess,
        )
    elif _runoff_scheme == "bucket":
        W_new, soil_evap_actual, runoff, _runoff_inf, _runoff_sat = partition_bucket_runoff(
            W_flat, P_input, soil_evap_demand, dt, W_max,
            config.K_infiltration, config.infil_suction_boost,
            infiltration_excess=config.infiltration_excess,
        )
    else:
        raise ValueError(
            f"Unknown land runoff_scheme {_runoff_scheme!r}; "
            "expected one of: 'bucket', 'topmodel'."
        )
    evap_rate_actual = jnp.where(has_snow, sublim_actual, soil_evap_actual)
    evap_excess_energy = (evap_rate_demand - evap_rate_actual) * L_eff
    lhflx_actual = evap_rate_actual * L_eff

    # Excess (unrealised) latent flux warms the slab.
    T_soil_new = T_soil_new + dt * evap_excess_energy / heat_cap_total

    # --- Reshape back to original (cubed-sphere or columnar) shape ---
    def _unflat(x):
        if x is None:
            return None
        if is_flat:
            return x
        return x.reshape(original_shape)

    T_soil_new   = _unflat(T_soil_new)
    W_new        = _unflat(W_new)
    snow_new     = _unflat(snow_new)
    snow_age_new = _unflat(snow_age_new)
    runoff       = _unflat(runoff)
    lhflx_actual_full = _unflat(lhflx_actual)

    # Advance state-carried TgC EMA if present.
    if state.TgC is not None:
        TgC_new = advance_TgC_ema(state.TgC, forcing.T_lowest, dt)
    else:
        TgC_new = None

    new_state = LandState(
        T_soil=state.T_soil.replace(data=T_soil_new),
        W_bucket=state.W_bucket.replace(data=W_new),
        snow_depth=state.snow_depth.replace(data=snow_new),
        snow_age=state.snow_age.replace(data=snow_age_new),
        runoff=runoff,
        TgC=TgC_new,
    )

    # Post-step coupler-facing surface state (re-uses canopy-derived
    # albedo and emissivity from surface_out).  T_soil_new is already
    # unflattened above; surface_out fields are still flat.
    # ``T_sfc`` is the AERODYNAMIC surface temperature for the atmosphere's
    # sensible-heat coupling: the canopy air-space temperature ``Tc`` (the
    # exchange node, H_tot = rho*cp*(Tc - Ta)/Ra), NOT the soil temperature.
    # The LW-derived radiometric ``T_surface`` is carried separately as T_rad.
    # (``T_soil_new`` remains the soil thermal prognostic in the land state.)
    response_T_surface     = _unflat(surface_out.T_canopy_air)
    response_albedo        = _unflat(surface_out.albedo)
    response_emissivity    = _unflat(surface_out.emissivity)
    response_z0            = _unflat(surface_out.z0)
    response_q_surface     = _unflat(surface_out.q_surface)
    response_shflx         = _unflat(surface_out.shflx)
    response_tau_x         = _unflat(surface_out.tau_x)
    response_tau_y         = _unflat(surface_out.tau_y)
    response_lw_up         = _unflat(surface_out.lw_up)

    # --- Carbon cycle ---
    if config.carbon.scheme != "none":
        lat_arr = lat if lat is not None else jnp.zeros_like(T_soil)
        if surface_out.gpp is not None:
            gpp_override = _unflat(surface_out.gpp)
        else:
            gpp_override = None
        # Slab carbon path: pass start-of-step bucket beta as the
        # moisture forcing (consistent with the existing SimpleSEB path).
        beta_soil_new = config.beta_min + (1.0 - config.beta_min) * jnp.clip(
            W_new / W_max, 0.0, 1.0)
        carbon_state_new, co2_flux = step_carbon(
            carbon_state, forcing.sw_down, T_soil_new, forcing.co2_ppmv,
            beta_soil_new, lat_arr, doy, forcing.precip_total,
            config.carbon, dt, gpp_override=gpp_override,
        )
    else:
        carbon_state_new = carbon_state
        co2_flux = jnp.zeros_like(T_soil_new)

    response = TileResponse(
        T_sfc=response_T_surface,
        # Emission-equivalent canopy temperature for the LW boundary: the
        # two-leaf ``surface_out.T_surface`` satisfies eps_col*sigma*T_surface^4
        # = LW_emit, whereas ``T_sfc`` above is the aerodynamic canopy air-space
        # temperature ``Tc`` used by the (linear) sensible-heat path.  The tile
        # blend MUST emit with this radiometric T_rad, not Tc, or LW conservation
        # breaks for vegetated cells.
        T_rad=_unflat(surface_out.T_surface),
        albedo=response_albedo,
        emissivity=response_emissivity,
        z0=response_z0,
        q_surface=response_q_surface,
        shflx=response_shflx,
        lhflx=lhflx_actual_full,
        tau_x=response_tau_x,
        tau_y=response_tau_y,
        lw_up=response_lw_up,
        u_ocean_sfc=jnp.zeros_like(T_soil_new),
        v_ocean_sfc=jnp.zeros_like(T_soil_new),
        co2_flux=co2_flux,
        # Freshwater leaving the column to the ocean = bucket overflow
        # runoff [kg/m²/s]; closes the coupler water budget.
        freshwater_flux=runoff,
        # Land does not extract heat or exert stress on the ocean.
        ocean_heat_extraction=jnp.zeros_like(T_soil_new),
        ocean_stress_x=jnp.zeros_like(T_soil_new),
        ocean_stress_y=jnp.zeros_like(T_soil_new),
        # Phase-aware moisture mass flux (kg/m²/s): the actual evaporation
        # rate that produced lhflx_actual_full.
        surface_mass_flux=_unflat(evap_rate_actual),
        # Land tile does not exchange salt with the ocean directly.
        salt_flux=jnp.zeros_like(T_soil_new),
    )

    return new_state, response, carbon_state_new
