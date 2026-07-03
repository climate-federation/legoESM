"""Multi-layer soil land model with pluggable surface scheme.

Dispatches between:

- ``SimpleSEBConfig`` (default): bulk-flux surface energy balance with
  skin T = ``T_soil[:, 0]``, optional Jarvis / Leuning stomatal coupling.
- ``TwoLeafCanopyConfig``: DifferBESS-style two-leaf canopy Newton +
  Picard closure on top of the same soil column.

Both surface schemes produce a ``SurfaceFluxOutput``; the post-flux
pipeline (snow, Richards, soil thermal, carbon, TileResponse) is shared.

Physics sequence each time step:

1. Pre-flux: root distribution, root-zone soil moisture stress,
   spatial parameter resolution.
2. Surface scheme dispatch: compute ``SurfaceFluxOutput`` (``shflx``,
   ``lhflx``, ``G_soil``, ``gpp``, ``tau_*``, surface state).
3. Snow budget (energy-limited melt).
4. Sublimation vs soil evaporation partition (water-limited).
5. Richards equation for soil moisture.
6. Soil thermal diffusion with the converged ground heat flux.
7. Post-step surface state (``q_surface``, ``lw_up``) for the coupler.
8. Carbon cycle (GPP override from surface scheme).
9. Build ``TileResponse``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice
from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm.core.surface_energy import surface_radiation_fluxes
from legoesm.land.carbon.config import CarbonState
from legoesm.land.carbon.carbon_cycle import step_carbon
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.snow_budget import update_snow
from legoesm.land.state import MultiLayerLandState
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.stomata_utils import compute_effective_beta
from legoesm.land.richards import solve_richards
from legoesm.land.soil_thermal import solve_soil_thermal
from legoesm.land.surface_scheme import (
    SimpleSEBConfig,
    TwoLeafCanopyConfig,
    compute_simple_seb_fluxes,
    compute_two_leaf_canopy_fluxes,
)
from legoesm.land.surface_scheme.two_leaf_canopy import (
    advance_TgC_ema,
    compute_prognostic_lai,
)
from legoesm.surface_albedo import land_albedo as compute_land_albedo


def _get(lp, name: str, fallback):
    """Read a per-column field from ``lp`` if present, else return ``fallback``.

    Uses the safe ``getattr(lp, name, fallback)`` form because ``lp`` can be
    either ``LandSurfaceParams`` (for SimpleSEB; full field set) or
    ``CanopyLandParams`` (for TwoLeafCanopy; disjoint field set).  Missing
    fields fall back to the caller-supplied default rather than raising.
    """
    if lp is None:
        return fallback
    return getattr(lp, name, fallback)


def step_multilayer_land_with_diagnostics(
    state: MultiLayerLandState,
    forcing: AtmToSurface,
    config: MultiLayerLandConfig,
    U_min: float,
    dt: float,
    lat: jnp.ndarray | None = None,
    carbon_state: CarbonState | None = None,
    doy: float = 0.0,
    land_params=None,
):
    """Like :func:`step_multilayer_land` but also returns the ``SurfaceFluxOutput``.

    The 4th element is the surface scheme's ``SurfaceFluxOutput`` with all
    scheme-specific diagnostic fields populated (``Tf_Sun``, ``Tf_Sh``,
    ``gs_Sun``, ``n_iters``, ``f_veg`` for the canopy scheme; the common
    flux / radiation / state fields for both schemes).  Intended for
    offline diagnostic runs — no performance cost beyond the extra pytree
    allocation.
    """
    return _step_multilayer_land_impl(
        state, forcing, config, U_min, dt,
        lat=lat, carbon_state=carbon_state, doy=doy, land_params=land_params)


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
    """Step the multi-layer land model forward by ``dt`` seconds.

    Dispatches between ``SimpleSEBConfig`` and ``TwoLeafCanopyConfig``
    surface schemes via ``isinstance(config.surface_scheme, ...)``.  The
    post-flux pipeline (snow, Richards, soil thermal, carbon, TileResponse)
    is shared between both branches.  Returns a 3-tuple; use
    :func:`step_multilayer_land_with_diagnostics` to also receive the raw
    ``SurfaceFluxOutput`` for diagnostic inspection.
    """
    new_state, response, carbon_new, _surface_out = _step_multilayer_land_impl(
        state, forcing, config, U_min, dt,
        lat=lat, carbon_state=carbon_state, doy=doy, land_params=land_params)
    return new_state, response, carbon_new


def _step_multilayer_land_impl(
    state: MultiLayerLandState,
    forcing: AtmToSurface,
    config: MultiLayerLandConfig,
    U_min: float,
    dt: float,
    lat: jnp.ndarray | None = None,
    carbon_state: CarbonState | None = None,
    doy: float = 0.0,
    land_params=None,
):
    """Internal 4-tuple (new_state, TileResponse, carbon, SurfaceFluxOutput).

    Kept non-public so the two public entry points (``step_multilayer_land``
    and ``step_multilayer_land_with_diagnostics``) can return different
    arities without branching inside the tight-loop code.
    """
    lp = land_params
    T_soil = state.T_soil        # (ncol, n_layers)
    psi = state.psi_soil         # (ncol, n_layers)
    theta = state.theta_soil     # (ncol, n_layers)
    snow = state.snow_depth      # (ncol,)
    snow_age = state.snow_age    # (ncol,)

    grid = make_soil_grid(config.soil_grid)

    # Spatially-varying surface parameters (or config scalar fallbacks).
    albedo_land = _get(lp, "albedo_veg", config.albedo_land)
    emissivity = _get(lp, "emissivity", config.emissivity_land)
    z0 = _get(lp, "z0", config.z0_land)

    # Spatially-varying root zone params.
    root_depth = _get(lp, "root_depth", config.root_depth)
    theta_wp = _get(lp, "theta_wp", config.theta_wp)
    theta_fc = _get(lp, "theta_fc", config.theta_fc)

    # Start-of-step skin temperature = top soil layer.
    T_surface = T_soil[:, 0]
    ncol = T_surface.shape[0]

    # Smooth wind speed floor.
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2)
    wind_dir_x = forcing.u_lowest / jnp.maximum(wind_speed, 1e-6)
    wind_dir_y = forcing.v_lowest / jnp.maximum(wind_speed, 1e-6)

    # --- Root distribution and per-layer moisture stress (shared) ---
    theta_r = config.hydraulics.theta_r
    z_centers = grid.z_node

    # ``root_depth``, ``theta_wp``, ``theta_fc`` may be scalars (from the
    # config) or per-column arrays (from ``LandSurfaceParams``).  Handle
    # both by promoting to (ncol,) shape.
    def _to_ncol(v):
        arr = jnp.asarray(v)
        if arr.ndim == 0:
            return jnp.broadcast_to(arr, (ncol,))
        return arr

    root_depth_c = _to_ncol(root_depth)
    theta_wp_c   = _to_ncol(theta_wp)
    theta_fc_c   = _to_ncol(theta_fc)

    root_frac = jnp.exp(-z_centers[None, :] / root_depth_c[:, None])
    root_frac = root_frac / jnp.sum(root_frac, axis=-1, keepdims=True)
    # Audit #6 / Iter-65: floor the (theta_fc - theta_wp) range at 1e-3
    # m³/m³ (~1 % of theta_sat).  Earlier ``+ 1e-10`` only protected against
    # exact equality; a misconfigured cell with theta_fc ≈ theta_wp (e.g. a
    # pathological PFT lookup row) still produced exploding beta_root because
    # the denominator could go ~O(theta).  Ported from main during the
    # jianing/land ↔ main sync (2026-06-03).
    _denom = jnp.maximum(
        theta_fc_c[:, None] - theta_wp_c[:, None], 1e-3,
    )
    beta_root = jnp.clip(
        (theta - theta_wp_c[:, None]) / _denom,
        0.0, 1.0,
    )
    w_frac_rz = jnp.clip(jnp.sum(root_frac * beta_root, axis=-1), 0.0, 1.0)

    beta_soil = config.beta_min + (1.0 - config.beta_min) * w_frac_rz

    # Top-layer effective saturation S_top = (theta0 - theta_r)/(theta_sat - theta_r),
    # used by BOTH surface schemes' bare-soil-evaporation resistance.  Compute it in
    # the (ncol, nlayers) layer space THEN take layer 0, so a per-column (ncol, 1)
    # theta_r / theta_sat (spatial hydraulics, e.g. the coupled land tile) broadcasts
    # against the 2-D theta rather than outer-producting against the 1-D theta[:, 0]
    # to (ncol, ncol) — the shape bug behind the coupled multilayer-land driver crash.
    # .astype(theta.dtype): a per-column theta_r/theta_sat may be float64 while the
    # coupled land state theta is float32 — keep S_top in the state precision so the
    # bare-soil-evap throttle does not silently promote the step output to float64
    # (the scan carry requires input/output dtypes to match).
    _S_top = jnp.clip(
        (theta - theta_r)
        / jnp.maximum(config.hydraulics.theta_sat - theta_r, 1e-6),
        1e-6, 1.0)[:, 0].astype(theta.dtype)

    # =================================================================
    # Surface scheme dispatch
    # =================================================================
    if isinstance(config.surface_scheme, TwoLeafCanopyConfig):
        # Canopy surface scheme: Newton closure with Picard loop that
        # advances soil thermal tentatively between passes.
        def _soil_thermal_cb(G, dt_):
            T_tent = solve_soil_thermal(
                T_soil, theta, grid,
                config.hydraulics, config.thermal,
                G, dt_,
            )
            return T_tent[:, 0]

        # State-carried 30-day TgC EMA takes precedence over any
        # ``CanopyLandParams.TgC`` override (which the canopy uses if
        # ``TgC_override`` is None).
        TgC_override = state.TgC

        # Phase 6 / Stage 2b: prognostic LAI feedback.  When
        # ``use_prognostic_lai`` (opt-in; off by default) and
        # differland carbon is active, LAI = C_fol / LCMA takes
        # precedence over any prescribed ``CanopyLandParams.LAI``.
        LAI_override = compute_prognostic_lai(
            carbon_state, config, config.surface_scheme)

        surface_out = compute_two_leaf_canopy_fluxes(
            T_soil_top=T_surface,
            forcing=forcing,
            canopy_config=config.surface_scheme,
            land_config=config,
            canopy_params=lp,
            w_frac_rz=w_frac_rz,
            wind_speed=wind_speed,
            wind_dir_x=wind_dir_x,
            wind_dir_y=wind_dir_y,
            soil_thermal_fn=_soil_thermal_cb,
            dt=dt,
            TgC_override=TgC_override,
            LAI_override=LAI_override,
            # Bare-soil evaporation efficiency = TWO complementary top-layer
            # limiters, applied as a beta conductance efficiency in the canopy
            # soil energy balance (both tie evaporation to the fast-drying
            # SURFACE, not the root-zone average):
            #   * Kelvin pore RELATIVE HUMIDITY  h_r = exp(psi_top g /(R_v T))
            #     — thermodynamic vapour-pressure lowering; only bites as the
            #     surface approaches residual (psi -> -inf).
            #   * Sellers-1992 / Lee-Pielke-1992 diffusion-crust resistance,
            #     S_top**soil_evap_resistance_exp with S_top the top-layer
            #     effective saturation — throttles evaporation even when the
            #     surface is WET (S_top<1), the regime the EC + DifferBESS
            #     comparison showed over-predicts soil evaporation several-fold.
            # This is the canopy-path counterpart of the SimpleSEB S_top**exp
            # throttle (#671); exp=0 recovers the Kelvin-only behaviour.
            w_frac_soil_evap=(
                jnp.exp(jnp.minimum(
                    psi[:, 0] * constants.g
                    / (constants.R_v * jnp.maximum(T_soil[:, 0], 1.0)), 0.0))
                # _S_top (top-layer effective saturation) is floored at 1e-6 (not 0)
                # in its shared definition above: keeps d(S_top**exp)/dS_top finite at
                # the residual-water boundary for a trainable exp < 1 (0**exp has an
                # infinite gradient) — AD-safe, negligible forward effect.
                * _S_top ** config.soil_evap_resistance_exp),
        )
    else:
        # SimpleSEB: bulk fluxes with skin T = T_soil[:, 0].
        surface_out = compute_simple_seb_fluxes(
            T_surface=T_surface,
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

    # =================================================================
    # Shared post-flux pipeline
    # =================================================================
    shflx = surface_out.shflx
    lhflx = surface_out.lhflx
    tau_x = surface_out.tau_x
    tau_y = surface_out.tau_y
    G_surface = surface_out.G_soil

    # --- Snow phase (iter-68 — consistent with simple_seb's flux calc) ---
    # Reuse the same warm-surface-snowfall gate as simple_seb.py so
    # downstream latent-mass partition (sublimation vs soil evap) is
    # consistent with the L_eff that produced the demand.  Ported from
    # main during the jianing/land ↔ main sync 2026-06-03.
    fresh_snow_mass = forcing.precip_snow * dt
    has_existing_snow = snow > 1e-6
    has_surviving_fresh_snow = (
        (fresh_snow_mass > 1e-6) & (T_surface < constants.T_freeze)
    )
    has_snow = has_existing_snow | has_surviving_fresh_snow

    # --- Snow budget (energy-limited melt) ---
    snow_new, snow_age_new, snow_melt = update_snow(
        snow, snow_age, T_surface, forcing.precip_snow, dt,
        Q_net=G_surface,
        snow_melt_rate=config.snow_melt_rate,
        T_snow_melt=config.T_snow_melt,
    )
    melt_energy = snow_melt * constants.L_f / dt
    G_surface = G_surface - melt_energy

    # --- Latent mass partition (sublimation vs soil evap, water-limited) ---
    rho_w = constants.rho_water
    L_eff = jnp.where(has_snow, constants.L_s, constants.L_v)
    evap_rate_demand = lhflx / L_eff

    snow_after_melt = snow_new
    max_sublim = jnp.maximum(snow_after_melt / dt, 0.0)
    sublim_demand = jnp.where(has_snow, evap_rate_demand, 0.0)
    sublim_actual = jnp.minimum(sublim_demand, max_sublim)
    sublim_actual = jnp.where(sublim_demand < 0.0, sublim_demand, sublim_actual)
    snow_new = jnp.maximum(snow_new - sublim_actual * dt, 0.0)

    dz = grid.dz
    extractable_water = jnp.sum(
        jnp.maximum(theta - theta_r, 0.0) * dz[None, :], axis=-1) * rho_w
    precip_rain = forcing.precip_total - forcing.precip_snow
    melt_rate = snow_melt / dt
    soil_evap_demand = jnp.where(has_snow, 0.0, evap_rate_demand)
    # Bare-soil evaporation resistance (#671, Sellers 1992 / Lee & Pielke 1992):
    # throttle the (positive, evaporative) bare-soil demand by the TOP-layer
    # effective saturation S_top**exp — the surface dries into a high-resistance
    # crust far faster than the root-zone mean.  GATED to SimpleSEB: the two-leaf
    # canopy path applies its OWN top-layer soil-evap throttle (Kelvin h_r) inside
    # the canopy energy balance, so surface_out.lhflx already reflects it; applying
    # S_top**exp again here would double-throttle AND wrongly throttle the canopy
    # transpiration folded into the total lhflx.  Dew (demand<0) left un-throttled.
    if not isinstance(config.surface_scheme, TwoLeafCanopyConfig):
        # _S_top (top-layer effective saturation, floored at 1e-6 in its shared
        # definition above so d(S_top**exp)/dS_top stays finite at the residual-water
        # boundary for a trainable exp < 1; AD-safe, negligible fwd).
        _beta_surf = (_S_top ** config.soil_evap_resistance_exp).astype(
            soil_evap_demand.dtype)
        soil_evap_demand = jnp.where(
            soil_evap_demand > 0.0, soil_evap_demand * _beta_surf, soil_evap_demand)
    max_soil_evap = jnp.maximum(
        extractable_water / dt + precip_rain + melt_rate, 0.0)
    soil_evap = jnp.minimum(soil_evap_demand, max_soil_evap)

    evap_rate = jnp.where(has_snow, sublim_actual, soil_evap)
    evap_excess_energy = (evap_rate_demand - evap_rate) * L_eff
    lhflx_actual = evap_rate * L_eff

    # --- Root water uptake partition ---
    # iter-23 + dew handling (ported from main 2026-06-03):
    # (a) Snow gate: the snowpack already swallowed sublim_actual upstream
    #     (line above), so the soil should NOT see any latent flux when
    #     ``has_snow`` is True — routing snow deposition through
    #     ``flux_top`` would double-count the mass (codex iter-23
    #     stop-time review).
    # (b) Dew handling: within the snow-free regime, negative ``evap_rate``
    #     (dew / downward deposition on bare soil) routes ENTIRELY to
    #     ``flux_top`` so the column water budget closes; transpiration
    #     sink is set to 0.  Vegetated-fraction dew on a snow-free cell
    #     is treated as bare-soil input (no separate canopy-storage
    #     reservoir in this model).
    f_veg = jnp.clip(w_frac_rz, 0.0, 1.0)
    soil_flux = jnp.where(has_snow, 0.0, evap_rate)
    is_dew = soil_flux < 0.0
    evap_bare = jnp.where(is_dew, soil_flux, soil_flux * (1.0 - f_veg))
    evap_transp = jnp.where(is_dew, 0.0, soil_flux * f_veg)
    flux_top = (precip_rain + melt_rate - evap_bare) / rho_w

    E_pot_transp = jnp.maximum(evap_transp, 0.0) / rho_w
    weight = root_frac * beta_root  # both are (ncol, n_layers)
    weight_sum = jnp.sum(weight, axis=-1, keepdims=True)
    weight_norm = weight / jnp.maximum(weight_sum, 1e-20)
    sink = weight_norm * E_pot_transp[:, None] / dz[None, :]

    # --- Richards equation (+ coupled surface ponding cell, #671) ---
    richards_out = solve_richards(
        psi, theta, grid,
        config.hydraulics, config.richards,
        flux_top, sink, dt,
        surface_water=state.surface_water,
    )
    # NB (#671): the former "evaporation water budget closure" — a clip of
    # theta_new to [theta_r, theta_sat] — is removed.  It was a non-conservative
    # band-aid for the old infiltration/evap mismatch; theta_from_psi is now
    # bounded below at theta_r by construction, the coupled surface cell carries
    # the ponded excess, and the mixed-form solve closes the water budget.

    # --- Soil thermal diffusion (final, with converged G) ---
    # Semi-implicit surface conductance (Robin BC): the SimpleSEB scheme returns a
    # linearised lambda = -dG/dT_sfc that makes the surface energy balance's
    # T_sfc-dependence implicit here, removing the explicit-coupling large-dt/thin-
    # top-layer instability.  None for the two-leaf canopy (its Newton closure owns
    # the coupling) and for slab builds that leave it unset -> explicit BC, unchanged.
    G_surface = G_surface + evap_excess_energy
    T_soil_new = solve_soil_thermal(
        T_soil, richards_out.theta_new, grid,
        config.hydraulics, config.thermal,
        G_surface, dt,
        surface_conductance=surface_out.surface_conductance,
    )

    # --- Advance the 30-day TgC EMA (only when state carries it) ---
    if state.TgC is not None:
        TgC_new = advance_TgC_ema(state.TgC, forcing.T_lowest, dt)
    else:
        TgC_new = None

    # --- Build new state ---
    # Preserve the INPUT state's precision.  The Richards + soil-thermal solves run
    # in an internal working precision that is float64 whenever the hydraulics config
    # carries float64 params (x64 enabled), even for a float32 coupled land state — so
    # cast each updated leaf back to its input dtype.  Without this the multilayer step
    # returns float64 leaves for a float32 carry and the driver's lax.scan rejects the
    # segment ("carry input/output dtypes must match").  No-op for a uniform-precision
    # (offline / test) state, where input dtype == working dtype.
    def _match(new, old):
        # Cast an updated leaf back to its input dtype (None-safe on either side —
        # e.g. TgC is None for the SimpleSEB scheme).
        if new is None or old is None:
            return new
        return new.astype(old.dtype)

    new_state = MultiLayerLandState(
        T_soil=_match(T_soil_new, state.T_soil),
        psi_soil=_match(richards_out.psi_new, state.psi_soil),
        theta_soil=_match(richards_out.theta_new, state.theta_soil),
        runoff_surface=_match(richards_out.runoff_surface, state.runoff_surface),
        runoff_subsurface=_match(richards_out.runoff_subsurface,
                                 state.runoff_subsurface),
        snow_depth=_match(snow_new, state.snow_depth),
        snow_age=_match(snow_age_new, state.snow_age),
        TgC=_match(TgC_new, state.TgC),
        surface_water=_match(richards_out.surface_water,   # coupled ponding cell (#671)
                             state.surface_water),
    )

    # --- Post-step surface state for coupler ---
    T_surface_new = T_soil_new[:, 0]

    if config.snow_albedo_feedback and lat is not None:
        alpha_new = compute_land_albedo(
            lat, snow_new, snow_age_new, config.land_albedo)
    else:
        alpha_new = surface_out.albedo

    # lw_up recomputed with post-step surface T and surface scheme's
    # effective emissivity (canopy RT vs scalar land emissivity).
    _, _, lw_up_new = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_surface_new, alpha_new,
        emissivity,
    )

    # --- Post-step q_surface ---
    # Reuse the same Audit #6 / Iter-65 1e-3 floor as the pre-step branch
    # above so degenerate PFT cells cannot blow up beta_root_new propagating
    # into the q_surface reported back to the atmosphere.
    theta_new = richards_out.theta_new
    _denom_new = jnp.maximum(
        theta_fc_c[:, None] - theta_wp_c[:, None], 1e-3,
    )
    beta_root_new = jnp.clip(
        (theta_new - theta_wp_c[:, None]) / _denom_new,
        0.0, 1.0,
    )
    w_frac_rz_new = jnp.clip(
        jnp.sum(root_frac * beta_root_new, axis=-1), 0.0, 1.0)
    beta_soil_new = config.beta_min + (1.0 - config.beta_min) * w_frac_rz_new
    # SimpleSEB: stomatal_ratio carries the stomatal limitation through
    # the updated moisture state.  Canopy: stomatal_ratio = 1 (LE is
    # computed from leaf-level gradients, not via beta * q_sat).
    stom_ratio = surface_out.stomatal_ratio
    if stom_ratio is None:
        stom_ratio = jnp.ones_like(beta_soil_new)
    beta_new = stom_ratio * beta_soil_new
    q_sat_liq_new = saturation_mixing_ratio(T_surface_new, forcing.p_surface)
    q_sat_ice_new = saturation_mixing_ratio_ice(T_surface_new, forcing.p_surface)
    has_snow_new = snow_new > 1e-6
    q_sat_sfc_new = jnp.where(has_snow_new, q_sat_ice_new, q_sat_liq_new)
    beta_effective_new = jnp.where(has_snow_new, 1.0, beta_new)
    q_sfc_new = beta_effective_new * q_sat_sfc_new

    # --- Carbon cycle ---
    if config.carbon.scheme != "none":
        lat_arr = lat if lat is not None else jnp.zeros(ncol)
        if surface_out.gpp is not None:
            # Canopy or SimpleSEB+stomata both populate this.
            gpp_override = surface_out.gpp
        else:
            # Neither scheme produced GPP (stomata disabled, carbon=none?).
            # Re-derive via compute_effective_beta on post-step state so
            # the carbon cycle sees a consistent end-of-step GPP.
            _, gpp_override = compute_effective_beta(
                T_surface_new, forcing, beta_soil_new, config, carbon_state,
                dt, land_params=lp)
        carbon_state_new, co2_flux = step_carbon(
            carbon_state, forcing.sw_down, T_surface_new, forcing.co2_ppmv,
            beta_soil_new, lat_arr, doy, forcing.precip_total, config.carbon,
            dt, gpp_override=gpp_override,
        )
    else:
        carbon_state_new = carbon_state
        co2_flux = jnp.zeros(ncol)

    # --- TileResponse ---
    # ``T_sfc`` is the AERODYNAMIC surface temperature for the atmosphere's
    # sensible-heat coupling.  For the two-leaf canopy that is the canopy
    # air-space temperature ``Tc`` (the exchange node, H_tot = rho*cp*(Tc-Ta)/Ra);
    # for SimpleSEB it is the top-soil surface temperature ``T_surface_new``.
    # Static dispatch on the (compile-time) surface-scheme type — feature gating,
    # not data-dependent selection.
    if isinstance(config.surface_scheme, TwoLeafCanopyConfig):
        response_T_sfc = surface_out.T_canopy_air   # aerodynamic (canopy air-space temp)
        # Upward LW MUST be the canopy's conservative top-of-canopy LW_out (the
        # flux consistent with T_rad / eps_col), NOT a recomputation from the
        # post-step top-SOIL temperature — otherwise the tile blend, the gray
        # brightness temperature (sigma*T_bb^4 = lw_up), and lw_net all carry a
        # soil-based flux that discards the canopy radiative state.  (The slab
        # canopy path likewise reports surface_out.lw_up.)
        response_lw_up = surface_out.lw_up          # canopy LW_out
    else:
        response_T_sfc = T_surface_new              # SimpleSEB: top-soil surface temp
        response_lw_up = lw_up_new                  # recomputed from post-step skin T
    response = TileResponse(
        T_sfc=response_T_sfc,
        # Emission-equivalent canopy temperature for the LW boundary
        # (eps_col*sigma*T_surface^4 = LW_emit); T_sfc above is the aerodynamic
        # (Tc for canopy / top-soil for SimpleSEB) temperature for sensible heat.
        # The tile blend emits with this radiometric T_rad, not the aerodynamic
        # temp, to conserve LW for vegetated cells.
        T_rad=surface_out.T_surface,
        albedo=alpha_new,
        emissivity=surface_out.emissivity,
        z0=surface_out.z0,
        q_surface=q_sfc_new,
        shflx=shflx,
        lhflx=lhflx_actual,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=response_lw_up,
        u_ocean_sfc=jnp.zeros(ncol),
        v_ocean_sfc=jnp.zeros(ncol),
        co2_flux=co2_flux,
        # Freshwater leaving the column to the ocean = total Richards runoff
        # (surface + subsurface) [kg/m²/s]; closes the coupler water budget.
        freshwater_flux=richards_out.runoff_surface + richards_out.runoff_subsurface,
        # Land does not extract heat or exert stress on the ocean.
        ocean_heat_extraction=jnp.zeros(ncol),
        ocean_stress_x=jnp.zeros(ncol),
        ocean_stress_y=jnp.zeros(ncol),
        # Phase-aware moisture mass flux: lhflx_actual was computed with
        # L_eff (L_s over snow, L_v otherwise) so dividing recovers mass.
        surface_mass_flux=lhflx_actual / L_eff,
        # Land tile does not exchange salt with the ocean directly.
        salt_flux=jnp.zeros(ncol),
    )

    return new_state, response, carbon_state_new, surface_out


def init_multilayer_land_state(
    ncol: int,
    config: MultiLayerLandConfig,
    T_init: float = 280.0,
    theta_init: float | None = None,
    TgC_init: float | None = None,
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
    TgC_init : float or None
        Initial value of the 30-day air-temperature EMA in [°C].  Only
        consumed by the two-leaf canopy surface scheme.  When ``None``,
        the state's ``TgC`` field is left as ``None`` and the canopy
        falls back to per-column ``CanopyLandParams.TgC`` (or to the
        instantaneous ``forcing.T_lowest - 273.15``).  Pass a numerical
        value (typically ``T_init - 273.15``) to enable the in-state
        EMA accumulator.

    Returns
    -------
    MultiLayerLandState
    """
    from legoesm.land.soil_hydraulics import psi_from_theta

    grid = make_soil_grid(config.soil_grid)
    nlayers = grid.n_layers

    if theta_init is None:
        theta_init = 0.5 * config.hydraulics.theta_sat

    # T_init may be a scalar (uniform column, legacy) or a per-column (ncol,)
    # array (spatial warm start, e.g. lat-varying near-surface air T) — #671.
    _T = jnp.asarray(T_init)
    if _T.ndim == 0:
        T_soil = jnp.full((ncol, nlayers), T_init)
    else:
        T_soil = jnp.broadcast_to(_T.reshape(ncol, 1), (ncol, nlayers))
    # broadcast_to (not full) so a PER-COLUMN theta_init (ncol,1) from a spatial
    # theta_sat works; scalar theta_init broadcasts identically.
    theta_soil = jnp.broadcast_to(jnp.asarray(theta_init), (ncol, nlayers))
    psi_soil = psi_from_theta(theta_soil, config.hydraulics)

    if TgC_init is not None:
        TgC = jnp.full(ncol, TgC_init)
    else:
        TgC = None

    return MultiLayerLandState(
        T_soil=T_soil,
        psi_soil=psi_soil,
        theta_soil=theta_soil,
        runoff_surface=jnp.zeros(ncol),
        runoff_subsurface=jnp.zeros(ncol),
        snow_depth=jnp.zeros(ncol),
        snow_age=jnp.zeros(ncol),
        TgC=TgC,
        surface_water=jnp.zeros(ncol),   # coupled ponding cell (#671)
    )


def aridity_theta_init(rh_surface, theta_wp, theta_fc):
    """Aridity-aware initial soil moisture from near-surface relative humidity.

    Cold-starting the deep Richards column at the moisture-uniform
    ``0.5 * theta_sat`` default (a soil *porosity* fraction, aridity-blind)
    leaves subtropical deserts holding rainforest-scale water — ~375 kg/m^2 in a
    3 m column.  A hot bare-soil skin then drives runaway *potential*
    evaporation: the supply limiter never binds because the water is there, the
    atmosphere moistens without bound, and the coupled AMIP run blows up after
    ~a week (issue #730, day-8 non-finite winds).

    Anchor the initial plant-available water to atmospheric aridity instead.  Map
    the near-surface relative humidity of the IC atmosphere linearly into the
    plant-available range ``[theta_wp, theta_fc]``::

        theta_init = theta_wp + clip(rh_surface, 0, 1) * (theta_fc - theta_wp)

    Arid columns (low near-surface RH) start near the wilting point, so from step
    one the root-zone ``beta`` is at its floor (``beta`` multiplies
    ``q_sat(T_skin)`` in the surface specific humidity) AND the extractable water
    is small — the evaporation limiter binds immediately.  Humid columns start
    near field capacity (the physical drained equilibrium).

    ``theta_wp`` / ``theta_fc`` MUST be the SAME per-column thresholds the tile's
    ``beta`` reads (``LandSurfaceParams.theta_wp`` / ``theta_fc``) so the seed is
    consistent with the running physics.  CLM setup enforces
    ``theta_fc > theta_wp``, and physically ``theta_wp >= theta_r``, so the result
    lies in ``[theta_wp, theta_fc] ⊂ [theta_r, theta_sat]`` — a valid Richards
    state for ``psi_from_theta``.

    Parameters
    ----------
    rh_surface : array
        Near-surface relative-humidity proxy ``q_v / q_sat`` (per column).
        Clipped to ``[0, 1]``.
    theta_wp, theta_fc : array or float
        Per-column wilting-point / field-capacity volumetric water content
        [m^3/m^3].

    Returns
    -------
    array
        Per-column initial volumetric water content (broadcast of the inputs).
    """
    rh = jnp.clip(rh_surface, 0.0, 1.0)
    return theta_wp + rh * (theta_fc - theta_wp)
