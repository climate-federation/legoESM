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
from legoesm.core.bulk_flux import simple_bulk_fluxes, compute_most_fluxes
from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm.land.soil_hydraulics import psi_from_theta
from legoesm.core.surface_energy import surface_radiation_fluxes
from legoesm.land.carbon.config import CarbonState
from legoesm.land.carbon.carbon_cycle import step_carbon
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.snow_budget import update_snow
from legoesm.land.snow_bands import (
    band_albedo,
    band_net_radiation,
    band_precip_snow,
    step_snow_bands,
)
from legoesm.land.state import MultiLayerLandState
from legoesm.land.surface_params import read_spatial_param as _get
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.stomata_utils import compute_effective_beta
from legoesm.land.richards import solve_richards
from legoesm.land.soil_thermal import solve_soil_thermal
from legoesm.surface_albedo import land_albedo as compute_land_albedo
from legoesm.surface_albedo import (
    dry_soil_brightening,
    land_vegetation_albedo,
    snow_albedo,
    snow_cover_fraction,
)

# Perturbation [K] for the one-sided finite-difference linearisation of the
# turbulent surface fluxes when building the semi-implicit surface conductance
# (see step_multilayer_land). Small enough for an accurate slope, large enough
# to stay well above bulk-flux round-off.
_SURFACE_LIN_DT_K = 0.1


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
    # CLM dry-soil brightening (Oleson 2013): the snow-free base albedo rises as the
    # top layer dries, so a dry DESERT is bright while moist bare soil / tundra stays
    # dark — the desert contrast a single per-PFT albedo cannot capture.  Applied to the
    # base BEFORE the snow blend so both the flux and the returned albedo use it.
    albedo_land = albedo_land + dry_soil_brightening(theta[:, 0], config.land_albedo)
    emissivity = _get(lp, "emissivity", config.emissivity_land)
    z0 = _get(lp, "z0", config.z0_land)

    # Spatially-varying root zone params
    root_depth = _get(lp, "root_depth", config.root_depth)
    theta_wp = _get(lp, "theta_wp", config.theta_wp)
    theta_fc = _get(lp, "theta_fc", config.theta_fc)

    # Surface temperature = top soil layer
    T_sfc = T_soil[:, 0]
    ncol = T_sfc.shape[0]

    # --- Smooth wind speed floor ---
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
    )

    # --- Root distribution and per-layer moisture stress ---
    # Compute these early so beta_soil reflects the full root zone,
    # not just the top layer.
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
            theta_fc[:, None] - theta_wp[:, None], 1e-3,  # coeff-ok: theta-range divide-safety floor
        )
        beta_root = jnp.clip(
            (theta - theta_wp[:, None]) / denom,
            0.0, 1.0,
        )
    else:
        denom = jnp.maximum(theta_fc - theta_wp, 1e-3)  # coeff-ok: theta-range divide-safety floor
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
    beta, _ = compute_effective_beta(
        T_sfc, forcing, beta_soil, config, carbon_state, dt,
        land_params=lp,
    )

    # Stomatal reduction factor: ratio of effective beta to soil-only beta.
    # This captures the stomatal limitation independent of soil moisture,
    # so it can be applied to updated soil moisture later (same as slab land).
    stomatal_ratio = beta / jnp.maximum(beta_soil, 1e-10)

    # --- Surface saturation humidity: use ice saturation over snow ---
    q_sat_liq = saturation_mixing_ratio(T_sfc, forcing.p_surface)
    q_sat_ice = saturation_mixing_ratio_ice(T_sfc, forcing.p_surface)
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
    # on T_sfc < T_freeze we only switch to snow phase when the
    # snow can survive.  Existing snow always uses snow phase
    # regardless of surface temperature (snow_budget handles melt
    # energy correctly).  Iter-68 audit fix.
    # --- Sub-grid elevation-band snow (static feature gate) ---
    # When enabled, the TOTAL cell precipitation is re-partitioned by phase PER
    # elevation band (lapse-rate-downscaled air temperature), so a warm cell whose
    # high sub-grid fractions sit below freezing still accumulates snow there.  The
    # cell-level snow logic below (phase of q_sat, L_eff, albedo, melt energy) then
    # runs on the area-weighted aggregates.
    bands = config.elev_bands
    if bands is not None:
        if state.snow_bands is None:
            raise ValueError(
                "config.elev_bands is set but state.snow_bands is None; initialise "
                "the state with init_multilayer_land_state(config=...) so the banded "
                "SWE field exists."
            )
        # Firn/glacier-ice reservoir (gap 4); a legacy restart predating it carries
        # ``None`` -> start from no perennial ice (equivalent to the old cap behaviour
        # until the reservoir fills), keeping backward compatibility.
        ice_bands_in = (jnp.zeros_like(state.snow_bands) if state.ice_bands is None
                        else state.ice_bands)
        snowfall_bands = band_precip_snow(
            forcing.T_lowest, forcing.precip_total, forcing.precip_snow, bands,
        )
        precip_snow_eff = jnp.mean(snowfall_bands, axis=-1)  # equal-area bands
    else:
        precip_snow_eff = forcing.precip_snow

    fresh_snow_mass = precip_snow_eff * dt
    has_existing_snow = snow > 1e-6
    has_surviving_fresh_snow = (fresh_snow_mass > 1e-6) & (T_sfc < constants.T_freeze)
    has_snow = has_existing_snow | has_surviving_fresh_snow
    q_sat_sfc = jnp.where(has_snow, q_sat_ice, q_sat_liq)
    # Over snow, moisture is freely available from the snowpack (beta=1)
    beta_effective = jnp.where(has_snow, 1.0, beta)
    q_sfc = beta_effective * q_sat_sfc

    # Phase-appropriate latent heat: sublimation over snow, vaporisation over bare soil
    L_eff = jnp.where(has_snow, constants.L_s, constants.L_v)

    # --- Bulk fluxes ---
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
            T_sfc, q_sfc, rho,
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
            T_sfc, q_sfc, rho, wind_speed,
            config.Cd_land, config.Ch_land,
            L_latent=L_eff,
        )

    # --- Surface albedo (snow-mass dependent) ---
    # Use the SAME effective snow mass as the bulk-flux phase decision
    # (iter-68 fix): existing snow always counts; fresh snow counts
    # only when T_sfc < T_freeze (it survives the step).  Without
    # this consistency, SW absorption would lag the LH/SH phase
    # transition by one step on every fresh-snow event.  Iter-71 fix.
    snow_effective = jnp.where(
        has_existing_snow | has_surviving_fresh_snow,
        snow + jnp.where(has_surviving_fresh_snow, fresh_snow_mass, 0.0),
        snow,
    )
    if bands is not None:
        # Banded surface energy balance (gap 1+2): per-band albedo -> per-band net
        # radiation (elevation-lapsed SW/LW down + per-band skin T).  Rn_bands drives
        # per-band melt; the area-weighted aggregate drives the shared soil column;
        # alpha_eff (SW-flux-weighted) is the atmosphere-facing albedo.  Retains the
        # elevation x albedo covariance the old cell-mean radiation averaged away.
        _n_bands = bands.band_dz.shape[-1]
        if config.snow_albedo_feedback and lat is not None:
            # Banded effective SWE for the albedo cover: existing band SWE plus fresh
            # band snowfall where it survives the step on that band (band-downscaled
            # surface temperature below freezing) — the band analogue of the cell-level
            # ``snow_effective`` rule.  Pre-step albedo uses the OLD per-band age (this
            # step's snowfall resets it post-step), matching the cell-mean path.
            _cover_fn = lambda s: snow_cover_fraction(s, config.land_albedo)
            # NB gap 3 (solar-zenith snow brightening) is intentionally NOT wired here:
            # the offline calibration forcing carries a constant placeholder cos_zenith
            # so the effect is inactive/uncalibratable offline, and in coupled mode it
            # needs radiation-time zenith to stay energy-consistent (see snow_albedo's
            # cos_zenith arg + docs).  Snow albedo here is the diffuse (age) value.
            _alb_fn = lambda a: snow_albedo(a, config.land_albedo)
            _T_sfc_band = T_sfc[:, None] - bands.lapse_rate_K_m * bands.band_dz
            _band_surviving = ((snowfall_bands * dt > 1e-6)
                               & (_T_sfc_band < constants.T_freeze))
            _bands_effective = state.snow_bands + jnp.where(
                _band_surviving, snowfall_bands * dt, 0.0,
            )
            # Snow-free base MUST match compute_land_albedo's: the per-cell CLM map
            # albedo when land_params are supplied, else the latitude-band vegetation
            # albedo (so at zero relief the banded path reproduces the cell-mean one).
            _snow_free_base = (jnp.broadcast_to(albedo_land, T_sfc.shape) if lp is not None
                               else land_vegetation_albedo(lat, config.land_albedo))
            alpha_bands = band_albedo(
                _bands_effective, state.snow_age_bands, _snow_free_base, _cover_fn,
                _alb_fn, ice_bands=ice_bands_in, cfg=bands,
            )
        else:
            # Snow-albedo feedback off (or no latitude): uniform snow-free base albedo
            # per band — the banded analogue of the cell-mean ``alpha = full(albedo_land)``
            # path (no snow blend, no latitude dependence); the elevation SW/LW lapse
            # (gap 2) still applies per band.  Preserves the non-banded gating semantics.
            alpha_bands = jnp.broadcast_to(
                jnp.reshape(albedo_land, (-1, 1)), (T_sfc.shape[0], _n_bands))
        band_rad = band_net_radiation(
            T_sfc, alpha_bands, forcing.sw_down, forcing.lw_down, emissivity, bands,
        )
        alpha = band_rad.alpha_eff
        sw_net, lw_net = band_rad.sw_net_agg, band_rad.lw_net_agg
    else:
        band_rad = None
        if config.snow_albedo_feedback and lat is not None:
            # Snow-free base = per-cell map albedo (CLM PFT) when land_params supplied.
            _base = None if lp is None else jnp.broadcast_to(albedo_land, T_sfc.shape)
            alpha = compute_land_albedo(
                lat, snow_effective, snow_age, config.land_albedo, base_albedo=_base,
            )
        else:
            alpha = jnp.full(T_sfc.shape, albedo_land, dtype=T_sfc.dtype)
        # --- Radiation (cell-mean) ---
        sw_net, lw_net, _ = surface_radiation_fluxes(
            forcing.sw_down, forcing.lw_down, T_sfc, alpha,
            emissivity,
        )

    # --- Ground heat flux (residual of surface energy balance) ---
    # G = SW_net + LW_net - SH - LH  (positive into soil)
    G_surface = sw_net + lw_net - shflx - lhflx

    # --- Snow budget (energy-limited melt) ---
    # G_surface drives the melt: M = max(0, G * dt / L_f)
    if bands is not None:
        # Per-band melt is limited by each band's OWN net radiation minus the (cell)
        # turbulent fluxes — the banded surface energy balance.  Its area-weighted
        # mean equals the aggregate G_surface, so soil-column energy is conserved.
        band_step = step_snow_bands(
            state.snow_bands, state.snow_age_bands, ice_bands_in, T_sfc, snowfall_bands, dt,
            Q_net=band_rad.Rn_bands - shflx[:, None] - lhflx[:, None],
            cfg=bands,
            T_snow_melt=config.T_snow_melt,
        )
        snow_bands_new = band_step.swe_bands
        ice_bands_new = band_step.ice_bands
        snow_age_bands_new = band_step.snow_age_bands
        snow_new = band_step.swe_total
        snow_age_new = band_step.snow_age
        snow_melt = band_step.snow_melt
        ice_melt = band_step.ice_melt
        # Glacier discharge (frozen) + ablation ice meltwater both leave as runoff.
        cap_runoff = band_step.ice_runoff + ice_melt / dt
    else:
        snow_new, snow_age_new, snow_melt = update_snow(
            snow, snow_age, T_sfc, precip_snow_eff, dt,
            Q_net=G_surface,
            snow_melt_rate=config.snow_melt_rate,
            T_snow_melt=config.T_snow_melt,
        )
        snow_bands_new = state.snow_bands
        snow_age_bands_new = state.snow_age_bands
        ice_bands_new = state.ice_bands
        ice_melt = jnp.zeros_like(snow_new)
        cap_runoff = jnp.zeros_like(snow_new)

    # Subtract melt energy from ground heat flux before soil thermal solve.  Both the
    # seasonal-snow melt and the ablation ice melt consume L_f per kg; the frozen
    # glacier discharge leaves as ice (no fusion) so it is NOT charged here.
    melt_energy = (snow_melt + ice_melt) * constants.L_f / dt  # W/m2 consumed by melt
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
    if bands is not None:
        # Distribute the aggregate sublimation/deposition across bands
        # proportionally to their SWE (preserves the band distribution and the
        # aggregate mass; a snow-free band stays snow-free).  AD-safe: guard the
        # denominator, fall back to unscaled bands when the pack is empty.
        _agg_ok = snow_after_melt > 1e-12
        _scale = jnp.where(
            _agg_ok, snow_new / jnp.where(_agg_ok, snow_after_melt, 1.0), 1.0,
        )
        # Empty-pack deposition (frost/dew grows the aggregate from ~0): scaling zero
        # bands would keep them zero and the deposited mass would vanish from the band
        # budget next step.  Distribute it EQUALLY across the equal-area bands so the
        # band store matches the aggregate (mean_k snow_bands_new == snow_new).
        _deposit_empty = (~_agg_ok) & (snow_new > 1e-12)
        snow_bands_new = jnp.where(
            _deposit_empty[:, None], snow_new[:, None], snow_bands_new * _scale[:, None],
        )

    # --- Water-limit soil evaporation (bare soil only) ---
    dz = grid.dz  # (n_layers,) layer thicknesses [m]
    extractable_water = jnp.sum(
        jnp.maximum(theta - theta_r, 0.0) * dz[None, :], axis=-1
    ) * rho_w  # (ncol,) kg/m2
    precip_rain = forcing.precip_total - precip_snow_eff
    melt_rate = snow_melt / dt  # kg/m2/s meltwater entering liquid budget
    soil_evap_demand = jnp.where(has_snow, 0.0, evap_rate_demand)
    # Surface soil resistance: the bulk latent flux throttles by the ROOT-ZONE mean
    # beta, but bare-soil evaporation is controlled by the TOP layer, which dries
    # into a high-resistance crust far faster.  Throttle the (positive, evaporative)
    # bare-soil demand by S_top**exp (beta-method soil-evaporation efficiency,
    # Sellers 1992 / Lee & Pielke 1992); dew/condensation (demand < 0) is left
    # un-throttled.  The suppressed latent energy is returned to the soil via
    # ``evap_excess_energy`` below (energy-conserving), so a dry crust warms the
    # surface instead of evaporating water that the deep column would have to supply.
    _theta_top = theta[:, :1]
    _S_top = jnp.clip((_theta_top - theta_r)
                      / jnp.maximum(config.hydraulics.theta_sat - theta_r, 1e-6),
                      0.0, 1.0)[:, 0]
    # Cast to the evaporation working dtype: theta_r / theta_sat may be float64
    # per-cell config arrays while the coupled state runs float32, and promoting
    # the latent flux here would change the land-state output dtype (a lax.scan
    # carry-type mismatch in the segment).
    _S_top_pow = jnp.where(_S_top > 0.0, _S_top, 1.0)
    _beta_surf_raw = _S_top_pow ** config.soil_evap_resistance_exp
    _beta_surf = jnp.where(_S_top > 0.0, _beta_surf_raw, 0.0).astype(
        soil_evap_demand.dtype)
    soil_evap_demand = jnp.where(soil_evap_demand > 0.0,
                                 soil_evap_demand * _beta_surf, soil_evap_demand)
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
    # Split surface flux between bare-soil and transpiration.  The
    # snowpack already swallowed sublim_actual upstream (line 286), so
    # the soil should NOT see ANY latent flux when has_snow=True —
    # routing snow deposition through flux_top double-counts the mass
    # (codex iter-23 stop-time review).  Only the snow-free regime
    # exposes the soil to direct latent exchange.
    #
    # Within the snow-free regime:
    #   evap_rate >= 0 (evaporation upward): bare/veg partition by
    #     ``1 - f_veg`` / ``f_veg``.
    #   evap_rate < 0 (dew / deposition downward on bare soil):
    #     route ALL of the negative flux to ``flux_top`` so the column
    #     water budget closes; transpiration sink set to 0.  The
    #     vegetated-fraction dew on a snow-free cell is treated as
    #     bare-soil input (no separate canopy-storage reservoir in
    #     this model).
    soil_flux = jnp.where(has_snow, 0.0, evap_rate)
    is_dew = soil_flux < 0.0
    evap_bare = jnp.where(is_dew, soil_flux, soil_flux * (1.0 - f_veg))
    evap_transp = jnp.where(is_dew, 0.0, soil_flux * f_veg)

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
    # AD-safe normalization.  ``weight / max(weight_sum, 1e-20)`` is finite in the
    # FORWARD when the root zone is all dry (weight_sum -> 0, weight -> 0), but its
    # reverse-mode VJP carries a 1/weight_sum**2 term that overflows to NaN as
    # weight_sum -> 0 (a finite-forward / NaN-backward singularity that breaks
    # gradient-based land calibration).  Branch on a non-tiny weight_sum and sanitise
    # the denominator in BOTH branches so no near-zero division ever enters the graph;
    # fall back to the root-density profile (E_pot_transp ~ 0 there, so the sink is
    # unchanged) when the zone is dry.  Forward-identical for weight_sum > 1e-12.
    _wok = weight_sum > 1e-12
    _denom = jnp.where(_wok, weight_sum, 1.0)
    weight_norm = jnp.where(_wok, weight / _denom, root_frac)
    sink = weight_norm * E_pot_transp[:, None] / dz[None, :]

    # --- Richards equation: update soil moisture (+ coupled surface ponding) ---
    richards_out = solve_richards(
        psi, theta, grid,
        config.hydraulics, config.richards,
        flux_top, sink, dt,
        surface_water=state.surface_water,
    )
    # NB: the former "evaporation water budget closure" — a clip of theta_new to
    # [theta_r, theta_sat] — is removed.  It was a non-conservative band-aid for the
    # old infiltration/evap mismatch; theta_from_psi is now bounded below at theta_r
    # by construction, the coupled surface cell carries the ponded excess, and the
    # mixed-form solve closes the water budget, so no clip is needed.

    # --- Soil thermal diffusion: update soil temperature ---
    # Excess energy from water-limited evaporation warms the soil
    G_surface = G_surface + evap_excess_energy

    # --- Semi-implicit surface conductance lambda = -dG/dT_sfc (>= 0) ---
    # Folding lambda into the soil thermal solve (Robin BC) linearises the
    # T_sfc-dependence of the surface energy balance, removing the explicit-
    # coupling instability that diverges (NaN) for a large dt + thin top layer
    # under a stiff surface (high roughness / high insolation).  Three parts:
    #   * longwave (analytic):  -d(LW_net)/dT_sfc = 4*eps*sigma*T_sfc^3
    #   * sensible (finite difference, scheme-agnostic): d(SH)/dT_sfc, the bulk
    #     SH re-evaluated at T_sfc + _SURFACE_LIN_DT_K.
    #   * latent: d(LH)/dT_sfc from the same FD, but the flux that ACTUALLY
    #     reaches the soil is the water/snow-LIMITED ``lhflx_actual``, not the
    #     demand ``lhflx``.  When evaporation is supply-limited the latent flux
    #     is ~insensitive to T_sfc, so scale the latent slope by the realised
    #     fraction lhflx_actual/lhflx (-> 0 when the limiter binds; 1 for
    #     unlimited evaporation, and for dew/condensation where lhflx <= 0).
    # All slopes are clamped >= 0 so lambda can only ADD damping.
    T_sfc_lin = T_sfc + _SURFACE_LIN_DT_K
    q_sfc_lin = beta_effective * jnp.where(
        has_snow,
        saturation_mixing_ratio_ice(T_sfc_lin, forcing.p_surface),
        saturation_mixing_ratio(T_sfc_lin, forcing.p_surface),
    )
    if config.bulk_scheme in ("most", "coare3", "large_yeager"):
        _, _, shflx_lin, lhflx_lin, _ = compute_most_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_sfc_lin, q_sfc_lin, rho,
            z_ref=config.z_ref,
            z0_init=z0,
            scheme=config.bulk_scheme,
            n_iter=config.bulk_n_iter,
            L_latent=L_eff,
        )
    else:
        _, _, shflx_lin, lhflx_lin = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_sfc_lin, q_sfc_lin, rho, wind_speed,
            config.Cd_land, config.Ch_land,
            L_latent=L_eff,
        )
    lambda_lw = 4.0 * emissivity * constants.sigma_sb * T_sfc ** 3
    lambda_sh = jnp.maximum((shflx_lin - shflx) / _SURFACE_LIN_DT_K, 0.0)
    # Realised-evaporation fraction in [0, 1]: the latent flux entering the soil
    # is supply-limited, so its T_sfc slope shrinks toward 0 as the limiter
    # binds.  ``lhflx <= 0`` is dew/condensation (never supply-limited) -> 1.
    # The denominator must be floored before the division, not only masked by the
    # outer ``where``.  At a dry top layer ``lhflx_actual`` can be exactly zero
    # while ``lhflx`` is a tiny positive number; the forward ratio is finite, but
    # the division VJP forms 0 / lhflx**2 and can produce 0/0 -> NaN.
    _lhflx_frac_denom = jnp.where(lhflx > 0.0, jnp.maximum(lhflx, 1e-12), 1.0)
    latent_realised_frac = jnp.where(
        lhflx > 0.0,
        jnp.clip(lhflx_actual / _lhflx_frac_denom, 0.0, 1.0),
        1.0,
    )
    lambda_lh = jnp.maximum(
        (lhflx_lin - lhflx) / _SURFACE_LIN_DT_K, 0.0
    ) * latent_realised_frac
    surface_conductance = lambda_lw + lambda_sh + lambda_lh

    T_soil_new = solve_soil_thermal(
        T_soil, richards_out.theta_new, grid,
        config.hydraulics, config.thermal,
        G_surface, dt,
        surface_conductance=surface_conductance,
    )

    # --- Build new state ---
    # ``cap_runoff`` (frozen glacier discharge + ablation ice meltwater) leaves via the
    # runoff/freshwater path WITHOUT infiltrating (glacier outflow, not soil water), so
    # it is added to the surface runoff after the Richards solve.
    runoff_surface_total = richards_out.runoff_surface + cap_runoff
    new_state = MultiLayerLandState(
        T_soil=T_soil_new,
        psi_soil=richards_out.psi_new,
        theta_soil=richards_out.theta_new,
        runoff_surface=runoff_surface_total,
        runoff_subsurface=richards_out.runoff_subsurface,
        snow_depth=snow_new,
        snow_age=snow_age_new,
        surface_water=richards_out.surface_water,
        snow_bands=snow_bands_new,
        snow_age_bands=snow_age_bands_new,
        ice_bands=ice_bands_new,
    )

    # --- Build TileResponse ---
    T_sfc_new = T_soil_new[:, 0]

    # Post-step albedo + up-welling LW for the atmosphere/coupler: reflect the updated
    # snow (+ exposed glacier ice) for the NEXT step.  The banded path re-runs the SAME
    # flux-weighted band radiation as the pre-step (gap 1) so the reported albedo is
    # alpha_eff and lw_up is the banded per-band emission aggregate — NOT a plain band
    # mean and NOT the cell-mean-T Stefan-Boltzmann value, which would be inconsistent
    # with the SW the surface actually absorbed.
    if bands is not None:
        if config.snow_albedo_feedback and lat is not None:
            _base_new = (jnp.broadcast_to(albedo_land, T_sfc_new.shape) if lp is not None
                         else land_vegetation_albedo(lat, config.land_albedo))
            alpha_bands_new = band_albedo(
                snow_bands_new, snow_age_bands_new, _base_new,
                lambda s: snow_cover_fraction(s, config.land_albedo),
                lambda a: snow_albedo(a, config.land_albedo),  # diffuse (gap 3 unwired)
                ice_bands=ice_bands_new, cfg=bands,
            )
        else:
            alpha_bands_new = jnp.broadcast_to(
                jnp.reshape(albedo_land, (-1, 1)),
                (T_sfc_new.shape[0], bands.band_dz.shape[-1]))
        band_rad_new = band_net_radiation(
            T_sfc_new, alpha_bands_new, forcing.sw_down, forcing.lw_down, emissivity, bands,
        )
        alpha_new = band_rad_new.alpha_eff
        lw_up_new = band_rad_new.lw_up_agg
    else:
        if config.snow_albedo_feedback and lat is not None:
            _base = None if lp is None else jnp.broadcast_to(albedo_land, T_sfc_new.shape)
            alpha_new = compute_land_albedo(
                lat, snow_new, snow_age_new, config.land_albedo, base_albedo=_base,
            )
        else:
            alpha_new = alpha
        _, _, lw_up_new = surface_radiation_fluxes(
            forcing.sw_down, forcing.lw_down, T_sfc_new, alpha_new,
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
            theta_fc[:, None] - theta_wp[:, None], 1e-3,  # coeff-ok: theta-range divide-safety floor
        )
        beta_root_new = jnp.clip(
            (theta_new - theta_wp[:, None]) / denom_new,
            0.0, 1.0,
        )
    else:
        denom_new = jnp.maximum(theta_fc - theta_wp, 1e-3)  # coeff-ok: theta-range divide-safety floor
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
    q_sat_liq_new = saturation_mixing_ratio(T_sfc_new, forcing.p_surface)
    q_sat_ice_new = saturation_mixing_ratio_ice(T_sfc_new, forcing.p_surface)
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
            T_sfc_new, forcing, beta_soil_new, config, carbon_state, dt,
            land_params=lp,
        )
        carbon_state_new, co2_flux = step_carbon(
            carbon_state, forcing.sw_down, T_sfc_new, forcing.co2_ppmv,
            beta_soil_new, lat_arr, doy, forcing.precip_total, config.carbon, dt,
            gpp_override=gpp_farq_new,
        )
    else:
        carbon_state_new = carbon_state
        co2_flux = jnp.zeros(ncol)

    response = TileResponse(
        T_sfc=T_sfc_new,
        albedo=alpha_new,
        emissivity=jnp.full(T_sfc.shape, emissivity, dtype=T_sfc.dtype),
        z0=jnp.full(T_sfc.shape, z0, dtype=T_sfc.dtype),
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
        # subsurface runoff (incl. snow-capping ice discharge).  All kg/m²/s.
        freshwater_flux=(
            runoff_surface_total + richards_out.runoff_subsurface
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
        # Land tile does not exchange salt with the ocean directly.
        salt_flux=jnp.zeros(ncol),
    )

    return new_state, response, carbon_state_new


def init_multilayer_land_state(
    ncol: int,
    config: MultiLayerLandConfig,
    T_init: float = 280.0,  # coeff-ok: initial soil temperature [K]
    theta_init: float | None = None,
) -> MultiLayerLandState:
    """Create initial multi-layer land state.

    Parameters
    ----------
    ncol : int
        Number of columns.
    config : MultiLayerLandConfig
        Land model configuration.
    T_init : float or array
        Initial soil temperature [K].  A scalar gives a uniform column
        (legacy default).  A per-column array of shape ``(ncol,)`` gives a
        spatially-structured warm start (e.g. the lat-varying near-surface air
        temperature), broadcast vertically across all soil layers — removes the
        artificial tropical cold-soil spin-up of the uniform 280 K default.
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

    _T = jnp.asarray(T_init)
    if _T.ndim == 0:
        T_soil = jnp.full((ncol, nlayers), T_init)
    else:
        # per-column (ncol,) -> broadcast across the vertical soil layers
        T_soil = jnp.broadcast_to(_T.reshape(ncol, 1), (ncol, nlayers))
    # broadcast_to (not full) so a PER-COLUMN theta_init (ncol,1) from a spatial
    # theta_sat works; scalar theta_init broadcasts identically.
    theta_soil = jnp.broadcast_to(jnp.asarray(theta_init), (ncol, nlayers))
    psi_soil = psi_from_theta(theta_soil, config.hydraulics)

    # Banded SWE/age/ice only when the elevation-band scheme is configured; ``None``
    # keeps the legacy pytree structure (single cell-mean snowpack, no ice reservoir).
    if config.elev_bands is not None:
        _nb = config.elev_bands.band_dz.shape[-1]
        # Match the soil-temperature dtype so the banded fields do not start float32
        # under a float64 (x64) state — a mixed-dtype pytree / scan-carry hazard.
        _bdt = T_soil.dtype
        snow_bands = jnp.zeros((ncol, _nb), dtype=_bdt)
        snow_age_bands = jnp.zeros((ncol, _nb), dtype=_bdt)
        ice_bands = jnp.zeros((ncol, _nb), dtype=_bdt)
    else:
        snow_bands = snow_age_bands = ice_bands = None

    return MultiLayerLandState(
        T_soil=T_soil,
        psi_soil=psi_soil,
        theta_soil=theta_soil,
        runoff_surface=jnp.zeros(ncol),
        runoff_subsurface=jnp.zeros(ncol),
        snow_depth=jnp.zeros(ncol),
        snow_age=jnp.zeros(ncol),
        surface_water=jnp.zeros(ncol),
        snow_bands=snow_bands,
        snow_age_bands=snow_age_bands,
        ice_bands=ice_bands,
    )
