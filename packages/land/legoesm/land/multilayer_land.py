"""Multi-layer soil land model with pluggable surface scheme.

Dispatches between:

- ``SimpleSEBConfig`` (default): bulk-flux surface energy balance with
  skin T = ``T_soil[:, 0]``, optional Jarvis / Leuning stomatal coupling.
- ``TwoLeafCanopyConfig``: DifferBESS-style two-leaf canopy Newton +
  Picard closure on top of the same soil column.
- ``CLMMLCanopyConfig``: CLM-ML-JAX multilayer canopy model (Phase 3).

Both ``SimpleSEBConfig`` and ``TwoLeafCanopyConfig`` produce a
``SurfaceFluxOutput``; the post-flux pipeline (snow, Richards, soil
thermal, carbon, TileResponse) is shared.

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

import math

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
from legoesm.land.soil_thermal import (
    liquid_water_content, moisture_fusion_heat_source, solve_soil_thermal)

# Sub-steps of the final soil-thermal solve when soil freeze/thaw is on: at the
# 1800 s land step a single apparent-heat-capacity step overshoots the 0 C
# curtain in a thin top layer; six 300 s sub-steps keep it on the curtain
# (user decision 2026-09-28).  A loop count, never config or trainable.
FINAL_THERMAL_SUBSTEPS = 6
from legoesm.land.canopy.config import CLMMLCanopyConfig
from legoesm.land.canopy.interception import (
    intercept_rain,
    wetted_fraction as interception_wetted_fraction,
)
from legoesm.land.surface_scheme import (
    SimpleSEBConfig,
    TwoLeafCanopyConfig,
    compute_simple_seb_fluxes,
    compute_two_leaf_canopy_fluxes,
)
from legoesm.land.canopy.radiative_transfer import broadband_albedo
from legoesm.land.soil_albedo import rewet_soil_bands
from legoesm.land.surface_scheme.two_leaf_canopy import (
    advance_TgC_ema,
    compute_prognostic_lai,
)
from legoesm.surface_albedo import land_albedo as compute_land_albedo
from legoesm.surface_albedo import (
    dry_soil_brightening,
    land_vegetation_albedo,
    snow_albedo,
    snow_cover_fraction,
)
from legoesm.land.snow_bands import (
    band_albedo,
    band_net_radiation,
    band_precip_snow,
    step_snow_bands,
)

# ln 10: CLM5 writes the ice impedance as a power of ten.
_LN10 = math.log(10.0)


def _get(lp, name: str, fallback):
    """Read a per-column field from ``lp`` if present, else return ``fallback``.

    Uses the safe ``getattr(lp, name, fallback)`` form because ``lp`` can be
    either ``LandSurfaceParams`` (for SimpleSEB; full field set) or
    ``CanopyLandParams`` (for TwoLeafCanopy; disjoint field set).  Missing
    fields fall back to the caller-supplied default rather than raising.

    A field that EXISTS but is ``None`` is also treated as absent: optional
    per-column params (``root_depth``/``theta_wp``/``theta_fc``) are declared on
    ``CanopyLandParams`` with a ``None`` default, so a params object that does
    not carry them must still fall back to the scalar config value rather than
    propagating ``None`` into the arithmetic.
    """
    if lp is None:
        return fallback
    v = getattr(lp, name, fallback)
    return fallback if v is None else v


def resolve_plant_wilting_point(land_params, config):
    """The PLANT wilting point that drives root-zone transpiration and GPP.

    It is deliberately separate from the SOIL wilting point (deep-rooted
    vegetation extracts water below the soil-evaporation cutoff), and it is
    resolved in one place because the two callers -- the multilayer land step
    and the coupler's land-tile beta -- disagreed: the step fell straight back
    to the SCALAR ``config.theta_wp`` and so ignored a per-column
    ``theta_wp``.  Any calibration that varies the wilting point by plant
    functional type therefore reached soil evaporation and was silently inert
    in transpiration and GPP -- the two arms of the same column disagreeing
    about how dry the soil is.

    Order, most specific first: a per-column ``theta_wp_plant``, then a
    per-column ``theta_wp``, then ``config.theta_wp_plant``, then the scalar
    ``config.theta_wp``.  With none of them set this reproduces the
    single-wilting-point behaviour exactly.
    """
    scalar = (config.theta_wp_plant if config.theta_wp_plant is not None
              else config.theta_wp)
    return _get(land_params, "theta_wp_plant",
                _get(land_params, "theta_wp", scalar))


def root_zone_moisture_stress(theta, beta_min, root_depth, theta_wp, theta_fc,
                              z_centers, ncol):
    """Root-zone soil-moisture evaporative efficiency ``beta_soil`` in
    ``[beta_min, 1]`` plus its per-layer pieces.

    Single source of the moisture-stress formula, shared by the land surface
    energy balance (:func:`_step_multilayer_land_impl`, pre- AND post-step) and
    the coupler's atmospheric land tile (via :func:`land_tile_beta_soil`) so
    BOTH throttle land evaporation by the IDENTICAL stress.  Previously the
    atmosphere re-derived the land surface humidity at ``beta = 1`` (a saturated
    swamp surface) while the land model throttled internally, giving a land
    latent flux ~10x too large (hfls ~775 W/m^2, Bowen ~0.04).

        beta_root  = clip((theta - theta_wp) / max(theta_fc - theta_wp, 1e-3), 0, 1)
        w_frac_rz  = clip(sum_layers(root_frac * beta_root), 0, 1)
        beta_soil  = beta_min + (1 - beta_min) * w_frac_rz

    ``root_frac`` is the exponential root density normalised over the column.
    ``root_depth``/``theta_wp``/``theta_fc`` may be scalars (config) or
    per-column arrays (``LandSurfaceParams``); both are promoted to ``(ncol,)``.
    Returns ``(beta_soil, root_frac, beta_root, w_frac_rz)``.
    """
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
    # Delegate the moisture-stress arithmetic (beta_root, beta_soil, and the
    # audit-#6 theta_fc-theta_wp range floor) to the single kernel
    # root_zone_beta_soil so the formula lives in exactly one place; add
    # root_frac + w_frac_rz here for callers that need the per-layer pieces.
    beta_soil, beta_root = root_zone_beta_soil(
        theta, root_frac, theta_wp_c, theta_fc_c, beta_min, spatial=True)
    w_frac_rz = jnp.clip(jnp.sum(root_frac * beta_root, axis=-1), 0.0, 1.0)
    return beta_soil, root_frac, beta_root, w_frac_rz


def land_tile_beta_soil(theta_soil, config, land_params=None):
    """Root-zone ``beta_soil`` for the coupler's atmospheric land tile, resolved
    from the SAME ``config`` / ``land_params`` thresholds the land SEB uses.

    The coupler forms the land-tile effective surface humidity
    ``q_sfc = beta_soil * q_sat(T_land)`` (the alpha-method, matching
    ``simple_seb`` ``q_sfc = beta_effective * q_sat_sfc``) so the atmospheric
    land latent flux is throttled by soil moisture instead of running at the
    saturated-surface potential rate.  ``theta_soil`` is the ``(ncol, n_layers)``
    soil-moisture field from the carried multilayer land state.
    """
    grid = make_soil_grid(config.soil_grid)
    root_depth = _get(land_params, "root_depth", config.root_depth)
    # Plant wilting point (transpiration extraction) drives this root-zone
    # availability; falls back to the soil wilting point when unset.
    theta_wp   = resolve_plant_wilting_point(land_params, config)
    theta_fc   = _get(land_params, "theta_fc", config.theta_fc)
    beta_soil, _, _, _ = root_zone_moisture_stress(
        theta_soil, config.beta_min, root_depth, theta_wp, theta_fc,
        grid.z_node, theta_soil.shape[0])
    return beta_soil


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
    clm_ml_grid_info=None,
    clm_ml_pft_per_col=None,
    clm_ml_vcmaxpft_jax=None,
    clm_ml_g1_medlyn_jax=None,
    soil_frozen_fraction: jnp.ndarray | None = None,
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
        lat=lat, carbon_state=carbon_state, doy=doy, land_params=land_params,
        clm_ml_grid_info=clm_ml_grid_info,
        clm_ml_pft_per_col=clm_ml_pft_per_col,
        clm_ml_vcmaxpft_jax=clm_ml_vcmaxpft_jax,
        clm_ml_g1_medlyn_jax=clm_ml_g1_medlyn_jax,
        soil_frozen_fraction=soil_frozen_fraction)


def root_zone_beta_soil(
    theta: jnp.ndarray,
    root_frac: jnp.ndarray,
    theta_wp,
    theta_fc,
    beta_min: float,
    spatial: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Root-zone soil-moisture stress: ``(beta_soil, beta_root)``.

    Single source of truth for the moisture-stress factor used both at the
    start of a step (from the incoming ``theta``) and at the end (from the
    Richards-updated ``theta_new``); the carbon-equilibrium validator reuses
    it so the diagnostic GPP sees exactly the model's beta_soil (no
    re-derived numerics).

    Parameters
    ----------
    theta : (ncol, n_layers) volumetric water content [m3/m3].
    root_frac : root-density weights summing to 1 along the layer axis;
        shape (n_layers,) (scalar params) or (ncol, n_layers) (spatial).
    theta_wp, theta_fc : wilting-point / field-capacity water content
        [m3/m3]; scalar when ``spatial`` is False, else (ncol,) arrays.
    beta_min : floor on availability [-].
    spatial : True when ``theta_wp``/``theta_fc`` are per-column arrays.

    Returns
    -------
    beta_soil : (ncol,) root-zone-weighted availability in [beta_min, 1],
        used for stomatal / GPP moisture stress.
    beta_root : (ncol, n_layers) per-layer stress in [0, 1], reused for the
        root-water-uptake sink partition.

    Notes
    -----
    The field-capacity minus wilting-point range is floored at 1e-3 m3/m3
    (~1 % of theta_sat) so even a pathological PFT lookup with
    ``theta_fc <= theta_wp`` cannot blow ``beta_root`` up: the denominator
    would otherwise go near-zero on the same scale as theta itself.  Audit
    finding #6.
    """
    if spatial:
        denom = jnp.maximum(
            theta_fc[:, None] - theta_wp[:, None], 1e-3,  # coeff-ok: theta-range divide-safety floor
        )
        beta_root = jnp.clip((theta - theta_wp[:, None]) / denom, 0.0, 1.0)
    else:
        denom = jnp.maximum(theta_fc - theta_wp, 1e-3)  # coeff-ok: theta-range divide-safety floor
        beta_root = jnp.clip((theta - theta_wp) / denom, 0.0, 1.0)
    # Root-zone weighted beta: integrates moisture stress across layers
    # weighted by root density, so a dry top with wet deeper layers still
    # permits transpiration.  Numpy broadcasting handles both ``root_frac``
    # shapes uniformly, no per-branch reduction needed.
    w_frac_rz = jnp.clip(jnp.sum(root_frac * beta_root, axis=-1), 0.0, 1.0)
    beta_soil = beta_min + (1.0 - beta_min) * w_frac_rz
    return beta_soil, beta_root


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
    clm_ml_grid_info=None,
    clm_ml_pft_per_col=None,
    clm_ml_vcmaxpft_jax=None,
    clm_ml_g1_medlyn_jax=None,
    soil_frozen_fraction: jnp.ndarray | None = None,
) -> tuple[MultiLayerLandState, TileResponse, CarbonState | None]:
    """Step the multi-layer land model forward by ``dt`` seconds.

    Dispatches between ``SimpleSEBConfig`` and ``TwoLeafCanopyConfig``
    surface schemes via ``isinstance(config.surface_scheme, ...)``.  The
    post-flux pipeline (snow, Richards, soil thermal, carbon, TileResponse)
    is shared between both branches.  Returns a 3-tuple; use
    :func:`step_multilayer_land_with_diagnostics` to also receive the raw
    ``SurfaceFluxOutput`` for diagnostic inspection.

    ``clm_ml_grid_info`` threads a concrete CLM-ML ``GridInfo`` to the canopy
    interface for multi-step differentiable rollouts;
    ``clm_ml_vcmaxpft_jax`` / ``clm_ml_g1_medlyn_jax`` thread the optional
    trainable per-PFT Vcmax25 / Medlyn-g1 overrides (see
    :func:`_step_multilayer_land_impl`); all ``None`` for forward-only /
    non-CLM-ML.
    """
    new_state, response, carbon_new, _surface_out = _step_multilayer_land_impl(
        state, forcing, config, U_min, dt,
        lat=lat, carbon_state=carbon_state, doy=doy, land_params=land_params,
        clm_ml_grid_info=clm_ml_grid_info,
        clm_ml_pft_per_col=clm_ml_pft_per_col,
        clm_ml_vcmaxpft_jax=clm_ml_vcmaxpft_jax,
        clm_ml_g1_medlyn_jax=clm_ml_g1_medlyn_jax,
        soil_frozen_fraction=soil_frozen_fraction)
    return new_state, response, carbon_new


def _partition_latent_root_top(soil_evap, has_snow, f_veg, le_canopy, le_soil):
    """Split the water-limited L_v evaporation stream (mass rate, kg/m^2/s) into a
    TOP-boundary (bare-soil) part and a ROOT-ZONE (transpiration) part.

    Transpiration is drawn from the root zone; below-canopy soil evaporation leaves
    the top-soil boundary.  When the canopy scheme reports its ACTUAL split
    (``le_canopy`` = transpiration, ``le_soil`` = soil evaporation, both W/m^2, not
    None — the two-leaf canopy AND the CLM-ML multilayer canopy both do), the
    transpiration fraction of the net stream is ``le_canopy / (le_canopy + le_soil)``.
    The prior code always used the root-zone-wetness heuristic ``f_veg``, which
    MIS-SOURCED transpiration whenever ``f_veg`` differed from the true transpiration
    share — e.g. a deciduous canopy transpiring hard over a drying surface (small
    ``f_veg``) had its transpiration wrongly charged to bare-soil top-layer water
    (codex 2026-07-20).

    Scope of the ratio (codex review): a single fraction of the NET stream can only
    represent SIGN-CONSISTENT components.  We therefore use the ratio ONLY when both
    components are evaporative (``le_canopy >= 0`` AND ``le_soil >= 0``); for
    mixed-sign steps (transpiration with soil/leaf dew, and vice-versa — a dawn/dusk
    edge case the model's storage-free canopy cannot route componentwise anyway) we
    defer to ``f_veg``, matching the prior net-based behaviour there.  SimpleSEB and
    any scheme reporting no split also use ``f_veg``.

    Over snow the stream is pure canopy transpiration (the ground component already
    sublimated from the pack upstream), so it is drawn ENTIRELY from the root zone
    (``transp_frac = 1``).  Net dew (``soil_evap < 0``) routes entirely to the top
    boundary (transpiration sink = 0).

    CONSERVATION-NEUTRAL: ``evap_bare + evap_transp == soil_evap`` (to floating-point
    rounding — the two fractions sum to 1), so only the SPLIT changes, never the
    total withdrawal.
    """
    if le_canopy is not None and le_soil is not None:
        _tot = le_canopy + le_soil
        _safe_tot = jnp.where(_tot > 1e-12, _tot, 1.0)    # double-where AD guard
        _both_evap = (le_canopy >= 0.0) & (le_soil >= 0.0)
        _ratio = jnp.where(_tot > 1e-12, le_canopy / _safe_tot, f_veg)
        transp_frac_veg = jnp.where(_both_evap, _ratio, f_veg)
    else:
        transp_frac_veg = f_veg
    transp_frac = jnp.where(has_snow, 1.0, transp_frac_veg)
    is_dew = soil_evap < 0.0
    evap_bare = jnp.where(is_dew, soil_evap, soil_evap * (1.0 - transp_frac))
    evap_transp = jnp.where(is_dew, 0.0, soil_evap * transp_frac)
    return evap_bare, evap_transp


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
    clm_ml_grid_info=None,
    clm_ml_pft_per_col=None,
    clm_ml_vcmaxpft_jax=None,
    clm_ml_g1_medlyn_jax=None,
    soil_frozen_fraction: jnp.ndarray | None = None,
):
    """Internal 4-tuple (new_state, TileResponse, carbon, SurfaceFluxOutput).

    Kept non-public so the two public entry points (``step_multilayer_land``
    and ``step_multilayer_land_with_diagnostics``) can return different
    arities without branching inside the tight-loop code.

    ``clm_ml_grid_info`` (concrete ``GridInfo``) is forwarded to the CLM-ML
    canopy interface for a MULTI-STEP differentiable rollout — extract it once
    from a warm-start state with
    ``legoesm.land.canopy.clm_ml_interface.extract_clm_ml_grid_info`` and close
    over it before the ``jax.grad`` scan, so the traced carried state is never
    ``int()``-ed.  ``clm_ml_vcmaxpft_jax`` / ``clm_ml_g1_medlyn_jax`` are the
    optional TRAINABLE per-PFT Vcmax25 / Medlyn-g1 overrides (traced leaves
    injected from the loss), forwarded so gradients w.r.t. these canopy
    parameters flow through the standard land rollout.  All three are ignored by
    the non-CLM-ML schemes and by forward-only runs.
    """
    lp = land_params
    T_soil = state.T_soil        # (ncol, n_layers)
    psi = state.psi_soil         # (ncol, n_layers)
    theta = state.theta_soil     # (ncol, n_layers)
    snow = state.snow_depth      # (ncol,)
    snow_age = state.snow_age    # (ncol,)

    grid = make_soil_grid(config.soil_grid)

    # Spatially-varying surface parameters (or config scalar fallbacks).
    _albedo_base = _get(lp, "albedo_veg", config.albedo_land)   # snow-free veg/soil base
    # Dry-soil brightening (Oleson et al. 2013 / CLM): snow-free bare soil brightens
    # by up to ``soil_dry_albedo_boost`` as the top layer dries out, so a desert
    # (theta~0.05) is bright while moist tundra (theta~0.3) stays dark.  This term was
    # dropped in the main land-refactor merge; re-wire it onto the per-cell base albedo
    # BEFORE the surface scheme / snow feedback so the SEB, the banded ``_base`` (line
    # ~372/599) and ``compute_land_albedo`` all see the brightened desert soil.  Uses the
    # START-of-step top-layer moisture here (this base drives the pre-step SEB fluxes);
    # the post-step reported albedo (coupler hand-off) re-brightens with END-of-step
    # moisture below for state consistency.
    albedo_land = _albedo_base + dry_soil_brightening(theta[:, 0], config.land_albedo)
    emissivity = _get(lp, "emissivity", config.emissivity_land)
    z0 = _get(lp, "z0", config.z0_land)

    # Spatially-varying root zone params.
    root_depth = _get(lp, "root_depth", config.root_depth)
    theta_wp = _get(lp, "theta_wp", config.theta_wp)          # SOIL wilting point
    theta_fc = _get(lp, "theta_fc", config.theta_fc)
    # PLANT wilting point drives the ROOT-ZONE transpiration / GPP stress and is
    # kept SEPARATE from the soil wilting point: deep-rooted vegetation extracts
    # water below the soil-evaporation cutoff.  Falls back to ``theta_wp`` (per-
    # column params first, then config) so an unset plant wp reproduces the
    # single-wilting-point behaviour exactly.
    theta_wp_plant = resolve_plant_wilting_point(lp, config)

    # Start-of-step skin temperature = top soil layer.
    T_surface = T_soil[:, 0]
    ncol = T_surface.shape[0]

    # --- Sub-grid elevation-band snow (opt-in gaps 1,2,4) ---
    # Re-partition the cell precipitation phase PER elevation band (lapse-downscaled
    # air T), so a warm cell whose high sub-grid fractions sit below freezing still
    # accumulates snow there.  The cell-level snow logic below then runs on the
    # area-weighted band aggregates; ``band_net_radiation`` (below) replaces the
    # cell-mean radiation in ``G_surface`` with the banded per-band balance.
    bands = config.elev_bands
    if bands is not None:
        if isinstance(config.surface_scheme,
                      (TwoLeafCanopyConfig, CLMMLCanopyConfig)):
            # The banded radiation replaces the surface scheme's radiation/G with a
            # BARE (no-canopy) per-band balance; mixing it with a canopy scheme's
            # radiative closure (two-leaf Kelvin RT, or CLM-ML's multilayer canopy
            # RT — both carry their own T_rad/lw_up) would be inconsistent.  Reject
            # rather than silently apply incompatible closures.  Applies to BOTH
            # canopy schemes (the guard predates CLMMLCanopyConfig, which needs the
            # same treatment as TwoLeafCanopyConfig).
            raise ValueError(
                "config.elev_bands (sub-grid snow bands) is not supported with the "
                "TwoLeafCanopyConfig or CLMMLCanopyConfig surface scheme — the "
                "banded radiation would override the canopy radiative closure.  "
                "Use SimpleSEBConfig with bands.")
        if state.snow_bands is None:
            raise ValueError(
                "config.elev_bands is set but state.snow_bands is None; initialise "
                "the state with init_multilayer_land_state(config=...) so the banded "
                "SWE field exists.")
        ice_bands_in = (jnp.zeros_like(state.snow_bands) if state.ice_bands is None
                        else state.ice_bands)
        snowfall_bands = band_precip_snow(
            forcing.T_lowest, forcing.precip_total, forcing.precip_snow, bands)
        precip_snow_eff = jnp.mean(snowfall_bands, axis=-1)   # equal-area bands
    else:
        precip_snow_eff = forcing.precip_snow

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
    # Root-zone stress uses the PLANT wilting point (transpiration extraction
    # limit); the soil wilting point ``theta_wp`` is the soil-column reference.
    theta_wp_c   = _to_ncol(theta_wp_plant)
    theta_fc_c   = _to_ncol(theta_fc)

    root_frac = jnp.exp(-z_centers[None, :] / root_depth_c[:, None])
    root_frac = root_frac / jnp.sum(root_frac, axis=-1, keepdims=True)

    # --- Moisture availability from root-zone water content ---
    # Root-zone moisture stress via the shared helper (single source of truth;
    # see root_zone_beta_soil for the audit-#6 / iter-65 theta_fc-theta_wp
    # range floor).  theta_wp_c / theta_fc_c are already promoted to (ncol,),
    # so use the spatial (per-column) path.  beta_root (per-layer) is reused
    # below for the root-water-uptake sink partition.  (#823 keeps this
    # root_zone_beta_soil path — numerically identical to main's canonical
    # root_zone_moisture_stress, which stays available for the coupler land
    # tile via land_tile_beta_soil — because the soil-C-spin-up + carbon audit
    # downstream reuse the inline root_frac / theta_wp_c / theta_fc_c / w_frac_rz.)
    beta_soil, beta_root = root_zone_beta_soil(
        theta, root_frac, theta_wp_c, theta_fc_c, config.beta_min,
        spatial=True,
    )
    # Root-zone-integrated wetness (vegetation-cover proxy), reused downstream
    # for the transpiration/bare-soil evaporation partition (f_veg).  Same
    # reduction the helper applies internally for beta_soil.
    w_frac_rz = jnp.clip(jnp.sum(root_frac * beta_root, axis=-1), 0.0, 1.0)

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
    canopy_state_new = None  # updated only by CLMMLCanopyConfig branch
    _alpha_applied = None    # set by the two-leaf branch: one albedo, absorbed + exported
    _lp_soil = None          # two-leaf: params with soil bands at start-of-step water
    if isinstance(config.surface_scheme, TwoLeafCanopyConfig):
        # Canopy surface scheme: Newton closure with Picard loop that
        # advances soil thermal tentatively between passes.  These run before
        # Richards on unchanged theta, so they carry no moisture fusion source.
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

        # Kelvin pore relative humidity h_r = exp(psi_top g/(R_v T)) — the
        # thermodynamic vapour-pressure lowering of the drying surface (bites only
        # near residual water).  Shape (ncol,), in state precision.
        _h_r_top = jnp.exp(jnp.minimum(
            psi[:, 0] * constants.g
            / (constants.R_v * jnp.maximum(T_soil[:, 0], 1.0)), 0.0)).astype(theta.dtype)
        # Top-layer RELATIVE saturation W_1 = theta_1/theta_sat for the Sellers-1992
        # surface resistance (computed in layer space then sliced, same broadcast-safe
        # pattern as _S_top above).
        _W1_top = jnp.clip(
            theta / jnp.maximum(config.hydraulics.theta_sat, 1e-6),
            1e-6, 1.0)[:, 0].astype(theta.dtype)

        # Wetted leaf fraction from the START-of-step canopy-water store, driving
        # the wet-leaf evaporation INSIDE the canopy energy balance (interception
        # loss: a wet leaf evaporates at the boundary-layer limit, so LE rises).
        # ``None`` when interception is off — the canopy then runs the pure-
        # stomatal balance unchanged.
        _fwet_pre = None
        if config.interception is not None and state.W_canopy is not None:
            _lai_i = jnp.broadcast_to(
                _get(lp, "LAI", jnp.zeros_like(T_surface)), T_surface.shape)
            _sai_i = _get(lp, "SAI", None)
            _pai_i = _lai_i + (0.0 if _sai_i is None
                               else jnp.broadcast_to(_sai_i, T_surface.shape))
            _fwet_pre = interception_wetted_fraction(
                state.W_canopy, _pai_i, config.interception)

        # Soil-colour bands follow the START-of-step top-layer water (CTSM
        # evaluates the soil albedo from the current h2osoi_vol); parameter
        # sets without soil-colour bounds carry a prescribed albedo and pass
        # through unchanged.
        lp = rewet_soil_bands(lp, theta[:, 0])
        _lp_soil = lp
        # ONE surface albedo for absorption and for export.  The canopy RT's
        # band albedos are the snow-free soil-colour background, so without
        # this the land absorbed sunlight through ~0.15 while the atmosphere
        # reflected the exported snow-aged 0.52 on the same cell -- nothing
        # reconciled them (energy created on snow-covered tundra, lost on
        # glacier).  Snow is layered on each band's own base, so snow-free
        # columns absorb exactly as before and the calibrated glacier bands
        # survive; the prognostic dry-soil brightening the old export added
        # is NOT applied here because the bands carry the CTSM soil-colour
        # moisture dependence (rewet above) and both reviewers flagged the
        # double count.  Absorption this step is exactly (1 - broadband of
        # these bands) * sw_down in daylight (the RT's own low-light fallback,
        # sw_down < 1 W/m2, is the only exception).  With snow feedback on, the
        # EXPORT (post-step block below) is the same construction on the
        # post-step snow and soil water, i.e. what the NEXT step absorbs with,
        # so the hand-off to the next radiation call is exact; within one step
        # they differ by that step's snow and top-layer water change.
        if (config.snow_albedo_feedback and lat is not None
                and lp is not None and hasattr(lp, "ALB_VIS")):
            _band = lambda a: compute_land_albedo(
                lat, snow, snow_age, config.land_albedo,
                base_albedo=jnp.broadcast_to(a, T_surface.shape))
            lp = lp._replace(ALB_VIS=_band(lp.ALB_VIS), ALB_NIR=_band(lp.ALB_NIR))
            _alpha_applied = broadband_albedo(lp.ALB_VIS, lp.ALB_NIR)
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
            fwet=_fwet_pre,
            # Warm start: the previous step's last converged canopy solution
            # (None on a state that does not carry the cache -> cold start).
            canopy_seed=state.canopy_x,
            # Bare-soil evaporation efficiency = TWO complementary top-layer
            # limiters, applied as a beta conductance efficiency in the canopy
            # soil energy balance (both tie evaporation to the fast-drying
            # SURFACE, not the root-zone average):
            #   * Kelvin pore RELATIVE HUMIDITY  h_r = exp(psi_top g /(R_v T))
            #     — thermodynamic vapour-pressure lowering; only bites as the
            #     surface approaches residual (psi -> -inf).
            #   * Soil-moisture control, ONE of two mutually-exclusive forms
            #     (never both — same Sellers-1992 physics, or the limitation
            #     double-counts):
            #       - series_resistance ON (default): Kelvin h_r ONLY here; the
            #         moisture control is the Sellers-1992 SURFACE RESISTANCE r_ss
            #         (+ SZ09 litter), added in SERIES with raw_below inside the
            #         canopy soil energy balance via ``soil_surface_relsat`` below.
            #         Fixes the aerodynamic-only path that let a wet forest floor
            #         evaporate at near-potential rate (LE_soil ~57% of total at
            #         US-MMS; <15% is physical).
            #       - series_resistance OFF (legacy): the beta EFFICIENCY
            #         S_top**soil_evap_resistance_exp (Sellers-1992 / Lee-Pielke-1992
            #         diffusion crust; the SimpleSEB #671 counterpart), which
            #         throttles the conductance but barely helps a surface that
            #         rewets to S_top~1.  exp=0 recovers Kelvin-only.
            w_frac_soil_evap=(
                _h_r_top if config.soil_evap_series_resistance
                # _S_top floored at 1e-6 keeps d(S_top**exp)/dS_top finite at the
                # residual boundary for a trainable exp<1 (AD-safe; see above).
                else _h_r_top * _S_top ** config.soil_evap_resistance_exp),
            soil_surface_relsat=_W1_top,
        )
    elif isinstance(config.surface_scheme, CLMMLCanopyConfig):
        # CLM-ML-JAX multilayer canopy scheme (Phase 3 implementation).
        # Architectural constraints:
        #   1. NOT jax.jit-compatible — CLM-ML uses Python/NumPy control flow internally.
        #   2. ``lon`` is not forwarded; a virtual longitude is inferred from
        #      forcing.cos_zenith via CLM's shr_orb_cosz inversion for round-trip consistency.
        #   3. ``canopy_state.t_a10_arr`` (np.ndarray, shape (ncol,)) carries the
        #      10-day running-mean air temperature on the Python host between steps.
        # Lazy import keeps clm_ml_jax optional.
        from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes

        # Prognostic LAI feedback for CLM-ML — mirrors the two-leaf branch above.
        # When ``CLMMLCanopyConfig.use_prognostic_lai`` is set AND differland
        # carbon is active, ``compute_prognostic_lai`` returns C_fol / LCMA and it
        # supersedes the prescribed ``LandSurfaceParams.LAI`` inside CLM-ML; it
        # returns ``None`` otherwise, so CLM-ML falls back to the prescribed
        # climatology (or its scalar default).  Canopy structure (SAI/htop/hbot)
        # stays prescribed — the carbon cycle produces no allometric mapping.
        # FORWARD-ONLY coupling: the whole CLM-ML interface is eager/non-jit
        # (Python-NumPy control flow, host-side ``float(lai_override[i])`` reads),
        # so this is a prognostic forward feedback (carbon evolves -> LAI updates
        # each step), NOT a differentiable one — do not ``jax.grad`` through it.
        LAI_override = compute_prognostic_lai(
            carbon_state, config, config.surface_scheme)

        # Wind-speed floor PARITY with the two-leaf arm: the CLM-ML backend takes
        # uref = sqrt(u^2 + v^2) with NO floor, so in calm/stable (night) air it
        # sees wind -> 0, collapsing u*/aerodynamic conductance and under-predicting
        # H.  Scale (u, v) direction-preserving so their magnitude equals the same
        # sqrt(u^2 + v^2 + U_min^2) floor the two-leaf arm applies above.
        _wsp_floored = jnp.sqrt(
            forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2)
        _wsp_raw = jnp.sqrt(forcing.u_lowest ** 2 + forcing.v_lowest ** 2)
        _calm = _wsp_raw > 1e-6
        # Direction-preserving rescale to the floored magnitude; at (near-)calm the
        # direction is undefined, so fall back to (U_min, 0) — a nonzero magnitude
        # (=U_min) with an arbitrary but definite direction, so CLM-ML's
        # sqrt(u^2+v^2) never collapses to 0 the way the raw wind would.
        _sc = jnp.where(_calm, _wsp_floored / jnp.maximum(_wsp_raw, 1e-6), 0.0)
        clmml_forcing = forcing._replace(
            u_lowest=jnp.where(_calm, forcing.u_lowest * _sc, U_min),
            v_lowest=jnp.where(_calm, forcing.v_lowest * _sc, 0.0))
        surface_out, canopy_state_new = compute_clm_ml_canopy_fluxes(
            T_soil_top=T_surface,
            forcing=clmml_forcing,
            canopy_config=config.surface_scheme,
            land_config=config,
            land_params=lp,
            canopy_state=state.canopy_state,
            dt=dt,
            T_soil=T_soil,
            psi_soil=psi,
            theta_soil=theta,
            lat=lat,
            doy=doy,
            lai_override=LAI_override,
            grid_info=clm_ml_grid_info,
            pft_per_col=clm_ml_pft_per_col,
            vcmaxpft_jax=clm_ml_vcmaxpft_jax,
            g1_medlyn_jax=clm_ml_g1_medlyn_jax,
        )
    elif isinstance(config.surface_scheme, SimpleSEBConfig):
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
    else:
        raise ValueError(
            f"Unknown surface_scheme type: {type(config.surface_scheme)!r}. "
            f"Expected TwoLeafCanopyConfig, CLMMLCanopyConfig, or SimpleSEBConfig."
        )

    # =================================================================
    # Shared post-flux pipeline
    # =================================================================
    shflx = surface_out.shflx
    # stflx_air + stflx_veg (CLM-ML canopy heat storage) are intentionally
    # NOT folded into shflx.  They are internal canopy redistribution terms,
    # not turbulent SH going to the atmosphere.  Standard CLM-CAM coupling
    # reports eflx_sh_tot = shflx_canopy without stflx (Bonan et al. 2021).
    # The coupler sees a small per-step residual that integrates to zero diurnally.
    lhflx = surface_out.lhflx
    tau_x = surface_out.tau_x
    tau_y = surface_out.tau_y
    G_surface = surface_out.G_soil

    # --- Banded surface radiation (gaps 1,2): override the cell-mean radiation in
    # G_surface with the area-weighted per-band balance (elevation-lapsed SW/LW +
    # per-band albedo + per-band skin T), keeping the (cell-mean) turbulent fluxes
    # from the surface scheme.  ``band_rad.Rn_bands`` drives per-band melt below.
    if bands is not None:
        # per-band cover with the canopy snow mask applied and clipped PER BAND
        # (post-aggregate scaling is wrong on saturated bands — see land_albedo)
        _scl = (1.0 if config.land_albedo.snow_cover_scale is None
                else jnp.asarray(config.land_albedo.snow_cover_scale)[:, None])
        _cover_fn = lambda s: jnp.clip(
            snow_cover_fraction(s, config.land_albedo) * _scl, 0.0, 1.0)
        # gap 3: solar-zenith snow brightening (cos_zenith per cell -> broadcast over
        # the band axis).  Inactive where cos_zenith is a constant placeholder.
        _cz = forcing.cos_zenith[:, None]
        _alb_fn = lambda a: snow_albedo(a, config.land_albedo, cos_zenith=_cz)
        _T_sfc_band = T_surface[:, None] - bands.lapse_rate_K_m * bands.band_dz
        _band_surviving = ((snowfall_bands * dt > 1e-6)
                           & (_T_sfc_band < constants.T_freeze))
        _bands_eff = state.snow_bands + jnp.where(
            _band_surviving, snowfall_bands * dt, 0.0)
        if config.snow_albedo_feedback and lat is not None:
            # Snow-free base = ``albedo_land`` (per-cell CLM map / lp.albedo_veg, or the
            # config scalar carrying the trainable pft_alb) so the base flows to the
            # gradient — NOT the latitude-band veg albedo, which would zero pft_alb.
            _base = jnp.broadcast_to(albedo_land, T_surface.shape)
            alpha_bands = band_albedo(
                _bands_eff, state.snow_age_bands, _base, _cover_fn, _alb_fn,
                ice_bands=ice_bands_in, cfg=bands)
        else:
            alpha_bands = jnp.broadcast_to(
                jnp.reshape(albedo_land, (-1, 1)), (ncol, bands.band_dz.shape[-1]))
        band_rad = band_net_radiation(
            T_surface, alpha_bands, forcing.sw_down, forcing.lw_down, emissivity, bands)
        # surface_out.G_soil = sw_net + lw_net - shflx - lhflx; swap the radiation.
        G_surface = band_rad.sw_net_agg + band_rad.lw_net_agg - shflx - lhflx
    else:
        band_rad = None

    # --- Snow phase (iter-68 — consistent with simple_seb's flux calc) ---
    # Reuse the same warm-surface-snowfall gate as simple_seb.py so
    # downstream latent-mass partition (sublimation vs soil evap) is
    # consistent with the L_eff that produced the demand.  Ported from
    # main during the jianing/land ↔ main sync 2026-06-03.  Uses the banded
    # ``precip_snow_eff`` (== forcing.precip_snow when bands are off).
    fresh_snow_mass = precip_snow_eff * dt
    has_existing_snow = snow > 1e-6
    has_surviving_fresh_snow = (
        (fresh_snow_mass > 1e-6) & (T_surface < constants.T_freeze)
    )
    has_snow = has_existing_snow | has_surviving_fresh_snow

    # --- Snow (+ firn/ice, banded) budget (energy-limited melt) ---
    if bands is not None:
        # Per-band melt limited by each band's OWN net radiation minus the (cell)
        # turbulent fluxes; area-weighted mean equals the aggregate G_surface, so
        # soil-column energy is conserved.  Firn/ice reservoir per gap 4.
        # Per-band rain (gap 6 refreeze input) + cell wind (gap 5 blowing snow).
        _precip_rain_bands = forcing.precip_total[:, None] - snowfall_bands
        band_step = step_snow_bands(
            state.snow_bands, state.snow_age_bands, ice_bands_in, T_surface,
            snowfall_bands, dt,
            Q_net=band_rad.Rn_bands - shflx[:, None] - lhflx[:, None],
            cfg=bands, T_snow_melt=config.T_snow_melt,
            snow_age_activation_K=config.land_albedo.snow_age_activation_K,
            precip_rain_bands=_precip_rain_bands, wind=wind_speed)
        snow_bands_new = band_step.swe_bands
        ice_bands_new = band_step.ice_bands
        snow_age_bands_new = band_step.snow_age_bands
        snow_new = band_step.swe_total
        snow_age_new = band_step.snow_age
        snow_melt = band_step.snow_melt
        ice_melt = band_step.ice_melt
        refreeze = band_step.refreeze          # (ncol,) rain refrozen [kg/m2] (gap 6)
        blow_subl = band_step.blow_subl        # (ncol,) blowing-snow sublimation [kg/m2/s]
        # Frozen glacier discharge + ablation ice meltwater both leave as runoff.
        cap_runoff = band_step.ice_runoff + ice_melt / dt
    else:
        snow_new, snow_age_new, snow_melt = update_snow(
            snow, snow_age, T_surface, precip_snow_eff, dt,
            Q_net=G_surface,
            snow_melt_rate=config.snow_melt_rate,
            T_snow_melt=config.T_snow_melt,
            snow_age_activation_K=config.land_albedo.snow_age_activation_K,
        )
        snow_bands_new = state.snow_bands
        snow_age_bands_new = state.snow_age_bands
        ice_bands_new = state.ice_bands
        ice_melt = jnp.zeros_like(snow_new)
        refreeze = jnp.zeros_like(snow_new)
        blow_subl = jnp.zeros_like(snow_new)
        cap_runoff = jnp.zeros_like(snow_new)
    # Energy into the surface budget: seasonal-snow + ablation-ice melt CONSUME L_f;
    # rain-on-snow refreezing (gap 6) RELEASES L_f; blowing-snow sublimation (gap 5)
    # consumes L_s.  (Frozen glacier discharge leaves as ice — no fusion.)
    melt_energy = ((snow_melt + ice_melt - refreeze) * constants.L_f / dt
                   + blow_subl * constants.L_s)
    G_surface = G_surface - melt_energy

    # --- Latent mass partition (component- and phase-correct, water-limited) ---
    # The surface latent flux is split into a SNOWPACK-sublimation energy stream
    # (charged at L_s, drawn from / deposited on the pack) and a SOIL / plant-
    # water evaporation stream (charged at L_v), by COMPONENT and phase, so each
    # unit of latent ENERGY removes the correct vapour MASS from the correct
    # reservoir:
    #   * SimpleSEB (bare surface): the whole lhflx is the ground flux -> the
    #     snowpack when snow covers the cell (L_s), else the soil top (L_v).
    #   * Two-leaf / CLM-ML canopy: lhflx = LE_canopy (leaf transpiration + wet-
    #     leaf evaporation drawn from soil / plant water ABOVE the snow, at L_v) +
    #     LE_soil (the BELOW-canopy GROUND latent).  Over snow the ground surface
    #     IS the snowpack, so LE_soil sublimates from the pack at L_s while the
    #     transpiration stream still draws soil water at L_v.  A NEGATIVE
    #     transpiration flux over snow is canopy dew with no canopy-water
    #     reservoir, so it frosts the pack (L_s) alongside the ground component.
    #     Charging the WHOLE canopy latent at L_s (pre-audit) mis-phased the
    #     transpiration (~12% mass error, wrong reservoir); charging it all at
    #     L_v soil (first audit pass) mis-routed the ground sublimation off the
    #     pack.  ``surface_out.LE_soil`` carries the ground component for BOTH
    #     canopy schemes (two-leaf ``LE_Soil``; CLM-ML ``lhsoi_soil``).  The
    #     ``LE_soil is None`` fallback (whole positive latent -> L_v soil) is a
    #     bounded last resort for a canopy scheme that does not expose a ground
    #     component (e.g. an older clm-ml-jax lacking ``lhsoi_soil``).
    # ``scheme_is_seb`` and ``LE_soil is None`` are STATIC (trace-time) branches;
    # ``transp_to_snow`` is the (traced) canopy-dew-over-snow mask.
    rho_w = constants.rho_water
    scheme_is_seb = isinstance(config.surface_scheme, SimpleSEBConfig)
    if scheme_is_seb:
        lhflx_ground = lhflx
        lhflx_transp = jnp.zeros_like(lhflx)
    elif surface_out.LE_soil is not None:
        lhflx_ground = surface_out.LE_soil
        lhflx_transp = lhflx - lhflx_ground
    else:  # canopy scheme without an exposed ground component (CLM-ML)
        lhflx_ground = jnp.zeros_like(lhflx)
        lhflx_transp = lhflx
    transp_to_snow = has_snow & (lhflx_transp < 0.0)
    snow_latent = (jnp.where(has_snow, lhflx_ground, 0.0)
                   + jnp.where(transp_to_snow, lhflx_transp, 0.0))
    soil_latent = lhflx - snow_latent

    # --- Snowpack sublimation / frost (L_s), pack-limited ---
    snow_after_melt = snow_new
    max_sublim = jnp.maximum(snow_after_melt / dt, 0.0)
    sublim_demand = snow_latent / constants.L_s
    sublim_actual = jnp.minimum(sublim_demand, max_sublim)
    sublim_actual = jnp.where(sublim_demand < 0.0, sublim_demand, sublim_actual)
    snow_new = jnp.maximum(snow_new - sublim_actual * dt, 0.0)
    if bands is not None:
        # Redistribute the aggregate sublimation/deposition across bands (preserve the
        # band distribution + aggregate mass); empty-pack deposition (frost) spreads
        # equally across the equal-area bands so mass is not lost.
        _agg_ok = snow_after_melt > 1e-12
        _scale = jnp.where(
            _agg_ok, snow_new / jnp.where(_agg_ok, snow_after_melt, 1.0), 1.0)
        _deposit_empty = (~_agg_ok) & (snow_new > 1e-12)
        snow_bands_new = jnp.where(
            _deposit_empty[:, None], snow_new[:, None], snow_bands_new * _scale[:, None])

    dz = grid.dz
    extractable_water = jnp.sum(
        jnp.maximum(theta - theta_r, 0.0) * dz[None, :], axis=-1) * rho_w
    # Rain that refroze into the pack (gap 6) is now snow, so it no longer infiltrates.
    precip_rain = forcing.precip_total - precip_snow_eff - refreeze / dt
    melt_rate = snow_melt / dt

    # --- Canopy interception, phase 1: intercept rain into the store ----------
    # Only the THROUGHFALL (direct + drip) infiltrates, so the water-availability
    # limiter below and the Richards top flux both see ``infil_rain`` (not raw
    # precip).  The wet-leaf evaporation (phase 2) runs after the transpiration
    # partition so it can be capped by the transpiration demand (closure).  Two-
    # leaf path only (CLM-ML has its own internal store; SimpleSEB has no canopy
    # latent stream).  See land/canopy/interception.py.
    # Gated ALSO on ``state.W_canopy is not None`` so the step never changes the
    # carry pytree structure (``None`` -> array would break a ``lax.scan``) and
    # never intercepts water it cannot store (which would leak): the store is
    # allocated at init / restored from restart when interception is configured,
    # so a genuine run always carries it (codex).
    _do_intercept = (config.interception is not None
                     and isinstance(config.surface_scheme, TwoLeafCanopyConfig)
                     and state.W_canopy is not None)
    infil_rain = precip_rain
    _W_int = None
    _pai = None
    if _do_intercept:
        _lai = jnp.broadcast_to(_get(lp, "LAI", jnp.zeros_like(precip_rain)),
                                precip_rain.shape)
        _sai = _get(lp, "SAI", None)
        _pai = _lai + (0.0 if _sai is None
                       else jnp.broadcast_to(_sai, precip_rain.shape))
        _W_int, _throughfall = intercept_rain(
            state.W_canopy, jnp.maximum(precip_rain, 0.0), _pai, dt,
            config.interception)
        # ``_throughfall`` already carries the canopy DRIP (which is nonzero even
        # at zero rain when the plant area — hence storage capacity — shrinks, so
        # it must NOT be discarded, else that water leaks; codex).  Add back any
        # negative "rain" (numerical / refreeze deficit) so the column budget is
        # unchanged in that edge case.
        infil_rain = _throughfall + jnp.minimum(precip_rain, 0.0)
    # --- Soil / plant-water evaporation (L_v), water-limited ---
    soil_evap_demand = soil_latent / constants.L_v
    # Bare-soil evaporation resistance (#671, Sellers 1992 / Lee & Pielke 1992):
    # throttle the (positive, evaporative) bare-soil demand by the TOP-layer
    # effective saturation S_top**exp — the surface dries into a high-resistance
    # crust far faster than the root-zone mean.  GATED to SimpleSEB ONLY: BOTH
    # canopy schemes (two-leaf Kelvin h_r, and CLM-ML's Philip (1957) rhg_soil
    # soil-humidity closure) apply their OWN top-layer soil-evap throttle inside
    # the canopy energy balance, so surface_out.lhflx already reflects it;
    # applying S_top**exp again here would double-throttle AND wrongly throttle
    # the canopy transpiration folded into the total lhflx.  Testing
    # ``isinstance(SimpleSEBConfig)`` (not ``not isinstance(TwoLeafCanopyConfig)``)
    # so the third scheme, CLMMLCanopyConfig, is correctly excluded too.  Dew
    # (demand<0) left un-throttled.
    if isinstance(config.surface_scheme, SimpleSEBConfig):
        # Bare-soil evaporation efficiency = Kelvin pore RELATIVE HUMIDITY x diffusion-crust
        # resistance, matching the two-leaf-canopy path (above) and CLM5 (Oleson 2013):
        #   * h_r = exp(psi_top g /(R_v T_top)) — pore-space vapour-pressure lowering; ~1
        #     except as the surface nears residual (psi -> -inf), where it shuts evap off.
        #     Previously MISSING on the SimpleSEB path (only S_top**exp), unlike the canopy.
        #   * _S_top (top-layer effective saturation, floored at 1e-6 in its shared
        #     definition above so d(S_top**exp)/dS_top stays finite at the residual-water
        #     boundary for a trainable exp < 1; AD-safe, negligible fwd).
        _h_r = jnp.exp(jnp.minimum(
            psi[:, 0] * constants.g
            / (constants.R_v * jnp.maximum(T_soil[:, 0], 1.0)), 0.0))
        _beta_surf = (_h_r * _S_top ** config.soil_evap_resistance_exp).astype(
            soil_evap_demand.dtype)
        soil_evap_demand = jnp.where(
            soil_evap_demand > 0.0, soil_evap_demand * _beta_surf, soil_evap_demand)
    # Water available to bare-soil evaporation uses the THROUGHFALL (infil_rain),
    # not raw precip, since interception removed the intercepted part upstream.
    max_soil_evap = jnp.maximum(
        extractable_water / dt + infil_rain + melt_rate, 0.0)
    soil_evap = jnp.minimum(soil_evap_demand, max_soil_evap)

    # --- Root water uptake partition ---
    # ``soil_flux`` is the L_v soil / plant-water stream ONLY: the snowpack
    # already swallowed the sublimation / frost stream (``sublim_actual``)
    # upstream, so routing any of it through ``flux_top`` would double-count the
    # mass (codex iter-23 stop-time review).  Within this stream a negative value
    # is dew / downward deposition on bare soil, which routes ENTIRELY to
    # ``flux_top`` (transpiration sink = 0) so the column water budget closes;
    # vegetated-fraction dew on a snow-free cell is treated as bare-soil input
    # (no separate canopy-storage reservoir in this model).
    #
    # Bare-soil-top vs root-sink partition: the snow-free MIXED stream (bare-soil
    # surface evaporation + transpiration) is split by the root-zone wetness
    # ``f_veg`` heuristic, sending (1 - f_veg) through the top boundary.  OVER SNOW
    # the soil stream is PURE canopy transpiration — the below-canopy ground
    # component (LE_soil) has already sublimated from the pack — so it must be
    # drawn from the ROOT ZONE in full (transp_frac = 1), never partly through the
    # top-soil boundary (which would corrupt the surface water balance under
    # snow).  SimpleSEB over snow leaves soil_flux == 0, so the branch is a no-op
    # for it.
    f_veg = jnp.clip(w_frac_rz, 0.0, 1.0)
    evap_bare, evap_transp = _partition_latent_root_top(
        soil_evap, has_snow, f_veg,
        surface_out.LE_canopy, surface_out.LE_soil)

    # --- Canopy interception, phase 2: deplete the store by the wet-leaf flux --
    # The canopy energy balance already computed the wet-leaf evaporation
    # (``surface_out.LE_wet_canopy``, the fwet share of the boosted LE — a real
    # interception-loss latent flux, NOT re-labelled transpiration).  That water
    # is drawn from the store; the REST of the canopy latent stays transpiration
    # from the root zone.  Cap the draw by the store (fwet is bounded, so this
    # rarely bites); any store shortfall reverts that flux to the soil sink, so
    # the total soil+canopy water budget closes against precip - ET - runoff and
    # the reported LE is unchanged.
    W_canopy_new = state.W_canopy
    _wet_evap = jnp.zeros_like(evap_transp)
    if _do_intercept:
        _wet_evap_demand = jnp.maximum(
            surface_out.LE_wet_canopy, 0.0) / constants.L_v   # kg m-2 s-1
        # Cap by BOTH the store (can't evaporate water it doesn't hold) AND the
        # transpiration the caller is about to draw (the wet-leaf flux re-sources
        # transpiration; drawing more than that from the store would remove more
        # surface water than the reported atmospheric latent flux — codex).
        _wet_evap = jnp.minimum(
            jnp.minimum(_wet_evap_demand, _W_int / dt),
            jnp.maximum(evap_transp, 0.0))
        W_canopy_new = jnp.maximum(_W_int - _wet_evap * dt, 0.0)
        # Wet-leaf water came from the store, so remove it from the root-zone
        # transpiration sink; wet_evap <= evap_transp keeps this >= 0.
        evap_transp = evap_transp - _wet_evap

    flux_top = (infil_rain + melt_rate - evap_bare) / rho_w

    E_pot_transp = jnp.maximum(evap_transp, 0.0) / rho_w
    weight = root_frac * beta_root  # both are (ncol, n_layers)
    weight_sum = jnp.sum(weight, axis=-1, keepdims=True)
    weight_norm = weight / jnp.maximum(weight_sum, 1e-20)
    sink = weight_norm * E_pot_transp[:, None] / dz[None, :]

    # --- Frozen-soil ice impedance (CLM5 IceImpedance), freeze/thaw lanes only ---
    # At start-of-step T and water, held for the step.  Off -> original Richards path.
    _log_imped = (soil_ice_log_impedance(T_soil, theta, config)
                  if config.thermal.enable_freeze_thaw else None)

    # --- Richards equation (+ coupled surface ponding cell, #671) ---
    richards_out = solve_richards(
        psi, theta, grid,
        config.hydraulics, config.richards,
        flux_top, sink, dt,
        surface_water=state.surface_water,
        log_impedance=_log_imped,
    )
    # NB (#671): the former "evaporation water budget closure" — a clip of
    # theta_new to [theta_r, theta_sat] — is removed.  It was a non-conservative
    # band-aid for the old infiltration/evap mismatch; theta_from_psi is now
    # bounded below at theta_r by construction, the coupled surface cell carries
    # the ponded excess, and the mixed-form solve closes the water budget.

    # --- Realised soil / plant-water evaporation (what the soil actually gave) ---
    # A draw the dry column cannot supply (bare-soil top flux or root sink) ends
    # at the Richards psi dry floor, which REFILLS it: the soil loses less than
    # the demand (measured: 0.51 of a 0.8 mm/d desert demand was refilled).  The
    # evaporation handed to the atmosphere is therefore the demand minus that
    # refill (``richards_out.refill``, bounded by the solver's draw, so a negative
    # or non-draw residual is never turned into evaporation or dew; the solver
    # takes non-draw created water back out of the soil).  Draws are NOT pre-capped by
    # start-of-step layer water: that would also cut legitimate capillary supply
    # from moister layers below (-3.7% of a moist-subsoil day's ET, measured).
    # Transpiration is rebuilt from the sink the solve received, so a column with
    # zero root weight reports none.  Units: kg m-2 s-1, positive = upward.
    evap_transp = rho_w * jnp.sum(sink * dz[None, :], axis=-1)
    soil_evap = evap_bare + evap_transp + _wet_evap - richards_out.refill * rho_w / dt

    # --- Combine the two phase streams ---
    # Total vapour mass leaving the surface = pack sublimation + soil / plant
    # evaporation; total latent energy = their L_s / L_v weighted sum.  Demand
    # unmet by a reservoir cap, the bare-soil resistance or the soil supply limit
    # returns to the ground heat flux as ``evap_excess_energy`` (below) so the
    # surface energy budget still closes (in - out - dStorage = 0); the skin
    # temperature is not re-solved this step.
    lhflx_actual = sublim_actual * constants.L_s + soil_evap * constants.L_v
    evap_excess_energy = lhflx - lhflx_actual

    # --- Soil thermal diffusion (final, with converged G) ---
    # Semi-implicit surface conductance (Robin BC): the SimpleSEB scheme returns a
    # linearised lambda = -dG/dT_sfc that makes the surface energy balance's
    # T_sfc-dependence implicit here, removing the explicit-coupling large-dt/thin-
    # top-layer instability.  None for the two-leaf canopy (its Newton closure owns
    # the coupling) and for slab builds that leave it unset -> explicit BC, unchanged.
    G_surface = G_surface + evap_excess_energy
    # Fusion heat of the ice change Richards made at fixed T (evaluated at the
    # start-of-step T the apparent heat capacity uses).
    _fusion_source = (
        moisture_fusion_heat_source(
            T_soil, theta, richards_out.theta_new, dz,
            config.thermal, dt)
        if config.thermal.enable_freeze_thaw else None)
    T_soil_new = solve_soil_thermal(
        T_soil, richards_out.theta_new, grid,
        config.hydraulics, config.thermal,
        G_surface, dt,
        surface_conductance=surface_out.surface_conductance,
        layer_source=_fusion_source,
        n_substeps=(FINAL_THERMAL_SUBSTEPS
                    if config.thermal.enable_freeze_thaw else 1),
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

    # Banded glacier discharge + ablation ice meltwater (``cap_runoff``) leave via the
    # runoff/freshwater path WITHOUT infiltrating (glacier outflow, not soil water).
    runoff_surface_total = richards_out.runoff_surface + cap_runoff
    new_state = MultiLayerLandState(
        T_soil=_match(T_soil_new, state.T_soil),
        psi_soil=_match(richards_out.psi_new, state.psi_soil),
        theta_soil=_match(richards_out.theta_new, state.theta_soil),
        runoff_surface=_match(runoff_surface_total, state.runoff_surface),
        runoff_subsurface=_match(richards_out.runoff_subsurface,
                                 state.runoff_subsurface),
        snow_depth=_match(snow_new, state.snow_depth),
        snow_age=_match(snow_age_new, state.snow_age),
        TgC=_match(TgC_new, state.TgC),
        surface_water=_match(richards_out.surface_water,   # coupled ponding cell (#671)
                             state.surface_water),
        snow_bands=_match(snow_bands_new, state.snow_bands),
        snow_age_bands=_match(snow_age_bands_new, state.snow_age_bands),
        ice_bands=_match(ice_bands_new, state.ice_bands),
        # CLM-ML canopy carry is a NamedTuple pytree (not a dtype-castable leaf),
        # so it bypasses ``_match``; ``None`` for the non-canopy schemes.
        canopy_state=canopy_state_new,
        # Intercepted-water store: structure-preserving — an array in, an array
        # out (updated on the two-leaf interception path, else carried), a
        # ``None`` in stays ``None``.  ``_do_intercept`` already requires the
        # store to exist, so an active interception step never introduces the
        # store (no ``None`` -> array carry-structure change under a scan).
        W_canopy=(_match(W_canopy_new, state.W_canopy)
                  if state.W_canopy is not None else None),
        # Canopy warm-start cache.  Carried ONLY when the incoming state already
        # carries it, so the pytree structure is invariant under a ``lax.scan``
        # (a None -> array transition mid-scan would be a carry-structure
        # change).  ``surface_out.canopy_x`` is None for every non-canopy
        # surface scheme, in which case the cache is dropped.
        canopy_x=(_match(surface_out.canopy_x, state.canopy_x)
                  if (state.canopy_x is not None
                      and surface_out.canopy_x is not None) else None),
    )

    # --- Post-step surface state for coupler ---
    T_surface_new = T_soil_new[:, 0]
    # Re-brighten the snow-free base with the END-of-step top-layer moisture so the albedo
    # handed to the coupler (drives the next radiation step) is consistent with the updated
    # T_surface_new / snow_new state — the pre-step ``albedo_land`` used start-of-step theta.
    albedo_land_post = _albedo_base + dry_soil_brightening(
        richards_out.theta_new[:, 0], config.land_albedo)

    if bands is not None:
        # Post-step banded albedo + up-welling LW for the atmosphere: the SAME
        # flux-weighted band radiation as the pre-step (gap 1), so the coupler sees a
        # consistent albedo (alpha_eff) and banded LW emission (not a cell-mean value).
        if config.snow_albedo_feedback and lat is not None:
            _base_new = jnp.broadcast_to(albedo_land_post, T_surface_new.shape)
            _cz_new = forcing.cos_zenith[:, None]
            alpha_bands_new = band_albedo(
                snow_bands_new, snow_age_bands_new, _base_new,
                lambda s: jnp.clip(
                    snow_cover_fraction(s, config.land_albedo)
                    * (1.0 if config.land_albedo.snow_cover_scale is None
                       else jnp.asarray(config.land_albedo.snow_cover_scale)[:, None]),
                    0.0, 1.0),
                lambda a: snow_albedo(a, config.land_albedo, cos_zenith=_cz_new),
                ice_bands=ice_bands_new, cfg=bands)
        else:
            alpha_bands_new = jnp.broadcast_to(
                jnp.reshape(albedo_land_post, (-1, 1)), (ncol, bands.band_dz.shape[-1]))
        band_rad_new = band_net_radiation(
            T_surface_new, alpha_bands_new, forcing.sw_down, forcing.lw_down,
            emissivity, bands)
        alpha_new = band_rad_new.alpha_eff
        lw_up_new = band_rad_new.lw_up_agg
    else:
        if _alpha_applied is not None:
            # Same bands, same snow layering, POST-step snow and soil water:
            # the export feeds the NEXT radiation call, whose canopy will
            # absorb with the post-step state (codex).  Absorption this step
            # used the pre-step bands (``_alpha_applied``); the two differ only
            # by one step's snow and top-layer water change.
            _lp_new = rewet_soil_bands(_lp_soil, richards_out.theta_new[:, 0])
            _band_new = lambda a: compute_land_albedo(
                lat, snow_new, snow_age_new, config.land_albedo,
                base_albedo=jnp.broadcast_to(a, T_surface_new.shape))
            alpha_new = broadband_albedo(_band_new(_lp_new.ALB_VIS),
                                         _band_new(_lp_new.ALB_NIR))
        elif getattr(_lp_soil, "ALB_VIS_DRY", None) is not None:
            # Two-leaf without snow layering: the soil bands at the post-step
            # water, i.e. what the next step absorbs with (same hand-off as
            # above, minus snow).
            _lp_new = rewet_soil_bands(_lp_soil, richards_out.theta_new[:, 0])
            alpha_new = broadband_albedo(_lp_new.ALB_VIS, _lp_new.ALB_NIR)
        elif config.snow_albedo_feedback and lat is not None:
            # Per-cell base albedo (CLM PFT / trainable), consistent with the SEB.
            alpha_new = compute_land_albedo(
                lat, snow_new, snow_age_new, config.land_albedo,
                base_albedo=jnp.broadcast_to(albedo_land_post, T_surface_new.shape))
        else:
            alpha_new = surface_out.albedo
        # lw_up recomputed with post-step surface T and surface scheme's
        # effective emissivity (canopy RT vs scalar land emissivity).
        _, _, lw_up_new = surface_radiation_fluxes(
            forcing.sw_down, forcing.lw_down, T_surface_new, alpha_new,
            emissivity,
        )

    # --- Post-step q_surface ---
    # Reuse the same Audit #6 / Iter-65 range floor as the pre-step branch
    # above (now inside root_zone_beta_soil) so degenerate PFT cells cannot
    # blow up beta_soil_new propagating into the q_surface reported back to
    # the atmosphere.  Computed UNCONDITIONALLY (not folded into the
    # non-canopy branch): the carbon cycle below consumes ``beta_soil_new``
    # for EVERY surface scheme, including CLM-ML canopy — moving it inside
    # ``else`` leaves it undefined for canopy + carbon runs.  theta_wp_c /
    # theta_fc_c are already (ncol,), so use the spatial path.
    theta_new = richards_out.theta_new
    beta_soil_new, _ = root_zone_beta_soil(
        theta_new, root_frac, theta_wp_c, theta_fc_c, config.beta_min,
        spatial=True,
    )
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
    # Snow that was present when the scheme computed its humidity but melted
    # away during the step leaves that humidity stale on the ICE curve; the
    # end-state reconstruction below is the honest value for that transition.
    _snow_melted_out = (snow > 1e-6) & ~has_snow_new
    if surface_out.q_surface is not None:
        # The surface scheme SOLVED for its own boundary humidity -- CLM-ML's
        # Philip soil relative humidity, the two-leaf canopy's canopy-air
        # humidity q_c (solved through the stomatal + soil + aerodynamic
        # resistance network), or SimpleSEB's bounded gradient form.  Use it.
        # This branch used to be CLM-ML only, and the else-branch OVERWROTE the
        # two-leaf canopy's solved q_c with the product form beta*q_sat -- the
        # resistance physics ran and was then discarded at the boundary (the
        # slab wrapper preserved it; this wrapper did not).  Snow still
        # overrides to the ice-saturation surface.
        q_sfc_new = jnp.where(has_snow_new, q_sat_sfc_new, surface_out.q_surface)
        q_sfc_new = jnp.where(
            _snow_melted_out,
            forcing.q_lowest
            + jnp.where(has_snow_new, 1.0, beta_new)
            * (q_sat_sfc_new - forcing.q_lowest),
            q_sfc_new)
    else:
        # Scheme returned no humidity: reconstruct the bounded GRADIENT form
        # (never the product form -- beta is a flux efficiency, and beta*q_sat
        # manufactures condensation over dry soil; see simple_seb.py).
        beta_effective_new = jnp.where(has_snow_new, 1.0, beta_new)
        q_sfc_new = (forcing.q_lowest
                     + beta_effective_new * (q_sat_sfc_new - forcing.q_lowest))

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
            _, gpp_override, _ = compute_effective_beta(
                T_surface_new, forcing, beta_soil_new, config, carbon_state,
                dt, land_params=lp)
        carbon_state_new, co2_flux = step_carbon(
            carbon_state, forcing.sw_down, T_surface_new, forcing.co2_ppmv,
            beta_soil_new, lat_arr, doy, forcing.precip_total, config.carbon,
            dt, gpp_override=gpp_override,
            soil_frozen_fraction=soil_frozen_fraction,
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
    if isinstance(config.surface_scheme, (TwoLeafCanopyConfig, CLMMLCanopyConfig)):
        # For both canopy schemes the coupler must receive the canopy-consistent
        # LW_up (computed inside the canopy RT, not recomputed from post-step soil T)
        # and the aerodynamic canopy-air temperature (not the soil skin T).
        # CLMMLCanopyConfig populates T_canopy_air in _extract_surface_fluxes;
        # fall back to T_surface if absent (should not happen post-Iter 9).
        response_T_sfc = (surface_out.T_canopy_air
                          if surface_out.T_canopy_air is not None
                          else surface_out.T_surface)
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
        # Latent heat to the atmosphere = the evaporative/sublimation demand PLUS the
        # blowing-snow sublimation (gap 5): its L_s was charged to the surface energy
        # budget, so it must reach the atmosphere as latent heat (0 when bands off).
        lhflx=lhflx_actual + blow_subl * constants.L_s,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=response_lw_up,
        u_ocean_sfc=jnp.zeros(ncol),
        v_ocean_sfc=jnp.zeros(ncol),
        co2_flux=co2_flux,
        # Freshwater leaving the column to the ocean = total Richards runoff
        # (surface + subsurface) [kg/m²/s]; closes the coupler water budget.
        freshwater_flux=runoff_surface_total + richards_out.runoff_subsurface,
        # Land does not extract heat or exert stress on the ocean.
        ocean_heat_extraction=jnp.zeros(ncol),
        ocean_stress_x=jnp.zeros(ncol),
        ocean_stress_y=jnp.zeros(ncol),
        # Phase-aware moisture mass flux (up): pack sublimation (``sublim_actual``,
        # L_s) + soil / plant-water evaporation (``soil_evap``, L_v) + the blowing-
        # snow sublimated SWE that left as vapor (gap 5) — so the vapor mass
        # balances the reported latent heat across BOTH phases.
        surface_mass_flux=sublim_actual + soil_evap + blow_subl,
        # Land tile does not exchange salt with the ocean directly.
        salt_flux=jnp.zeros(ncol),
    )

    # The hold must be ATOMIC over everything this step advanced. The carbon
    # pools are stepped above from the SAME rejected GPP and surface
    # temperature, so a column held in the soil but advanced in carbon would
    # carry that inconsistency into the restart file.
    new_state, response, carbon_state_new, _held_mask, _n_held = (
        _hold_unsolved_columns(state, new_state, response, surface_out,
                               forcing, config, ncol,
                               carbon_old=carbon_state,
                               carbon_new=carbon_state_new))
    surface_out = surface_out._replace(held=_held_mask, n_held=_n_held)

    return new_state, response, carbon_state_new, surface_out


# ---------------------------------------------------------------------------
# Per-column containment: an unsolved column must not be able to kill the model
# ---------------------------------------------------------------------------

def _hold_unsolved_columns(state, new_state, response, surface_out, forcing,
                           config, ncol, carbon_old=None, carbon_new=None):
    """Freeze any column the land model failed to solve, and say so.

    Two things make a column's new state untrustworthy:

    * the surface scheme's iterative closure did not reach a root
      (``SurfaceFluxOutput.converged is False``), so its fluxes are a stopped
      iterate rather than a solution of the surface energy balance; or
    * some leaf of the new state or of the tile response came back non-finite.

    Either way the column is HELD: its state (soil, snow, ponding, carbon)
    reverts to the start of the step and its tile response reports no turbulent
    exchange and no runoff, with a surface temperature equal to the (finite)
    previous top-soil temperature and a surface humidity equal to the air's.

    That does NOT mean the atmosphere exchanges nothing with a held column.
    On the coupled path the land tile hands back only the skin temperature and
    albedo and the atmosphere recomputes its own sensible and latent fluxes
    from them, so it keeps exchanging with the held surface — it simply does so
    against a finite, frozen surface instead of a diverging one.  Making the
    atmosphere's own flux law honour the hold needs the mask threaded to it and
    is NOT done here.

    Why this exists.  Before it, ONE column of 2562 that went non-finite
    reached the atmosphere through the land skin temperature and, via the
    dynamical core's global mass fixer, made every column of the model
    non-finite within a single step: a 5-day AMIP run died 7 hours in with a
    NaN in every field. Containing the damage to the column that produced it
    turns a dead run into a reported defect.

    This is CONTAINMENT, NOT PHYSICS. A held column conserves neither energy
    nor water over the step it is held, so it must never be absorbed silently —
    a run whose land is quietly frozen somewhere is worse than one that stops.
    Returning the per-column mask and the count is how that is made visible:
    this function runs inside the jitted step, where a host print is not
    available on a GPU-only runtime (``jax.debug.print`` raises there), so the
    caller is responsible for surfacing them.  ``SurfaceFluxOutput.held`` and
    ``SurfaceFluxOutput.n_held`` carry them out.

    Returns
    -------
    held_state, held_response, held_carbon, held_mask (ncol,) bool,
    n_held () int32
    """
    def _is_float_leaf(leaf):
        # NOT ``dtype.kind in "fc"``: bfloat16 is an extension dtype whose kind
        # is "V", so a kind test silently skips it and a bfloat16 NaN would go
        # unheld (reproduced by review).  ``jnp.issubdtype(..., jnp.inexact)``
        # covers every float and complex dtype JAX supports.
        dtype = getattr(leaf, "dtype", None)
        return dtype is not None and jnp.issubdtype(dtype, jnp.inexact)

    def _col_bad(leaf):
        """Per-column non-finiteness of one (ncol, ...) array leaf."""
        arr = jnp.asarray(leaf)
        if arr.ndim == 0 or arr.shape[0] != ncol:
            return jnp.zeros(ncol, dtype=bool)
        flat = arr.reshape(ncol, -1)
        return jnp.any(~jnp.isfinite(flat), axis=-1)

    def _any_bad(tree):
        # Over tree LEAVES, so a nested carrier (the CLM-ML canopy state) is
        # inspected too, not just the top-level fields.
        acc = jnp.zeros(ncol, dtype=bool)
        for leaf in jax.tree.leaves(tree):
            if _is_float_leaf(leaf):
                acc = acc | _col_bad(leaf)
        return acc

    bad = _any_bad(new_state) | _any_bad(response)
    if surface_out.converged is not None:
        bad = bad | ~jnp.asarray(surface_out.converged).reshape(-1).astype(bool)

    n_held = jnp.sum(bad.astype(jnp.int32))

    def _hold_leaf(new_leaf, old_leaf):
        if not _is_float_leaf(new_leaf) or old_leaf is None:
            return new_leaf
        if new_leaf.ndim == 0 or new_leaf.shape[0] != ncol:
            return new_leaf
        old_arr = jnp.asarray(old_leaf)
        if old_arr.shape != new_leaf.shape:
            return new_leaf
        mask = bad.reshape((ncol,) + (1,) * (new_leaf.ndim - 1))
        return jnp.where(mask, old_arr, new_leaf)

    def _hold_field(new_field, old_field):
        # Per FIELD rather than one tree.map over the whole state: a field that
        # is None on one side and an array on the other (an optional store
        # switched on mid-run) makes the two states different pytrees, which a
        # single tree.map cannot walk.  Inside a field we DO recurse, so a
        # nested carrier (the CLM-ML canopy state) is held as well.
        if new_field is None or old_field is None:
            return new_field
        if _is_float_leaf(new_field):
            return _hold_leaf(new_field, old_field)
        try:
            return jax.tree.map(_hold_leaf, new_field, old_field)
        except (ValueError, TypeError):
            # Structurally different this step (e.g. a carrier rebuilt from
            # scratch): nothing to revert to, so leave it. Loud rather than
            # silent — the count below still reports the column as held.
            return new_field

    held_state = new_state._replace(
        **{name: _hold_field(getattr(new_state, name), getattr(state, name))
           for name in new_state._fields})

    # Inert-surface response for a held column.
    T_prev = state.T_soil[:, 0]
    eps = jnp.full(ncol, config.emissivity_land)
    inert = dict(
        T_sfc=T_prev, T_rad=T_prev,
        albedo=jnp.full(ncol, config.albedo_land),
        emissivity=eps,
        z0=jnp.full(ncol, config.z0_land),
        # Equal to the air, so any consumer that forms a humidity GRADIENT from
        # this field gets zero.  Note the coupled atmosphere is not such a
        # consumer — it recomputes surface humidity from saturation at the skin
        # temperature and ignores this field (see the note in the docstring).
        q_surface=jnp.asarray(forcing.q_lowest).reshape(-1),
        lw_up=eps * constants.sigma_sb * T_prev ** 4,
    )

    def _hold_response(name, leaf):
        if leaf is None or not _is_float_leaf(leaf):
            return leaf
        arr = jnp.asarray(leaf)
        if arr.ndim == 0 or arr.shape[0] != ncol:
            return leaf
        fallback = inert.get(name, jnp.zeros_like(arr))
        return jnp.where(bad, jnp.broadcast_to(fallback, arr.shape), arr)

    held_response = response._replace(
        **{name: _hold_response(name, getattr(response, name))
           for name in response._fields})

    held_carbon = carbon_new
    if carbon_old is not None and carbon_new is not None:
        try:
            held_carbon = jax.tree.map(_hold_leaf, carbon_new, carbon_old)
        except (ValueError, TypeError):
            # The carbon carrier changed structure this step, so there is no
            # matching value to revert to. Leave it rather than guess; the
            # column is still reported held by the count below.
            held_carbon = carbon_new

    return held_state, held_response, held_carbon, bad, n_held


def soil_ice_log_impedance(
    T_soil: jnp.ndarray,
    theta: jnp.ndarray,
    config: MultiLayerLandConfig,
) -> jnp.ndarray:
    """ln of the CLM5 frozen-soil conductivity multiplier, per layer.

    ``10**(-e * icefrac)`` with ``icefrac = min(1, vol_ice / theta_sat)`` and
    ``vol_ice`` the ice (``theta - theta_liq(T)``, the thermal solve's own split)
    at ice density, as CLM5 SoilHydrologyMod / IceImpedance.  ``e`` =
    ``config.richards.ice_impedance_exponent``.  Zero-porosity cells get no ice.
    """
    liq, _ = liquid_water_content(T_soil, theta, config.thermal)
    theta_sat = jnp.broadcast_to(config.hydraulics.theta_sat, theta.shape)
    ice_vol = (theta - liq) * (constants.rho_water / constants.rho_ice)
    icefrac = jnp.where(
        theta_sat > 0.0,
        jnp.clip(ice_vol / jnp.maximum(theta_sat, 1e-6), 0.0, 1.0), 0.0)
    return -config.richards.ice_impedance_exponent * _LN10 * icefrac


def init_multilayer_land_state(
    ncol: int,
    config: MultiLayerLandConfig,
    T_init: float = 280.0,  # coeff-ok: initial condition (default land skin/soil temperature)
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

    # Banded SWE/age/ice only when the elevation-band scheme is configured; ``None``
    # keeps the legacy pytree (single cell-mean snowpack, no ice reservoir).  Match the
    # soil dtype so the banded fields do not start float32 under a float64 state.
    if config.elev_bands is not None:
        _nb = config.elev_bands.band_dz.shape[-1]
        _bdt = T_soil.dtype
        snow_bands = jnp.zeros((ncol, _nb), dtype=_bdt)
        snow_age_bands = jnp.zeros((ncol, _nb), dtype=_bdt)
        ice_bands = jnp.zeros((ncol, _nb), dtype=_bdt)
    else:
        snow_bands = snow_age_bands = ice_bands = None

    # Initialize canopy state for CLM-ML-JAX scheme.
    # On cold start the mlcanopy_type is not allocated here — the interface
    # allocates it lazily on the first call to compute_clm_ml_canopy_fluxes
    # when canopy_state is None.
    if isinstance(config.surface_scheme, CLMMLCanopyConfig):
        from legoesm.land.canopy.state import CanopyState
        canopy_state = CanopyState(mlcanopy=None)
    else:
        canopy_state = None

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
        snow_bands=snow_bands,
        snow_age_bands=snow_age_bands,
        ice_bands=ice_bands,
        canopy_state=canopy_state,
        # Dry canopy at start; carried only when interception is configured.
        W_canopy=(jnp.zeros(ncol) if config.interception is not None else None),
        # Canopy warm-start cache, allocated (as "no converged solution yet")
        # only for the scheme that has a Newton closure to seed.  It must be
        # ALLOCATED here rather than left None and filled on the first step: a
        # ``lax.scan`` carry cannot change pytree structure mid-scan.  All-NaN
        # is the honest sentinel — the canopy tests it with ``isfinite`` and
        # cold-starts every column on the first step, exactly as before.
        canopy_x=(jnp.full((ncol, 6), jnp.nan, dtype=T_soil.dtype)
                  if (isinstance(config.surface_scheme, TwoLeafCanopyConfig)
                      and not isinstance(config.surface_scheme,
                                         CLMMLCanopyConfig)) else None),
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


# --- Packed land columns (2026-08-25 step-cost profile) ---------------------
# The coupled driver solves the tile only on the f_land > 0 columns and keeps
# the full-grid state between calls; these three helpers are the whole
# contract, factored here so the gather/scatter convention is testable
# without a driver.  A leaf participates iff its LEADING axis is the full
# column count — every other leaf (scalars, per-PFT tables) passes through.

def gather_land_columns(tree, idx, ncol_full: int):
    """Gather leading-``ncol_full`` leaves of ``tree`` onto columns ``idx``."""
    return jax.tree_util.tree_map(
        lambda x: (x[idx]
                   if (hasattr(x, "shape") and getattr(x, "ndim", 0) >= 1
                       and x.shape[0] == ncol_full)
                   else x),
        tree)


def scatter_land_columns(full_tree, packed_tree, idx, ncol_full: int):
    """Write packed leaves back into the full-grid tree at columns ``idx``.

    Non-column leaves take the PACKED (advanced) value — they were passed
    through the solve unpacked, so the solved value is the current one.
    """
    return jax.tree_util.tree_map(
        lambda full, packed: (
            full.at[idx].set(packed)
            if (hasattr(full, "shape") and getattr(full, "ndim", 0) >= 1
                and full.shape[0] == ncol_full)
            else packed),
        full_tree, packed_tree)


def scatter_cells(v, idx, ncol_full: int):
    """Packed per-column vector -> full-grid cells, zero fill (never NaN:
    the consumers multiply by f_land, and 0 * NaN would contaminate them)."""
    return jnp.zeros((ncol_full,), v.dtype).at[idx].set(v.reshape(-1))
