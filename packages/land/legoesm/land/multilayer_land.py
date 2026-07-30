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

from typing import NamedTuple

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
from legoesm.land.snow_column import (
    apply_sublimation,
    pack_top_temperature,
    snow_base_interface_conductance,
    step_snow_column,
    total_water,
)
from legoesm.land.state import MultiLayerLandState

# Snow thermal scheme dispatch (validated on the static config value at the shared
# impl entry — see MultiLayerLandConfig.snow_scheme).  Grow this set as schemes are
# wired; an unknown value must raise, never silently fall through to a default.
_SUPPORTED_SNOW_SCHEMES = ("single", "multilayer")

# Minimum soil-top half-interface distance [m] for the snow<->soil interface
# conductance (mirrors snow_column._DZ_HALF_MIN): bounds k/(dz/2) for a vanishing
# top-layer thickness so g_iface stays finite.
_SOIL_TOP_DZ_HALF_MIN = 1e-4
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.stomata_utils import compute_effective_beta
from legoesm.land.richards import solve_richards
from legoesm.land.soil_thermal import compute_thermal_conductivity, solve_soil_thermal
from legoesm.land.canopy.config import CLMMLCanopyConfig
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


class LandStepDiagnostics(NamedTuple):
    """Per-step budget terms that close the land energy and water budgets.

    The taped ``Rnet``/``shflx``/``lhflx`` alone CANNOT close the surface energy
    budget, because ``SurfaceFluxOutput.G_soil`` is *defined* as the residual
    ``Rnet - SH - LH`` and the snow phase change is then removed from it:

        Rnet - SH - LH  ==  g_soil + melt_energy

    so a diagnosed "residual" of ``Rnet - SH - LH`` is the ground heat flux
    BEFORE melt, not an energy leak.  Taping ``g_soil`` and ``melt_energy``
    separately makes the identity checkable instead of inferable.

    Likewise the water budget needs the snow-phase fluxes: the ``ET`` tape is
    ``lhflx / L_v``, which is NOT the mass flux over snow (where the latent heat
    carries ``L_s``), and melt moves mass from the pack into the soil without
    appearing in any existing tape variable.

    All fluxes are per unit GROUND area.  Mass fluxes are rates [kg m-2 s-1],
    positive in the direction named; energies [W m-2].
    """
    g_soil: jnp.ndarray        # heat flux INTO the soil column, post-phase-change
    melt_energy: jnp.ndarray   # energy consumed by melt (+) / released by refreeze (-)
    snowmelt: jnp.ndarray      # snow + ablation-ice melt leaving the pack as liquid
    refreeze: jnp.ndarray      # rain-on-snow refrozen in the pack (mass, +)
    sublimation: jnp.ndarray   # pack -> vapour (+ = mass leaving; < 0 = frost deposition)


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
    # Audit #6 / Iter-65: floor the (theta_fc - theta_wp) range at 1e-3 m^3/m^3
    # (~1 % of theta_sat) so a pathological PFT row (theta_fc ~ theta_wp) cannot
    # explode beta_root through a ~0 denominator.
    _denom = jnp.maximum(
        theta_fc_c[:, None] - theta_wp_c[:, None], 1e-3,  # coeff-ok: floor (theta_fc - theta_wp) range to avoid /~0 in beta_root
    )
    beta_root = jnp.clip((theta - theta_wp_c[:, None]) / _denom, 0.0, 1.0)
    w_frac_rz = jnp.clip(jnp.sum(root_frac * beta_root, axis=-1), 0.0, 1.0)
    beta_soil = beta_min + (1.0 - beta_min) * w_frac_rz
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
    theta_wp   = _get(land_params, "theta_wp", config.theta_wp)
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
):
    """Like :func:`step_multilayer_land` but also returns the ``SurfaceFluxOutput``.

    The 4th element is the surface scheme's ``SurfaceFluxOutput`` with all
    scheme-specific diagnostic fields populated (``Tf_Sun``, ``Tf_Sh``,
    ``gs_Sun``, ``n_iters``, ``f_veg`` for the canopy scheme; the common
    flux / radiation / state fields for both schemes).  Intended for offline
    diagnostic runs — no performance cost beyond the extra pytree allocation.
    For the energy/water budget-closure terms use
    :func:`step_multilayer_land_with_budget`, which returns those as a 5th element.
    """
    return _step_multilayer_land_impl(
        state, forcing, config, U_min, dt,
        lat=lat, carbon_state=carbon_state, doy=doy, land_params=land_params)[:4]


def step_multilayer_land_with_budget(
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
    """Like :func:`step_multilayer_land_with_diagnostics` plus a 5th element:
    a :class:`LandStepDiagnostics` with the ENERGY and WATER budget-closure terms.

    Separate from the 4-tuple variant on purpose — that signature has many call
    sites (EC-site driver, canopy validators, tests) which have no use for the
    budget terms, and widening it would churn all of them for no benefit.
    """
    return _step_multilayer_land_impl(
        state, forcing, config, U_min, dt,
        lat=lat, carbon_state=carbon_state, doy=doy, land_params=land_params)


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
) -> tuple[MultiLayerLandState, TileResponse, CarbonState | None]:
    """Step the multi-layer land model forward by ``dt`` seconds.

    Dispatches between ``SimpleSEBConfig`` and ``TwoLeafCanopyConfig``
    surface schemes via ``isinstance(config.surface_scheme, ...)``.  The
    post-flux pipeline (snow, Richards, soil thermal, carbon, TileResponse)
    is shared between both branches.  Returns a 3-tuple; use
    :func:`step_multilayer_land_with_diagnostics` to also receive the raw
    ``SurfaceFluxOutput`` for diagnostic inspection.
    """
    new_state, response, carbon_new, _surface_out, _diags = _step_multilayer_land_impl(
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
    # Dispatch hardening: validate the snow scheme on the STATIC config value at fn
    # entry (never inside the traced body).  Unknown value -> hard error so a typo
    # cannot silently run the wrong snow physics.
    if config.snow_scheme not in _SUPPORTED_SNOW_SCHEMES:
        raise ValueError(
            f"Unknown snow_scheme {config.snow_scheme!r}; expected one of "
            f"{_SUPPORTED_SNOW_SCHEMES}.")
    if config.snow_scheme == "multilayer":
        # Phase 2b Stage 3: the prognostic multi-layer snow column is coupled below
        # (canopy -> column -> soil sequential operator split).  Scoped to the
        # two-leaf canopy surface scheme (the LMIP + flux-site config); the SEB /
        # CLM-ML coupling is a later extension.  Fail loudly rather than silently
        # run the single-node budget under a 'multilayer' label.
        # See docs/land/phase2b_snow_thermal_plan.md.
        if not isinstance(config.surface_scheme, TwoLeafCanopyConfig):
            raise ValueError(
                "snow_scheme='multilayer' is currently supported only with the "
                "TwoLeafCanopyConfig surface scheme (the LMIP / flux-site config); "
                f"got {type(config.surface_scheme).__name__}. Use snow_scheme='single' "
                "with other surface schemes.")
        if config.elev_bands is not None:
            raise ValueError(
                "snow_scheme='multilayer' and elev_bands (sub-grid snow bands) are "
                "mutually exclusive snow representations; enable only one.")
        if state.snow_column is None:
            raise ValueError(
                "snow_scheme='multilayer' requires state.snow_column; initialise with "
                "init_multilayer_land_state(config=...) so the prognostic column exists.")
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
    theta_wp = _get(lp, "theta_wp", config.theta_wp)
    theta_fc = _get(lp, "theta_fc", config.theta_fc)

    # Start-of-step skin temperature = top soil layer.
    T_surface = T_soil[:, 0]
    ncol = T_surface.shape[0]

    # --- Multi-layer snow column: lagged (start-of-step) diagnostics (Stage 3) ---
    # When snow_scheme=="multilayer" the prognostic column buffers the surface energy
    # flux from the soil.  Compute the lagged pack SWE / top-T here so the surface
    # scheme's ground boundary is the PACK-TOP temperature where a pack exists (the
    # skin the atmosphere actually exchanges with under snow), and the cell-mean
    # ``snow`` handed to the albedo/latent blocks is the column total SWE.  Sequential
    # operator split: these lagged values also set G_bottom below (see the snow phase).
    _use_snow_column = config.snow_scheme == "multilayer"
    if _use_snow_column:
        _pack_swe_lagged = total_water(state.snow_column)              # (ncol,)
        # THERMALLY-ACTIVE pack (zero-layer-snow threshold): only a pack thick enough
        # to carry its own surface energy budget insulates the soil.  A thin dusting
        # accumulates + brightens albedo but the surface flux still reaches the soil
        # (routing a full Q_top into a ~0-heat-capacity layer would melt it in one
        # step).  See SnowColumnConfig.thermal_active_swe.
        _has_pack_lagged = _pack_swe_lagged > config.snow_column.thermal_active_swe
        _pack_top_T_lagged = pack_top_temperature(state.snow_column)   # (ncol,)
        # Canopy ground boundary = pack-top T where thermally packed, else soil top.
        T_surface = jnp.where(_has_pack_lagged, _pack_top_T_lagged, T_surface)
        # Cell-mean SWE the snow/albedo/latent blocks read = the column total (the
        # column is authoritative for SWE, incl. sub-threshold snow for albedo).
        snow = _pack_swe_lagged

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
    theta_wp_c   = _to_ncol(theta_wp)
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

        surface_out, canopy_state_new = compute_clm_ml_canopy_fluxes(
            T_soil_top=T_surface,
            forcing=forcing,
            canopy_config=config.surface_scheme,
            land_config=config,
            land_params=lp,
            w_frac_rz=w_frac_rz,
            wind_speed=wind_speed,
            canopy_state=state.canopy_state,
            dt=dt,
            T_soil=T_soil,
            psi_soil=psi,
            theta_soil=theta,
            lat=lat,
            doy=doy,
            lai_override=LAI_override,
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

    # Snow-free base albedo that the snow feedback (and banded radiation) blends
    # on top of.  The canopy schemes (TwoLeafCanopy / CLM-ML) diagnose a per-cell
    # snow-free surface albedo from their OWN shortwave radiative transfer — the
    # soil-colour ``ALB_VIS``/``ALB_NIR`` (moisture-darkened) blended with the
    # vegetation — and return it in ``surface_out.albedo``.  The SEB path carries
    # the per-cell veg/soil blend (``lp.albedo_veg`` + dry-soil brightening) in
    # ``albedo_land``.  ``CanopyLandParams`` has NO ``albedo_veg`` field, so
    # ``_get(lp, "albedo_veg", config.albedo_land)`` above silently collapses the
    # canopy base to the scalar ``config.albedo_land``; without this split the
    # reported/coupled canopy albedo degenerates to a uniform base + snow bands
    # and drops all soil-colour + vegetation structure (while the canopy energy
    # balance still used the real per-cell albedo — an inconsistency too).  The
    # canopy RT already accounts for soil reflectance, so no dry-soil brightening
    # is re-applied to ``surface_out.albedo``.
    _is_canopy = isinstance(
        config.surface_scheme, (TwoLeafCanopyConfig, CLMMLCanopyConfig))
    snowfree_base = surface_out.albedo if _is_canopy else albedo_land

    # --- Banded surface radiation (gaps 1,2): override the cell-mean radiation in
    # G_surface with the area-weighted per-band balance (elevation-lapsed SW/LW +
    # per-band albedo + per-band skin T), keeping the (cell-mean) turbulent fluxes
    # from the surface scheme.  ``band_rad.Rn_bands`` drives per-band melt below.
    if bands is not None:
        _cover_fn = lambda s: snow_cover_fraction(s, config.land_albedo)
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
            # Snow-free base = ``snowfree_base`` (per-cell canopy RT albedo for the
            # canopy schemes, else the SEB per-cell CLM map / lp.albedo_veg + dry-soil
            # brightening scalar carrying the trainable pft_alb) so the base flows to
            # the gradient — NOT the latitude-band veg albedo, which would zero pft_alb.
            _base = jnp.broadcast_to(snowfree_base, T_surface.shape)
            alpha_bands = band_albedo(
                _bands_eff, state.snow_age_bands, _base, _cover_fn, _alb_fn,
                ice_bands=ice_bands_in, cfg=bands)
        else:
            alpha_bands = jnp.broadcast_to(
                jnp.reshape(snowfree_base, (-1, 1)), (ncol, bands.band_dz.shape[-1]))
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
    elif _use_snow_column:
        # --- Multi-layer prognostic snow column (Stage 3, sequential split) ---
        # Q_top = the surface net flux (G_surface, computed by the canopy against the
        # lagged pack-top T) enters the pack TOP; a prescribed conductive flux
        # G_bottom = g_iface*(T_pack_base - T_soil_top) (positive DOWNWARD into soil)
        # leaves the base and is handed UNCHANGED to the soil top BC below, so the
        # snow<->soil seam conserves energy.  Fusion is handled INSIDE the column
        # (enthalpy method) -> no melt_energy term here.  Empty-pack columns bypass
        # the column (Q_top=0 into the inert pack; the soil gets the full G_surface),
        # so a snow-free cell is bit-identical to snow_scheme="single".
        _sc = state.snow_column
        # Interface conductance g_iface = harmonic mean of the pack-base and soil-top
        # half-conductances (both >= 0).  Lagged (start-of-step) T -> explicit split.
        _g_base = snow_base_interface_conductance(_sc, config.snow_column)   # W/m2/K
        _k_soil_top = compute_thermal_conductivity(
            theta, config.hydraulics, config.thermal)[:, 0]                  # W/m/K
        _g_soil_top = _k_soil_top / jnp.maximum(
            0.5 * grid.dz[0], _SOIL_TOP_DZ_HALF_MIN)                         # W/m2/K
        _g_iface = jnp.where(
            _has_pack_lagged,
            2.0 * _g_base * _g_soil_top / jnp.maximum(_g_base + _g_soil_top, 1e-30),
            0.0)
        G_bottom = _g_iface * (_sc.T[..., -1] - T_soil[:, 0])   # +downward into soil
        # Only packed columns take Q_top / lose G_bottom; empty ones just accumulate
        # fresh snow (at T_air) and stay inert this step.
        _Q_top_col = jnp.where(_has_pack_lagged, G_surface, 0.0)
        _G_bottom_col = jnp.where(_has_pack_lagged, G_bottom, 0.0)
        _sc_new, drainage, drainage_heat = step_snow_column(
            _sc, precip_snow_eff, forcing.T_lowest,
            _Q_top_col, _G_bottom_col, dt, config.snow_column)
        # Soil top BC flux: G_bottom + the drained meltwater's enthalpy where packed,
        # else the full surface flux (empty-pack bypass).
        G_surface = jnp.where(
            _has_pack_lagged, G_bottom + drainage_heat / dt, G_surface)
        snow_new = total_water(_sc_new)              # pre-sublimation cell SWE [kg/m2]
        snow_melt = drainage                         # meltwater leaving the base [kg/m2]
        # Column has no separate age clock: reset on fresh snowfall, else age; zero
        # once the pack is empty (mirrors update_snow's albedo-age semantics).
        _fresh = precip_snow_eff * dt > 1e-6
        snow_age_new = jnp.where(_fresh, 0.0, snow_age + dt)
        snow_age_new = jnp.where(
            snow_new > config.snow_column.min_pack_swe, snow_age_new, 0.0)
        snow_bands_new = state.snow_bands
        snow_age_bands_new = state.snow_age_bands
        ice_bands_new = state.ice_bands
        ice_melt = jnp.zeros_like(snow_new)
        refreeze = jnp.zeros_like(snow_new)
        blow_subl = jnp.zeros_like(snow_new)
        cap_runoff = jnp.zeros_like(snow_new)
        _soil_surface_conductance = jnp.where(
            _has_pack_lagged, _g_iface, surface_out.surface_conductance)
    else:
        snow_new, snow_age_new, snow_melt = update_snow(
            snow, snow_age, T_surface, precip_snow_eff, dt,
            Q_net=G_surface,
            snow_melt_rate=config.snow_melt_rate,
            T_snow_melt=config.T_snow_melt,
        )
        snow_bands_new = state.snow_bands
        snow_age_bands_new = state.snow_age_bands
        ice_bands_new = state.ice_bands
        ice_melt = jnp.zeros_like(snow_new)
        refreeze = jnp.zeros_like(snow_new)
        blow_subl = jnp.zeros_like(snow_new)
        cap_runoff = jnp.zeros_like(snow_new)
    # Defined on EVERY branch: the multilayer snow column handles fusion inside
    # the pack (enthalpy method), so its explicit melt_energy term is zero and the
    # identity Rnet-SH-LH == g_soil + melt_energy still holds there.
    melt_energy = jnp.zeros_like(G_surface)
    if not _use_snow_column:
        # Energy into the surface budget: seasonal-snow + ablation-ice melt CONSUME
        # L_f; rain-on-snow refreezing (gap 6) RELEASES L_f; blowing-snow sublimation
        # (gap 5) consumes L_s.  (Frozen glacier discharge leaves as ice — no fusion.)
        # Multilayer: fusion is handled INSIDE the column, so this is skipped.
        melt_energy = ((snow_melt + ice_melt - refreeze) * constants.L_f / dt
                       + blow_subl * constants.L_s)
        G_surface = G_surface - melt_energy
        _soil_surface_conductance = surface_out.surface_conductance

    # --- Latent mass partition (sublimation vs soil evap, water-limited) ---
    rho_w = constants.rho_water
    L_eff = jnp.where(has_snow, constants.L_s, constants.L_v)
    evap_rate_demand = lhflx / L_eff

    snow_after_melt = snow_new
    max_sublim = jnp.maximum(snow_after_melt / dt, 0.0)
    sublim_demand = jnp.where(has_snow, evap_rate_demand, 0.0)
    sublim_actual = jnp.minimum(sublim_demand, max_sublim)
    sublim_actual = jnp.where(sublim_demand < 0.0, sublim_demand, sublim_actual)
    if _use_snow_column:
        # Remove the sublimated ICE mass (+ its sensible enthalpy) from the pack; the
        # L_s ENERGY is already booked in Q_top (=G_surface) as -lhflx over snow, so
        # mass-removal here closes the coupled seam.  Deposition (sublim<0) adds frost.
        _sc_new, _ = apply_sublimation(
            _sc_new, sublim_actual * dt, config.snow_column)
        snow_new = total_water(_sc_new)
    else:
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
    soil_evap_demand = jnp.where(has_snow, 0.0, evap_rate_demand)
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
    # Multilayer snow (packed cells): the soil top is driven by G_bottom =
    # g_iface*(T_pack_base - T_soil_top), so the implicit Robin term is g_iface itself
    # (= -dG_bottom/dT_soil_top >= 0) rather than the canopy's pack-top conductance
    # (``_soil_surface_conductance`` above already blends the two by pack presence).
    G_surface = G_surface + evap_excess_energy
    T_soil_new = solve_soil_thermal(
        T_soil, richards_out.theta_new, grid,
        config.hydraulics, config.thermal,
        G_surface, dt,
        surface_conductance=_soil_surface_conductance,
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
        # Snow-column carry: the stepped column for snow_scheme="multilayer"
        # (Stage 3), else None (single-scheme passthrough).  ``_sc_new`` is the
        # post-thermal, post-sublimation column; ``snow_depth`` above == its total
        # SWE (set via ``snow_new``) for albedo/diagnostics/restart consistency.
        snow_column=(_sc_new if _use_snow_column else state.snow_column),
    )

    # --- Post-step surface state for coupler ---
    T_surface_new = T_soil_new[:, 0]
    # Re-brighten the snow-free base with the END-of-step top-layer moisture so the albedo
    # handed to the coupler (drives the next radiation step) is consistent with the updated
    # T_surface_new / snow_new state — the pre-step ``albedo_land`` used start-of-step theta.
    albedo_land_post = _albedo_base + dry_soil_brightening(
        richards_out.theta_new[:, 0], config.land_albedo)
    # Canopy schemes' snow-free base is the per-cell RT albedo (single per-step
    # value; no start/end-of-step brightening split — the RT already used the
    # step's soil moisture).  See ``snowfree_base`` above.
    snowfree_base_post = surface_out.albedo if _is_canopy else albedo_land_post

    if bands is not None:
        # Post-step banded albedo + up-welling LW for the atmosphere: the SAME
        # flux-weighted band radiation as the pre-step (gap 1), so the coupler sees a
        # consistent albedo (alpha_eff) and banded LW emission (not a cell-mean value).
        if config.snow_albedo_feedback and lat is not None:
            _base_new = jnp.broadcast_to(snowfree_base_post, T_surface_new.shape)
            _cz_new = forcing.cos_zenith[:, None]
            alpha_bands_new = band_albedo(
                snow_bands_new, snow_age_bands_new, _base_new,
                lambda s: snow_cover_fraction(s, config.land_albedo),
                lambda a: snow_albedo(a, config.land_albedo, cos_zenith=_cz_new),
                ice_bands=ice_bands_new, cfg=bands)
        else:
            alpha_bands_new = jnp.broadcast_to(
                jnp.reshape(snowfree_base_post, (-1, 1)), (ncol, bands.band_dz.shape[-1]))
        band_rad_new = band_net_radiation(
            T_surface_new, alpha_bands_new, forcing.sw_down, forcing.lw_down,
            emissivity, bands)
        alpha_new = band_rad_new.alpha_eff
        lw_up_new = band_rad_new.lw_up_agg
    else:
        if config.snow_albedo_feedback and lat is not None:
            # Per-cell snow-free base: the canopy RT albedo (soil-colour +
            # vegetation) for the canopy schemes, else the SEB per-cell CLM
            # PFT / trainable blend — snow feedback blends on top.
            alpha_new = compute_land_albedo(
                lat, snow_new, snow_age_new, config.land_albedo,
                base_albedo=jnp.broadcast_to(snowfree_base_post, T_surface_new.shape))
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
    if isinstance(config.surface_scheme, CLMMLCanopyConfig):
        # CLM-ML computes q_surface via the Philip (1957) soil-humidity formula
        # (rhg_soil * q_sat) internally and returns it in surface_out.q_surface.
        # Use it directly so the coupler sees the same humidity as CLM-ML used
        # for soil evaporation.  Override with q_sat_ice over snow (physically
        # correct; CLM-ML always runs with snl=0, so this path is dormant).
        q_sfc_new = jnp.where(has_snow_new, q_sat_sfc_new, surface_out.q_surface)
    else:
        # SimpleSEB / TwoLeafCanopy: beta·qsat with the updated moisture state
        # (``beta_new`` computed unconditionally above).
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
            _, gpp_override, _ = compute_effective_beta(
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
        # Phase-aware moisture mass flux (up): the evaporative/sublimation demand
        # (lhflx_actual / L_eff) PLUS the blowing-snow sublimated SWE that left as
        # vapor (gap 5) — so the vapor mass balances the reported latent heat.
        surface_mass_flux=lhflx_actual / L_eff + blow_subl,
        # Land tile does not exchange salt with the ocean directly.
        salt_flux=jnp.zeros(ncol),
    )

    step_diags = LandStepDiagnostics(
        # G_surface at this point is the flux handed to the soil thermal solve:
        # the surface residual minus the phase-change sink (single/banded), or the
        # snow-column base flux + drained meltwater enthalpy (column).
        g_soil=G_surface,
        melt_energy=melt_energy,
        # snow_melt / ice_melt / refreeze are MASSES over the step -> rates.
        snowmelt=(snow_melt + ice_melt) / dt,
        refreeze=refreeze / dt,
        # sublim_actual is already a rate; blow_subl (blowing-snow) likewise.
        sublimation=sublim_actual + blow_subl,
    )
    return new_state, response, carbon_state_new, surface_out, step_diags


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

    # Prognostic multi-layer snow column: an empty pack on cold start (all layers
    # zero SWE, T at freezing, fresh-snow density) when the multilayer scheme is
    # selected; ``None`` keeps the legacy single cell-mean pytree.  Match the soil
    # dtype so the column does not start float32 under a float64 state.
    if config.snow_scheme == "multilayer":
        from legoesm.land.snow_column import initial_snow_state
        snow_column = initial_snow_state((ncol,), config.snow_column,
                                         dtype=T_soil.dtype)
    else:
        snow_column = None

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
        snow_column=snow_column,
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
