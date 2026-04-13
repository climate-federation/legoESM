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

import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice
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
from legoesm.land.surface_scheme import (
    SimpleSEBConfig,
    TwoLeafCanopyConfig,
    compute_simple_seb_fluxes,
    compute_two_leaf_canopy_fluxes,
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
    beta_root = jnp.clip(
        (theta - theta_wp_c[:, None])
        / (theta_fc_c[:, None] - theta_wp_c[:, None] + 1e-10),
        0.0, 1.0)
    w_frac_rz = jnp.clip(jnp.sum(root_frac * beta_root, axis=-1), 0.0, 1.0)

    beta_soil = config.beta_min + (1.0 - config.beta_min) * w_frac_rz

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

        surface_out = compute_two_leaf_canopy_fluxes(
            T_soil_top=T_surface,
            forcing=forcing,
            canopy_config=config.surface_scheme,
            land_config=config,
            canopy_params=lp,
            root_frac=root_frac,
            beta_root=beta_root,
            w_frac_rz=w_frac_rz,
            wind_speed=wind_speed,
            wind_dir_x=wind_dir_x,
            wind_dir_y=wind_dir_y,
            soil_thermal_fn=_soil_thermal_cb,
            dt=dt,
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

    # --- Snow budget (energy-limited melt) ---
    has_snow = snow > 1e-6
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
    max_soil_evap = jnp.maximum(
        extractable_water / dt + precip_rain + melt_rate, 0.0)
    soil_evap = jnp.minimum(soil_evap_demand, max_soil_evap)

    evap_rate = jnp.where(has_snow, sublim_actual, soil_evap)
    evap_excess_energy = (evap_rate_demand - evap_rate) * L_eff
    lhflx_actual = evap_rate * L_eff

    # --- Root water uptake partition ---
    f_veg = jnp.clip(w_frac_rz, 0.0, 1.0)
    evap_bare = evap_rate * (1.0 - f_veg)
    evap_transp = evap_rate * f_veg
    flux_top = (precip_rain + melt_rate - evap_bare) / rho_w

    E_pot_transp = jnp.maximum(evap_transp, 0.0) / rho_w
    weight = root_frac * beta_root  # both are (ncol, n_layers)
    weight_sum = jnp.sum(weight, axis=-1, keepdims=True)
    weight_norm = weight / jnp.maximum(weight_sum, 1e-20)
    sink = weight_norm * E_pot_transp[:, None] / dz[None, :]

    # --- Richards equation ---
    richards_out = solve_richards(
        psi, theta, grid,
        config.hydraulics, config.richards,
        flux_top, sink, dt,
    )
    theta_corrected = jnp.clip(
        richards_out.theta_new,
        config.hydraulics.theta_r, config.hydraulics.theta_sat,
    )
    richards_out = richards_out._replace(theta_new=theta_corrected)

    # --- Soil thermal diffusion (final, with converged G) ---
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
    theta_new = richards_out.theta_new
    beta_root_new = jnp.clip(
        (theta_new - theta_wp_c[:, None])
        / (theta_fc_c[:, None] - theta_wp_c[:, None] + 1e-10),
        0.0, 1.0)
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
    response = TileResponse(
        T_surface=T_surface_new,
        albedo=alpha_new,
        emissivity=surface_out.emissivity,
        z0=surface_out.z0,
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

    return new_state, response, carbon_state_new, surface_out


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
