"""Sea ice model: thermodynamics, dynamics, and multi-category ice.

Modes (controlled by ``SeaIceConfig``):
- **Slab** (``dynamics="none"``, ``n_categories=1``): Original thermodynamic
  slab with diagnostic free-drift velocity. Fully backward compatible.
- **Free drift** (``dynamics="free_drift"``): Free-drift velocity with
  optional tracer advection.
- **EVP** (``dynamics="evp"``): Elastic-Viscous-Plastic rheology with
  subcycled momentum solver (Hunke & Dukowicz 1997).

Multi-category ice (``n_categories > 1``) follows the CICE framework
with linear remapping (Lipscomb 2001) to maintain the ice thickness
distribution.

Thermodynamics:
    Surface energy balance determines T_ice.
    Conductive flux through ice: F_cond = k_ice * (T_freeze - T_ice) / (h + h_min)
    Growth/melt: dh/dt = (F_cond - F_ocean) / (rho_ice * L_f)
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio_ice
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import (
    SeaIceState,
    DynamicSeaIceState,
    dynamic_to_slab,
)
from legoesm.surface_albedo import ice_albedo as compute_ice_albedo


# ==============================================================================
# Main entry point
# ==============================================================================

def step_sea_ice(
    state,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
    grid=None,
):
    """Step the sea ice model forward by dt seconds.

    Dispatches to slab or dynamic model based on config.

    Parameters
    ----------
    state : SeaIceState or DynamicSeaIceState
        Current ice state.
    forcing : AtmToSurface
        Atmospheric forcing fields.
    ocean_sst : jnp.ndarray
        Ocean SST [K], shape (6, n, n).
    ocean_u, ocean_v : jnp.ndarray
        Ocean surface currents [m/s], shape (6, n, n).
    config : SeaIceConfig
        Sea ice model parameters.
    U_min : float
        Minimum wind speed floor [m/s].
    dt : float
        Time step [s].
    grid : CubedSphereGrid, optional
        Required when ``dynamics != "none"`` or ``transport != "none"``.

    Returns
    -------
    new_state : SeaIceState or DynamicSeaIceState
    response : TileResponse
    """
    if config.dynamics == "none" and config.n_categories == 1:
        # Original slab path — fully backward compatible
        if isinstance(state, DynamicSeaIceState):
            state = dynamic_to_slab(state)
        return _step_slab(state, forcing, ocean_sst, ocean_u, ocean_v,
                          config, U_min, dt)
    else:
        return _step_dynamic(state, forcing, ocean_sst, ocean_u, ocean_v,
                             config, U_min, dt, grid)


# ==============================================================================
# Slab thermodynamics (original implementation)
# ==============================================================================

def _step_slab(
    state: SeaIceState,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
) -> tuple[SeaIceState, TileResponse]:
    """Original slab sea ice model (backward compatible)."""
    h = state.h_ice.data
    T_ice = state.T_ice.data
    conc = state.concentration.data

    ice_mask = h > 0.0
    h_eff = jnp.maximum(h, config.h_ice_min)

    # ---------- Surface fluxes ----------
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
    )
    rho = forcing.rho_lowest
    q_sfc = saturation_mixing_ratio_ice(T_ice, forcing.p_surface)

    if config.bulk_scheme in ("most", "coare3", "large_yeager"):
        from legoesm.coupler.bulk_flux import compute_most_fluxes
        tau_x, tau_y, shflx, lhflx, _ = compute_most_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_ice, q_sfc, rho,
            z_ref=config.z_ref,
            z0_init=config.z0_ice,
            scheme=config.bulk_scheme,
            n_iter=config.bulk_n_iter,
        )
    else:
        tau_x = -rho * config.Cd_ice * wind_speed * forcing.u_lowest
        tau_y = -rho * config.Cd_ice * wind_speed * forcing.v_lowest
        shflx = rho * constants.c_pd * config.Ch_ice * wind_speed * (T_ice - forcing.T_lowest)
        lhflx = rho * constants.L_v * config.Ch_ice * wind_speed * (q_sfc - forcing.q_lowest)

    # ---------- Ice albedo ----------
    if config.temp_dependent_albedo:
        alpha_ice = compute_ice_albedo(T_ice, config.ice_albedo)
    else:
        alpha_ice = jnp.broadcast_to(jnp.array(config.albedo_ice), h.shape)

    # ---------- Radiation ----------
    sw_net = (1.0 - alpha_ice) * forcing.sw_down
    lw_down_abs = config.emissivity_ice * forcing.lw_down
    lw_up = config.emissivity_ice * constants.sigma_sb * T_ice ** 4
    lw_net = lw_down_abs - lw_up
    Q_sfc = sw_net + lw_net - shflx - lhflx

    # ---------- Conductive flux ----------
    F_cond = jnp.where(
        ice_mask,
        config.k_ice * (config.T_freeze_ocean - T_ice) / h_eff,
        0.0,
    )

    # ---------- Surface temperature evolution ----------
    skin_cap = config.rho_ice * config.c_ice * h_eff * 0.5
    dT_dt = (Q_sfc - F_cond) / skin_cap
    T_ice_trial = T_ice + dt * dT_dt
    T_ice_new = jnp.where(
        ice_mask,
        jnp.clip(T_ice_trial, config.T_ice_min, config.T_freeze_ocean),
        jnp.broadcast_to(jnp.array(config.T_freeze_ocean), T_ice.shape),
    )

    # ---------- Growth / melt ----------
    F_ocean = config.k_ice * jnp.maximum(ocean_sst - config.T_freeze_ocean, 0.0) / h_eff
    dh_dt_ice = (F_cond - F_ocean) / (config.rho_ice * config.L_f)
    freeze_flux_open = jnp.maximum(-Q_sfc, 0.0)
    dh_dt_open = freeze_flux_open / (config.rho_ice * config.L_f)
    dh_dt = jnp.where(ice_mask, dh_dt_ice, dh_dt_open)
    h_new = jnp.maximum(h + dt * dh_dt, 0.0)

    # ---------- Concentration ----------
    dconc_growth = jnp.maximum(dh_dt, 0.0) * (1.0 - conc) / config.h_new_ice
    dconc_melt = jnp.minimum(dh_dt, 0.0) * conc / (h_eff)
    conc_new = conc + dt * (dconc_growth + dconc_melt)
    conc_new = jnp.clip(conc_new, 0.0, 1.0)

    # ---------- Ice velocity (free drift, diagnostic) ----------
    u_ice = (config.drag_ocean * ocean_u
             + config.drag_atm * (config.rho_air_ref / config.rho_ice) * forcing.u_lowest)
    v_ice = (config.drag_ocean * ocean_v
             + config.drag_atm * (config.rho_air_ref / config.rho_ice) * forcing.v_lowest)

    new_state = SeaIceState(
        h_ice=state.h_ice.replace(data=h_new),
        T_ice=state.T_ice.replace(data=T_ice_new),
        concentration=state.concentration.replace(data=conc_new),
    )

    lw_up_new = config.emissivity_ice * constants.sigma_sb * T_ice_new ** 4
    q_sfc_new = saturation_mixing_ratio_ice(T_ice_new, forcing.p_surface)

    response = TileResponse(
        T_surface=T_ice_new,
        albedo=alpha_ice,
        emissivity=jnp.broadcast_to(jnp.array(config.emissivity_ice), h.shape),
        z0=jnp.broadcast_to(jnp.array(config.z0_ice), h.shape),
        q_surface=q_sfc_new,
        shflx=shflx,
        lhflx=lhflx,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up_new,
        u_ocean_sfc=u_ice,
        v_ocean_sfc=v_ice,
        co2_flux=jnp.zeros_like(h),
    )

    return new_state, response


# ==============================================================================
# Dynamic sea ice (EVP + optional multi-category)
# ==============================================================================

def _step_dynamic(
    state,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
    grid=None,
):
    """Dynamic sea ice step: dynamics → transport → thermodynamics → ITD remap.

    Parameters
    ----------
    state : DynamicSeaIceState
    grid : CubedSphereGrid, required for dynamics/transport
    """
    from legoesm.ice.dynamics import evp_solver, free_drift_velocity
    from legoesm.ice.transport import advect_ice_tracers
    from legoesm.ice.itd import aggregate_state, linear_remap

    h = state.h_ice.data
    T_ice = state.T_ice.data
    conc = state.concentration.data
    u_ice = state.u_ice.data
    v_ice = state.v_ice.data
    s11 = state.sigma_11.data
    s22 = state.sigma_22.data
    s12 = state.sigma_12.data

    # For multi-category: aggregate for coupler response and dynamics
    if h.ndim > 3:
        h_agg, T_agg, conc_agg = aggregate_state(h, T_ice, conc)
    else:
        h_agg, T_agg, conc_agg = h, T_ice, conc

    # ---- 1. Dynamics ----
    if config.dynamics == "evp" and grid is not None:
        u_ice, v_ice, s11, s22, s12 = evp_solver(
            u_ice, v_ice, s11, s22, s12,
            h_agg, conc_agg,
            forcing.u_lowest, forcing.v_lowest,
            ocean_u, ocean_v,
            grid, dt,
            N_evp=config.N_evp,
            e_yield=config.e_yield,
            P_star=config.P_star,
            C_strength=config.C_strength,
            T_evp=config.T_evp,
            rho_ice=config.rho_ice,
            rho_air=config.rho_air_ref,
            rho_ocean=config.rho_ocean_ref,
            C_ai=config.drag_atm,
            C_oi=config.drag_ocean,
            differentiable=config.differentiable_dynamics,
        )
    elif config.dynamics == "free_drift":
        u_ice, v_ice = free_drift_velocity(
            ocean_u, ocean_v,
            forcing.u_lowest, forcing.v_lowest,
            drag_ocean=config.drag_ocean,
            drag_atm=config.drag_atm,
            rho_air=config.rho_air_ref,
            rho_ice=config.rho_ice,
        )

    # ---- 2. Transport ----
    if config.transport == "advect" and grid is not None:
        if h.ndim > 3:
            # Multi-category: advect each category
            n_cat = h.shape[-1]
            h_list, conc_list, T_list = [], [], []
            for k in range(n_cat):
                h_k, c_k, T_k = advect_ice_tracers(
                    h[..., k], conc[..., k], T_ice[..., k],
                    u_ice, v_ice, grid, dt,
                )
                h_list.append(h_k)
                conc_list.append(c_k)
                T_list.append(T_k)
            h = jnp.stack(h_list, axis=-1)
            conc = jnp.stack(conc_list, axis=-1)
            T_ice = jnp.stack(T_list, axis=-1)
        else:
            h, conc, T_ice = advect_ice_tracers(
                h, conc, T_ice, u_ice, v_ice, grid, dt,
            )

    # ---- 3. Thermodynamics (per category or single) ----
    if h.ndim > 3:
        # Multi-category: apply thermodynamics per category
        h_old = h.copy()
        conc_old = conc.copy()
        n_cat = h.shape[-1]
        h_list, T_list, conc_list = [], [], []
        for k in range(n_cat):
            h_k, T_k, conc_k = _thermo_single(
                h[..., k], T_ice[..., k], conc[..., k],
                forcing, ocean_sst, config, U_min, dt,
            )
            h_list.append(h_k)
            T_list.append(T_k)
            conc_list.append(conc_k)
        h = jnp.stack(h_list, axis=-1)
        T_ice = jnp.stack(T_list, axis=-1)
        conc = jnp.stack(conc_list, axis=-1)

        # ---- 4. ITD remap ----
        h, conc = linear_remap(h_old, conc_old, h, conc, n_cat)
    else:
        h, T_ice, conc = _thermo_single(
            h, T_ice, conc, forcing, ocean_sst, config, U_min, dt,
        )

    # Re-aggregate for coupler response
    if h.ndim > 3:
        h_agg, T_agg, conc_agg = aggregate_state(h, T_ice, conc)
    else:
        h_agg, T_agg, conc_agg = h, T_ice, conc

    # ---- Build response ----
    response = _build_response(
        h_agg, T_agg, conc_agg, u_ice, v_ice,
        forcing, config, U_min,
    )

    new_state = DynamicSeaIceState(
        h_ice=state.h_ice.replace(data=h),
        T_ice=state.T_ice.replace(data=T_ice),
        concentration=state.concentration.replace(data=conc),
        u_ice=state.u_ice.replace(data=u_ice),
        v_ice=state.v_ice.replace(data=v_ice),
        sigma_11=state.sigma_11.replace(data=s11),
        sigma_22=state.sigma_22.replace(data=s22),
        sigma_12=state.sigma_12.replace(data=s12),
    )

    return new_state, response


# ==============================================================================
# Thermodynamics for a single thickness class
# ==============================================================================

def _thermo_single(
    h: jnp.ndarray,
    T_ice: jnp.ndarray,
    conc: jnp.ndarray,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Thermodynamic update for a single thickness class."""
    ice_mask = h > 0.0
    h_eff = jnp.maximum(h, config.h_ice_min)

    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
    )
    rho = forcing.rho_lowest
    q_sfc = saturation_mixing_ratio_ice(T_ice, forcing.p_surface)

    # Bulk fluxes
    tau_x = -rho * config.Cd_ice * wind_speed * forcing.u_lowest
    tau_y = -rho * config.Cd_ice * wind_speed * forcing.v_lowest
    shflx = rho * constants.c_pd * config.Ch_ice * wind_speed * (T_ice - forcing.T_lowest)
    lhflx = rho * constants.L_v * config.Ch_ice * wind_speed * (q_sfc - forcing.q_lowest)

    # Albedo
    if config.temp_dependent_albedo:
        alpha_ice = compute_ice_albedo(T_ice, config.ice_albedo)
    else:
        alpha_ice = jnp.broadcast_to(jnp.array(config.albedo_ice), h.shape)

    # Radiation
    sw_net = (1.0 - alpha_ice) * forcing.sw_down
    lw_down_abs = config.emissivity_ice * forcing.lw_down
    lw_up = config.emissivity_ice * constants.sigma_sb * T_ice ** 4
    Q_sfc = sw_net + lw_down_abs - lw_up - shflx - lhflx

    # Conductive flux
    F_cond = jnp.where(
        ice_mask,
        config.k_ice * (config.T_freeze_ocean - T_ice) / h_eff,
        0.0,
    )

    # Temperature
    skin_cap = config.rho_ice * config.c_ice * h_eff * 0.5
    dT_dt = (Q_sfc - F_cond) / skin_cap
    T_trial = T_ice + dt * dT_dt
    T_new = jnp.where(
        ice_mask,
        jnp.clip(T_trial, config.T_ice_min, config.T_freeze_ocean),
        jnp.broadcast_to(jnp.array(config.T_freeze_ocean), T_ice.shape),
    )

    # Growth/melt
    F_ocean = config.k_ice * jnp.maximum(ocean_sst - config.T_freeze_ocean, 0.0) / h_eff
    dh_dt_ice = (F_cond - F_ocean) / (config.rho_ice * config.L_f)
    freeze_flux_open = jnp.maximum(-Q_sfc, 0.0)
    dh_dt_open = freeze_flux_open / (config.rho_ice * config.L_f)
    dh_dt = jnp.where(ice_mask, dh_dt_ice, dh_dt_open)
    h_new = jnp.maximum(h + dt * dh_dt, 0.0)

    # Concentration
    dconc_growth = jnp.maximum(dh_dt, 0.0) * (1.0 - conc) / config.h_new_ice
    dconc_melt = jnp.minimum(dh_dt, 0.0) * conc / h_eff
    conc_new = jnp.clip(conc + dt * (dconc_growth + dconc_melt), 0.0, 1.0)

    return h_new, T_new, conc_new


# ==============================================================================
# Build TileResponse
# ==============================================================================

def _build_response(
    h: jnp.ndarray,
    T_ice: jnp.ndarray,
    conc: jnp.ndarray,
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    forcing: AtmToSurface,
    config: SeaIceConfig,
    U_min: float,
) -> TileResponse:
    """Build coupler response from aggregated ice fields."""
    q_sfc = saturation_mixing_ratio_ice(T_ice, forcing.p_surface)

    if config.temp_dependent_albedo:
        alpha_ice = compute_ice_albedo(T_ice, config.ice_albedo)
    else:
        alpha_ice = jnp.broadcast_to(jnp.array(config.albedo_ice), h.shape)

    lw_up = config.emissivity_ice * constants.sigma_sb * T_ice ** 4

    # Recompute surface fluxes from aggregated state
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
    )
    rho = forcing.rho_lowest
    shflx = rho * constants.c_pd * config.Ch_ice * wind_speed * (T_ice - forcing.T_lowest)
    lhflx = rho * constants.L_v * config.Ch_ice * wind_speed * (q_sfc - forcing.q_lowest)
    tau_x = -rho * config.Cd_ice * wind_speed * forcing.u_lowest
    tau_y = -rho * config.Cd_ice * wind_speed * forcing.v_lowest

    return TileResponse(
        T_surface=T_ice,
        albedo=alpha_ice,
        emissivity=jnp.broadcast_to(jnp.array(config.emissivity_ice), h.shape),
        z0=jnp.broadcast_to(jnp.array(config.z0_ice), h.shape),
        q_surface=q_sfc,
        shflx=shflx,
        lhflx=lhflx,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up,
        u_ocean_sfc=u_ice,
        v_ocean_sfc=v_ice,
        co2_flux=jnp.zeros_like(h),
    )
