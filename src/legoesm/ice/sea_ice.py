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

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio_ice
from legoesm.coupler.bulk_flux import simple_bulk_fluxes
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.coupler.surface_energy import surface_radiation_fluxes
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

    # ---------- Surface fluxes (MOST dispatch stays in slab path) ----------
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
        tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_ice, q_sfc, rho, wind_speed,
            config.Cd_ice, config.Ch_ice,
        )

    # ---------- Thermodynamics (delegate to shared routine) ----------
    h_new, T_ice_new, conc_new = _thermo_single(
        h, T_ice, conc, forcing, ocean_sst, config, U_min, dt,
        shflx=shflx, lhflx=lhflx,
    )

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

    # ---------- Build response ----------
    if config.temp_dependent_albedo:
        alpha_ice = compute_ice_albedo(T_ice_new, config.ice_albedo)
    else:
        alpha_ice = jnp.broadcast_to(jnp.array(config.albedo_ice), h.shape)

    _, _, lw_up_new = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_ice_new, alpha_ice,
        config.emissivity_ice,
    )
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
            # Multi-category: advect each category via vmap over last axis
            def _advect_cat(h_k, conc_k, T_k):
                return advect_ice_tracers(
                    h_k, conc_k, T_k, u_ice, v_ice, grid, dt,
                )
            # Move category axis to front for vmap, then back
            h_t = jnp.moveaxis(h, -1, 0)
            conc_t = jnp.moveaxis(conc, -1, 0)
            T_t = jnp.moveaxis(T_ice, -1, 0)
            h_t, conc_t, T_t = jax.vmap(_advect_cat)(h_t, conc_t, T_t)
            h = jnp.moveaxis(h_t, 0, -1)
            conc = jnp.moveaxis(conc_t, 0, -1)
            T_ice = jnp.moveaxis(T_t, 0, -1)
        else:
            h, conc, T_ice = advect_ice_tracers(
                h, conc, T_ice, u_ice, v_ice, grid, dt,
            )

    # ---- 3. Thermodynamics (per category or single) ----
    if h.ndim > 3:
        # Multi-category: apply thermodynamics per category via vmap
        n_cat = h.shape[-1]
        h_old = h
        conc_old = conc

        def _thermo_cat(h_k, T_k, conc_k):
            return _thermo_single(
                h_k, T_k, conc_k,
                forcing, ocean_sst, config, U_min, dt,
            )

        h_t = jnp.moveaxis(h, -1, 0)
        T_t = jnp.moveaxis(T_ice, -1, 0)
        conc_t = jnp.moveaxis(conc, -1, 0)
        h_t, T_t, conc_t = jax.vmap(_thermo_cat)(h_t, T_t, conc_t)
        h = jnp.moveaxis(h_t, 0, -1)
        T_ice = jnp.moveaxis(T_t, 0, -1)
        conc = jnp.moveaxis(conc_t, 0, -1)

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
    shflx: jnp.ndarray | None = None,
    lhflx: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Thermodynamic update for a single thickness class.

    Parameters
    ----------
    shflx, lhflx : jnp.ndarray or None
        Pre-computed sensible and latent heat fluxes.  When *None*
        (the default), ``simple_bulk_fluxes`` is called internally.
        Pass pre-computed values when the caller uses a different
        bulk-flux scheme (e.g. MOST).
    """
    ice_mask = h > 0.0
    h_eff = jnp.maximum(h, config.h_ice_min)

    # Bulk fluxes — use caller-supplied values when available.
    if shflx is None or lhflx is None:
        wind_speed = jnp.sqrt(
            forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
        )
        rho = forcing.rho_lowest
        q_sfc = saturation_mixing_ratio_ice(T_ice, forcing.p_surface)
        _, _, shflx, lhflx = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_ice, q_sfc, rho, wind_speed,
            config.Cd_ice, config.Ch_ice,
        )

    # Albedo
    if config.temp_dependent_albedo:
        alpha_ice = compute_ice_albedo(T_ice, config.ice_albedo)
    else:
        alpha_ice = jnp.broadcast_to(jnp.array(config.albedo_ice), h.shape)

    # Radiation
    sw_net, lw_net, lw_up = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_ice, alpha_ice,
        config.emissivity_ice,
    )
    Q_sfc = sw_net + lw_net - shflx - lhflx

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

    # Surface melt: if T_trial exceeds freezing, the excess enthalpy melts
    # ice from the top instead of being discarded by the temperature clamp.
    excess_energy = skin_cap * jnp.maximum(
        T_trial - config.T_freeze_ocean, 0.0
    ) / dt  # [W/m²]
    dh_dt_surface_melt = -excess_energy / (config.rho_ice * config.L_f)

    # Growth/melt — turbulent ocean heat transfer (not conductive scaling)
    F_ocean = config.ocean_heat_transfer_coeff * jnp.maximum(
        ocean_sst - config.T_freeze_ocean, 0.0,
    )
    dh_dt_basal = (F_cond - F_ocean) / (config.rho_ice * config.L_f)

    # Combine surface and basal melt/growth for existing ice
    dh_dt_ice = dh_dt_basal + dh_dt_surface_melt

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

    _, _, lw_up = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_ice, alpha_ice,
        config.emissivity_ice,
    )

    # Recompute surface fluxes from aggregated state
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
    )
    rho = forcing.rho_lowest
    tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
        forcing.u_lowest, forcing.v_lowest,
        forcing.T_lowest, forcing.q_lowest,
        T_ice, q_sfc, rho, wind_speed,
        config.Cd_ice, config.Ch_ice,
    )

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
