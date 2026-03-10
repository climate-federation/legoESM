"""Thermodynamic slab sea ice model.

Thermodynamics:
    Surface energy balance determines T_ice.
    Conductive flux through ice: F_cond = k_ice * (T_freeze - T_ice) / (h + h_min)
    Growth/melt: dh/dt = (F_cond - F_ocean) / (rho_ice * L_f)

Transport (simplified free-drift):
    u_ice ~ drag_ocean * u_ocean + drag_atm * (rho_air/rho_ice) * u_wind
    (Advection of h_ice and concentration is not included in this slab version;
     only the thermodynamic response and velocity are computed.)
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio_ice
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import SeaIceState
from legoesm.surface_albedo import ice_albedo as compute_ice_albedo


def step_sea_ice(
    state: SeaIceState,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
) -> tuple[SeaIceState, TileResponse]:
    """Step the sea ice model forward by dt seconds.

    Parameters
    ----------
    state : SeaIceState
        Current ice state.
    forcing : AtmToSurface
        Atmospheric forcing fields.
    ocean_sst : jnp.ndarray
        Ocean SST [K], shape (6, n, n). Used as bottom boundary.
    ocean_u, ocean_v : jnp.ndarray
        Ocean surface currents [m/s], shape (6, n, n).
    config : SeaIceConfig
        Sea ice model parameters.
    U_min : float
        Minimum wind speed floor [m/s].
    dt : float
        Time step [s].
    """
    h = state.h_ice.data
    T_ice = state.T_ice.data
    conc = state.concentration.data

    # Only existing ice supports conductive flux through the slab.
    ice_mask = h > 0.0
    h_eff = jnp.maximum(h, config.h_ice_min)

    # ---------- Surface fluxes ----------
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
    )
    rho = forcing.rho_lowest

    # Surface humidity: ice-phase saturation
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

    # Net surface energy into ice surface
    Q_sfc = sw_net + lw_net - shflx - lhflx

    # ---------- Conductive flux ----------
    F_cond = jnp.where(
        ice_mask,
        config.k_ice * (config.T_freeze_ocean - T_ice) / h_eff,
        0.0,
    )

    # ---------- Surface temperature evolution ----------
    # Only the thin surface layer responds: use rho_ice * c_ice * h_eff / 2
    skin_cap = config.rho_ice * config.c_ice * h_eff * 0.5
    dT_dt = (Q_sfc - F_cond) / skin_cap
    T_ice_trial = T_ice + dt * dT_dt
    # Existing ice stays in physically plausible bounds; open water carries
    # no ice-skin state and is pinned to freezing.
    T_ice_new = jnp.where(
        ice_mask,
        jnp.clip(T_ice_trial, config.T_ice_min, config.T_freeze_ocean),
        jnp.broadcast_to(jnp.array(config.T_freeze_ocean), T_ice.shape),
    )

    # ---------- Growth / melt ----------
    # Ocean heat flux: when SST > T_freeze, ocean melts ice from below
    F_ocean = config.k_ice * jnp.maximum(ocean_sst - config.T_freeze_ocean, 0.0) / h_eff
    dh_dt_ice = (F_cond - F_ocean) / (config.rho_ice * config.L_f)
    # For open-water cells (h=0), new ice forms only when the net surface
    # energy budget extracts heat (Q_sfc < 0).
    freeze_flux_open = jnp.maximum(-Q_sfc, 0.0)
    dh_dt_open = freeze_flux_open / (config.rho_ice * config.L_f)
    dh_dt = jnp.where(ice_mask, dh_dt_ice, dh_dt_open)
    h_new = jnp.maximum(h + dt * dh_dt, 0.0)

    # ---------- Concentration ----------
    # Growing: open-water freezing creates new ice at h_new_ice thickness
    # Melting: concentration decreases proportionally to thickness loss
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

    # Recompute q_surface from updated ice temperature for consistency
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
