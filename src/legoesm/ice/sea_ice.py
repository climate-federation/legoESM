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

    # Effective thickness for smooth division
    h_eff = h + config.h_ice_min

    # ---------- Surface fluxes ----------
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
    )
    rho = forcing.rho_lowest

    # Surface humidity: ice-phase saturation
    q_sfc = saturation_mixing_ratio_ice(T_ice, forcing.p_surface)

    tau_x = -rho * config.Cd_ice * wind_speed * forcing.u_lowest
    tau_y = -rho * config.Cd_ice * wind_speed * forcing.v_lowest
    shflx = rho * constants.c_pd * config.Ch_ice * wind_speed * (T_ice - forcing.T_lowest)
    lhflx = rho * constants.L_v * config.Ch_ice * wind_speed * (q_sfc - forcing.q_lowest)

    # ---------- Radiation ----------
    sw_net = (1.0 - config.albedo_ice) * forcing.sw_down
    lw_down_abs = config.emissivity_ice * forcing.lw_down
    lw_up = config.emissivity_ice * constants.sigma_sb * T_ice ** 4
    lw_net = lw_down_abs - lw_up

    # Net surface energy into ice surface
    Q_sfc = sw_net + lw_net - shflx - lhflx

    # ---------- Conductive flux ----------
    F_cond = config.k_ice * (config.T_freeze_ocean - T_ice) / h_eff

    # ---------- Surface temperature evolution ----------
    # Only the thin surface layer responds: use rho_ice * c_ice * h_eff / 2
    skin_cap = config.rho_ice * config.c_ice * h_eff * 0.5
    dT_dt = (Q_sfc - F_cond) / skin_cap
    T_ice_new = T_ice + dt * dT_dt

    # Cap at freezing — surface melt doesn't superheat
    T_ice_new = jnp.minimum(T_ice_new, config.T_freeze_ocean)

    # ---------- Growth / melt ----------
    # Ocean heat flux: when SST > T_freeze, ocean melts ice from below
    F_ocean = config.k_ice * jnp.maximum(ocean_sst - config.T_freeze_ocean, 0.0) / h_eff
    dh_dt = (F_cond - F_ocean) / (config.rho_ice * config.L_f)
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

    response = TileResponse(
        T_surface=T_ice_new,
        albedo=jnp.broadcast_to(jnp.array(config.albedo_ice), h.shape),
        emissivity=jnp.broadcast_to(jnp.array(config.emissivity_ice), h.shape),
        z0=jnp.broadcast_to(jnp.array(config.z0_ice), h.shape),
        q_surface=q_sfc,
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
