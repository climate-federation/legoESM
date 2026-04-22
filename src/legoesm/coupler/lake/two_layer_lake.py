"""Two-layer lake model (epilimnion + hypolimnion).

Epilimnion energy balance:
    rho*c*h_epi * dT_epi/dt = SW_net + LW_net - SH - LH - F_mix

Vertical mixing:
    F_mix = rho*c*k_mix_eff * (T_epi - T_hypo) / (0.5*(h_epi + h_hypo))
    k_mix_eff = k_mix * (1 + alpha * |V|)

Hypolimnion:
    rho*c*h_hypo * dT_hypo/dt = F_mix
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_specific_humidity
from legoesm.coupler.bulk_flux import simple_bulk_fluxes
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.coupler.surface_energy import surface_radiation_fluxes
from legoesm.coupler.lake.config import LakeConfig
from legoesm.coupler.lake.state import LakeState


def step_lake(
    state: LakeState,
    forcing: AtmToSurface,
    config: LakeConfig,
    U_min: float,
    dt: float,
) -> tuple[LakeState, TileResponse]:
    """Step the two-layer lake model forward by dt seconds."""
    T_epi = state.T_epi.data
    T_hypo = state.T_hypo.data

    # Wind speed with smooth floor
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
    )

    # Surface humidity: saturated at epilimnion temperature
    q_sfc = saturation_specific_humidity(T_epi, forcing.p_surface)

    # Bulk fluxes
    rho = forcing.rho_lowest

    if config.bulk_scheme in ("most", "coare3", "large_yeager"):
        from legoesm.coupler.bulk_flux import compute_most_fluxes
        tau_x, tau_y, shflx, lhflx, _ = compute_most_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_epi, q_sfc, rho,
            z_ref=config.z_ref,
            z0_init=config.z0_lake,
            scheme=config.bulk_scheme,
            n_iter=config.bulk_n_iter,
        )
    else:
        tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_epi, q_sfc, rho, wind_speed,
            config.Cd_lake, config.Ch_lake,
        )

    # Radiation
    sw_net, lw_net, lw_up = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_epi, config.albedo_lake,
        config.emissivity_lake,
    )

    # Vertical mixing: wind-enhanced
    k_eff = config.k_mix * (1.0 + config.wind_mix_alpha * wind_speed)
    d_mid = 0.5 * (config.h_epi + config.h_hypo)
    F_mix = config.rho_water * config.c_water * k_eff * (T_epi - T_hypo) / d_mid

    # Epilimnion energy balance
    cap_epi = config.rho_water * config.c_water * config.h_epi
    dT_epi_dt = (sw_net + lw_net - shflx - lhflx - F_mix) / cap_epi
    T_trial_epi = T_epi + dt * dT_epi_dt

    # Lake freezing: clamp T at freezing point.
    # Energy removed by clamping is diagnosed as Q_freeze (ice-formation
    # latent heat flux).  This closes the layer energy budget:
    #   cap * dT/dt = (net flux in) - Q_freeze
    T_freeze = config.T_freeze
    T_epi_new = jnp.maximum(T_trial_epi, T_freeze)
    Q_freeze_epi = cap_epi * jnp.maximum(T_freeze - T_trial_epi, 0.0) / dt

    # Hypolimnion: receives mixing flux only
    cap_hypo = config.rho_water * config.c_water * config.h_hypo
    dT_hypo_dt = F_mix / cap_hypo
    T_trial_hypo = T_hypo + dt * dT_hypo_dt
    T_hypo_new = jnp.maximum(T_trial_hypo, T_freeze)
    Q_freeze_hypo = cap_hypo * jnp.maximum(T_freeze - T_trial_hypo, 0.0) / dt

    Q_freeze_total = Q_freeze_epi + Q_freeze_hypo

    from legoesm.core.field import Field
    Q_freeze_field = None
    if state.Q_freeze is not None:
        Q_freeze_field = state.Q_freeze.replace(data=Q_freeze_total)
    else:
        Q_freeze_field = Field(
            data=Q_freeze_total, name="Q_freeze",
            dims=state.T_epi.dims, units="W/m2",
        )

    new_state = LakeState(
        T_epi=state.T_epi.replace(data=T_epi_new),
        T_hypo=state.T_hypo.replace(data=T_hypo_new),
        Q_freeze=Q_freeze_field,
    )

    _, _, lw_up_new = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_epi_new, config.albedo_lake,
        config.emissivity_lake,
    )

    # Recompute q_surface from updated epilimnion temperature for consistency
    q_sfc_new = saturation_specific_humidity(T_epi_new, forcing.p_surface)

    response = TileResponse(
        T_surface=T_epi_new,
        albedo=jnp.broadcast_to(jnp.array(config.albedo_lake), T_epi.shape),
        emissivity=jnp.broadcast_to(jnp.array(config.emissivity_lake), T_epi.shape),
        z0=jnp.broadcast_to(jnp.array(config.z0_lake), T_epi.shape),
        q_surface=q_sfc_new,
        shflx=shflx,
        lhflx=lhflx,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up_new,
        u_ocean_sfc=jnp.zeros_like(T_epi),
        v_ocean_sfc=jnp.zeros_like(T_epi),
        co2_flux=jnp.zeros_like(T_epi),
    )

    return new_state, response
