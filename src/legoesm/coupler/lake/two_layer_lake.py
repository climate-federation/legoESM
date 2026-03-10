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
from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
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
    q_sfc = saturation_mixing_ratio(T_epi, forcing.p_surface)

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
        tau_x = -rho * config.Cd_lake * wind_speed * forcing.u_lowest
        tau_y = -rho * config.Cd_lake * wind_speed * forcing.v_lowest
        shflx = rho * constants.c_pd * config.Ch_lake * wind_speed * (T_epi - forcing.T_lowest)
        lhflx = rho * constants.L_v * config.Ch_lake * wind_speed * (q_sfc - forcing.q_lowest)

    # Radiation
    sw_net = (1.0 - config.albedo_lake) * forcing.sw_down
    lw_down_abs = config.emissivity_lake * forcing.lw_down
    lw_up = config.emissivity_lake * constants.sigma_sb * T_epi ** 4
    lw_net = lw_down_abs - lw_up

    # Vertical mixing: wind-enhanced
    k_eff = config.k_mix * (1.0 + config.wind_mix_alpha * wind_speed)
    d_mid = 0.5 * (config.h_epi + config.h_hypo)
    F_mix = config.rho_water * config.c_water * k_eff * (T_epi - T_hypo) / d_mid

    # Epilimnion energy balance
    cap_epi = config.rho_water * config.c_water * config.h_epi
    dT_epi_dt = (sw_net + lw_net - shflx - lhflx - F_mix) / cap_epi
    T_epi_new = T_epi + dt * dT_epi_dt

    # Hypolimnion: receives mixing flux only
    cap_hypo = config.rho_water * config.c_water * config.h_hypo
    dT_hypo_dt = F_mix / cap_hypo
    T_hypo_new = T_hypo + dt * dT_hypo_dt

    new_state = LakeState(
        T_epi=state.T_epi.replace(data=T_epi_new),
        T_hypo=state.T_hypo.replace(data=T_hypo_new),
    )

    lw_up_new = config.emissivity_lake * constants.sigma_sb * T_epi_new ** 4

    # Recompute q_surface from updated epilimnion temperature for consistency
    q_sfc_new = saturation_mixing_ratio(T_epi_new, forcing.p_surface)

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
