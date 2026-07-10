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
from legoesm.core.field import Field
from legoesm.thermo import (
    saturation_mixing_ratio,
    saturation_mixing_ratio_ice,
)
from legoesm.core.bulk_flux import simple_bulk_fluxes, compute_most_fluxes
from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm.core.surface_energy import surface_radiation_fluxes
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

    # Surface humidity: saturated at epilimnion temperature.  Use
    # the ice saturation form when the lake is at or below freezing
    # (frozen-over lakes sublimate, not evaporate).
    is_frozen = T_epi <= config.T_freeze
    q_sfc_liq = saturation_mixing_ratio(T_epi, forcing.p_surface)
    q_sfc_ice = saturation_mixing_ratio_ice(T_epi, forcing.p_surface)
    q_sfc = jnp.where(is_frozen, q_sfc_ice, q_sfc_liq)

    # Latent heat: sublimation (L_s) when frozen, vaporization (L_v)
    # otherwise.  ``simple_bulk_fluxes`` and ``compute_most_fluxes``
    # both return ``lhflx`` as a positive-up energy flux equal to
    # ``L · evap_rate``, so phase-correct L is the only switch needed.
    # Earlier the lake always used L_v, biasing lhflx by ~13% over
    # frozen lakes (coupler-conservation audit F17).
    L_eff = jnp.where(is_frozen, constants.L_s, constants.L_v)

    # Bulk fluxes
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
            T_epi, q_sfc, rho,
            z_ref=config.z_ref,
            z0_init=config.z0_lake,
            scheme=config.bulk_scheme,
            n_iter=config.bulk_n_iter,
            L_latent=L_eff,
        )
    else:
        tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_epi, q_sfc, rho, wind_speed,
            config.Cd_lake, config.Ch_lake,
            L_latent=L_eff,
        )

    # Radiation.  A frozen lake reflects like ice/snow, not open water —
    # completes the ``is_frozen`` switch already applied to q_sfc and L_eff.
    # (Diagnostic-only ice: no prognostic thickness yet; upgrade = FLake ice.)
    albedo_eff = jnp.where(is_frozen, config.albedo_lake_ice, config.albedo_lake)
    sw_net, lw_net, lw_up = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_epi, albedo_eff,
        config.emissivity_lake,
    )

    # Vertical mixing: wind-enhanced
    k_eff = config.k_mix * (1.0 + config.wind_mix_alpha * wind_speed)
    d_mid = 0.5 * (config.h_epi + config.h_hypo)
    F_mix = config.rho_water * config.c_water_mass * k_eff * (T_epi - T_hypo) / d_mid

    # Epilimnion energy balance
    cap_epi = config.rho_water * config.c_water_mass * config.h_epi
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
    cap_hypo = config.rho_water * config.c_water_mass * config.h_hypo
    dT_hypo_dt = F_mix / cap_hypo
    T_trial_hypo = T_hypo + dt * dT_hypo_dt
    T_hypo_new = jnp.maximum(T_trial_hypo, T_freeze)
    Q_freeze_hypo = cap_hypo * jnp.maximum(T_freeze - T_trial_hypo, 0.0) / dt

    # Convective overturn for freshwater density inversion.
    # Freshwater density peaks at T_max ≈ 3.983 °C (277.133 K) — its
    # local anomaly is well-approximated by
    #     ρ(T) = ρ_max · (1 − α · (T − T_max)²)
    # with α ≈ 8.0e-6 K⁻² from the Kell (1975) polynomial.  In autumn
    # / winter the epilimnion can cool below the hypolimnion's
    # temperature while both layers remain above T_max — making ρ_epi
    # > ρ_hypo even though T_epi < T_hypo (T_epi is closer to the
    # max-density point).  Without convective overturn the model
    # freezes this static instability in place; CICE / ALMA / FLake
    # instantaneously homogenize the two layers.  The α constant
    # cancels in the boolean ρ_epi > ρ_hypo comparison, but using the
    # canonical Kell value keeps the EOS reusable for future buoyancy
    # / N² diagnostics.
    rho_epi = 1.0 - constants.rho_freshwater_curvature * (
        T_epi_new - constants.T_freshwater_max_density
    ) ** 2
    rho_hypo = 1.0 - constants.rho_freshwater_curvature * (
        T_hypo_new - constants.T_freshwater_max_density
    ) ** 2
    unstable = rho_epi > rho_hypo
    T_mix = (
        cap_epi * T_epi_new + cap_hypo * T_hypo_new
    ) / (cap_epi + cap_hypo)
    T_epi_new = jnp.where(unstable, T_mix, T_epi_new)
    T_hypo_new = jnp.where(unstable, T_mix, T_hypo_new)

    Q_freeze_total = Q_freeze_epi + Q_freeze_hypo

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

    # Upward longwave with explicit reflected component, matching the
    # convention used by every other tile (slab/multilayer land via
    # ``surface_radiation_fluxes``; sea-ice; ocean):
    #     lw_up = ε σ T⁴ + (1 − ε) · lw_down
    # The earlier formulation dropped the reflection term, biasing the
    # lake-tile lw_up low by (1−ε)·lw_down ≈ 10 W/m² (ε=0.97,
    # lw_down ≈ 350 W/m²) and feeding that bias up into the atmosphere
    # TOA budget proportional to lake fraction.
    lw_up_new = (
        config.emissivity_lake * constants.sigma_sb * T_epi_new ** 4
        + (1.0 - config.emissivity_lake) * forcing.lw_down
    )

    # Recompute q_surface from updated epilimnion temperature for
    # consistency.  Apply the same liquid / ice saturation phase
    # switch used pre-step (line 50-53): frozen lakes (T_epi ≤
    # T_freeze) sublimate, not evaporate.  Without the switch, the
    # response.q_surface reported to the atmosphere was biased high
    # by ~14 % at T = −10 °C (liquid vs ice saturation), feeding the
    # next-step bulk-flux computation a non-physical q_sfc.
    is_frozen_new = T_epi_new <= config.T_freeze
    q_sfc_liq_new = saturation_mixing_ratio(T_epi_new, forcing.p_surface)
    q_sfc_ice_new = saturation_mixing_ratio_ice(T_epi_new, forcing.p_surface)
    q_sfc_new = jnp.where(is_frozen_new, q_sfc_ice_new, q_sfc_liq_new)

    # ``jnp.full(shape, scalar, dtype=...)`` lowers to a single
    # ``Broadcast`` HLO op, whereas ``jnp.broadcast_to(jnp.array(scalar), ...)``
    # also forces a ``ConvertElementType`` for the implicit dtype
    # promotion of the Python float.  Tiny per-call savings but
    # this fires every coupler step.
    _t_dtype = T_epi.dtype
    # Lake → ocean freshwater: P − E (E uses L_eff to handle frozen
    # lakes correctly; we recover the mass via E = lhflx / L_eff).
    # Lakes are typically internally drained but for the tile-blend
    # accounting we still report the surface mass-flux signal.
    evap_rate = lhflx / L_eff
    freshwater_flux = forcing.precip_total - evap_rate
    response = TileResponse(
        T_sfc=T_epi_new,
        # Reported albedo tracks the POST-step phase (like q_surface_new): the
        # atmosphere sees the end-of-step surface next.  (SW absorbed THIS step
        # used the pre-step albedo_eff above — correct for the state it had.)
        albedo=jnp.where(is_frozen_new, config.albedo_lake_ice,
                         config.albedo_lake).astype(_t_dtype),
        emissivity=jnp.full(T_epi.shape, config.emissivity_lake, dtype=_t_dtype),
        z0=jnp.full(T_epi.shape, config.z0_lake, dtype=_t_dtype),
        q_surface=q_sfc_new,
        shflx=shflx,
        lhflx=lhflx,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up_new,
        u_ocean_sfc=jnp.zeros_like(T_epi),
        v_ocean_sfc=jnp.zeros_like(T_epi),
        co2_flux=jnp.zeros_like(T_epi),
        freshwater_flux=freshwater_flux,
        # Lake tiles do not extract heat from the ocean.
        ocean_heat_extraction=jnp.zeros_like(T_epi),
        # Lake tiles do not exert stress on the ocean.
        ocean_stress_x=jnp.zeros_like(T_epi),
        ocean_stress_y=jnp.zeros_like(T_epi),
        # Phase-aware moisture mass flux (L_eff already switches to
        # L_s on frozen lakes, iter-11; ``evap_rate`` was just
        # computed above for freshwater_flux).
        surface_mass_flux=evap_rate,
        # Lake tile does not exchange salt with the ocean.
        salt_flux=jnp.zeros_like(T_epi),
    )

    return new_state, response
