"""Simple bulk-flux surface energy balance (SimpleSEB).

``SimpleSEB`` is the default surface scheme for the slab and multilayer
land models.  The surface behaves as a homogeneous skin with temperature
equal to the top-layer soil T (multilayer) or the single-slab T.  Fluxes
are computed via bulk transfer with an optional stomatal down-regulation
(Jarvis or coupled Leuning Farquhar + Ball-Berry / Medlyn via
``legoesm.land.stomata_utils.compute_effective_beta``).

``SimpleSEBConfig`` is a marker NamedTuple — it has no scheme-specific
parameters because the stomatal / carbon configuration is still read
from ``config.stomata`` at the top level of ``LandConfig`` /
``MultiLayerLandConfig``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice
from legoesm.coupler.bulk_flux import simple_bulk_fluxes
from legoesm.coupler.coupling_fields import AtmToSurface
from legoesm.coupler.surface_energy import surface_radiation_fluxes
from legoesm.land.carbon.config import CarbonState
from legoesm.land.stomata_utils import compute_effective_beta
from legoesm.land.surface_scheme.base import SurfaceFluxOutput
from legoesm.surface_albedo import land_albedo as compute_land_albedo


class SimpleSEBConfig(NamedTuple):
    """Marker config for the simple bulk-flux surface scheme.

    Has no fields — the stomatal / carbon configuration lives on
    ``config.stomata`` at the top level of ``LandConfig`` /
    ``MultiLayerLandConfig``.  This NamedTuple only exists so that the
    ``surface_scheme`` dispatch in ``step_*_land`` can distinguish it
    from ``TwoLeafCanopyConfig`` via ``isinstance``.
    """


def compute_simple_seb_fluxes(
    *,
    T_surface: jnp.ndarray,
    snow: jnp.ndarray,
    snow_age: jnp.ndarray,
    beta_soil: jnp.ndarray,
    forcing: AtmToSurface,
    land_config,
    U_min: float,
    lat: jnp.ndarray | None,
    carbon_state: CarbonState | None,
    dt: float,
    land_params,
    albedo_land,
    emissivity,
    z0,
) -> SurfaceFluxOutput:
    """Compute SimpleSEB surface fluxes for one time step.

    Operates identically for slab (``T_surface = T_soil``,
    ``beta_soil`` from bucket) and multilayer (``T_surface = T_soil[:, 0]``,
    ``beta_soil`` from root-zone weighted theta).  The caller prepares
    the per-column ``beta_soil`` and surface parameters so this function
    is agnostic to whether soil state is a slab or a multi-layer column.

    Parameters
    ----------
    T_surface : (ncol,) skin temperature [K]
    snow      : (ncol,) snow water equivalent [kg/m^2]
    snow_age  : (ncol,) snow age [s]
    beta_soil : (ncol,) root-zone weighted soil moisture beta [0-1]
    forcing   : atmospheric forcing fields
    land_config : LandConfig or MultiLayerLandConfig
                  (must have ``stomata``, ``carbon``, ``snow_albedo_feedback``,
                  ``land_albedo``, ``bulk_scheme``, ``Cd_land``, ``Ch_land``,
                  ``z_ref``, ``bulk_n_iter``)
    U_min     : minimum wind speed floor [m/s]
    lat       : latitude [rad] (for snow-albedo feedback)
    carbon_state : current carbon pool state (for Farquhar coupling)
    dt        : time step [s]
    land_params : optional LandSurfaceParams
    albedo_land : spatial or scalar bare-land albedo
    emissivity  : spatial or scalar emissivity
    z0          : spatial or scalar roughness length [m]

    Returns
    -------
    SurfaceFluxOutput with ``gpp`` populated if ``stomata.enabled`` and
    ``carbon.scheme == "differland"``.
    """
    # --- Wind speed with floor ---
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2)

    # --- Stomatal + Farquhar beta (uses forcing.sw_down, co2, q_lowest) ---
    beta, gpp_farq = compute_effective_beta(
        T_surface, forcing, beta_soil, land_config, carbon_state, dt,
        land_params=land_params)

    # Ratio used in the post-flux q_surface recomputation to carry
    # stomatal limitation through the updated soil-moisture state.
    stomatal_ratio = beta / jnp.maximum(beta_soil, 1e-10)

    # --- Surface saturation humidity: ice over snow, liquid over bare soil ---
    q_sat_liq = saturation_mixing_ratio(T_surface, forcing.p_surface)
    q_sat_ice = saturation_mixing_ratio_ice(T_surface, forcing.p_surface)
    # Iter-68 audit fix (ported from main during the jianing/land ↔ main
    # sync 2026-06-03).  A brief warm-surface snowfall event would
    # otherwise flip the latent-heat phase to L_s for the whole step even
    # though the new snow melts in seconds.  Gate the phase decision on
    # (existing snowpack) OR (fresh snowfall AND T_surface < T_freeze) so
    # we only enter snow phase when the snow can actually survive the step.
    fresh_snow_mass = forcing.precip_snow * dt
    has_existing_snow = snow > 1e-6
    has_surviving_fresh_snow = (
        (fresh_snow_mass > 1e-6) & (T_surface < constants.T_freeze)
    )
    has_snow = has_existing_snow | has_surviving_fresh_snow
    q_sat_sfc = jnp.where(has_snow, q_sat_ice, q_sat_liq)
    # Snow surface is freely evaporating (snowpack limits later).
    beta_effective = jnp.where(has_snow, 1.0, beta)
    q_sfc = beta_effective * q_sat_sfc

    # Phase-appropriate latent heat (consistent with iter-68 gate above).
    L_eff = jnp.where(has_snow, constants.L_s, constants.L_v)

    # --- Bulk fluxes ---
    rho = forcing.rho_lowest
    if land_config.bulk_scheme in ("most", "coare3", "large_yeager"):
        from legoesm.coupler.bulk_flux import compute_most_fluxes
        tau_x, tau_y, shflx, lhflx, _ = compute_most_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_surface, q_sfc, rho,
            z_ref=land_config.z_ref,
            z0_init=z0,
            scheme=land_config.bulk_scheme,
            n_iter=land_config.bulk_n_iter,
            L_latent=L_eff,
        )
    else:
        tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_surface, q_sfc, rho, wind_speed,
            land_config.Cd_land, land_config.Ch_land,
            L_latent=L_eff,
        )

    # --- Surface albedo (iter-71 audit fix ported from main 2026-06-03) ---
    # Use the SAME effective snow mass as the iter-68 bulk-flux phase
    # decision above so SW absorption does not lag the LH/SH phase
    # transition by one step on every fresh-snow event.  Without this
    # consistency, albedo would treat the column as snow-free for one
    # step while LH was already computed as snow phase.
    snow_effective = jnp.where(
        has_existing_snow | has_surviving_fresh_snow,
        snow + jnp.where(has_surviving_fresh_snow, fresh_snow_mass, 0.0),
        snow,
    )
    if land_config.snow_albedo_feedback and lat is not None:
        alpha = compute_land_albedo(
            lat, snow_effective, snow_age, land_config.land_albedo)
    else:
        alpha = jnp.broadcast_to(jnp.asarray(albedo_land), T_surface.shape)

    # --- Radiation ---
    sw_net, lw_net, lw_up = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_surface, alpha, emissivity)

    # --- Ground heat flux (residual of surface energy balance) ---
    G_soil = sw_net + lw_net - shflx - lhflx

    return SurfaceFluxOutput(
        shflx=shflx, lhflx=lhflx,
        tau_x=tau_x, tau_y=tau_y,
        sw_net=sw_net, lw_net=lw_net, lw_up=lw_up,
        G_soil=G_soil,
        T_surface=T_surface,
        q_surface=q_sfc,
        albedo=alpha,
        emissivity=jnp.broadcast_to(jnp.asarray(emissivity), T_surface.shape),
        z0=jnp.broadcast_to(jnp.asarray(z0), T_surface.shape),
        gpp=gpp_farq,
        stomatal_ratio=stomatal_ratio,
        # Canopy-specific diagnostics left as None
    )
