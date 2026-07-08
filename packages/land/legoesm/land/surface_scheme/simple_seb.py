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
from legoesm.core.bulk_flux import simple_bulk_fluxes, compute_most_fluxes
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.core.surface_energy import surface_radiation_fluxes
from legoesm.land.carbon.config import CarbonState
from legoesm.land.stomata_utils import compute_effective_beta
from legoesm.land.surface_scheme.base import SurfaceFluxOutput
from legoesm.surface_albedo import land_albedo as compute_land_albedo


# --- SimpleSEB cold-start numerics safety guards (peers; NOT climate-tuning knobs) ---
# Public so the slab-land inline flux path (slab_land.step_land) applies the SAME
# two guards instead of carrying its own copy (single source of truth).
# Ceiling on the neutral-equivalent bulk transfer coefficient for the MOST land
# fluxes: the default floor lets C_e reach ~0.64 at extreme cold-start instability,
# turning a ~0.6 g/kg humidity gradient into a spurious ~4900 W/m2 latent shock that
# NaNs the thin top soil layer.  0.02 is a generous strong-instability bound (>> the
# ~3.4e-3 neutral value), so it binds ONLY on pathological cold-start columns and is
# climatologically inert (byte-identical) once the surface has spun up.
LAND_MAX_EXCHANGE_COEFF = 0.02
# Floor on the CONDENSATION (negative) latent flux [W/m2] (issue #730): a cold/dry
# surface under moister advected air can produce a spurious ~-3000 W/m2 condensation
# flux (vs real frost/dew ~O(10-100)) which the SEB balances at an unphysical hot skin
# T -> thin top-layer runaway (land skin 224 -> 1156 K -> NaN in coupled AMIP without
# it).  -150 W/m2 is safely ABOVE any real frost/dew, so near-inert; only CONDENSATION
# is floored (evaporation stays free so the SEB keeps its self-limiting feedback).
# None disables it.
LAND_CONDENSATION_FLOOR_W = -150.0
# Perturbation [K] for the one-sided finite-difference linearisation of the turbulent
# fluxes when building the semi-implicit surface conductance (Robin BC).  Small enough
# for an accurate slope, large enough to stay above bulk-flux round-off.
_SURFACE_LIN_DT_K = 0.1


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
    beta, gpp_farq, sif_farq = compute_effective_beta(
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
    # Dispatch hardening (restores the guard lost when this dispatch moved
    # out of step_multilayer_land in the surface-scheme refactor): an unknown
    # bulk_scheme must raise, not silently run the constant-coefficient else.
    _valid_bulk = ("constant", "most", "coare3", "large_yeager")
    if land_config.bulk_scheme not in _valid_bulk:
        raise ValueError(
            f"Unknown bulk_scheme {land_config.bulk_scheme!r}; "
            f"expected one of {_valid_bulk}."
        )
    rho = forcing.rho_lowest
    if land_config.bulk_scheme in ("most", "coare3", "large_yeager"):
        tau_x, tau_y, shflx, lhflx, _ = compute_most_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_surface, q_sfc, rho,
            z_ref=land_config.z_ref,
            z0_init=z0,
            scheme=land_config.bulk_scheme,
            n_iter=land_config.bulk_n_iter,
            L_latent=L_eff,
            max_exchange_coeff=LAND_MAX_EXCHANGE_COEFF,
        )
    else:
        tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_surface, q_sfc, rho, wind_speed,
            land_config.Cd_land, land_config.Ch_land,
            L_latent=L_eff,
        )

    # Cold-start condensation floor (issue #730): bound the spurious (negative)
    # condensation shock BEFORE it enters G_soil / the returned demand; evaporation
    # (positive lhflx) stays free.  See LAND_CONDENSATION_FLOOR_W.
    if LAND_CONDENSATION_FLOOR_W is not None:
        lhflx = jnp.maximum(lhflx, LAND_CONDENSATION_FLOOR_W)

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
        # Snow-free base = the per-cell ``albedo_land`` (CLM PFT / soil-colour map or
        # the trainable per-PFT albedo), NOT the latitude-band vegetation albedo —
        # otherwise the calibrated per-cell / per-PFT albedo (and its gradient) is
        # dropped whenever the snow feedback is on (bright deserts + trainable pft_alb).
        alpha = compute_land_albedo(
            lat, snow_effective, snow_age, land_config.land_albedo,
            base_albedo=jnp.broadcast_to(jnp.asarray(albedo_land), T_surface.shape))
    else:
        alpha = jnp.broadcast_to(jnp.asarray(albedo_land), T_surface.shape)

    # --- Radiation ---
    sw_net, lw_net, lw_up = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_surface, alpha, emissivity)

    # --- Ground heat flux (residual of surface energy balance) ---
    G_soil = sw_net + lw_net - shflx - lhflx

    # --- Semi-implicit surface conductance lambda = -dG_soil/dT_sfc (>= 0) for the
    # soil-thermal Robin BC (removes the explicit-coupling large-dt/thin-layer/stiff-
    # surface instability that otherwise diverges to NaN).  Longwave slope is analytic
    # (4 eps sigma T^3); the sensible + latent slopes are one-sided finite differences
    # re-evaluating the SAME bulk-flux scheme at T_surface + _SURFACE_LIN_DT_K.  All
    # slopes are clamped >= 0 so lambda can only ADD damping.  The latent slope uses
    # the DEMAND lhflx (the caller's post-hoc water-limiting is not visible here); when
    # evaporation is supply-limited that slightly OVER-damps, which errs toward
    # stability (the guard's purpose) and never destabilises.  Consumed by
    # solve_soil_thermal(surface_conductance=...).
    T_sfc_lin = T_surface + _SURFACE_LIN_DT_K
    q_sfc_lin = beta_effective * jnp.where(
        has_snow,
        saturation_mixing_ratio_ice(T_sfc_lin, forcing.p_surface),
        saturation_mixing_ratio(T_sfc_lin, forcing.p_surface),
    )
    if land_config.bulk_scheme in ("most", "coare3", "large_yeager"):
        _, _, shflx_lin, lhflx_lin, _ = compute_most_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_sfc_lin, q_sfc_lin, rho,
            z_ref=land_config.z_ref, z0_init=z0,
            scheme=land_config.bulk_scheme, n_iter=land_config.bulk_n_iter,
            L_latent=L_eff, max_exchange_coeff=LAND_MAX_EXCHANGE_COEFF,
        )
    else:
        _, _, shflx_lin, lhflx_lin = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_sfc_lin, q_sfc_lin, rho, wind_speed,
            land_config.Cd_land, land_config.Ch_land, L_latent=L_eff,
        )
    if LAND_CONDENSATION_FLOOR_W is not None:
        lhflx_lin = jnp.maximum(lhflx_lin, LAND_CONDENSATION_FLOOR_W)
    _emis_b = jnp.broadcast_to(jnp.asarray(emissivity), T_surface.shape)
    lambda_lw = 4.0 * _emis_b * constants.sigma_sb * T_surface ** 3
    lambda_sh = jnp.maximum((shflx_lin - shflx) / _SURFACE_LIN_DT_K, 0.0)
    lambda_lh = jnp.maximum((lhflx_lin - lhflx) / _SURFACE_LIN_DT_K, 0.0)
    surface_conductance = lambda_lw + lambda_sh + lambda_lh

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
        sif=sif_farq,
        stomatal_ratio=stomatal_ratio,
        surface_conductance=surface_conductance,
        # Canopy-specific diagnostics left as None
    )
