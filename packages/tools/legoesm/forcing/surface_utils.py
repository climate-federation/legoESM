"""Surface blending utilities for AMIP-style experiments.

Provides common surface-property computations shared across scripts:
- Sea-ice / ocean temperature blending
- Sea-ice / ocean albedo and emissivity blending
- Column AOD distribution to model layers
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants


def blend_surface_temperature(
    sst: jnp.ndarray,
    sic: jnp.ndarray,
    T_ice: float,
) -> jnp.ndarray:
    """Blend ocean SST and sea-ice temperature by ice concentration.

    Parameters
    ----------
    sst : array
        Sea surface temperature [K].
    sic : array
        Sea-ice concentration [0, 1].
    T_ice : float
        Sea-ice temperature [K].

    Returns
    -------
    T_sfc : array
        Blended surface temperature [K].
    """
    return sic * T_ice + (1.0 - sic) * sst


def land_lapse_adjusted_surface_temperature(
    T_sfc: jnp.ndarray,
    f_land: jnp.ndarray,
    z_sfc: jnp.ndarray,
    lapse_K_per_m: float,
) -> jnp.ndarray:
    """Lower the LAND fraction of a surface-temperature anchor by a lapse rate.

    AMIP SST loaders fill land cells with the nearest-ocean SST (a sea-level
    temperature).  Anchoring surface fluxes / radiation to that value at an
    elevated cell overheats the surface by ``lapse * z`` (e.g. ~+13 K at 2 km
    for 6.5 K/km), driving spurious surface fluxes and convection over
    highlands.  This applies the standard-atmosphere correction on the land
    fraction only::

        T_eff = T_sfc - f_land * lapse_K_per_m * max(z_sfc, 0)

    Sign convention: ``z_sfc`` positive up [m]; ``lapse_K_per_m > 0`` cools
    with height.  The ocean fraction (``f_land = 0``) is unchanged; negative
    elevations (below-sea-level basins) are clipped to zero rather than
    warmed.

    Parameters
    ----------
    T_sfc : array
        Surface-temperature anchor [K] (ocean/ice blended, land = nearest-
        ocean fill).
    f_land : array
        Land fraction in [0, 1], same shape as ``T_sfc``.
    z_sfc : array
        Surface elevation [m] (e.g. ``phis / g``), same shape.
    lapse_K_per_m : float
        Lapse rate [K/m]; 0 disables (returns ``T_sfc`` unchanged up to
        floating-point identity).

    Returns
    -------
    T_eff : array
        Lapse-adjusted surface temperature [K].
    """
    return T_sfc - f_land * lapse_K_per_m * jnp.maximum(z_sfc, 0.0)


def blend_surface_property(
    sic: jnp.ndarray,
    value_ice: float | jnp.ndarray,
    value_ocean: float | jnp.ndarray,
) -> jnp.ndarray:
    """Blend an ice and ocean surface property by ice concentration.

    Useful for albedo, emissivity, or any linearly-blended quantity.

    Parameters
    ----------
    sic : array
        Sea-ice concentration [0, 1].
    value_ice : float or array
        Property value over ice.
    value_ocean : float or array
        Property value over ocean.

    Returns
    -------
    blended : array
        Ice-concentration-weighted blend.
    """
    return sic * value_ice + (1.0 - sic) * value_ocean


def surface_temperature_for_lw_boundary(
    radiation: str,
    *,
    T_rad: jnp.ndarray,
    lw_up: jnp.ndarray,
) -> jnp.ndarray:
    """Surface temperature to feed a scheme's longwave boundary.

    A scheme reproduces the surface's true upward LW flux ``LW_out`` only if it
    is fed the temperature consistent with its OWN emissivity convention:

    * ``rrtmgp`` / ``rrtmg`` — use the radiative-equivalent ``T_rad`` paired with
      the tile-blended ``eps_col`` (``eps_col*sigma*T_rad^4 = LW_emit``); the
      ``(1-eps_col)*La`` reflection term then completes ``LW_out``.
    * ``gray`` / ``none`` — gray emits as a BLACK surface (``eps = 1``) and
      cannot honour ``eps_col``, so feeding it ``T_rad`` would emit
      ``sigma*T_rad^4 = LW_emit/eps_col`` and OVERSTATE the flux by ``1/eps_col``.
      Instead derive a black-surface BRIGHTNESS temperature from the complete
      upward flux, ``T_bb = (lw_up/sigma)^0.25``, so ``sigma*T_bb^4 = LW_out``
      exactly.

    Parameters
    ----------
    radiation : str
        Active radiation scheme.
    T_rad : array
        Radiative-equivalent skin temperature (``eps_col*sigma*T_rad^4 =
        LW_emit``).
    lw_up : array
        Total upward LW flux at the surface (``LW_out``) [W/m^2].

    Returns
    -------
    T : array
        Surface temperature to hand the LW boundary.
    """
    if radiation in ("rrtmgp", "rrtmg"):
        return T_rad
    # gray / none: black-surface brightness temperature from the full upward flux.
    return (jnp.maximum(lw_up, 1.0e-6) / constants.sigma_sb) ** 0.25


def surface_emissivity_for_lw_inversion(
    radiation: str,
    *,
    dynamic_emissivity: jnp.ndarray | None,
    static_sfc_emissivity: float | jnp.ndarray,
) -> float | jnp.ndarray:
    """Surface emissivity to invert a held ``lw_net`` back to gross ``lw_down``.

    The coupled drivers reconstruct gross ``lw_down`` from the held net surface
    longwave via ``lw_down = (lw_net + eps*sigma*T^4) / eps``.  For that round
    trip to be exact, ``eps`` MUST equal the emissivity the radiation scheme
    actually EMITTED the boundary with — otherwise a persistent O(1 W/m^2)
    surface-energy bias leaks in (the emissivity mismatch is independent of the
    skin temperature used).  This returns that matching emissivity:

    * ``rrtmgp`` / ``rrtmg`` WITH the dynamic surface-radiation feedback — the
      tile-blended ``eps_col`` (``dynamic_emissivity``, not ``None``), exactly
      what ``solve_columns(sfc_emissivity=emis_col)`` used.
    * ``rrtmgp`` / ``rrtmg`` WITHOUT the feedback (or before the first coupler
      response) — the static config surface emissivity it emitted with.
    * ``gray`` / ``none`` — an idealized BLACK surface (``eps = 1.0``).  Gray
      radiation keeps ``GrayRadiationConfig.sfc_emissivity = 1.0`` and the
      dynamic ``emis_col`` is intentionally NOT threaded into it, so the
      reconstruction must also use ``1.0`` regardless of any config emissivity.

    Parameters
    ----------
    radiation : str
        Active radiation scheme (``"rrtmgp"``, ``"rrtmg"``, ``"gray"``,
        ``"none"``, ...).
    dynamic_emissivity : array or None
        The coupler's tile-blended surface emissivity for this segment, or
        ``None`` when the dynamic feedback is off / unavailable.
    static_sfc_emissivity : float or array
        The static config surface emissivity the scheme falls back to.

    Returns
    -------
    eps : float or array
        Emissivity to use for the ``lw_net`` -> ``lw_down`` inversion.
    """
    if radiation in ("rrtmgp", "rrtmg"):
        if dynamic_emissivity is not None:
            return dynamic_emissivity
        return static_sfc_emissivity
    # gray / none: idealized black surface (emit with eps = 1.0).
    return 1.0
def snow_fraction(
    T_low: jnp.ndarray,
    T_freeze: float,
    transition_center_offset_K: float = 2.0,
    transition_width_K: float = 4.0,
) -> jnp.ndarray:
    """Smooth rain/snow partition fraction (Wigmosta 1994 / Dai 2008).

    Linear ramp from 0 (all rain) at
    ``T_low = T_freeze + transition_center_offset_K`` down to 1 (all snow)
    at ``T_low = T_freeze + transition_center_offset_K - transition_width_K``.
    Replaces the hard step ``where(T_low < T_freeze, 1, 0)``, which zeroes
    ``d(snow)/d(T_low)`` on training/DA paths and miscounts mixed-phase
    precipitation in the 0–4 °C band. Single source of truth shared by the
    coupled and earth-system drivers so the two cannot silently diverge.

    Parameters
    ----------
    T_low : array
        Lowest-model-level air temperature [K].
    T_freeze : float
        Freezing point [K] (pass ``constants.T_freeze``).
    transition_center_offset_K : float
        Upper edge of the mixed-phase band above freezing [K] (default 2.0).
    transition_width_K : float
        Total width of the linear ramp [K] (default 4.0).

    Returns
    -------
    snow_frac : array
        Snow fraction in [0, 1].
    """
    return jnp.clip(
        (T_freeze + transition_center_offset_K - T_low) / transition_width_K,
        0.0,
        1.0,
    )


def distribute_column_aod_to_layers(
    aod_col: jnp.ndarray,
    p_half_col: jnp.ndarray,
) -> jnp.ndarray:
    """Distribute column aerosol optical depth to layers by pressure thickness.

    Parameters
    ----------
    aod_col : array, shape (ncol,)
        Total column AOD per column.
    p_half_col : array, shape (ncol, nlev+1)
        Half-level pressures.

    Returns
    -------
    aod_layers : array, shape (ncol, nlev)
        Layer-distributed AOD.
    """
    dp = jnp.clip(p_half_col[:, 1:] - p_half_col[:, :-1], 1.0e-12, None)
    w = dp / jnp.sum(dp, axis=1, keepdims=True)
    return jnp.clip(aod_col, 0.0, None)[:, None] * w
