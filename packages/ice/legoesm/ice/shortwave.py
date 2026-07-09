"""Sea-ice shortwave albedo and SW penetration.

Three schemes share a common ``compute_ice_sw`` entry point:

- ``"constant"`` — single broadband albedo, no penetration.
- ``"maykut_untersteiner"`` — thin-ice α(T_sfc, h_ice) ramp.
- ``"delta_eddington"`` — two-band (VIS + NIR) Briegleb & Light
  (2007) albedo with snow + pond modifications and an interior
  penetration fraction ``i0_vis`` for the visible band.

The Delta-Eddington implementation here is a tabulated /
empirical surrogate for the full radiative-transfer table used in
CICE6.  It captures the leading dependencies (snow vs bare,
melting vs cold, snow depth, pond fraction) with smooth
interpolations so that the gradient through the SW kernel
remains well-defined for adjoint applications.  The exact CICE6
delta-Eddington lookup table is a future extension if higher
fidelity is needed.

All functions are JAX-compatible.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants

# AD-safety floor for the thickness inside the sqrt thickness-ramps.  sqrt'(0) is
# infinite, so differentiating sqrt(h_ice) at the open-water limit h_ice == 0
# (every ice-edge cell) yields an inf/NaN gradient even though the forward value
# is finite.  Flooring the sqrt argument at this negligible thickness (0.1 nm)
# keeps the forward ramp bit-identical for any physical h_ice while giving a
# finite (zero, via the max) gradient at h_ice == 0.
_H_SQRT_FLOOR = 1e-12  # m

# --- albedo-scheme default coefficients (Maykut & Untersteiner 1971; delta-
# Eddington Briegleb & Light 2007) ---------------------------------------------
# Published scheme constants for the internal band-albedo construction. The
# user-facing tunable broadband albedo is SeaIceConfig.albedo_ice (spec tier 1);
# these internal coefficients are fixed scheme defaults.
_ALBEDO_COLD_BARE_DEFAULT = 0.7   # cold bare-ice broadband albedo (MU 1971)
_ALBEDO_MELT_BARE_DEFAULT = 0.5   # melting bare-ice broadband albedo (MU 1971)
_H_SNOW_SAT_M = 0.05              # snow depth saturating the snow-albedo ramp [m]
_H_POND_SAT_M = 0.3               # pond depth saturating the pond-albedo ramp [m]
_H_RAMP_M = 0.5                   # thickness saturating the thin-ice albedo ramp [m]
_H_BARE_SAT_M = 0.5               # bare-ice thickness ramp saturation [m]
_F_VIS = 0.52                     # visible-band fraction of incident SW (Briegleb-Light)
_H_SNOW_MASK_M = 0.02             # snow depth fully masking bare-ice albedo [m]
_T_MELT_WIDTH_K = 1.0            # smooth melt-transition half-width [K]


# ==============================================================================
# Result container
# ==============================================================================

class IceSWResult(NamedTuple):
    """Outputs of the ice shortwave kernel.

    ``albedo_eff`` is the area-weighted broadband albedo for the
    ice tile (mixes snow / bare / pond contributions).
    ``sw_penetrated`` is the shortwave that goes *through* the ice
    and warms the ocean column below — non-zero only for the
    delta-Eddington scheme.
    """
    albedo_eff: jnp.ndarray            # [0-1]
    sw_absorbed_surface: jnp.ndarray   # [W/m²]  (heats the surface skin)
    sw_penetrated: jnp.ndarray         # [W/m²]  (penetrates to ocean)


# ==============================================================================
# Maykut-Untersteiner (CCSM3) thin-ice albedo
# ==============================================================================

def maykut_untersteiner_albedo(
    T_sfc: jnp.ndarray,
    h_ice: jnp.ndarray,
    *,
    albedo_cold_bare: float = _ALBEDO_COLD_BARE_DEFAULT,
    albedo_melt_bare: float = _ALBEDO_MELT_BARE_DEFAULT,
    h_ramp: float = _H_RAMP_M,
    T_melt: float = constants.T_freeze,
    T_width: float = _T_MELT_WIDTH_K,
) -> jnp.ndarray:
    """Temperature- and thickness-dependent broadband ice albedo.

    Smooth interpolation between two regimes:
        * Cold (T_sfc < T_melt − T_width/2): albedo = α_cold
        * Melting (T_sfc ≥ T_melt):           albedo = α_melt
    multiplied by a thickness ramp ``(h / h_ramp)^0.5`` saturating
    at 1 once ``h ≥ h_ramp`` — thin growing ice has a smaller
    albedo because some shortwave penetrates the thin column.

    Parameters
    ----------
    T_sfc : array
        Surface temperature [K].
    h_ice : array
        Ice thickness [m].
    albedo_cold_bare, albedo_melt_bare : float
        Saturated thick-ice albedos in the two regimes.
    h_ramp : float
        Thickness above which the thickness factor saturates [m].
    T_melt, T_width : float
        Centre and width of the smooth melt transition [K].
    """
    melt_fraction = jnp.clip(0.5 + (T_sfc - T_melt) / T_width, 0.0, 1.0)
    alpha_regime = (
        (1.0 - melt_fraction) * albedo_cold_bare
        + melt_fraction * albedo_melt_bare
    )
    thickness_factor = jnp.clip(
        jnp.sqrt(jnp.maximum(h_ice, _H_SQRT_FLOOR) / jnp.maximum(h_ramp, 1e-6)),
        0.0,
        1.0,
    )
    return alpha_regime * thickness_factor


# ==============================================================================
# Delta-Eddington two-band albedo (Briegleb & Light 2007 surrogate)
# ==============================================================================

def _band_albedo_snow(
    melt_fraction: jnp.ndarray,
    h_snow: jnp.ndarray,
    *,
    alpha_cold_vis: float,
    alpha_melt_vis: float,
    alpha_cold_nir: float,
    alpha_melt_nir: float,
    h_snow_sat: float = _H_SNOW_SAT_M,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Snow-surface albedo per band, with a thin-snow ramp."""
    ramp = jnp.clip(h_snow / jnp.maximum(h_snow_sat, 1e-6), 0.0, 1.0)
    alpha_vis = (1.0 - melt_fraction) * alpha_cold_vis + melt_fraction * alpha_melt_vis
    alpha_nir = (1.0 - melt_fraction) * alpha_cold_nir + melt_fraction * alpha_melt_nir
    return alpha_vis * ramp, alpha_nir * ramp


def _band_albedo_bare_ice(
    melt_fraction: jnp.ndarray,
    h_ice: jnp.ndarray,
    *,
    alpha_cold_vis: float,
    alpha_melt_vis: float,
    alpha_cold_nir: float,
    alpha_melt_nir: float,
    h_sat: float = _H_BARE_SAT_M,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Bare-ice two-band albedo with thickness ramp."""
    ramp = jnp.clip(
        jnp.sqrt(jnp.maximum(h_ice, _H_SQRT_FLOOR) / jnp.maximum(h_sat, 1e-6)),
        0.0,
        1.0,
    )
    alpha_vis = (1.0 - melt_fraction) * alpha_cold_vis + melt_fraction * alpha_melt_vis
    alpha_nir = (1.0 - melt_fraction) * alpha_cold_nir + melt_fraction * alpha_melt_nir
    return alpha_vis * ramp, alpha_nir * ramp


def _band_albedo_pond(
    h_pond: jnp.ndarray,
    alpha_ice_vis: jnp.ndarray,
    alpha_ice_nir: jnp.ndarray,
    *,
    alpha_deep_vis: float,
    alpha_deep_nir: float,
    h_pond_sat: float = _H_POND_SAT_M,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Melt-pond two-band albedo — DECREASES with depth.

    A shallow pond looks like the wet ice it sits on; deepening water
    absorbs more in the column and darkens toward the deep-pond floor
    ``alpha_deep_*`` (Ebert & Curry 1993; Briegleb & Light 2007).  The
    old form ramped UP with depth (a zero-depth pond went perfectly
    black, a deep pond was brightest) — the sign was inverted.
    """
    # depth 0 -> ice albedo (fresh pond ~ wet ice); depth >= h_sat -> deep floor.
    # Clamp the deep floor to the underlying ice albedo so thin (dark) ice can't
    # brighten as the pond deepens — monotone non-increasing in depth (codex).
    decay = jnp.clip(h_pond / jnp.maximum(h_pond_sat, 1e-6), 0.0, 1.0)
    deep_vis = jnp.minimum(alpha_deep_vis, alpha_ice_vis)
    deep_nir = jnp.minimum(alpha_deep_nir, alpha_ice_nir)
    return (alpha_ice_vis + (deep_vis - alpha_ice_vis) * decay,
            alpha_ice_nir + (deep_nir - alpha_ice_nir) * decay)


def delta_eddington_albedo(
    T_sfc: jnp.ndarray,
    h_ice: jnp.ndarray,
    h_snow: jnp.ndarray,
    pond_area: jnp.ndarray,
    pond_depth: jnp.ndarray,
    *,
    f_vis: float = _F_VIS,
    T_melt: float = constants.T_freeze,
    T_width: float = _T_MELT_WIDTH_K,
    i0_vis: float = constants.i0_vis,
    i0_nir: float = constants.i0_nir,
    h_snow_mask: float = _H_SNOW_MASK_M,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Two-band albedo following the Briegleb-Light surrogate.

    The visible (≈ 0.2–0.7 μm) and near-IR (≈ 0.7–5 μm) bands are
    treated separately.  Snow on top of ice masks the bare-ice
    optics; ponds lower the visible band albedo strongly.  ``f_vis``
    is the assumed visible fraction of incident solar irradiance.

    Returns
    -------
    albedo_total : array
        Broadband ice-tile albedo (after band-averaging with
        ``f_vis`` weighting).
    transmittance_vis : array
        Fraction of incident shortwave that penetrates *through*
        the bare-ice band (only meaningful where there is no snow
        / pond cover above).  Multiplied by the *non-pond* and
        *non-snow* fraction of the cell and ``i0_vis`` to get the
        net flux into the ocean.

    Notes
    -----
    Surrogate-fidelity limitation (disclosed): ``i0_vis`` is applied as a
    fraction of *incident* SW with no in-ice Beer-Lambert attenuation,
    whereas full CICE6 delta-Eddington defines ``I0`` as a fraction of the
    *net-absorbed* SW that then decays as ``exp(-κ_ice·h_ice)`` through the
    ice.  Column energy is still conserved (``compute_ice_sw`` sets
    ``absorbed = (1-α)·F - penetrated``), but the surface/ocean *partition*
    is biased toward the ocean for thick bare ice.  A faithful upgrade
    needs the CICE ``I0`` convention + an ice extinction coefficient
    ``κ_ice`` (reference value required — not guessed) + validation.
    Opt-in scheme; default ice SW is ``constant``.
    """
    melt_fraction = jnp.clip(0.5 + (T_sfc - T_melt) / T_width, 0.0, 1.0)

    snow_vis, snow_nir = _band_albedo_snow(
        melt_fraction, h_snow,
        alpha_cold_vis=constants.alpha_snow_cold_vis,
        alpha_melt_vis=constants.alpha_snow_melt_vis,
        alpha_cold_nir=constants.alpha_snow_cold_nir,
        alpha_melt_nir=constants.alpha_snow_melt_nir,
    )
    ice_vis, ice_nir = _band_albedo_bare_ice(
        melt_fraction, h_ice,
        alpha_cold_vis=constants.alpha_ice_cold_vis,
        alpha_melt_vis=constants.alpha_ice_melt_vis,
        alpha_cold_nir=constants.alpha_ice_cold_nir,
        alpha_melt_nir=constants.alpha_ice_melt_nir,
    )
    pond_vis, pond_nir = _band_albedo_pond(
        pond_depth, ice_vis, ice_nir,
        alpha_deep_vis=constants.alpha_pond_max_vis,
        alpha_deep_nir=constants.alpha_pond_max_nir,
    )

    # Coverage fractions (snow on top wins; ponds occupy a fraction of
    # the bare-ice surface).  ``h_snow_mask`` is the snow depth [m] at which
    # the surface is fully snow-covered (lifted from a bare 0.02 literal — no
    # magic numbers in the JAX albedo body; CLAUDE.md).
    f_snow = jnp.clip(jnp.minimum(h_snow / jnp.maximum(h_snow_mask, 1e-6), 1.0), 0.0, 1.0)
    f_bare_after_snow = 1.0 - f_snow
    f_pond_on_bare = jnp.clip(pond_area, 0.0, 1.0)
    f_pond = f_bare_after_snow * f_pond_on_bare
    f_bare = f_bare_after_snow * (1.0 - f_pond_on_bare)

    alpha_vis = f_snow * snow_vis + f_bare * ice_vis + f_pond * pond_vis
    alpha_nir = f_snow * snow_nir + f_bare * ice_nir + f_pond * pond_nir
    alpha_total = f_vis * alpha_vis + (1.0 - f_vis) * alpha_nir

    # Penetration fraction: only the visible band penetrates the
    # *bare-ice* portion that is not snow-covered.  Pond surfaces
    # are also transparent in VIS but pond water re-absorbs, so we
    # approximate this with the bare ice transparency only.
    transmittance = f_vis * f_bare * i0_vis + (1.0 - f_vis) * f_bare * i0_nir
    return alpha_total, transmittance


# ==============================================================================
# Unified dispatch entry point
# ==============================================================================

def compute_ice_sw(
    sw_down: jnp.ndarray,
    T_sfc: jnp.ndarray,
    h_ice: jnp.ndarray,
    h_snow: jnp.ndarray,
    pond_area: jnp.ndarray,
    pond_depth: jnp.ndarray,
    *,
    scheme: str,
    albedo_const: float,
) -> IceSWResult:
    """Compute albedo and absorbed / penetrated SW for ice tile.

    Parameters
    ----------
    sw_down : array
        Downward shortwave at the surface [W/m²].
    T_sfc : array
        Surface skin temperature [K].
    h_ice, h_snow : array
        Ice and snow thickness [m].
    pond_area, pond_depth : array
        Pond fraction (relative to ice area) and mean depth [m].
        Set to zero when ``MeltPondConfig.enabled`` is False.
    scheme : {"constant", "maykut_untersteiner", "delta_eddington"}
    albedo_const : float
        Fallback albedo for the constant scheme.

    Returns
    -------
    result : :class:`IceSWResult`
    """
    if scheme == "constant":
        alpha = jnp.full_like(T_sfc, albedo_const)
        absorbed = (1.0 - alpha) * sw_down
        penetrated = jnp.zeros_like(sw_down)
    elif scheme == "maykut_untersteiner":
        alpha = maykut_untersteiner_albedo(T_sfc, h_ice)
        absorbed = (1.0 - alpha) * sw_down
        penetrated = jnp.zeros_like(sw_down)
    elif scheme == "delta_eddington":
        alpha, transmittance = delta_eddington_albedo(
            T_sfc, h_ice, h_snow, pond_area, pond_depth,
        )
        # SW energy balance (positive into the column): the SW that is not
        # reflected enters the column and is partitioned into a surface-absorbed
        # part and a part that penetrates through to the ocean below.
        #   reflected + absorbed + penetrated == sw_down  (exactly).
        # ``transmittance`` and ``(1 - alpha)`` come from SEPARATE fits, so the
        # raw ``transmittance * sw_down`` can exceed ``net_into_column``; the
        # previous code clamped ``absorbed`` at 0 but left ``penetrated``
        # unbounded, so reflected + absorbed + penetrated could EXCEED sw_down
        # (energy created).  Bound the penetrated flux by the column input and
        # take the surface-absorbed part as the remainder so the balance closes
        # and both parts stay non-negative (finding #8).
        net_into_column = (1.0 - alpha) * sw_down
        penetrated = jnp.minimum(transmittance * sw_down, net_into_column)
        absorbed = net_into_column - penetrated
    else:
        raise ValueError(
            f"compute_ice_sw: unknown scheme {scheme!r}.  Expected one of "
            "'constant', 'maykut_untersteiner', 'delta_eddington'."
        )
    return IceSWResult(
        albedo_eff=alpha,
        sw_absorbed_surface=absorbed,
        sw_penetrated=penetrated,
    )
