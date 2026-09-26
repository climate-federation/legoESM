"""Sea-ice shortwave albedo and SW penetration.

Three schemes share a common ``compute_ice_sw`` entry point:

- ``"constant"`` — single broadband albedo, optional fixed transmission.
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
_T_MELT_WIDTH_K = 1.0            # melt-transition width below T_melt [K]

# Bulk shortwave extinction coefficient of sea ice for the penetrating (i0)
# fraction, used in the Beer-Lambert decay exp(-kappa_ice*h_ice).  ~1.4 1/m is
# the CICE bare-ice interior value (Briegleb & Light 2007, following Grenfell &
# Maykut 1977): the i0 fraction that enters the ice column is attenuated by the
# ice above the base so that thick ice transmits ~nothing to the ocean.
_KAPPA_ICE_M_INV = 1.4          # sea-ice SW extinction coefficient [1/m]


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
    albedo_ocean: float = constants.alpha_ocean_broadband,
) -> jnp.ndarray:
    """Temperature- and thickness-dependent broadband ice albedo.

    Smooth interpolation between two regimes:
        * Cold (T_sfc ≤ T_melt − T_width): albedo = α_cold
        * Melting (T_sfc ≥ T_melt):        albedo = α_melt
    multiplied by a thickness ramp ``(h / h_ramp)^0.5`` saturating
    at 1 once ``h ≥ h_ramp`` — thin growing ice has a smaller
    albedo because some shortwave penetrates the thin column — and
    floored at the open-water albedo so vanishing ice asymptotes to
    ocean, not black.

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
        Upper edge and width of the smooth melt transition [K]; the
        ramp spans ``[T_melt − T_width, T_melt]``.
    albedo_ocean : float
        Open-water albedo used as the thin-ice floor.
    """
    # Sign/formula walk: the skin temperature is clamped at T_melt elsewhere, so
    # the melt ramp must reach 1.0 *at* T_melt (not 0.5).  Ramp linearly over
    # [T_melt − T_width, T_melt]: melt_fraction = 0 at the cold edge, = 1 at
    # T_melt, monotonically increasing with T_sfc (warmer -> more melting ->
    # lower albedo).
    melt_fraction = jnp.clip(1.0 + (T_sfc - T_melt) / T_width, 0.0, 1.0)
    alpha_regime = (
        (1.0 - melt_fraction) * albedo_cold_bare
        + melt_fraction * albedo_melt_bare
    )
    thickness_factor = jnp.clip(
        jnp.sqrt(jnp.maximum(h_ice, _H_SQRT_FLOOR) / jnp.maximum(h_ramp, 1e-6)),
        0.0,
        1.0,
    )
    # Sign/formula walk: the sqrt ramp -> 0 as h_ice -> 0 drives the albedo below
    # the open-water value (1 mm ice would be darker than the ocean it sits in).
    # Floor at albedo_ocean so thin ice asymptotes to open water, never darker.
    return jnp.maximum(alpha_regime * thickness_factor, albedo_ocean)


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
    alpha_underlying_vis: jnp.ndarray,
    alpha_underlying_nir: jnp.ndarray,
    h_snow_sat: float = _H_SNOW_SAT_M,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate from underlying ice to optically thick snow per band.

    The existing empirical depth ramp must approach the ice albedo as snow
    vanishes, not zero (black). Coverage is applied separately by the caller.
    This repairs the surrogate's thin-snow darkening without changing its
    band endpoints or ramp lengths; it is not the SI3 exponential snow fit.
    """
    ramp = jnp.clip(h_snow / jnp.maximum(h_snow_sat, 1e-6), 0.0, 1.0)
    alpha_vis = (1.0 - melt_fraction) * alpha_cold_vis + melt_fraction * alpha_melt_vis
    alpha_nir = (1.0 - melt_fraction) * alpha_cold_nir + melt_fraction * alpha_melt_nir
    return (alpha_underlying_vis + (alpha_vis - alpha_underlying_vis) * ramp,
            alpha_underlying_nir + (alpha_nir - alpha_underlying_nir) * ramp)


def _band_albedo_bare_ice(
    melt_fraction: jnp.ndarray,
    h_ice: jnp.ndarray,
    *,
    alpha_cold_vis: float,
    alpha_melt_vis: float,
    alpha_cold_nir: float,
    alpha_melt_nir: float,
    h_sat: float = _H_BARE_SAT_M,
    albedo_ocean: float = constants.alpha_ocean_broadband,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Bare-ice two-band albedo with thickness ramp.

    Floored per band at the open-water albedo: the sqrt ramp -> 0 as
    h_ice -> 0 would otherwise make vanishing ice darker than the ocean.
    """
    ramp = jnp.clip(
        jnp.sqrt(jnp.maximum(h_ice, _H_SQRT_FLOOR) / jnp.maximum(h_sat, 1e-6)),
        0.0,
        1.0,
    )
    alpha_vis = (1.0 - melt_fraction) * alpha_cold_vis + melt_fraction * alpha_melt_vis
    alpha_nir = (1.0 - melt_fraction) * alpha_cold_nir + melt_fraction * alpha_melt_nir
    # Sign/formula walk: floor each band at albedo_ocean so thin ice asymptotes
    # to open water (both bands), never below the ocean it replaces.
    return (
        jnp.maximum(alpha_vis * ramp, albedo_ocean),
        jnp.maximum(alpha_nir * ramp, albedo_ocean),
    )


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
    The penetrating (``i0``) fraction is attenuated by an in-ice Beer-Lambert
    factor ``exp(-κ_ice·h_ice)`` (``κ_ice = _KAPPA_ICE_M_INV``, CICE bare-ice
    value) so that thick bare ice transmits ~nothing to the ocean, while thin /
    marginal ice still lets a physical fraction through.  Column energy is
    conserved by ``compute_ice_sw`` (``absorbed = (1-α)·F - penetrated`` with
    ``penetrated`` capped at the column input).  Remaining surrogate
    simplification vs full CICE6: ``i0`` is scaled off *incident* rather than
    *net-absorbed* SW, a small partition difference now that the dominant thick-
    ice over-transmission is removed.  Opt-in scheme; default ice SW is
    ``constant``.
    """
    # Sign/formula walk (matches maykut_untersteiner_albedo): skin T is clamped
    # at T_melt elsewhere, so ramp melt_fraction over [T_melt − T_width, T_melt]
    # to reach 1.0 *at* T_melt (a 0.5-centred ramp would top out at 0.5, so the
    # melting albedos were never fully reached).
    melt_fraction = jnp.clip(1.0 + (T_sfc - T_melt) / T_width, 0.0, 1.0)

    ice_vis, ice_nir = _band_albedo_bare_ice(
        melt_fraction, h_ice,
        alpha_cold_vis=constants.alpha_ice_cold_vis,
        alpha_melt_vis=constants.alpha_ice_melt_vis,
        alpha_cold_nir=constants.alpha_ice_cold_nir,
        alpha_melt_nir=constants.alpha_ice_melt_nir,
    )
    snow_vis, snow_nir = _band_albedo_snow(
        melt_fraction, h_snow,
        alpha_underlying_vis=ice_vis, alpha_underlying_nir=ice_nir,
        alpha_cold_vis=constants.alpha_snow_cold_vis,
        alpha_melt_vis=constants.alpha_snow_melt_vis,
        alpha_cold_nir=constants.alpha_snow_cold_nir,
        alpha_melt_nir=constants.alpha_snow_melt_nir,
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
    #
    # Sign/formula walk (Beer-Lambert): the i0 fraction that enters the ice
    # base decays as exp(-kappa_ice*h_ice) through the ice column, so only a
    # thin/marginal ice layer transmits appreciable SW to the ocean.  Without
    # this factor a fixed ~i0 fraction of *incident* SW was dumped through
    # arbitrarily thick ice into the ocean.  h_ice >= 0 => attenuation in (0,1],
    # so the transmitted fraction is strictly reduced, never amplified.
    attenuation = jnp.exp(-_KAPPA_ICE_M_INV * jnp.maximum(h_ice, 0.0))
    transmittance = (
        f_vis * f_bare * i0_vis + (1.0 - f_vis) * f_bare * i0_nir
    ) * attenuation
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
    sw_transmittance_const: float = 0.0,
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
    sw_transmittance_const : float
        Constant-scheme SW transmittance through ice+snow to the ocean
        (fraction of INCIDENT ``sw_down``; bounded by the non-reflected
        column input so ``reflected + absorbed + penetrated == sw_down``
        exactly).  Default 0.0 = bit-identical legacy.  Ignored by the
        other schemes (delta_eddington computes its own transmittance;
        maykut_untersteiner remains a no-penetration albedo fit).

    Returns
    -------
    result : :class:`IceSWResult`
    """
    if scheme == "constant":
        alpha = jnp.full_like(T_sfc, albedo_const)
        # SW budget (positive into the column): reflected + absorbed +
        # penetrated == sw_down exactly — the penetrated flux is DEBITED from
        # the surface-absorbed part (same closure as delta_eddington below),
        # so transmitting SW to the ocean costs the ice energy balance.
        net_into_column = (1.0 - alpha) * sw_down
        penetrated = jnp.minimum(
            sw_transmittance_const * sw_down, net_into_column)
        absorbed = net_into_column - penetrated
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
