"""Lightweight saturation thermodynamics for legoESM.

This module provides `saturation_mixing_ratio` and
`saturation_mixing_ratio_ice` with *no* dependency on
``atmosphere.physics`` so that ``land/``, ``ice/``, ``ocean/``, and
``coupler/`` modules can import them without pulling in the full
atmosphere physics package.

All operations are pure JAX and compatible with jit, grad, vmap, scan.

Conventions — water-vapor mass variables
----------------------------------------
This module returns the **mixing ratio** ``r_sat = ε e_sat / (p - e_sat)``
(mass of water vapor per unit mass of *dry* air).  Throughout the
``atmosphere/physics`` source tree the prognostic field is named
``q_v`` and many docstrings call it "specific humidity".  In the
typical atmospheric regime where ``e_sat ≪ p``, mixing ratio and
specific humidity differ by ``q ≈ r / (1 + r)`` — about 1% for
``r = 0.01``.  The codebase uses these interchangeably; physics that
needs the distinction (vertical-flux conservation in saturated tropical
columns, q_c bookkeeping) should read this caveat carefully and
convert explicitly when the 1% drift matters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants


def saturation_vapor_pressure(T: jax.Array) -> jax.Array:
    """Compute saturation vapor pressure using Tetens formula.

    e_sat = 611.2 * exp(17.67 * T_c / (T_c + 243.5))

    Parameters
    ----------
    T : jax.Array
        Temperature [K].

    Returns
    -------
    jax.Array
        Saturation vapor pressure [Pa].
    """
    # AD-safe temperature floor.  The Tetens denominator is
    # ``T_c + 243.5 = T - 29.65 K``; as ``T → 29.65 K`` from below the
    # exponent → +∞ and ``exp`` OVERFLOWS to inf.  The forward is often
    # masked downstream (the smooth cap in ``saturation_mixing_ratio``
    # clamps q_sat to 1), but the REVERSE-mode gradient then hits
    # ``0 × inf`` and the whole adjoint goes non-finite — this silently
    # NaN'd carry-based differentiable training whenever a single
    # pathological surface/atmos cell dipped toward the singularity
    # (AIMIP, job 8533906: ``inf encountered in exp``).  Clip to 150 K
    # (far below any real atmospheric/surface temperature, so the forward
    # is bit-identical everywhere it matters; the clip's zero gradient
    # below the floor × the finite e_sat'(150 K) gives a finite gradient
    # there instead of inf).  satcurve-ok: identical Tetens curve for
    # T ≥ 150 K; this is an AD floor, not a new saturation formula.
    T_c = jnp.clip(T, 150.0, None) - constants.T_freeze
    return 611.2 * jnp.exp(17.67 * T_c / (T_c + 243.5))


# ---------------------------------------------------------------------------
# Alduchov & Eskridge (1996) "AERK" water + "AERKi" ice saturation curve.
#
# A single smooth (C-infinity), differentiable curve over the whole physical
# range: the over-water Magnus form (AERK, Eq. 21) and the over-ice Magnus form
# (AERKi, Eq. 23) blended across 0 degC by a logistic weight.  Unlike the plain
# over-water ``saturation_vapor_pressure`` above (Bolton 17.67/243.5), this
# branches to ice below freezing — the over-water extrapolation over-estimates
# e_s by ~10-60 % at -10..-50 degC.  Ported to match DifferBESS
# ``process/thermo.py`` (the two-big-leaf canopy oracle); the canopy energy
# balance uses this curve and its analytic derivatives for VPD/RH and the
# Penman-Monteith linearisation.  Alduchov & Eskridge (1996) J. Appl. Meteorol.
# 35(4) 601-609:  water (AERK, Eq.21) 610.94 Pa / 17.625 / 243.04 (<0.384% over
# -40..+50C);  ice (AERKi, Eq.23) 611.21 Pa / 22.587 / 273.86 (<0.213% over
# -80..0C).  Analytic derivatives match jax.grad / grad(grad) to round-off.
# ---------------------------------------------------------------------------
_AERK_C1_WATER, _AERK_A_WATER, _AERK_B_WATER = 610.94, 17.625, 243.04  # satcurve-ok: AERK water (thermo)
_AERK_C1_ICE,   _AERK_A_ICE,   _AERK_B_ICE   = 611.21, 22.587, 273.86  # satcurve-ok: AERKi ice (thermo)
_AERK_BLEND_HALFWIDTH_C = 1.0   # logistic blend half-width [degC]


def _aerk_magnus(C1, A, B, Tc):
    return C1 * jnp.exp(A * Tc / (Tc + B))


def _aerk_d_magnus(C1, A, B, Tc):
    e = _aerk_magnus(C1, A, B, Tc)
    return e * (A * B) * (Tc + B) ** (-2)


def _aerk_dd_magnus(C1, A, B, Tc):
    e  = _aerk_magnus(C1, A, B, Tc)
    de = _aerk_d_magnus(C1, A, B, Tc)
    return (A * B) * (de * (Tc + B) ** (-2) - 2.0 * e * (Tc + B) ** (-3))


def _aerk_w(Tc):
    """Logistic weight on the WATER branch (-> 1 warm, -> 0 cold)."""
    return jax.nn.sigmoid(Tc / _AERK_BLEND_HALFWIDTH_C)


def saturation_vapor_pressure_aerk(T: jax.Array) -> jax.Array:
    """Saturation vapour pressure [Pa] from T [K] — AERK water + AERKi ice blend.

    Over-water/over-ice Magnus forms (Alduchov & Eskridge 1996) blended across
    0 degC.  Matches DifferBESS ``process/thermo.py``.  Use in place of the plain
    over-water :func:`saturation_vapor_pressure` where sub-freezing accuracy
    matters (the canopy energy balance) — the over-water form over-estimates e_s
    by ~10-60 % below -10 degC.
    """
    Tc = T - constants.T_freeze
    w  = _aerk_w(Tc)
    ew = _aerk_magnus(_AERK_C1_WATER, _AERK_A_WATER, _AERK_B_WATER, Tc)
    ei = _aerk_magnus(_AERK_C1_ICE,   _AERK_A_ICE,   _AERK_B_ICE,   Tc)
    return w * ew + (1.0 - w) * ei


def d_saturation_vapor_pressure_aerk(T: jax.Array) -> jax.Array:
    """First derivative d e_s/dT [Pa K-1] of the AERK blend (analytic)."""
    Tc = T - constants.T_freeze
    w  = _aerk_w(Tc)
    dw = w * (1.0 - w) / _AERK_BLEND_HALFWIDTH_C
    ew  = _aerk_magnus(_AERK_C1_WATER, _AERK_A_WATER, _AERK_B_WATER, Tc)
    ei  = _aerk_magnus(_AERK_C1_ICE,   _AERK_A_ICE,   _AERK_B_ICE,   Tc)
    dew = _aerk_d_magnus(_AERK_C1_WATER, _AERK_A_WATER, _AERK_B_WATER, Tc)
    dei = _aerk_d_magnus(_AERK_C1_ICE,   _AERK_A_ICE,   _AERK_B_ICE,   Tc)
    return dw * (ew - ei) + w * dew + (1.0 - w) * dei


def dd_saturation_vapor_pressure_aerk(T: jax.Array) -> jax.Array:
    """Second derivative d^2 e_s/dT^2 [Pa K-2] of the AERK blend (analytic).

    Uses the saturation curve only (no actual vapour pressure) — fixing the
    historical ``e_c``-instead-of-``e_s`` Penman-Monteith bug at the source.
    """
    Tc  = T - constants.T_freeze
    w   = _aerk_w(Tc)
    dw  = w * (1.0 - w) / _AERK_BLEND_HALFWIDTH_C
    ddw = w * (1.0 - w) * (1.0 - 2.0 * w) / _AERK_BLEND_HALFWIDTH_C ** 2
    ew   = _aerk_magnus(_AERK_C1_WATER, _AERK_A_WATER, _AERK_B_WATER, Tc)
    ei   = _aerk_magnus(_AERK_C1_ICE,   _AERK_A_ICE,   _AERK_B_ICE,   Tc)
    dew  = _aerk_d_magnus(_AERK_C1_WATER, _AERK_A_WATER, _AERK_B_WATER, Tc)
    dei  = _aerk_d_magnus(_AERK_C1_ICE,   _AERK_A_ICE,   _AERK_B_ICE,   Tc)
    ddew = _aerk_dd_magnus(_AERK_C1_WATER, _AERK_A_WATER, _AERK_B_WATER, Tc)
    ddei = _aerk_dd_magnus(_AERK_C1_ICE,   _AERK_A_ICE,   _AERK_B_ICE,   Tc)
    return ddw * (ew - ei) + 2.0 * dw * (dew - dei) + w * ddew + (1.0 - w) * ddei


def saturation_vapor_pressure_goff(T: jax.Array) -> jax.Array:
    """Saturation vapour pressure over liquid water, WMO Goff (1957) [Pa].

    Exact port of NEMO/aerobulk ``sbc_phy.F90::e_sat_sclr`` (the curve NEMO's
    bulk formulas use), kept alongside the Tetens default so the OMIP/CORE-II
    faithful ocean forcing reproduces NEMO's saturation humidity bit-for-bit.
    Differs from :func:`saturation_vapor_pressure` by ~0.1–0.3 % over the
    ocean-temperature range.  The temperature is floored at 180 K exactly as
    NEMO does (guards masked/garbage cells).

    Parameters
    ----------
    T : jax.Array
        Temperature [K].

    Returns
    -------
    jax.Array
        Saturation vapor pressure [Pa].
    """
    T_K = jnp.maximum(T, 180.0)
    rt = constants.T_freeze / T_K
    log10_T = jnp.log10(T_K / constants.T_freeze)
    return 100.0 * 10.0 ** (
        10.79574 * (1.0 - rt)
        - 5.028 * log10_T
        + 1.50475e-4 * (1.0 - 10.0 ** (-8.2969 * (T_K / constants.T_freeze - 1.0)))
        + 0.42873e-3 * (10.0 ** (4.76955 * (1.0 - rt)) - 1.0)
        + 0.78614
    )


# ---------------------------------------------------------------------------
# Flatau et al. (1992) polynomial saturation vapor pressure
# ---------------------------------------------------------------------------
# CLUBB (and CAM, via ``saturation_formula = flatau``) closes its assumed-PDF
# cloud scheme on the Flatau 8th-order polynomial fit to the SVP curve rather
# than Tetens. These are the CANONICAL Flatau curves (added here, in the shared
# thermo module, so the CLUBB port consumes saturation only from ``thermo`` per
# the CLAUDE.md "no saturation re-impl" rule — the curves are NOT re-derived
# inside the physics tree). Faithful to ``saturation.F90`` /
# ``CLUBB-JAX/.../saturation.py`` (coefficients verbatim). Reference:
# Flatau, P. J., Walko, R. L., & Cotton, W. R. (1992). Polynomial fits to
# saturation vapor pressure. J. Appl. Meteorol., 31, 1507-1513, Tables 3-4.

_FLATAU_MIN_T_C = -85.0       # liquid polynomial valid range floor [deg C]
_FLATAU_ICE_MIN_T_C = -90.0   # ice polynomial valid range floor [deg C]

# IFS homogeneous-freezing RH-over-ice ramp, as adopted by gSAM 1.8.7
# MICRO_SAM1MOM/cloud.f90: ``rh_homo = 2.583 - tabs/207.8`` below 235 K.
_IFS_RH_HOMO_A = 2.583        # ramp intercept [-]
_IFS_RH_HOMO_B = 207.8        # ramp temperature scale [K]
_IFS_RH_HOMO_T_MAX = 235.0    # allowance applies only below this [K]

# Flatau ice polynomial coefficients (Table 4), x100 as in saturation.F90.
_FLATAU_ICE_A = (
    100.0 * 6.09868993,
    100.0 * 0.499320233,
    100.0 * 0.184672631e-1,
    100.0 * 0.402737184e-3,
    100.0 * 0.565392987e-5,
    100.0 * 0.521693933e-7,
    100.0 * 0.307839583e-9,
    100.0 * 0.105785160e-11,
    100.0 * 0.161444444e-14,
)


def saturation_vapor_pressure_flatau(T: jax.Array) -> jax.Array:
    """Flatau (1992) polynomial saturation vapor pressure over liquid water.

    8th-order factored polynomial fit, valid roughly -85 to +50 deg C. This is
    the SVP curve CLUBB/CAM use by default (``saturation_formula = flatau``).

    Parameters
    ----------
    T : jax.Array
        Temperature [K].

    Returns
    -------
    jax.Array
        Saturation vapor pressure over liquid [Pa].
    """
    T_c = jnp.clip(T - constants.T_freeze, _FLATAU_MIN_T_C, None)
    T_sqd = T_c ** 2
    return (
        -3.21582393e-14
        * (T_c - 646.5835252598777)
        * (T_c + 90.72381630364440)
        * (T_sqd + 111.0976961559954 * T_c + 6459.629194243118)
        * (T_sqd + 152.3131930092453 * T_c + 6499.774954705265)
        * (T_sqd + 174.4279584934021 * T_c + 7721.679732114084)
    )


def saturation_vapor_pressure_ice_flatau(T: jax.Array) -> jax.Array:
    """Flatau (1992) polynomial saturation vapor pressure over ice.

    8th-order Horner polynomial (Table 4), valid roughly -90 to 0 deg C.

    Parameters
    ----------
    T : jax.Array
        Temperature [K].

    Returns
    -------
    jax.Array
        Saturation vapor pressure over ice [Pa].
    """
    T_c = jnp.clip(T - constants.T_freeze, _FLATAU_ICE_MIN_T_C, None)
    a = _FLATAU_ICE_A
    return (
        a[0] + T_c * (a[1] + T_c * (a[2] + T_c * (
            a[3] + T_c * (a[4] + T_c * (a[5] + T_c * (
                a[6] + T_c * (a[7] + T_c * a[8])))))))
    )


def saturation_mixing_ratio(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute saturation mixing ratio using Tetens formula.

    e_sat = 611.2 * exp(17.67 * T_c / (T_c + 243.5))   where T_c = T - 273.15
    q_sat = epsilon * e_sat / (p - e_sat)

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Saturation mixing ratio [kg/kg].
    """
    return _mixing_ratio_from_esat(saturation_vapor_pressure(T), p)


def _mixing_ratio_from_esat(e_sat: jax.Array, p: jax.Array) -> jax.Array:
    """Saturation mixing ratio from a saturation vapour pressure [Pa].

    The shared (differentiable, smooth-floored/capped) ``e_sat -> q_sat``
    conversion used by every saturation curve (Tetens, Goff), so the curve
    is the ONLY thing that varies between conventions — no re-derived
    conversion numerics (#762).
    """
    # Smooth floor on denominator: preserves gradients near e_sat ≈ p
    # instead of a hard clip that creates a zero-gradient plateau.
    # softplus(x - 1) + 1 ≈ x for x >> 1, ≈ 1 for x << 1, smooth at x = 1.
    denom = jax.nn.softplus(p - e_sat - 1.0) + 1.0
    q_sat = constants.epsilon * e_sat / denom
    # Smooth cap at 1.0 kg/kg: prevents singularity at low-pressure levels
    # while allowing gradients to flow (unlike hard jnp.minimum).
    # Uses LogSumExp smooth-min: 1 - softplus(β(1 - x))/β with β = 20.
    return 1.0 - jax.nn.softplus(20.0 * (1.0 - q_sat)) / 20.0


def saturation_mixing_ratio_goff(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Saturation mixing ratio from the WMO Goff (1957) curve [kg/kg].

    The NEMO/AeroBulk air-sea convention for the surface saturation
    humidity (issue #762): identical smooth ``e_sat -> q_sat`` conversion
    as :func:`saturation_mixing_ratio`, but over the Goff vapour-pressure
    curve instead of Tetens — ~0.5-1 % on Δq (hence LH) at warm SST.
    """
    return _mixing_ratio_from_esat(saturation_vapor_pressure_goff(T), p)


def saturation_mixing_ratio_ice(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute saturation mixing ratio over ice (Clausius-Clapeyron).

    e_sat_i = 611.2 * exp(L_s/R_v * (1/T_freeze - 1/T))
    q_sat_i = epsilon * e_sat_i / (p - e_sat_i)

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Ice saturation mixing ratio [kg/kg].
    """
    e_sat_i = 611.2 * jnp.exp(
        constants.L_s / constants.R_v * (1.0 / constants.T_freeze - 1.0 / T)
    )
    denom = jax.nn.softplus(p - e_sat_i - 1.0) + 1.0
    q_sat_i = constants.epsilon * e_sat_i / denom
    return 1.0 - jax.nn.softplus(20.0 * (1.0 - q_sat_i)) / 20.0


def homogeneous_freezing_rh_factor(
    T: jax.Array,
    q_ice: jax.Array | None = None,
    *,
    enabled: bool = True,
    q_ice_threshold: float = 1.0e-8,
) -> jax.Array:
    """IFS homogeneous-freezing ice-supersaturation allowance (SAM ``rh_homo``).

    gSAM 1.8.7 ``MICRO_SAM1MOM/cloud.f90`` (Khairoutdinov 2023, "Modeled after
    IFS model") scales the ICE saturation target so that pristine, very cold
    air may stay ice-supersaturated up to the homogeneous-freezing threshold::

        rh_homo = 2.583 - T / 207.8      for T < 235 K with no pre-existing ice
        rh_homo = 1                      otherwise

    i.e. ~1.45 at 235 K rising to ~1.67 at 190 K.  The gate is physical: with
    no ice surface present there is nothing for the vapour to deposit onto, so
    it accumulates until homogeneous freezing of solution droplets fires.  Where
    ice is already present the allowance is withdrawn — cloud.f90: "if ice
    already exists - do as usual, that is no supersaturation over ice".

    SCOPE, precisely: ``q_ice`` is the CLOUD-ICE mass the caller passes at scheme
    ENTRY, matching cloud.f90's ``qci`` gate.  Ice nucleated later in the same
    microphysics call does not withdraw the allowance until the next step, and
    precipitating ice (snow/graupel) is not counted — gSAM gates on ``qci``
    alone, so counting those surfaces here would depart from the oracle.

    Multiply an ice saturation mixing ratio by this factor to get the target a
    deposition/adjustment step should relax toward.

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    q_ice : jax.Array or None
        Pre-existing cloud-ice mixing ratio [kg/kg].  ``None`` treats the air
        as ice-free (the pristine branch) everywhere.
    enabled : bool, default True
        ``False`` returns 1.0 — the no-allowance behaviour, i.e. deposition
        targets plain ice saturation.  This is the OFF switch component
        configs thread through; it is a static Python bool, so the disabled
        path compiles to the original expression.
    q_ice_threshold : float, default 1e-8
        The ``qci < 1e-8`` ice-presence gate from cloud.f90 [kg/kg].

    Returns
    -------
    jax.Array
        Multiplicative factor >= 1 applied to the ice saturation target.
    """
    if not enabled:
        return jnp.ones_like(T)
    rh_homo = jnp.maximum(
        _IFS_RH_HOMO_A - T / _IFS_RH_HOMO_B, 1.0,
    )
    cold = T < _IFS_RH_HOMO_T_MAX
    if q_ice is not None:
        cold = cold & (q_ice < q_ice_threshold)
    return jnp.where(cold, rh_homo, 1.0)


def saturation_mixing_ratio_blend(
    T: jax.Array,
    p: jax.Array,
    T_blend_top: float | None = None,
    T_blend_width: float = 20.0,
) -> jax.Array:
    """FV3_3D iter 720: saturation mixing ratio with liquid/ice blend.

    Reusable helper matching FV3's ``compute_qs(..., es_over_liq_and_ice=
    .true.)`` behaviour:

        w_liq  = clip((T − (T_top − width)) / width, 0, 1)
        q_sat  = w_liq · q_sat_liq + (1 − w_liq) · q_sat_ice

    Defaults:
        T_blend_top = ``constants.T_freeze``     (273.15 K)
        T_blend_width = 20.0 K

    Generalizes the inline blend in ``rh_calc_fv3 do_cmip=True``
    (iter-715) for reuse by other diagnostics.

    Parameters
    ----------
    T : jax.Array
        Temperature (K).
    p : jax.Array
        Pressure (Pa).
    T_blend_top : float, optional
        Upper temperature above which q_sat = q_sat_liq exactly.
        Default ``constants.T_freeze``.
    T_blend_width : float, default 20.0 K.
        Linear-blend width.

    Returns
    -------
    q_sat : jax.Array
        Blended saturation mixing ratio (kg/kg).
    """
    if T_blend_top is None:
        T_blend_top = constants.T_freeze
    T_blend_bot = T_blend_top - T_blend_width
    qs_liq = saturation_mixing_ratio(T, p)
    qs_ice = saturation_mixing_ratio_ice(T, p)
    w_liq = jnp.clip(
        (T - T_blend_bot) / (T_blend_top - T_blend_bot),
        0.0, 1.0,
    )
    return w_liq * qs_liq + (1.0 - w_liq) * qs_ice


def saturation_mixing_ratio_dT(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Analytic derivative d(q_sat)/dT consistent with ``saturation_mixing_ratio``.

    Uses the same Tetens vapor-pressure formula as
    ``saturation_vapor_pressure`` and the same hard ``p - e_sat`` floor as
    historically used by closure schemes (CLUBB-style PDF widths).  The
    smooth softplus floor used by ``saturation_mixing_ratio`` itself
    is intentionally NOT applied here — for derivative use cases
    (e.g. Gaussian PDF width scaling), the simpler ``max(p - e_sat, 1)``
    floor is the standard convention.

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        d(q_sat)/dT [kg/kg/K].
    """
    e_sat = saturation_vapor_pressure(T)
    T_c = T - constants.T_freeze
    de_dT = e_sat * 17.67 * 243.5 / (T_c + 243.5) ** 2
    p_eff = jnp.clip(p - e_sat, 1.0)
    return constants.epsilon * de_dT * p / p_eff ** 2


def saturation_specific_humidity(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute saturation specific humidity from saturation mixing ratio.

    q = w_sat / (1 + w_sat)

    where w_sat = epsilon * e_sat / (p - e_sat) is the saturation mixing
    ratio.  Use this function when working with specific humidity fields
    (q = m_v / (m_v + m_d)) rather than mixing ratio (w = m_v / m_d).

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Saturation specific humidity [kg/kg].
    """
    w_sat = saturation_mixing_ratio(T, p)
    return w_sat / (1.0 + w_sat)


def vapor_pressure_from_specific_humidity(
    q: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Vapor pressure [Pa] from specific humidity and total pressure.

    ``e = q · p / (ε + (1 − ε) · q)`` — the inverse of the specific-humidity
    definition ``q = ε e / (p − (1 − ε) e)`` (NOT the mixing-ratio form
    ``q p / (ε + q)``, which biases e by ~1% at tropical q).  Centralised here
    so the canopy stomatal (Jarvis / coupled-Farquhar) and canopy-air
    energy-balance VPD paths share one derivation.

    Parameters
    ----------
    q : jax.Array
        Specific humidity [kg/kg].
    p : jax.Array
        Total pressure [Pa].

    Returns
    -------
    jax.Array
        Vapor pressure [Pa].
    """
    return q * p / (constants.epsilon + (1.0 - constants.epsilon) * q)


def virtual_temperature(
    T: jax.Array,
    q: jax.Array,
) -> jax.Array:
    """Virtual temperature [K] from temperature and specific humidity.

    ``T_v = T (1 + (1/ε − 1) q)`` — the specific-humidity form (ε = R_d/R_v so
    1/ε − 1 = R_v/R_d − 1).  Centralised for the land surface-forcing paths
    (CRU-JRA assembly, eddy-covariance site loader) that form moist-air density
    from it; mirrors ``atmosphere.physics._shared.virtual_temperature``.

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    q : jax.Array
        Specific humidity [kg/kg].

    Returns
    -------
    jax.Array
        Virtual temperature [K].
    """
    return T * (1.0 + (1.0 / constants.epsilon - 1.0) * q)


def moist_air_density(
    T: jax.Array,
    p: jax.Array,
    q: jax.Array,
) -> jax.Array:
    """Moist-air density [kg/m3]: ``ρ = p / (R_d T_v)`` with virtual temperature
    ``T_v`` from :func:`virtual_temperature`."""
    return p / (constants.R_d * virtual_temperature(T, q))


def mixing_ratio_to_specific_humidity(
    mixing_ratio: jax.Array,
) -> jax.Array:
    """Convert water-vapor mixing ratio to specific humidity.

    The mixing ratio convention is ``r = m_v / m_d``; specific humidity is
    ``q = m_v / (m_v + m_d)``.  This helper centralizes the conversion for
    data-ingest paths that cross between those conventions.
    """
    r = jnp.maximum(jnp.asarray(mixing_ratio), 0.0)
    return r / (1.0 + r)


def specific_humidity_to_mixing_ratio(
    specific_humidity: jax.Array,
    *,
    denominator_floor: float = 1.0e-12,
) -> jax.Array:
    """Convert specific humidity to water-vapor mixing ratio.

    ``q`` is clipped below one AND the denominator is explicitly floored so
    malformed input cannot divide by zero; valid atmospheric values are
    unchanged.  The explicit ``jnp.maximum(1 - q, floor)`` (matching
    :func:`specific_humidity_tendency_to_mixing_ratio_tendency`) is required for
    float32 inputs, where ``1 - 1e-12`` rounds to exactly ``1.0`` so the clip
    alone would still divide by zero (→ ``inf``) at ``q ≥ 1``.
    """
    q = jnp.clip(jnp.asarray(specific_humidity), 0.0, 1.0 - denominator_floor)
    denom = jnp.maximum(1.0 - q, denominator_floor)
    return q / denom


def specific_humidity_tendency_to_mixing_ratio_tendency(
    specific_humidity: jax.Array,
    specific_humidity_tendency: jax.Array,
    *,
    denominator_floor: float = 1.0e-12,
) -> jax.Array:
    """Convert ``dq/dt`` to ``dr/dt`` for ``r = q / (1 - q)``.

    DEPHY and reanalysis files often provide tendencies for specific humidity,
    while the atmospheric physics path consumes water-vapor mixing ratio.  The
    derivative is ``dr/dt = dq/dt / (1 - q)^2``.
    """
    q = jnp.clip(jnp.asarray(specific_humidity), 0.0, 1.0 - denominator_floor)
    denom = jnp.maximum(1.0 - q, denominator_floor)
    return jnp.asarray(specific_humidity_tendency) / (denom * denom)


def relative_humidity(
    T: jax.Array,
    p: jax.Array,
    mixing_ratio: jax.Array,
) -> jax.Array:
    """Saturation ratio ``S = e / e_sat`` (WMO relative humidity, as a fraction).

    Computes the water-vapor partial pressure from the vapor MIXING RATIO
    ``r = m_v / m_d`` [kg/kg dry air] and divides by the saturation vapor
    pressure::

        e = p * r / (epsilon + r)
        S = e / saturation_vapor_pressure(T)

    This is the saturation ratio that diffusional droplet growth uses (its
    ``(S - 1)`` supersaturation), and is distinct from the mixing-ratio ratio
    ``r / r_sat`` — the two differ by ``O(r/epsilon, e_sat/p)`` (~1 %), which is
    a large fractional error in the small ``S - 1`` activation signal.

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].
    mixing_ratio : jax.Array
        Water-vapor mixing ratio ``r = m_v/m_d`` [kg/kg].

    Returns
    -------
    jax.Array
        Saturation ratio ``S = e/e_sat`` [-] (1.0 at saturation).
    """
    e = p * mixing_ratio / (constants.epsilon + mixing_ratio)
    return e / saturation_vapor_pressure(T)


def latent_heat_vaporization_sst(T_sfc_K: jax.Array) -> jax.Array:
    """SST-dependent latent heat of vaporization [J/kg].

    The NEMO/AeroBulk air-sea convention (sbc_phy ``L_vap``, also
    COARE/Fairall): ``L = L_v - L_v_sst_slope (T - T_freeze)``; equals
    ``constants.L_v`` at 0 degC by construction.  Up to ~3 % smaller than
    the constant at warm SST (issue #762).  Dtype-preserving — the OMIP
    NEMO-parity path wraps this with its float64 pin.
    """
    return constants.L_v - constants.L_v_sst_slope * (
        T_sfc_K - constants.T_freeze
    )


def moist_air_cp(q_air: jax.Array) -> jax.Array:
    """Moist-air specific heat [J/(kg K)], NEMO/AeroBulk convention.

    ``cp = rCp_dry + rCp_vap q`` (NEMO sbc_phy ``cp_air``) — the
    convention set of the transcribed bulk schemes, NOT the
    mixture-weighted ``c_pd (1-q) + c_pv q`` (issue #762).  ~1-2 % above
    dry ``c_pd`` in the humid tropics.  Dtype-preserving.
    """
    return constants.c_p_dry_air_nemo + constants.c_p_vapor_nemo * q_air
