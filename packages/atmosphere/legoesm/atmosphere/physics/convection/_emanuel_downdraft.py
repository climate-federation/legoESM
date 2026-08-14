"""Emanuel's precipitating unsaturated downdraft — faithful port of CONVECT.

Port of ``convect43c.f`` lines 713-830 (the downdraft loop) together with the
tendency terms at lines 843-849 and 932-934 that consume it.  Oracle at
``.physics-validator/emanuel/oracle/convect43c.f`` (1075 lines, from
``texmex.mit.edu/pub/emanuel/CONVECT4``); every line number cited below is
that file's.

WHAT IT REPLACES.  ``EmanuelConfig.enable_unsaturated_downdraft`` previously
selected a stand-in that moved a fixed fraction of the column-integrated
condensate into a below-LCL cooling+moistening tendency.  That has no
downdraft mass flux, no rain-water budget, no fall speed and no dependence on
sub-cloud humidity, so it cannot do the one thing a downdraft does: import air
of LOW moist static energy from aloft into the sub-cloud layer.  The oracle's
version does, through ``MP`` (the downdraft mass flux) and ``QP`` (its mixing
ratio).

THE PHYSICS, in the oracle's own order:

1. ``WDTRAIN`` — condensate detrained into the precipitation shaft at level i,
   from the adiabatic updraught (``EP*M*CLW``) plus every mixture detraining
   there (``max(0, ELIJ-(1-EP)*CLW)*MENT``).
2. A quadratic for the rain-water content: writing ``REVAP = sqrt(WATER)``,
   the steady balance between what falls in from above, what is detrained and
   what evaporates gives ``REVAP = 0.5*(-B6 + sqrt(B6^2 + 4*C6))`` — always
   the positive root.
3. ``EVAP`` follows; the evaporative cooling drives a downdraft mass flux
   ``MP`` under a hydrostatic approximation, smoothed by a small inertia term
   and forced linearly to zero below ~950 hPa.
4. ``QP``, the shaft's mixing ratio, integrates downward: mixing with the
   environment where the mass flux increases downward, moist-adiabatic descent
   where it does not.

A PROPERTY WORTH KNOWING BEFORE READING THE TENDENCIES.  ``FT`` receives NO
mass-flux advection term from the downdraft — only evaporative cooling and the
sensible heat carried by the falling rain.  ``FQ`` receives the full
``MP(i+1)*(QP(i+1)-Q(i)) - MP(i)*(QP(i)-Q(i-1))`` flux divergence.  That
asymmetry is the oracle's, not a porting omission: CONVECT carries the
downdraft's thermal effect through the ``QP``/``H`` formulation rather than
through a separate dry-static-energy flux.

INDEX CONVENTION.  Everything here is SURFACE-FIRST (``i=0`` is the lowest
model level, ``i+1`` is ABOVE ``i``), exactly as the oracle indexes it and as
``_emanuel_mixing`` works internally.  The caller flips.  Getting this
backwards produces a plausible-looking wrong answer with every shape intact,
so ``emanuel_downdraft_tendencies`` asserts the orientation of its inputs
rather than trusting it.

DIFFERENTIABILITY.  Every oracle branch becomes smooth or masked.  Three
places need the double-``where`` guard, because a plain ``jnp.where`` still
differentiates the unselected branch and propagates its NaN: the division by
``MP`` in the ``QP`` update, the square root whose argument reaches zero on a
non-convecting column, and the ``P(1)-P(JTT)`` denominator of the surface
taper (zero whenever the two lowest levels share a pressure).

Reference: Emanuel, K. A. (1991), J. Atmos. Sci. 48, 2313-2335; Emanuel &
Zivkovic-Rothman (1999), J. Atmos. Sci. 56, 1766-1782.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from legoesm import constants

__physics_contract__ = {
    "units": (
        "mp [kg/m^2/s] downdraft mass flux; qp [kg/kg] downdraft mixing "
        "ratio; evap [(kg/kg)/s]; water [kg/kg] rain-water content; "
        "wt [Pa/s] hydrometeor fall speed; dT_dt [K/s]; "
        "dq_v_dt [(kg/kg)/s]; precip_mm_day [mm/day]. Profiles SURFACE-FIRST."
    ),
    "signs": (
        "mp >= 0 (a downward flux, magnitude only) and mp == 0 exactly at the "
        "lowest level. evap >= 0. Evaporation COOLS (negative dT_dt) and "
        "MOISTENS (positive dq_v_dt) the layer it occurs in, which for rain "
        "falling out of cloud base is the sub-cloud layer."
    ),
    "conserves": ["water"],
    "differentiable": True,
    "reference": "Emanuel (1991) CONVECT v4.3c, convect43c.f lines 713-934",
    "idealized_test": (
        "A column with no detrained condensate (ep == 0) gives exactly zero "
        "mp, evap and tendencies. Evaporation floors at zero only when the "
        "MIXTURE of environmental and downdraft air is supersaturated: a "
        "saturated ENVIRONMENT still evaporates, because AFAC is driven by "
        "QS(I) - 0.5*(Q(I) + QP(I+1)) and the descending shaft is drier than "
        "its surroundings (measured, not assumed)."
    ),
}

# --- Unit conversions carried by the oracle's hPa/Pa mixture ------------- #
_HPA_TO_PA = 100.0
_PA_TO_HPA = 0.01
_SEC_PER_DAY = 86400.0

# --- Published CONVECT v4.3c constants (convect43c.f lines 186-201) ------ #
#: Specific heat of liquid water [J/kg/K] AS THE ORACLE SETS IT (line 190).
#: The textbook value is ~4190; CONVECT uses 2500, and an oracle port
#: reproduces the oracle.  It enters only the ``(CL - CPD)`` sensible-heat
#: term carried by falling precipitation.
_CL_ORACLE = 2500.0
#: Density of liquid water [kg/m^3] used by the oracle's precipitation
#: conversion (line 189).
_ROWL_ORACLE = 1000.0

# --- AD-safe floors.  Each guards a DERIVATIVE, not a value. ------------- #
_SQRT_FLOOR = 1.0e-30
_MP_FLOOR = 1.0e-12
_DENOM_FLOOR = 1.0e-12


class EmanuelDowndraftOutput(NamedTuple):
    """Downdraft diagnosis and its tendency contributions (SURFACE-FIRST)."""

    mp: jax.Array
    qp: jax.Array
    evap: jax.Array
    water: jax.Array
    wt: jax.Array
    dT_dt: jax.Array
    dq_v_dt: jax.Array
    precip_mm_day: jax.Array


def _safe_sqrt(x: jax.Array) -> jax.Array:
    """``sqrt`` with a finite derivative at ``x == 0``.

    ``d sqrt(x)/dx`` diverges at 0 and the discriminant ``B6^2 + 4*C6`` is
    exactly 0 on a column with no detrained condensate — precisely the common
    case, so the gradient must stay finite there.  The double-``where`` masks
    the zero out of the differentiated branch while returning the exact value
    forward.
    """
    safe = jnp.where(x > _SQRT_FLOOR, x, jnp.ones_like(x))
    return jnp.where(x > _SQRT_FLOOR, jnp.sqrt(safe), jnp.zeros_like(x))


def _safe_divide(num: jax.Array, den: jax.Array, floor: float) -> jax.Array:
    """``num/den``, exactly 0 with a 0 derivative where ``|den| <= floor``."""
    ok = jnp.abs(den) > floor
    safe_den = jnp.where(ok, den, jnp.ones_like(den))
    return jnp.where(ok, num / safe_den, jnp.zeros_like(num))


def _above(x: jax.Array) -> jax.Array:
    """``x[i+1]`` (the level ABOVE i in surface-first order), clamped at top."""
    return jnp.concatenate([x[:, 1:], x[:, -1:]], axis=-1)


def _below(x: jax.Array) -> jax.Array:
    """``x[i-1]`` (the level BELOW i in surface-first order), clamped at 0."""
    return jnp.concatenate([x[:, :1], x[:, :-1]], axis=-1)


def _fall_speed_and_coefficient(
    T: jax.Array,
    *,
    omtrain: float,
    omtsnow: float,
    coeffr: float,
    coeffs: float,
    freeze_transition_K: float,
) -> tuple[jax.Array, jax.Array]:
    """Smooth replacement for the oracle's ``IF(T(I).GT.273.0)`` switch.

    The oracle flips fall speed and evaporation coefficient discontinuously
    (line 749).  A step is not differentiable and puts a derivative-free tuner
    on a cliff; blending over ``freeze_transition_K`` recovers both limits away
    from the transition and is monotone through it.

    The threshold is the ORACLE's 273.0 K, deliberately not the model's
    freezing point — reproducing the oracle means reproducing its constant,
    and the difference is far inside the blend width.
    """
    warm = jax.nn.sigmoid((T - 273.0) / freeze_transition_K)  # const-ok: oracle threshold, not a freezing point
    wt = omtsnow + (omtrain - omtsnow) * warm
    coeff = coeffs + (coeffr - coeffs) * warm
    return wt, coeff


def _taper_mass_flux_to_surface(
    mp: jax.Array, p_full: jax.Array, *, taper_p_fraction: float,
) -> jax.Array:
    """Force ``MP`` linearly to zero between ~950 hPa and the surface.

    Oracle lines 779-784::

        IF(P(I).GT.(0.949*P(1)))THEN
           JTT=MAX(JTT,I)
           MP(I)=MP(JTT)*(P(1)-P(I))/(P(1)-P(JTT))
        END IF

    ``JTT`` is a running maximum over a loop DESCENDING from cloud top, and
    ``P`` decreases upward, so the pressure test is true exactly on a prefix
    of levels near the surface.  The loop meets that band at its TOP first;
    there ``JTT`` is fixed for good and the formula is the identity, so every
    lower level reads the same anchor.  The construct is therefore a one-pass
    linear taper:

        jtt   = max(1, largest i with p[i] > f * p[0])         (0-based)
        mp[i] = mp[jtt] * (p[0]-p[i]) / (p[0]-p[jtt])   for 1 <= i <= jtt

    Two consequences worth stating because they are why a second pass is
    EXACT rather than approximate: the anchor ``mp[jtt]`` comes from the
    inertia recursion and is never itself tapered, and the recursion's output
    at levels inside the band is dead — the taper overwrites it before
    anything reads it.

    ``i = 0`` is excluded because the oracle's ``IF(I.EQ.1)GOTO 360`` jumps
    past this whole block at the lowest level, leaving ``MP(1)`` at its zero
    initialisation (line 481).  The caller enforces that zero; the outcome is
    the same either way, and stating which mechanism produces it keeps the
    port checkable against the Fortran.

    The anchor INDEX is discrete in the oracle and stays discrete here; the
    gradient flows through the gathered ``mp[jtt]`` and the pressures.
    """
    _ncol, nlev = mp.shape
    idx = jnp.arange(nlev)
    p_sfc = p_full[:, :1]
    band = p_full > taper_p_fraction * p_sfc
    # Floored at 1 to match ``JTT`` starting at 2 in the oracle's 1-based
    # indexing; also keeps the anchor off the surface level itself.
    jtt = jnp.maximum(jnp.max(jnp.where(band, idx[None, :], -1), axis=-1), 1)
    mp_anchor = jnp.take_along_axis(mp, jtt[:, None], axis=-1)
    p_anchor = jnp.take_along_axis(p_full, jtt[:, None], axis=-1)
    # Zero denominator whenever the two lowest levels share a pressure — the
    # oracle would produce NaN there.
    ramp = _safe_divide(p_sfc - p_full, p_sfc - p_anchor, _DENOM_FLOOR)
    inside = band & (idx[None, :] <= jtt[:, None]) & (idx[None, :] > 0)
    return jnp.where(inside, mp_anchor * ramp, mp)


def emanuel_downdraft(
    *,
    T: jax.Array,
    q: jax.Array,
    qs: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    h_moist: jax.Array,
    gz: jax.Array,
    lv: jax.Array,
    cpn: jax.Array,
    m_profile: jax.Array,
    ment: jax.Array,
    elij: jax.Array,
    clw: jax.Array,
    ep: jax.Array,
    sigd: float,
    sigs: float,
    omtrain: float,
    omtsnow: float,
    coeffr: float,
    coeffs: float,
    freeze_transition_K: float,
    inertia_scale_hPa: float,
    taper_p_fraction: float,
    dhdp_min: float,
    ep_gate_threshold: float,
    ep_gate_width: float,
) -> EmanuelDowndraftOutput:
    """Diagnose the precipitating downdraft and its tendencies.

    ALL inputs and outputs are SURFACE-FIRST: ``i = 0`` is the lowest model
    level and ``i+1`` is above ``i``.

    Parameters
    ----------
    T, q, qs : (ncol, nlev) environmental temperature [K] and vapour /
        saturation mixing ratio [kg/kg].
    p_full : (ncol, nlev) full-level pressure [Pa], oracle ``P``.
    p_half : (ncol, nlev) pressure of the interface at the BOTTOM of level i
        [Pa], oracle ``PH`` — so ``p_half[i] - p_half[i+1]`` is the mass
        thickness of level i, which is how the oracle uses it.
    h_moist, gz, lv, cpn : (ncol, nlev) oracle ``H``, ``GZ``, ``LV``, ``CPN``.
    m_profile, ment, elij, clw, ep : updraught mass flux, mixing matrix,
        mixture condensate, adiabatic cloud water and precipitation efficiency
        from ``_emanuel_mixing`` (all surface-first).
    """
    _ncol, nlev = T.shape
    g = constants.g
    ginv = 1.0 / g
    cpd = constants.c_pd

    # -- 1. Detrained precipitation (oracle 729-736) -----------------------
    # WDTRAIN(i) = G*EP(i)*M(i)*CLW(i)
    #              + sum_{j<i} G*max(0, ELIJ(j,i) - (1-EP(i))*CLW(i))*MENT(j,i)
    # The sum runs over ORIGIN levels j strictly BELOW the detrainment level
    # i, i.e. strictly lower-triangular in (j, i).  Transposing it feeds the
    # shaft from the wrong mixtures while every shape still matches, so the
    # mask is written from the Fortran loop bounds (DO 320 J=1,I-1) rather
    # than from intuition.
    idx = jnp.arange(nlev)
    lower = (idx[:, None] < idx[None, :])[None, :, :]
    awat = jnp.maximum(elij - (1.0 - ep)[:, None, :] * clw[:, None, :], 0.0)
    wdtrain = g * ep * m_profile * clw + g * jnp.sum(
        jnp.where(lower, awat * ment, 0.0), axis=1)

    # -- 2. Fall speed / evaporation coefficient / precipitating fraction --
    wt, coeff = _fall_speed_and_coefficient(
        T, omtrain=omtrain, omtsnow=omtsnow, coeffr=coeffr, coeffs=coeffs,
        freeze_transition_K=freeze_transition_K)
    # SIGP(i) is uniformly SIGS in the oracle (lines 448, 462), clipped.
    sigt = jnp.clip(jnp.full_like(T, sigs), 0.0, 1.0)

    # Oracle arithmetic is in hPa wherever PH/P appear with the 100./0.01
    # conversions, so convert once here rather than sprinkling factors.
    ph_hpa = p_half * _PA_TO_HPA
    p_hpa = p_full * _PA_TO_HPA
    dph = ph_hpa - _above(ph_hpa)                 # PH(I) - PH(I+1)  > 0
    dph_below = _below(ph_hpa) - ph_hpa           # PH(I-1) - PH(I)  > 0
    dp_below = _below(p_hpa) - p_hpa              # P(I-1) - P(I)    > 0
    h_below = _below(h_moist)
    qs_below = _below(qs)                         # QSTM
    T_above = _above(T)
    lv_above = _above(lv)
    gz_above = _above(gz)

    is_surface = (idx[None, :] == 0)
    is_top = (idx[None, :] == nlev - 1)

    # -- 3. Downward sweep (oracle DO 400 I=INB,1,-1) ----------------------
    # Sequential: level i needs WATER(i+1), WT(i+1), MP(i+1), QP(i+1).  In
    # surface-first order that is DESCENDING index, so the scan runs reversed.
    def step(carry, x):
        water_up, wt_up, mp_up, qp_up = carry
        (T_i, q_i, qs_i, wt_i, coeff_i, sigt_i, dph_i, dph_below_i,
         dp_below_i, wdtrain_i, ph_i, lv_i, h_i, h_below_i, gz_i, gz_up_i,
         T_up_i, lv_up_i, qs_below_i, surf_i, top_i) = x

        qsm = 0.5 * (q_i + qp_up)
        afac = coeff_i * ph_i * (qs_i - qsm) / (1.0e4 + 2.0e3 * ph_i * qs_i)
        afac = jnp.maximum(afac, 0.0)
        b6 = _HPA_TO_PA * dph_i * sigt_i * afac / wt_i
        c6 = (water_up * wt_up + wdtrain_i / sigd) / wt_i
        revap = 0.5 * (-b6 + _safe_sqrt(b6 * b6 + 4.0 * c6))
        evap_i = sigt_i * afac * revap
        water_i = revap * revap

        # Hydrostatic downdraft mass flux + inertia (oracle 769-777).  Skipped
        # at the lowest level, where MP keeps its zero initialisation.
        dhdp = jnp.maximum(
            _safe_divide(h_i - h_below_i, dp_below_i, _DENOM_FLOOR), dhdp_min)
        mp_raw = jnp.maximum(
            _HPA_TO_PA * ginv * lv_i * sigd
            * _safe_divide(evap_i, dhdp, _DENOM_FLOOR), 0.0)
        fac = _safe_divide(jnp.full_like(dph_below_i, inertia_scale_hPa),
                           dph_below_i, _DENOM_FLOOR)
        mp_i = jnp.where(surf_i, jnp.zeros_like(mp_raw),
                         (fac * mp_up + mp_raw) / (1.0 + fac))

        # QP (oracle 790-817).  Skipped at cloud top, where QP keeps its
        # initialisation QP(I) = Q(I-1) (line 501).
        rat = _safe_divide(mp_up, mp_i, _MP_FLOOR)
        qp_mix = (qp_up * rat + q_i * (1.0 - rat)
                  + _HPA_TO_PA * ginv * sigd * dph_i
                  * _safe_divide(evap_i, mp_i, _MP_FLOOR))
        qp_desc = _safe_divide(
            gz_up_i - gz_i + qp_up * (lv_up_i + T_up_i * (_CL_ORACLE - cpd))
            + cpd * (T_up_i - T_i),
            lv_i + T_i * (_CL_ORACLE - cpd), _DENOM_FLOOR)
        qp_new = jnp.where(mp_i > mp_up, qp_mix,
                           jnp.where(mp_up > 0.0, qp_desc, qp_up))
        qp_new = jnp.clip(qp_new, 0.0, qs_below_i)
        qp_i = jnp.where(top_i, qp_up, qp_new)

        return ((water_i, wt_i, mp_i, qp_i),
                (mp_i, qp_i, evap_i, water_i))

    # Carry above cloud top: WATER=0, WT=OMTSNOW, MP=0 (oracle 478-481) and
    # QP(INB+1)=Q(INB) from the QP initialisation at line 501.
    zeros = jnp.zeros_like(T[:, 0])
    carry0 = (zeros, jnp.full_like(zeros, omtsnow), zeros, q[:, -1])
    xs = (T, q, qs, wt, coeff, sigt, dph, dph_below, dp_below, wdtrain,
          ph_hpa, lv, h_moist, h_below, gz, gz_above, T_above, lv_above,
          qs_below, jnp.broadcast_to(is_surface, T.shape),
          jnp.broadcast_to(is_top, T.shape))
    xs = tuple(jnp.moveaxis(a, 1, 0) for a in xs)
    _carry, out = lax.scan(step, carry0, xs, reverse=True)
    mp, qp, evap, water = (jnp.moveaxis(a, 0, 1) for a in out)

    mp = _taper_mass_flux_to_surface(
        mp, p_full, taper_p_fraction=taper_p_fraction)
    # ``IF(I.EQ.1)GOTO 360`` leaves MP at the lowest level at zero; enforced
    # after the taper so neither path can reintroduce a surface mass flux.
    mp = jnp.where(is_surface, 0.0, mp)

    # -- 4. Whole-downdraft gate (oracle 717: IF(EP(INB).LT.1e-4) skip) ----
    # A column that detrains no precipitation has no shaft at all.  Smooth so
    # the gate is differentiable rather than a cliff at the threshold.
    gate = jax.nn.sigmoid(
        (ep[:, -1:] - ep_gate_threshold) / ep_gate_width)
    mp = mp * gate
    qp = qp * gate + q * (1.0 - gate)
    evap = evap * gate
    water = water * gate

    # -- 5. Tendencies (oracle 843-849 for i=0, 932-934 for i>0) -----------
    # One formula for every level: at i = 0 the oracle's expression is the
    # general one with MP(1) == 0, which the surface mask above guarantees.
    dpinv = _safe_divide(jnp.ones_like(dph), dph, _DENOM_FLOOR)  # 1/hPa
    lvcp = lv / cpn
    water_above = _above(water)
    wt_above = _above(wt)
    mp_above = _above(mp)
    qp_above = _above(qp)
    q_below = _below(q)

    dT_dt = (-sigd * lvcp * evap
             + sigd * wt_above * (_CL_ORACLE - cpd) * water_above
             * (T_above - T) * dpinv * _PA_TO_HPA / cpn)
    dq_v_dt = (sigd * evap
               + g * (mp_above * (qp_above - q) - mp * (qp - q_below))
               * dpinv * _PA_TO_HPA)

    # -- 6. Surface precipitation (oracle 822) -----------------------------
    # PRECIP = WT(1)*SIGD*WATER(1)*3600.*24000./(ROWL*G); the 24000 is
    # 24 h x 1000 mm/m, i.e. seconds-per-day times the mm conversion.
    precip_mm_day = (wt[:, 0] * sigd * water[:, 0] * _SEC_PER_DAY
                     * 1.0e3 / (_ROWL_ORACLE * g))

    return EmanuelDowndraftOutput(
        mp=mp, qp=qp, evap=evap, water=water, wt=wt,
        dT_dt=dT_dt, dq_v_dt=dq_v_dt, precip_mm_day=precip_mm_day,
    )
