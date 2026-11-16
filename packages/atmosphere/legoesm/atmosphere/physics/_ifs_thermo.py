"""IFS convection thermodynamics: fcttre.func.h, CUADJTQ and SATUR.

Faithful port of OpenIFS (github.com/ecmwf-ifs/openifs @ 8f6f722f73,
CY48R1 / OpenIFS 48r1.1) ``ifs-source/arpifs/phys_ec/cuadjtq.F90`` (the
``LPHYLIN = .FALSE.`` branch, KCALL 1, 2, 3 and 6), the statement functions of
``fcttre.func.h`` that the convection scheme calls, and the PQSEN formula of
``satur.F90`` (KFLAG = 0, i.e. FOEEWM) that the caller of CUMASTRN uses.

Every function works on ONE column point (scalars) or elementwise on arrays;
the convection scheme vmaps over columns.

Constants are a bundle (:class:`IFSThermo`), built by :func:`ifs_thermo` from
the base values.  The production default (:func:`lego_thermo`) is legoESM's
``legoesm.constants`` plus the liquid saturation curve of
``legoesm.thermo.saturation_vapor_pressure`` (the Tetens form IFS uses, with
legoESM's coefficients); the ice branch, the mixed-phase temperatures and the
von Karman constant are the IFS values below.  The oracle test feeds the
IFS-native bundle as well, so both are certified.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm import thermo

__physics_contract__ = {
    "summary": (
        "IFS fcttre statement functions (mixed-phase Tetens saturation, latent "
        "heats), CUADJTQ saturation adjustment (KCALL 1/2/3/6) and the SATUR "
        "PQSEN formula, one point at a time."
    ),
    "inputs": {"T": "K", "q": "kg kg-1 (specific humidity)", "p": "Pa"},
    "outputs": {"T": "K (adjusted)", "q": "kg kg-1 (adjusted)", "qsat": "kg kg-1"},
    "sign_convention": "condensation warms (T + L/cp * dq) and dries (q - dq)",
    "conserves": ["none"],
    "differentiable": True,
    "reference": "OpenIFS 8f6f722 cuadjtq.F90, fcttre.func.h, satur.F90",
    "idealized_test": "tests/atmosphere/test_bechtold_ifs_oracle.py",
}

# --- IFS thermodynamic constants without a legoESM counterpart (yoethf /
# suphec.F90 / sucumf.F90, OpenIFS 8f6f722) ---
R3IES = 22.587           # suphec.F90:241  ice Tetens slope
R4IES = -0.7             # suphec.F90:243  ice Tetens offset [K]
RTICE_OFFSET_K = 23.0    # suphec.F90:249  RTICE = RTT - 23 (mixed-phase end, cloud)
RTICECU_OFFSET_K = 38.0  # sucumf.F90:272  RTICECU = RTT - 38 (LMFGLAC, convection)
RTBER_OFFSET_K = 5.0     # suphec.F90:247-248  RTBER = RTBERCU = RTT - 5
RKAP = constants.kappa_von_karman   # parphy.F90 RKAP = 0.4 (same value)
QSAT_CAP = 0.5           # cuadjtq.F90 ZQMAX / satur.F90 MIN(0.5, ZQS)
# --- RULE 5 windows of the convection thermodynamics (#2281 part 2), used only
# when a caller passes an IfsSmooth ``sm`` (CUMASTRN); w = 0.01 x a scale of the
# switched quantity (P2 over the CUMASTRN oracle fixture,
# scripts/validate/oracle/bechtold_ifs/switch_scales.py); O(1) ratios 0.01 x
# the threshold. ---
_W_REL = 0.01
_S_TCU = 1.87             # [K] temperature (P2, fixture)
_S_CONDCU = 1.42e-6       # [kg/kg] condensation increment (P2, fixture)
_W_TCU, _W_CONDCU = _W_REL * _S_TCU, _W_REL * _S_CONDCU
_W_EPCU = _W_REL * QSAT_CAP   # e/p against QSAT_CAP (O(1) ratio)
RKOOP1 = 2.583           # suphec.F90:253  Koop homogeneous-nucleation fit
RKOOP2 = 0.48116e-2      # suphec.F90:254  [K-1]


class IFSThermo(NamedTuple):
    """yomcst / yoethf values used by the IFS convection (no defaults: build
    with :func:`ifs_thermo`)."""
    RG: float
    RD: float
    RV: float
    RCPD: float
    RLVTT: float
    RLMLT: float
    RLSTT: float
    RTT: float
    RA: float
    RPI: float
    RETV: float
    R2ES: float
    R3LES: float
    R4LES: float
    R3IES: float
    R4IES: float
    R5LES: float
    R5IES: float
    R5ALVCP: float
    R5ALSCP: float
    RALVDCP: float
    RALSDCP: float
    RALFDCP: float
    RTWAT: float
    RTBER: float
    RTBERCU: float
    RTICE: float
    RTICECU: float
    RTWAT_RTICE_R: float
    RTWAT_RTICECU_R: float
    RKOOP1: float = RKOOP1
    RKOOP2: float = RKOOP2


def ifs_thermo(*, RG, RD, RV, RCPD, RLVTT, RLMLT, RTT, RA, R2ES, R3LES, R4LES,
               R3IES=R3IES, R4IES=R4IES):
    """Derived constants exactly as suphec.F90 / sucumf.F90 form them."""
    RLSTT = RLVTT + RLMLT
    R5LES = R3LES * (RTT - R4LES)
    R5IES = R3IES * (RTT - R4IES)
    RTWAT = RTT
    RTICE = RTT - RTICE_OFFSET_K
    RTICECU = RTT - RTICECU_OFFSET_K
    return IFSThermo(
        RG=RG, RD=RD, RV=RV, RCPD=RCPD, RLVTT=RLVTT, RLMLT=RLMLT, RLSTT=RLSTT,
        RTT=RTT, RA=RA, RPI=2.0 * math.asin(1.0), RETV=RV / RD - 1.0,
        R2ES=R2ES, R3LES=R3LES, R4LES=R4LES, R3IES=R3IES, R4IES=R4IES,
        R5LES=R5LES, R5IES=R5IES,
        R5ALVCP=R5LES * RLVTT / RCPD, R5ALSCP=R5IES * RLSTT / RCPD,
        RALVDCP=RLVTT / RCPD, RALSDCP=RLSTT / RCPD, RALFDCP=RLMLT / RCPD,
        RTWAT=RTWAT, RTBER=RTT - RTBER_OFFSET_K, RTBERCU=RTT - RTBER_OFFSET_K,
        RTICE=RTICE, RTICECU=RTICECU,
        RTWAT_RTICE_R=1.0 / (RTWAT - RTICE),
        RTWAT_RTICECU_R=1.0 / (RTWAT - RTICECU))


def lego_thermo() -> IFSThermo:
    """legoESM constants; liquid Tetens = legoesm.thermo.saturation_vapor_pressure
    (e = E0*exp(A*(T-RTT)/(T-(RTT-B))), R2ES = E0*RD/RV as in suphec)."""
    return ifs_thermo(
        RG=constants.g, RD=constants.R_d, RV=constants.R_v, RCPD=constants.c_pd,
        RLVTT=constants.L_v, RLMLT=constants.L_f,  # latent-ok: IFS oracle constants RLVTT/RLMLT
        RTT=constants.T_freeze,
        RA=constants.R_earth,
        R2ES=thermo.TETENS_E0_PA * constants.R_d / constants.R_v,
        R3LES=thermo.TETENS_A,
        R4LES=constants.T_freeze - thermo.TETENS_B_K)


# ---------------------------------------------------------------------------
# fcttre.func.h
# ---------------------------------------------------------------------------
def foealfa(th, t):
    return jnp.minimum(1.0, ((jnp.maximum(th.RTICE, jnp.minimum(th.RTWAT, t))
                              - th.RTICE) * th.RTWAT_RTICE_R) ** 2)


def foealfcu(th, t, sm=None):
    """Convective liquid fraction.  With an IfsSmooth ``sm`` (not exact) the
    upper clip at RTWAT (a slope jump) is C1, smoothed on the cold side so the
    fraction never exceeds 1; the lower clip at RTICECU is already C1 (the
    ramp is quadratic).  The smooth form keeps no min(1, .) (it cannot bind)."""
    if sm is None:
        return jnp.minimum(1.0, ((jnp.maximum(th.RTICECU, jnp.minimum(th.RTWAT, t))
                                  - th.RTICECU) * th.RTWAT_RTICECU_R) ** 2)
    tc = th.RTWAT - sm.pos(th.RTWAT - t, _W_TCU)    # <= RTWAT: fraction <= 1
    a = ((jnp.maximum(th.RTICECU, tc) - th.RTICECU) * th.RTWAT_RTICECU_R) ** 2
    return jnp.minimum(1.0, a) if sm.hard else a


def ew_mixed(th, a, t):
    return th.R2ES * (a * jnp.exp(th.R3LES * (t - th.RTT) / (t - th.R4LES))
                      + (1.0 - a) * jnp.exp(th.R3IES * (t - th.RTT) / (t - th.R4IES)))


def foeeliq(th, t):
    """fcttre.func.h FOEELIQ: liquid-only Tetens."""
    return th.R2ES * jnp.exp(th.R3LES * (t - th.RTT) / (t - th.R4LES))


def foeeice(th, t):
    """fcttre.func.h FOEEICE: ice-only Tetens."""
    return th.R2ES * jnp.exp(th.R3IES * (t - th.RTT) / (t - th.R4IES))


def foeewm(th, t):
    return ew_mixed(th, foealfa(th, t), t)


def foeewmcu(th, t, sm=None):
    return ew_mixed(th, foealfcu(th, t, sm), t)


def dem_mixed(th, a, t):
    return (a * th.R5ALVCP * (1.0 / (t - th.R4LES) ** 2)
            + (1.0 - a) * th.R5ALSCP * (1.0 / (t - th.R4IES) ** 2))


def foedem(th, t):
    return dem_mixed(th, foealfa(th, t), t)


def foedemcu(th, t, sm=None):
    return dem_mixed(th, foealfcu(th, t, sm), t)


def foeldcpm(th, t):
    a = foealfa(th, t)
    return a * th.RALVDCP + (1.0 - a) * th.RALSDCP


def foeldcpmcu(th, t, sm=None):
    a = foealfcu(th, t, sm)
    return a * th.RALVDCP + (1.0 - a) * th.RALSDCP


def foelhmcu(th, t, sm=None):
    a = foealfcu(th, t, sm)
    return a * th.RLVTT + (1.0 - a) * th.RLSTT


def fminj(x, y):
    """cuadjtq.func.h FMINJ(X,Y) = Y - 0.5*(ABS(X-Y)-(X-Y))."""
    return y - 0.5 * (jnp.abs(x - y) - (x - y))


def satur_qsat(th, t, p, kflag=0):
    """satur.F90 (LDPHYLIN = .FALSE.): ZQS = MIN(0.5, E(T)/P), PQSAT = ZQS/(1 - RETV*ZQS),
    E = FOEEWM (KFLAG 0, as the convection driver forms PQSEN) or FOEEWMCU
    (KFLAG 1, as VDFHGHTN calls it)."""
    if kflag not in (0, 1):
        raise ValueError(f"satur_qsat: kflag must be 0 or 1, got {kflag!r}")
    e = foeewmcu(th, t) if kflag == 1 else foeewm(th, t)
    zqs = jnp.minimum(QSAT_CAP, e / p)
    return zqs / (1.0 - th.RETV * zqs)


# ---------------------------------------------------------------------------
# CUADJTQ (LPHYLIN = .FALSE.), one point
# ---------------------------------------------------------------------------
def _gate0(sm, zcond, zcond1):
    """where(zcond == 0, 0, zcond1); smooth: zcond1 x the one-sided weight of
    zcond > 0 (exact at zcond <= 0 and zcond >= the window)."""
    if sm is None or sm.hard:
        return jnp.where(zcond == 0.0, 0.0, zcond1)
    return zcond1 * sm.on(zcond, 0.0, _W_CONDCU)


def _pos(sm, x):
    return jnp.maximum(x, 0.0) if sm is None else sm.pos(x, _W_CONDCU)


def _cap(sm, x):
    return jnp.minimum(QSAT_CAP, x) if sm is None else sm.min(x, QSAT_CAP, _W_EPCU)


def _adj_mixed(th, psp, t, q, sm=None):
    """cuadjtq.F90 KCALL 1 (and the mixed-phase half of KCALL 6), :156-187."""
    zqp = 1.0 / psp

    def cond(t, q, cap):
        zl = 1.0 / (t - th.R4LES)
        zi = 1.0 / (t - th.R4IES)
        a = foealfcu(th, t, sm)
        zqsat = th.R2ES * (a * jnp.exp(th.R3LES * (t - th.RTT) * zl)
                           + (1.0 - a) * jnp.exp(th.R3IES * (t - th.RTT) * zi))
        zqsat = cap(zqsat * zqp)
        zcor = 1.0 - th.RETV * zqsat
        zf = a * th.R5ALVCP * zl ** 2 + (1.0 - a) * th.R5ALSCP * zi ** 2
        return (q * zcor ** 2 - zqsat * zcor) / (zcor ** 2 + zqsat * zf)

    # FMINJ is MIN in a different arithmetic: the same cap when smooth
    cap2 = (lambda x: fminj(QSAT_CAP, x)) if sm is None or sm.hard else (lambda x: _cap(sm, x))
    zcond = _pos(sm, cond(t, q, lambda x: _cap(sm, x)))
    t = t + foeldcpmcu(th, t, sm) * zcond
    q = q - zcond
    zcond1 = cond(t, q, cap2)
    zcond1 = _gate0(sm, zcond, zcond1)
    t = t + foeldcpmcu(th, t, sm) * zcond1
    q = q - zcond1
    return t, q


def _adj_liquid(th, psp, t, q, sm=None):
    """cuadjtq.F90 KCALL 6, liquid-only half (LDOFLAG), :191-215."""
    zqp = 1.0 / psp

    def cond(t, q, cap):
        zl = 1.0 / (t - th.R4LES)
        zqsat = cap(th.R2ES * jnp.exp(th.R3LES * (t - th.RTT) * zl) * zqp)
        zcor = 1.0 - th.RETV * zqsat
        zf = th.R5ALVCP * zl ** 2
        return (q * zcor ** 2 - zqsat * zcor) / (zcor ** 2 + zqsat * zf)

    cap2 = (lambda x: fminj(QSAT_CAP, x)) if sm is None or sm.hard else (lambda x: _cap(sm, x))
    zcond = _pos(sm, cond(t, q, lambda x: _cap(sm, x)))
    t = t + th.RALVDCP * zcond
    q = q - zcond
    zcond1 = cond(t, q, cap2)
    zcond1 = _gate0(sm, zcond, zcond1)
    t = t + th.RALVDCP * zcond1
    q = q - zcond1
    return t, q


def _adj_cu_unclamped(th, psp, t, q, clamp_first, clamp_second, sm=None):
    """cuadjtq.F90 KCALL 2 (:221-243) and KCALL 3 (:316-335): FOE*CU form."""
    zqp = 1.0 / psp

    def cond(t, q):
        zqsat = _cap(sm, foeewmcu(th, t, sm) * zqp)
        zcor = 1.0 / (1.0 - th.RETV * zqsat)
        zqsat = zqsat * zcor
        return (q - zqsat) / (1.0 + zqsat * zcor * foedemcu(th, t, sm))

    zcond = clamp_first(cond(t, q))
    t = t + foeldcpmcu(th, t, sm) * zcond
    q = q - zcond
    zcond1 = clamp_second(cond(t, q), zcond)
    t = t + foeldcpmcu(th, t, sm) * zcond1
    q = q - zcond1
    return t, q


def cuadjtq(th, kcall, psp, t, q, flag=True, oflag=None, sm=None):
    """CUADJTQ for one point.  ``kcall`` is a static python int; ``flag``
    (LDFLAG) is a boolean and ``oflag`` (LDOFLAG, KCALL 6 only) a boolean or,
    with ``sm``, a weight in [0, 1].  Points with ``flag`` False are returned
    unchanged (KCALL 3 has no flag).  ``sm`` (an IfsSmooth, CUMASTRN only):
    the condensation clamps, the saturation cap and the second-iteration gate
    take their C1 forms (windows ``_W_*CU``); None = exact."""
    if kcall == 1:
        tn, qn = _adj_mixed(th, psp, t, q, sm)
    elif kcall == 6:
        tm, qm = _adj_mixed(th, psp, t, q, sm)
        tl, ql = _adj_liquid(th, psp, t, q, sm)
        if sm is None or sm.hard:
            tn = jnp.where(oflag, tl, tm)
            qn = jnp.where(oflag, ql, qm)
        else:
            tn = oflag * tl + (1.0 - oflag) * tm
            qn = oflag * ql + (1.0 - oflag) * qm
    elif kcall == 2:
        if sm is None or sm.hard:
            tn, qn = _adj_cu_unclamped(
                th, psp, t, q, lambda c: jnp.minimum(c, 0.0),
                lambda c1, c: jnp.where(c == 0.0, jnp.minimum(c1, 0.0), c1), sm)
        else:
            # min(c, 0) = -pos(-c); the second clamp min(c1, 0) = c1 - pos(c1)
            # applies where the first one bound (c = 0): weight 1 - on(-c, 0)
            tn, qn = _adj_cu_unclamped(
                th, psp, t, q, lambda c: -sm.pos(-c, _W_CONDCU),
                lambda c1, c: c1 - (1.0 - sm.on(-c, 0.0, _W_CONDCU))
                * sm.pos(c1, _W_CONDCU), sm)
    elif kcall == 3:
        return _adj_cu_unclamped(th, psp, t, q, lambda c: c, lambda c1, c: c1, sm)
    else:
        raise ValueError(f"cuadjtq: KCALL {kcall} is not ported (1, 2, 3, 6 are)")
    return jnp.where(flag, tn, t), jnp.where(flag, qn, q)
