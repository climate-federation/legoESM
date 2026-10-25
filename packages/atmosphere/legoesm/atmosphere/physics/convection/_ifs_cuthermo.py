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


def foealfcu(th, t):
    return jnp.minimum(1.0, ((jnp.maximum(th.RTICECU, jnp.minimum(th.RTWAT, t))
                              - th.RTICECU) * th.RTWAT_RTICECU_R) ** 2)


def _ew_mixed(th, a, t):
    return th.R2ES * (a * jnp.exp(th.R3LES * (t - th.RTT) / (t - th.R4LES))
                      + (1.0 - a) * jnp.exp(th.R3IES * (t - th.RTT) / (t - th.R4IES)))


def foeewm(th, t):
    return _ew_mixed(th, foealfa(th, t), t)


def foeewmcu(th, t):
    return _ew_mixed(th, foealfcu(th, t), t)


def _dem(th, a, t):
    return (a * th.R5ALVCP * (1.0 / (t - th.R4LES) ** 2)
            + (1.0 - a) * th.R5ALSCP * (1.0 / (t - th.R4IES) ** 2))


def foedem(th, t):
    return _dem(th, foealfa(th, t), t)


def foedemcu(th, t):
    return _dem(th, foealfcu(th, t), t)


def foeldcpm(th, t):
    a = foealfa(th, t)
    return a * th.RALVDCP + (1.0 - a) * th.RALSDCP


def foeldcpmcu(th, t):
    a = foealfcu(th, t)
    return a * th.RALVDCP + (1.0 - a) * th.RALSDCP


def foelhmcu(th, t):
    a = foealfcu(th, t)
    return a * th.RLVTT + (1.0 - a) * th.RLSTT


def fminj(x, y):
    """cuadjtq.func.h FMINJ(X,Y) = Y - 0.5*(ABS(X-Y)-(X-Y))."""
    return y - 0.5 * (jnp.abs(x - y) - (x - y))


def satur_qsat(th, t, p):
    """satur.F90 KFLAG=0 (as the convection driver forms PQSEN):
    ZQS = MIN(0.5, FOEEWM(T)/P);  PQSEN = ZQS/(1 - RETV*ZQS)."""
    zqs = jnp.minimum(QSAT_CAP, foeewm(th, t) / p)
    return zqs / (1.0 - th.RETV * zqs)


# ---------------------------------------------------------------------------
# CUADJTQ (LPHYLIN = .FALSE.), one point
# ---------------------------------------------------------------------------
def _adj_mixed(th, psp, t, q):
    """cuadjtq.F90 KCALL 1 (and the mixed-phase half of KCALL 6), :156-187."""
    zqp = 1.0 / psp

    def cond(t, q, cap):
        zl = 1.0 / (t - th.R4LES)
        zi = 1.0 / (t - th.R4IES)
        a = foealfcu(th, t)
        zqsat = th.R2ES * (a * jnp.exp(th.R3LES * (t - th.RTT) * zl)
                           + (1.0 - a) * jnp.exp(th.R3IES * (t - th.RTT) * zi))
        zqsat = cap(zqsat * zqp)
        zcor = 1.0 - th.RETV * zqsat
        zf = a * th.R5ALVCP * zl ** 2 + (1.0 - a) * th.R5ALSCP * zi ** 2
        return (q * zcor ** 2 - zqsat * zcor) / (zcor ** 2 + zqsat * zf)

    zcond = jnp.maximum(cond(t, q, lambda x: jnp.minimum(QSAT_CAP, x)), 0.0)
    t = t + foeldcpmcu(th, t) * zcond
    q = q - zcond
    zcond1 = cond(t, q, lambda x: fminj(QSAT_CAP, x))
    zcond1 = jnp.where(zcond == 0.0, 0.0, zcond1)
    t = t + foeldcpmcu(th, t) * zcond1
    q = q - zcond1
    return t, q


def _adj_liquid(th, psp, t, q):
    """cuadjtq.F90 KCALL 6, liquid-only half (LDOFLAG), :191-215."""
    zqp = 1.0 / psp

    def cond(t, q, cap):
        zl = 1.0 / (t - th.R4LES)
        zqsat = cap(th.R2ES * jnp.exp(th.R3LES * (t - th.RTT) * zl) * zqp)
        zcor = 1.0 - th.RETV * zqsat
        zf = th.R5ALVCP * zl ** 2
        return (q * zcor ** 2 - zqsat * zcor) / (zcor ** 2 + zqsat * zf)

    zcond = jnp.maximum(cond(t, q, lambda x: jnp.minimum(QSAT_CAP, x)), 0.0)
    t = t + th.RALVDCP * zcond
    q = q - zcond
    zcond1 = cond(t, q, lambda x: fminj(QSAT_CAP, x))
    zcond1 = jnp.where(zcond == 0.0, 0.0, zcond1)
    t = t + th.RALVDCP * zcond1
    q = q - zcond1
    return t, q


def _adj_cu_unclamped(th, psp, t, q, clamp_first, clamp_second):
    """cuadjtq.F90 KCALL 2 (:221-243) and KCALL 3 (:316-335): FOE*CU form."""
    zqp = 1.0 / psp

    def cond(t, q):
        zqsat = jnp.minimum(QSAT_CAP, foeewmcu(th, t) * zqp)
        zcor = 1.0 / (1.0 - th.RETV * zqsat)
        zqsat = zqsat * zcor
        return (q - zqsat) / (1.0 + zqsat * zcor * foedemcu(th, t))

    zcond = clamp_first(cond(t, q))
    t = t + foeldcpmcu(th, t) * zcond
    q = q - zcond
    zcond1 = clamp_second(cond(t, q), zcond)
    t = t + foeldcpmcu(th, t) * zcond1
    q = q - zcond1
    return t, q


def cuadjtq(th, kcall, psp, t, q, flag=True, oflag=None):
    """CUADJTQ for one point.  ``kcall`` is a static python int; ``flag``
    (LDFLAG) and ``oflag`` (LDOFLAG, KCALL 6 only) are booleans.  Points
    with ``flag`` False are returned unchanged (KCALL 3 has no flag)."""
    if kcall == 1:
        tn, qn = _adj_mixed(th, psp, t, q)
    elif kcall == 6:
        tm, qm = _adj_mixed(th, psp, t, q)
        tl, ql = _adj_liquid(th, psp, t, q)
        tn = jnp.where(oflag, tl, tm)
        qn = jnp.where(oflag, ql, qm)
    elif kcall == 2:
        tn, qn = _adj_cu_unclamped(
            th, psp, t, q, lambda c: jnp.minimum(c, 0.0),
            lambda c1, c: jnp.where(c == 0.0, jnp.minimum(c1, 0.0), c1))
    elif kcall == 3:
        return _adj_cu_unclamped(th, psp, t, q, lambda c: c, lambda c1, c: c1)
    else:
        raise ValueError(f"cuadjtq: KCALL {kcall} is not ported (1, 2, 3, 6 are)")
    return jnp.where(flag, tn, t), jnp.where(flag, qn, q)
