"""RRTMG longwave gas optics as ecRad (OpenIFS 49r1, 8f6f722) calls it.

Line-by-line port of, in the oracle tree ``bld49/src/``:

* ``rrtm_setcoef_140gp.F90``                          -> :func:`rrtm_setcoef_140gp`
* ``rrtm_gas_optical_depth.F90`` + ``rrtm_taumol1..16.F90`` -> :func:`rrtm_gas_optical_depth`
* ``radiation_ifs_rrtm.F90`` planck_function_atmos/surf -> :func:`planck_function_atmos`,
  :func:`planck_function_surf`
* ``radiation_ifs_rrtm.F90`` gas_optics, LW part (McICA = non-reordered branch:
  level reversal, ``max(min_gas_od_lw, .)``, ``lw_emission*(1-lw_albedo)``)
  -> :func:`gas_optics_lw`

Inputs of the gas optics are the :mod:`.rrtmg_prepare` outputs (RRTM order,
index 0 = lowest layer, column first). Tables come only from :mod:`.tables`.
``ZTAUAERL`` is identically zero inside gas_optics (radiation_ifs_rrtm.F90:458)
and is not carried.

Fortran literals written WITHOUT ``_JPRB`` are default REAL(4) in the oracle
build; their float32 value is what the oracle multiplies by (taumol4 upper
g-point factors, taumol13 ``3.55E-4``), so they are stored as
``float(np.float32(...))``.

Smoothing (one static ``sharpness``; ``math.inf`` reproduces the oracle's hard
form exactly; the default 100 is in absolute units of each argument):

1. tropopause split ``ln p > 4.56`` (lower vs upper k-table formulas, every band
   and the setcoef branch quantities): per-layer blend weight
   ``w = 1 - sigmoid_switch(4.56 - ln p)`` (V1-lw); each branch is evaluated at
   its own table-valid ``jp`` (lower ``min(jp,12)``, upper ``max(jp,13)``) with
   fp/jt/jt1/fac/ratios recomputed for that jp (V2-lw).
2. setcoef ``Z_FP = MAX(-1, MIN(1, Z_FP))`` -> ``smooth_clamp``.
3. specparm ``MIN(ONEMINUS, .)`` (all eta parameters) -> ``smooth_min``.
4. taumol 3-point (``specparm < 0.125`` / ``> 0.875``) vs 2-point formulas ->
   sigmoid blend of the three formulas.
5. ratio adjustments (``ZRATN2O > 1.5``, ``ZRATCO2 > 3``) -> sigmoid blend.
6. taumol1 ``IF (ZPP < 250)`` correction -> sigmoid blend.
7. Planck lower cap ``T < 160 K`` -> ``smooth_max(T, 160)``.
8. ``max(min_gas_od_lw, od)`` -> ``smooth_max`` with sharpness / min_gas_od_lw
   (V3-lw, relative width).

Not smoothed: integer table indices (jp, jt, jt1, js, indfor, indself,
indminor, Planck ind and their clamps) feeding continuous interpolations; the
zero-column floors ``IF (col == 0) col = 1e-32*coldry`` (V4-lw); the integer
diagnostics laytrop/layswtch/laylow. KNOWN DIFFERENCE at the default sharpness:
see ``tests/atmosphere/test_ecrad_ifs_lw_gas_oracle.py`` (stored per-column
gaps, RULE 5).
"""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.smooth import (
    sigmoid_switch,
    smooth_clamp,
    smooth_max,
    smooth_min,
    smooth_relu,
)

from . import tables
from .rrtmg_prepare import const_operand

__physics_contract__ = {
    "summary": (
        "RRTMG-IFS longwave gas optics (ecRad 49r1, 140 g-points): setcoef, the "
        "16 taumol bands, Planck source at half levels and surface emission."
    ),
    "inputs": {
        "pavel": "hPa", "tavel": "K", "coldry": "molec/cm^2",
        "wbrodl": "molec/cm^2", "wkl": "molec/cm^2", "wx": "1e20 molec/cm^2",
        "temperature_hl": "K", "skin_temperature": "K", "lw_albedo": "1",
    },
    "outputs": {
        "od_lw": "1 (layer optical depth per g-point)",
        "planck_hl": "W/m^2 (per g-point, half levels)",
        "lw_emission": "W/m^2 (per g-point)",
    },
    "sign_convention": (
        "optical depth >= min_gas_od_lw; emission and Planck fluxes positive; "
        "od_lw/planck_hl top-down (ecRad order), internal arrays bottom-up (RRTM)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "OpenIFS 49r1 (8f6f722) rrtm_setcoef_140gp.F90, rrtm_taumol1..16.F90, "
        "radiation_ifs_rrtm.F90; Mlawer et al. 1997 JGR 102:16663"
    ),
    "idealized_test": (
        "tests/atmosphere/test_ecrad_ifs_lw_gas_oracle.py: stage-by-stage match "
        "to the 49r1 oracle dumps at 1e-12 relative (sharpness=inf)"
    ),
}

# --- setcoef constants (rrtm_setcoef_140gp.F90, line numbers of 49r1) ---
_STPFAC = 296.0 / 1013.0                  # :190
_JP_OFFSET = 36.0                         # :202 INT(36 - 5*(plog+0.04))
_JP_SLOPE = 5.0                           # :202, :215
_JP_PLOG_SHIFT = 0.04                     # :202
_JP_MAX = 58                              # :205
_JT_DT = 15.0                             # :225, :232, :237
_PLOG_TROP = 4.56                         # :252 (Z_PLOG > 4.56)
_FOR_T_LO = 332.0                         # :258
_FOR_DT_LO = 36.0                         # :258
_SELF_T0 = 188.0                          # :269, :329
_SELF_DT = 7.2                            # :269
_SELF_IOFF = 7                            # :270
_INDSELF_MAX = 9                          # :270
_FOR_DT_UP = 36.0                         # :329
_INDFOR_UP = 3                            # :330
_MINOR_T0 = 180.8                         # :278
_MINOR_DT = 7.2                           # :278
_INDMINOR_MAX = 18                        # :279
_COL_SCALE = 1.0e-20                      # :295-301
_COL_FLOOR = 1.0e-32                      # :302-304
_CO2REG = 3.55e-24                        # :305
_CO2MULT_A = 272.63                       # :307
_CO2MULT_B = 1919.4                       # :307
_CO2MULT_C = 8.7604e-4                    # :307
_LO_JP_MAX = 12                           # lower table: 13 pressures, jp+1 <= 13
_UP_JP_MIN = 13                           # upper table: IND0 uses jp-13 >= 0

# --- taumol constants ---
_ONEMINUS = 1.0 - 1.0e-6                  # radiation_ifs_rrtm.F90:354
_SPEC_LO = 0.125                          # rrtm_taumol3.F90 et al. 3-point switch
_SPEC_HI = 0.875
_JS_HIGH = 8                              # JS of the specparm > 0.875 branch (specmult in (7,8))
_ETA_A = 8.0                              # lower specmult
_ETA_B = 4.0                              # upper specmult
_ADJ_SCALE = 1.0e20                       # ZRATN2O/ZRATCO2 = 1e20*chi/chi_ref
_B1_P_CORR = 250.0                        # rrtm_taumol1.F90:217-220
_B1_A = 0.15
_B1_D_LO = 154.4
_B1_D_UP = 95.6                           # rrtm_taumol1.F90:254
_B2_A = 0.05                              # rrtm_taumol2.F90 ZCORRADJ
_B2_P0 = 100.0
_B2_D = 900.0
_B13_CO2_REF = 3.55e-4                    # rrtm_taumol13.F90:160 (_JPRB)
_B13_CO2_REF_R4 = float(np.float32(3.55e-4))   # rrtm_taumol13.F90:163 (REAL(4))
_B4_UP_FACTORS = tuple(float(np.float32(v)) for v in
                       (0.92, 0.88, 1.07, 1.1, 0.99, 0.88, 0.943))  # taumol4:317-323, g 8..14
_B7_UP_FACTORS = (0.92, 0.88, 1.07, 1.1, 0.99, 0.855)               # taumol7 upper, g 6..11
# ratio adjustments (threshold, offset, exponent): taumol3/9 n2o, 6/7/8/13 co2
_ADJ_N2O = (1.5, 0.5, 0.65)
_ADJ_B6 = (3.0, 2.0, 0.77)
_ADJ_B7_LO = (3.0, 3.0, 0.79)
_ADJ_B7_UP = (3.0, 2.0, 0.79)
_ADJ_B8 = (3.0, 2.0, 0.65)
_ADJ_B13 = (3.0, 2.0, 0.68)

# --- Planck (radiation_ifs_rrtm.F90:651-690) ---
_ZFLUXFAC = 2.0 * math.asin(1.0) * 1.0e4
_PLANCK_T_MIN = 160.0
_PLANCK_T_OFF = 159.0
_PLANCK_IND_MAX = 180

# CHI_MLS gas rows (yoerrtrf), zero-based: h2o, co2, o3, n2o, (co), ch4, o2
_H2O, _CO2, _O3, _N2O, _CH4 = 0, 1, 2, 3, 5
# WKL rows (rrtm_prepare_gases), zero-based
_W_H2O, _W_CO2, _W_O3, _W_N2O, _W_CH4, _W_O2 = 0, 1, 2, 3, 5, 6

NBANDS = 16
# Planck-fraction reference levels ZREFRAT_PLANCK_A/B (rrtm_taumol3:110, 4:105/106, 5:111/114)
_PLANCK_REF_LO = {3: 9, 4: 11, 5: 5}
_PLANCK_REF_UP = {4: 13, 5: 43}


def _t(name):
    return jnp.asarray(tables.table(name))


def _chi(gas, jp):
    """CHI_MLS(gas+1, jp) with jp 1-based (array)."""
    return jnp.take(_t("yoerrtrf.CHI_MLS")[gas], jp - 1, mode="clip")


def _chi_s(gas, jp):
    """CHI_MLS(gas+1, jp) with a static Fortran jp."""
    return float(tables.table("yoerrtrf.CHI_MLS")[gas, jp - 1])


def _ref(a, b, k):
    """CHI_MLS(a,k)/CHI_MLS(b,k) at a static reference level k (ZREFRAT_*)."""
    return _chi_s(a, k) / _chi_s(b, k)


def _trunc_int(x):
    return jnp.trunc(x).astype(jnp.int32)


def _x(a):
    return a[..., None]


# =====================================================================
# RRTM_SETCOEF_140GP
# =====================================================================
class Branch(NamedTuple):
    """jp-dependent setcoef quantities of ONE k-table branch (lower or upper)."""
    jp: jnp.ndarray
    jt: jnp.ndarray
    jt1: jnp.ndarray
    fac00: jnp.ndarray
    fac01: jnp.ndarray
    fac10: jnp.ndarray
    fac11: jnp.ndarray
    indfor: jnp.ndarray
    forfrac: jnp.ndarray
    rat_h2oco2: jnp.ndarray
    rat_h2oco2_1: jnp.ndarray
    rat_h2oo3: jnp.ndarray
    rat_h2oo3_1: jnp.ndarray
    rat_h2on2o: jnp.ndarray
    rat_h2on2o_1: jnp.ndarray
    rat_h2och4: jnp.ndarray
    rat_h2och4_1: jnp.ndarray
    rat_n2oco2: jnp.ndarray
    rat_n2oco2_1: jnp.ndarray
    rat_o3co2: jnp.ndarray
    rat_o3co2_1: jnp.ndarray


class LwCoef(NamedTuple):
    """RRTM_SETCOEF_140GP outputs. Oracle-named fields are the branch-selected
    (blended) values the oracle returns; ``lo``/``up`` carry each branch at its
    own jp and ``w`` the lower-branch weight (1 = troposphere) per layer."""
    fac00: jnp.ndarray
    fac01: jnp.ndarray
    fac10: jnp.ndarray
    fac11: jnp.ndarray
    forfac: jnp.ndarray
    forfrac: jnp.ndarray
    indfor: jnp.ndarray
    jp: jnp.ndarray
    jt: jnp.ndarray
    jt1: jnp.ndarray
    colh2o: jnp.ndarray
    colco2: jnp.ndarray
    colo3: jnp.ndarray
    coln2o: jnp.ndarray
    colch4: jnp.ndarray
    colo2: jnp.ndarray
    co2mult: jnp.ndarray
    colbrd: jnp.ndarray
    laytrop: jnp.ndarray
    layswtch: jnp.ndarray
    laylow: jnp.ndarray
    selffac: jnp.ndarray
    selffrac: jnp.ndarray
    indself: jnp.ndarray
    indminor: jnp.ndarray
    scaleminor: jnp.ndarray
    scaleminorn2: jnp.ndarray
    minorfrac: jnp.ndarray
    rat_h2oco2: jnp.ndarray
    rat_h2oco2_1: jnp.ndarray
    rat_h2oo3: jnp.ndarray
    rat_h2oo3_1: jnp.ndarray
    rat_h2on2o: jnp.ndarray
    rat_h2on2o_1: jnp.ndarray
    rat_h2och4: jnp.ndarray
    rat_h2och4_1: jnp.ndarray
    rat_n2oco2: jnp.ndarray
    rat_n2oco2_1: jnp.ndarray
    rat_o3co2: jnp.ndarray
    rat_o3co2_1: jnp.ndarray
    # carried for the taumols
    w: jnp.ndarray
    lo: Branch
    up: Branch
    pavel: jnp.ndarray
    coldry: jnp.ndarray
    wx: jnp.ndarray


def _plog_bounds(upper):
    """ln p range of a branch where Z_FP = MAX(-1, MIN(1, .)) does not bind
    (V9-lw): fp = 1 at PREFLOG(jp_max+1), fp = -1 at PREFLOG(jp_min) + 1/5."""
    pref = tables.table("yoerrtrf.PREFLOG")
    jmin, jmax = (_UP_JP_MIN, _JP_MAX) if upper else (1, _LO_JP_MAX)
    return float(pref[jmax]), float(pref[jmin - 1]) + 1.0 / _JP_SLOPE


def _branch(plog, tavel, jp, indfor, forfrac, upper, sharpness) -> Branch:
    """rrtm_setcoef_140gp.F90:213-238 and the CHI_MLS ratios at a given jp.

    V9-lw: the fp clamp is applied to ln p (branch bounds, :func:`_plog_bounds`)
    instead of to fp: exact inside the table, continuous across jp boundaries,
    equal to the oracle's clamped fp to ~1e-15 off the table at sharpness=inf."""
    preflog = jnp.take(_t("yoerrtrf.PREFLOG"), jp - 1, mode="clip")
    tref = _t("yoerrtrf.TREF")
    tr0 = jnp.take(tref, jp - 1, mode="clip")
    tr1 = jnp.take(tref, jp, mode="clip")                    # TREF(JP1)
    lo_b, hi_b = _plog_bounds(upper)
    fp = _JP_SLOPE * (preflog - smooth_clamp(plog, lo_b, hi_b, sharpness))
    jt = jnp.clip(_trunc_int(3.0 + (tavel - tr0) / const_operand(_JT_DT)), 1, 4)
    ft = ((tavel - tr0) / const_operand(_JT_DT)) - (jt - 3).astype(tavel.dtype)
    jt1 = jnp.clip(_trunc_int(3.0 + (tavel - tr1) / const_operand(_JT_DT)), 1, 4)
    ft1 = ((tavel - tr1) / const_operand(_JT_DT)) - (jt1 - 3).astype(tavel.dtype)
    compfp = 1.0 - fp

    def r(a, b, k):
        return _chi(a, jp + k) / _chi(b, jp + k)

    return Branch(
        jp=jp, jt=jt, jt1=jt1,
        fac00=compfp * (1.0 - ft), fac01=fp * (1.0 - ft1),
        fac10=compfp * ft, fac11=fp * ft1,
        indfor=indfor, forfrac=forfrac,
        rat_h2oco2=r(_H2O, _CO2, 0), rat_h2oco2_1=r(_H2O, _CO2, 1),
        rat_h2oo3=r(_H2O, _O3, 0), rat_h2oo3_1=r(_H2O, _O3, 1),
        rat_h2on2o=r(_H2O, _N2O, 0), rat_h2on2o_1=r(_H2O, _N2O, 1),
        rat_h2och4=r(_H2O, _CH4, 0), rat_h2och4_1=r(_H2O, _CH4, 1),
        rat_n2oco2=r(_N2O, _CO2, 0), rat_n2oco2_1=r(_N2O, _CO2, 1),
        rat_o3co2=r(_O3, _CO2, 0), rat_o3co2_1=r(_O3, _CO2, 1),
    )


def rrtm_setcoef_140gp(coldry, wbroad, wkl, pavel, tavel, wx,
                       sharpness: float = 100.0) -> LwCoef:
    """Port of RRTM_SETCOEF_140GP. All inputs (ncol, nlev) RRTM order, wkl
    (ncol, 35, nlev); wx (ncol, 4, nlev) is only carried for the taumols."""
    plog = jnp.log(pavel)
    jp = jnp.clip(_trunc_int(_JP_OFFSET - _JP_SLOPE * (plog + _JP_PLOG_SHIFT)), 1, _JP_MAX)
    lower = plog > _PLOG_TROP                                  # integer diagnostics only
    w = 1.0 - sigmoid_switch(_PLOG_TROP - plog, sharpness)     # 1 where Z_PLOG > 4.56
    water = wkl[:, _W_H2O] / coldry
    scalefac = pavel * _STPFAC / tavel
    forfac0 = scalefac / (1.0 + water)
    # lower-branch-only quantities
    f = (_FOR_T_LO - tavel) / const_operand(_FOR_DT_LO)
    indfor_lo = jnp.minimum(2, jnp.maximum(1, _trunc_int(f)))
    forfrac_lo = f - indfor_lo.astype(f.dtype)
    f = (tavel - _SELF_T0) / const_operand(_SELF_DT)
    indself = jnp.minimum(_INDSELF_MAX, jnp.maximum(1, _trunc_int(f) - _SELF_IOFF))
    selffrac = f - (indself + _SELF_IOFF).astype(f.dtype)
    # upper-branch-only quantities
    indfor_up = jnp.full_like(jp, _INDFOR_UP)
    forfrac_up = (tavel - _SELF_T0) / const_operand(_FOR_DT_UP) - 1.0
    # both branches
    scaleminor = pavel / tavel
    scaleminorn2 = (pavel / tavel) * (wbroad / (coldry + wkl[:, _W_H2O]))
    f = (tavel - _MINOR_T0) / const_operand(_MINOR_DT)
    indminor = jnp.minimum(_INDMINOR_MAX, jnp.maximum(1, _trunc_int(f)))
    minorfrac = f - indminor.astype(f.dtype)

    def col(i):
        return _COL_SCALE * wkl[:, i]

    def floor0(cx):
        # IF (col == 0) col = 1e-32*coldry; V4-lw smooth form, exact at inf:
        # col + floor * H(-col/floor) (H(x) = x >= 0 at inf; col >= 0).
        fl = _COL_FLOOR * coldry
        return cx + fl * sigmoid_switch(-cx / fl, sharpness)

    colh2o = col(_W_H2O)
    colco2 = floor0(col(_W_CO2))
    colo3 = col(_W_O3)
    coln2o = floor0(col(_W_N2O))
    colch4 = floor0(col(_W_CH4))
    colo2 = col(_W_O2)
    colbrd = _COL_SCALE * wbroad
    co2reg = _CO2REG * coldry
    co2mult = ((colco2 - co2reg) * _CO2MULT_A * jnp.exp(-_CO2MULT_B / tavel)
               / (_CO2MULT_C * tavel))
    selffac = colh2o * (water * forfac0)
    forfac = colh2o * forfac0

    lo = _branch(plog, tavel, jnp.minimum(jp, _LO_JP_MAX), indfor_lo, forfrac_lo, False,
                 sharpness)
    up = _branch(plog, tavel, jnp.maximum(jp, _UP_JP_MIN), indfor_up, forfrac_up, True,
                 sharpness)

    def bl(a, b):
        return w * a + (1.0 - w) * b

    def sel(a, b):
        return jnp.where(lower, a, b)

    laytrop = jnp.sum(lower, axis=-1).astype(jnp.int32)
    ncol = pavel.shape[0]
    return LwCoef(
        fac00=bl(lo.fac00, up.fac00), fac01=bl(lo.fac01, up.fac01),
        fac10=bl(lo.fac10, up.fac10), fac11=bl(lo.fac11, up.fac11),
        forfac=forfac, forfrac=bl(forfrac_lo, forfrac_up),
        indfor=sel(indfor_lo, indfor_up), jp=jp, jt=sel(lo.jt, up.jt), jt1=sel(lo.jt1, up.jt1),
        colh2o=colh2o, colco2=colco2, colo3=colo3, coln2o=coln2o, colch4=colch4,
        colo2=colo2, co2mult=co2mult, colbrd=colbrd,
        laytrop=laytrop, layswtch=jnp.zeros(ncol, jnp.int32),
        laylow=jnp.ones(ncol, jnp.int32),
        selffac=selffac, selffrac=selffrac, indself=indself, indminor=indminor,
        scaleminor=scaleminor, scaleminorn2=scaleminorn2, minorfrac=minorfrac,
        rat_h2oco2=bl(lo.rat_h2oco2, up.rat_h2oco2),
        rat_h2oco2_1=bl(lo.rat_h2oco2_1, up.rat_h2oco2_1),
        rat_h2oo3=lo.rat_h2oo3, rat_h2oo3_1=lo.rat_h2oo3_1,
        rat_h2on2o=lo.rat_h2on2o, rat_h2on2o_1=lo.rat_h2on2o_1,
        rat_h2och4=lo.rat_h2och4, rat_h2och4_1=lo.rat_h2och4_1,
        rat_n2oco2=lo.rat_n2oco2, rat_n2oco2_1=lo.rat_n2oco2_1,
        rat_o3co2=up.rat_o3co2, rat_o3co2_1=up.rat_o3co2_1,
        w=w, lo=lo, up=up, pavel=pavel, coldry=coldry, wx=wx,
    )


# =====================================================================
# taumol building blocks (shapes: per-layer (ncol, nlev), per-g (ncol, nlev, ng))
# =====================================================================
def _rows(tab, idx):
    """TAB(IDX, :) with a 1-based integer index array; out-of-table indices
    (only in inactive, zero-weight branches) are clipped."""
    return jnp.take(tab, idx - 1, axis=0, mode="clip")


def _lerp(tab, idx, frac):
    a = _rows(tab, idx)
    return a + _x(frac) * (_rows(tab, idx + 1) - a)


def _lerp2(tab, j, m, f, minorfrac):
    """K(J,M,:) eta + minor-temperature interpolation (taumol3 ZN2OM1/2 form)."""

    def g(jj, mm):
        return tab.at[jj - 1, mm - 1].get(mode="clip")

    m1 = g(j, m) + _x(f) * (g(j + 1, m) - g(j, m))
    m2 = g(j, m + 1) + _x(f) * (g(j + 1, m + 1) - g(j, m + 1))
    return m1 + _x(minorfrac) * (m2 - m1)


def _ind_lo(b: Branch, nsp, js):
    return (((b.jp - 1) * 5 + (b.jt - 1)) * nsp + js,
            (b.jp * 5 + (b.jt1 - 1)) * nsp)


def _ind_up(b: Branch, nsp, js):
    return (((b.jp - 13) * 5 + (b.jt - 1)) * nsp + js,
            ((b.jp - 12) * 5 + (b.jt1 - 1)) * nsp)


def _simple(colx, b: Branch, tab, upper, nsp):
    """COL * (FAC00*A(IND0) + FAC10*A(IND0+1) + FAC01*A(IND1) + FAC11*A(IND1+1)),
    IND0/IND1 = (...)*NSPA|NSPB + 1. NSPB(16) = 0 in the 49r1 tables, so band 16
    upper always reads ABSB(1:2); ported as is (oracle behaviour)."""
    i0, i1b = (_ind_up if upper else _ind_lo)(b, nsp, 1)
    i1 = i1b + 1
    return _x(colx) * (_x(b.fac00) * _rows(tab, i0) + _x(b.fac10) * _rows(tab, i0 + 1)
                       + _x(b.fac01) * _rows(tab, i1) + _x(b.fac11) * _rows(tab, i1 + 1))


def _eta_param(sp_raw, mult, sharpness):
    """MIN(ONEMINUS, specparm); specmult; JS = 1+INT(); FS = MOD(specmult, 1)."""
    sp = smooth_min(sp_raw, _ONEMINUS, sharpness)
    sm = mult * sp
    tr = jnp.trunc(sm)
    return sp, 1 + tr.astype(jnp.int32), sm - tr


def _eta(num, rat, other, mult, sharpness):
    speccomb = num + rat * other
    sp, js, fs = _eta_param(num / speccomb, mult, sharpness)
    return speccomb, sp, js, fs


def _major3(speccomb, sp, sm, fa, fb, tab, base, nsp, sharpness):
    """Lower-atmosphere major-species term with the 3-point (specparm < 0.125
    or > 0.875) / 2-point blend (rrtm_taumol3.F90:180-276 form).

    base = IND - JS; sm = specmult. V10-lw: each formula uses its branch-local
    stencil (low JS=1, high JS=8, mid JS clamped to [2,7]) so it extends
    smoothly through the thresholds; at sharpness=inf the active branch has
    exactly the oracle's JS/FS."""

    def at(ind, off):
        return _rows(tab, ind + off)

    def fk(p):
        p2 = p * p
        p4 = p2 * p2
        return p4, 1.0 - p - 2.0 * p4, p + p4

    fa_, fb_ = _x(fa), _x(fb)
    tr = jnp.trunc(sm)
    ind = base + 1                                     # low: JS = 1, FS = sm
    fs = sm
    k0, k1, k2 = (_x(v) for v in fk(fs - 1.0))
    t_low = (k0 * fa_ * at(ind, 0) + k1 * fa_ * at(ind, 1) + k2 * fa_ * at(ind, 2)
             + k0 * fb_ * at(ind, nsp) + k1 * fb_ * at(ind, nsp + 1) + k2 * fb_ * at(ind, nsp + 2))
    ind = base + _JS_HIGH                                     # high: JS = 8
    fs = sm - (_JS_HIGH - 1)
    k0, k1, k2 = (_x(v) for v in fk(-fs))
    t_high = (k2 * fa_ * at(ind, -1) + k1 * fa_ * at(ind, 0) + k0 * fa_ * at(ind, 1)
              + k2 * fb_ * at(ind, nsp - 1) + k1 * fb_ * at(ind, nsp) + k0 * fb_ * at(ind, nsp + 1))
    jsm = jnp.clip(1 + tr.astype(jnp.int32), 2, 7)     # mid: JS in [2, 7]
    ind = base + jsm
    fs_ = _x(sm - (jsm - 1).astype(sm.dtype))
    t_mid = ((1.0 - fs_) * fa_ * at(ind, 0) + fs_ * fa_ * at(ind, 1)
             + (1.0 - fs_) * fb_ * at(ind, nsp) + fs_ * fb_ * at(ind, nsp + 1))
    w_low = 1.0 - sigmoid_switch(sp - _SPEC_LO, sharpness)     # sp < 0.125
    w_high = 1.0 - sigmoid_switch(_SPEC_HI - sp, sharpness)    # sp > 0.875
    w_mid = 1.0 - w_low - w_high
    sc = _x(speccomb)
    return (_x(w_low) * (sc * t_low) + _x(w_high) * (sc * t_high)
            + _x(w_mid) * (sc * t_mid))


def _major_lo(e, e1, b: Branch, tab, nsp, sharpness):
    """ZTAU_MAJOR + ZTAU_MAJOR1 of the lower atmosphere (e = eta tuples)."""
    sc, sp, _, _ = e
    sc1, sp1, _, _ = e1
    i0b, i1b = _ind_lo(b, nsp, 0)
    return (_major3(sc, sp, _ETA_A * sp, b.fac00, b.fac10, tab, i0b, nsp, sharpness)
            + _major3(sc1, sp1, _ETA_A * sp1, b.fac01, b.fac11, tab, i1b, nsp, sharpness))


def _major_up(e, e1, b: Branch, tab, nsp):
    """Upper-atmosphere 2-point eta term (rrtm_taumol3.F90:330-339 form)."""
    sc, _, js, fs = e
    sc1, _, js1, fs1 = e1
    i0, i1b = _ind_up(b, nsp, js)
    i1 = i1b + js1
    fs_, fs1_ = _x(fs), _x(fs1)
    f00, f10, f01, f11 = _x(b.fac00), _x(b.fac10), _x(b.fac01), _x(b.fac11)

    def bt(i, off):
        return _rows(tab, i + off)

    return (_x(sc) * ((1.0 - fs_) * f00 * bt(i0, 0) + fs_ * f00 * bt(i0, 1)
                      + (1.0 - fs_) * f10 * bt(i0, nsp) + fs_ * f10 * bt(i0, nsp + 1))
            + _x(sc1) * ((1.0 - fs1_) * f01 * bt(i1, 0) + fs1_ * f01 * bt(i1, 1)
                         + (1.0 - fs1_) * f11 * bt(i1, nsp) + fs1_ * f11 * bt(i1, nsp + 1)))


def _self(c: LwCoef, tab):
    return _x(c.selffac) * _lerp(tab, c.indself, c.selffrac)


def _for(c: LwCoef, b: Branch, tab):
    return _x(c.forfac) * _lerp(tab, b.indfor, b.forfrac)


def _minor1(c: LwCoef, tab):
    return _lerp(tab, c.indminor, c.minorfrac)


def _pfrac_eta(tab, jpl, fpl):
    """FRACREF(IG,JPL) + FPL*(FRACREF(IG,JPL+1) - FRACREF(IG,JPL))."""
    return _lerp(tab.T, jpl, fpl)


def _pfrac_const(tab, shape):
    return jnp.broadcast_to(tab, shape + tab.shape)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _safe_pow(base, expo):
    """base**expo (base >= 0, 0 < expo < 1, expo a static Python float) whose
    tangent is 0 where base == 0 (V12-lw). Values are unchanged; the guard only
    removes the inf*0 = NaN that a base underflowed to 0 (fp32 softplus far
    below a threshold, or the hard max at sharpness=inf) puts in the gradient."""
    return base ** expo


@_safe_pow.defjvp
def _safe_pow_jvp(expo, primals, tangents):
    (b,), (db,) = primals, tangents
    pos = b > 0.0
    bs = jnp.where(pos, b, 1.0)
    # primal through the safe base too, so jvp-of-jvp never differentiates b**e at 0
    return (jnp.where(pos, bs ** expo, 0.0),
            jnp.where(pos, expo * bs ** (expo - 1.0), 0.0) * db)


def _adjcol(col, coldry, chi_rat, chi_col, spec, sharpness):
    """IF (ZRAT > thr) ADJFAC = off + (ZRAT-off)**expo ... ELSE col (blended)."""
    thr, off, expo = spec
    rat = _ADJ_SCALE * (col / coldry) / const_operand(chi_rat)
    # V5-lw: the inactive branch is frozen at its threshold value through the
    # smooth max max(rat-off, thr-off) = (thr-off) + relu(rat-thr), written with
    # softplus: logsumexp rounds the base to exactly 0 for thr == off (taumol7
    # lower) far below the threshold, and 0**(expo-1) then makes the gradient
    # NaN. The softplus form stays > 0 for rat >= 0 in fp64 at sharpness 100;
    # where it still reaches 0 (fp32, or the hard max at sharpness=inf),
    # _safe_pow (V12-lw) gives the power a zero tangent instead of NaN.
    base = (thr - off) + smooth_relu(rat - thr, sharpness)
    adj = (off + _safe_pow(base, expo)) * chi_col * coldry * _COL_SCALE
    ws = 1.0 - sigmoid_switch(thr - rat, sharpness)            # rat > thr
    return ws * adj + (1.0 - ws) * col


def _chi_step(gas, pavel, upper, sharpness):
    """CHI_MLS(gas, jp_b+1) of a branch as a smooth staircase in
    u = 36 - 5 (ln p + 0.04) (V11-lw): CHI(kmin+1) + sum_k (CHI(k+1) - CHI(k))
    H(u - k), k = kmin+1..kmax. At sharpness=inf it telescopes to
    CHI(clip(INT(u), kmin, kmax) + 1) (summation rounding ~1e-15)."""
    chi = tables.table("yoerrtrf.CHI_MLS")[gas]
    kmin, kmax = (_UP_JP_MIN, _JP_MAX) if upper else (1, _LO_JP_MAX)
    ks = np.arange(kmin + 1, kmax + 1)
    d = jnp.asarray(chi[ks] - chi[ks - 1])
    u = _JP_OFFSET - _JP_SLOPE * (jnp.log(pavel) + _JP_PLOG_SHIFT)
    return float(chi[kmin]) + jnp.sum(d * sigmoid_switch(_x(u) - ks, sharpness), axis=-1)


def _adj_jp(col, gas, upper, c: LwCoef, spec, sharpness):
    """_adjcol with the reference CHI_MLS(gas, jp+1) of the branch."""
    chi = _chi_step(gas, c.pavel, upper, sharpness)
    return _adjcol(col, c.coldry, chi, chi, spec, sharpness)


def _band_tables(b):
    m = f"yoerrta{b}."
    names = [n.split(".", 1)[1] for n in tables.load_tables() if n.startswith(m)]
    return {n: _t(m + n) for n in names}


def _nsp(b, upper):
    return int(tables.table("yoerrtwn.NSPB" if upper else "yoerrtwn.NSPA")[b - 1])


# =====================================================================
# the 16 bands: each returns (tau, pfrac) for one branch
# =====================================================================
def _band(b, c: LwCoef, upper, sharpness):
    tb = _band_tables(b)
    br = c.up if upper else c.lo
    shp = c.colh2o.shape
    ng = tb["FRACREFA"].shape[0]
    nsp = _nsp(b, upper)
    sh = sharpness
    if b == 1:
        if not upper:
            wlt = 1.0 - sigmoid_switch(c.pavel - _B1_P_CORR, sh)          # ZPP < 250
            corr_lo = 1.0 - _B1_A * (_B1_P_CORR - c.pavel) / const_operand(_B1_D_LO)
            corr = wlt * corr_lo + (1.0 - wlt) * 1.0
            tn2 = _x(c.colbrd * c.scaleminorn2) * _minor1(c, tb["KA_MN2"])
            tau = _x(corr) * (_simple(c.colh2o, br, tb["ABSA"], False, nsp)
                              + _self(c, tb["SELFREF"]) + _for(c, br, tb["FORREF"]) + tn2)
            return tau, _pfrac_const(tb["FRACREFA"], shp)
        corr = 1.0 - _B1_A * (c.pavel / const_operand(_B1_D_UP))
        tn2 = _x(c.colbrd * c.scaleminorn2) * _minor1(c, tb["KB_MN2"])
        tau = _x(corr) * (_simple(c.colh2o, br, tb["ABSB"], True, nsp)
                          + _for(c, br, tb["FORREF"]) + tn2)
        return tau, _pfrac_const(tb["FRACREFB"], shp)
    if b == 2:
        if not upper:
            corr = 1.0 - _B2_A * (c.pavel - _B2_P0) / const_operand(_B2_D)
            tau = _x(corr) * (_simple(c.colh2o, br, tb["ABSA"], False, nsp)
                              + _self(c, tb["SELFREF"]) + _for(c, br, tb["FORREF"]))
            return tau, _pfrac_const(tb["FRACREFA"], shp)
        tau = _simple(c.colh2o, br, tb["ABSB"], True, nsp) + _for(c, br, tb["FORREF"])
        return tau, _pfrac_const(tb["FRACREFB"], shp)
    if b in (3, 4, 5):
        if not upper:
            e = _eta(c.colh2o, br.rat_h2oco2, c.colco2, _ETA_A, sh)
            e1 = _eta(c.colh2o, br.rat_h2oco2_1, c.colco2, _ETA_A, sh)
            kpl = _PLANCK_REF_LO[b]
            _, _, jpl, fpl = _eta(c.colh2o, _ref(_H2O, _CO2, kpl), c.colco2, _ETA_A, sh)
            tau = (_major_lo(e, e1, br, tb["ABSA"], nsp, sh)
                   + _self(c, tb["SELFREF"]) + _for(c, br, tb["FORREF"]))
            if b == 3:
                _, _, jm, fm = _eta(c.colh2o, _ref(_H2O, _CO2, 3), c.colco2, _ETA_A, sh)
                adj = _adj_jp(c.coln2o, _N2O, upper, c, _ADJ_N2O, sh)
                tau = tau + _x(adj) * _lerp2(tb["KA_MN2O"], jm, c.indminor, fm, c.minorfrac)
            if b == 5:
                _, _, jm, fm = _eta(c.colh2o, _ref(_H2O, _CO2, 7), c.colco2, _ETA_A, sh)
                tau = (tau + _lerp2(tb["KA_MO3"], jm, c.indminor, fm, c.minorfrac) * _x(c.colo3)
                       + _x(c.wx[:, 0]) * tb["CCL4"])
            return tau, _pfrac_eta(tb["FRACREFA"], jpl, fpl)
        if b == 3:
            e = _eta(c.colh2o, br.rat_h2oco2, c.colco2, _ETA_B, sh)
            e1 = _eta(c.colh2o, br.rat_h2oco2_1, c.colco2, _ETA_B, sh)
            _, _, jm, fm = _eta(c.colh2o, _ref(_H2O, _CO2, 13), c.colco2, _ETA_B, sh)
            adj = _adj_jp(c.coln2o, _N2O, upper, c, _ADJ_N2O, sh)
            _, _, jpl, fpl = _eta(c.colh2o, _ref(_H2O, _CO2, 13), c.colco2, _ETA_B, sh)
            tau = (_major_up(e, e1, br, tb["ABSB"], nsp) + _for(c, br, tb["FORREF"])
                   + _x(adj) * _lerp2(tb["KB_MN2O"], jm, c.indminor, fm, c.minorfrac))
            return tau, _pfrac_eta(tb["FRACREFB"], jpl, fpl)
        e = _eta(c.colo3, br.rat_o3co2, c.colco2, _ETA_B, sh)
        e1 = _eta(c.colo3, br.rat_o3co2_1, c.colco2, _ETA_B, sh)
        kpl = _PLANCK_REF_UP[b]
        _, _, jpl, fpl = _eta(c.colo3, _ref(_O3, _CO2, kpl), c.colco2, _ETA_B, sh)
        tau = _major_up(e, e1, br, tb["ABSB"], nsp)
        if b == 4:
            fac = jnp.ones(ng).at[7:14].set(jnp.asarray(_B4_UP_FACTORS))
            tau = tau * fac
        else:
            tau = tau + _x(c.wx[:, 0]) * tb["CCL4"]
        return tau, _pfrac_eta(tb["FRACREFB"], jpl, fpl)
    if b == 6:
        if not upper:
            adj = _adj_jp(c.colco2, _CO2, upper, c, _ADJ_B6, sh)
            tau = (_simple(c.colh2o, br, tb["ABSA"], False, nsp) + _self(c, tb["SELFREF"])
                   + _for(c, br, tb["FORREF"])
                   + _x(c.wx[:, 1]) * tb["CFC11ADJ"] + _x(c.wx[:, 2]) * tb["CFC12"]
                   + _x(adj) * _minor1(c, tb["KA_MCO2"]))
        else:
            tau = 0.0 + _x(c.wx[:, 1]) * tb["CFC11ADJ"] + _x(c.wx[:, 2]) * tb["CFC12"]
        return tau, _pfrac_const(tb["FRACREFA"], shp)
    if b == 7:
        if not upper:
            sc = c.colh2o + br.rat_h2oo3 * c.colo3
            e = (sc,) + _eta_param(1.0 / (1.0 + br.rat_h2oo3 / c.colh2o * c.colo3), _ETA_A, sh)
            sc1 = c.colh2o + br.rat_h2oo3_1 * c.colo3
            e1 = (sc1,) + _eta_param(1.0 / (1.0 + br.rat_h2oo3_1 / c.colh2o * c.colo3), _ETA_A, sh)
            ref = _ref(_H2O, _O3, 3)
            _, jm, fm = _eta_param(1.0 / (1.0 + ref / c.colh2o * c.colo3), _ETA_A, sh)
            adj = _adj_jp(c.colco2, _CO2, upper, c, _ADJ_B7_LO, sh)
            _, jpl, fpl = _eta_param(1.0 / (1.0 + ref / c.colh2o * c.colo3), _ETA_A, sh)
            tau = (_major_lo(e, e1, br, tb["ABSA"], nsp, sh) + _self(c, tb["SELFREF"])
                   + _for(c, br, tb["FORREF"])
                   + _x(adj) * _lerp2(tb["KA_MCO2"], jm, c.indminor, fm, c.minorfrac))
            return tau, _pfrac_eta(tb["FRACREFA"], jpl, fpl)
        adj = _adj_jp(c.colco2, _CO2, upper, c, _ADJ_B7_UP, sh)
        tau = _simple(c.colo3, br, tb["ABSB"], True, nsp) + _x(adj) * _minor1(c, tb["KB_MCO2"])
        fac = jnp.ones(ng).at[5:11].set(jnp.asarray(_B7_UP_FACTORS))
        return tau * fac, _pfrac_const(tb["FRACREFB"], shp)
    if b == 8:
        adj = _adj_jp(c.colco2, _CO2, upper, c, _ADJ_B8, sh)
        if not upper:
            tau = (_simple(c.colh2o, br, tb["ABSA"], False, nsp) + _self(c, tb["SELFREF"])
                   + _for(c, br, tb["FORREF"])
                   + _x(adj) * _minor1(c, tb["KA_MCO2"])
                   + _x(c.colo3) * _minor1(c, tb["KA_MO3"])
                   + _x(c.coln2o) * _minor1(c, tb["KA_MN2O"])
                   + _x(c.wx[:, 2]) * tb["CFC12"] + _x(c.wx[:, 3]) * tb["CFC22ADJ"])
            return tau, _pfrac_const(tb["FRACREFA"], shp)
        tau = (_simple(c.colo3, br, tb["ABSB"], True, nsp)
               + _x(adj) * _minor1(c, tb["KB_MCO2"])
               + _x(c.coln2o) * _minor1(c, tb["KB_MN2O"])
               + _x(c.wx[:, 2]) * tb["CFC12"] + _x(c.wx[:, 3]) * tb["CFC22ADJ"])
        return tau, _pfrac_const(tb["FRACREFB"], shp)
    if b == 9:
        adj = _adj_jp(c.coln2o, _N2O, upper, c, _ADJ_N2O, sh)
        if not upper:
            e = _eta(c.colh2o, br.rat_h2och4, c.colch4, _ETA_A, sh)
            e1 = _eta(c.colh2o, br.rat_h2och4_1, c.colch4, _ETA_A, sh)
            _, _, jm, fm = _eta(c.colh2o, _ref(_H2O, _CH4, 3), c.colch4, _ETA_A, sh)
            _, _, jpl, fpl = _eta(c.colh2o, _ref(_H2O, _CH4, 9), c.colch4, _ETA_A, sh)
            tau = (_major_lo(e, e1, br, tb["ABSA"], nsp, sh) + _self(c, tb["SELFREF"])
                   + _for(c, br, tb["FORREF"])
                   + _x(adj) * _lerp2(tb["KA_MN2O"], jm, c.indminor, fm, c.minorfrac))
            return tau, _pfrac_eta(tb["FRACREFA"], jpl, fpl)
        tau = _simple(c.colch4, br, tb["ABSB"], True, nsp) + _x(adj) * _minor1(c, tb["KB_MN2O"])
        return tau, _pfrac_const(tb["FRACREFB"], shp)
    if b == 10:
        if not upper:
            tau = (_simple(c.colh2o, br, tb["ABSA"], False, nsp) + _self(c, tb["SELFREF"])
                   + _for(c, br, tb["FORREF"]))
            return tau, _pfrac_const(tb["FRACREFA"], shp)
        tau = _simple(c.colh2o, br, tb["ABSB"], True, nsp) + _for(c, br, tb["FORREF"])
        return tau, _pfrac_const(tb["FRACREFB"], shp)
    if b == 11:
        so2 = _x(c.colo2 * c.scaleminor)
        if not upper:
            tau = (_simple(c.colh2o, br, tb["ABSA"], False, nsp) + _self(c, tb["SELFREF"])
                   + _for(c, br, tb["FORREF"]) + so2 * _minor1(c, tb["KA_MO2"]))
            return tau, _pfrac_const(tb["FRACREFA"], shp)
        tau = (_simple(c.colh2o, br, tb["ABSB"], True, nsp) + _for(c, br, tb["FORREF"])
               + so2 * _minor1(c, tb["KB_MO2"]))
        return tau, _pfrac_const(tb["FRACREFB"], shp)
    if b == 12:
        if not upper:
            e = _eta(c.colh2o, br.rat_h2oco2, c.colco2, _ETA_A, sh)
            e1 = _eta(c.colh2o, br.rat_h2oco2_1, c.colco2, _ETA_A, sh)
            _, _, jpl, fpl = _eta(c.colh2o, _ref(_H2O, _CO2, 10), c.colco2, _ETA_A, sh)
            tau = (_major_lo(e, e1, br, tb["ABSA"], nsp, sh) + _self(c, tb["SELFREF"])
                   + _for(c, br, tb["FORREF"]))
            return tau, _pfrac_eta(tb["FRACREFA"], jpl, fpl)
        z = jnp.zeros(shp + (ng,))
        return z, z
    if b == 13:
        if not upper:
            e = _eta(c.colh2o, br.rat_h2on2o, c.coln2o, _ETA_A, sh)
            e1 = _eta(c.colh2o, br.rat_h2on2o_1, c.coln2o, _ETA_A, sh)
            _, _, jm, fm = _eta(c.colh2o, _ref(_H2O, _N2O, 1), c.coln2o, _ETA_A, sh)
            adj = _adjcol(c.colco2, c.coldry, _B13_CO2_REF, _B13_CO2_REF_R4, _ADJ_B13, sh)
            _, _, jmc, fmc = _eta(c.colh2o, _ref(_H2O, _N2O, 3), c.coln2o, _ETA_A, sh)
            _, _, jpl, fpl = _eta(c.colh2o, _ref(_H2O, _N2O, 5), c.coln2o, _ETA_A, sh)
            colco = jnp.zeros(shp)                                       # Z_COLCO = 0
            tau = (_major_lo(e, e1, br, tb["ABSA"], nsp, sh) + _self(c, tb["SELFREF"])
                   + _for(c, br, tb["FORREF"])
                   + _x(adj) * _lerp2(tb["KA_MCO2"], jm, c.indminor, fm, c.minorfrac)
                   + _x(colco) * _lerp2(tb["KA_MCO"], jmc, c.indminor, fmc, c.minorfrac))
            return tau, _pfrac_eta(tb["FRACREFA"], jpl, fpl)
        tau = _x(c.colo3) * _minor1(c, tb["KB_MO3"])
        return tau, _pfrac_const(tb["FRACREFB"], shp)
    if b == 14:
        if not upper:
            tau = (_simple(c.colco2, br, tb["ABSA"], False, nsp) + _self(c, tb["SELFREF"])
                   + _for(c, br, tb["FORREF"]))
            return tau, _pfrac_const(tb["FRACREFA"], shp)
        return _simple(c.colco2, br, tb["ABSB"], True, nsp), _pfrac_const(tb["FRACREFB"], shp)
    if b == 15:
        if not upper:
            e = _eta(c.coln2o, br.rat_n2oco2, c.colco2, _ETA_A, sh)
            e1 = _eta(c.coln2o, br.rat_n2oco2_1, c.colco2, _ETA_A, sh)
            ref = _ref(_N2O, _CO2, 1)
            _, _, jm, fm = _eta(c.coln2o, ref, c.colco2, _ETA_A, sh)
            _, _, jpl, fpl = _eta(c.coln2o, ref, c.colco2, _ETA_A, sh)
            tn2 = _x(c.colbrd * c.scaleminor) * _lerp2(tb["KA_MN2"], jm, c.indminor, fm,
                                                        c.minorfrac)
            tau = (_major_lo(e, e1, br, tb["ABSA"], nsp, sh) + _self(c, tb["SELFREF"])
                   + _for(c, br, tb["FORREF"]) + tn2)
            return tau, _pfrac_eta(tb["FRACREFA"], jpl, fpl)
        z = jnp.zeros(shp + (ng,))
        return z, z
    if b == 16:
        if not upper:
            e = _eta(c.colh2o, br.rat_h2och4, c.colch4, _ETA_A, sh)
            e1 = _eta(c.colh2o, br.rat_h2och4_1, c.colch4, _ETA_A, sh)
            _, _, jpl, fpl = _eta(c.colh2o, _ref(_H2O, _CH4, 6), c.colch4, _ETA_A, sh)
            tau = (_major_lo(e, e1, br, tb["ABSA"], nsp, sh) + _self(c, tb["SELFREF"])
                   + _for(c, br, tb["FORREF"]))
            return tau, _pfrac_eta(tb["FRACREFA"], jpl, fpl)
        return _simple(c.colch4, br, tb["ABSB"], True, nsp), _pfrac_const(tb["FRACREFB"], shp)
    raise ValueError(f"unknown RRTMG LW band {b}")


def rrtm_gas_optical_depth(c: LwCoef, sharpness: float = 100.0):
    """Port of RRTM_GAS_OPTICAL_DEPTH (taumol1..16, aerosol od = 0).

    Returns (tau, pfrac), both (ncol, nlev, 140), RRTM layer order (index 0 =
    lowest layer). Fortran ZOD_LW(ig, jlev, jcol) = tau[jcol, jlev, ig] and
    ZPFRAC(jcol, ig, jlev) = pfrac[jcol, jlev, ig]."""
    w = _x(c.w)
    taus, pfs = [], []
    for b in range(1, NBANDS + 1):
        t_lo, p_lo = _band(b, c, False, sharpness)
        t_up, p_up = _band(b, c, True, sharpness)
        taus.append(w * t_lo + (1.0 - w) * t_up)
        pfs.append(w * p_lo + (1.0 - w) * p_up)
    return jnp.concatenate(taus, axis=-1), jnp.concatenate(pfs, axis=-1)


# =====================================================================
# Planck functions (radiation_ifs_rrtm.F90:576-808)
# =====================================================================
def _planck_bands(temperature, sharpness):
    """planck_store(..., jband) for temperatures of any shape -> (..., 16)."""
    tc = smooth_max(temperature, _PLANCK_T_MIN, sharpness)       # cap below 160 K
    ind = jnp.minimum(_trunc_int(tc - _PLANCK_T_OFF), _PLANCK_IND_MAX)
    frac = tc - (ind.astype(tc.dtype) + _PLANCK_T_OFF)
    tot = _t("yoerrtwn.TOTPLNK")                                  # (181, 16)
    factor = _ZFLUXFAC * _t("yoerrtwn.DELWAVE")
    a = _rows(tot, ind)
    return factor * (a + _x(frac) * (_rows(tot, ind + 1) - a))


def _band_of_g():
    return jnp.asarray(tables.table("yoerrtftr.NGB")[:140] - 1)


def planck_function_atmos(temperature_hl, pfrac, sharpness: float = 100.0):
    """temperature_hl (ncol, nlev+1) top-down; pfrac (ncol, nlev, 140) RRTM
    order. Returns planck_hl (ncol, nlev+1, 140) top-down
    (Fortran planck_hl(jg, jlev, jcol) = out[jcol, jlev, jg])."""
    pst = _planck_bands(temperature_hl, sharpness)[..., _band_of_g()]
    prev = pfrac[:, ::-1]                       # top layer first
    pf_hl = jnp.concatenate([prev[:, :1], prev], axis=1)   # jlev=1 -> PFRAC(nlev)
    return pst * pf_hl


def planck_function_surf(skin_temperature, pfrac_surf, sharpness: float = 100.0):
    """skin_temperature (ncol,), pfrac_surf (ncol, 140) = PFRAC(:,:,1)."""
    return _planck_bands(skin_temperature, sharpness)[..., _band_of_g()] * pfrac_surf


# =====================================================================
# gas_optics LW part
# =====================================================================
class LwGasOptics(NamedTuple):
    od_lw: jnp.ndarray          # (ncol, nlev, 140) top-down
    planck_hl: jnp.ndarray      # (ncol, nlev+1, 140) top-down
    lw_emission: jnp.ndarray    # (ncol, 140)
    tau: jnp.ndarray            # (ncol, nlev, 140) RRTM order, before the floor
    pfrac: jnp.ndarray          # (ncol, nlev, 140) RRTM order
    coef: LwCoef


def lw_outputs(tau, min_gas_od_lw: float, sharpness: float = 100.0):
    """od_lw(:,jlev,jcol) = max(min_gas_od_lw, ZOD_LW(:,nlev+1-jlev,jcol)).
    min_gas_od_lw must be a Python float (the smooth width is relative to it,
    V3-lw)."""
    return smooth_max(tau[:, ::-1, :], min_gas_od_lw, sharpness / min_gas_od_lw)


def gas_optics_lw(prep, temperature_hl, skin_temperature, lw_albedo,
                  min_gas_od_lw: float, sharpness: float = 100.0) -> LwGasOptics:
    """LW part of radiation_ifs_rrtm gas_optics (simple surface, McICA).

    prep: :class:`.rrtmg_prepare.PreparedGases`; temperature_hl (ncol, nlev+1)
    top-down; skin_temperature (ncol,); lw_albedo (ncol, 140)."""
    c = rrtm_setcoef_140gp(prep.coldry, prep.wbrodl, prep.wkl, prep.pavel,
                           prep.tavel, prep.wx, sharpness)
    tau, pfrac = rrtm_gas_optical_depth(c, sharpness)
    planck_hl = planck_function_atmos(temperature_hl, pfrac, sharpness)
    emis = planck_function_surf(skin_temperature, pfrac[:, 0, :], sharpness) * (1.0 - lw_albedo)
    return LwGasOptics(lw_outputs(tau, min_gas_od_lw, sharpness), planck_hl, emis,
                       tau, pfrac, c)
