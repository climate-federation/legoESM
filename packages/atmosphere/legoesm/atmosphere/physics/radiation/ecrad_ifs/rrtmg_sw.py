"""RRTMG shortwave gas optics as ecRad (OpenIFS 49r1, 8f6f722) calls it.

Line-by-line port of, in the oracle tree ``bld49/src/``:

* ``srtm_setcoef.F90``            -> :func:`srtm_setcoef`
* ``srtm_gas_optical_depth.F90``  -> :func:`srtm_gas_optical_depth`
* ``srtm_taumol16.F90`` ... ``srtm_taumol29.F90`` -> :func:`band_terms`
* ``radiation_ifs_rrtm.F90`` gas_optics, SW part (SRTM calls, the
  ``solar_irradiance/sum(ZINCSOL)`` scaling, level reversal and the
  ``max(min_gas_od_sw, .)`` floor, McICA = non-reordered branch)
  -> :func:`gas_optics_sw`

Inputs are the RRTM_PREPARE_GASES outputs (``ZPAVEL`` [hPa], ``ZTAVEL`` [K],
``ZCOLDRY``/``ZWKL`` [molec/cm^2]) in RRTM order (index 0 = lowest layer);
arrays are column-first: ``(ncol, nlev)`` and ``wkl (ncol, 35, nlev)``.
Tables come only from :mod:`.tables` (vote V35).

Fortran literals written WITHOUT ``_JPRB`` are default REAL(4) in the oracle
build (gfortran, no -fdefault-real-8): their float32 value is what the oracle
multiplies by, so they are stored below as ``float(np.float32(...))``.

Departures from 49r1, all decided by vote:
* V37: srtm_taumol16/17/27/28/29 keep ``I_LAYSOLFR`` as ONE scalar across the
  column loop of a block; it is ported PER COLUMN (ecRad 1.7.2 form). The two
  agree on the oracle inputs (tested) and differ for a block mixing columns
  whose LAYREFFR crossing sits at different layers (tested, documented there).
* Smoothing (one static ``sharpness``; ``math.inf`` = the oracle's hard form):
  the specparm clamp ``IF (specparm >= ONEMINUS)`` -> ``smooth_min``; the
  tropopause split (lower vs upper k-table formula) -> per-layer blend with
  ``sigmoid_switch(jpc - 13)``, ``jpc = 36 - 5 (ln p + 0.04)``.
  the solar-source layer selection (I_LAYSOLFR crossing tests on the
  continuous pressure index, see :func:`solar_source`); the ``cos_sza > 0``
  day/night branch -> ``smoothstep_cubic(cos_sza*sharpness)``. Still hard:
  ``max(min_gas_od_sw, od)`` (PENDING user decision: every smooth form at the
  approved 0.01 width biases the 63% of layers with od < 0.01) and the exact
  ``IF (col == 0) col = 1e-32*coldry`` replacement.
* Out-of-bounds cases of the oracle (votes): ``K_JP(lay-1)`` at lay=1 counts
  as no crossing; a column with no layer above LAYTROP takes the default layer.
  The tropopause split uses the per-layer ``jp >= 13`` test of SRTM_SETCOEF;
  it equals the taumol ``lay > LAYTROP`` count test whenever pressure is
  monotone (otherwise the oracle indexes its tables out of bounds).
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.smooth import sigmoid_switch, smooth_clamp, smooth_min, smoothstep_cubic

from . import tables

__physics_contract__ = {
    "summary": (
        "ecRad 49r1 RRTMG (SRTM) shortwave gas optics: per-g-point layer optical "
        "depth, single-scattering albedo (Rayleigh/total) and TOA incoming solar "
        "flux for the 112 SW g-points (14 bands)."
    ),
    "inputs": {
        "pavel": "hPa", "tavel": "K", "coldry": "molec/cm^2",
        "wkl": "molec/cm^2", "cos_sza": "1", "solar_irradiance": "W/m^2",
    },
    "outputs": {"od_sw": "1", "ssa_sw": "1", "incoming_sw": "W/m^2"},
    "sign_convention": (
        "od_sw >= min_gas_od_sw >= 0, ssa_sw in [0, 1], incoming_sw >= 0 and "
        "sums to solar_irradiance over g-points in daylight; output levels are "
        "top-down (ecRad order), RRTM inputs bottom-up."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Mlawer & Clough (1997) RRTM-SW; Iacono et al. (2008) JGR 113 D13103; "
        "oracle OpenIFS 49r1 8f6f722 radiation_ifs_rrtm.F90 / srtm_*.F90."
    ),
    "idealized_test": (
        "tests/atmosphere/test_ecrad_ifs_sw_gas_oracle.py: every stage and the "
        "end-to-end outputs match the Fortran oracle to 1e-12 (fp64); night "
        "columns give od=ssa=incoming=0."
    ),
}

# --- radiation_ifs_rrtm.F90:353 ---------------------------------------------
_ONEMINUS = 1.0 - 1.0e-6  # ZONEMINUS = 1.0_jprb - 1.0e-6_jprb

# --- srtm_setcoef.F90 (lines 91-201) ----------------------------------------
_STPFAC = 296.0 / 1013.0  # Z_STPFAC = 296._JPRB/1013._JPRB
_JP_OFFSET = 36.0  # INT(36._JPRB - 5._JPRB*(Z_PLOG+0.04_JPRB))
_JP_SLOPE = 5.0
_JP_PLOG_SHIFT = 0.04
_JP_MAX = 58
_JP_TROP = 13  # IF (KJP >= 13) GO TO 5300 (above LAYTROP)
_JT_STEP = 15.0  # INT(3. + (PTAVEL-TREF)/15.)
_FP_SCALE = 5.0  # Z_FP = 5. * (PREFLOG(JP) - Z_PLOG)
_FOR_T_LO = 332.0  # Z_FACTOR = (332.0-PTAVEL)/36.0
_FOR_DT = 36.0
_SELF_T0 = 188.0  # (PTAVEL-188.0)/7.2, (PTAVEL-188.0)/36.0
_SELF_DT = float(np.float32(7.2))  # REAL(4) literal
_SELF_OFFSET = 7
_INDSELF_MAX = 9
_INDFOR_UP = 3
_COL_SCALE = float(np.float32(1.0e-20))  # REAL(4) literal 1.E-20
_COL_FLOOR = float(np.float32(1.0e-32))  # REAL(4) literal 1.E-32
_WKL_H2O, _WKL_CO2, _WKL_O3, _WKL_CH4, _WKL_O2 = 0, 1, 2, 5, 6  # PWKL(:,1/2/3/6/7,:)

# --- susrtm.F90:46-47 (NSPA/NSPB, not exported as tables) -------------------
_NSPA = dict(zip(range(16, 30), (9, 9, 9, 9, 1, 9, 9, 1, 9, 1, 0, 1, 9, 1)))
_NSPB = dict(zip(range(16, 30), (1, 5, 1, 1, 1, 5, 1, 0, 1, 0, 0, 1, 5, 1)))

# --- srtm_taumol*.F90 literals -----------------------------------------------
_SPEC_LO = 8.0  # Z_SPECMULT = 8.*(Z_SPECPARM)   (lower atmosphere)
_SPEC_UP = 4.0  # Z_SPECMULT = 4.*(Z_SPECPARM)   (upper, bands 17/21/28)
_O2ADJ_22 = 1.6  # srtm_taumol22.F90 Z_O2ADJ = 1.6_JPRB
_O2CONT_22 = float(np.float32(4.35e-4))  # 4.35e-4*P_COLO2/(350.0*2.0), REAL(4)
_O2CONT_DIV_22 = float(np.float32(350.0) * np.float32(2.0))

_BANDS = tuple(range(16, 30))
(_B16, _B17, _B18, _B19, _B20, _B21, _B22, _B23, _B24, _B25, _B26, _B27,
 _B28, _B29) = _BANDS
# Bands whose 49r1 I_LAYSOLFR is the scalar of V37 ("top": search above LAYTROP)
_TOP_BANDS = (16, 17, 27, 28, 29)


def _t(name):
    return jnp.asarray(tables.table(name))


def _ngc():
    return {b: int(n) for b, n in zip(_BANDS, tables.table("yoesrtwn.NGC"))}


def _trunc_int(x):
    """Fortran INT (truncation toward zero)."""
    return jnp.trunc(x).astype(jnp.int32)


class SwCoef(NamedTuple):
    """SRTM_SETCOEF outputs; ``*_lo``/``*_up`` are the below/above-LAYTROP branch
    values (srtm_setcoef.F90 lines 136-201), ``w_up`` the smooth split weight."""

    laytrop: jax.Array
    jp: jax.Array
    jt: jax.Array
    jt1: jax.Array
    fac00: jax.Array
    fac01: jax.Array
    fac10: jax.Array
    fac11: jax.Array
    colh2o: jax.Array
    colco2: jax.Array
    colo3: jax.Array
    colch4: jax.Array
    colo2: jax.Array
    colmol: jax.Array
    forfac: jax.Array
    forfrac_lo: jax.Array
    indfor_lo: jax.Array
    forfrac_up: jax.Array
    indfor_up: jax.Array
    selffac_lo: jax.Array
    selffrac_lo: jax.Array
    indself_lo: jax.Array
    w_up: jax.Array
    jpc: jax.Array

    def hard(self):
        """The oracle's per-layer branch selection (what SRTM_SETCOEF stores)."""
        up = self.jp >= _JP_TROP
        zero = jnp.zeros_like(self.selffac_lo)
        return dict(
            forfrac=jnp.where(up, self.forfrac_up, self.forfrac_lo),
            indfor=jnp.where(up, self.indfor_up, self.indfor_lo),
            selffac=jnp.where(up, zero, self.selffac_lo),
            selffrac=jnp.where(up, zero, self.selffrac_lo),
            indself=jnp.where(up, 0, self.indself_lo),
        )


def srtm_setcoef(pavel, tavel, coldry, wkl, sharpness: float = 100.0) -> SwCoef:
    """SRTM_SETCOEF for all columns (the oracle skips cos_sza<=0 columns, whose
    outputs it leaves stale; those columns are zeroed in gas_optics_sw)."""
    preflog = _t("yoesrtwn.PREFLOG")
    tref = _t("yoesrtwn.TREF")
    plog = jnp.log(pavel)
    jpc = _JP_OFFSET - _JP_SLOPE * (plog + _JP_PLOG_SHIFT)
    jp = jnp.clip(_trunc_int(jpc), 1, _JP_MAX)
    fp = _FP_SCALE * (preflog[jp - 1] - plog)
    dt0 = (tavel - tref[jp - 1]) / _JT_STEP
    dt1 = (tavel - tref[jp]) / _JT_STEP
    jt = jnp.clip(_trunc_int(3.0 + dt0), 1, 4)
    ft = dt0 - (jt - 3)
    jt1 = jnp.clip(_trunc_int(3.0 + dt1), 1, 4)
    ft1 = dt1 - (jt1 - 3)

    water = wkl[:, _WKL_H2O, :] / coldry
    scalefac = pavel * _STPFAC / tavel
    forfac = scalefac / (1.0 + water)
    # below LAYTROP
    factor = (_FOR_T_LO - tavel) / _FOR_DT
    indfor_lo = jnp.minimum(2, jnp.maximum(1, _trunc_int(factor)))
    forfrac_lo = factor - indfor_lo
    selffac_lo = water * forfac
    factor = (tavel - _SELF_T0) / _SELF_DT
    indself_lo = jnp.minimum(_INDSELF_MAX, jnp.maximum(1, _trunc_int(factor) - _SELF_OFFSET))
    selffrac_lo = factor - (indself_lo + _SELF_OFFSET)
    # above LAYTROP
    forfrac_up = (tavel - _SELF_T0) / _FOR_DT - 1.0

    colh2o = _COL_SCALE * wkl[:, _WKL_H2O, :]

    def _col(i):
        c = _COL_SCALE * wkl[:, i, :]
        return jnp.where(c == 0.0, _COL_FLOOR * coldry, c)

    compfp = 1.0 - fp
    return SwCoef(
        laytrop=jnp.sum(jp < _JP_TROP, axis=-1).astype(jnp.int32),
        jp=jp, jt=jt, jt1=jt1,
        fac00=compfp * (1.0 - ft), fac01=fp * (1.0 - ft1),
        fac10=compfp * ft, fac11=fp * ft1,
        colh2o=colh2o, colco2=_col(_WKL_CO2),
        colo3=_COL_SCALE * wkl[:, _WKL_O3, :], colch4=_col(_WKL_CH4),
        colo2=_col(_WKL_O2), colmol=_COL_SCALE * coldry + colh2o,
        forfac=forfac, forfrac_lo=forfrac_lo, indfor_lo=indfor_lo,
        forfrac_up=forfrac_up, indfor_up=jnp.full_like(jp, _INDFOR_UP),
        selffac_lo=selffac_lo, selffrac_lo=selffrac_lo, indself_lo=indself_lo,
        w_up=sigmoid_switch(jpc - _JP_TROP, sharpness),
        jpc=smooth_clamp(jpc, 1.0, _JP_MAX, sharpness),
    )


# ----------------------------------------------------------------------------
# k-table helpers (Fortran 1-based indices; out-of-branch indices are clipped
# and carry zero weight in the split blend)


def _g(tab, idx, ng):
    """tab(idx, 1:ng) -> (ncol, nlev, ng)."""
    return jnp.take(tab[:, :ng], idx - 1, axis=0, mode="clip")


def _interp4(tab, ind0, ind1, dt, c, ng):
    """A(ind0)*fac00 + A(ind0+dt)*fac10 + A(ind1)*fac01 + A(ind1+dt)*fac11."""
    f = lambda a: a[..., None]  # noqa: E731
    return (_g(tab, ind0, ng) * f(c.fac00) + _g(tab, ind0 + dt, ng) * f(c.fac10)
            + _g(tab, ind1, ng) * f(c.fac01) + _g(tab, ind1 + dt, ng) * f(c.fac11))


def _pair(tab, ind0, ind1, dt, fs, speccomb, c, ng):
    """Z_SPECCOMB*((1-Z_FS)*interp(IND) + Z_FS*interp(IND+1))."""
    return speccomb[..., None] * (
        (1.0 - fs[..., None]) * _interp4(tab, ind0, ind1, dt, c, ng)
        + fs[..., None] * _interp4(tab, ind0 + 1, ind1 + 1, dt, c, ng))


def _spec(a, b, strrat, mult, sharpness):
    """Z_SPECCOMB, JS, Z_FS with IF (Z_SPECPARM >= ONEMINUS) smoothed."""
    speccomb = a + strrat * b
    specparm = smooth_min(a / speccomb, _ONEMINUS, sharpness)
    specmult = mult * specparm
    return speccomb, 1 + _trunc_int(specmult), jnp.fmod(specmult, 1.0)


def _ind_lo(c, nsp, js):
    ind0 = ((c.jp - 1) * 5 + (c.jt - 1)) * nsp + js
    ind1 = (c.jp * 5 + (c.jt1 - 1)) * nsp + js
    return ind0, ind1


def _ind_up(c, nsp, js):
    ind0 = ((c.jp - 13) * 5 + (c.jt - 1)) * nsp + js
    ind1 = ((c.jp - 12) * 5 + (c.jt1 - 1)) * nsp + js
    return ind0, ind1


def _self(tab, c, ng):
    s0 = _g(tab, c.indself_lo, ng)
    return c.selffac_lo[..., None] * (s0 + c.selffrac_lo[..., None] * (_g(tab, c.indself_lo + 1, ng) - s0))


def _for(tab, c, ng, up):
    ind, frac = (c.indfor_up, c.forfrac_up) if up else (c.indfor_lo, c.forfrac_lo)
    f0 = _g(tab, ind, ng)
    return c.forfac[..., None] * (f0 + frac[..., None] * (_g(tab, ind + 1, ng) - f0))


def _sfl_js(tab, js, fs, ng):
    """SFLUXREFC(IG,JS) + Z_FS*(SFLUXREFC(IG,JS+1)-SFLUXREFC(IG,JS))."""
    t = tab[:ng].T
    s0 = jnp.take(t, js - 1, axis=0, mode="clip")
    return s0 + fs[..., None] * (jnp.take(t, js, axis=0, mode="clip") - s0)


def band_terms(b: int, c: SwCoef, sharpness: float = 100.0):
    """SRTM_TAUMOL<b>: (taug, taur, sflux_candidate, mode), each (ncol, nlev, ng).

    ``sflux_candidate[:, k]`` is what the band writes to P_SFLUXZEN if layer k
    is its solar-source layer; ``mode`` is "top" (search above LAYTROP, bands
    16/17/27/28/29) or "trop" (search below LAYTROP), see :func:`solar_source`.
    """
    ng = _ngc()[b]
    m = f"yoesrta{b}."
    t = lambda n: _t(m + n)  # noqa: E731
    e = lambda x: x[..., None]  # noqa: E731
    na, nb = _NSPA[b], _NSPB[b]
    shp = c.jp.shape + (ng,)
    one = lambda x: jnp.broadcast_to(x[:ng], shp)  # noqa: E731
    rayl = lambda: e(c.colmol) * t("RAYL")[0]  # noqa: E731
    raylc = lambda: e(c.colmol) * t("RAYLC")[:ng]  # noqa: E731
    h2o, co2, o3, ch4, o2 = c.colh2o, c.colco2, c.colo3, c.colch4, c.colo2

    if b in (_B16, _B18):
        strrat = t("STRRAT1" if b == _B16 else "STRRAT")[0]
        sc, js, fs = _spec(h2o, ch4, strrat, _SPEC_LO, sharpness)
        i0, i1 = _ind_lo(c, na, js)
        lo = _pair(t("ABSA"), i0, i1, na, fs, sc, c, ng) + e(h2o) * (
            _self(t("SELFREFC"), c, ng) + _for(t("FORREFC"), c, ng, False))
        i0, i1 = _ind_up(c, nb, 1)
        up = e(ch4) * _interp4(t("ABSB"), i0, i1, 1, c, ng)
        taur_lo = taur_up = rayl()
        if b == _B16:
            cand, mode = one(t("SFLUXREFC")), "top"
        else:
            cand, mode = _sfl_js(t("SFLUXREFC"), js, fs, ng), "trop"
    elif b in (_B17, _B21):
        strrat = t("STRRAT")[0]
        sc, js, fs = _spec(h2o, co2, strrat, _SPEC_LO, sharpness)
        i0, i1 = _ind_lo(c, na, js)
        lo = _pair(t("ABSA"), i0, i1, na, fs, sc, c, ng) + e(h2o) * (
            _self(t("SELFREFC"), c, ng) + _for(t("FORREFC"), c, ng, False))
        cand_lo = _sfl_js(t("SFLUXREFC"), js, fs, ng)
        scu, jsu, fsu = _spec(h2o, co2, strrat, _SPEC_UP, sharpness)
        i0, i1 = _ind_up(c, nb, jsu)
        # P_COLH2O * P_FORFAC * (FORREFC(INDF) + ...), in the oracle's order
        f0 = _g(t("FORREFC"), c.indfor_up, ng)
        up = _pair(t("ABSB"), i0, i1, nb, fsu, scu, c, ng) + e(h2o) * e(c.forfac) * (
            f0 + e(c.forfrac_up) * (_g(t("FORREFC"), c.indfor_up + 1, ng) - f0))
        taur_lo = taur_up = rayl()
        if b == _B17:
            cand, mode = _sfl_js(t("SFLUXREFC"), jsu, fsu, ng), "top"
        else:
            cand, mode = cand_lo, "trop"
    elif b == _B19:
        sc, js, fs = _spec(h2o, co2, t("STRRAT")[0], _SPEC_LO, sharpness)
        i0, i1 = _ind_lo(c, na, js)
        lo = _pair(t("ABSA"), i0, i1, na, fs, sc, c, ng) + e(h2o) * (
            _self(t("SELFREFC"), c, ng) + _for(t("FORREFC"), c, ng, False))
        i0, i1 = _ind_up(c, nb, 1)
        up = e(co2) * _interp4(t("ABSB"), i0, i1, 1, c, ng)
        taur_lo = taur_up = rayl()
        cand, mode = _sfl_js(t("SFLUXREFC"), js, fs, ng), "trop"
    elif b == _B20:
        i0, i1 = _ind_lo(c, na, 1)
        lo = e(h2o) * (_interp4(t("ABSA"), i0, i1, 1, c, ng) + _self(t("SELFREFC"), c, ng)
                       + _for(t("FORREFC"), c, ng, False)) + e(ch4) * t("ABSCH4C")[:ng]
        i0, i1 = _ind_up(c, nb, 1)
        up = e(h2o) * (_interp4(t("ABSB"), i0, i1, 1, c, ng)
                       + _for(t("FORREFC"), c, ng, True)) + e(ch4) * t("ABSCH4C")[:ng]
        taur_lo = taur_up = rayl()
        cand, mode = one(t("SFLUXREFC")), "trop"
    elif b == _B22:
        o2cont = e(_O2CONT_22 * o2 / _O2CONT_DIV_22)
        sc, js, fs = _spec(h2o, o2, _O2ADJ_22 * t("STRRAT")[0], _SPEC_LO, sharpness)
        i0, i1 = _ind_lo(c, na, js)
        lo = _pair(t("ABSA"), i0, i1, na, fs, sc, c, ng) + e(h2o) * (
            _self(t("SELFREFC"), c, ng) + _for(t("FORREFC"), c, ng, False)) + o2cont
        i0, i1 = _ind_up(c, nb, 1)
        up = e(o2 * _O2ADJ_22) * _interp4(t("ABSB"), i0, i1, 1, c, ng) + o2cont
        taur_lo = taur_up = rayl()
        cand, mode = _sfl_js(t("SFLUXREFC"), js, fs, ng), "trop"
    elif b == _B23:
        i0, i1 = _ind_lo(c, na, 1)
        lo = e(h2o) * (t("GIVFAC")[0] * _interp4(t("ABSA"), i0, i1, 1, c, ng)
                       + _self(t("SELFREFC"), c, ng) + _for(t("FORREFC"), c, ng, False))
        up = jnp.zeros(shp)
        taur_lo = taur_up = raylc()
        cand, mode = one(t("SFLUXREFC")), "trop"
    elif b == _B24:
        sc, js, fs = _spec(h2o, o2, t("STRRAT")[0], _SPEC_LO, sharpness)
        i0, i1 = _ind_lo(c, na, js)
        lo = (_pair(t("ABSA"), i0, i1, na, fs, sc, c, ng) + e(o3) * t("ABSO3AC")[:ng]
              + e(h2o) * (_self(t("SELFREFC"), c, ng) + _for(t("FORREFC"), c, ng, False)))
        taur_lo = e(c.colmol) * _sfl_js(t("RAYLAC"), js, fs, ng)
        i0, i1 = _ind_up(c, nb, 1)
        up = e(o2) * _interp4(t("ABSB"), i0, i1, 1, c, ng) + e(o3) * t("ABSO3BC")[:ng]
        taur_up = e(c.colmol) * t("RAYLBC")[:ng]
        cand, mode = _sfl_js(t("SFLUXREFC"), js, fs, ng), "trop"
    elif b == _B25:
        i0, i1 = _ind_lo(c, na, 1)
        lo = e(h2o) * _interp4(t("ABSA"), i0, i1, 1, c, ng) + e(o3) * t("ABSO3AC")[:ng]
        up = e(o3) * t("ABSO3BC")[:ng]
        taur_lo = taur_up = raylc()
        cand, mode = one(t("SFLUXREFC")), "trop"
    elif b == _B26:
        lo = up = jnp.zeros(shp)
        taur_lo = taur_up = raylc()
        cand, mode = one(t("SFLUXREFC")), "trop26"
    elif b == _B27:
        i0, i1 = _ind_lo(c, na, 1)
        lo = e(o3) * _interp4(t("ABSA"), i0, i1, 1, c, ng)
        i0, i1 = _ind_up(c, nb, 1)
        up = e(o3) * _interp4(t("ABSB"), i0, i1, 1, c, ng)
        taur_lo = taur_up = raylc()
        cand, mode = one(t("SCALEKUR")[0] * t("SFLUXREFC")), "top"
    elif b == _B28:
        strrat = t("STRRAT")[0]
        sc, js, fs = _spec(o3, o2, strrat, _SPEC_LO, sharpness)
        i0, i1 = _ind_lo(c, na, js)
        lo = _pair(t("ABSA"), i0, i1, na, fs, sc, c, ng)
        scu, jsu, fsu = _spec(o3, o2, strrat, _SPEC_UP, sharpness)
        i0, i1 = _ind_up(c, nb, jsu)
        up = _pair(t("ABSB"), i0, i1, nb, fsu, scu, c, ng)
        taur_lo = taur_up = rayl()
        cand, mode = _sfl_js(t("SFLUXREFC"), jsu, fsu, ng), "top"
    elif b == _B29:
        i0, i1 = _ind_lo(c, na, 1)
        lo = e(h2o) * (_interp4(t("ABSA"), i0, i1, 1, c, ng) + _self(t("SELFREFC"), c, ng)
                       + _for(t("FORREFC"), c, ng, False)) + e(co2) * t("ABSCO2C")[:ng]
        i0, i1 = _ind_up(c, nb, 1)
        up = e(co2) * _interp4(t("ABSB"), i0, i1, 1, c, ng) + e(h2o) * t("ABSH2OC")[:ng]
        taur_lo = taur_up = rayl()
        cand, mode = one(t("SFLUXREFC")), "top"
    else:
        raise ValueError(f"no SRTM band {b}")

    w = e(c.w_up)
    taug = w * up + (1.0 - w) * lo
    taur = w * taur_up + (1.0 - w) * taur_lo
    return taug, taur, cand, mode


def solar_source(cand, jpc, w_up, mode, layreffr, sharpness: float = 100.0):
    """P_SFLUXZEN per column: the value written at the last layer where the
    band's ``IF (I_LAY == I_LAYSOLFR)`` holds, I_LAYSOLFR held PER COLUMN (V37).

    "top" (srtm_taumol16/17/27/28/29 upper loop): I_LAYSOLFR=nlev; for
    lay>laytrop, if jp(lay-1)<LAYREFFR<=jp(lay) then I_LAYSOLFR=lay.
    "trop" (18-25 lower loop): I_LAYSOLFR=laytrop; for lay<=laytrop, if
    jp(lay)<LAYREFFR<=jp(min(nlev,lay+1)) then I_LAYSOLFR=min(lay+1,laytrop).
    "trop26": no crossing test. So the result is the candidate at the last
    crossing (top: that layer; trop: min(layer+1, laytrop)), else at the
    default layer (top: nlev; trop: laytrop).

    Smoothed (re-vote): each crossing test ``jp >= LAYREFFR`` is
    ``sigmoid_switch(jpc - LAYREFFR)`` on the continuous clamped pressure index
    (exact at sharpness=inf since LAYREFFR is an integer and jp = INT(jpc)),
    ``lay > laytrop`` is the split weight ``w_up``, and "last crossing wins" is a
    lerp scan ``val = x*new + (1-x)*val`` (exact for x in {0, 1}).
    Oracle out-of-bounds cases (votes): no crossing at lay=1 (jp(0)); a "top"
    column with no layer above laytrop takes layer nlev; a "trop" column with
    laytrop=0 has no lower layer and gives 0.
    """
    sig = lambda x: sigmoid_switch(x, sharpness)  # noqa: E731
    lower = 1.0 - w_up  # lay <= laytrop
    zero_col = jnp.zeros_like(lower[:, :1])
    if mode != "trop26":
        ge = sig(jpc - layreffr)  # jp >= LAYREFFR
        lt = 1.0 - ge  # jp < LAYREFFR
    if mode == "top":
        x = w_up * jnp.concatenate([zero_col, lt[:, :-1]], -1) * ge
        new = cand
        init = cand[:, -1]
    else:
        lower_above = jnp.concatenate([lower[:, 1:], zero_col], -1)
        init = jnp.sum((lower * (1.0 - lower_above))[..., None] * cand, axis=1)
        if mode == "trop":
            ge_next = jnp.concatenate([ge[:, 1:], ge[:, -1:]], -1)  # MIN(nlev, lay+1)
            x = lower * lt * ge_next
        else:
            x = jnp.zeros_like(lower)
        cand_next = jnp.concatenate([cand[:, 1:], cand[:, -1:]], 1)
        new = lower_above[..., None] * cand_next + (1.0 - lower_above[..., None]) * cand

    def step(val, k):
        xk = x[:, k][:, None]
        return xk * new[:, k] + (1.0 - xk) * val, None

    val, _ = jax.lax.scan(step, init, jnp.arange(cand.shape[1]))
    return val


def _layreffr(b):
    """LAYREFFR of band b (srtm_taumol26 has none: no crossing test)."""
    name = f"yoesrta{b}.LAYREFFR"
    return int(tables.table(name)[0]) if name in tables.load_tables() else None


def srtm_gas_optical_depth(c: SwCoef, sharpness: float = 100.0):
    """SRTM_GAS_OPTICAL_DEPTH for daylight columns: (POD, PSSA, PINCSOL), shapes
    (ncol, nlev, 112), (ncol, nlev, 112), (ncol, 112), g-points band-ordered."""
    od, ssa, inc = [], [], []
    for b in _BANDS:
        taug, taur, cand, mode = band_terms(b, c, sharpness)
        inc.append(solar_source(cand, c.jpc, c.w_up, mode, _layreffr(b), sharpness))
        pod = taur + taug
        od.append(pod)
        ssa.append(taur / pod)
    return (jnp.concatenate(od, -1), jnp.concatenate(ssa, -1), jnp.concatenate(inc, -1))


def gas_optics_sw(pavel, tavel, coldry, wkl, cos_sza, solar_irradiance,
                  min_gas_od_sw, sharpness: float = 100.0):
    """SW part of radiation_ifs_rrtm.F90 gas_optics (McICA, no spectral solar
    scaling: config49r1_rrtmg.nam). Returns column-first, levels top-down:
    od_sw, ssa_sw (ncol, nlev, 112) and incoming_sw (ncol, 112)."""
    c = srtm_setcoef(pavel, tavel, coldry, wkl, sharpness)
    zod, zssa, zinc = srtm_gas_optical_depth(c, sharpness)
    return sw_outputs(zod, zssa, zinc, cos_sza, solar_irradiance, min_gas_od_sw, sharpness)


def _day_weight(cos_sza, sharpness):
    """IF (cos_sza > 0): smoothstep_cubic(cos_sza*sharpness) (re-vote), exactly 0
    for cos_sza <= 0 and 1 for cos_sza >= 1/sharpness; hard at sharpness=inf."""
    if sharpness == math.inf:
        return (cos_sza > 0.0).astype(jnp.result_type(cos_sza, float))
    return smoothstep_cubic(cos_sza * sharpness)


def sw_outputs(zod, zssa, zinc, cos_sza, solar_irradiance, min_gas_od_sw,
               sharpness: float = 100.0):
    """radiation_ifs_rrtm.F90 lines 589-650 from day-branch ZOD/ZSSA/ZINCSOL.

    Night columns are zero (the oracle's initialisation at line 563) and their
    incoming_sw_scale is 1; the day/night choice is the weight w of
    :func:`_day_weight`. The od floor is still the hard max (decision pending
    with the user: both smooth forms at width 0.01 bias thin layers).
    """
    w = _day_weight(cos_sza, sharpness)[:, None]
    zod = w[..., None] * zod
    zssa = w[..., None] * zssa
    zinc_m = w * zinc
    total = jnp.where((cos_sza > 0.0)[:, None], jnp.sum(zinc, -1, keepdims=True), 1.0)  # guards 0/0 only
    scale = w * (solar_irradiance / total) + (1.0 - w)
    od_sw = jnp.maximum(min_gas_od_sw, zod[:, ::-1, :])
    return od_sw, zssa[:, ::-1, :], scale * zinc_m
