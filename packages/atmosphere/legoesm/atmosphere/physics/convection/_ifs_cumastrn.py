"""IFS cumulus master routine CUMASTRN with downdraught, fluxes and tendencies.

Faithful port of OpenIFS (github.com/ecmwf-ifs/openifs @ 8f6f722f73,
OpenIFS 48r1.1) ``ifs-source/arpifs/phys_ec/{cumastrn,cudlfsn,cuddrafn,
cuflxn,cudtdqn,cubidiag,cududv}.F90``; the updraught routines are in
:mod:`._ifs_cuascent`, the thermodynamics in :mod:`._ifs_cuthermo`.

Configuration ported (sucumf.F90 defaults, the deterministic ECMWF physics):
LMFPEN = LMFSCV = LMFMID = LMFDD = LMFDUDV = LMFDSNOW = LMFWETB = LMFGLAC =
LSCVLIQ = LMFUVDIS = .TRUE.; LMFENTHCONS = LMFCUCA = LNEWTAU = LMFWSTAR =
LMFSMOOTH = .FALSE.; RMFSOLTQ = RMFSOLUV = 1, RMFSOLRHS = RMFADVW = 0,
RCAPDCYCL = 2, RCAPQADV = 0.8; LDTDKMF = LDMCAPEA = LPHYLIN = .FALSE.;
SPP off; no chemical tracers (KTRAC = 0); LEPCLD = .TRUE.; PPLRG = PPLDARE
= 1.  Branches of other switch values are NOT ported.

One column per call (KLON = 1, as the oracle was run), vmapped over columns.
Arrays are 1-based padded (see :mod:`._ifs_cuascent`).  ``land`` is the
LDLAND flag as a fraction: every ``IF(LDLAND)`` is evaluated as
``land*A + (1-land)*B``, which is the source exactly for land in {0, 1}
(legoESM's existing convention for fractional land; a decision is pending).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.smooth import smooth_max

from .mass_flux import ifs_ztaures
from ._ifs_cuthermo import cuadjtq, foealfcu, foeewmcu, foelhmcu
from ._ifs_cuascent import (
    cuascn, cubasen, cuinin, kget, kset, sdiv, spow, ssqrt,
    RMFCMIN, RDEPTHS, RMFLIA, RMFCFL, ENTRORG, DETRPEN, ENTSHALP, RPRCON, ZDNOPRC,
)

__physics_contract__ = {
    "summary": (
        "IFS mass-flux convection (Tiedtke 1989 / Bechtold et al. 2008, 2014) "
        "as CUMASTRN runs it: trigger, first-guess flux, single ascent, "
        "saturated downdraught (CUDLFSN/CUDDRAFN), CAPE / PBL / mid-level "
        "closures with rescaling, precipitation, melting and sub-cloud "
        "evaporation (CUFLXN), implicit T/q (CUDTDQN) and momentum (CUDUDV) "
        "tendencies, kinetic-energy dissipation heating."
    ),
    "inputs": {
        "T": "K", "q": "kg kg-1", "u": "m s-1", "v": "m s-1",
        "p_full": "Pa", "p_half": "Pa", "geo_full": "m2 s-2", "geo_half": "m2 s-2",
        "omega": "Pa s-1", "plitot": "kg kg-1", "qsat": "kg kg-1",
        "ptent/ptenq": "K s-1 / kg kg-1 s-1 (total tendency before convection)",
        "ptenta/ptenqa": "K s-1 / kg kg-1 s-1 (dynamics tendency)",
        "ahfs": "W m-2 (IFS sign: negative = upward)",
        "qhfl": "kg m-2 s-1 (IFS sign: negative = upward)",
        "land": "1", "dx": "m", "dt": "s",
    },
    "outputs": {
        "ptent/ptenq/ptenu/ptenv": "input tendency + convective tendency",
        "pmflxr/pmflxs": "kg m-2 s-1 (rain / snow flux on half levels, + down)",
        "plude, psnde": "kg m-2 s-1 (detrained condensate / precipitation)",
        "pmfu/pmfd": "kg m-2 s-1", "pcape": "J kg-1",
    },
    "sign_convention": (
        "mass fluxes positive upward (PMFD <= 0); precipitation fluxes "
        "positive downward; tendencies added to the incoming ones."
    ),
    "conserves": ["moisture"],
    "differentiable": True,
    "reference": "OpenIFS 8f6f722 cumastrn.F90 and callees (Apache-2.0)",
    "idealized_test": "tests/atmosphere/test_bechtold_ifs_oracle.py",
}

# --- yoecumf values (sucumf.F90 @ 8f6f722) ---
ENTRDD = 3.0e-4        # sucumf.F90:155  downdraught entrainment [1/m]
RMFDEPS = 0.30         # sucumf.F90:165  LFS downdraught mass-flux fraction
RCUCOV = 0.05          # sucumf.F90:186  convective precipitation area fraction
RCPECONS_G = 5.44e-4   # sucumf.F90:187  RCPECONS = 5.44E-4/RG
RCVRFACTOR = 5.09e-3   # sucumf.F90:188
RTAUMEL = 5.0 * 3.6e3 * 0.66   # sucumf.F90:189  melting time scale [s]
RHEBC = 0.92           # sucumf.F90:190  ocean RH break (sub-cloud evaporation)
RTAUA = 1.0            # sucumf.F90:193
RCAPQADV = 0.8         # sucumf.F90:237
RMINCAPE = 0.05        # sucumf.F90:239
RUVPER = 0.3           # sucumf.F90:256
# --- cuflxn.F90 fixed numbers ---
RHEBC_LAND = 0.75      # cuflxn.F90:221
RHEBC_LAND_DEEP = 0.70 # cuflxn.F90:222
RHEBC_DEEP = 0.85      # cuflxn.F90:225
ZTW1, ZTW2, ZTW3, ZTW4, ZTW5 = 1329.31, 0.0074615, 0.85e5, 40.637, 275.0  # :160-164
ZMELT_DT = 0.5         # :382  ZCONS1 = ZCONS1A*(1+0.5*(T-RTT))
ZDD_SCALE_MAX = 0.98   # :346
ZDD_TINY = 1.0e-15     # :347
ZEVAP_EXP = 0.5777     # :475
ZRFL_MIN = 1.0e-12     # :472,:489
ZRHM_MIN = 0.8         # :450
ZRHM_SCALE = 0.025     # :450
ZRCUCOV_DEEP = float(np.float32(0.6))   # :452  default-kind literal 0.6
# --- cudlfsn / cuddrafn ---
ZHSMIN0 = 1.0e8        # cudlfsn.F90:181
ZLFS_RAIN = 10.0       # cudlfsn.F90:256
ZDD_MIX_MIN = 0.3      # cuddrafn.F90:189
ZDD_DET_MAX = 0.75     # cuddrafn.F90:190
ZTD_MIN, ZTD_MAX = 100.0, 400.0   # cuddrafn.F90:209-210
# --- cumastrn fixed numbers ---
ZMFUB_DEEP = 0.1       # :560,:579
ZDQMIN_FRAC = 0.01     # :567
ZDQMIN_FLOOR = 1.0e-10 # :567
ZDH_DQ = 1.0e5         # :569
ZDX_MIN = 1.0e2        # :729
ZTAU_W0 = 2.0          # :803
ZTAU_WMAX = 15.0       # :803
ZPBL_DP = 50.0e2       # :813
ZDCYC_DZ = 1.0e4       # :815,:821
ZDUTEN0 = 2.0          # :820
ZTAURES_MIN = 1.25     # :827
ZSATFR_MAX = float(np.float32(0.94))   # :852  default-kind literal 0.94
ZOMEGA_RES = -100.0    # :852
ZCAPE_MAX = 5000.0     # :858
ZHEAT_MIN = 1.0e-4     # :859
ZXTAU_MIN = 3.6e3 / 5.0        # :861
ZXTAU_MAX = 3.0 * 3.6e3        # :861
ZMFUB1_MIN = 0.001     # :863
ZEPS_MF = 1.0e-10      # :882,:944
ZDH_MIN = 0.1          # :897  RG*MAX(ZDH, 0.1*RCPD)
ZSHAL_MF = 0.9         # :900
ZSHAL_DH = 0.5         # :900
ZSHAL_DH2 = 0.75       # :901
ZMID_MF = 5.0          # :911
ZMID_GEO = 1.5e3       # :914
ZMID_CFL = 1.05        # :915
ZMFS_MIN = 1.0e-10     # :979
ZDDTOP_Q = 0.3         # :1140-1144
ZMFA_MIN = 1.0e-15     # :1195,:1201
ZRSUD_AR, ZRSUD_AS = 20.89, 29.51   # :1583-1584
ZRSUD_BR, ZRSUD_BS = 1.15, 1.10     # :1585-1586
ZRSUD_UNIT = 1.0e-3    # :1601
ZRSUD_HR = 3600.0      # :1601
ZRSUD_SNOW = 10.0      # :1602
ZAR = 1.0 / 20.89      # :1568
ZAS = 1.0 / 29.51      # :1569
ZBR = 1.0 / 1.15       # :1570
ZBS = 1.0 / 1.10       # :1571
ZDISS_EXP = float(np.float32(0.3333))  # :1596  default-kind literal 0.3333
ZBIDIAG_EPS = 1.0e-35  # cubidiag.F90:72,:76
# --- sucumf.F90:284-290 NJKT* pressure thresholds [Pa] ---
NJKT1_P, NJKT2_P, NJKT3_P = 350.0e2, 60.0e2, 950.0e2
NJKT5_P, NJKT6_P, NJKT7_P = 500.0e2, 700.0e2, 100.0e2


class IfsTunables(NamedTuple):
    """Run-set CUMASTRN constants a host may override; defaults are sucumf's."""
    entrorg: float = ENTRORG
    detrpen: float = DETRPEN
    entshalp: float = ENTSHALP
    rprcon: float = RPRCON
    zdnoprc: float = ZDNOPRC
    rcapqadv: float = RCAPQADV
    # Shallow closure with no sub-cloud supply (ZDHPBL <= 0), main #2253 (user
    # decision 2026-10-28).  "first_guess" is the source (:912 ZMFUB1=ZMFUB);
    # "zero" closes on smooth_max(ZDHPBL, 0) of width shallow_supply_width_w_m2
    # [W/m2] (0 = hard max).  Defaults are the source's; the host passes its
    # run's BechtoldConfig.ifs_shallow_nosupply / ifs_shallow_supply_width_w_m2.
    shallow_nosupply: str = "first_guess"
    shallow_supply_width_w_m2: float = 1.0


SHALLOW_NOSUPPLY_MODES = ("zero", "first_guess")


class CumastrnOut(NamedTuple):
    """CUMASTRN outputs for one column (1-based padded; see field notes)."""
    ptent: jax.Array
    ptenq: jax.Array
    ptenu: jax.Array
    ptenv: jax.Array
    ldcum: jax.Array
    ktype: jax.Array
    kcbot: jax.Array
    kctop: jax.Array
    kbotsc: jax.Array
    ldsc: jax.Array
    ptu: jax.Array
    pqu: jax.Array
    plu: jax.Array
    plude: jax.Array
    pludeli: jax.Array      # (4, K+1): liquid, ice, vapour, T detrainment
    psnde: jax.Array        # (2, K+1): detrained rain, snow
    penth: jax.Array
    pmflxr: jax.Array       # (K+2,) half levels
    pmflxs: jax.Array
    prain: jax.Array
    plrain: jax.Array
    prsud: jax.Array        # (2, K+1)
    pmfu: jax.Array
    pmfd: jax.Array
    pmfude_rate: jax.Array
    pmfdde_rate: jax.Array
    pcape: jax.Array
    pwmean: jax.Array
    pvdiscu: jax.Array
    pdiss: jax.Array
    pqsen: jax.Array        # INOUT: melting-adjusted saturation humidity (cuflxn)
    pdmfup: jax.Array       # local ZDMFUP handed to CUDTDQN (precipitation sink of q)


def njkt_levels(K, pap):
    """sucumf.F90:279-293 with STPRE := the given full-level pressure (1-based):
    NJKT1 (350 hPa), NJKT2 (60 hPa), NJKT3 (950 hPa), NJKT5 (500 hPa),
    NJKT6 (700 hPa), NJKT7 (100 hPa)."""
    lev = jnp.arange(K + 1)
    rng = (lev >= 2) & (lev <= K)

    def top(thr, init):
        ok = rng & (pap > thr)
        return jnp.where(jnp.any(ok), jnp.min(jnp.where(ok, lev, K + 1)), init)

    njkt1 = top(NJKT1_P, 2)
    njkt2 = top(NJKT2_P, 2)
    njkt3 = jnp.minimum(K - 2, top(NJKT3_P, K - 2))
    # NJKT5/6/7 have no initial value in sucumf (unset if no level is below
    # the threshold); 0 here
    njkt5 = top(NJKT5_P, 0)
    njkt6 = top(NJKT6_P, 0)
    njkt7 = top(NJKT7_P, 0)
    return njkt1, njkt2, njkt3, njkt5, njkt6, njkt7


# ---------------------------------------------------------------------------
# CUDLFSN (cudlfsn.F90:173-276), LPHYLIN = .FALSE.
# ---------------------------------------------------------------------------
def cudlfsn(th, K, kcbot, kctop, ldcum, ptenh, pqenh, pten, pqsen, pgeo, pgeoh,
            paph, ptu, pqu, pmfub, prfl, ptd, pqd, pmfd, pmfds, pmfdq, pdmfdp):
    lev = jnp.arange(K + 1)
    rcpd, retv = th.RCPD, th.RETV
    zhsk = rcpd * pten + pgeo + foelhmcu(th, pten) * pqsen
    rng = (lev >= 3) & (lev <= K - 2)
    zhs = jnp.where(rng, zhsk, ZHSMIN0)
    # first strict minimum below ZHSMIN = 1e8, else KLEV+1
    below = rng & (zhsk < ZHSMIN0)
    ikhsmin = jnp.where(jnp.any(below), jnp.argmin(zhs), K + 1)
    st = dict(lddraf=jnp.bool_(False), kdtop=jnp.int32(K + 1), prfl=prfl, ptd=ptd,
              pqd=pqd, pmfd=pmfd, pmfds=pmfds, pmfdq=pmfdq, pdmfdp=pdmfdp)

    def body(i, s):
        jk = 3 + i
        s = dict(s)
        llo2 = (ldcum & (s["prfl"] > 0.0) & ~s["lddraf"] & (jk < kcbot)
                & (jk > kctop) & (jk >= ikhsmin))
        twb, qwb = cuadjtq(th, 2, kget(paph, jk), kget(ptenh, jk), kget(pqenh, jk), llo2)
        zttest = 0.5 * (kget(ptu, jk) + twb)
        zqtest = 0.5 * (kget(pqu, jk) + qwb)
        zbuo = zttest * (1.0 + retv * zqtest) - kget(ptenh, jk) * (1.0 + retv * kget(pqenh, jk))
        zcond = kget(pqenh, jk) - qwb
        zmftop = -RMFDEPS * pmfub
        ok = llo2 & (zbuo < 0.0) & (s["prfl"] > ZLFS_RAIN * zmftop * zcond)
        s["kdtop"] = jnp.where(ok, jk, s["kdtop"])
        s["lddraf"] = s["lddraf"] | ok
        s["ptd"] = kset(s["ptd"], jk, zttest, ok)
        s["pqd"] = kset(s["pqd"], jk, zqtest, ok)
        s["pmfd"] = kset(s["pmfd"], jk, zmftop, ok)
        s["pmfds"] = kset(s["pmfds"], jk, zmftop * (rcpd * zttest + kget(pgeoh, jk)), ok)
        s["pmfdq"] = kset(s["pmfdq"], jk, zmftop * zqtest, ok)
        dp = -0.5 * zmftop * zcond
        s["pdmfdp"] = kset(s["pdmfdp"], jk - 1, dp, ok)
        s["prfl"] = jnp.where(ok, s["prfl"] + dp, s["prfl"])
        return s

    return jax.lax.fori_loop(0, max(K - 5, 0), body, st)


# ---------------------------------------------------------------------------
# CUDDRAFN (cuddrafn.F90:156-296)
# ---------------------------------------------------------------------------
def cuddrafn(th, K, lddraf, ptenh, pqenh, pgeo, pgeoh, paph, prfl, ptd, pqd,
             pmfu, pmfd, pmfds, pmfdq, pdmfdp, njkt3):
    rg, rcpd, retv = th.RG, th.RCPD, th.RETV
    zrg = 1.0 / rg
    f = ptenh.dtype
    zero = jnp.zeros(K + 1, f)
    itopde = njkt3
    st = dict(prfl=prfl, ptd=ptd, pqd=pqd, pmfd=pmfd, pmfds=pmfds, pmfdq=pmfdq,
              pdmfdp=pdmfdp, pmfdde_rate=zero, zoentr=jnp.zeros((), f),
              zbuoy=jnp.zeros((), f))

    def body(i, s):
        jk = 3 + i
        s = dict(s)
        mfd1 = kget(s["pmfd"], jk - 1)
        llo2 = lddraf & (mfd1 < 0.0)
        zentr = ENTRDD * mfd1 * (kget(pgeoh, jk - 1) - kget(pgeoh, jk)) * zrg
        zdmfen = zentr
        zdmfde = zentr
        lowp = jk > itopde
        zdmfen = jnp.where(lowp, 0.0, zdmfen)
        zdmfde = jnp.where(lowp, kget(s["pmfd"], itopde) * (kget(paph, jk) - kget(paph, jk - 1))
                           / (paph[K + 1] - kget(paph, itopde)), zdmfde)
        zdz = -(kget(pgeoh, jk - 1) - kget(pgeoh, jk)) * zrg
        en = zdmfen + s["zoentr"] * zdz * mfd1
        en = jnp.maximum(en, ZDD_MIX_MIN * mfd1)
        en = jnp.maximum(en, -ZDD_DET_MAX * kget(pmfu, jk) - (mfd1 - zdmfde))
        en = jnp.minimum(en, 0.0)
        zdmfen = jnp.where(lowp, zdmfen, en)
        mfd = mfd1 + zdmfen - zdmfde
        zseen = (rcpd * kget(ptenh, jk - 1) + kget(pgeoh, jk - 1)) * zdmfen
        zqeen = kget(pqenh, jk - 1) * zdmfen
        zsdde = (rcpd * kget(s["ptd"], jk - 1) + kget(pgeoh, jk - 1)) * zdmfde
        zqdde = kget(s["pqd"], jk - 1) * zdmfde
        zmfdsk = kget(s["pmfds"], jk - 1) + zseen - zsdde
        zmfdqk = kget(s["pmfdq"], jk - 1) + zqeen - zqdde
        inv = 1.0 / jnp.minimum(-RMFCMIN, mfd)
        qd = zmfdqk * inv
        td = (zmfdsk * inv - kget(pgeoh, jk)) / rcpd
        td = jnp.maximum(ZTD_MIN, jnp.minimum(ZTD_MAX, td))
        zcond = qd
        td, qd = cuadjtq(th, 2, kget(paph, jk), td, qd, llo2)
        zcond = zcond - qd
        zbuo = td * (1.0 + retv * qd) - kget(ptenh, jk) * (1.0 + retv * kget(pqenh, jk))
        rain = (s["prfl"] > 0.0) & (kget(pmfu, jk) > 0.0)
        zbuo = jnp.where(rain, zbuo - td * sdiv(s["prfl"], kget(pmfu, jk), rain), zbuo)
        stop = (zbuo >= 0.0) | (s["prfl"] <= mfd * zcond)
        mfd = jnp.where(stop, 0.0, mfd)
        zbuo = jnp.where(stop, 0.0, zbuo)
        dmfdp = -mfd * zcond
        zbuoyz = zbuo / kget(ptenh, jk)
        zbuoyz = jnp.minimum(zbuoyz, 0.0)
        zdzg = -(kget(pgeo, jk - 1) - kget(pgeo, jk))
        zbuoy = s["zbuoy"] + zbuoyz * zdzg
        zoentr = rg * zbuoyz * 0.5 / (1.0 + zbuoy)
        s["pmfd"] = kset(s["pmfd"], jk, mfd, llo2)
        s["ptd"] = kset(s["ptd"], jk, td, llo2)
        s["pqd"] = kset(s["pqd"], jk, qd, llo2)
        s["pmfds"] = kset(s["pmfds"], jk, (rcpd * td + kget(pgeoh, jk)) * mfd, llo2)
        s["pmfdq"] = kset(s["pmfdq"], jk, qd * mfd, llo2)
        s["pdmfdp"] = kset(s["pdmfdp"], jk - 1, dmfdp, llo2)
        s["prfl"] = jnp.where(llo2, s["prfl"] + dmfdp, s["prfl"])
        s["zbuoy"] = jnp.where(llo2, zbuoy, s["zbuoy"])
        s["zoentr"] = jnp.where(llo2, zoentr, s["zoentr"])
        s["pmfdde_rate"] = kset(s["pmfdde_rate"], jk, -zdmfde, llo2)
        return s

    # PKINED (downdraught kinetic energy) is not used by CUMASTRN: not ported
    return jax.lax.fori_loop(0, K - 2, body, st)


# ---------------------------------------------------------------------------
# CUFLXN (cuflxn.F90:193-510), LPHYLIN = LDTDKMF = .FALSE.
# ---------------------------------------------------------------------------
def cuflxn(th, K, dt, pten, pqen, pqsen, ptenh, pqenh, paph, pap, pgeoh, land,
           ldcum, kcbot, kctop, kdtop, ktype, lddraf, pmfu, pmfd, pmfus, pmfds,
           pmfuq, pmfdq, pmful, plude, pludeli1, pludeli2, plrain, pdmfup, pdmfdp,
           plglac, pmfude_rate, pmfdde_rate):
    rg, rcpd, rtt = th.RG, th.RCPD, th.RTT
    lev = jnp.arange(K + 1)
    f = pten.dtype
    zcons1a = rcpd / (th.RLMLT * rg * RTAUMEL)
    zcons2 = RMFCFL / (rg * dt)
    rcpecons = RCPECONS_G / rg
    lddraf = lddraf & ldcum & ~(kdtop < kctop)
    ktype = jnp.where(ldcum, ktype, 0)
    rh_land = jnp.where(ktype == 1, RHEBC_LAND_DEEP, RHEBC_LAND)
    rh_sea = jnp.where(ktype == 1, RHEBC_DEEP, RHEBC)
    zrhebc = land * rh_land + (1.0 - land) * rh_sea
    # 1.0 final fluxes, :229-283 (JK = 2..KLEV)
    rngk = (lev >= 2) & (lev <= K)
    inc = rngk & ldcum & (lev >= kctop)
    gh = pgeoh[:K + 1]
    llddraf = lddraf & (lev >= kdtop)
    pmfus = jnp.where(inc, pmfus - pmfu * (rcpd * ptenh + gh), pmfus)
    pmfuq = jnp.where(inc, pmfuq - pmfu * pqenh, pmfuq)
    plglac = jnp.where(inc, pmfu * plglac, plglac)
    pmfds = jnp.where(inc & llddraf, pmfds - pmfd * (rcpd * ptenh + gh), pmfds)
    pmfdq = jnp.where(inc & llddraf, pmfdq - pmfd * pqenh, pmfdq)
    nxt = lambda a: jnp.concatenate([a[1:], a[-1:]])   # a(JK+1), clamped at KLEV
    # IDBAS (:264-266) reads PMFD(JK+1) before iteration JK+1 zeroes it.
    isb = inc & llddraf & (pmfd < 0.0) & (nxt(pmfd) == 0.0)
    idbas = jnp.where(jnp.any(isb), jnp.max(jnp.where(isb, lev, 0)), 1)
    ddoff = inc & ~llddraf
    pmfd = jnp.where(ddoff, 0.0, pmfd)
    pmfds = jnp.where(ddoff, 0.0, pmfds)
    pmfdq = jnp.where(ddoff, 0.0, pmfdq)
    prv = lambda m: jnp.concatenate([m[1:], jnp.zeros((1,), bool)])  # level JK-1 from JK
    pdmfdp = jnp.where(prv(ddoff), 0.0, pdmfdp)
    out = rngk & ~inc
    for_zero = (pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq, pmful, plglac)
    pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq, pmful, plglac = (
        jnp.where(out, 0.0, a) for a in for_zero)
    outp = prv(out)
    pdmfup = jnp.where(outp, 0.0, pdmfup)
    pdmfdp = jnp.where(outp, 0.0, pdmfdp)
    plude = jnp.where(outp, 0.0, plude)
    pludeli1 = jnp.where(outp, 0.0, pludeli1)
    pludeli2 = jnp.where(outp, 0.0, pludeli2)
    # 1.5 below cloud base, :295-343
    ikb = kcbot
    ik = ikb + 1
    zzp = (paph[K + 1] - kget(paph, ik)) / (paph[K + 1] - kget(paph, ikb))
    zzp = jnp.where(ktype == 3, zzp * zzp, zzp)
    mfu_b = kget(pmfu, ikb)
    pmfu = kset(pmfu, ik, mfu_b * zzp, ldcum)
    pmfus = kset(pmfus, ik, (kget(pmfus, ikb) - foelhmcu(th, kget(ptenh, ikb)) * kget(pmful, ikb)) * zzp, ldcum)
    pmfuq = kset(pmfuq, ik, (kget(pmfuq, ikb) + kget(pmful, ikb)) * zzp, ldcum)
    pmful = kset(pmful, ik, 0.0, ldcum)
    # :316-329 sequential: the SQRT test reads PMFU(IDBAS), which this loop
    # overwrites once JK passes IDBAS.
    ikb1 = kcbot + 1
    zzp_all = (paph[K + 1] - paph[:K + 1]) / (paph[K + 1] - kget(paph, ikb1))

    def blw(i, c):
        jk = 2 + i
        mu, mus, muq, mul = c
        on = ldcum & (jk > kcbot + 1)
        zzp_ = zzp_all[jk]
        sq = (idbas >= K - 1) & (-kget(pmfd, idbas) > 0.5 * kget(mu, idbas))
        zzp_ = jnp.where(sq, ssqrt(zzp_), zzp_)
        zzp_ = jnp.where(ktype == 3, zzp_ * zzp_, zzp_)
        return (kset(mu, jk, kget(mu, ikb1) * zzp_, on), kset(mus, jk, kget(mus, ikb1) * zzp_, on),
                kset(muq, jk, kget(muq, ikb1) * zzp_, on), kset(mul, jk, 0.0, on))

    pmfu, pmfus, pmfuq, pmful = jax.lax.fori_loop(0, K - 1, blw, (pmfu, pmfus, pmfuq, pmful))
    llb = lddraf & (lev > idbas) & (idbas < K) & rngk
    c1 = llb & (idbas == kcbot + 1)
    zzpd = (paph[K + 1] - paph[:K + 1]) / (paph[K + 1] - kget(paph, idbas))
    zzpd = jnp.where(ktype == 3, zzpd * zzpd, zzpd)
    pmfd = jnp.where(c1, kget(pmfd, idbas) * zzpd, pmfd)
    pmfds = jnp.where(c1, kget(pmfds, idbas) * zzpd, pmfds)
    pmfdq = jnp.where(c1, kget(pmfdq, idbas) * zzpd, pmfdq)
    pmfd_up = jnp.concatenate([pmfd[:1], pmfd[:-1]])          # PMFD(JK-1), after the update
    c2 = llb & (idbas != kcbot + 1) & (lev == idbas + 1)
    pmfdde_rate = jnp.where(c1 | c2, -(pmfd_up - pmfd), pmfdde_rate)
    # rescale DD fluxes if the total mass flux becomes negative, :345-366
    rng2 = (lev >= 2) & (lev <= K)
    sel = rng2 & lddraf & (lev >= kdtop - 1)
    zmfmax = jnp.abs(pmfu) * ZDD_SCALE_MAX
    neg = sel & (pmfd + zmfmax + ZDD_TINY < 0.0)
    zmfs = jnp.min(jnp.where(neg, -zmfmax / jnp.where(neg, pmfd, -1.0), 1.0))
    scl = rng2 & (zmfs < 1.0) & (lev >= kdtop - 1)
    pmfd = jnp.where(scl, pmfd * zmfs, pmfd)
    pmfds = jnp.where(scl, pmfds * zmfs, pmfds)
    pmfdq = jnp.where(scl, pmfdq * zmfs, pmfdq)
    pmfdde_rate = jnp.where(scl, pmfdde_rate * zmfs, pmfdde_rate)
    pdmfdp = jnp.where(scl, pdmfdp * zmfs, pdmfdp)

    # 2. rain/snow fall, melting, :370-440 (JK = 2..KLEV, sequential)
    zero_h = jnp.zeros(K + 2, f)
    st = dict(pmflxr=zero_h, pmflxs=zero_h, pdpmel=jnp.zeros(K + 1, f),
              psnde1=jnp.zeros(K + 1, f), psnde2=jnp.zeros(K + 1, f),
              pqsen=pqsen, plglac=plglac, pdmfdp=pdmfdp, prain=jnp.zeros((), f))

    def fall(i, s):
        jk = 2 + i
        s = dict(s)
        on = ldcum & (jk >= kctop - 1)
        dmfup = kget(pdmfup, jk)
        s["prain"] = jnp.where(on, s["prain"] + dmfup, s["prain"])
        tk, qk, qsk, pk = kget(pten, jk), kget(pqen, jk), kget(s["pqsen"], jk), kget(pap, jk)
        ztwetb = tk - jnp.maximum(0.0, qsk - qk) * (ZTW1 + ZTW2 * (pk - ZTW3) - ZTW4 * (tk - ZTW5))
        zten = ztwetb   # LMFWETB: ZWETB = 1
        rk = kget(s["pmflxr"], jk)
        sk = kget(s["pmflxs"], jk)
        melt = on & (sk > 0.0) & (zten > rtt)
        zcons1 = zcons1a * (1.0 + ZMELT_DT * (zten - rtt))
        zfac = zcons1 * (kget(paph, jk + 1) - kget(paph, jk))
        zsnmlt = jnp.minimum(sk, zfac * (zten - rtt))
        dpmel = jnp.where(melt, zsnmlt, 0.0)
        qsn = foeewmcu(th, zten - zsnmlt / jnp.where(melt, zfac, 1.0)) / pk
        s["pqsen"] = kset(s["pqsen"], jk, qsn, melt)
        zalfaw = foealfcu(th, zten)
        dmfdp = kget(s["pdmfdp"], jk)
        frz = (tk <= rtt) & (zalfaw > 0.0)
        # ZGLAC = 0 (LMFGLAC): the PDMFUP term of :410 vanishes.
        s["plglac"] = kset(s["plglac"], jk, kget(s["plglac"], jk) + zalfaw * dmfdp, on & frz)
        zalfaw = jnp.where(frz, 0.0, zalfaw)
        dsn = jk <= kcbot          # LMFDSNOW
        sn2 = kget(plrain, jk + 1) * kget(pmfude_rate, jk)
        sn1 = sn2 * zalfaw
        sn2 = sn2 * (1.0 - zalfaw)
        sn1 = jnp.minimum(sn1, zalfaw * (dmfup + dmfdp))
        sn2 = jnp.minimum(sn2, (1.0 - zalfaw) * (dmfup + dmfdp) - dpmel)
        sn1 = jnp.where(dsn, jnp.maximum(sn1, 0.0), 0.0)
        sn2 = jnp.where(dsn, jnp.maximum(sn2, 0.0), 0.0)
        r1 = rk + zalfaw * (dmfup + dmfdp) + dpmel - sn1
        s1 = sk + (1.0 - zalfaw) * (dmfup + dmfdp) - dpmel - sn2
        allneg = r1 + s1 < 0.0
        rneg = ~allneg & (r1 < 0.0)
        sneg = ~allneg & ~rneg & (s1 < 0.0)
        dmfdp_n = jnp.where(allneg, -(rk + sk + dmfup), dmfdp)
        r2 = jnp.where(allneg, 0.0, jnp.where(rneg, 0.0, jnp.where(sneg, r1 + s1, r1)))
        s2 = jnp.where(allneg, 0.0, jnp.where(rneg, s1 + r1, jnp.where(sneg, 0.0, s1)))
        dpmel = jnp.where(allneg, 0.0, dpmel)
        s["pmflxr"] = kset(s["pmflxr"], jk + 1, r2, on)
        s["pmflxs"] = kset(s["pmflxs"], jk + 1, s2, on)
        s["pdpmel"] = kset(s["pdpmel"], jk, dpmel, on)
        s["psnde1"] = kset(s["psnde1"], jk, sn1, on)
        s["psnde2"] = kset(s["psnde2"], jk, sn2, on)
        s["pdmfdp"] = kset(s["pdmfdp"], jk, dmfdp_n, on)
        return s

    st = jax.lax.fori_loop(0, K - 1, fall, st)
    pqsen = st["pqsen"]
    # cover fraction, :445-461
    ikb, ikt = kcbot, kctop
    zrhm = 0.5 * (kget(pqen, ikb) / kget(pqsen, ikb) + kget(pqen, ikt) / kget(pqsen, ikt))
    zrcucov = jnp.where(ktype >= 2,
                        RCUCOV * (1.0 + (jnp.maximum(ZRHM_MIN, zrhm) - ZRHM_MIN) / ZRHM_SCALE),
                        RCUCOV * ZRCUCOV_DEEP)

    # evaporation below cloud base, :463-508
    def evap(i, s):
        jk = 2 + i
        s = dict(s)
        on = ldcum & (jk >= kcbot)
        rk = kget(s["pmflxr"], jk)
        sk = kget(s["pmflxs"], jk)
        zrfl = rk + sk
        has = zrfl > ZRFL_MIN
        dp = kget(paph, jk + 1) - kget(paph, jk)
        qsk = kget(pqsen, jk)
        qk = kget(pqen, jk)
        arg = jnp.sqrt(kget(paph, jk) / paph[K + 1]) / RCVRFACTOR * zrfl / zrcucov
        zdrfl1 = rcpecons * jnp.maximum(0.0, qsk - qk) * zrcucov * spow(arg, ZEVAP_EXP) * dp
        zrnew = zrfl - zdrfl1
        zrmin = zrfl - zrcucov * jnp.maximum(0.0, zrhebc * qsk - qk) * zcons2 * dp
        zrnew = jnp.maximum(zrnew, zrmin)
        zrfln = jnp.maximum(zrnew, 0.0)
        zdrfl = jnp.minimum(0.0, zrfln - zrfl)
        zalfaw = jnp.where(kget(pten, jk) < rtt, 0.0, foealfcu(th, kget(pten, jk)))
        dmfdp = kget(s["pdmfdp"], jk)
        zpdr = zalfaw * dmfdp
        zpds = (1.0 - zalfaw) * dmfdp
        zdenom = 1.0 / jnp.maximum(ZRFL_MIN, rk + sk)
        dpmel = kget(s["pdpmel"], jk)
        r1 = rk + zpdr + dpmel + zdrfl * rk * zdenom
        s1 = sk + zpds - dpmel + zdrfl * sk * zdenom
        dmfup = kget(s["pdmfup"], jk) + zdrfl
        allneg = r1 + s1 < 0.0
        rneg = ~allneg & (r1 < 0.0)
        sneg = ~allneg & ~rneg & (s1 < 0.0)
        dmfup = jnp.where(allneg, dmfup - (r1 + s1), dmfup)
        r2 = jnp.where(allneg, 0.0, jnp.where(rneg, 0.0, jnp.where(sneg, r1 + s1, r1)))
        s2 = jnp.where(allneg, 0.0, jnp.where(rneg, s1 + r1, jnp.where(sneg, 0.0, s1)))
        dpmel_n = jnp.where(allneg, 0.0, dpmel)
        a = on & has
        b = on & ~has
        s["pmflxr"] = kset(s["pmflxr"], jk + 1, jnp.where(a, r2, 0.0), on)
        s["pmflxs"] = kset(s["pmflxs"], jk + 1, jnp.where(a, s2, 0.0), on)
        s["pdmfup"] = kset(s["pdmfup"], jk, dmfup, a)
        s["pdpmel"] = kset(s["pdpmel"], jk, jnp.where(a, dpmel_n, 0.0), on)
        s["pdmfdp"] = kset(s["pdmfdp"], jk, 0.0, b)
        return s

    st["pdmfup"] = pdmfup
    st = jax.lax.fori_loop(0, K - 1, evap, st)
    return dict(lddraf=lddraf, ktype=ktype, pmfu=pmfu, pmfd=pmfd, pmfus=pmfus,
                pmfds=pmfds, pmfuq=pmfuq, pmfdq=pmfdq, pmful=pmful, plude=plude,
                pludeli1=pludeli1, pludeli2=pludeli2, psnde1=st["psnde1"],
                psnde2=st["psnde2"], pdmfup=st["pdmfup"], pdmfdp=st["pdmfdp"],
                pdpmel=st["pdpmel"], plglac=st["plglac"], pmflxr=st["pmflxr"],
                pmflxs=st["pmflxs"], prain=st["prain"], pmfude_rate=pmfude_rate,
                pmfdde_rate=pmfdde_rate, pqsen=pqsen)


# ---------------------------------------------------------------------------
# CUBIDIAG (cubidiag.F90:64-82)
# ---------------------------------------------------------------------------
def cubidiag(K, kctop, mask, pa, pb, pr):
    def body(i, pu):
        jk = 2 + i
        m = kget(mask, jk)
        zbet = 1.0 / (kget(pb, jk) + ZBIDIAG_EPS)
        first = jk == kctop - 1
        val = jnp.where(first, kget(pr, jk) * zbet,
                        (kget(pr, jk) - kget(pa, jk) * kget(pu, jk - 1)) * zbet)
        return kset(pu, jk, val, m & (jk >= kctop - 1))
    return jax.lax.fori_loop(0, K - 1, body, jnp.zeros_like(pr))




def _up(a):
    """a(JK-1) at index JK (index 0, 1 duplicate a(0)/a(1); never selected)."""
    return jnp.concatenate([a[:1], a[:-1]])


def _nx(a):
    """a(JK+1) at index JK (the last entry repeats a(KLEV); never selected)."""
    return jnp.concatenate([a[1:], a[-1:]])


def _zdp(th, K, paph):
    """ZDP(JK) = RG/(PAPH(JK+1)-PAPH(JK)) at index JK (index 0 = index 1)."""
    zdp = th.RG / (paph[2:K + 2] - paph[1:K + 1])   # physical layers only (cudtdqn.F90:234)
    return jnp.concatenate([zdp[:1], zdp])


# ---------------------------------------------------------------------------
# CUDTDQN (cudtdqn.F90:197-460): RMFSOLTQ = 1, RMFSOLRHS = RMFADVW = 0,
# LMFENTHCONS = LDTDKMF = LPHYLIN = .FALSE., LEPCLD = .TRUE.
# ---------------------------------------------------------------------------
def cudtdqn(th, K, dt, kctop, ldcum, lddraf, kdtop, pten, pqen, paph, pgeo,
            pgeoh, ptenh, pqenh, pqsen, plglac, plude, pludeli1, pludeli2, psnde1,
            psnde2, pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq, pmful, pdmfup, pdpmel,
            ptent, ptenq):
    rcpd = th.RCPD
    lev = jnp.arange(K + 1)
    zorcpd = 1.0 / rcpd
    zimp = 0.0                                             # 1 - RMFSOLTQ
    zdp = _zdp(th, K, paph)
    on = (lev >= 2) & (lev <= K) & ldcum & (lev >= kctop - 1)
    # 2.0 fluxes recomputed for the implicit solution, :257-283
    zgq = (pqenh - _up(pqen)) / pqsen
    zgh = rcpd * pten + pgeo
    zgs = (rcpd * (ptenh - _up(pten)) + pgeoh[:K + 1] - _up(pgeo)) / zgh
    zs = rcpd * (zimp * _up(pten) + zgs * pten) + _up(pgeo) + zgs * pgeo
    zq = zimp * _up(pqen) + zgq * pqsen
    zmfus = jnp.where(on, pmfus - pmfu * zs, pmfus)
    zmfuq = jnp.where(on, pmfuq - pmfu * zq, pmfuq)
    dd = on & lddraf & (lev >= kdtop)
    zmfds = jnp.where(dd, pmfds - pmfd * zs, pmfds)
    zmfdq = jnp.where(dd, pmfdq - pmfd * zq, pmfdq)
    # 3.0 tendencies (ZINT = 0: LMFENTHCONS off), :290-380
    zalv = foelhmcu(th, pten)
    zdtdt = zdp * zorcpd * (_nx(zmfus) - zmfus + _nx(zmfds) - zmfds
                            + (th.RLMLT * (plglac - pdpmel)
                               - zalv * (_nx(pmful) - pmful) + zalv * pdmfup)
                            + th.RLVTT * (pludeli1 + psnde1)
                            + th.RLSTT * (pludeli2 + psnde2))
    zdqdt = zdp * (_nx(zmfuq) - zmfuq + _nx(zmfdq) - zmfdq + _nx(pmful) - pmful
                   - plude - pdmfup - psnde1 - psnde2)
    zdtdt = zdtdt.at[K].set(-zdp[K] * zorcpd * (
        zmfus[K] + zmfds[K] + (th.RLMLT * (pdpmel[K] - plglac[K]) - zalv[K] * pmful[K]
                               - zalv[K] * pdmfup[K])))
    zdqdt = zdqdt.at[K].set(-zdp[K] * (zmfuq[K] + zmfdq[K] + (pmful[K] + pdmfup[K])))
    # 3.2 implicit solution, :400-455.  RMFSOLRHS = 0 and ZADVW = 0 make the
    # RHS (ZDTDT + PTENT*0)*PTSPHY + PTEN - 0 = ZDTDT*PTSPHY + PTEN exactly.
    zzp = zdp * dt
    za = jnp.where(on, -zzp * (pmfu + pmfd), 0.0)
    zb = jnp.where(on & (lev < K), 1.0 + zzp * (_nx(pmfu) + _nx(pmfd)), 1.0)
    zr1 = cubidiag(K, kctop, on, za, zb, zdtdt * dt + pten)
    zr2 = cubidiag(K, kctop, on, za, zb, zdqdt * dt + pqen)
    ztsphy = 1.0 / dt
    ptent = jnp.where(on, ptent + (zr1 - pten) * ztsphy, ptent)
    ptenq = jnp.where(on, ptenq + (zr2 - pqen) * ztsphy, ptenq)
    penth = jnp.where(on, (zr1 - pten) * ztsphy, 0.0)
    return ptent, ptenq, penth


# ---------------------------------------------------------------------------
# CUDUDV (cududv.F90:130-406): RMFSOLUV = 1, RMFSOLRHS = RMFADVW = 0, SPP off
# ---------------------------------------------------------------------------
def cududv(th, K, dt, kctop, ldcum, paph, puen, pven, pmfu, pmfd, puu, pud, pvu,
           pvd, ptenu, ptenv):
    lev = jnp.arange(K + 1)
    zimp = 0.0
    zdp = _zdp(th, K, paph)
    rng = (lev >= 2) & (lev <= K)
    zmfuu = jnp.where(rng & ldcum, pmfu * (puu - zimp * _up(puen)), 0.0)
    zmfuv = jnp.where(rng & ldcum, pmfu * (pvu - zimp * _up(pven)), 0.0)
    zmfdu = jnp.where(rng & ldcum, pmfd * (pud - zimp * _up(puen)), 0.0)
    zmfdv = jnp.where(rng & ldcum, pmfd * (pvd - zimp * _up(pven)), 0.0)
    zdudt = zdp * (_nx(zmfuu) - zmfuu + _nx(zmfdu) - zmfdu)
    zdvdt = zdp * (_nx(zmfuv) - zmfuv + _nx(zmfdv) - zmfdv)
    zdudt = zdudt.at[K].set(-zdp[K] * (zmfuu[K] + zmfdu[K]))
    zdvdt = zdvdt.at[K].set(-zdp[K] * (zmfuv[K] + zmfdv[K]))
    on = rng & ldcum & (lev >= kctop - 1)
    zzp = zdp * dt
    za = jnp.where(on, -zzp * (pmfu + pmfd), 0.0)
    zb = jnp.where(on & (lev < K), 1.0 + zzp * (_nx(pmfu) + _nx(pmfd)), 1.0)
    zr1 = cubidiag(K, kctop, on, za, zb, zdudt * dt + puen)
    zr2 = cubidiag(K, kctop, on, za, zb, zdvdt * dt + pven)
    ztsphy = 1.0 / dt
    ptenu = jnp.where(on, ptenu + (zr1 - puen) * ztsphy, ptenu)
    ptenv = jnp.where(on, ptenv + (zr2 - pven) * ztsphy, ptenv)
    return ptenu, ptenv




# ---------------------------------------------------------------------------
# CUMASTRN (cumastrn.F90:429-1626), one column
# ---------------------------------------------------------------------------
def cumastrn_column(th, K, dt, pgaw, pten, pqen, puen, pven, plitot, pvervel, pqsen,
                    pqhfl_sfc, pahfs_sfc, pap, paph, pgeo, pgeoh, land, ptent,
                    ptenq, ptenu, ptenv, ptenta, ptenqa, ldshcv, njkt,
                    tun=IfsTunables()):
    """CUMASTRN for one column.  Arrays are 1-based padded (full levels K+1,
    half levels K+2; index 0 a finite copy of index 1).  ``njkt`` is the
    tuple of :func:`njkt_levels`; ``pgaw`` the Gaussian weight PGAW (ZDX =
    2*RA*SQRT(RPI*PGAW), :728).  Returns :class:`CumastrnOut`."""
    rg, rcpd, retv, rlvtt = th.RG, th.RCPD, th.RETV, th.RLVTT
    njkt1, njkt2, njkt3, njkt5, njkt6, njkt7 = njkt
    lev = jnp.arange(K + 1)
    zcons2 = RMFCFL / (rg * dt)
    zcons = 1.0 / (rg * dt)
    zorcpd = 1.0 / rcpd
    zrg = 1.0 / rg
    zero = jnp.zeros(K + 1, pten.dtype)
    zdz = paph[2:K + 2] - paph[1:K + 1]                 # PAPH(JK+1)-PAPH(JK), JK = 1..K
    zdz = jnp.concatenate([zdz[:1], zdz])               # index 0: finite copy

    # 2. CUININ, :436-445
    ztenh, zqenh, _zqsenh, zuu, zvu = cuinin(
        th, K, pten, pqen, pqsen, puen, pven, pgeo, paph, pgeoh, njkt2)
    zud, zvd, ztd, zqd = zuu, zvu, ztenh, zqenh
    # 3. CUBASEN, :450-461 (ZKMFL = 0)
    b = cubasen(th, K, ztenh, zqenh, pgeoh, paph, pqhfl_sfc, pahfs_sfc, ptenqa,
                pten, pqen, pqsen, pgeo, land, njkt1, njkt2, njkt6, entrorg=tun.entrorg)
    ldcum, kcbot, ictop0, idpl = b["ldcum"], b["kcbot"], b["kctop"], b["kdpl"]
    # sub-cloud budgets, :463-479
    zkhfl = -pahfs_sfc - rlvtt * pqhfl_sfc
    sub = ldcum & (lev >= njkt2) & (lev >= kcbot) & (lev >= 1)
    zdhpbl = jnp.sum(jnp.where(sub, (rlvtt * ptenq + rcpd * ptent) * zdz, 0.0))
    zcappbl = jnp.sum(jnp.where(sub, (ptent + retv * pten * ptenq) * zdz, 0.0))
    # initial cloud type, :509-523
    zpbmpt = kget(paph, kcbot) - kget(paph, ictop0)
    ktype = jnp.where(ldcum, jnp.where(zpbmpt >= RDEPTHS, 1, 2), 0)
    # first-guess cloud-base mass flux, :539-561
    ikb = kcbot
    zmfmax = (kget(paph, ikb) - kget(paph, ikb - 1)) * zcons2
    zqumqe = kget(b["pqu"], ikb) + kget(b["plu"], ikb) - kget(zqenh, ikb)
    zdqmin = jnp.maximum(ZDQMIN_FRAC * kget(zqenh, ikb), ZDQMIN_FLOOR)
    zdh = rcpd * (kget(b["ptu"], ikb) - kget(ztenh, ikb)) + rlvtt * zqumqe
    zdh = rg * jnp.maximum(zdh, ZDH_DQ * zdqmin)
    sh_ok = zdhpbl > 0.0
    zmfub_sh = jnp.where(sh_ok, jnp.minimum(sdiv(zdhpbl, zdh, sh_ok), zmfmax),
                         zmfmax * ZMFUB_DEEP)
    zmfub = jnp.where(ldcum, jnp.where(ktype == 1, zmfmax * ZMFUB_DEEP, zmfub_sh), 0.0)
    ldcum = ldcum & ~((ktype == 2) & ~sh_ok)

    # 4. CUASCN, :611-625
    a = cuascn(th, K, dt, ztenh, zqenh, pten, pqen, pqsen, plitot, pgeo, pgeoh,
               pap, paph, pvervel, b["pwubase"], ldcum, ktype, b["klab"], b["ptu"],
               b["pqu"], b["plu"], zmfub, kcbot, idpl, njkt7, rprcon=tun.rprcon,
               zdnoprc=tun.zdnoprc, entrorg=tun.entrorg, detrpen=tun.detrpen,
               entshalp=tun.entshalp)
    ztenh, zqenh = a["ptenh"], a["pqenh"]
    ldcum, ktype, kcbot, kctop = a["ldcum"], a["ktype"], a["kcbot"], a["kctop"]
    ptu, pqu, plu, plrain = a["ptu"], a["pqu"], a["plu"], a["plrain"]
    pmfu, zmfub = a["pmfu"], a["pmfub"]
    zlglac, zmfus, zmfuq, zmful = a["plglac"], a["pmfus"], a["pmfuq"], a["pmful"]
    plude, pludeli1, pludeli2 = a["plude"], a["pludeli1"], a["pludeli2"]
    zdmfup, pmfude_rate, pwmean = a["pdmfup"], a["pmfude_rate"], a["pwmean"]
    # reclassify, :634-643
    zpbmpt = kget(paph, kcbot) - kget(paph, kctop)
    ktype = jnp.where(ldcum & (ktype == 1) & (zpbmpt < RDEPTHS), 2, ktype)
    ktype = jnp.where(ldcum & (ktype == 2) & (zpbmpt >= RDEPTHS), 1, ktype)
    zrfl = jnp.sum(zdmfup[1:])
    ztenu, ztenv = ptenu, ptenv

    # 5. downdraught, LMFDD, :674-709
    d = cudlfsn(th, K, kcbot, kctop, ldcum, ztenh, zqenh, pten, pqsen, pgeo, pgeoh,
                paph, ptu, pqu, zmfub, zrfl, ztd, zqd, zero, zero, zero, zero)
    lddraf, idtop = d["lddraf"], d["kdtop"]
    d = cuddrafn(th, K, lddraf, ztenh, zqenh, pgeo, pgeoh, paph, d["prfl"], d["ptd"],
                 d["pqd"], pmfu, d["pmfd"], d["pmfds"], d["pmfdq"], d["pdmfdp"], njkt3)
    ztd, zqd, pmfd, zmfds, zmfdq = d["ptd"], d["pqd"], d["pmfd"], d["pmfds"], d["pmfdq"]
    zdmfdp, pmfdde_rate = d["pdmfdp"], d["pmfdde_rate"]

    # 6. deep closure, :721-867
    zdx = jnp.maximum(2.0 * th.RA * jnp.sqrt(th.RPI * pgaw), ZDX_MIN)
    deep = ldcum & (ktype == 1)
    rng2 = (lev >= njkt2) & (lev <= K)
    llo3 = deep & rng2 & (lev <= kcbot) & (lev > kctop)
    tup, qup, gup = _up(pten), _up(pqen), _up(pgeo)
    zheat = jnp.sum(jnp.where(llo3, jnp.maximum(
        0.0, ((tup - pten + (gup - pgeo) * zorcpd) / ztenh + retv * (qup - pqen))
        * (rg * (pmfu + pmfd))), 0.0))
    zdzp = pap - _up(pap)
    zcape = jnp.sum(jnp.where(llo3, ((ptu - ztenh) / ztenh + retv * (pqu - zqenh) - plu)
                              * zdzp, 0.0))
    ztenh2 = ztenh - 0.5 * (ptenta + _up(ptenta)) * dt
    zqenh2 = zqenh - 0.5 * (ptenqa + _up(ptenqa)) * dt
    zcape2 = jnp.sum(jnp.where(llo3, ((ptu - ztenh2) / ztenh2 + retv * (pqu - zqenh2) - plu)
                               * zdzp, 0.0))
    zdqcv = jnp.sum(jnp.where(deep & rng2, ptenqa * zdz * (pqen / pqsen), 0.0))
    zsatfr = jnp.sum(jnp.where(deep & rng2 & (lev >= kctop), (pqen / pqsen) * zdz, 0.0))
    ztaures = ifs_ztaures(zdx)
    ikd, ikb, ik = idpl, kcbot, kctop
    ztau = ((kget(pgeoh, ik) - kget(pgeoh, ikb))
            / ((ZTAU_W0 + jnp.minimum(ZTAU_WMAX, pwmean)) * rg) * ztaures * RTAUA)
    zxtau = ztau
    llo1 = paph[K + 1] - kget(paph, ikd) < ZPBL_DP
    zduten = ZDUTEN0 + ssqrt(0.5 * (kget(puen, ikb) ** 2 + kget(pven, ikb) ** 2
                                  + kget(puen, njkt3) ** 2 + kget(pven, njkt3) ** 2))
    ztaupbl = jnp.minimum(ZDCYC_DZ, kget(pgeoh, ikb) - pgeoh[K + 1]) / (rg * zduten)
    # LDLAND as a land fraction: exact for land in {0, 1}
    zcapdcycl = jnp.where(llo1, land * (zcappbl * ztau / ztaures)
                          + (1.0 - land) * (zcappbl * ztaupbl), 0.0)
    zdqcv = zdqcv * rlvtt / kget(pgeoh, ik) * zxtau / jnp.maximum(ZTAURES_MIN, ztaures) * tun.rcapqadv
    zsatfr = zsatfr / (paph[K + 1] - kget(paph, ik))
    zcape2 = tun.rcapqadv * zcape2 + (1.0 - tun.rcapqadv) * zcape
    zcapdcycl = jnp.maximum(zcapdcycl, -2.0 * zcape2)
    br = (zsatfr <= ZSATFR_MAX) | (kget(pvervel, njkt5) < ZOMEGA_RES)
    zcape_c = jnp.where(br, jnp.maximum(RMINCAPE * zcape, zcape2 - zcapdcycl + zdqcv),
                        jnp.maximum(RMINCAPE * zcape, zcape - zcapdcycl))
    zcape_c = jnp.minimum(zcape_c, ZCAPE_MAX)
    zheat_c = jnp.maximum(ZHEAT_MIN, zheat)
    zxtau_c = jnp.maximum(ZXTAU_MIN, jnp.minimum(ZXTAU_MAX, zxtau))
    zmfub1 = jnp.maximum((zcape_c * zmfub) / (zheat_c * zxtau_c), ZMFUB1_MIN)   # ZFACCA = 1
    zmfmax = (kget(paph, ikb) - kget(paph, ikb - 1)) * zcons2
    zmfub1 = jnp.where(deep, jnp.minimum(zmfub1, zmfmax), zmfub)
    # shallow and mid-level, :881-936
    pmfd_b = kget(pmfd, ikb)
    zeps = jnp.where(pmfd_b < 0.0, -pmfd_b / jnp.maximum(zmfub, ZEPS_MF), 0.0)
    zqumqe = (kget(pqu, ikb) + kget(plu, ikb) - zeps * kget(zqd, ikb)
              - (1.0 - zeps) * kget(zqenh, ikb))
    zdh = (rcpd * (kget(ptu, ikb) - zeps * kget(ztd, ikb) - (1.0 - zeps) * kget(ztenh, ikb))
           + rlvtt * zqumqe)
    zdh2 = (rcpd * (kget(ptu, ikb - 1) - zeps * kget(ztd, ikb - 1)
                    - (1.0 - zeps) * kget(ztenh, ikb - 1)) + rlvtt * zqumqe)
    zdh = 0.5 * (zdh + zdh2)
    zdh = rg * jnp.maximum(zdh, ZDH_MIN * rcpd)
    # KNOWN DIFFERENCE (RULE 5 / departure from OpenIFS, main #2253, user
    # decision 2026-10-28) when tun.shallow_nosupply == "zero": a column first
    # triggered deep and reclassified shallow with no sub-cloud supply keeps
    # the first guess 0.1*ZMFMAX in OpenIFS (:912), which grows as 1/dt; "zero"
    # uses the supply smooth_max(ZDHPBL, 0) (width shallow_supply_width_w_m2,
    # ZDHPBL being W/m2 times g) in place of ZDHPBL, so the no-supply flux is
    # ln2*g*width/ZDH and a supply of one width gets ln(1+e) = 1.31x the
    # source.  Width 0 is the hard max; "first_guess" is the source.  Per-case
    # gaps: tests/atmosphere/test_bechtold_ifs_oracle.py.
    if tun.shallow_nosupply == "first_guess":
        supply = zdhpbl
    else:
        w = tun.shallow_supply_width_w_m2
        supply = smooth_max(zdhpbl, 0.0,
                            sharpness=float("inf") if w == 0.0 else 1.0 / (rg * w))
    m2 = supply / zdh
    weak = (m2 > ZSHAL_MF * zmfmax / RMFCFL) & (zdh < ZSHAL_DH * rg * rcpd)
    m2 = jnp.where(weak, supply / (ZSHAL_DH2 * rcpd * rg), m2)
    if tun.shallow_nosupply == "first_guess":
        m2 = jnp.where(zdhpbl > 0.0, m2, zmfub)                    # :905 / :912
    m2 = jnp.minimum(m2, zmfmax)
    m3 = jnp.maximum(zmfub, ZMID_MF * zkhfl / kget(pgeoh, ikb)) * (1.0 + zeps)
    m3 = jnp.where(kget(pgeoh, ikb) < ZMID_GEO, jnp.minimum(m3, zmfmax / (ZMID_CFL * RMFCFL)),
                   jnp.minimum(m3, zmfmax))
    zmfub1 = jnp.where(ldcum & (ktype == 2), m2, zmfub1)
    zmfub1 = jnp.where(ldcum & (ktype == 3), m3, zmfub1)
    # rescale DD fluxes, :940-953
    rdd = lddraf & ((ktype == 1) | (ktype == 2))
    zfac = jnp.where(rdd, zmfub1 / jnp.maximum(zmfub, ZEPS_MF), 1.0)
    pmfd, zmfds, zmfdq, zdmfdp, pmfdde_rate = (
        jnp.where(rdd, x * zfac, x) for x in (pmfd, zmfds, zmfdq, zdmfdp, pmfdde_rate))
    # 6.2 final closure = scaling, :961-1021 (RMFADVW = 0)
    zmfs0 = zmfub1 / jnp.maximum(RMFCMIN, zmfub)

    def lim(i, c):
        jk = 2 + i
        mu, zm = c
        on = ldcum & (jk >= kctop - 1)
        zdz_ = (paph[K + 1] - kget(paph, jk)) / (paph[K + 1] - kget(paph, kcbot))
        mu = kset(mu, jk, kget(mu, kcbot) * zdz_, on & (jk > kcbot))
        zmfmx = jnp.minimum((kget(paph, jk) - kget(paph, jk - 1)) * zcons2, RMFLIA)
        mk = kget(mu, jk)
        hit = on & (mk * zm > zmfmx)
        zm = jnp.where(hit, jnp.maximum(jnp.minimum(zm, sdiv(zmfmx, mk, hit)), ZMFS_MIN), zm)
        return mu, zm

    pmfu, zmfs = jax.lax.fori_loop(0, K - 1, lim, (pmfu, zmfs0))
    scl = (lev >= 2) & ldcum & (lev <= kcbot) & (lev >= kctop - 1)
    (pmfu, zmfus, zmfuq, zmful, zdmfup, plude, pludeli1, pludeli2, pmfude_rate) = (
        jnp.where(scl, x * zmfs, x) for x in
        (pmfu, zmfus, zmfuq, zmful, zdmfup, plude, pludeli1, pludeli2, pmfude_rate))
    # 6.5 pathological shallow / switched-off shallow, :1032-1047
    patho = (ktype == 2) & (kcbot == kctop) & (kcbot >= K - 1)
    ldcum = ldcum & ~patho
    ktype = jnp.where(patho, 0, ktype)
    ldcum = ldcum & ~(~ldshcv & (ktype == 2))

    # 7.0 final fluxes, :1082-1115
    idtop = jnp.where(lddraf & (idtop <= kctop), kctop + 1, idtop)
    above = lddraf & (lev >= 2) & (lev < idtop)
    pmfd, zmfds, zmfdq, zdmfdp = (jnp.where(above, 0.0, x) for x in (pmfd, zmfds, zmfdq, zdmfdp))
    pmfdde_rate = jnp.where(above | (lddraf & (lev >= 2) & (lev == idtop)), 0.0, pmfdde_rate)
    fx = cuflxn(th, K, dt, pten, pqen, pqsen, ztenh, zqenh, paph, pap, pgeoh, land,
                ldcum, kcbot, kctop, idtop, ktype, lddraf, pmfu, pmfd, zmfus, zmfds,
                zmfuq, zmfdq, zmful, plude, pludeli1, pludeli2, plrain, zdmfup, zdmfdp,
                zlglac, pmfude_rate, pmfdde_rate)
    lddraf, ktype = fx["lddraf"], fx["ktype"]
    pmfu, pmfd, zmfus, zmfds = fx["pmfu"], fx["pmfd"], fx["pmfus"], fx["pmfds"]
    zmfuq, zmfdq, zmful = fx["pmfuq"], fx["pmfdq"], fx["pmful"]
    plude, pludeli1, pludeli2 = fx["plude"], fx["pludeli1"], fx["pludeli2"]
    psnde1, psnde2 = fx["psnde1"], fx["psnde2"]
    zdmfup, zdmfdp, zdpmel, zlglac = fx["pdmfup"], fx["pdmfdp"], fx["pdpmel"], fx["plglac"]
    pmflxr, pmflxs, prain = fx["pmflxr"], fx["pmflxs"], fx["prain"]
    pmfude_rate, pmfdde_rate, pqsen_c = fx["pmfude_rate"], fx["pmfdde_rate"], fx["pqsen"]

    # rate / precipitation consistency, :1121-1141 (JK = 2..KLEV-1)
    r = (lev >= 2) & (lev <= K - 1)
    zer = -pmfd + _up(pmfd) + pmfdde_rate
    pmfdde_rate = jnp.where(r & lddraf & (lev >= idtop - 1) & (zer < 0.0),
                            pmfdde_rate - zer, pmfdde_rate)
    cud = r & ldcum & (lev >= kctop - 1)
    zer = pmfu - _nx(pmfu) + pmfude_rate
    pmfude_rate = jnp.where(cud & (zer < 0.0), pmfude_rate - zer, pmfude_rate)
    zdmfup = jnp.where(cud, pmflxr[1:] + pmflxs[1:] - pmflxr[:-1] - pmflxs[:-1], zdmfup)
    zdmfdp = jnp.where(cud, 0.0, zdmfdp)
    # downdraught top humidity, :1145-1157 (RMFSOLTQ > 0)
    ikd = jnp.minimum(idtop + 1, K)
    pmfd = kset(pmfd, idtop, ZDDTOP_Q * kget(pmfd, ikd),
              lddraf & (kget(zmfdq, idtop) < ZDDTOP_Q * kget(zmfdq, ikd)))
    # negative humidities near cloud top, :1161-1188 (JK = 2..KLEV)
    rk = (lev >= 2) & (lev <= K)
    near = rk & ldcum & (lev >= kctop - 1) & (lev < kcbot)
    zdzt = dt * rg / zdz
    zmfa = (_nx(zmfuq) + _nx(zmfdq) - zmfuq - zmfdq + _nx(zmful) - zmful + zdmfup
            - psnde1 - psnde2)
    zmfa = (zmfa - plude) * zdzt
    lim2 = near & (pqen + zmfa < 0.0) & (plude > 0.0)
    zfacl = jnp.minimum(-(pqen + zmfa) / zdzt, plude)
    plude_n = jnp.where(lim2, plude + zfacl, plude)
    pludeli1 = jnp.where(lim2, pludeli1 + zfacl * sdiv(pludeli1, plude_n, lim2), pludeli1)
    pludeli2 = jnp.where(lim2, pludeli2 + zfacl * sdiv(pludeli2, plude_n, lim2), pludeli2)
    negl = near & (plude_n < 0.0)
    plude = jnp.where(negl, 0.0, plude_n)
    pludeli1 = jnp.where(negl, 0.0, pludeli1)
    pludeli2 = jnp.where(negl, 0.0, pludeli2)
    pmfude_rate = jnp.where(rk & ~ldcum, 0.0, pmfude_rate)
    pmfdde_rate = jnp.where(rk & (_up(pmfd) == 0.0), 0.0, pmfdde_rate)

    # 8.0 draught properties back from the fluxes, :1194-1224 (RMFSOLTQ > 0)
    gh = pgeoh[:K + 1]
    blwb = ldcum & (lev >= 2) & (lev > kcbot)
    inc = ldcum & (lev >= 2) & (lev <= kcbot) & (lev >= kctop)
    zmfa_u = 1.0 / jnp.maximum(ZMFA_MIN, pmfu)
    pqu = jnp.where(blwb, zqenh + zmfuq * zmfa_u, pqu)
    ptu = jnp.where(blwb, ztenh + zmfus * zmfa_u * zorcpd, ptu)
    ddb = blwb & lddraf
    zmfa_d = 1.0 / jnp.minimum(-ZMFA_MIN, pmfd)
    zqd = jnp.where(ddb, zqenh + zmfdq * zmfa_d, zqd)
    ztd = jnp.where(ddb, ztenh + zmfds * zmfa_d * zorcpd, ztd)
    zmfus = jnp.where(blwb | inc, pmfu * (rcpd * ptu + gh), zmfus)
    zmfuq = jnp.where(blwb | inc, pmfu * pqu, zmfuq)
    zmfdq = jnp.where(ddb | inc, pmfd * zqd, zmfdq)
    zmfds = jnp.where(ddb | inc, pmfd * (rcpd * ztd + gh), zmfds)
    ptent, ptenq, penth = cudtdqn(
        th, K, dt, kctop, ldcum, lddraf, idtop, pten, pqen, paph, pgeo, pgeoh, ztenh,
        zqenh, pqsen_c, zlglac, plude, pludeli1, pludeli2, psnde1, psnde2, pmfu, pmfd,
        zmfus, zmfds, zmfuq, zmfdq, zmful, zdmfup, zdpmel, ptent, ptenq)

    # 9.0 momentum, LMFDUDV, :1242-1382
    def uu_body(i, c):
        jk = K - 1 - i
        zuu_c, zvu_c = c
        base12 = ldcum & (jk == kcbot) & (ktype < 3)
        base3 = ldcum & (jk == kcbot) & (ktype == 3)
        zuu_c = kset(zuu_c, jk, jnp.where(base3, kget(puen, jk - 1), kget(puen, idpl - 1)), base12 | base3)
        zvu_c = kset(zvu_c, jk, jnp.where(base3, kget(pven, jk - 1), kget(pven, idpl - 1)), base12 | base3)
        inc_ = ldcum & (jk < kcbot) & (jk >= kctop)
        zerate = kget(pmfu, jk) - kget(pmfu, jk + 1) + kget(pmfude_rate, jk)
        zderate = kget(pmfude_rate, jk)
        zmfa_ = 1.0 / jnp.maximum(RMFCMIN, kget(pmfu, jk))
        nu = (kget(zuu_c, jk + 1) * kget(pmfu, jk + 1) + zerate * kget(puen, jk)
              - zderate * kget(zuu_c, jk + 1)) * zmfa_
        nv = (kget(zvu_c, jk + 1) * kget(pmfu, jk + 1) + zerate * kget(pven, jk)
              - zderate * kget(zvu_c, jk + 1)) * zmfa_
        return kset(zuu_c, jk, nu, inc_), kset(zvu_c, jk, nv, inc_)

    zuu, zvu = jax.lax.fori_loop(0, K - 2, uu_body, (zuu, zvu))

    def ud_body(i, c):
        jk = 3 + i
        zud_c, zvd_c = c
        at_ = ldcum & (jk == idtop)
        bl_ = ldcum & (jk > idtop)
        zerate = -kget(pmfd, jk) + kget(pmfd, jk - 1) + kget(pmfdde_rate, jk)
        zmfa_ = 1.0 / jnp.minimum(-RMFCMIN, kget(pmfd, jk))
        nu = (kget(zud_c, jk - 1) * kget(pmfd, jk - 1) - zerate * kget(puen, jk - 1)
              + kget(pmfdde_rate, jk) * kget(zud_c, jk - 1)) * zmfa_
        nv = (kget(zvd_c, jk - 1) * kget(pmfd, jk - 1) - zerate * kget(pven, jk - 1)
              + kget(pmfdde_rate, jk) * kget(zvd_c, jk - 1)) * zmfa_
        nu = jnp.where(at_, 0.5 * (kget(zuu, jk) + kget(puen, jk - 1)), nu)
        nv = jnp.where(at_, 0.5 * (kget(zvu, jk) + kget(pven, jk - 1)), nv)
        return kset(zud_c, jk, nu, at_ | bl_), kset(zvd_c, jk, nv, at_ | bl_)

    zud, zvd = jax.lax.fori_loop(0, K - 2, ud_body, (zud, zvd))
    # momentum mass-flux limiter (RMFSOLUV <= 1), :1294-1318
    zmfmx = (paph[:K + 1] - _up(paph[:K + 1])) * zcons
    hit = (lev >= 2) & ldcum & (lev >= kctop) & (pmfu > zmfmx)
    zmfs_m = jnp.min(jnp.where(hit, sdiv(zmfmx, pmfu, hit), 1.0))
    cs = ldcum & (lev >= kctop - 1)
    zmfuus = jnp.where(cs, pmfu * zmfs_m, pmfu)
    zmfdus = jnp.where(cs, pmfd * zmfs_m, pmfd)
    # below-cloud-base draught winds and the updraught perturbation, :1323-1366
    zmfuub = kget(zmfuus, kcbot) * (kget(zuu, kcbot) - kget(puen, kcbot - 1))
    zmfuvb = kget(zmfuus, kcbot) * (kget(zvu, kcbot) - kget(pven, kcbot - 1))
    zdzb = (paph[K + 1] - paph[:K + 1]) / (paph[K + 1] - kget(paph, kcbot))
    zdzb = jnp.where(ktype == 3, zdzb * zdzb, zdzb)
    bl = ldcum & (lev >= 2) & (lev > kcbot)
    zmfa_b = 1.0 / jnp.maximum(RMFCMIN, zmfuus)
    zuu = jnp.where(bl, _up(puen) + zmfuub * zdzb * zmfa_b, zuu)
    zvu = jnp.where(bl, _up(pven) + zmfuvb * zdzb * zmfa_b, zvu)
    zmfdus = jnp.where(bl, kget(zmfdus, kcbot) * zdzb, zmfdus)
    zud = jnp.where(bl, _up(puen) + kget(zud, kcbot) - kget(puen, kcbot - 1), zud)
    zvd = jnp.where(bl, _up(pven) + kget(zvd, kcbot) - kget(pven, kcbot - 1), zvd)
    pert = ldcum & (lev >= 2) & (lev >= kctop)
    zuu = jnp.where(pert, zuu - jnp.minimum(jnp.abs(zuu), RUVPER) * jnp.copysign(1.0, zuu), zuu)
    zvu = jnp.where(pert, zvu - jnp.minimum(jnp.abs(zvu), RUVPER) * jnp.copysign(1.0, zvu), zvu)
    ptenu, ptenv = cududv(th, K, dt, kctop, ldcum, paph, puen, pven, zmfuus, zmfdus,
                          zuu, zud, zvu, zvd, ptenu, ptenv)
    # dissipation heating, LMFUVDIS, :1384-1421 (JK = 1..KLEV)
    on = ldcum & (lev >= 1) & (lev >= kctop - 1)
    zduten, zdvten = ptenu - ztenu, ptenv - ztenv
    zuv2 = jnp.where(on, ssqrt(zduten ** 2 + zdvten ** 2), 0.0)
    zsum22 = jnp.sum(jnp.where(on, zuv2 * zdz, 0.0))
    zsum12 = jnp.sum(jnp.where(on, -(puen * zduten + pven * zdvten) * zdz, 0.0))
    pvdiscu = zsum12 * zrg
    ptent = jnp.where(on, ptent + zorcpd * zsum12 * zuv2 / jnp.maximum(ZMFA_MIN, zsum22), ptent)

    # 12. final outputs, :1567-1626 (JK = 1..KLEV)
    ikn = jnp.minimum(lev + 1, K)
    ikp = jnp.maximum(lev - 1, 1)
    fin = ldcum & (lev >= 1) & (lev >= kctop - 1)
    lde3 = pmfude_rate * (pqu[ikn] - zqenh[ikn]) + pmfdde_rate * (zqd[ikp] - zqenh[ikp])
    lde4 = pmfude_rate * (ptu[ikn] - ztenh[ikn]) + pmfdde_rate * (ztd[ikp] - ztenh[ikp])
    dgeo = pgeoh[1:K + 1] - pgeoh[2:K + 2]
    zro = rg / jnp.concatenate([dgeo[:1], dgeo])         # 1/dz, index 0: finite copy
    zrho_inv = th.RD * pten * (1.0 + retv * pqen) / pap
    zdiss = spow(jnp.abs(puen * zduten + pven * zdvten), ZDISS_EXP)
    rsud1 = ZRSUD_UNIT * zrho_inv * spow(pmflxr[:K + 1] * ZRSUD_HR * ZAR, ZBR)
    rsud2 = 0.5 * ZRSUD_UNIT * zrho_inv * spow(ZRSUD_SNOW * pmflxs[:K + 1] * ZRSUD_HR * ZAS, ZBS)
    z = lambda x: jnp.where(fin, x, 0.0)
    return CumastrnOut(
        ptent=ptent, ptenq=ptenq, ptenu=ptenu, ptenv=ptenv, ldcum=ldcum, ktype=ktype,
        kcbot=kcbot, kctop=kctop, kbotsc=b["kbotsc"], ldsc=b["ldsc"],
        ptu=jnp.where(fin, ptu, pten), pqu=jnp.where(fin, pqu, pqen), plu=z(plu),
        plude=z(plude), pludeli=jnp.stack([z(pludeli1), z(pludeli2), z(lde3), z(lde4)]),
        psnde=jnp.stack([z(psnde1), z(psnde2)]), penth=z(penth), pmflxr=pmflxr,
        pmflxs=pmflxs, prain=prain, plrain=plrain, prsud=jnp.stack([z(rsud1), z(rsud2)]),
        pmfu=z(pmfu), pmfd=z(pmfd), pmfude_rate=z(pmfude_rate * zro),
        pmfdde_rate=z(pmfdde_rate * zro), pcape=b["pcape"], pwmean=pwmean,
        pvdiscu=pvdiscu, pdiss=z(zdiss), pqsen=pqsen_c,
        pdmfup=z(zdmfup))


def _pad(a):
    """(..., n) surface-last -> (..., n+1) 1-based, index 0 = copy of index 1."""
    return jnp.concatenate([a[..., :1], a], axis=-1)


def cumastrn(th, dt, pgaw, T, q, u, v, plitot, omega, qsat, ahfs, qhfl, p_full,
             p_half, geo_full, geo_half, land, ptent, ptenq, ptenu, ptenv, ptenta,
             ptenqa, ldshcv=True, tun=IfsTunables()):
    """CUMASTRN over a batch of columns.

    Arrays are (ncol, nlev) top-down (half levels (ncol, nlev+1)); ``ahfs``,
    ``qhfl``, ``land``, ``pgaw`` are (ncol,) (``ahfs``/``qhfl`` the surface
    values PAHFS(KLEV+1)/PQHFL(KLEV+1), IFS sign).  NJKT* follow sucumf.F90
    with STPRE := each column's own p_full.  Returns a :class:`CumastrnOut`
    with the padding removed: level arrays (ncol, nlev) [half: nlev+1],
    level indices (KCBOT, KCTOP, KBOTSC) in Fortran numbering 1..nlev.
    """
    if tun.shallow_nosupply not in SHALLOW_NOSUPPLY_MODES:
        raise ValueError(
            f"unknown shallow-closure no-supply scheme "
            f"shallow_nosupply={tun.shallow_nosupply!r}; choose one of "
            f"{SHALLOW_NOSUPPLY_MODES}.")
    if not tun.shallow_supply_width_w_m2 >= 0.0:
        raise ValueError(
            "shallow_supply_width_w_m2 must be >= 0 (0 = hard max(ZDHPBL, 0)), "
            f"got {tun.shallow_supply_width_w_m2!r}.")
    K = T.shape[-1]

    def col(pg, t, qq, uu, vv, lit, om, qs, hf, qf, pf, ph, gf, gh, ld, tt, tq,
            tu, tv, ta, qa):
        pap = _pad(pf)
        o = cumastrn_column(
            th, K, dt, pg, _pad(t), _pad(qq), _pad(uu), _pad(vv), _pad(lit), _pad(om),
            _pad(qs), qf, hf, pap, _pad(ph), _pad(gf), _pad(gh), ld, _pad(tt), _pad(tq),
            _pad(tu), _pad(tv), _pad(ta), _pad(qa), ldshcv, njkt_levels(K, pap), tun)
        return o._replace(**{k: getattr(o, k)[..., 1:] for k in o._fields
                             if getattr(o, k).ndim >= 1})

    return jax.vmap(col)(pgaw, T, q, u, v, plitot, omega, qsat, ahfs, qhfl, p_full,
                         p_half, geo_full, geo_half, land, ptent, ptenq, ptenu, ptenv,
                         ptenta, ptenqa)
