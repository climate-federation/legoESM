"""IFS convection updraught: CUININ, CUBASEN, CUASCN (with CUBASMCN, CUENTR).

Faithful port of OpenIFS (github.com/ecmwf-ifs/openifs @ 8f6f722f73) files
``ifs-source/arpifs/phys_ec/{cuinin,cubasen,cuascn,cubasmcn,cuentr}.F90`` in
the configuration CUMASTRN runs them in the deterministic ECMWF physics:
LPHYLIN = LDTDKMF = LDMIXS = .FALSE., SPP off, NACTAERO = 0, PPLRG = PPLDARE
= 1 (Earth), LSCVLIQ = LMFGLAC = LMFMID = .TRUE., yoecumf values of
sucumf.F90.

Layout: ONE column, 1-based padded arrays exactly as the Fortran indexes
them -- full levels ``a[1..K]`` (``a[0]`` unused), half levels
``a[1..K+1]``; level indices (KCBOT, KCTOP, ...) carry the Fortran values.
The caller vmaps over columns (KLON = 1 per call, as the oracle was run).
Hard switches are ``jnp.where`` (piecewise differentiable, exactly the
source); branches that are not taken are evaluated on sanitised arguments so
no inf/NaN reaches the gradient (value-identical, see ``ssqrt``/``spow``/``sdiv``).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ._ifs_cuthermo import (
    cuadjtq, foealfa, foealfcu, foeewm, QSAT_CAP, RKAP,
)

__physics_contract__ = {
    "summary": (
        "IFS updraught: half-level environment (CUININ), departure-level / "
        "cloud-base search with test parcel (CUBASEN), entraining-detraining "
        "ascent with mid-level initiation, in-plume precipitation and kinetic "
        "energy (CUASCN, CUBASMCN, CUENTR); one column."
    ),
    "inputs": {"T": "K", "q": "kg kg-1", "p_full": "Pa", "p_half": "Pa",
               "geopotential": "m2 s-2", "surface fluxes": "IFS sign (negative = upward)"},
    "outputs": {"updraught T/q/l": "K, kg kg-1", "mass flux": "kg m-2 s-1",
                "detrainment, precipitation fluxes": "kg m-2 s-1"},
    "sign_convention": "mass fluxes positive upward; levels top-down, surface = K",
    "conserves": ["none"],
    "differentiable": True,
    "reference": "OpenIFS 8f6f722 cuinin/cubasen/cuascn/cubasmcn/cuentr.F90",
    "idealized_test": "tests/atmosphere/test_bechtold_ifs_oracle.py",
}

# --- yoecumf values (sucumf.F90 @ 8f6f722, RPLRG = RPLDARE = 1) ---
ENTRORG = 1.75e-3     # sucumf.F90:136  organized entrainment [1/m]
ENTR_RH = 1.0         # sucumf.F90:140
ENTSHALP = 2.0        # sucumf.F90:144  shallow/mid-level entrainment factor
ENTSTPC1 = 0.8        # sucumf.F90:147  test-parcel 1/z entrainment
ENTSTPC2 = 2.0e-4     # sucumf.F90:148
DETRPEN = 0.75e-4     # sucumf.F90:129  turbulent detrainment [1/m]
RMFCMIN = 1.0e-8      # sucumf.F90:160  minimum mass flux (safety)
RDEPTHS = 2.0e4       # sucumf.F90:170  deep/shallow depth split [Pa]
RPRCON = 1.4e-3       # sucumf.F90:175  conversion rate [1/m]
RMFLIA = 2.0          # sucumf.F90:241  absolute mass-flux limit
RMFCFL = 3.0          # sucumf.F90:248  CFL multiple (RMFSOLTQ > 0.5)
RLMIN = 1.0e-8        # sucldp.F90 default (YRECLDP%RLMIN)

# --- cubasen.F90 fixed numbers ---
ZTEXC0 = 0.2          # :331,:373  minimum T excess [K]
ZQEXC0 = 1.0e-4       # :332,:374  minimum q excess [kg/kg]
ZTEX_MAX = 1.0        # :346,:381  maximum T excess [K]
ZQEX_MAX = 5.0e-4     # :347,:382
ZUST_MIN = 0.1        # :338  ustar = MAX(SQRT(PKMFL), 0.1), PKMFL = 0 from CUMASTRN
ZWS_FAC = 1.5         # :339,:343-344  w* / excess factor
ZWS_EXP = 0.3333      # :341
ZWS_SCALE = 1.2       # :341
ZWU2_SFC_ADD = 0.1    # :352  surface parcel KE offset [m2/s2]
ZRH_ADV = 0.9         # :386  land advection perturbation RH limit
ZQADV_MAX = 3.0e-4    # :387
ZQADV_TAU = 600.0     # :387  [s]
ZMIXL_DP = 60.0e2     # :394  mixed-layer departure test [Pa]
ZMIXL_DEPTH = 50.0e2  # :399  mixed-layer depth [Pa]
ZMIX_DEEP = 0.4       # :460  test-parcel entrainment factor (deep)

# --- cuascn.F90 fixed numbers ---
ZDNOPRC = 3.0e-4      # :277  condensate threshold for precipitation
Z_CLDMAX = 5.0e-3     # :278
Z_CWIFRAC = 0.5       # :279
Z_CPRC2 = 0.5         # :280
Z_CWDRAG = (3.0 / 8.0) * 0.506 / 0.2   # :281
ZFACBUO = 0.5 / (1.0 + 0.5)            # :274
ZKCTOP0_P = 4.0e4     # :380  cloud-top estimate [Pa]
ZDETR_MAX = 0.75      # :463
ZOENTR_MAX = 0.4      # :472,:621
ZDETR_RH = 1.6        # :489,:606
ZTU_MIN = 100.0       # :515
ZTU_MAX = 400.0       # :516
ZMID_RH = 0.80        # cubasmcn.F90:116
ZMID_MF_MIN = 1.0e-6  # cubasmcn.F90:122
ZLSCV_DT = 20.0       # :444  liquid-only mid-level if T(KCBOT) > RTT-20
ZBUO_MID = -0.5       # :572
ZBUO_REINT = -0.1     # :587
ZKE_MIN = -1.0e3      # :606
ZKEDKE_FLOOR = 1.0e-3 # :610
ZBUO_ORG = -0.2       # :617
ZOENTR_RH = 0.3       # :618
ZBUO_TOP = -2.0       # :634
ZLAPSE_TOP = -3.0e-3  # :635  [K/m]
ZWU_MAX = 15.0        # :656
ZKE_PRC_MIN = 0.5     # :656
ZKE_RAIN_MIN = 0.1    # :731
ZZCO_LIQ = 0.3        # :660
ZPRC_W = 0.75         # :671
ZVW_COEF = 21.18      # :724
ZVW_EXP = 0.2         # :724
ZWMEAN_MIN = 1.0e-2   # :863  floor of the mean updraught KE [m2 s-2]
ZDPMEAN_MIN = 1.0     # :863  floor of the averaging depth [Pa]


# ---------------------------------------------------------------------------
# helpers (one column, 1-based arrays)
# ---------------------------------------------------------------------------
def kget(a, k):
    """a(k) with the index clipped into range (only ever read in a branch the
    Fortran guards; clipping keeps the discarded lane finite)."""
    return a[jnp.clip(k, 0, a.shape[0] - 1)]


def kset(a, k, v, m=True):
    """a(k) = v where m."""
    kk = jnp.clip(k, 0, a.shape[0] - 1)
    return a.at[kk].set(jnp.where(m, v, a[kk]))


def ssqrt(x):
    """SQRT(x) for x >= 0; x <= 0 -> 0 without an infinite derivative."""
    pos = x > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 1.0)), 0.0)


def spow(x, e):
    """x**e for x >= 0 (real exponent); 0 -> 0 without an infinite derivative."""
    pos = x > 0.0
    return jnp.where(pos, jnp.power(jnp.where(pos, x, 1.0), e), 0.0)


def sdiv(a, b, ok):
    """a/b where ok; a/1 elsewhere (the discarded lane stays finite)."""
    return a / jnp.where(ok, b, 1.0)


def _fsign_half(x):
    """0.5 + SIGN(0.5, x): 1 for x >= 0, 0 for x < 0."""
    return jnp.where(x >= 0.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# CUININ (cuinin.F90:139-199)
# ---------------------------------------------------------------------------
def cuinin(th, K, pten, pqen, pqsen, puen, pven, pgeo, paph, pgeoh, njkt2):
    jk = jnp.arange(K + 1)
    up = jnp.concatenate([pten[:1], pten[:-1]])     # PTEN(JK-1)
    gup = jnp.concatenate([pgeo[:1], pgeo[:-1]])
    qup = jnp.concatenate([pqen[:1], pqen[:-1]])
    qsup = jnp.concatenate([pqsen[:1], pqsen[:-1]])
    rcpd = th.RCPD
    zorcpd = 1.0 / rcpd
    ptenh = (jnp.maximum(rcpd * up + gup, rcpd * pten + pgeo) - pgeoh[:K + 1]) * zorcpd
    pqenh = qup
    pqsenh = qsup
    adj = (jk >= 2) & (jk < K - 1) & (jk >= njkt2)
    # levels outside the window (incl. the model top, PAPH(1) may be 0) are
    # never adjusted (cuinin.F90:154); a finite pressure there keeps the
    # discarded lane's gradient finite (value-identical)
    ta, qsa = cuadjtq(th, 3, jnp.where(adj, paph[:K + 1], paph[K]), ptenh, pqsenh)
    ptenh = jnp.where(adj, ta, ptenh)
    pqsenh = jnp.where(adj, qsa, pqsenh)
    pqenh = jnp.where(adj, jnp.maximum(jnp.minimum(qup, qsup) + (pqsenh - qsup), 0.0), pqenh)
    ptenh = ptenh.at[K].set((rcpd * pten[K] + pgeo[K] - pgeoh[K]) * zorcpd)
    pqenh = pqenh.at[K].set(pqen[K])
    ptenh = ptenh.at[1].set(pten[1])
    pqenh = pqenh.at[1].set(pqen[1])
    # index 0 has no Fortran counterpart; a finite copy of level 1 keeps
    # masked-out lanes (and their gradients) finite.
    ptenh = ptenh.at[0].set(pten[1])
    pqenh = pqenh.at[0].set(pqen[1])
    ik = jnp.maximum(jk - 1, 1)
    puu = puen[ik]
    pvu = pven[ik]
    return ptenh, pqenh, pqsenh, puu, pvu


# ---------------------------------------------------------------------------
# CUBASEN (cubasen.F90:247-746), KINDEX = NJKT1, LDMIXS = LDTDKMF = .FALSE.
# ---------------------------------------------------------------------------
def cubasen(th, K, ptenh, pqenh, pgeoh, paph, pqhfl_sfc, pahfs_sfc, ptenqa,
            pten, pqen, pqsen, pgeo, land, njkt1, njkt2, njkt6, entrorg=ENTRORG):
    """Returns dict: ptu, pqu, plu, klab, ldcum, ldsc, kcbot, kbotsc, kctop
    (ICTOP0), kdpl, pwubase, pcape."""
    rg, rcpd, rd, retv = th.RG, th.RCPD, th.RD, th.RETV
    zrg = 1.0 / rg
    zrcpd = 1.0 / rcpd
    f = pten.dtype
    i32 = jnp.int32
    zs_enh = rcpd * ptenh + pgeoh[:K + 1]             # ZSENH(1..K)
    zqenh = pqenh                                      # ZQENH(1..K)
    zero = jnp.zeros(K + 1, f)
    kindex = njkt1
    jkt2 = njkt2

    st = dict(
        ztu=ptenh, zqu=pqenh, zlu=zero, ilab=jnp.zeros(K + 1, i32),
        zsuh=zero, zbuoh=zero, zwu2h=zero, zcape=zero,
        go=jnp.bool_(True), first=jnp.bool_(True), deep=jnp.bool_(False),
        ztex=jnp.zeros((), f), zqex=jnp.zeros((), f),
        idpl=i32(K), icbot=i32(K), ibotsc=i32(K - 1), ictop=i32(K - 1),
        lldcum=jnp.bool_(False), lldsc=jnp.bool_(False),
        # outputs
        ptu=ptenh, pqu=pqenh, plu=zero, klab=jnp.zeros(K + 1, i32),
        ldcum=jnp.bool_(False), ldsc=jnp.bool_(False), kcbot=i32(-1),
        kbotsc=i32(-1), kctop=i32(-1), kdpl=i32(K), pwubase=jnp.zeros((), f),
    )

    def outer(i, st):
        jkk = K - i
        active = jkk >= kindex
        st = dict(st)
        go0 = st["go"] & active
        # 1.2 initialise at the departure half level
        st["idpl"] = jnp.where(go0, jkk, st["idpl"])
        st["icbot"] = jnp.where(go0, jkk, st["icbot"])
        st["ibotsc"] = jnp.where(go0, K - 1, st["ibotsc"])
        st["ictop"] = jnp.where(go0, K - 1, st["ictop"])
        st["lldcum"] = jnp.where(go0, False, st["lldcum"])
        st["lldsc"] = jnp.where(go0, False, st["lldsc"])
        surf = jkk == K
        # --- surface departure (JKK == KLEV), :335-370
        zrho_s = paph[K + 1] / (rd * (pten[K] * (1.0 + retv * pqen[K])))
        zkhvfl = (pahfs_sfc * zrcpd + retv * pten[K] * pqhfl_sfc) / zrho_s
        zws0 = (ZUST_MIN ** 3.0
                - ZWS_FAC * RKAP * zkhvfl * (pgeoh[K] - pgeoh[K + 1]) / pten[K])
        unst = zkhvfl < 0.0
        zws = ZWS_SCALE * spow(jnp.where(unst, zws0, 1.0), ZWS_EXP)
        zws_ok = jnp.where(unst, zws, 1.0)
        ztex = jnp.maximum(-ZWS_FAC * pahfs_sfc / (zrho_s * zws_ok * rcpd), ZTEXC0)
        zqex = jnp.maximum(-ZWS_FAC * pqhfl_sfc / (zrho_s * zws_ok), ZQEXC0)
        ztex = jnp.minimum(ztex, ZTEX_MAX)
        zqex = jnp.minimum(zqex, ZQEX_MAX)
        s_qu = zqenh[K] + zqex
        s_suh = zs_enh[K] + rcpd * ztex
        s_tu = (zs_enh[K] - pgeoh[K]) * zrcpd + ztex
        s_wu2 = zws ** 2 + ZWU2_SFC_ADD
        # --- elevated departure (JKK < KLEV), :372-413
        ztexc = jnp.asarray(ZTEXC0, f)
        zqexc = jnp.asarray(ZQEXC0, f)
        km1 = jkk == K - 1
        ztexc = jnp.where(km1, jnp.minimum(jnp.maximum(ztexc, st["ztex"]), ZTEX_MAX), ztexc)
        zqexc = jnp.where(km1, jnp.minimum(jnp.maximum(zqexc, st["zqex"]), ZQEX_MAX), zqexc)
        gate = ((jkk < K - 1) & (jkk > njkt6)
                & (kget(zqenh, jkk) / kget(pqsen, jkk) < ZRH_ADV))
        zadv = jnp.minimum(ZQADV_MAX, jnp.maximum(0.0, kget(ptenqa, jkk) * ZQADV_TAU))
        # LDLAND as the land fraction (exact for land in {0, 1})
        zqexc = zqexc + land * jnp.where(gate, zadv, 0.0)
        e_qu = kget(zqenh, jkk) + zqexc
        e_suh = kget(zs_enh, jkk) + rcpd * ztexc
        e_tu = (kget(zs_enh, jkk) - kget(pgeoh, jkk)) * zrcpd + ztexc
        mixl = paph[K + 1] - kget(paph, jkk - 1) < ZMIXL_DP
        zw1 = jnp.zeros((), f)
        sq = jnp.zeros((), f)
        ss = jnp.zeros((), f)
        for d in (1, 0, -1):                               # JK = JKK+1, JKK, JKK-1
            jkm = jkk + d
            take = zw1 < ZMIXL_DEPTH
            zw2 = kget(paph, jkm) - kget(paph, jkm - 1)
            zw1 = jnp.where(take, zw1 + zw2, zw1)
            sq = jnp.where(take, sq + kget(zqenh, jkm) * zw2, sq)
            ss = jnp.where(take, ss + kget(zs_enh, jkm) * zw2, ss)
        zw1s = jnp.where(mixl, zw1, 1.0)
        m_qu = sq / zw1s + zqexc
        m_suh = ss / zw1s + rcpd * ztexc
        m_tu = (m_suh - kget(pgeoh, jkk)) * zrcpd + ztexc
        e_qu = jnp.where(mixl, m_qu, e_qu)
        e_suh = jnp.where(mixl, m_suh, e_suh)
        e_tu = jnp.where(mixl, m_tu, e_tu)
        # combine
        set_s = go0 & surf & unst
        set_e = go0 & ~surf
        setp = set_s | set_e
        qu0 = jnp.where(surf, s_qu, e_qu)
        suh0 = jnp.where(surf, s_suh, e_suh)
        tu0 = jnp.where(surf, s_tu, e_tu)
        wu20 = jnp.where(surf, s_wu2, 1.0)
        ztvenh = (1.0 + retv * kget(zqenh, jkk)) * (kget(zs_enh, jkk) - kget(pgeoh, jkk)) * zrcpd
        ztvuh = (1.0 + retv * qu0) * tu0
        buo0 = (ztvuh - ztvenh) * rg / ztvenh
        st["ilab"] = kset(st["ilab"], jkk, 1, setp)
        st["zqu"] = kset(st["zqu"], jkk, qu0, setp)
        st["zsuh"] = kset(st["zsuh"], jkk, suh0, setp)
        st["ztu"] = kset(st["ztu"], jkk, tu0, setp)
        st["zlu"] = kset(st["zlu"], jkk, 0.0, setp)
        st["zwu2h"] = kset(st["zwu2h"], jkk, wu20, setp)
        st["zbuoh"] = kset(st["zbuoh"], jkk, buo0, setp)
        st["ztex"] = jnp.where(go0 & surf, jnp.where(unst, ztex, 0.0), st["ztex"])
        st["zqex"] = jnp.where(go0 & surf, jnp.where(unst, zqex, 0.0), st["zqex"])
        st["go"] = jnp.where(go0 & surf & ~unst, False, st["go"])
        shallow = surf | (kindex == K - 1)

        # 2.0 ascent JK = JKK-1 .. JKT2
        def inner(i2, s):
            jk = K - 1 - i2
            s = dict(s)
            on = s["go"] & active & (jk <= jkk - 1) & (jk >= jkt2)
            zdz = (kget(pgeoh, jk) - kget(pgeoh, jk + 1)) * zrg
            zqf = (kget(pqenh, jk + 1) + kget(pqenh, jk)) * 0.5
            zsf = (kget(zs_enh, jk + 1) + kget(zs_enh, jk)) * 0.5
            qu1 = kget(s["zqu"], jk + 1)
            suh1 = kget(s["zsuh"], jk + 1)
            # shallow: 1/z mixing, :428-441
            zeps = ENTSTPC1 / ((kget(pgeo, jk) - pgeoh[K + 1]) * zrg) + ENTSTPC2
            mix_s = jnp.minimum(1.0, 0.5 * zdz * zeps)
            ztmp = 1.0 / (1.0 + mix_s)
            qu_s = (qu1 * (1.0 - mix_s) + 2.0 * mix_s * zqf) * ztmp
            suh_s = (suh1 * (1.0 - mix_s) + 2.0 * mix_s * zsf) * ztmp
            # deep: :454-472
            mix_d = ZMIX_DEEP * entrorg * zdz * jnp.minimum(
                1.0, (kget(pqsen, jk) / pqsen[K]) ** 3)
            mix_d = jnp.minimum(1.0, mix_d)
            qu_d = qu1 * (1.0 - mix_d) + zqf * mix_d
            suh_d = suh1 * (1.0 - mix_d) + zsf * mix_d
            zmix = jnp.where(shallow, mix_s, mix_d)
            qu = jnp.where(shallow, qu_s, qu_d)
            suh = jnp.where(shallow, suh_s, suh_d)
            zqold = qu
            tu = (suh - kget(pgeoh, jk)) * zrcpd
            # lanes that are not ascending carry no parcel: evaluate them on the
            # environment so no overflow reaches the gradient (value-identical,
            # nothing from such a lane is written)
            tu = jnp.where(on, tu, kget(ptenh, jk))
            qu = jnp.where(on, qu, kget(pqenh, jk))
            tu, qu = cuadjtq(th, 1, kget(paph, jk), tu, qu, on)
            zdq = jnp.maximum(zqold - qu, 0.0)
            lu = kget(s["zlu"], jk + 1) + zdq
            zlglac = zdq * ((1.0 - foealfcu(th, tu))
                            - (1.0 - foealfcu(th, kget(s["ztu"], jk + 1))))
            lu = 0.5 * lu
            tu = tu + th.RALFDCP * zlglac
            suh = rcpd * tu + kget(pgeoh, jk)
            ztvuh = (1.0 + retv * qu - lu) * tu + th.RALFDCP * zlglac
            ztvenh = (1.0 + retv * kget(zqenh, jk)) * (kget(zs_enh, jk) - kget(pgeoh, jk)) * zrcpd
            buoh = (ztvuh - ztvenh) * rg / ztvenh
            zbuof = (buoh + kget(s["zbuoh"], jk + 1)) * 0.5
            ztmp = 1.0 / (1.0 + 2.0 * zmix)
            wu2 = (kget(s["zwu2h"], jk + 1) * (1.0 - 2.0 * zmix) + 2.0 * zbuof * zdz) * ztmp
            s["zcape"] = kset(s["zcape"], jkk, kget(s["zcape"], jkk)
                            + jnp.maximum(0.0, zbuof * zdz), on)
            # exact cloud base, :517-555
            ik = jk + 1
            tuk = kget(s["ztu"], ik)
            pk = kget(paph, ik)
            zqsu = foeewm(th, tuk) / pk
            zesdp = zqsu
            zqsu = jnp.minimum(QSAT_CAP, zqsu)
            zcor = 1.0 / (1.0 - retv * zqsu)
            zqsu = zqsu * zcor
            zdqb = jnp.minimum(0.0, kget(s["zqu"], ik) - zqsu)
            zalfaw = foealfa(th, tuk)
            zfacw = th.R5LES / ((tuk - th.R4LES) ** 2)
            zfaci = th.R5IES / ((tuk - th.R4IES) ** 2)
            zfac = zalfaw * zfacw + (1.0 - zalfaw) * zfaci
            zcor = 1.0 / (1.0 - retv * zesdp)
            zdqsdt = zfac * zcor * zqsu
            zdtdp = rd * tuk / (rcpd * pk)
            zdp = zdqb / (zdqsdt * zdtdp)
            zcbase = pk + zdp
            ztop = zcbase - kget(paph, jk)
            zbot = kget(paph, jk + 1) - zcbase
            cb = on & (lu > 0.0) & (kget(s["ilab"], jk + 1) == 1)
            c1 = cb & (ztop > zbot) & (kget(s["zwu2h"], jk + 1) > 0.0)
            c2 = cb & ~c1 & (ztop <= zbot) & (wu2 > 0.0)
            jkb = jnp.minimum(K - 1, jk + 1)
            # write this level (needed before the base flags overwrite ILAB)
            s["zqu"] = kset(s["zqu"], jk, qu, on)
            s["zsuh"] = kset(s["zsuh"], jk, suh, on)
            s["ztu"] = kset(s["ztu"], jk, tu, on)
            s["zlu"] = kset(s["zlu"], jk, lu, on)
            s["zbuoh"] = kset(s["zbuoh"], jk, buoh, on)
            s["zwu2h"] = kset(s["zwu2h"], jk, wu2, on)
            s["ilab"] = kset(s["ilab"], jkb, 2, c1)
            s["ilab"] = kset(s["ilab"], jk, 2, c1 | c2)
            s["lldsc"] = s["lldsc"] | c1 | c2
            s["ibotsc"] = jnp.where(c1, jkb, jnp.where(c2, jk, s["ibotsc"]))
            s["icbot"] = jnp.where(c1, jkb, jnp.where(c2, jk, s["icbot"]))
            s["zlu"] = kset(s["zlu"], jk + 1, RLMIN, c1)
            # convection presence / top, :560-573
            stop = on & (wu2 < 0.0)
            hasl = kget(s["zlu"], jk + 1) > 0.0
            s["ictop"] = jnp.where(stop & hasl, jk, s["ictop"])
            s["lldcum"] = jnp.where(stop, hasl, s["lldcum"])
            s["go"] = jnp.where(stop, False, s["go"])
            s["ilab"] = kset(s["ilab"], jk, jnp.where(lu > 0.0, 2, 1), on & ~stop)
            return s

        st = jax.lax.fori_loop(0, K - 2, inner, st)

        # surface-parcel bookkeeping, :578-617
        sfc = active & (surf | (km1 & (kindex == K - 1)))
        jkt, jkb = st["ictop"], st["icbot"]
        lldeep = kget(paph, jkb) - kget(paph, jkt) > RDEPTHS
        lldcum = jnp.where(lldeep & (kindex < K - 1), False, st["lldcum"])
        go_s = jnp.where(kindex == K - 1, ~lldcum, True)
        wub = ssqrt(jnp.maximum(kget(st["zwu2h"], jkb), 0.0))
        st["lldcum"] = jnp.where(sfc, lldcum, st["lldcum"])
        st["ldsc"] = jnp.where(sfc, st["lldsc"], st["ldsc"])
        st["kbotsc"] = jnp.where(sfc, jnp.where(st["lldsc"], st["ibotsc"], -1), st["kbotsc"])
        st["deep"] = jnp.where(sfc, False, st["deep"])
        st["go"] = jnp.where(sfc, go_s, st["go"])
        st["kcbot"] = jnp.where(sfc, jnp.where(lldcum, jkb, -1), st["kcbot"])
        st["kctop"] = jnp.where(sfc, jnp.where(lldcum, jkt, -1), st["kctop"])
        st["kdpl"] = jnp.where(sfc, jnp.where(lldcum, st["idpl"], K - 1), st["kdpl"])
        st["ldcum"] = jnp.where(sfc, lldcum, st["ldcum"])
        st["pwubase"] = jnp.where(sfc, jnp.where(lldcum, wub, 0.0), st["pwubase"])
        lev = jnp.arange(K + 1)
        cp = sfc & ((lev >= jkt) | ((kindex >= K - 1) & surf)) & (lev >= 1)
        for nm, src in (("klab", "ilab"), ("ptu", "ztu"), ("pqu", "zqu"), ("plu", "zlu")):
            st[nm] = jnp.where(cp, st[src], st[nm])

        # deep departure selection, :619-670
        el = active & (jkk < K)
        jkt, jkb = st["ictop"], st["icbot"]
        newdeep = jnp.where(st["deep"], True,
                            kget(paph, jkb) - kget(paph, jkt) >= RDEPTHS)
        deep = jnp.where(el, newdeep, st["deep"])
        reset = el & deep & st["first"]
        jkbd = st["idpl"]
        inc = (lev <= jkbd) & (lev >= jkt)
        st["klab"] = jnp.where(reset, jnp.where(lev < jkt, 0, jnp.where(inc, st["ilab"], 1)),
                               st["klab"])
        st["ptu"] = jnp.where(reset, jnp.where(inc, st["ztu"], ptenh), st["ptu"])
        st["pqu"] = jnp.where(reset, jnp.where(inc, st["zqu"], pqenh), st["pqu"])
        st["plu"] = jnp.where(reset, jnp.where(inc, st["zlu"], 0.0), st["plu"])
        st["kdpl"] = jnp.where(reset, st["idpl"], st["kdpl"])
        st["kctop"] = jnp.where(reset, st["ictop"], st["kctop"])
        st["kcbot"] = jnp.where(reset, st["icbot"], st["kcbot"])
        st["ldcum"] = jnp.where(reset, st["lldcum"], st["ldcum"])
        st["ldsc"] = jnp.where(reset, False, st["ldsc"])
        st["kbotsc"] = jnp.where(reset, -1, st["kbotsc"])
        st["pwubase"] = jnp.where(
            reset, ssqrt(jnp.maximum(kget(st["zwu2h"], st["icbot"]), 0.0)), st["pwubase"])
        st["first"] = jnp.where(reset, False, st["first"])
        st["go"] = jnp.where(el, ~deep, st["go"])
        st["deep"] = deep
        return st

    st = jax.lax.fori_loop(0, K - 1, outer, st)
    pcape = jnp.max(st["zcape"][1:])
    return dict(ptu=st["ptu"], pqu=st["pqu"], plu=st["plu"], klab=st["klab"],
                ldcum=st["ldcum"], ldsc=st["ldsc"], kcbot=st["kcbot"],
                kbotsc=st["kbotsc"], kctop=st["kctop"], kdpl=st["kdpl"],
                pwubase=st["pwubase"], pcape=pcape)


# ---------------------------------------------------------------------------
# CUASCN (cuascn.F90:283-866) with CUBASMCN and CUENTR inlined
# ---------------------------------------------------------------------------
def cuascn(th, K, dt, ptenh, pqenh, pten, pqen, pqsen, plitot, pgeo, pgeoh,
           pap, paph, pvervel, pwubase, ldcum, ktype, klab, ptu, pqu, plu,
           pmfub, kcbot, kdpl, njkt7, rprcon=RPRCON, zdnoprc=ZDNOPRC,
           entrorg=ENTRORG, detrpen=DETRPEN, entshalp=ENTSHALP):
    """Returns dict of the CUASCN outputs (incl. the INOUT PTENH/PQENH)."""
    rg, rcpd, retv, rtt = th.RG, th.RCPD, th.RETV, th.RTT
    f = pten.dtype
    i32 = jnp.int32
    zcons2 = RMFCFL / (rg * dt)
    zrg = 1.0 / rg
    zorcpd = 1.0 / rcpd
    zprcdgw = rprcon / rg
    zero = jnp.zeros(K + 1, f)
    lev = jnp.arange(K + 1)

    # 2. defaults, :334-409
    kcbot = jnp.where(ldcum, kcbot, -1)
    pmfub = jnp.where(ldcum, pmfub, 0.0)
    pqu = pqu.at[K].set(jnp.where(ldcum, pqu[K], 0.0))
    ktype = jnp.where(ldcum, ktype, 0)
    lscv = ktype > 1                                   # LSCVLIQ = .TRUE.
    llklab = ~ldcum | (ktype == 3)
    plu = jnp.where(lev == kcbot, plu, 0.0)
    klab = jnp.where(llklab, 0, klab)
    ldcum = jnp.where(ktype == 3, False, ldcum)
    # 3. cloud base values, :414-424
    kctop = kcbot
    ikb = kcbot
    pkineu = zero
    pmfu = zero
    pmfus = zero
    pmfuq = zero
    pmful = zero
    pkineu = kset(pkineu, ikb, 0.5 * pwubase ** 2, ldcum)
    pmfu = kset(pmfu, ikb, pmfub, ldcum)
    pmfus = kset(pmfus, ikb, pmfub * (rcpd * kget(ptu, ikb) + kget(pgeoh, ikb)), ldcum)
    pmfuq = kset(pmfuq, ikb, pmfub * kget(pqu, ikb), ldcum)
    pmful = kset(pmful, ikb, pmfub * kget(plu, ikb), ldcum)

    st = dict(
        ptenh=ptenh, pqenh=pqenh, ldcum=ldcum, ktype=ktype, klab=klab,
        ptu=ptu, pqu=pqu, plu=plu, plrain=zero, pmfu=pmfu, pmfub=pmfub,
        plglac=zero, pmfus=pmfus, pmfuq=pmfuq, pmful=pmful, plude=zero,
        pludeli1=zero, pludeli2=zero, pdmfup=zero, pdmfen=zero, kcbot=kcbot,
        kctop=kctop, pmfude_rate=zero, pkineu=pkineu, zbuo=zero,
        pwmean=jnp.zeros((), f), zdpmean=jnp.zeros((), f),
        zoentr=jnp.zeros((), f), zluold=jnp.zeros((), f), lscv=lscv,
        llo3=jnp.bool_(False), zdmfen=jnp.zeros((), f), zdmfde=jnp.zeros((), f),
    )

    def level(i, st):
        jk = K - 1 - i
        s = dict(st)
        # --- CUBASMCN (KK = JK), cubasmcn.F90:113-138
        kk = jk
        mid = ((~s["ldcum"]) & (kget(s["klab"], kk + 1) == 0) & (kk > njkt7)
               & (kget(pqen, kk) > ZMID_RH * kget(pqsen, kk)))
        tum = (rcpd * kget(pten, kk) + kget(pgeo, kk) - kget(pgeoh, kk + 1)) * zorcpd
        zzzmb = jnp.minimum(jnp.maximum(ZMID_MF_MIN, -kget(pvervel, kk) * zrg), RMFLIA)
        s["ptu"] = kset(s["ptu"], kk + 1, tum, mid)
        s["pqu"] = kset(s["pqu"], kk + 1, kget(pqen, kk), mid)
        s["plu"] = kset(s["plu"], kk + 1, 0.0, mid)
        s["pmfub"] = jnp.where(mid, zzzmb, s["pmfub"])
        s["pmfu"] = kset(s["pmfu"], kk + 1, zzzmb, mid)
        s["pmfus"] = kset(s["pmfus"], kk + 1, zzzmb * (rcpd * tum + kget(pgeoh, kk + 1)), mid)
        s["pmfuq"] = kset(s["pmfuq"], kk + 1, zzzmb * kget(pqen, kk), mid)
        s["pmful"] = kset(s["pmful"], kk + 1, 0.0, mid)
        s["pdmfup"] = kset(s["pdmfup"], kk + 1, 0.0, mid)
        s["plrain"] = kset(s["plrain"], kk + 1, 0.0, mid)
        s["kcbot"] = jnp.where(mid, kk, s["kcbot"])
        s["klab"] = kset(s["klab"], kk + 1, 1, mid)
        s["ktype"] = jnp.where(mid, 3, s["ktype"])
        # --- per-level preamble, :442-466
        kt3 = s["ktype"] == 3
        s["lscv"] = s["lscv"] | (kt3 & (kget(pten, s["kcbot"]) > rtt - ZLSCV_DT))
        klab_up = kget(s["klab"], jk + 1)
        s["klab"] = kset(s["klab"], jk, 0, klab_up == 0)
        llflag = (s["ldcum"] & (klab_up == 2)) | (kt3 & (klab_up == 1))
        zph = kget(paph, jk)
        zmfmax = (kget(paph, jk) - kget(paph, jk - 1)) * zcons2
        capm = kt3 & (jk == s["kcbot"]) & (s["pmfub"] > zmfmax)
        zfac = sdiv(zmfmax, s["pmfub"], capm)
        for nm in ("pmfu", "pmfus", "pmfuq"):
            s[nm] = kset(s[nm], jk + 1, kget(s[nm], jk + 1) * zfac, capm)
        s["pmfub"] = jnp.where(capm, zmfmax, s["pmfub"])
        s["llo3"] = s["llo3"] | (klab_up > 0)
        llo3 = s["llo3"]
        # --- CUENTR (LDWORK = LLO3), cuentr.F90:124-151
        ldc, kcb = s["ldcum"], s["kcbot"]
        zdz = (kget(pgeoh, jk) - kget(pgeoh, jk + 1)) * zrg
        zmf = kget(s["pmfu"], jk + 1) * zdz
        ent = ldc & (jk < kcb)
        zdmfen = jnp.where(llo3, jnp.where(ent, 0.0 * zmf, 0.0), s["zdmfen"])
        zdmfde = jnp.where(llo3, jnp.where(ent, detrpen * zmf, 0.0), s["zdmfde"])
        # --- ascent for LLFLAG, :478-524
        mfu1 = kget(s["pmfu"], jk + 1)
        fl = llo3 & llflag
        zdmfde_f = jnp.minimum(zdmfde, ZDETR_MAX * mfu1)
        rh = jnp.minimum(1.0, kget(pqen, jk) / kget(pqsen, jk))
        atb = jk == kcb
        zoentr_b = -entrorg * (rh - ENTR_RH) * (kget(pgeoh, jk) - kget(pgeoh, jk + 1)) * zrg
        zoentr_b = jnp.minimum(ZOENTR_MAX, zoentr_b) * mfu1
        zoentr = jnp.where(fl & atb, zoentr_b, s["zoentr"])
        blw = jk < kcb
        zxs = jnp.maximum(mfu1 - zmfmax, 0.0)
        s["pwmean"] = jnp.where(fl & blw, s["pwmean"] + kget(s["pkineu"], jk + 1)
                                * (kget(pap, jk + 1) - kget(pap, jk)), s["pwmean"])
        s["zdpmean"] = jnp.where(fl & blw, s["zdpmean"] + kget(pap, jk + 1) - kget(pap, jk),
                                 s["zdpmean"])
        en_b = zoentr
        en_b = jnp.where(s["ktype"] >= 2, entshalp * en_b, en_b)
        de_b = jnp.where(s["ktype"] >= 2, en_b, zdmfde_f)
        de_b = de_b * (ZDETR_RH - rh)
        zmftest = mfu1 + en_b - de_b
        zchange = jnp.maximum(zmftest - zmfmax, 0.0)
        zxe = jnp.maximum(zchange - zxs, 0.0)
        en_b = en_b - zxe
        zchange = zchange - zxe
        de_b = de_b + zchange
        zdmfen = jnp.where(fl, jnp.where(blw, en_b, zdmfen), zdmfen)
        zdmfde = jnp.where(fl, jnp.where(blw, de_b, zdmfde_f), zdmfde)
        mfu = mfu1 + zdmfen - zdmfde
        zqeen = kget(s["pqenh"], jk + 1) * zdmfen
        zseen = (rcpd * kget(s["ptenh"], jk + 1) + kget(pgeoh, jk + 1)) * zdmfen
        lit = kget(plitot, jk)
        zleen = jnp.where(lit > RLMIN, lit * zdmfen, 0.0)
        zscde = (rcpd * kget(s["ptu"], jk + 1) + kget(pgeoh, jk + 1)) * zdmfde
        zqude = kget(s["pqu"], jk + 1) * zdmfde
        lude = kget(s["plu"], jk + 1) * zdmfde
        zmfusk = kget(s["pmfus"], jk + 1) + zseen - zscde
        zmfuqk = kget(s["pmfuq"], jk + 1) + zqeen - zqude
        zmfulk = kget(s["pmful"], jk + 1) + zleen - lude
        zfac = 1.0 / jnp.maximum(RMFCMIN, mfu)
        lu = zmfulk * zfac
        qu = zmfuqk * zfac
        tu = jnp.minimum(ZTU_MAX, jnp.maximum(ZTU_MIN, (zmfusk * zfac - kget(pgeoh, jk)) * zorcpd))
        lrain = kget(s["plrain"], jk + 1) * jnp.maximum(0.0, mfu1 - zdmfde) * zfac
        s["pdmfen"] = kset(s["pdmfen"], jk, zdmfen - zdmfde, fl)
        s["pmfu"] = kset(s["pmfu"], jk, mfu, fl)
        s["plude"] = kset(s["plude"], jk, lude, fl)
        s["plu"] = kset(s["plu"], jk, lu, fl)
        s["pqu"] = kset(s["pqu"], jk, qu, fl)
        s["ptu"] = kset(s["ptu"], jk, tu, fl)
        s["plrain"] = kset(s["plrain"], jk, lrain, fl)
        zqold = jnp.where(fl, qu, 0.0)
        zluold = jnp.where(fl, lu, s["zluold"])
        # below departure level, :526-534
        bel = llo3 & (jk > kdpl)
        s["ptu"] = kset(s["ptu"], jk, kget(s["ptenh"], jk), bel)
        s["pqu"] = kset(s["pqu"], jk, kget(s["pqenh"], jk), bel)
        s["plu"] = kset(s["plu"], jk, 0.0, bel)
        zluold = jnp.where(bel, 0.0, zluold)
        # CUADJTQ KCALL 6 (LSCVLIQ), :540-548
        tu0 = kget(s["ptu"], jk)
        qu0 = kget(s["pqu"], jk)
        tu, qu = cuadjtq(th, 6, zph, tu0, qu0, fl, s["lscv"])
        s["ptu"] = kset(s["ptu"], jk, tu, fl)
        s["pqu"] = kset(s["pqu"], jk, qu, fl)
        cnd = fl & (qu != zqold)
        # glaciation, :571-587
        glac = kget(s["plu"], jk) * ((1.0 - foealfcu(th, tu))
                                    - (1.0 - foealfcu(th, kget(s["ptu"], jk + 1))))
        glac = jnp.where(s["lscv"], 0.0, glac)
        zfacg = foealfcu(th, kget(pten, jk))
        glac = glac + zfacg * kget(s["pdmfup"], jk + 1) / jnp.maximum(RMFCMIN, mfu1) \
            * _fsign_half(rtt - kget(pten, jk)) * 1.0
        tu = tu + th.RALFDCP * glac
        s["plglac"] = kset(s["plglac"], jk, glac, cnd)
        s["ptu"] = kset(s["ptu"], jk, tu, cnd)
        # buoyancy, cloud top, detrainment, :594-664
        lu = kget(s["plu"], jk) + zqold - qu
        klab1 = kget(s["klab"], jk + 1)
        zbc = tu * (1.0 + retv * qu - kget(s["plu"], jk + 1) - kget(s["plrain"], jk + 1))
        zbe = kget(s["ptenh"], jk) * (1.0 + retv * kget(s["pqenh"], jk))
        zbuo = zbc - zbe
        s["klab"] = kset(s["klab"], jk, 2, cnd)
        s["plu"] = kset(s["plu"], jk, lu, cnd)
        s["zbuo"] = kset(s["zbuo"], jk, zbuo, cnd)
        # mid-level initiation
        mi = cnd & (s["ktype"] == 3) & (klab1 == 1)
        mok = mi & (zbuo > ZBUO_MID)
        mno = mi & ~mok
        s["ldcum"] = jnp.where(mok, True, s["ldcum"])
        s["kctop"] = jnp.where(mok, jk, s["kctop"])
        s["pkineu"] = kset(s["pkineu"], jk, 0.5, mok)
        s["klab"] = kset(s["klab"], jk, 0, mno)
        s["pmfu"] = kset(s["pmfu"], jk, 0.0, mno)
        s["pmfu"] = kset(s["pmfu"], jk + 1, 0.0, mno)
        s["plude"] = kset(s["plude"], jk, 0.0, mno)
        s["plu"] = kset(s["plu"], jk, 0.0, mno)
        # in-cloud (KLAB(JK+1) == 2)
        ic = cnd & (klab1 == 2)
        reint = ic & (zbuo < ZBUO_REINT)
        teh = 0.5 * (kget(pten, jk) + kget(pten, jk - 1))
        qeh = 0.5 * (kget(pqen, jk) + kget(pqen, jk - 1))
        s["ptenh"] = kset(s["ptenh"], jk, teh, reint)
        s["pqenh"] = kset(s["pqenh"], jk, qeh, reint)
        zbuo = jnp.where(reint, zbc - teh * (1.0 + retv * qeh), zbuo)
        s["zbuo"] = kset(s["zbuo"], jk, zbuo, ic)
        tvh = kget(s["ptenh"], jk) * (1.0 + retv * kget(s["pqenh"], jk))
        zbuoc = 0.5 * (zbuo + kget(s["zbuo"], jk + 1)) / tvh
        zdkbuo = (kget(pgeoh, jk) - kget(pgeoh, jk + 1)) * ZFACBUO * zbuoc
        mfden = jnp.maximum(RMFCMIN, mfu1)
        zdken = jnp.minimum(1.0, (1.0 + Z_CWDRAG) * jnp.where(zdmfen > 0.0, zdmfen, zdmfde) / mfden)
        ke = jnp.maximum(ZKE_MIN, (kget(s["pkineu"], jk + 1) * (1.0 - zdken) + zdkbuo)
                         / (1.0 + zdken))
        neg = zbuoc < 0.0
        zkedke = ke / jnp.maximum(ZKEDKE_FLOOR, kget(s["pkineu"], jk + 1))
        zkedke = jnp.maximum(0.0, jnp.minimum(1.0, zkedke))
        zmfun = (ZDETR_RH - rh) * ssqrt(zkedke)
        de_n = jnp.maximum(zdmfde, mfu1 * (1.0 - zmfun))
        zdmfde_ic = jnp.where(neg, de_n, zdmfde)
        lude_ic = jnp.where(neg, kget(s["plu"], jk + 1) * de_n, kget(s["plude"], jk))
        mfu_ic = jnp.where(neg, mfu1 + zdmfen - de_n, kget(s["pmfu"], jk))
        # organized entrainment for the next level, :617-625
        org = zbuo > ZBUO_ORG
        rhm1 = jnp.minimum(1.0, kget(pqen, jk - 1) / kget(pqsen, jk - 1))
        zoe = (entrorg * (ZOENTR_RH - (rhm1 - ENTR_RH))
               * (kget(pgeoh, jk - 1) - kget(pgeoh, jk)) * zrg
               * jnp.minimum(1.0, kget(pqsen, jk) / kget(pqsen, s["kcbot"])) ** 3)
        zoe = jnp.minimum(ZOENTR_MAX, zoe) * mfu_ic
        zoentr = jnp.where(ic, jnp.where(org, zoe, 0.0), zoentr)
        # below departure: erase
        bd = jk > kdpl
        mfu_ic = jnp.where(bd, mfu1, mfu_ic)
        ke = jnp.where(bd, 0.5, ke)
        lapse = (kget(pten, jk - 1) - kget(pten, jk)) / (zrg * (kget(pgeo, jk - 1) - kget(pgeo, jk)))
        top = (ke > 0.0) & (mfu_ic > 0.0) & ((zbuo > ZBUO_TOP) | (lapse < ZLAPSE_TOP))
        ktop = ic & top
        kno = ic & ~top
        s["kctop"] = jnp.where(ktop, jk, s["kctop"])
        llo1 = ktop
        zdmfde_ic = jnp.where(kno, mfu1, zdmfde_ic)
        lude_ic = jnp.where(kno, kget(s["plu"], jk + 1) * mfu1, lude_ic)
        s["klab"] = kset(s["klab"], jk, 0, kno)
        s["pmfu"] = kset(s["pmfu"], jk, jnp.where(top, mfu_ic, 0.0), ic)
        s["pkineu"] = kset(s["pkineu"], jk, jnp.where(top, ke, 0.0), ic)
        s["plude"] = kset(s["plude"], jk, lude_ic, ic)
        s["pmfude_rate"] = kset(s["pmfude_rate"], jk, zdmfde_ic, ic & (mfu1 > 0.0))
        zdmfde = jnp.where(ic, zdmfde_ic, zdmfde)
        # no condensation: plume stops (KTYPE <= 2) / mid-level reset
        nc = fl & ~cnd
        nc12 = nc & (s["ktype"] <= 2)
        s["klab"] = kset(s["klab"], jk, 0, nc12)
        s["pmfu"] = kset(s["pmfu"], jk, 0.0, nc12)
        s["pkineu"] = kset(s["pkineu"], jk, 0.0, nc12)
        s["plude"] = kset(s["plude"], jk, kget(s["plu"], jk + 1) * mfu1, nc12)
        s["pmfude_rate"] = kset(s["pmfude_rate"], jk, mfu1, nc12)
        zdmfde = jnp.where(nc12, mfu1, zdmfde)
        nc3 = nc & (s["ktype"] == 3)
        s["pmfu"] = kset(s["pmfu"], jk + 1, 0.0, nc3 & (klab1 == 1))
        s["pmfu"] = kset(s["pmfu"], jk, 0.0, nc3)
        # precipitation by analytic integration, :641-712
        luk = kget(s["plu"], jk)
        tuk = kget(s["ptu"], jk)
        pr = llo3 & llo1 & (luk > zdnoprc)
        zwu = jnp.minimum(ZWU_MAX, jnp.sqrt(2.0 * jnp.maximum(ZKE_PRC_MIN, kget(s["pkineu"], jk + 1))))
        zzco = 1.0 + ZZCO_LIQ * foealfcu(th, tuk)
        zprcon = zprcdgw / (ZPRC_W * zwu) * zzco
        zdt = jnp.minimum(th.RTBERCU - th.RTICECU, jnp.maximum(th.RTBERCU - tuk, 0.0))
        zcbf = 1.0 + Z_CPRC2 * ssqrt(zdt)
        zzco = zprcon * zcbf
        zlcrit = zdnoprc / zcbf
        zdfi = kget(pgeoh, jk) - kget(pgeoh, jk + 1)
        zc = luk - zluold
        zd = zzco * (1.0 - jnp.exp(-(luk / zlcrit) ** 2)) * zdfi
        zd = jnp.where(pr, zd, 1.0)
        zint = jnp.exp(-zd)
        zlnew = zluold * zint + zc / zd * (1.0 - zint)
        zlnew = jnp.maximum(0.0, jnp.minimum(luk, zlnew))
        zlnew = jnp.minimum(Z_CLDMAX, zlnew)
        zprecip = jnp.where(pr, jnp.maximum(0.0, zluold + zc - zlnew), 0.0)
        s["pdmfup"] = kset(s["pdmfup"], jk, zprecip * kget(s["pmfu"], jk), pr)
        s["plrain"] = kset(s["plrain"], jk, kget(s["plrain"], jk) + zprecip, pr)
        s["plu"] = kset(s["plu"], jk, zlnew, pr)
        # rain fall-out, :740-761
        lr = kget(s["plrain"], jk)
        rf = llo3 & llo1 & (lr > 0.0)
        zvw = ZVW_COEF * spow(lr, ZVW_EXP)
        zvi = Z_CWIFRAC * zvw
        zalfaw = jnp.where(s["lscv"], 1.0, foealfcu(th, tuk))
        zvv = zalfaw * zvw + (1.0 - zalfaw) * zvi
        zrold = lr - zprecip
        zwu = jnp.minimum(ZWU_MAX, jnp.sqrt(2.0 * jnp.maximum(ZKE_RAIN_MIN, kget(s["pkineu"], jk))))
        zd = jnp.where(rf, zvv / zwu, 1.0)
        zint = jnp.exp(-zd)
        zrnew = zrold * zint + zprecip / zd * (1.0 - zint)
        zrnew = jnp.maximum(0.0, jnp.minimum(lr, zrnew))
        s["plrain"] = kset(s["plrain"], jk, zrnew, rf)
        # fluxes, :764-773
        mfuk = kget(s["pmfu"], jk)
        luk = kget(s["plu"], jk)
        tuk = kget(s["ptu"], jk)
        s["pmful"] = kset(s["pmful"], jk, luk * mfuk, fl)
        s["pmfus"] = kset(s["pmfus"], jk, (rcpd * tuk + kget(pgeoh, jk)) * mfuk, fl)
        s["pmfuq"] = kset(s["pmfuq"], jk, kget(s["pqu"], jk) * mfuk, fl)
        za = jnp.where(s["lscv"], 1.0, foealfcu(th, tuk))
        lde = kget(s["plude"], jk)
        s["pludeli1"] = kset(s["pludeli1"], jk, za * lde, fl)
        s["pludeli2"] = kset(s["pludeli2"], jk, (1.0 - za) * lde, fl)
        s["zoentr"] = jnp.where(llo3, zoentr, s["zoentr"])
        s["zluold"] = jnp.where(llo3, zluold, s["zluold"])
        s["zdmfen"] = zdmfen
        s["zdmfde"] = zdmfde
        return s

    st = jax.lax.fori_loop(0, K - 3, level, st)
    # 5. final, :859-866
    ldc = jnp.where(st["kctop"] == -1, False, st["ldcum"])
    kcbot = jnp.maximum(st["kcbot"], st["kctop"])
    pw = jnp.maximum(ZWMEAN_MIN, st["pwmean"] / jnp.maximum(ZDPMEAN_MIN, st["zdpmean"]))
    pwmean = jnp.where(ldc, jnp.sqrt(2.0 * pw), st["pwmean"])
    out = dict(st)
    out.update(ldcum=ldc, kcbot=kcbot, pwmean=pwmean)
    return out
