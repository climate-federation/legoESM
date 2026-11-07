"""Tiedtke (1989) mass-flux convection as WRF v4.8.0 runs it.

:func:`tiedtke_wrf` is a FAITHFUL port of WRF v4.8.0
``phys/module_cu_tiedtke.F`` (SHA 06d4240ae989cc3e50af412bb472df3d9048783c;
Tiedtke 1989 + Nordeng CAPE closure + organized entrainment/detrainment +
cumulus friction), water-only with legoESM constants.  It matches the running
Fortran to max relative error <= 1e-10 in fp64
(tests/atmosphere/test_tiedtke_wrf_oracle.py).

:func:`tiedtke_convection` is the host entry every ``convection: tiedtke``
lane calls: it feeds the port the oracle's own inputs and maps its outputs
onto :class:`ConvectionOutput`.  Two departures from the oracle, both
user-approved: the land mask enters as a smooth weight of the land fraction
(exactly the oracle at land fraction 0 or 1), and the port's internal hard
switches are kept as in WRF (explicit exception to the smooth-switch rule;
smoothing is a separate change).

References
----------
* Tiedtke, M. (1989). A comprehensive mass flux scheme for cumulus
  parameterization in large-scale models.  *Mon. Wea. Rev.*, 117, 1779-1800.
* Nordeng, T. E. (1994). Extended versions of the convective parametrization
  scheme at ECMWF.  ECMWF Tech. Memo. 206.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.smooth import sigmoid_switch
from legoesm.thermo import saturation_vapor_pressure
from legoesm.atmosphere.physics.convection.output import ConvectionOutput

__all__ = ("tiedtke_convection", "tiedtke_wrf", "TiedtkeWRFOutput")

__physics_contract__ = {
    "summary": (
        "WRF v4.8.0 Tiedtke mass-flux convection (cu_tiedtke/tiecnv), faithful "
        "port: deep/shallow/midlevel types, Nordeng CAPE closure, downdrafts, "
        "cumulus friction; surface convective rain and snow."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg (specific humidity)", "p_full": "Pa",
        "p_half": "Pa", "u": "m/s", "v": "m/s", "dt": "s",
        "geopotential": "m^2/s^2 (full levels)", "omega": "Pa/s",
        "dq_v_dt_dyn": "kg/kg/s (dynamics q_v tendency, WRF qvften)",
        "dq_v_dt_pbl": "kg/kg/s (boundary-layer q_v tendency, WRF qvpblten)",
        "surface_moisture_flux": "kg/m^2/s (positive up, WRF qfx)",
        "land_frac": "1 (None = ocean)", "sigma": "1 (full-level p/p_s)",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "dq_c_conv_dt": "kg/kg/s (detrained liquid)",
        "dq_i_conv_dt": "kg/kg/s (detrained ice)",
        "du_dt_conv": "m/s^2", "dv_dt_conv": "m/s^2",
        "dq_r_conv_dt": "kg/kg/s (surface precip placed in the lowest layer)",
        "precip_frozen_sfc": "kg/m^2/s (surface snow)",
        "conv_prog_profile_new": "kg/m^2/s (updraft mass flux, diagnostic)",
    },
    "sign_convention": (
        "levels top-down, surface at [:, -1]; dq_r_conv_dt column-integrates "
        "(sum dq_r dp / g) to the oracle's surface precipitation, which leaves "
        "the column (surface route)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "WRF v4.8.0 phys/module_cu_tiedtke.F; Tiedtke (1989); Nordeng (1994)",
    "idealized_test": (
        "tests/atmosphere/test_tiedtke_wrf_oracle.py (oracle match); "
        "tests/unit/test_tiedtke.py (host mapping)."
    ),
}

# Land weight: sigmoid of (land fraction - 0.5) at a 0.01 land-fraction width
# (user-approved smooth form of WRF's land mask ``lndj``).
_LAND_SWITCH_SHARPNESS = 100.0


def tiedtke_convection(T, q_v, p_full, p_half, u, v, dt, *, geopotential,
                       omega, dq_v_dt_dyn, dq_v_dt_pbl, surface_moisture_flux,
                       land_frac, sigma):
    """Run :func:`tiedtke_wrf` and map its outputs onto ConvectionOutput.

    Profiles (ncol, nlev) top-down.  Returns ``(ConvectionOutput, mfu)``;
    ``mfu`` [kg/m^2/s] is carried as ``conv_prog_profile`` (diagnostic, the
    scheme keeps no memory).  The oracle's surface precipitation is emitted as
    ``dq_r_conv_dt`` in the lowest layer so its column integral is exactly that
    precipitation; hosts route it to the surface (trait ``rain_is_net_flux``).
    WRF produces no CAPE diagnostic: ``cape`` is zeros.
    """
    if land_frac is None:
        land_w = jnp.zeros(T.shape[:1], T.dtype)
    else:
        land_w = sigmoid_switch(jnp.asarray(land_frac, T.dtype).reshape(-1) - 0.5,
                                sharpness=_LAND_SWITCH_SHARPNESS)
    o = tiedtke_wrf(T, q_v, u, v, p_full, p_half, geopotential, omega,
                    dq_v_dt_dyn, dq_v_dt_pbl, surface_moisture_flux, land_w,
                    sigma, dt)
    dp_bot = p_half[:, -1] - p_half[:, -2]
    dq_r = jnp.zeros_like(T).at[:, -1].set(constants.g * o.precip / dp_bot)
    out = ConvectionOutput(
        dT_dt=o.dT_dt, dq_v_dt=o.dq_v_dt, dq_c_conv_dt=o.dq_c_dt,
        cape=jnp.zeros_like(o.precip),
        convective_mask=o.ldcum.astype(T.dtype),
        du_dt_conv=o.du_dt, dv_dt_conv=o.dv_dt, dq_r_conv_dt=dq_r,
        precip_frozen_sfc=o.precip_snow, dq_i_conv_dt=o.dq_i_dt,
    )
    return out, o.mfu


# =============================================================================
# FAITHFUL PORT: WRF v4.8.0 ``module_cu_tiedtke.F`` (tiecnv + cumastr_new chain)
# =============================================================================
# Oracle: https://github.com/wrf-model/WRF tag v4.8.0,
# SHA 06d4240ae989cc3e50af412bb472df3d9048783c, phys/module_cu_tiedtke.F
# (md5 1d9127ddd05b9e314ae95b847043e155).  Every routine of the call chain is
# ported line by line: tiecnv, cumastr_new, cuini, cubase, cuasc_new,
# cuentr_new, cubasmc, cuadjtq, cudlfs, cuddraf, cuflx, cudtdq, cududv,
# tlucua/b/c.  Hard switches are ported as hard switches (``jnp.where``),
# so the port is piecewise differentiable and reproduces the oracle to
# round-off (fp64; tests/atmosphere/test_tiedtke_wrf_oracle.py).
#
# Thermodynamic constants are legoESM's (``legoesm.constants``) and the
# saturation curve is ``legoesm.thermo.saturation_vapor_pressure`` -- the
# oracle was RUN with exactly these (audit variant "V1": stub
# module_model_constants with the legoESM values, Tetens coefficients set to
# the legoesm.thermo water curve on both the liquid and the ice branch).
# legoESM convection is WATER ONLY, so the oracle's L_s := L_v and L_f := 0:
# every ``alf`` (fusion) term of the oracle is identically zero and is not
# carried here, the ``als`` branches equal the ``alv`` ones, and the
# snow-melt rate ``cpd/(alf*g*dt)`` is infinite (all snow melts at the first
# level warmer than tmelt+2 K) -- which is what the oracle computes with
# alf = 0.  The ICE phase of stock WRF (L_s, L_f, ice Tetens branch) is NOT
# ported.
#
# Column semantics: one oracle call per column (WRF tile of klon = 1), as the
# audit ran it.  WRF couples the columns of a tile in exactly three places
# (the ``is.eq.0`` cycles of the level loops, ``icum.eq.0`` early return and
# ``itop = min(kctop)`` in cuflx); with klon = 1 each is per column.
#
# Indexing: arrays are padded with one leading element so that Fortran index
# k (k = 1 top ... klev surface) is Python index k; ``paph`` likewise
# (k = 1 ... klev+1).  Index 0 is only touched where the Fortran would read or
# write outside its arrays (ktopm2 = 1); the pad holds the level-1 value.

# --- WRF module_cu_tiedtke.F parameters (lines 31-108; tunable in WRF only
#     by editing the source, so module constants here) ---
_WRF_ENTRPEN = 1.0e-4      # entrainment rate, penetrative convection [1/m]
_WRF_ENTRSCV = 1.2e-3      # entrainment rate, shallow convection [1/m]
_WRF_ENTRMID = 1.0e-4      # entrainment rate, midlevel convection [1/m]
_WRF_ENTRDD = 2.0e-4       # entrainment rate, downdrafts [1/m]
_WRF_CMFCTOP = 0.30        # relative mass flux above the non-buoyancy level
_WRF_CMFCMAX = 1.0         # max midlevel cloud-base mass flux [kg/m2/s]
_WRF_CMFCMIN = 1.0e-10     # min mass flux (safety) [kg/m2/s]
_WRF_CMFDEPS = 0.30        # downdraft fraction of cloud-base flux at the LFS
_WRF_CPRCON_G = 1.1e-3     # cprcon * g: cloud-water -> rain conversion [1/m]
_WRF_ZDNOPRC = 1.5e4       # cloud depth below which no precipitation [Pa]
_WRF_RHC = 0.80            # RH above which detrained condensate is kept
_WRF_RHM = 1.0             # RH of full condensate retention
_WRF_ZBUO0 = 0.50          # cloud-base virtual-temperature tolerance [K]
_WRF_CRIRH = 0.70          # cloud-layer RH required by the CAPE closure
_WRF_FDBK = 1.0            # detrained-condensate feedback factor
_WRF_ZTAU = 2400.0         # CAPE-closure adjustment time [s]
_WRF_CEVAPCU1_G = 1.93e-6 * 261.0 * 0.5   # cevapcu1 * g (Kessler evaporation)
_WRF_CEVAPCU2 = 1.0e3 / (38.3 * 0.293)    # cevapcu2
_WRF_HGFR = 233.15         # homogeneous-freezing temperature [K]
# --- hard-coded literals of the routines (module_cu_tiedtke.F line cited) ---
_WRF_VIRT = 0.608          # virtual-temperature factor (lines 886, 1026, 1639, 1753)
_WRF_TYPE1_EVAP_FACTOR = 1.1   # deep if MC > 1.1 * evaporation (line 846)
_WRF_LAND_ENTR_FACTOR = 1.1    # land entrainment enhancement (lines 945, 969)
_WRF_MFUB_FALLBACK = 0.01  # cloud-base mass flux when the closure fails (863, 1041)
_WRF_DQMIN_FRAC = 0.01     # zdqmin = max(0.01*qenh, 1e-10) (lines 858, 1065)
_WRF_DQMIN_FLOOR = 1.0e-10
_WRF_MFUB_FLOOR = 1.0e-10  # zfac = zmfub1/max(zmfub, 1e-10) (line 1083)
_WRF_ZBI_DEPTH = 25.0      # zbi = 1/(25 g) (line 916)
_WRF_KCTOP0_P = 4.0e4      # cloud-top estimate for non-convecting columns [Pa] (1610)
_WRF_ORG_MAX = 1.0e-3      # organized entrainment / detrainment cap [1/m] (1646, 2944)
_WRF_DETR_MAX = 0.75       # max detrained fraction of the mass flux (1720, 1743, 1834)
_WRF_TU_MIN = 100.0        # updraft / downdraft temperature clip [K] (1769, 2154)
_WRF_TU_MAX = 400.0
_WRF_MFU_TOP_FRAC = 0.01   # buoyant level needs mfu > 0.01 mfub (line 1789)
_WRF_PI = 3.1415           # pi as written in cuentr_new (lines 2937-2939)
_WRF_TINY = 1.0e-20        # tiny-flux tests (lines 1820, 2341, 2361)
_WRF_MID_RH = 0.80         # midlevel trigger q > 0.8 qsat (line 2662)
_WRF_MID_NLEV = 13         # midlevel convection only in the lowest 13 levels (1663)
_WRF_KTOP0_MIN = 12        # CAPE-closure RH layer top index (line 1019)
_WRF_LFS_RAIN = 10.0       # LFS needs rain > 10 * downdraft evaporation (2021)
_WRF_SHALLOW_TOL = 0.2     # shallow closure update tolerance (line 1074)
_WRF_CUCOV = 0.05          # convective precipitation area fraction (line 2245)
_WRF_EVAP_RH = 0.8         # sub-cloud evaporation limit 0.8 qsat (line 2347)
_WRF_FLIQ_A = 0.0059       # liquid fraction of detrained condensate (line 625)
_WRF_FLIQ_B = 0.9941
_WRF_FLIQ_C = 0.003102
_WRF_QSAT_CAP = 0.5        # qsat cap in tiecnv / cuadjtq (lines 589, 2739)
_WRF_MELT_DT = 2.0         # snow melts above tmelt + 2 K (line 2246)


class TiedtkeWRFOutput(NamedTuple):
    """Output of :func:`tiedtke_wrf` (WRF v4.8.0 cu_tiedtke), shape (ncol, nlev)
    unless noted; levels top-down, surface at [:, -1].

    dT_dt [K/s], dq_v_dt [kg/kg/s] (specific humidity; the oracle's
    ``pqte - zqq``), du_dt / dv_dt [m/s2] (cumulus friction, cududv),
    dq_c_dt / dq_i_dt [kg/kg/s] (detrained condensate ``pcte`` split by the
    tiecnv liquid fraction), detrained_condensate [kg/kg/s] (``pcte``),
    precip [kg/m2/s, (ncol,)] (``prsfc + pssfc``, surface convective
    precipitation), mfu / mfd [kg/m2/s] (final up/downdraft mass flux on the
    TOP interface of each level), mfub [kg/m2/s, (ncol,)] (final cloud-base
    flux, 0 where not convecting), ldcum [bool, (ncol,)], ktype [int,
    (ncol,); 1 deep, 2 shallow, 3 midlevel, 0 none; where the first ascent
    finds no cloud the oracle returns early and keeps that ascent's type],
    kcbot / kctop [int, (ncol,); 0-based level index, top = 0; meaningful
    only where ldcum], precip_snow [kg/m2/s, (ncol,)] (``pssfc``, the frozen
    part of ``precip``).
    """
    dT_dt: jax.Array
    dq_v_dt: jax.Array
    du_dt: jax.Array
    dv_dt: jax.Array
    dq_c_dt: jax.Array
    dq_i_dt: jax.Array
    detrained_condensate: jax.Array
    precip: jax.Array
    mfu: jax.Array
    mfd: jax.Array
    mfub: jax.Array
    ldcum: jax.Array
    ktype: jax.Array
    kcbot: jax.Array
    kctop: jax.Array
    precip_snow: jax.Array


def _wrf_dlnes_dT(T):
    """d ln e_s / dT of ``legoesm.thermo.saturation_vapor_pressure`` (the
    oracle's tlucub / c5les*(T-c4les)**-2 for the legoESM Tetens curve)."""
    es, des = jax.jvp(saturation_vapor_pressure, (T,), (jnp.ones_like(T),))
    return des / es


def _wrf_qsat(T, p):
    """tiecnv / cuadjtq saturation specific humidity: tlucua(T)/p, capped at
    0.5, divided by (1 - vtmpc1*qs).  Returns (qs, 1/(1-vtmpc1*qs_capped))."""
    vtmpc1 = constants.R_v / constants.R_d - 1.0
    qs = jnp.minimum(_WRF_QSAT_CAP,
                     constants.epsilon * saturation_vapor_pressure(T) * (1.0 / p))
    cor = 1.0 / (1.0 - vtmpc1 * qs)
    return qs * cor, cor


def _wrf_cuadjtq(T, q, p, flag, kcall):
    """cuadjtq (lines 2694-2849): two-pass saturation adjustment.

    kcall 0: unconditional; 1: condensation only; 2: evaporation only.  The
    second pass runs where the first changed the state (kcall 1/2) or always
    (kcall 0), exactly as the Fortran ``isum``/``zcond.ne.0`` logic per column.
    """
    ldcp = constants.L_v / constants.c_pd

    def cond(T, q):
        qs, cor = _wrf_qsat(T, p)
        return (q - qs) / (1.0 + qs * cor * (_wrf_dlnes_dT(T) * ldcp))

    c = cond(T, q)
    if kcall == 1:
        c = jnp.maximum(c, 0.0)
    elif kcall == 2:
        c = jnp.minimum(c, 0.0)
    c = jnp.where(flag, c, 0.0)
    T1 = T + ldcp * c
    q1 = q - c
    second = flag if kcall == 0 else flag & (c != 0.0)
    c1 = cond(T1, q1)
    return (jnp.where(second, T1 + ldcp * c1, T1),
            jnp.where(second, q1 - c1, q1))


def _sdiv(a, b, ok):
    """a / b where ``ok``; the masked lanes divide by 1 so no inf/NaN reaches
    the (discarded) branch -- forward identical, gradient NaN-free."""
    return a / jnp.where(ok, b, 1.0)


def _wrf_cuasc(c, s):
    """cuasc_new (lines 1468-1911) incl. cuentr_new and cubasmc; one column.

    ``c``: dict of fixed column fields; ``s``: dict of the state cuasc reads
    and writes.  Returns the updated state dict (Fortran inout arguments).
    """
    K = c["K"]
    g, cpd, rd, rv, alv = (constants.g, constants.c_pd, constants.R_d,
                           constants.R_v, constants.L_v)
    rcpd = 1.0 / cpd
    zrg = 1.0 / g
    vtmpc1 = rv / rd - 1.0
    zcons2 = 1.0 / (g * c["dt"])
    paph, pgeo, pgeoh = c["paph"], c["pgeo"], c["pgeoh"]
    ptenh, pqenh, pqsenh = c["ptenh"], c["pqenh"], c["pqsenh"]
    pten, pqen, pqsen, pap = c["pten"], c["pqen"], c["pqsen"], c["pap"]
    puen, pven, pverv = c["puen"], c["pven"], c["pverv"]
    lev = jnp.arange(K + 1)
    zero = jnp.zeros_like(pten)

    ldcum, ktype = s["ldcum"], s["ktype"]
    # 2. defaults
    ktype = jnp.where(ldcum, ktype, 0)
    klab = jnp.where((~ldcum) | (ktype == 3), 0, s["klab"])
    hi = (paph[: K + 1] < _WRF_KCTOP0_P) & (lev >= 1)
    kctop0 = jnp.where((~ldcum) & jnp.any(hi),
                       jnp.max(jnp.where(hi, lev, 0)), s["kctop0"])
    # 3. lifting level
    kctop = jnp.int32(K - 1)
    kcbot = jnp.where(ldcum, s["kcbot"], K - 1)
    pmfub = jnp.where(ldcum, s["pmfub"], 0.0)
    pqu = s["pqu"].at[K].set(jnp.where(ldcum, s["pqu"][K], 0.0))
    ptu, puu, pvu = s["ptu"], s["puu"], s["pvu"]
    pmfu = zero.at[K].set(pmfub)
    pmfus = zero.at[K].set(pmfub * (cpd * ptu[K] + pgeoh[K]))
    pmfuq = zero.at[K].set(pmfub * pqu[K])
    zmfuu = pmfub * puu[K]
    zmfuv = pmfub * pvu[K]
    # 3.1 organized entrainment at cloud base
    ikb = kcbot
    zbuoy = g * ((ptu[ikb] - ptenh[ikb]) / ptenh[ikb]
                 + _WRF_VIRT * (pqu[ikb] - pqenh[ikb]))
    ok = (ktype == 1) & (zbuoy > 0.0)
    zdz = (pgeo[ikb - 1] - pgeo[ikb]) * zrg
    zdrodz = -jnp.log(pten[ikb - 1] / pten[ikb]) / zdz - g / (rd * ptenh[ikb])
    oe = jnp.clip(zbuoy * 0.5 / (1.0 + zbuoy * zdz) + zdrodz, None, _WRF_ORG_MAX)
    zoentr = zero.at[ikb - 1].set(jnp.where(ok, jnp.maximum(oe, 0.0), 0.0))
    zbuoy = jnp.where(ktype == 1, zbuoy, 0.0)
    zodetr = zero
    st = dict(ktype=ktype, klab=klab, kctop0=kctop0, kctop=kctop, kcbot=kcbot,
              pmfub=pmfub, pentr=s["pentr"], ldcum=jnp.bool_(False),
              ptu=ptu, pqu=pqu, plu=zero, puu=puu, pvu=pvu, pmfu=pmfu,
              pmfus=pmfus, pmfuq=pmfuq, pmful=zero, plude=zero, pdmfup=zero,
              zoentr=zoentr, zodetr=zodetr, zbuoy=zbuoy, zmfuu=zmfuu,
              zmfuv=zmfuv)
    khmin, phhatt, phcbase = c["khmin"], c["zhhatt"], c["zhcbase"]

    def level(i, st):
        jk = K - 1 - i
        st = dict(st)
        # cubasmc (lines 2613-2688), kk = jk
        kk = jk
        mid = ((kk < K - 1) & (kk > K - _WRF_MID_NLEV) & (~st["ldcum"])
               & (st["klab"][kk + 1] == 0) & (pqen[kk] > _WRF_MID_RH * pqsen[kk]))
        zzzmb = jnp.minimum(jnp.maximum(_WRF_CMFCMIN, -pverv[kk] / g), _WRF_CMFCMAX)
        tu_m = (cpd * pten[kk] + pgeo[kk] - pgeoh[kk + 1]) * rcpd

        def setk(name, val, at=kk + 1, m=mid):
            st[name] = st[name].at[at].set(jnp.where(m, val, st[name][at]))

        setk("ptu", tu_m)
        setk("pqu", pqen[kk])
        setk("plu", 0.0)
        st["pmfub"] = jnp.where(mid, zzzmb, st["pmfub"])
        setk("pmfu", st["pmfub"])
        setk("pmfus", st["pmfub"] * (cpd * st["ptu"][kk + 1] + pgeoh[kk + 1]))
        setk("pmfuq", st["pmfub"] * st["pqu"][kk + 1])
        setk("pmful", 0.0)
        setk("pdmfup", 0.0)
        st["kcbot"] = jnp.where(mid, kk, st["kcbot"])
        setk("klab", 1)
        st["ktype"] = jnp.where(mid, 3, st["ktype"])
        st["pentr"] = jnp.where(mid, _WRF_ENTRMID, st["pentr"])
        setk("puu", puen[kk])
        setk("pvu", pven[kk])
        st["zmfuu"] = jnp.where(mid, st["pmfub"] * st["puu"][kk + 1], st["zmfuu"])
        st["zmfuv"] = jnp.where(mid, st["pmfub"] * st["pvu"][kk + 1], st["zmfuv"])
        # per-level preamble
        klab_up = st["klab"][jk + 1]
        st["klab"] = st["klab"].at[jk].set(jnp.where(klab_up == 0, 0, st["klab"][jk]))
        loflag = klab_up > 0
        cap3 = (st["ktype"] == 3) & (jk == st["kcbot"])
        zmfmax = (paph[jk] - paph[jk - 1]) * zcons2
        capm = cap3 & (st["pmfub"] > zmfmax)
        zfac = _sdiv(zmfmax, st["pmfub"], capm)
        for nm in ("pmfu", "pmfus", "pmfuq"):
            setk(nm, st[nm][jk + 1] * zfac, at=jk + 1, m=capm)
        st["zmfuu"] = jnp.where(capm, st["zmfuu"] * zfac, st["zmfuu"])
        st["zmfuv"] = jnp.where(capm, st["zmfuv"] * zfac, st["zmfuv"])
        st["pmfub"] = jnp.where(capm, zmfmax, st["pmfub"])
        # -- everything below runs only where loflag (``is.eq.0`` cycle) --
        ldc, kty, kcb, kt0 = st["ldcum"], st["ktype"], st["kcbot"], st["kctop0"]
        pmfu, ptu, pqu, plu = st["pmfu"], st["ptu"], st["pqu"], st["plu"]
        # cuentr_new (lines 2855-2950), kk = jk
        zpbase = paph[kcb]
        zrrho = (rd * ptenh[jk + 1]) / paph[jk + 1]
        zdprho = (paph[jk + 1] - paph[jk]) * zrg
        zpmid = 0.5 * (zpbase + paph[kt0])
        zentr = st["pentr"] * pmfu[jk + 1] * zdprho * zrrho
        llo1 = (jk < kcb) & ldc
        zdmfde = jnp.where(llo1, zentr, 0.0)
        llo2 = llo1 & (kty == 2) & (((zpbase - paph[jk]) < _WRF_ZDNOPRC)
                                    | (paph[jk] > zpmid))
        zdmfen = jnp.where(llo2, zentr, 0.0)
        iklwmin = jnp.maximum(c["klwmin"], kt0 + 2)
        llo2 = llo1 & (kty == 3) & ((jk >= iklwmin) | (pap[jk] > zpmid))
        zdmfen = jnp.where(llo2, zentr, zdmfen)
        llo2 = llo1 & (kty == 1)
        zdmfen = jnp.where(llo2, zentr, zdmfen)
        ikt, ikh = kt0, khmin
        org = llo2 & (jk <= khmin) & (jk >= kt0) & (ikh > ikt)
        zzmzk = -(pgeoh[ikh] - pgeoh[jk]) * zrg
        ztmzk = -(pgeoh[ikh] - pgeoh[ikt]) * zrg
        arg = _WRF_PI * _sdiv(zzmzk, ztmzk, org) * 0.5
        zorgde = jnp.tan(jnp.where(org, arg, 0.0)) * _WRF_PI * 0.5 / jnp.where(org, ztmzk, 1.0)
        zdprho2 = (paph[jk + 1] - paph[jk]) * (zrg * zrrho)
        zodetr_k = jnp.where(org, jnp.minimum(zorgde, _WRF_ORG_MAX) * pmfu[jk + 1] * zdprho2, 0.0)
        # ascent
        blw = jk < kcb
        zmftest = pmfu[jk + 1] + zdmfen - zdmfde
        zmfmax = jnp.minimum(zmftest, (paph[jk] - paph[jk - 1]) * zcons2)
        zdmfen = jnp.where(blw, jnp.maximum(zdmfen - jnp.maximum(zmftest - zmfmax, 0.0), 0.0), zdmfen)
        zdmfde = jnp.minimum(zdmfde, _WRF_DETR_MAX * pmfu[jk + 1])
        mfu_k = pmfu[jk + 1] + zdmfen - zdmfde
        zdp2 = (pgeoh[jk] - pgeoh[jk + 1]) * zrg
        zoentr_k = st["zoentr"][jk] * zdp2 * pmfu[jk + 1]
        zmftest = mfu_k + zoentr_k - zodetr_k
        zmfmax = jnp.minimum(zmftest, (paph[jk] - paph[jk - 1]) * zcons2)
        zoentr_k = jnp.where(blw, jnp.maximum(zoentr_k - jnp.maximum(zmftest - zmfmax, 0.0), 0.0),
                             st["zoentr"][jk])
        lim = (kty == 1) & blw & (jk <= khmin)
        zmse = cpd * ptu[jk + 1] + alv * pqu[jk + 1] + pgeoh[jk + 1]
        znevn = (pgeoh[kt0] - pgeoh[jk + 1]) * (zmse - phhatt[jk + 1]) * zrg
        znevn = jnp.where(znevn <= 0.0, 1.0, znevn)
        zodmax = jnp.maximum(((phcbase - zmse) / znevn) * zdp2 * pmfu[jk + 1], 0.0)
        zodetr_k = jnp.where(lim, jnp.minimum(zodetr_k, zodmax), zodetr_k)
        zodetr_k = jnp.minimum(zodetr_k, _WRF_DETR_MAX * mfu_k)
        mfu_k = mfu_k + zoentr_k - zodetr_k
        sh = cpd * ptenh[jk + 1] + pgeoh[jk + 1]
        zqeen = pqenh[jk + 1] * zdmfen
        zqeen = zqeen + pqenh[jk + 1] * zoentr_k
        zseen = sh * zdmfen
        zseen = zseen + sh * zoentr_k
        zscde = (cpd * ptu[jk + 1] + pgeoh[jk + 1]) * zdmfde
        zga = alv * pqsenh[jk + 1] / (rv * (ptenh[jk + 1] ** 2))
        zdt = ((plu[jk + 1] - _WRF_VIRT * (pqsenh[jk + 1] - pqenh[jk + 1]))
               / (1.0 / ptenh[jk + 1] + _WRF_VIRT * zga))
        zscod = cpd * ptenh[jk + 1] + pgeoh[jk + 1] + cpd * zdt
        zscde = zscde + zodetr_k * zscod
        zqude = pqu[jk + 1] * zdmfde
        zqcod = pqsenh[jk + 1] + zga * zdt
        zqude = zqude + zodetr_k * zqcod
        lude_k = plu[jk + 1] * zdmfde
        lude_k = lude_k + plu[jk + 1] * zodetr_k
        zmfusk = st["pmfus"][jk + 1] + zseen - zscde
        zmfuqk = st["pmfuq"][jk + 1] + zqeen - zqude
        zmfulk = st["pmful"][jk + 1] - lude_k
        rm = 1.0 / jnp.maximum(_WRF_CMFCMIN, mfu_k)
        lu_k = zmfulk * rm
        qu_k = zmfuqk * rm
        tu_k = jnp.clip((zmfusk * rm - pgeoh[jk]) * rcpd, _WRF_TU_MIN, _WRF_TU_MAX)
        zqold = qu_k
        tu_k, qu_k = _wrf_cuadjtq(tu_k, qu_k, paph[jk], loflag, 1)
        cnd = loflag & (qu_k != zqold)
        lu_k = jnp.where(cnd, lu_k + zqold - qu_k, lu_k)
        zbuo = (tu_k * (1.0 + vtmpc1 * qu_k - lu_k)
                - ptenh[jk] * (1.0 + vtmpc1 * pqenh[jk]))
        zbuo = jnp.where(st["klab"][jk + 1] == 1, zbuo + _WRF_ZBUO0, zbuo)
        top = cnd & (zbuo > 0.0) & (mfu_k > _WRF_MFU_TOP_FRAC * st["pmfub"]) & (jk >= kt0)
        zprcon = jnp.where(zpbase - paph[jk] >= _WRF_ZDNOPRC, _WRF_CPRCON_G / g, 0.0)
        zlnew = lu_k / (1.0 + zprcon * (pgeoh[jk] - pgeoh[jk + 1]))
        dmfup_k = jnp.where(top, jnp.maximum(0.0, (lu_k - zlnew) * mfu_k), 0.0)
        lu_k = jnp.where(top, zlnew, lu_k)
        klab_k = jnp.where(cnd, jnp.where(top, 2, 0), st["klab"][jk])
        mfu_k = jnp.where(cnd & ~top, 0.0, mfu_k)
        new_kctop = jnp.where(top, jk, st["kctop"])
        new_ldcum = st["ldcum"] | top
        # lmfdudv (cumulus friction, lines 1813-1846)
        zdmfen_u = zdmfen + zoentr_k
        zdmfde_u = zdmfde + zodetr_k
        deep3 = (kty == 1) | (kty == 3)
        small = zdmfen_u <= _WRF_TINY
        zz = jnp.where(deep3, jnp.where(small, 3.0, 2.0), jnp.where(small, 1.0, 0.0))
        zdmfeu = zdmfen_u + zz * zdmfde_u
        zdmfdu = jnp.minimum(zdmfde_u + zz * zdmfde_u, _WRF_DETR_MAX * pmfu[jk + 1])
        zmfuu = st["zmfuu"] + zdmfeu * puen[jk] - zdmfdu * st["puu"][jk + 1]
        zmfuv = st["zmfuv"] + zdmfeu * pven[jk] - zdmfdu * st["pvu"][jk + 1]
        pos = mfu_k > 0.0
        rmu = _sdiv(1.0, mfu_k, pos)
        uu_k = jnp.where(pos, zmfuu * rmu, st["puu"][jk])
        vu_k = jnp.where(pos, zmfuv * rmu, st["pvu"][jk])
        # organized entrainment for the next level (lines 1851-1864)
        oz = kty == 1
        zbuoyz = jnp.maximum(g * ((tu_k - ptenh[jk]) / ptenh[jk]
                                  + _WRF_VIRT * (qu_k - pqenh[jk]) - lu_k), 0.0)
        zdz = (pgeo[jk - 1] - pgeo[jk]) * zrg
        zdrodz = -jnp.log(pten[jk - 1] / pten[jk]) / zdz - g / (rd * ptenh[jk])
        zbuoy_n = st["zbuoy"] + zbuoyz * zdz
        oe = jnp.maximum(jnp.minimum(zbuoyz * 0.5 / (1.0 + zbuoy_n) + zdrodz, _WRF_ORG_MAX), 0.0)

        m = loflag

        def put(name, at, val):
            st[name] = st[name].at[at].set(jnp.where(m, val, st[name][at]))

        put("pmfu", jk, mfu_k)
        put("plude", jk, lude_k)
        put("plu", jk, lu_k)
        put("pqu", jk, qu_k)
        put("ptu", jk, tu_k)
        put("klab", jk, klab_k)
        put("pdmfup", jk, dmfup_k)
        put("pmful", jk, lu_k * mfu_k)
        put("pmfus", jk, (cpd * tu_k + pgeoh[jk]) * mfu_k)
        put("pmfuq", jk, qu_k * mfu_k)
        put("zoentr", jk, zoentr_k)
        put("zodetr", jk, zodetr_k)
        put("puu", jk, uu_k)
        put("pvu", jk, vu_k)
        m = loflag & oz
        put("zoentr", jk - 1, oe)
        st["zbuoy"] = jnp.where(m, zbuoy_n, st["zbuoy"])
        st["zmfuu"] = jnp.where(loflag, zmfuu, st["zmfuu"])
        st["zmfuv"] = jnp.where(loflag, zmfuv, st["zmfuv"])
        st["kctop"] = jnp.where(loflag, new_kctop, st["kctop"])
        st["ldcum"] = jnp.where(loflag, new_ldcum, st["ldcum"])
        return st

    st = jax.lax.fori_loop(0, K - 2, level, st)
    # 5. fluxes above the non-buoyancy level
    ldcum = st["ldcum"] & (st["kctop"] != K - 1)
    st["kcbot"] = jnp.maximum(st["kcbot"], st["kctop"])
    jk = st["kctop"] - 1
    zdmfde = (1.0 - _WRF_CMFCTOP) * st["pmfu"][jk + 1]
    mfu_k = st["pmfu"][jk + 1] - zdmfde
    upd = dict(
        plude=(jk, zdmfde * st["plu"][jk + 1]),
        pmfu=(jk, mfu_k),
        pmfus=(jk, (cpd * st["ptu"][jk] + pgeoh[jk]) * mfu_k),
        pmfuq=(jk, st["pqu"][jk] * mfu_k),
        pmful=(jk, st["plu"][jk] * mfu_k),
        pdmfup=(jk, 0.0),
        puu=(jk, st["puu"][jk + 1]),
        pvu=(jk, st["pvu"][jk + 1]),
    )
    for nm, (at, val) in upd.items():
        st[nm] = st[nm].at[at].set(jnp.where(ldcum, val, st[nm][at]))
    st["plude"] = st["plude"].at[jk - 1].set(
        jnp.where(ldcum, st["pmful"][jk], st["plude"][jk - 1]))
    st["ldcum"] = ldcum
    return st


def _wrf_column(pten, pqen, puen, pven, pverv, pap, paph, pgeo, pqvf, pqvbl,
                pqhfl, lndj, sig1, ztmst):
    """tiecnv + cumastr_new for ONE column, 1-based padded arrays."""
    K = pten.shape[0] - 1
    g, cpd, rd, rv, alv = (constants.g, constants.c_pd, constants.R_d,
                           constants.R_v, constants.L_v)
    rcpd = 1.0 / cpd
    vtmpc1 = rv / rd - 1.0
    zcons2 = 1.0 / (g * ztmst)
    tmelt = constants.T_freeze
    lev = jnp.arange(K + 1)
    zero = jnp.zeros_like(pten)
    i32 = jnp.int32

    # ---- tiecnv (lines 575-594) ----
    pqsen = jnp.minimum(_WRF_QSAT_CAP, constants.epsilon * saturation_vapor_pressure(pten) / pap)
    pqsen = pqsen / (1.0 - vtmpc1 * pqsen)
    zqq = pqvf + pqvbl
    pqte0 = zqq

    # ---- cuini (lines 1166-1300) ----
    pgeoh = zero.at[2:].set(pgeo[2:] + (pgeo[1:-1] - pgeo[2:]) * 0.5)
    ptenh = zero.at[2:].set((jnp.maximum(cpd * pten[1:-1] + pgeo[1:-1],
                                         cpd * pten[2:] + pgeo[2:]) - pgeoh[2:]) * rcpd)
    pqsenh = zero.at[2:].set(pqsen[1:-1])
    ptenh_a, pqsenh_a = _wrf_cuadjtq(ptenh, pqsenh, jnp.where(lev >= 2, paph[: K + 1], 1.0),
                                     lev >= 2, 0)
    ptenh = jnp.where(lev >= 2, ptenh_a, ptenh)
    pqsenh = jnp.where(lev >= 2, pqsenh_a, pqsenh)
    pqenh = zero.at[2:].set(jnp.maximum(
        jnp.minimum(pqen[1:-1], pqsen[1:-1]) + (pqsenh[2:] - pqsen[1:-1]), 0.0))
    ptenh = ptenh.at[K].set((cpd * pten[K] + pgeo[K] - pgeoh[K]) * rcpd)
    pqenh = pqenh.at[K].set(pqen[K])
    ptenh = ptenh.at[1].set(pten[1])
    pqenh = pqenh.at[1].set(pqen[1])
    pgeoh = pgeoh.at[1].set(pgeo[1])

    def stab(i, t):
        jk = K - 1 - i
        zzs = jnp.maximum(cpd * t[jk] + pgeoh[jk], cpd * t[jk + 1] + pgeoh[jk + 1])
        return t.at[jk].set((zzs - pgeoh[jk]) * rcpd)
    ptenh = jax.lax.fori_loop(0, K - 2, stab, ptenh)

    def wmin(i, c):
        jk = K - i
        zw, kl = c
        m = pverv[jk] < zw
        return jnp.where(m, pverv[jk], zw), jnp.where(m, jk, kl)
    _, klwmin = jax.lax.fori_loop(0, K - 2, wmin, (jnp.zeros((), pten.dtype), i32(K)))
    shift = lambda a: a.at[1:].set(jnp.concatenate([a[1:2], a[1:-1]]))
    puu0, pvu0 = shift(puen), shift(pven)
    ptu, pqu = ptenh, pqenh

    # ---- cubase (lines 1305-1463) ----
    dpK = paph[K + 1] - paph[K]
    cb = dict(ptu=ptu, pqu=pqu, plu=zero, klab=jnp.zeros(K + 1, i32).at[K].set(1),
              kcbot=i32(K - 1), ldcum=jnp.bool_(False), ldbase=jnp.bool_(False),
              uacc=puen[K] * dpK, vacc=pven[K] * dpK)

    def base(i, s):
        jk = K - 1 - i
        s = dict(s)
        lof = (s["klab"][jk + 1] == 1) | (s["ldcum"] & (s["kcbot"] == jk + 1))
        acc = lof & ~s["ldbase"]
        dp = paph[jk + 1] - paph[jk]
        s["uacc"] = jnp.where(acc, s["uacc"] + puen[jk] * dp, s["uacc"])
        s["vacc"] = jnp.where(acc, s["vacc"] + pven[jk] * dp, s["vacc"])
        qu = s["pqu"][jk + 1]
        tu = (cpd * s["ptu"][jk + 1] + pgeoh[jk + 1] - pgeoh[jk]) * rcpd
        zqold = qu
        tu, qu = _wrf_cuadjtq(tu, qu, paph[jk], lof, 1)
        same = qu == zqold
        zbuo = tu * (1.0 + vtmpc1 * qu) - ptenh[jk] * (1.0 + vtmpc1 * pqenh[jk]) + _WRF_ZBUO0
        klab_k = jnp.where(same, jnp.where(zbuo > 0.0, 1, s["klab"][jk]), 2)
        lu = jnp.where(same, s["plu"][jk], s["plu"][jk] + zqold - qu)
        isbase = lof & ~same & (zbuo > 0.0) & (s["klab"][jk + 1] == 1)
        s["pqu"] = s["pqu"].at[jk].set(jnp.where(lof, qu, s["pqu"][jk]))
        s["ptu"] = s["ptu"].at[jk].set(jnp.where(lof, tu, s["ptu"][jk]))
        s["klab"] = s["klab"].at[jk].set(jnp.where(lof, klab_k, s["klab"][jk]))
        s["plu"] = s["plu"].at[jk].set(jnp.where(lof, lu, s["plu"][jk]))
        s["kcbot"] = jnp.where(isbase, jk, s["kcbot"])
        s["ldcum"] = s["ldcum"] | isbase
        s["ldbase"] = s["ldbase"] | isbase
        return s
    cb = jax.lax.fori_loop(0, K - 2, base, cb)
    ldcum, kcbot, klab = cb["ldcum"], cb["kcbot"], cb["klab"]
    ptu, pqu, plu = cb["ptu"], cb["pqu"], cb["plu"]
    zz = 1.0 / (paph[K + 1] - paph[kcbot])
    puu = puu0.at[K].set(jnp.where(ldcum, cb["uacc"] * zz, puen[K - 1]))
    pvu = pvu0.at[K].set(jnp.where(ldcum, cb["vacc"] * zz, pven[K - 1]))

    # ---- cumastr_new 3(b): moisture convergence and cloud type ----
    dp = paph[2:] - paph[1:-1]                      # level k -> dp[k-1]
    dpf = zero.at[1:].set(dp)
    zdqcv = jnp.sum(pqte0[1:] * dp)
    zdqpbl = jnp.sum(jnp.where((lev >= kcbot) & (lev >= 2), pqte0 * dpf, 0.0))
    ktype = jnp.where(zdqcv > jnp.maximum(0.0, _WRF_TYPE1_EVAP_FACTOR * pqhfl * g), 1, 2)
    ikb = kcbot
    zqumqe = pqu[ikb] + plu[ikb] - pqenh[ikb]
    zdqmin = jnp.maximum(_WRF_DQMIN_FRAC * pqenh[ikb], _WRF_DQMIN_FLOOR)
    okc = (zdqpbl > 0.0) & (zqumqe > zdqmin) & ldcum
    zmfub = jnp.where(okc, zdqpbl / (g * jnp.maximum(zqumqe, zdqmin)), _WRF_MFUB_FALLBACK)
    ldcum = ldcum & okc
    zmfmax = (paph[ikb] - paph[ikb - 1]) * zcons2
    zmfub = jnp.minimum(zmfub, zmfmax)
    zhcbase = cpd * ptu[ikb] + pgeoh[ikb] + alv * pqu[ikb]
    # 4(a): zhhatt and ictop0 (lines 879-903)
    zalvdcp = alv / cpd
    zqalv = 1.0 / alv
    zhsat = cpd * ptenh + pgeoh + alv * pqsenh
    zgam = (_wrf_dlnes_dT(ptenh) * zalvdcp) * pqsenh / (1.0 - vtmpc1 * pqsenh)
    zzz = cpd * ptenh * _WRF_VIRT
    zhhatt = zhsat - (zzz + zgam * zzz) / (1.0 + zgam * zzz * zqalv) * jnp.maximum(pqsenh - pqenh, 0.0)
    rng = (lev >= 3) & (lev <= K - 1)
    zhhatt = jnp.where(rng | (lev == kcbot), zhhatt, 0.0)
    cand = rng & (lev < kcbot - 1) & (zhcbase > zhhatt)
    ictop0 = jnp.where(jnp.any(cand), jnp.min(jnp.where(cand, lev, K + 1)), kcbot - 1)
    # lowest organized-detrainment level (lines 907-938)
    deep = ldcum & (ktype == 1)
    zbi = 1.0 / (_WRF_ZBI_DEPTH * g)

    def hmin(i, c):
        jk = K - i
        zh, ih = c
        llo1 = deep & (ih == kcbot) & (jk < kcbot) & (jk >= ictop0)
        zro = rd * ptenh[jk] / (g * jnp.where(llo1, paph[jk], 1.0))
        zdz = (paph[jk] - paph[jk - 1]) * zro
        dgeo = pgeo[jk - 1] - pgeo[jk]
        zdhdz = ((cpd * (pten[jk - 1] - pten[jk]) + alv * (pqen[jk - 1] - pqen[jk])
                  + dgeo) * g / jnp.where(llo1, dgeo, 1.0))
        zfac = jnp.sqrt(jnp.where(llo1, 1.0 + (pgeoh[jk] - pgeoh[kcbot]) * zbi, 1.0))
        zh_n = zh + zdhdz * zfac * zdz
        zrh = -alv * (pqsenh[jk] - pqenh[jk]) * zfac
        return (jnp.where(llo1, zh_n, zh),
                jnp.where(llo1 & (zh_n > zrh), jk, ih))
    _, ihmin = jax.lax.fori_loop(0, K, hmin, (jnp.zeros((), pten.dtype),
                                               jnp.where(deep, kcbot, i32(-1))))
    ihmin = jnp.where(deep & (ihmin < ictop0), ictop0, ihmin)
    land = lndj
    zentr = jnp.where(ktype == 1, _WRF_ENTRPEN, _WRF_ENTRSCV)
    # land weight in [0, 1]: 1 + (1.1 - 1)*w is exactly 1.1 at w = 1 (oracle)
    zentr = zentr * (1.0 + (_WRF_LAND_ENTR_FACTOR - 1.0) * land)

    c = dict(K=K, dt=ztmst, paph=paph, pgeo=pgeo, pgeoh=pgeoh, ptenh=ptenh,
             pqenh=pqenh, pqsenh=pqsenh, pten=pten, pqen=pqen, pqsen=pqsen,
             pap=pap, puen=puen, pven=pven, pverv=pverv, klwmin=klwmin,
             khmin=ihmin, zhhatt=zhhatt, zhcbase=zhcbase)
    s = dict(ldcum=ldcum, ktype=ktype, klab=klab, kctop0=ictop0, kcbot=kcbot,
             pmfub=zmfub, pentr=zentr, ptu=ptu, pqu=pqu, puu=puu, pvu=pvu)
    a1 = _wrf_cuasc(c, s)
    icum1 = a1["ldcum"]
    # (c) cloud depth -> type, precipitation for the downdraft (lines 963-978)
    ldcum, ktype, kcbot, kctop = a1["ldcum"], a1["ktype"], a1["kcbot"], a1["kctop"]
    zpbmpt = paph[kcbot] - paph[kctop]
    ictop0 = jnp.where(ldcum, kctop, a1["kctop0"])
    ktype = jnp.where(ldcum & (ktype == 1) & (zpbmpt < _WRF_ZDNOPRC), 2, ktype)
    zentr = jnp.where(ktype == 2, _WRF_ENTRSCV * (1.0 + (_WRF_LAND_ENTR_FACTOR - 1.0) * land),
                      a1["pentr"])
    zrfl = jnp.sum(a1["pdmfup"][1:])
    zmfub = a1["pmfub"]
    ptu, pqu, plu, puu, pvu = a1["ptu"], a1["pqu"], a1["plu"], a1["puu"], a1["pvu"]
    pmfu = a1["pmfu"]

    # ---- 5. downdrafts: cudlfs (lines 1917-2046) ----
    ptd, pqd = ptenh, pqenh
    pud, pvd = puu0, pvu0
    dd = dict(ptd=ptd, pqd=pqd, pud=pud, pvd=pvd, pmfd=zero, pmfds=zero, pmfdq=zero,
              pdmfdp=zero, kdtop=i32(K + 1), lddraf=jnp.bool_(False), prfl=zrfl)

    def lfs(i, d):
        jk = 3 + i
        d = dict(d)
        llo2 = (ldcum & (d["prfl"] > 0.0) & ~d["lddraf"]
                & (jk < kcbot) & (jk > kctop))
        twb, qwb = _wrf_cuadjtq(ptenh[jk], pqenh[jk], paph[jk], llo2, 2)
        zttest = 0.5 * (ptu[jk] + twb)
        zqtest = 0.5 * (pqu[jk] + qwb)
        zbuo = zttest * (1.0 + vtmpc1 * zqtest) - ptenh[jk] * (1.0 + vtmpc1 * pqenh[jk])
        zcond = pqenh[jk] - qwb
        zmftop = -_WRF_CMFDEPS * zmfub
        hit = llo2 & (zbuo < 0.0) & (d["prfl"] > _WRF_LFS_RAIN * zmftop * zcond)
        dmfdp = -0.5 * zmftop * zcond

        def put(nm, at, val):
            d[nm] = d[nm].at[at].set(jnp.where(hit, val, d[nm][at]))
        put("ptd", jk, zttest)
        put("pqd", jk, zqtest)
        put("pmfd", jk, zmftop)
        put("pmfds", jk, zmftop * (cpd * zttest + pgeoh[jk]))
        put("pmfdq", jk, zmftop * zqtest)
        put("pdmfdp", jk - 1, dmfdp)
        d["prfl"] = jnp.where(hit, d["prfl"] + dmfdp, d["prfl"])
        d["kdtop"] = jnp.where(hit, jk, d["kdtop"])
        d["lddraf"] = d["lddraf"] | hit
        put("pud", jk, 0.5 * (puu[jk] + puen[jk - 1]))
        put("pvd", jk, 0.5 * (pvu[jk] + pven[jk - 1]))
        return d
    dd = jax.lax.fori_loop(0, K - 5, lfs, dd)
    lddraf = dd["lddraf"]

    # ---- cuddraf (lines 2052-2193) ----
    itopde = K - 2

    def descent(i, d):
        jk = 3 + i
        d = dict(d)
        llo2 = lddraf & (d["pmfd"][jk - 1] < 0.0)
        mfd_up = d["pmfd"][jk - 1]
        zentr_d = (_WRF_ENTRDD * mfd_up * rd * ptenh[jk - 1] / (g * paph[jk - 1])
                   * (paph[jk] - paph[jk - 1]))
        top = jk > itopde
        zdmfen = jnp.where(top, 0.0, zentr_d)
        zdmfde = jnp.where(top, d["pmfd"][itopde] * (paph[jk] - paph[jk - 1])
                           / (paph[K + 1] - paph[itopde]), zentr_d)
        mfd_k = mfd_up + zdmfen - zdmfde
        zseen = (cpd * ptenh[jk - 1] + pgeoh[jk - 1]) * zdmfen
        zqeen = pqenh[jk - 1] * zdmfen
        zsdde = (cpd * d["ptd"][jk - 1] + pgeoh[jk - 1]) * zdmfde
        zqdde = d["pqd"][jk - 1] * zdmfde
        zmfdsk = d["pmfds"][jk - 1] + zseen - zsdde
        zmfdqk = d["pmfdq"][jk - 1] + zqeen - zqdde
        rm = 1.0 / jnp.minimum(-_WRF_CMFCMIN, mfd_k)
        qd = zmfdqk * rm
        td = jnp.maximum(jnp.minimum((zmfdsk * rm - pgeoh[jk]) * rcpd, _WRF_TU_MAX), _WRF_TU_MIN)
        zcond = qd
        td, qd = _wrf_cuadjtq(td, qd, paph[jk], llo2, 2)
        zcond = zcond - qd
        zbuo = td * (1.0 + vtmpc1 * qd) - ptenh[jk] * (1.0 + vtmpc1 * pqenh[jk])
        mfd_k = jnp.where((zbuo >= 0.0) | (d["prfl"] <= mfd_k * zcond), 0.0, mfd_k)
        zdmfdp = -mfd_k * zcond
        frc = llo2 & (mfd_k < 0.0)
        zmfduk = mfd_up * d["pud"][jk - 1] + zdmfen * puen[jk - 1] - zdmfde * d["pud"][jk - 1]
        zmfdvk = mfd_up * d["pvd"][jk - 1] + zdmfen * pven[jk - 1] - zdmfde * d["pvd"][jk - 1]
        rmu = 1.0 / jnp.minimum(-_WRF_CMFCMIN, mfd_k)

        def put(nm, at, val, m=llo2):
            d[nm] = d[nm].at[at].set(jnp.where(m, val, d[nm][at]))
        put("pmfd", jk, mfd_k)
        put("pqd", jk, qd)
        put("ptd", jk, td)
        put("pmfds", jk, (cpd * td + pgeoh[jk]) * mfd_k)
        put("pmfdq", jk, qd * mfd_k)
        put("pdmfdp", jk - 1, zdmfdp)
        d["prfl"] = jnp.where(llo2, d["prfl"] + zdmfdp, d["prfl"])
        put("pud", jk, zmfduk * rmu, frc)
        put("pvd", jk, zmfdvk * rmu, frc)
        return d
    dd = jax.lax.fori_loop(0, K - 2, descent, dd)
    pmfd, zmfds, zmfdq, zdmfdp = dd["pmfd"], dd["pmfds"], dd["pmfdq"], dd["pdmfdp"]
    pqd, pud, pvd, idtop = dd["pqd"], dd["pud"], dd["pvd"], dd["kdtop"]

    # ---- 5.1 CAPE closure for deep convection (lines 1010-1050) ----
    deep = ldcum & (ktype == 1)
    ktop0 = jnp.maximum(_WRF_KTOP0_MIN, kctop)
    ikb = kcbot
    inl = (lev >= 2) & (lev <= kcbot) & (lev > kctop)
    prv = jnp.concatenate([pten[:1], pten[:-1]])
    prq = jnp.concatenate([pqen[:1], pqen[:-1]])
    paphm = paph[: K + 1].at[1:].set(paph[:K])
    pths = jnp.where(inl, ptenh, 1.0)   # masked lanes (top/pad) kept finite for AD
    zro = jnp.where(inl, paph[: K + 1], 1.0) / (rd * pths)
    zdz = (paph[: K + 1] - paphm) / (g * zro)
    hterm = (((prv - pten + g * zdz / cpd) / pths + _WRF_VIRT * (prq - pqen))
             * (pmfu + pmfd) * g / zro)
    cterm = g * ((ptu * (1.0 + _WRF_VIRT * pqu - plu)) / (pths * (1.0 + _WRF_VIRT * pqenh)) - 1.0) * zdz
    zheat = jnp.sum(jnp.where(inl, hterm, 0.0))
    zcape = jnp.sum(jnp.where(inl, cterm, 0.0))
    inr = (lev >= 2) & (lev <= kcbot) & (lev > ktop0)
    den = paph[ikb + 1] - paph[ktop0 + 1]
    dept = (paph[1:] - paph[:-1]) / jnp.where(jnp.any(inr), den, 1.0)
    zrelh = jnp.sum(jnp.where(inr, dept * pqen / pqsen, 0.0))
    okr = deep & (zrelh >= _WRF_CRIRH)
    zht = jnp.maximum(0.0, zcape) / (_WRF_ZTAU * jnp.where(okr, zheat, 1.0))
    zmfmax = (paph[ikb] - paph[ikb - 1]) * zcons2
    zmfub1 = jnp.where(okr, jnp.minimum(jnp.maximum(zmfub * zht, _WRF_MFUB_FALLBACK), zmfmax),
                       jnp.where(deep, _WRF_MFUB_FALLBACK, zmfub))
    zmfub = jnp.where(deep & ~okr, _WRF_MFUB_FALLBACK, zmfub)
    ldcum = ldcum & ~(deep & ~okr)
    # ---- 5.2 shallow: downdraft effect on the PBL moisture budget ----
    zeps = jnp.where((pmfd[ikb] < 0.0) & lddraf, _WRF_CMFDEPS, 0.0)
    zqumqe = pqu[ikb] + plu[ikb] - zeps * pqd[ikb] - (1.0 - zeps) * pqenh[ikb]
    zdqmin = jnp.maximum(_WRF_DQMIN_FRAC * pqenh[ikb], _WRF_DQMIN_FLOOR)
    oks = (zdqpbl > 0.0) & (zqumqe > zdqmin) & ldcum & (zmfub < zmfmax)
    z1 = jnp.where(oks, zdqpbl / (g * jnp.maximum(zqumqe, zdqmin)), zmfub)
    llo1 = (ktype == 2) & (jnp.abs(z1 - zmfub) < _WRF_SHALLOW_TOL * zmfub)
    z1 = jnp.minimum(jnp.where(llo1, z1, zmfub), zmfmax)
    zmfub1 = jnp.where(ktype != 1, z1, zmfub1)
    zfac = zmfub1 / jnp.maximum(zmfub, _WRF_MFUB_FLOOR)
    pmfd, zmfds, zmfdq, zdmfdp = (jnp.where(ldcum, a * zfac, 0.0)
                                  for a in (pmfd, zmfds, zmfdq, zdmfdp))
    zmfub = jnp.where(ldcum, zmfub1, 0.0)

    # ---- 6. final ascent ----
    s = dict(ldcum=ldcum, ktype=ktype, klab=a1["klab"], kctop0=ictop0, kcbot=kcbot,
             pmfub=zmfub, pentr=zentr, ptu=ptu, pqu=pqu, puu=puu, pvu=pvu)
    a2 = _wrf_cuasc(c, s)
    ldcum, ktype, kcbot, kctop = a2["ldcum"], a2["ktype"], a2["kcbot"], a2["kctop"]
    pmfu, pmfus, pmfuq, pmful = a2["pmfu"], a2["pmfus"], a2["pmfuq"], a2["pmful"]
    plude, pdmfup, puu, pvu = a2["plude"], a2["pdmfup"], a2["puu"], a2["pvu"]
    zmfub = a2["pmfub"]

    # ---- 7. cuflx (lines 2199-2367) ----
    itop = jnp.minimum(K, kctop)
    lddraf = lddraf & ldcum & ~(idtop < kctop)
    ktype = jnp.where(ldcum, ktype, 0)
    ktopm2 = itop - 2
    inr = lev >= ktopm2
    upper = ldcum & (lev >= kctop - 1)
    sh = cpd * ptenh + pgeoh
    ddm = lddraf & (lev >= idtop)
    nz_up = inr & upper
    nz_dd = nz_up & ddm
    pmfus = jnp.where(nz_up, pmfus - pmfu * sh, jnp.where(inr, 0.0, pmfus))
    pmfuq = jnp.where(nz_up, pmfuq - pmfu * pqenh, jnp.where(inr, 0.0, pmfuq))
    zmfds = jnp.where(nz_dd, zmfds - pmfd * sh, jnp.where(inr, 0.0, zmfds))
    zmfdq = jnp.where(nz_dd, zmfdq - pmfd * pqenh, jnp.where(inr, 0.0, zmfdq))
    pmfd = jnp.where(inr & ~nz_dd, 0.0, pmfd)
    pmfu = jnp.where(inr & ~upper, 0.0, pmfu)
    pmful = jnp.where(inr & ~upper, 0.0, pmful)
    # writes at jk-1 (jk = ktopm2 .. K): k = jk-1 in [ktopm2-1, K-1]
    nxt = lambda a: jnp.concatenate([a[1:], a[-1:]])   # value of jk = k+1
    inr1, upper1, nzdd1 = nxt(inr), nxt(upper), nxt(nz_dd)
    zdmfdp = jnp.where(inr1 & ~nzdd1, 0.0, zdmfdp).at[K].set(zdmfdp[K])
    pdmfup = jnp.where(inr1 & ~upper1, 0.0, pdmfup).at[K].set(pdmfup[K])
    plude = jnp.where(inr1 & ~upper1, 0.0, plude).at[K].set(plude[K])
    below = inr & ldcum & (lev > kcbot)
    zzp = (paph[K + 1] - paph[: K + 1]) / (paph[K + 1] - paph[kcbot])
    zzp = jnp.where(ktype == 3, zzp ** 2, zzp)
    pmfu = jnp.where(below, pmfu[kcbot] * zzp, pmfu)
    pmfus = jnp.where(below, pmfus[kcbot] * zzp, pmfus)
    pmfuq = jnp.where(below, pmfuq[kcbot] * zzp, pmfuq)
    pmful = jnp.where(below, pmful[kcbot] * zzp, pmful)
    ztmelp2 = tmelt + _WRF_MELT_DT

    def fall(i, f):
        jk = i + 1
        prain, prfl, psfl = f
        on = ldcum & (jk >= ktopm2)
        warm = pten[jk] > tmelt
        melt = warm & (psfl > 0.0) & (pten[jk] > ztmelp2)
        # alf = 0 (water only): cpd/(alf*g*dt) is infinite, so zsnmlt = psfl
        prfl_w = prfl + pdmfup[jk] + zdmfdp[jk]
        prfl_n = jnp.where(warm, jnp.where(melt, prfl_w + psfl, prfl_w), prfl)
        psfl_n = jnp.where(warm, jnp.where(melt, 0.0, psfl), psfl + pdmfup[jk] + zdmfdp[jk])
        return (jnp.where(on, prain + pdmfup[jk], prain),
                jnp.where(on, prfl_n, prfl), jnp.where(on, psfl_n, psfl))
    z0 = jnp.zeros((), pten.dtype)
    prain, prfl, psfl = jax.lax.fori_loop(0, K, fall, (z0, z0, z0))
    prfl = jnp.maximum(prfl, 0.0)
    psfl = jnp.maximum(psfl, 0.0)

    def evap(i, f):
        jk = i + 1
        zps, dmfup = f
        on = ldcum & (jk >= ktopm2) & (jk >= kcbot) & (zps > _WRF_TINY)
        cevapcu = (_WRF_CEVAPCU1_G / g) * jnp.sqrt(_WRF_CEVAPCU2 * jnp.sqrt(sig1[jk]))
        dpk = paph[jk + 1] - paph[jk]
        zr = jnp.where(on, zps, 1.0)
        zrnew = (jnp.maximum(0.0, jnp.sqrt(zr / _WRF_CUCOV) - cevapcu * dpk
                             * jnp.maximum(0.0, pqsen[jk] - pqen[jk]))) ** 2 * _WRF_CUCOV
        zrmin = zr - _WRF_CUCOV * jnp.maximum(0.0, _WRF_EVAP_RH * pqsen[jk] - pqen[jk]) * zcons2 * dpk
        zrfln = jnp.maximum(jnp.maximum(zrnew, zrmin), 0.0)
        zdrfl = jnp.minimum(0.0, zrfln - zr)
        return (jnp.where(on, zrfln, zps),
                dmfup.at[jk].set(jnp.where(on, dmfup[jk] + zdrfl, dmfup[jk])))
    zpsubcl, pdmfup = jax.lax.fori_loop(0, K, evap, (prfl + psfl, pdmfup))
    zdpevap = zpsubcl - (prfl + psfl)
    def rcp_floor(x):
        # 1./max(1e-20, x) with identical values; the floored branch is a
        # constant so its (overflowing in fp32) 1/x**2 derivative never forms.
        big = x > _WRF_TINY
        return jnp.where(big, 1.0 / jnp.where(big, x, 1.0), 1.0 / _WRF_TINY)
    def share(x, tot):
        # x + zdpevap*x*(1/max(1e-20, tot)); for x == 0 the term is exactly 0
        # (and x is piecewise constant there), so it is skipped: identical
        # values, and no 0 * 1/tot**2 (= NaN in fp32 once the rain has fully
        # evaporated and tot ~ 1e-20).
        nz = x != 0.0
        return x + jnp.where(nz, zdpevap * x * rcp_floor(jnp.where(nz, tot, 1.0)), 0.0)
    prfl = share(prfl, prfl + psfl)
    psfl = share(psfl, prfl + psfl)

    # ---- 8. cudtdq (lines 2373-2493); alf = 0, als = alv ----
    act = ldcum & (lev >= ktopm2) & (lev >= 1)
    rhk = jnp.minimum(1.0, pqen / pqsen)
    rhcoe = jnp.maximum(0.0, (rhk - _WRF_RHC) / (_WRF_RHM - _WRF_RHC))
    pldfd = jnp.maximum(0.0, rhcoe * _WRF_FDBK * plude)
    gdp = g / (paph[1:] - paph[:-1]).at[0].set(1.0)   # level k: g/(paph(k+1)-paph(k))
    srcp = pdmfup + zdmfdp
    dtdt_in = gdp * rcpd * (nxt(pmfus) - pmfus + nxt(zmfds) - zmfds
                            - alv * (nxt(pmful) - pmful - pldfd - srcp))
    dqdt_in = gdp * (nxt(pmfuq) - pmfuq + nxt(zmfdq) - zmfdq + nxt(pmful) - pmful - pldfd - srcp)
    dtdt_K = -gdp * rcpd * (pmfus + zmfds - alv * (pmful + pdmfup + zdmfdp + pldfd))
    dqdt_K = -gdp * (pmfuq + zmfdq + pldfd + (pmful + pdmfup + zdmfdp))
    isK = lev == K
    ptte = jnp.where(act, jnp.where(isK, dtdt_K, dtdt_in), 0.0)
    pqte = pqte0 + jnp.where(act, jnp.where(isK, dqdt_K, dqdt_in), 0.0)
    pcte = jnp.where(act, gdp * pldfd, 0.0)

    # ---- 9. cududv (lines 2499-2602) ----
    puen_m = jnp.concatenate([puen[:1], puen[:-1]])
    pven_m = jnp.concatenate([pven[:1], pven[:-1]])
    zmfuu = pmfu * (puu - puen_m)
    zmfuv = pmfu * (pvu - pven_m)
    zmfdu = pmfd * (pud - puen_m)
    zmfdv = pmfd * (pvd - pven_m)
    zmfuu, zmfuv, zmfdu, zmfdv = (jnp.where(below, a[kcbot] * zzp, a)
                                  for a in (zmfuu, zmfuv, zmfdu, zmfdv))
    du_in = gdp * (nxt(zmfuu) - zmfuu + nxt(zmfdu) - zmfdu)
    dv_in = gdp * (nxt(zmfuv) - zmfuv + nxt(zmfdv) - zmfdv)
    pvom = jnp.where(act, jnp.where(isK, -gdp * (zmfuu + zmfdu), du_in), 0.0)
    pvol = jnp.where(act, jnp.where(isK, -gdp * (zmfuv + zmfdv), dv_in), 0.0)

    # ---- tiecnv: detrained condensate split (lines 612-635; alf = 0) ----
    ztpp1 = pten + ptte * ztmst
    ztc = ztpp1 - constants.T_freeze
    fliq = jnp.where(ztpp1 >= constants.T_freeze, 1.0,
                     jnp.where(ztpp1 <= _WRF_HGFR, 0.0,
                               _WRF_FLIQ_A + _WRF_FLIQ_B * jnp.exp(-_WRF_FLIQ_C * ztc * ztc)))
    pos = pcte > 0.0
    dqc = jnp.where(pos, fliq * pcte, 0.0)
    dqi = jnp.where(pos, (1.0 - fliq) * pcte, 0.0)

    # icum = 0 after the first ascent: cumastr_new returns with no tendencies
    k1 = icum1
    z = lambda a: jnp.where(k1, a, 0.0)
    return dict(
        dT_dt=z(ptte), dq_v_dt=z(pqte - zqq), du_dt=z(pvom), dv_dt=z(pvol),
        dq_c_dt=z(dqc), dq_i_dt=z(dqi), detrained_condensate=z(pcte),
        precip=z(prfl + psfl), precip_snow=z(psfl), mfu=z(pmfu), mfd=z(pmfd),
        mfub=jnp.where(k1 & ldcum, zmfub, 0.0), ldcum=k1 & ldcum,
        ktype=jnp.where(k1, ktype, a1["ktype"]), kcbot=kcbot - 1, kctop=kctop - 1)


def tiedtke_wrf(T, q_v, u, v, p_full, p_half, geopotential, omega,
                dq_v_dt_dyn, dq_v_dt_pbl, surface_moisture_flux, land,
                sigma, dt) -> TiedtkeWRFOutput:
    """WRF v4.8.0 Tiedtke convection (``cu_tiedtke``/``tiecnv``), faithful port.

    All profiles (ncol, nlev), top-down (surface at [:, -1]).

    T [K]; q_v [kg/kg] specific humidity (tiecnv's ``zqp1``); u, v [m/s];
    p_full [Pa] (``pap``); p_half (ncol, nlev+1) [Pa] (``paph``);
    geopotential [m2/s2] at full levels (``pgeo``); omega [Pa/s] (``pverv``);
    dq_v_dt_dyn, dq_v_dt_pbl [kg/kg/s] large-scale and boundary-layer
    moisture tendencies (``qvften``, ``qvpblten``, added unconverted as in
    tiecnv); surface_moisture_flux [kg/m2/s] (``qfx``, upward); land (ncol,)
    land weight in [0, 1] (``lndj`` as 0/1: entrainment x(1 + 0.1 w), exactly
    the oracle's x1.1 at w = 1); sigma (ncol, nlev) full-level sigma
    (``znu``, sub-cloud evaporation); dt [s].

    Returns :class:`TiedtkeWRFOutput`.
    """
    def padf(a):
        return jnp.concatenate([a[:, :1], a], axis=1)

    args = (padf(T), padf(q_v), padf(u), padf(v), padf(omega), padf(p_full),
            padf(p_half), padf(geopotential), padf(dq_v_dt_dyn), padf(dq_v_dt_pbl),
            surface_moisture_flux, jnp.asarray(land, T.dtype), padf(sigma))
    out = jax.vmap(lambda *a: _wrf_column(*a, dt))(*args)
    return TiedtkeWRFOutput(**{k: (v[:, 1:] if v.ndim == 2 else v) for k, v in out.items()})
