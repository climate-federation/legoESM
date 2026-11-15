"""OpenIFS 49r1 vertical diffusion (VDF) column kernels, stage 1a.

Faithful port of OpenIFS (github.com/ecmwf-ifs/openifs @ 8f6f722f73, cycle
49r1) ``ifs-source/arpifs/phys_ec/``:

* ``vdfdpbl.F90`` -- diagnostic boundary-layer height (bulk Richardson
  number 0.25 crossing) :func:`ifs_vdfdpbl`;
* ``vdfexcu.F90`` -- exchange coefficients between the model levels above the
  surface layer (Louis-Tiedtke-Geleyn stable functions with a PBL-height
  asymptotic length, Monin-Obukhov unstable functions, the K-profile of the
  mixed layer with cloud-top driven K and top entrainment)
  :func:`ifs_vdfexcu`;
* ``vdftofdc.F90`` -- the turbulent orographic form drag coefficient
  (Beljaars, Brown & Wood 2004) :func:`ifs_tofd_coefficient`.

Configuration ported: the 49r1 defaults, ``LRDIFF_STRATO = .FALSE.``
(sugwd.F90:200, the stratospheric reduction branch is not ported), no SPP
perturbation (``LSPP = .FALSE.``), Earth (``RPLRG = RPLDARE = 1``).
``vdffblend.F90`` (blending-height wind for the surface postprocessing
SURFPP) is not ported: SURFPP is not ported (VOTES.md vote 3, S1).

Inputs that VDFHGHTN produces (``kinv``, ``kcbase``, ``kctop``, ``kpbltype``,
``zinv``, ``ri``, ``khvfl``) are arguments; stage 1b ports VDFHGHTN.

Differentiability (oracle_audit/ifs_vdf/VOTES.md vote 3): integer level picks
(the Ri = 0.25 crossing level, KINV/KCBASE/KCTOP/KPBLTYPE) stay discrete as in
the merged IFS convection and SSO ports; every value switch, floor and cap on
traced state is smooth (C1, exact outside its band) with a width in
:class:`IFSVDFConfig`.  :func:`hard_config` (all widths 0) is the oracle's
hard form.

KNOWN DIFFERENCES at the smooth default (measured per fixture column and
stored with the test, tests/atmosphere/test_ifs_vdf_oracle.py): the smooth
switches move the exchange coefficients only for columns with a level inside
a switch band; the hard form matches the oracle to 1e-12.

Levels run top -> surface; full-level arrays have KLEV entries (Fortran
1..KLEV), half-level arrays KLEV+1 entries (Fortran 0..KLEV).  One column per
call, vmapped over columns.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.smooth import max_c1, min_c1, smoothstep_cubic, sqrt_c1

__physics_contract__ = {
    "summary": (
        "OpenIFS 49r1 VDF column kernels: diagnostic PBL height (VDFDPBL), "
        "exchange coefficients above the surface layer (VDFEXCU), TOFD "
        "coefficient (VDFTOFDC)."
    ),
    "inputs": {
        "u, v": "m s-1 (KLEV, top -> surface)", "T": "K", "q": "kg kg-1",
        "geo_full": "m2 s-2 above the surface", "geo_half": "m2 s-2 (KLEV+1)",
        "p_half": "Pa (KLEV+1)", "cptgz": "J kg-1 (dry static energy)",
        "kmfl": "m2 s-2 (kinematic momentum flux)", "khfl": "K m s-1 (downward +)",
        "kqfl": "m s-1 (downward +)", "khvfl": "K m s-1 (buoyancy flux, downward +)",
        "ri": "1 (gradient Richardson number from VDFHGHTN)",
        "hrlw, hrsw": "K s-1", "z0m": "m",
    },
    "outputs": {
        "pblh": "m", "cfm, cfh": "kg m-2 s-1 x (1/m) scaled exchange (C-star, VDFDIFH/M units)",
        "km, kh": "m2 s-1", "tofdc": "s-1 (du/dt = -tofdc*u)",
    },
    "sign_convention": (
        "levels top -> surface; surface fluxes positive DOWNWARD (IFS), so a "
        "negative khvfl is an unstable surface layer; exchange coefficients >= 0."
    ),
    # coefficients only: the VDFDIFH/VDFDIFM solvers (stage 2) carry the budgets
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Beljaars & Viterbo (1998); Koehler et al. (2011) QJRMS 137, 43-57; "
        "Sandu et al. (2013) JAMES 5, 117-133; Lock et al. (2000) MWR 128, 3187; "
        "Beljaars, Brown & Wood (2004) QJRMS 130, 1327-1347; OpenIFS 49r1 "
        "(8f6f722) vdfdpbl.F90 / vdfexcu.F90 / vdftofdc.F90 / suvdf.F90 / suvdfs.F90"
    ),
    "idealized_test": "tests/atmosphere/test_ifs_vdf_oracle.py (hard form vs running OpenIFS)",
}

__param_spec__ = {
    "IFSVDFConfig": {
        "scheme_key": "atm.turb.IFSVDFConfig",
        "excluded": {
            "repdu2_width": "smoothing width",
            "repust_width_m_s": "smoothing width",
            "khvfl_gate_width": "smoothing width",
            "ri_width": "smoothing width",
            "pblh_switch_width_m": "smoothing width",
            "lmin_width_m": "smoothing width",
            "lmax_width_m": "smoothing width",
            "hr_min_width": "smoothing width",
            "dradflx_width": "smoothing width",
            "flux_test_width_w_m2": "smoothing width",
            "cbase_ratio_width": "smoothing width",
            "wtventr_width": "smoothing width",
            "dtv_width_k": "smoothing width",
            "kcap_width": "smoothing width",
            "cbrt_width": "smoothing width",
            "sqrt_width": "smoothing width",
            "guard_width_frac": "smoothing width",
        },
        "params": {},
    },
    "IFSTOFDConfig": {
        "scheme_key": "atm.turb.IFSTOFDConfig",
        "excluded": {"zmax_width_m": "smoothing width", "repdu2_width_frac": "smoothing width"},
        "params": {
            "tofd_alpha": {"units": "1", "bounds": (5.0, 50.0), "tunable_tier": 1,
                           "transform": "sigmoid", "category": "form_drag",
                           "reference": "Beljaars et al. (2004); OpenIFS 49r1 suvdf RTOFDALPHA",
                           "shape": None},
        },
    },
}

# --- shared numbers (parphy.F90, surf suvexc_mod.F90) ---
_REPDU2 = 0.1 ** 2           # m2 s-2, parphy.F90:25
_REPUST = 0.0001             # m s-1, suvexc_mod.F90:160 (SURF_INQ PREPUST)
_RKAP = 0.4                  # parphy.F90:24 von Karman (= constants.kappa_von_karman)
# --- vdfdpbl.F90 / vdfexcu.F90 fixed numbers (49r1) ---
_RICRI = 0.25                # critical bulk Richardson number
_PARZI_M = 1000.0            # vdfdpbl.F90 ZPARZI (no crossing found)
_EPS_SV = 1.0e-10            # ZEPS added to the surface s_v (vdfdpbl/vdfexcu)
_ENTRSFC = 0.2               # vdfexcu.F90 ZENTRSFC, surface-driven top entrainment
_ENTRRAD = 0.3               # vdfexcu.F90 ZENTRRAD, radiation-driven top entrainment
_LTG_D = 1.0                 # vdfexcu.F90 ZCD
_LTG_B = 5.0                 # vdfexcu.F90 ZCB
_LMIN_M = 30.0               # vdfexcu.F90 ZLMIN, free-shear asymptotic length
_LMAX_M = 300.0              # vdfexcu.F90 cap of the stable PBL length
_LSTAB_FRAC = 0.1            # vdfexcu.F90 asymptotic length = 0.1 x PBL height
_PHIK_H = 39.0               # vdfexcu.F90 ZPHIKH = (1 - 39 z/L)^(-1/3)
_PHIK_M = 15.0               # vdfexcu.F90 ZPHIKM = (1 - 15 z/L)^(-1/3)
_KCLD_COEF = 0.85            # vdfexcu.F90 Lock et al. (2000) eq. 5 prefactor
_FLUX_TEST_W_M2 = -0.1       # vdfexcu.F90 RCPD*PKHFL < -0.1
_FLUX_TEST_BOWEN = -0.25     # vdfexcu.F90 -RCPD*PKHFL < -0.25*RLVTT*PKQFL
_DTV_MIN_K = 1.0e-3          # vdfexcu.F90 ZDTV > 1E-3
_KCAP_M2_S = 1.0e6           # vdfexcu.F90 safety limit on K
_EPS_FLUX = 1.0e-10          # vdfexcu.F90 ZEPS in the Obukhov length
_EPS_MACH = 100.0 * float(np.finfo(np.float64).eps)   # vdfexcu.F90 ZEPSILON
_RLAM_M = 150.0              # suvdf.F90:124 RLAM, unstable asymptotic length
_RCDHALF = 16.0              # suvdfs.F90:159, PHIMU/PHIHU (fcvdfs.func.h)
# --- branch-evaluation bounds (numerics, vote 4 G1): inside the validity of
# the LTG denominators and (1-16Ri)^(-1/4), outside the Ri blend band ---
_RI_STABLE_LO = -0.03          # the LTG momentum denominator vanishes at Ri = -0.095
_RI_UNSTABLE_HI = 0.03
# --- TOFD spectral fit (Beljaars et al. 2004; vdftofdc.F90).  The Fortran
# assigns these as default-REAL (single precision) literals, so the oracle
# computes with their float32 values; kept bit-identical here. ---
_f4 = lambda x: float(np.float32(x))
_TOFD_BETA = _f4(1.0)
_TOFD_CMD = _f4(0.005)
_TOFD_CORR = _f4(0.6)
_TOFD_K1 = _f4(0.003)
_TOFD_IH = _f4(0.00102)
_TOFD_KFLT = _f4(0.00035)
_TOFD_N1 = _f4(-1.9)
_TOFD_N2 = _f4(-2.8)
_TOFD_IC = _f4(2.109)
_TOFD_IZ_M = _f4(1500.0)
_TOFD_IN = _f4(-1.2)
_TOFD_EXP = _f4(1.5)
_TOFD_ZMAX_M = _f4(5000.0)
_TOFD_FACT1 = _TOFD_K1 ** (_TOFD_N1 - _TOFD_N2) / (_TOFD_IH * _TOFD_KFLT ** _TOFD_N1)


class IFSVDFConfig(NamedTuple):
    """Smoothing widths (0 = the oracle's hard form; VOTES.md vote 3)."""

    repdu2_width: float = 1.0e-4          # m2 s-2, MAX(REPDU2, du^2)
    repust_width_m_s: float = 1.0e-6      # MAX(sqrt(KMFL), REPUST)
    khvfl_gate_width: float = 1.0e-3      # K m s-1, stable PBL height only if KHVFL > 0
    ri_width: float = 0.0025              # stable (Ri > 0) / unstable blend
    pblh_switch_width_m: float = 10.0     # z <= PBL height (stable length)
    lmin_width_m: float = 0.3             # MAX(30 m, 0.1 h)
    lmax_width_m: float = 3.0             # MIN(300 m, .)
    hr_min_width: float = 1.0e-6          # K s-1, MIN of the two heating rates
    dradflx_width: float = 1.0e-4         # K m s-1, MAX(radiative jump, 0)
    flux_test_width_w_m2: float = 1.0     # the two surface-flux tests
    cbase_ratio_width: float = 0.01       # MIN(1, z/z_cloudbase)
    wtventr_width: float = 1.0e-4         # K m s-1, MAX(0, entrainment flux)
    dtv_width_k: float = 1.0e-5           # ZDTV > 1e-3 K
    kcap_width: float = 1.0e4             # m2 s-1, MIN(K, 1e6)
    cbrt_width: float = 0.0033            # m3 s-3, C1 bridge of the radiative velocity cube root
    sqrt_width: float = 0.01              # C1 bridge of (1 - z/zml)^0.5
    guard_width_frac: float = 0.01        # |L| >= ZEPSILON and MAX(ZEPSILON, z_inv) guards


class IFSTOFDConfig(NamedTuple):
    """OpenIFS 49r1 TOFD setting (suvdf.F90) and smooth widths (0 = hard)."""

    tofd_alpha: float = 24.0             # RTOFDALPHA, suvdf.F90:128
    zmax_width_m: float = 50.0           # 0.01 x the 5000 m cut-off
    repdu2_width_frac: float = 0.01


class IFSVDFExcuOutput(NamedTuple):
    """VDFEXCU outputs; entry KLEV (the surface layer) is set by the caller."""

    cfm: jax.Array    # scaled momentum exchange (VDFDIFM units), KLEV-1 interfaces + 0
    cfh: jax.Array    # scaled heat exchange (VDFDIFH units)
    km: jax.Array     # m2 s-1
    kh: jax.Array     # m2 s-1


def hard_config(cfg: IFSVDFConfig = IFSVDFConfig()) -> IFSVDFConfig:
    """The oracle's hard form: every smoothing width 0 (oracle-match test only)."""
    return cfg._replace(**{f: 0.0 for f in cfg._fields})


def hard_tofd_config(cfg: IFSTOFDConfig = IFSTOFDConfig()) -> IFSTOFDConfig:
    """The oracle's hard form (oracle-match test only)."""
    return cfg._replace(zmax_width_m=0.0, repdu2_width_frac=0.0)


def _step(x, width, strict=True):
    """Heaviside(x), C1 cubic over |x| < width/2, exact outside (vote 3 W1).

    Hard (width 0): ``x > 0`` if strict else ``x >= 0``, as the oracle's test.
    """
    x = jnp.asarray(x)
    if width == 0.0:
        return ((x > 0) if strict else (x >= 0)).astype(x.dtype)
    return smoothstep_cubic(x / width + 0.5)


def _max(a, b, width):
    """MAX(a, b): exact for a - b >= width, C1 below (vote 3 W2)."""
    return max_c1(a, b, math.inf if width == 0.0 else 1.0 / width)


def _min(a, b, width):
    return min_c1(a, b, math.inf if width == 0.0 else 1.0 / width)


def _cbrt_c1(x, width):
    """C1 x^(1/3) for x >= 0: exact for x >= width, cubic bridge below (vote 3 Wm).

    The bridge cbrt(width) * (8/3 t^2 - 5/3 t^3), t = x/width, matches value
    and slope at t = 1 and is 0 with zero slope at t = 0.
    """
    if width == 0.0:
        ok = x > 0.0
        return jnp.where(ok, jnp.where(ok, x, 1.0) ** (1.0 / 3.0), 0.0)
    t = jnp.clip(x / width, 0.0, 1.0)
    bridge = width ** (1.0 / 3.0) * t * t * (8.0 / 3.0 - 5.0 / 3.0 * t)  # coeff-ok: C1 Hermite bridge coefficients
    above = x >= width
    return jnp.where(above, jnp.where(above, x, width) ** (1.0 / 3.0), bridge)


def _pbl_height_crossing(sv, geo_full, u, v, sv_bot, cfg, g):
    """First upward level where the bulk Ri exceeds 0.25 and the interpolated
    height (vdfdpbl.F90 / vdfexcu.F90 loop, identical formula).

    Returns (height_m, found).  The crossing level is a discrete pick
    (vote 3 S3); the interpolation is differentiable.
    """
    klev = sv.shape[0]
    du2 = _max(u[:-1] ** 2 + v[:-1] ** 2, _REPDU2, cfg.repdu2_width)
    droro = (sv[:-1] - sv_bot) / (sv[:-1] - geo_full[:-1])
    rilev = (geo_full[:-1] - geo_full[-1]) * droro / du2          # levels 1..KLEV-1
    # ZRIB(JL) is the previous (lower) level's Ri; 0 below level KLEV-1
    riprev = jnp.concatenate([rilev[1:], jnp.zeros((1,), rilev.dtype)])
    above = rilev > _RICRI
    idx = jnp.arange(klev - 1)
    # first crossing going up from KLEV-1 = the largest index with above
    k = jnp.max(jnp.where(above, idx, -1))
    found = k >= 0
    kk = jnp.maximum(k, 0)
    ri_k, ri_p = rilev[kk], riprev[kk]
    den = jnp.where(found, ri_k - ri_p, 1.0)     # no crossing: result discarded (vote 4 G3)
    h = ((ri_k - _RICRI) * geo_full[kk + 1] + (_RICRI - ri_p) * geo_full[kk]) / (den * g)
    return h, found


def ifs_vdfdpbl(u, v, T, q, geo_full, cfg: IFSVDFConfig = IFSVDFConfig(),
                g: float = constants.g, cpd: float = constants.c_pd,
                retv: float = constants.R_v / constants.R_d - 1.0):
    """VDFDPBL: diagnostic PBL height (m), one column (levels top -> surface)."""
    sv = cpd * T * (1.0 + retv * q) + geo_full
    h, found = _pbl_height_crossing(sv, geo_full, u, v, sv[-1] + _EPS_SV, cfg, g)
    return jnp.where(found, h, _PARZI_M)


def _phimu(eta):
    return 1.0 / jnp.sqrt(jnp.sqrt(1.0 - _RCDHALF * eta))


def _phihu(eta):
    return 1.0 / jnp.sqrt(1.0 - _RCDHALF * eta)


def ifs_vdfexcu(u, v, T, q, p_half, geo_full, geo_half, cptgz, hrlw, hrsw, z0m,
                kmfl, khfl, kqfl, khvfl, ri, zinv, kinv, kcbase, kctop, kpbltype,
                cfg: IFSVDFConfig = IFSVDFConfig(), g: float = constants.g,
                rd: float = constants.R_d, cpd: float = constants.c_pd,
                retv: float = constants.R_v / constants.R_d - 1.0,
                lv: float = constants.L_v, rlam: float = _RLAM_M):
    """VDFEXCU, one column.  Integer arguments are 1-based Fortran levels.

    Returns :class:`IFSVDFExcuOutput` with entries 1..KLEV-1 (Python 0..KLEV-2)
    filled and entry KLEV zero (VDFEXCU leaves the surface layer to the caller).
    ``zinv`` is accepted for signature parity (VDFEXCU reads it only in a
    commented-out test) and is unused.
    """
    del zinv
    klev = T.shape[0]
    dt = T.dtype
    rg = 1.0 / g
    rcpd = 1.0 / cpd
    iinv = kinv - 1                                    # vdfexcu.F90: IINV = KINV-1
    # --- 2. mixed-layer scaling ---
    # sqrt_c1 alters sqrt only below REPUST/2, where the MAX returns REPUST
    # exactly; it keeps the gradient finite at KMFL = 0 (vote 4 G2)
    ust = _max(sqrt_c1(kmfl, (0.5 * _REPUST) ** 2), _REPUST, cfg.repust_width_m_s)
    # 1/L for the unstable branch (the stable value L = 1 is never read: the
    # mixed-layer branch needs IINV < KLEV-1, which only an unstable PBL sets)
    inv_l = _RKAP * g * (khvfl - _EPS_FLUX) / (ust ** 3 * T[-1])
    inv_l_cap = 1.0 / _EPS_MACH
    inv_l = jnp.sign(inv_l) * _min(jnp.abs(inv_l), inv_l_cap, cfg.guard_width_frac * inv_l_cap)

    # cloud-top radiative flux jump (KPBLTYPE == 2), K m s-1
    jk = jnp.clip(iinv, 2, klev - 1)                   # Fortran index (>= 2: reads JK-1); KINV >= NJKT5
    hr_k = hrlw[jk - 1] + hrsw[jk - 1]
    hr_km1 = hrlw[jk - 2] + hrsw[jk - 2]
    drad = -_min(hr_k, hr_km1, cfg.hr_min_width) * (geo_half[jk - 1] - geo_half[jk]) * rg
    drad = _max(drad, 0.0, cfg.dradflx_width)
    drad = jnp.where(kpbltype == 2, drad, 0.0)

    # --- 3. stable PBL height (only if KHVFL > 0) ---
    sv = cpd * T * (1.0 + retv * q) + geo_full
    h, found = _pbl_height_crossing(sv, geo_full, u, v, sv[-1] + _EPS_SV, cfg, g)
    pblh = jnp.where(found, h, 0.0) * _step(khvfl, cfg.khvfl_gate_width)

    # --- interfaces JK = 1..KLEV-1 (python i = JK-1): levels JK and JK+1 ---
    JK = jnp.arange(1, klev)
    tu, tl = T[:-1], T[1:]
    qu, ql = q[:-1], q[1:]
    gu, gl = geo_full[:-1], geo_full[1:]
    mgeom = gu - gl
    du2 = _max((u[:-1] - u[1:]) ** 2 + (v[:-1] - v[1:]) ** 2, _REPDU2, cfg.repdu2_width)
    dudz = jnp.sqrt(du2) / mgeom * g
    zzh = 0.5 * rg * (gu + gl) + z0m

    # statically stable: LTG functions with the PBL-height asymptotic length
    # each branch is evaluated unclipped through the blend band and clipped only
    # where its weight is exactly 0, so the blend is C1 (vote 4 G1)
    ri_pos = jnp.maximum(ri[:-1], _RI_STABLE_LO)
    in_pbl = _step(pblh - gu * rg, cfg.pblh_switch_width_m, strict=False)   # z <= h
    lent_pbl = _min(_max(_LMIN_M, _LSTAB_FRAC * pblh, cfg.lmin_width_m), _LMAX_M, cfg.lmax_width_m)
    lent_s = in_pbl * lent_pbl + (1.0 - in_pbl) * _LMIN_M
    klen2_s = _RKAP * zzh * lent_s / (_RKAP * zzh + lent_s)
    scf = jnp.sqrt(1.0 + _LTG_D * ri_pos)
    cfm_s = dudz * klen2_s ** 2 / (1.0 + 2.0 * _LTG_B * ri_pos / scf)
    cfh_s = dudz * klen2_s ** 2 / (1.0 + 2.0 * _LTG_B * ri_pos * scf)
    # statically unstable: Monin-Obukhov functions with eta = Ri, length RLAM
    ri_neg = jnp.minimum(ri[:-1], _RI_UNSTABLE_HI)
    klen2_u = _RKAP * zzh * rlam / (_RKAP * zzh + rlam)
    phim, phih = _phimu(ri_neg), _phihu(ri_neg)
    cfm_u = dudz * klen2_u ** 2 * (1.0 / phim ** 2)
    cfh_u = dudz * klen2_u ** 2 * (1.0 / (phim * phih))
    stab = _step(ri[:-1], cfg.ri_width)                                     # Ri > 0
    cfm = stab * cfm_s + (1.0 - stab) * cfm_u
    cfh = stab * cfh_s + (1.0 - stab) * cfh_u

    # mixed layer (surface and outer layer) below the entrainment level
    mixed = (JK > iinv) & (iinv < klev - 1)
    zz = geo_half[1:klev] * rg                                              # PGEOH(JK)
    zinv_geo = geo_half[jnp.clip(iinv, 0, klev)] * rg
    gz = cfg.guard_width_frac * _EPS_MACH
    dh = zz / _max(zinv_geo, _EPS_MACH, gz)
    dl = jnp.where(mixed, zz * inv_l, 0.0)
    phikh = (1.0 - _PHIK_H * dl) ** (-1.0 / 3.0)
    phikm = (1.0 - _PHIK_M * dl) ** (-1.0 / 3.0)
    znorm = zz * (1.0 - dh) ** 2
    cu_ok = (kpbltype >= 2) & (kcbase > kctop + 1)
    sens = cpd * khfl
    flux_ok = (_step(_FLUX_TEST_W_M2 - sens, cfg.flux_test_width_w_m2)
               * _step(_FLUX_TEST_BOWEN * lv * kqfl + sens, cfg.flux_test_width_w_m2))
    zcb = geo_half[jnp.clip(kcbase - 1, 0, klev)] * rg
    dh2 = _min(1.0, zz / _max(zcb, _EPS_MACH, gz), cfg.cbase_ratio_width)
    znorm_cu = zz * (1.0 - dh) ** 4 + zz * (1.0 - dh2) ** 2
    w_cu = jnp.where(cu_ok, flux_ok, 0.0)
    znorm = w_cu * znorm_cu + (1.0 - w_cu) * znorm
    cfh_ml = _RKAP / phikh * ust * znorm
    cfm_ml = _RKAP / phikm * ust * znorm
    # cloud-top driven K (Lock et al. 2000 eq. 5), stratocumulus only
    zml = zinv_geo
    tl_lev = T[:-1]                                                         # PTM1(JL,JK)
    vsc = _cbrt_c1(g / tl_lev * zml * drad, cfg.cbrt_width)
    zml_s = _max(zml, _EPS_MACH, gz)
    kcld = (_KCLD_COEF * _RKAP * vsc * zz ** 2 / zml_s
            * sqrt_c1(1.0 - zz / zml_s, cfg.sqrt_width))
    kcld = jnp.where(kpbltype == 2, kcld, 0.0)
    cfh = jnp.where(mixed, cfh_ml + kcld, cfh)
    cfm = jnp.where(mixed, cfm_ml + kcld, cfm)

    # top entrainment at the entrainment level (Lock & MacVean 1999 eq. 11)
    entr = (JK == iinv) & (iinv < klev - 1)
    wtv = -_ENTRSFC * khvfl + jnp.where(kpbltype == 2, _ENTRRAD * drad, 0.0)
    wtv = _max(wtv, 0.0, cfg.wtventr_width)
    i_e = jnp.clip(iinv, 2, klev - 1)                                      # Fortran JK = IINV
    dtv = ((cptgz[i_e - 2] - cptgz[i_e - 1]) * rcpd
           + retv * 0.5 * (q[i_e - 2] - q[i_e - 1]) * (T[i_e - 1] + T[i_e - 2]))
    ok = _step(dtv - _DTV_MIN_K, cfg.dtv_width_k)
    dtv_safe = jnp.where(dtv > 0.5 * _DTV_MIN_K, dtv, _DTV_MIN_K)
    k_entr = jnp.where(ok > 0.0, ok * wtv * (geo_full[i_e - 2] - geo_full[i_e - 1]) * rg / dtv_safe, 0.0)
    cfh = jnp.where(entr, k_entr, cfh)
    cfm = jnp.where(entr, k_entr, cfm)

    # safety limit and conversion to VDFDIFH/M units
    cfh = _min(cfh, _KCAP_M2_S, cfg.kcap_width)
    cfm = _min(cfm, _KCAP_M2_S, cfg.kcap_width)
    kh, km = cfh, cfm
    cfnc1 = g * p_half[1:klev] / (0.5 * rd * mgeom * (tu * (1.0 + retv * qu) + tl * (1.0 + retv * ql)))
    z = jnp.zeros((1,), dt)
    return IFSVDFExcuOutput(
        cfm=jnp.concatenate([cfm * cfnc1, z]), cfh=jnp.concatenate([cfh * cfnc1, z]),
        km=jnp.concatenate([km, z]), kh=jnp.concatenate([kh, z]))


def ifs_tofd_coefficient(geo_full, u, v, sigflt, cfg: IFSTOFDConfig = IFSTOFDConfig(),
                         g: float = constants.g):
    """VDFTOFDC: TOFD implicit coefficient (s-1) on (ncol, klev) full levels.

    geo_full: geopotential above the surface (m2 s-2); sigflt: standard
    deviation of the filtered subgrid orography (m), one per column.
    """
    fact2 = cfg.tofd_alpha * _TOFD_BETA * _TOFD_CMD * _TOFD_CORR * _TOFD_IC * _TOFD_FACT1
    coef = fact2 * sigflt[:, None] ** 2
    zz = geo_full / g
    usq = u ** 2 + v ** 2
    w = cfg.repdu2_width_frac * _REPDU2
    uabs = jnp.sqrt(_max(usq, _REPDU2, w))
    zs = jnp.where(zz > 0.0, zz, 1.0)
    c = coef * uabs * jnp.exp(-(zs / _TOFD_IZ_M) ** _TOFD_EXP) * zs ** _TOFD_IN
    return jnp.where(zz > 0.0, c, 0.0) * _step(_TOFD_ZMAX_M - zz, cfg.zmax_width_m, strict=False)
