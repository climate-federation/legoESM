"""O9: BL99/P07 heat diffusion, growth/melt and ice_thd_temp, five categories x (10+5) layers.

The oracle below is an independent per-cell CPython transcription of
``ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/{icethd_zdf_bl99,icethd_dh,icethd,icevar}.f90``
(no model code): its constants are the build's literals, its ``MIN``/``MAX``
follow the record object's emitted ``minsd``/``maxsd`` operand roles, and its
``EXP``/``**`` call glibc through ctypes.  CPython double arithmetic is IEEE
round-to-nearest with no contraction, like the -O3 build without -mfma.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.nemo_namelist import overlay, parse_namelist  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ice import bitz_lipscomb as bl  # noqa: E402
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG  # noqa: E402

DATA = Path(__file__).parent / "data" / "orca2_sas_ice_expref"
_LIBM = ctypes.CDLL("libm.so.6")
_LIBM.exp.argtypes = (ctypes.c_double,)
_LIBM.exp.restype = ctypes.c_double
_LIBM.pow.argtypes = (ctypes.c_double, ctypes.c_double)
_LIBM.pow.restype = ctypes.c_double

# phycst.f90:50,68-76 and eosbn2.f90:1944-1945 (ln_teos10); epsi10/epsi20 from par_ice.
RT0, RHOS, RHOI, RCND_I, RCPI, RLFUS, RTMLT = 273.15, 330.0, 917.0, 2.034396, 2096.7, 0.3333601e06, 0.054  # const-ok: independent oracle transcribes the record build's phycst literals, never the model's constants
RHO0, RCP = 1026.0, 3991.86795711963
EPSI10, EPSI20 = 1.0e-10, 1.0e-20
R1_RHOI, R1_RHOS, R1_RCPI = 1.0 / RHOI, 1.0 / RHOS, 1.0 / RCPI
RDT = 10800.0  # ORCA2_SAS_ICE rn_Dt x nn_fsbc
JPL, NLAY_I, NLAY_S = 5, 10, 5
NPTI = (7, 3, 11, 1, 5)
FLUXES = list(bl.SI3DHFluxes._fields)


# Per-site hooks for the survivor tests: SWAP flips one site's emitted roles (a plant);
# TIES counts ties whose operands differ in bits (signed zeros) or carry a NaN.
SWAP: set = set()
TIES: dict = {}


def _tie(dst, src, site):
    if site is not None and (dst != dst or src != src
                             or (dst == src and math.copysign(1.0, dst) != math.copysign(1.0, src))):
        TIES[site] = TIES.get(site, 0) + 1
    return (src, dst) if site in SWAP else (dst, src)


def MAXSD(dst, src, site=None):  # noqa: N802 - x86 `maxsd src, dst`
    dst, src = _tie(dst, src, site)
    return dst if dst > src else src


def MINSD(dst, src, site=None):  # noqa: N802 - x86 `minsd src, dst`
    dst, src = _tie(dst, src, site)
    return dst if dst < src else src


def SUM(xs):  # noqa: N802
    acc = 0.0
    for x in xs:
        acc = acc + x
    return acc


def _deck():
    v = overlay(parse_namelist(DATA / "namelist_ice_ref", "ice"), parse_namelist(DATA / "namelist_ice_cfg", "ice"))

    def g(group, name):
        value = v[("ice", group, name.lower())][0]
        if name.startswith(("nn_", "jpl", "nlay")):  # INTEGER in ice.f90/par_ice.f90; the parser yields floats
            assert float(value).is_integer(), (name, value)
            return int(value)
        return value

    return {
        "jpl": g("nampar", "jpl"), "nlay_i": g("nampar", "nlay_i"), "nlay_s": g("nampar", "nlay_s"),
        "ln_virtual_itd": g("nampar", "ln_virtual_itd"),
        "rn_cnd_s": g("namthd_zdf", "rn_cnd_s"), "rn_kappa_i": g("namthd_zdf", "rn_kappa_i"),
        "rn_kappa_s": g("namthd_zdf", "rn_kappa_s"), "ln_cndi_P07": g("namthd_zdf", "ln_cndi_P07"),
        "ln_zdf_chkcvg": g("namthd_zdf", "ln_zdf_chkcvg"),
        "nn_qtrice": g("namsbc", "nn_qtrice"), "nn_snwfra": g("namsbc", "nn_snwfra"),
        "ln_cndflx": g("namsbc", "ln_cndflx"), "ln_cndemulate": g("namsbc", "ln_cndemulate"),
        "rn_snwblow": g("namsbc", "rn_snwblow"), "ln_icedH": g("namthd", "ln_icedH"),
        "rn_sinew": g("namthd_sal", "rn_sinew"), "nn_icesal": g("namthd_sal", "nn_icesal"),
    }


# --------------------------------------------------------------------------- oracle
def oracle_zdf(c, nml):
    """ice_thd_zdf_BL99(np_cnd_OFF) for one cell (icethd_zdf_bl99.f90:148-879)."""
    n, m = len(c["t_i"]), len(c["t_s"])
    rcnd_s, kap_i, kap_s = nml["rn_cnd_s"], nml["rn_kappa_i"], nml["rn_kappa_s"]
    r1n, r1m = 1.0 / float(n), 1.0 / float(m)
    zg1s = zg1 = 2.0
    zgamma, zkimin, ztsu_err, zdti_bnd, zhs_ssl, zhi_ssl, zh_min = 18009.0, 0.10, 1.e-5, 1.e-4, 0.03, 0.10, 1.e-3
    h_i, h_s, a_i, t_bo = c["h_i"], c["h_s"], c["a_i"], c["t_bo"]
    sz = [None] + list(c["sz_i"])
    t_i = [None] + list(c["t_i"])
    t_s = [None] + list(c["t_s"])
    qns, qsr, dqns, qtr = c["qns_ice"], c["qsr_ice"], c["dqns_ice"], c["qtr_ice_top"]
    zq_ini = SUM(c["e_i"]) * h_i * r1n + SUM(c["e_s"]) * h_s * r1m
    za = h_s / (h_s + 0.02)
    zraext = kap_s
    if h_i > 0.0:
        zh_i = MAXSD(h_i, zh_min, 'zdf:173') * r1n
        z1_h_i = 1.0 / zh_i
    else:
        zh_i = z1_h_i = 0.0
    if h_s > 0.0:
        zh_s = MAXSD(h_s, zh_min, 'zdf:181') * r1m
        z1_h_s = 1.0 / zh_s
        isnow = 1.0
    else:
        zh_s = z1_h_s = isnow = 0.0
    isnow_comb = h_s / zh_min if h_s < zh_min else 1.0
    t_su = c["t_su"]
    ztsub = t_su
    t_su = MINSD(t_su, RT0 - ztsu_err, 'zdf:202')
    zdqns_b, zqns_b = dqns, qns
    ztsold, ztiold = list(t_s), list(t_i)
    trs = [qtr] + [0.0] * m
    abs_ = [None] + [0.0] * m
    for jk in range(1, m + 1):
        trs[jk] = trs[0] * _LIBM.exp(-(zraext * MAXSD(zh_s * float(jk) - zhs_ssl, 0.0, 'zdf:219')))
        abs_[jk] = trs[jk - 1] - trs[jk]
    tri = [trs[m] * za + qtr * (1.0 - za)] + [0.0] * n
    abi = [None] + [0.0] * n
    for jk in range(1, n + 1):
        tri[jk] = (za * trs[m] * _LIBM.exp(-(kap_i * MAXSD(zh_i * float(jk) - zh_min, 0.0, 'zdf:229.max1')))
                   + (1.0 - za) * qtr * _LIBM.exp(-(kap_i * MAXSD(zh_i * float(jk) - zhi_ssl, 0.0, 'zdf:229.max2'))))
        abi[jk] = tri[jk - 1] - tri[jk]
    tcond = [0.0] * (n + 1)
    ks = [0.0] * (m + 1)
    ki = [0.0] * (n + 1)
    comb = 0.0
    eta_i = [None] + [0.0] * n
    eta_s = [None] + [0.0] * m
    conv = False
    iconv = 0
    while (not conv) and iconv < 200:
        iconv += 1
        cp = [0.0] * (n + 1)
        cp[0] = RCND_I + 0.09 * sz[1] / MINSD(t_i[1] - RT0, -EPSI10, 'zdf:274') - 0.011 * (t_i[1] - RT0)
        cp[n] = RCND_I + 0.09 * sz[n] / MINSD(t_bo - RT0, -EPSI10, 'zdf:276') - 0.011 * (t_bo - RT0)
        for jk in range(1, n):
            cp[jk] = (RCND_I + 0.09 * 0.5 * (sz[jk] + sz[jk + 1])
                      / MINSD(0.5 * (t_i[jk] + t_i[jk + 1]) - RT0, -EPSI10, 'zdf:281')
                      - 0.011 * (0.5 * (t_i[jk] + t_i[jk + 1]) - RT0))
        tcond = [MAXSD(x, zkimin, 'zdf:292') for x in cp]
        zghe = 1.0
        for jk in range(m):
            ks[jk] = zghe * rcnd_s * z1_h_s
        ks[m] = isnow * zghe * rcnd_s * tcond[0] / (0.5 * (tcond[0] * zh_s + rcnd_s * zh_i))
        for jk in range(n + 1):
            ki[jk] = zghe * tcond[jk] * z1_h_i
        comb = isnow_comb * ks[0] + (1.0 - isnow_comb) * ki[0]
        if h_s > 0.0:
            ki[0] = ks[m]
        for jk in range(1, n + 1):
            zcpi = RCPI + zgamma * sz[jk] / MAXSD((t_i[jk] - RT0) * (ztiold[jk] - RT0), EPSI10, 'zdf:353')
            eta_i[jk] = RDT * R1_RHOI * z1_h_i / zcpi
        for jk in range(1, m + 1):
            eta_s[jk] = RDT * R1_RHOS * R1_RCPI * z1_h_s
        qns = qns + dqns * (t_su - ztsub)
        zfnet = qsr - qtr + qns
        ztib, ztsb = list(t_i), list(t_s)
        N = n + m + 1
        tr = [[0.0, 0.0, 0.0, 0.0] for _ in range(N + 1)]
        ind = [0.0] * (N + 1)
        indb = [0.0] * (N + 1)
        diagb = [0.0] * (N + 1)
        for jm in range(m + 2, m + n + 1):
            jk = jm - m - 1
            tr[jm][1] = - eta_i[jk] * ki[jk - 1]
            tr[jm][2] = 1.0 + eta_i[jk] * (ki[jk - 1] + ki[jk])
            tr[jm][3] = - eta_i[jk] * ki[jk]
            ind[jm] = ztiold[jk] + eta_i[jk] * abi[jk]
        jm = m + n + 1
        tr[jm][1] = - eta_i[n] * ki[n - 1]
        tr[jm][2] = 1.0 + eta_i[n] * (ki[n - 1] + ki[n] * zg1)
        tr[jm][3] = 0.0
        ind[jm] = ztiold[n] + eta_i[n] * (abi[n] + ki[n] * zg1 * t_bo)
        if h_s > 0.0:
            for jm in range(3, m + 2):
                jk = jm - 1
                tr[jm][1] = - eta_s[jk] * ks[jk - 1]
                tr[jm][2] = 1.0 + eta_s[jk] * (ks[jk - 1] + ks[jk])
                tr[jm][3] = - eta_s[jk] * ks[jk]
                ind[jm] = ztsold[jk] + eta_s[jk] * abs_[jk]
            if t_su < RT0:
                jm_min, jm_max = 1, n + m + 1
                tr[1][1] = 0.0
                tr[1][2] = zdqns_b - zg1s * ks[0]
                tr[1][3] = zg1s * ks[0]
                ind[1] = zdqns_b * t_su - zfnet
                tr[2][1] = - eta_s[1] * ks[0] * zg1s
                tr[2][2] = 1.0 + eta_s[1] * (ks[1] + ks[0] * zg1s)
                tr[2][3] = - eta_s[1] * ks[1]
                ind[2] = ztsold[1] + eta_s[1] * abs_[1]
            else:
                jm_min, jm_max = 2, n + m + 1
                tr[2][1] = 0.0
                tr[2][2] = 1.0 + eta_s[1] * (ks[1] + ks[0] * zg1s)
                tr[2][3] = - eta_s[1] * ks[1]
                ind[2] = ztsold[1] + eta_s[1] * (abs_[1] + ks[0] * zg1s * t_su)
        else:
            if t_su < RT0:
                jm_min, jm_max = m + 1, n + m + 1
                tr[jm_min][1] = 0.0
                tr[jm_min][2] = zdqns_b - ki[0] * zg1
                tr[jm_min][3] = ki[0] * zg1
                ind[jm_min] = zdqns_b * t_su - zfnet
                tr[jm_min + 1][1] = - eta_i[1] * ki[0] * zg1
                tr[jm_min + 1][2] = 1.0 + eta_i[1] * (ki[1] + ki[0] * zg1)
                tr[jm_min + 1][3] = - eta_i[1] * ki[1]
                ind[jm_min + 1] = ztiold[1] + eta_i[1] * abi[1]
            else:
                jm_min, jm_max = m + 2, n + m + 1
                tr[jm_min][1] = 0.0
                tr[jm_min][2] = 1.0 + eta_i[1] * (ki[1] + ki[0] * zg1)
                tr[jm_min][3] = - eta_i[1] * ki[1]
                ind[jm_min] = ztiold[1] + eta_i[1] * (abi[1] + ki[0] * zg1 * t_su)
        indb[jm_min] = ind[jm_min]
        diagb[jm_min] = tr[jm_min][2]
        for jm in range(jm_min + 1, jm_max + 1):
            diagb[jm] = tr[jm][2] - tr[jm][1] * tr[jm - 1][3] / diagb[jm - 1]
            indb[jm] = ind[jm] - tr[jm][1] * indb[jm - 1] / diagb[jm - 1]
        t_i[n] = indb[jm_max] / diagb[jm_max]
        for jm in range(n + m, m + 1, -1):
            jk = jm - m - 1
            t_i[jk] = (indb[jm] - tr[jm][3] * t_i[jk + 1]) / diagb[jm]
        if h_s > 0.0:
            t_s[m] = (indb[m + 1] - tr[m + 1][3] * t_i[1]) / diagb[m + 1]
        for jm in range(m, 1, -1):
            jk = jm - 1
            if h_s > 0.0:
                t_s[jk] = (indb[jm] - tr[jm][3] * t_s[jk + 1]) / diagb[jm]
        ztsub = t_su
        if t_su < RT0:
            t_su = (indb[jm_min] - tr[jm_min][3] * (isnow * t_s[1] + (1.0 - isnow) * t_i[1])) / diagb[jm_min]
        zdti = 0.0
        t_su = MAXSD(MINSD(t_su, RT0, 'zdf:577.min'), RT0 - 100.0, 'zdf:577.max')
        zdti = max(zdti, abs(t_su - ztsub))
        if h_s > 0.0:
            for jk in range(1, m + 1):
                t_s[jk] = MAXSD(MINSD(t_s[jk], RT0, 'zdf:582.min'), RT0 - 100.0, 'zdf:582.max')
                zdti = max(zdti, abs(t_s[jk] - ztsb[jk]))
        for jk in range(1, n + 1):
            ztm = -(RTMLT * sz[jk]) + RT0
            t_i[jk] = MAXSD(MINSD(ztm, t_i[jk], 'zdf:589.min'), RT0 - 100.0, 'zdf:589.max')
            zdti = max(zdti, abs(t_i[jk] - ztib[jk]))
        if zdti < zdti_bnd:
            conv = True
    out = {}
    out["qcn_ice_bot"] = - ki[n] * zg1 * (t_bo - t_i[n])
    out["qcn_ice_top"] = - isnow * ks[0] * zg1s * (t_s[1] - t_su) - (1.0 - isnow) * ki[0] * zg1 * (t_i[1] - t_su)
    hfx_err_dif = c["hfx_err_dif"] - (qns - zqns_b) * a_i
    e_i = [None] + [0.0] * n
    for jk in range(1, n + 1):
        ztm = -(RTMLT * sz[jk])
        t_i[jk] = MINSD(ztm + RT0, t_i[jk], 'var:955')
        e_i[jk] = RHOI * (RCPI * (ztm - (t_i[jk] - RT0)) + RLFUS * MAXSD(1.0 - ztm / MINSD(t_i[jk] - RT0, -EPSI10, 'var:957.min'), 0.0, 'var:957.max')
                          - RCP * ztm)
    e_s = [None] + [RHOS * (RCPI * (RT0 - t_s[jk]) + RLFUS) for jk in range(1, m + 1)]
    zdq = -zq_ini + (SUM(e_i[1:]) * h_i * r1n + SUM(e_s[1:]) * h_s * r1m)
    if t_su < RT0:
        zhfx = (qns + qsr - tri[n] - out["qcn_ice_bot"] + zdq * (1.0 / RDT)) * a_i
    else:
        zhfx = (out["qcn_ice_top"] + qtr - tri[n] - out["qcn_ice_bot"] + zdq * (1.0 / RDT)) * a_i
    out["hfx_err_dif"] = hfx_err_dif + zhfx
    out["hfx_dif"] = c["hfx_dif"] - zdq * (1.0 / RDT) * a_i
    out["cnd_ice"] = 2.0 * comb if h_i >= zhi_ssl else 2.0 * tcond[0] / zhi_ssl
    out["t1_ice"] = isnow * t_s[1] + (1.0 - isnow) * t_i[1]
    if h_s >= zhs_ssl:
        out["t_si"] = ((rcnd_s * h_i * r1n * t_s[m] + tcond[1] * h_s * r1m * t_i[1])
                       / (rcnd_s * h_i * r1n + tcond[1] * h_s * r1m))
    else:
        out["t_si"] = t_su
    out.update(t_su=t_su, t_i=t_i[1:], t_s=t_s[1:], e_i=e_i[1:], e_s=e_s[1:], qns_ice=qns, qtr_ice_bot=tri[n],
               n_picard=iconv)
    return out


def _remap(ph, pe_or_content, nl, products, site):
    """snw_ent (products=True, icethd_dh.f90:584-619) / ice_var_vremap (icevar.f90:1010-1046)."""
    k0 = len(ph)
    zc0, zh0 = [0.0] * (k0 + 1), [0.0] * (k0 + 1)
    for j in range(1, k0 + 1):
        zc0[j] = zc0[j - 1] + (pe_or_content[j - 1] * ph[j - 1] if products else pe_or_content[j - 1])
        zh0[j] = zh0[j - 1] + ph[j - 1]
    zhnew = SUM(ph) * (1.0 / float(nl))
    zh1 = [0.0] * (nl + 1)
    for j in range(1, nl + 1):
        zh1[j] = zh1[j - 1] + zhnew
    zc1 = [0.0] * (nl + 1)
    for j0 in range(1, k0 + 1):
        for j1 in range(1, nl):
            if zh1[j1] <= zh0[j0] and zh1[j1] > zh0[j0 - 1]:
                zc1[j1] = (zc0[j0 - 1] * (zh0[j0] - zh1[j1]) + zc0[j0] * (zh1[j1] - zh0[j0 - 1])) / (zh0[j0] - zh0[j0 - 1])
    zc1[nl] = zc0[k0]
    return [MAXSD(zc1[j] - zc1[j - 1], 0.0, site + '.max1') / MAXSD(zhnew, EPSI20, site + '.max2') for j in range(1, nl + 1)]


def oracle_dh(c, z, nml):
    """ice_thd_dh for one cell (icethd_dh.f90:102-541), from the ZDF exit ``z``."""
    n, m = len(c["t_i"]), len(c["t_s"])
    r1n, r1m, r1dt = 1.0 / float(n), 1.0 / float(m), 1.0 / RDT
    a_i, at_i, sss, sst, t_bo = c["a_i"], c["at_i"], c["sss"], c["sst"], c["t_bo"]
    fx = {k: c["flux_" + k] for k in FLUXES}
    dh = dict.fromkeys(bl.SI3DHIncrements._fields, 0.0)
    zsw = 1.0
    zsnw = 1.0 - _LIBM.pow(1.0 - at_i, nml["rn_snwblow"])
    z1_rho = 1.0 / (RHOS + RHO0 - RHOI)
    t_su, h_i, h_s = z["t_su"], c["h_i"], c["h_s"]
    t_i, t_s, e_i, e_s = [None] + z["t_i"], [None] + z["t_s"], [None] + z["e_i"], [None] + z["e_s"]
    sz = [None] + list(c["sz_i"])
    qml = (z["qns_ice"] + c["qsr_ice"] - c["qtr_ice_top"] - z["qcn_ice_top"]) if t_su >= RT0 else 0.0
    zq_top = MAXSD(qml * RDT, 0.0, 'dh:128')
    zf_tt = z["qcn_ice_bot"] + c["qsb_ice_bot"] + c["fhld"] + z["qtr_ice_bot"] * c["frq_m"]
    zq_bot = MAXSD(zf_tt * RDT, 0.0, 'dh:130')
    zs = sz if nml["nn_icesal"] == 4 else [None] + [c["s_i"]] * n
    zs_o, ze_o, zh_o, zh = ([0.0] * (n + 2) for _ in range(4))
    for jk in range(1, n + 1):
        zs_o[jk] = h_i * r1n * zs[jk]
        ze_o[jk] = h_i * r1n * e_i[jk]
        zh_o[jk] = h_i * r1n
        zh[jk] = h_i * r1n
    zhs, zes = [0.0] * (m + 1), [0.0] * (m + 1)
    for jk in range(1, m + 1):
        zhs[jk] = h_s * r1m
        zes[jk] = e_s[jk]
    for jk in range(1, m + 1):
        if t_s[jk] > RT0:
            fx["hfx_res"] = fx["hfx_res"] - zes[jk] * zhs[jk] * a_i * r1dt
            fx["wfx_snw_sum"] = fx["wfx_snw_sum"] + RHOS * zhs[jk] * a_i * r1dt
            dh["dh_s_itm"] = dh["dh_s_itm"] - zhs[jk]
            h_s = MAXSD(h_s - zhs[jk], 0.0, 'dh:170')
            zhs[jk] = zes[jk] = 0.0
    if c["sprecip"] > 0.0:
        zhs[0] = zsnw * c["sprecip"] * RDT * R1_RHOS / at_i
        zes[0] = MAXSD(- c["qprec_ice"], 0.0, 'dh:180')
        fx["hfx_spr"] = fx["hfx_spr"] + zes[0] * zhs[0] * a_i * r1dt
        fx["wfx_spr"] = fx["wfx_spr"] - RHOS * zhs[0] * a_i * r1dt
        h_s = h_s + zhs[0]
    zdeltah = -MINSD(c["evap_ice"] * R1_RHOS * RDT, h_s, 'dh:194')   # emitted -MIN(A, h_s)
    zevap = c["evap_ice"] * RDT + zdeltah * RHOS
    if zdeltah > 0.0 and zes[0] == 0.0:
        zes[0] = RHOS * (RLFUS - RCPI * (t_su - RT0))
    for jk in range(0, m + 1):
        zdum = MAXSD(-zhs[jk], zdeltah, 'dh:200')
        fx["hfx_sub"] = fx["hfx_sub"] + zes[jk] * zdum * a_i * r1dt
        fx["wfx_snw_sub"] = fx["wfx_snw_sub"] - RHOS * zdum * a_i * r1dt
        h_s = MAXSD(h_s + zdum, 0.0, 'dh:206')
        zhs[jk] = MAXSD(zhs[jk] + zdum, 0.0, 'dh:207')
        zdeltah = MINSD(zdeltah - zdum, 0.0, 'dh:211')
    for jk in range(0, m + 1):
        if zhs[jk] > 0.0 and zq_top > 0.0:
            zdum = -MINSD(zq_top / MAXSD(EPSI20, zes[jk], 'dh:221'), zhs[jk], 'dh:228')   # emitted -MIN(zq/e, zh_s)
            fx["hfx_snw"] = fx["hfx_snw"] - zes[jk] * zdum * a_i * r1dt
            fx["wfx_snw_sum"] = fx["wfx_snw_sum"] - RHOS * zdum * a_i * r1dt
            dh["dh_s_sum"] = dh["dh_s_sum"] + zdum
            zq_top = MAXSD(zq_top + zdum * zes[jk], 0.0, 'dh:229')
            h_s = MAXSD(h_s + zdum, 0.0, 'dh:230')
            zhs[jk] = MAXSD(zhs[jk] + zdum, 0.0, 'dh:231')
    icount = [None] + [0] * n
    # surface (icethd_dh.f90:251-290) and basal (:388-427) melt share this body
    _SITE = {"sum": ("dh:251", "dh:272", "dh:274", "dh:289", "dh:290"),
             "bom": ("dh:388", "dh:409", "dh:411", "dh:426", "dh:427")}

    def layer(jk, heat, kind):
        nonlocal h_i
        ztm = - RTMLT * sz[jk]
        if t_i[jk] >= (ztm + RT0):
            zEi = - e_i[jk] * R1_RHOI
            zdum = MINSD(- zh[jk], 0.0, _SITE[kind][0])
            zfmdt = - zdum * RHOI
            dh["dh_i_itm"] = dh["dh_i_itm"] + zdum
            fx["hfx_res"] = fx["hfx_res"] + zEi * zfmdt * a_i * r1dt
            fx["wfx_res"] = fx["wfx_res"] - RHOI * zdum * a_i * r1dt
            fx["sfx_res"] = fx["sfx_res"] - RHOI * zdum * zs[jk] * a_i * r1dt
        else:
            zEi = - e_i[jk] * R1_RHOI
            zEw = RCP * ztm
            zdE = zEi - zEw
            zfmdt = - heat / zdE
            zdum = - zfmdt * R1_RHOI
            zdum = MINSD(MAXSD(zdum, - zh[jk], _SITE[kind][1] + '.max'), 0.0, _SITE[kind][1] + '.min')
            heat = MAXSD(heat - zdum * RHOI * zdE, 0.0, _SITE[kind][2])
            dh["dh_i_" + kind] = dh["dh_i_" + kind] + zdum
            zfmdt = - RHOI * zdum
            fx["hfx_thd"] = fx["hfx_thd"] + zEw * zfmdt * a_i * r1dt
            fx["hfx_" + kind] = fx["hfx_" + kind] - zdE * zfmdt * a_i * r1dt
            fx["wfx_" + kind] = fx["wfx_" + kind] - RHOI * zdum * a_i * r1dt
            fx["sfx_" + kind] = fx["sfx_" + kind] - RHOI * zdum * zs[jk] * a_i * r1dt
        zh[jk] = MAXSD(zh[jk] + zdum, 0.0, _SITE[kind][3])
        h_i = MAXSD(h_i + zdum, 0.0, _SITE[kind][4])
        zs_o[jk] = zs_o[jk] + zdum * zs[jk]
        ze_o[jk] = ze_o[jk] + zdum * e_i[jk]
        zh_o[jk] = zh_o[jk] + zdum
        return heat

    for jk in range(1, n + 1):
        zq_top = layer(jk, zq_top, "sum")
        zdum = -MINSD(zevap * R1_RHOI, zh[jk], 'dh:295')   # emitted -MIN(zevap*r1, zh_i)
        fx["hfx_sub"] = fx["hfx_sub"] + e_i[jk] * zdum * a_i * r1dt
        fx["wfx_ice_sub"] = fx["wfx_ice_sub"] - RHOI * zdum * a_i * r1dt
        fx["sfx_sub"] = fx["sfx_sub"] - RHOI * zdum * zs[jk] * a_i * r1dt
        zevap = zevap + zdum * RHOI
        zh[jk] = MAXSD(zh[jk] + zdum, 0.0, 'dh:310')
        h_i = MAXSD(h_i + zdum, 0.0, 'dh:311')
        dh["dh_i_sub"] = dh["dh_i_sub"] + zdum
        zs_o[jk] = zs_o[jk] + zdum * zs[jk]
        ze_o[jk] = ze_o[jk] + zdum * e_i[jk]
        zh_o[jk] = zh_o[jk] + zdum
        icount[jk] = 0 if zh[jk] > 0.0 else 1
    fx["wfx_err_sub"] = fx["wfx_err_sub"] - zevap * a_i * r1dt
    zs_new = 0.0
    if zf_tt < 0.0:
        zs_new = zsw * nml["rn_sinew"] * sss + (1. - zsw) * zs[1]
        ztm = - RTMLT * zs_new
        zt_new = zsw * t_bo + (1. - zsw) * t_i[n]
        zEi = (RCPI * (zt_new - (ztm + RT0)) - RLFUS * MAXSD(1.0 - ztm / (MINSD(zt_new - RT0, -EPSI10, 'dh:348.min')), 0.0, 'dh:348.max')
               + RCP * ztm)
        zEw = RCP * (t_bo - RT0)
        zdE = zEi - zEw
        dh["dh_i_bog"] = RDT * MAXSD(zf_tt / (zdE * RHOI), 0.0, 'dh:355')
        zfmdt = - RHOI * dh["dh_i_bog"]
        fx["hfx_thd"] = fx["hfx_thd"] + zEw * zfmdt * a_i * r1dt
        fx["hfx_bog"] = fx["hfx_bog"] - zdE * zfmdt * a_i * r1dt
        fx["wfx_bog"] = fx["wfx_bog"] - RHOI * dh["dh_i_bog"] * a_i * r1dt
        fx["sfx_bog"] = fx["sfx_bog"] - RHOI * dh["dh_i_bog"] * zs_new * a_i * r1dt
        zh[n + 1] = zh[n + 1] + dh["dh_i_bog"]
        h_i = h_i + dh["dh_i_bog"]
        zs_o[n + 1] = zs_o[n + 1] + dh["dh_i_bog"] * zs_new
        ze_o[n + 1] = ze_o[n + 1] + dh["dh_i_bog"] * (-zEi * RHOI)
        zh_o[n + 1] = zh_o[n + 1] + dh["dh_i_bog"]
    for jk in range(n, 0, -1):
        if zf_tt > 0.0 and jk > icount[jk]:
            zq_bot = layer(jk, zq_bot, "bom")
    if h_i == 0.0:
        for jk in range(0, m + 1):
            fx["hfx_res"] = fx["hfx_res"] - zes[jk] * zhs[jk] * a_i * r1dt
            fx["wfx_res"] = fx["wfx_res"] + RHOS * zhs[jk] * a_i * r1dt
            h_s = 0.0
            zes[jk] = zhs[jk] = 0.0
    dsni = MAXSD((RHOS * h_s + (RHOI - RHO0) * h_i) * z1_rho, 0.0, 'dh:456')
    dh["dh_snowice"] = dsni
    h_i = h_i + dsni
    h_s = h_s - dsni
    zfmdt = (RHOS - RHOI) * dsni
    zEw = RCP * sst
    fx["hfx_thd"] = fx["hfx_thd"] + zEw * zfmdt * a_i * r1dt
    fx["sfx_sni"] = fx["sfx_sni"] + sss * zfmdt * a_i * r1dt
    fx["wfx_sni"] = fx["wfx_sni"] - dsni * RHOI * a_i * r1dt
    fx["wfx_snw_sni"] = fx["wfx_snw_sni"] + dsni * RHOS * a_i * r1dt
    zh[0] = zh[0] + dsni
    zdeltah = dsni
    zs_o[0] = zs_o[0] - zfmdt * sss * R1_RHOI
    zh_o[0] = zh_o[0] + dsni
    ze_o[0] = ze_o[0] + zfmdt * zEw
    for jk in range(m, -1, -1):
        zdum = MINSD(zhs[jk], zdeltah, 'dh:490')   # emitted `minsd zdeltah, zh_s`
        zhs[jk] = MAXSD(zhs[jk] - zdum, 0.0, 'dh:491')
        ze_o[0] = ze_o[0] + zdum * zes[jk]
        zdeltah = MAXSD(zdeltah - zdum, 0.0, 'dh:494')
    e_s_new = _remap(zhs, zes, m, products=True, site="dh:619")
    if h_s > 0.0:
        t_s_new = [RT0 + (- e * R1_RHOS * R1_RCPI + RLFUS * R1_RCPI) for e in e_s_new]
    else:
        t_s_new = [RT0] * m
    e_i_new = _remap(zh_o, ze_o, n, products=False, site="var:1045")
    s_i = c["s_i"]
    if nml["nn_icesal"] == 4:
        sz_new = _remap(zh_o, zs_o, n, products=False, site="var:1045")
    else:
        sz_new = list(c["sz_i"])
        zs_sni = sss * (RHOI - RHOS) * R1_RHOI
        zds = (zs_sni - s_i) * dsni / MAXSD(h_i, EPSI10, 'dh:525')
        zds = zds + (zs_new - s_i) * dh["dh_i_bog"] / MAXSD(h_i, EPSI10, 'dh:525')
        s_i = s_i + zds
    a_out = a_i
    if h_i == 0.0:
        a_out, h_s, t_su = 0.0, 0.0, RT0
    return dict(h_i=h_i, h_s=h_s, t_su=t_su, a_i=a_out, e_i=e_i_new, e_s=e_s_new, t_s=t_s_new,
                sz_i=sz_new, s_i=s_i, qml_ice=qml, dh=dh, fluxes=fx)


def oracle_temp(h_i, e_i, sz):
    """ice_thd_temp (icethd.f90:279-291)."""
    out = []
    for e, s in zip(e_i, sz):
        if h_i > 0.0:
            ztm = - RTMLT * s
            zbbb = (RCP - RCPI) * ztm + e * R1_RHOI - RLFUS
            zccc = math.sqrt(MAXSD(zbbb * zbbb - 4.0 * RCPI * RLFUS * ztm, 0.0, 'thd:285'))
            out.append(RT0 - (zbbb + zccc) * 0.5 * R1_RCPI)
        else:
            out.append(RT0)
    return out


# --------------------------------------------------------------------------- inputs
KINDS = ("cold", "melt", "nosnow", "thin", "flood", "gone", "deposit", "warmsnow", "brine")


def _cell(rng, kind):
    n, m = NLAY_I, NLAY_S
    h_i = {"cold": 1.6, "melt": 0.9, "nosnow": 2.2, "thin": 0.04, "flood": 0.3, "gone": 0.012,
           "deposit": 1.1, "warmsnow": 0.7, "brine": 0.5}[kind] * (1.0 + 0.1 * rng.random())
    h_s = {"cold": 0.3, "melt": 0.08, "nosnow": 0.0, "thin": 0.002, "flood": 0.55, "gone": 0.0,
           "deposit": 0.0, "warmsnow": 0.0, "brine": 0.05}[kind] * (1.0 + 0.1 * rng.random())
    t_bo = RT0 - 1.8 - 0.1 * rng.random()
    t_su = {"melt": RT0 - 0.01, "thin": RT0 - 2.0, "gone": RT0 - 0.5}.get(kind, RT0 - 15.0 - 10.0 * rng.random())
    sz = [MAXSD(0.1, 2.0 + 4.0 * rng.random() * (k + 1) / n) for k in range(n)]
    if kind == "brine":  # melting point below t_bo near the base: ZDF clamps there, DH melts internally
        sz = [2.0 + 38.0 * (k + 1) / n for k in range(n)]
    t_i = [t_su + (t_bo - t_su) * (k + 0.5) / n for k in range(n)]
    t_i = [min(t, RT0 - RTMLT * s - 1.0e-3) for t, s in zip(t_i, sz)]
    if kind == "brine":  # lower layers enter at their melting point under a warmer base
        t_i[n - 3:] = [-(RTMLT * s) + RT0 for s in sz[n - 3:]]
    t_s = [t_su + (t_i[0] - t_su) * (k + 0.5) / m if h_s > 0 else RT0 for k in range(m)]
    if kind == "warmsnow":
        t_s = [RT0 + 0.5] * m
    e_i = []
    for t, s in zip(t_i, sz):
        ztm = -RTMLT * s
        e_i.append(RHOI * (RCPI * (ztm - (t - RT0)) + RLFUS * MAXSD(1.0 - ztm / MINSD(t - RT0, -EPSI10), 0.0) - RCP * ztm))
    e_s = [RHOS * (RCPI * (RT0 - t) + RLFUS) for t in t_s]
    a_i = 0.1 + 0.18 * rng.random()
    c = dict(h_i=h_i, h_s=h_s, t_su=t_su, t_i=t_i, t_s=t_s, e_i=e_i, e_s=e_s, sz_i=sz, a_i=a_i,
             at_i=min(0.997, 5 * a_i), t_bo=t_bo, s_i=SUM(sz) / n, sss=34.0 + rng.random(),
             sst=-1.7 - 0.1 * rng.random())
    c["qns_ice"] = {"melt": 260.0, "thin": 150.0, "gone": 40.0}.get(kind, -60.0 - 20.0 * rng.random())
    c["qsr_ice"] = {"melt": 180.0, "thin": 90.0}.get(kind, 20.0 * rng.random())
    c["dqns_ice"] = -(8.0 + 4.0 * rng.random())
    c["qtr_ice_top"] = {"nosnow": 12.0, "thin": 30.0, "cold": 0.0}.get(kind, 25.0 * rng.random())
    c["qsb_ice_bot"] = {"gone": 600.0, "melt": 15.0}.get(kind, -5.0 * rng.random())
    c["fhld"] = {"gone": 900.0}.get(kind, 0.0)
    c["frq_m"] = 1.0
    c["sprecip"] = {"cold": 2.0e-5, "melt": 1.0e-6}.get(kind, 0.0)
    c["qprec_ice"] = {"cold": -RHOS * (RLFUS + 5.0 * RCPI)}.get(kind, 0.0)
    c["evap_ice"] = {"deposit": -3.0e-6, "cold": 1.0e-6}.get(kind, 2.0e-6 * rng.random())
    c["hfx_err_dif"] = 0.25 * rng.standard_normal()
    c["hfx_dif"] = -0.0
    for k in FLUXES:
        c["flux_" + k] = [-0.0, 0.0, 1.0e-3 * rng.standard_normal()][int(rng.integers(3))]
    return c


def _categories(seed=0):
    rng = np.random.default_rng(seed)
    return [[_cell(rng, KINDS[(jl + j) % len(KINDS)]) for j in range(npti)] for jl, npti in enumerate(NPTI)]


ARRAY_KEYS = ("h_i", "h_s", "t_su", "t_i", "t_s", "e_i", "e_s", "sz_i", "a_i", "at_i", "t_bo", "s_i", "sss",
              "sst", "qns_ice", "qsr_ice", "dqns_ice", "qtr_ice_top", "qsb_ice_bot", "fhld", "frq_m", "sprecip",
              "qprec_ice", "evap_ice", "hfx_err_dif", "hfx_dif")


def _entry(cells):
    cpu = jax.devices("cpu")[0]
    entry = {k: jax.device_put(jnp.asarray(np.array([c[k] for c in cells], dtype=np.float64)), cpu)
             for k in ARRAY_KEYS}
    fluxes = bl.SI3DHFluxes(*(jax.device_put(jnp.asarray(np.array([c["flux_" + k] for c in cells])), cpu)
                              for k in FLUXES))
    return entry, fluxes


@pytest.fixture(autouse=True)
def libm_policy():
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    jax.clear_caches()
    yield
    set_policy(previous)
    jax.clear_caches()


def _stages(entry, fluxes, nml, constants=NEMO_SI3_CONSTANTS_CONFIG):
    return bl.si3_ice_thd_category_stages(entry=entry, fluxes=fluxes, namelist=nml, rDt_ice=RDT,
                                          constants=constants)


_JIT_STAGES = jax.jit(lambda e, f, items: _stages(e, f, dict(items)), static_argnums=2)


def _model(cells, nml, *, eager=False):
    entry, fluxes = _entry(cells)
    if eager:
        with jax.disable_jit():
            return _stages(entry, fluxes, nml)
    return _JIT_STAGES(entry, fluxes, tuple(sorted(nml.items())))


def _oracle_rows(cells, nml):
    rows = []
    for c in cells:
        z = oracle_zdf(c, nml)
        d = oracle_dh(c, z, nml)
        rows.append((z, d, oracle_temp(d["h_i"], d["e_i"], d["sz_i"])))
    return rows


def _leaves(stages):
    out = {"zdf." + k: v for k, v in stages.zdf._asdict().items()}
    for k, v in stages.dh._asdict().items():
        if k in ("dh", "fluxes"):
            out.update({f"dh.{k}.{kk}": vv for kk, vv in v._asdict().items()})
        else:
            out["dh." + k] = v
    out["temp.t_i"] = stages.t_i
    return out


def _oracle_leaves(rows):
    out = {}
    for name in bl.SI3ZDFOutputs._fields:
        out["zdf." + name] = np.array([r[0][name] for r in rows], dtype=np.float64)
    for name in bl.SI3DHOutputs._fields:
        if name == "dh":
            for kk in bl.SI3DHIncrements._fields:
                out[f"dh.dh.{kk}"] = np.array([r[1]["dh"][kk] for r in rows])
        elif name == "fluxes":
            for kk in FLUXES:
                out[f"dh.fluxes.{kk}"] = np.array([r[1]["fluxes"][kk] for r in rows])
        else:
            out["dh." + name] = np.array([r[1][name] for r in rows], dtype=np.float64)
    out["temp.t_i"] = np.array([r[2] for r in rows], dtype=np.float64)
    return out


def _bits(x):
    return np.asarray(x, dtype=np.float64).view(np.uint64)


def _diff(model, oracle):
    m = _leaves(model)
    assert set(m) == set(oracle)
    return {k: int(np.sum(_bits(m[k]) != _bits(ref))) for k, ref in oracle.items()
            if np.any(_bits(m[k]) != _bits(ref))}


def _digest(leaves):
    return hashlib.sha256(b"".join(_bits(leaves[k]).tobytes() for k in sorted(leaves))).hexdigest()


# --------------------------------------------------------------------------- tests
def test_deck_resolves_production_arm():
    nml = _deck()
    assert (nml["jpl"], nml["nlay_i"], nml["nlay_s"]) == (JPL, NLAY_I, NLAY_S)
    assert nml["nn_icesal"] == 4 and nml["ln_icedH"] is True and nml["rn_cnd_s"] == 0.31
    c = NEMO_SI3_CONSTANTS_CONFIG
    assert (c.T0, c.rho_snow, c.rho_ice, c.k_ice, c.c_ice, c.latent_fusion, c.liquidus_slope, c.rho_ocean,
            c.c_ocean) == (RT0, RHOS, RHOI, RCND_I, RCPI, RLFUS, RTMLT, RHO0, RCP)


def test_five_categories_bit_identical_to_independent_transcription():
    nml = _deck()
    branch = dict.fromkeys(("melting_surface", "snow", "no_snow", "internal_ice_melt", "basal_growth",
                            "basal_melt", "flooding", "ice_gone", "deposition", "warm_snow", "picard>1"), 0)
    for cells in _categories():
        rows = _oracle_rows(cells, nml)
        assert _diff(_model(cells, nml), _oracle_leaves(rows)) == {}
        for c, (z, d, _) in zip(cells, rows):
            branch["melting_surface"] += z["t_su"] >= RT0
            branch["snow"] += c["h_s"] > 0
            branch["no_snow"] += c["h_s"] == 0
            branch["internal_ice_melt"] += d["dh"]["dh_i_itm"] < 0
            branch["basal_growth"] += d["dh"]["dh_i_bog"] > 0
            branch["basal_melt"] += d["dh"]["dh_i_bom"] < 0
            branch["flooding"] += d["dh"]["dh_snowice"] > 0
            branch["ice_gone"] += d["h_i"] == 0.0
            branch["deposition"] += c["evap_ice"] < 0
            branch["warm_snow"] += max(z["t_s"]) > RT0
            branch["picard>1"] += z["n_picard"] > 1
    assert not [k for k, v in branch.items() if v == 0], branch


@pytest.mark.parametrize("jl", [2, 3])
def test_eager_equals_jit_every_bit(jl):
    # jl = 3 is a one-cell category: compiled for a batch of one, XLA re-expanded zeta and
    # zkappa inside the -zeta*zkappa*zg1s matrix entry and regrouped them (1 ULP, 26 leaves)
    # until every BL99 product was materialised; batches of two or more never showed it.
    nml = _deck()
    cells = _categories(seed=3)[jl]
    jit_leaves, eager_leaves = _leaves(_model(cells, nml)), _leaves(_model(cells, nml, eager=True))
    for k in jit_leaves:
        assert np.array_equal(_bits(jit_leaves[k]), _bits(eager_leaves[k])), k
    oracle = _oracle_leaves(_oracle_rows(cells, nml))
    assert _diff(_model(cells, nml), oracle) == {}


def test_inexact_value_pins():
    nml = _deck()
    assert float(nml["rn_cnd_s"]).hex() == "0x1.3d70a3d70a3d7p-2"  # 0.31 is not exact
    assert (1.0 / NLAY_I).hex() == "0x1.999999999999ap-4"  # r1_nlay_i = 0.1 rounds up
    assert (1.0 / NLAY_S).hex() == "0x1.999999999999ap-3"
    assert float(nml["rn_snwblow"]).hex() == "0x1.51eb851eb851fp-1"
    cells = _categories(seed=0)[0]
    assert _digest(_leaves(_model(cells, nml))) == _digest(_oracle_leaves(_oracle_rows(cells, nml)))


PLANTS = ("rn_cnd_s_ulp", "rn_sinew_ulp", "kappa_i_ulp", "native_transcendentals", "reversed_tie_roles",
          "sst_from_t_bo")


@pytest.mark.parametrize("plant", PLANTS)
def test_production_path_plants_fire(plant, monkeypatch):
    nml = _deck()
    cells = [c for cat in _categories(seed=1) for c in cat]
    if plant == "native_transcendentals":
        # 1-ULP EXP differences in absorbed shortwave vanish under ~260 K; transmitted
        # shortwave feeds basal growth directly, so give this plant snow-free sunlit columns.
        rng = np.random.default_rng(7)
        cells = [_cell(rng, "nosnow") for _ in range(64)]
        for c in cells:
            c["qtr_ice_top"] = 300.0 * rng.random()
    oracle = _oracle_leaves(_oracle_rows(cells, nml))
    assert _diff(_model(cells, nml), oracle) == {}
    if plant == "rn_cnd_s_ulp":
        nml = dict(nml, rn_cnd_s=math.nextafter(nml["rn_cnd_s"], 1.0))
    elif plant == "rn_sinew_ulp":
        nml = dict(nml, rn_sinew=math.nextafter(nml["rn_sinew"], 1.0))
    elif plant == "kappa_i_ulp":
        nml = dict(nml, rn_kappa_i=math.nextafter(nml["rn_kappa_i"], 2.0))
    elif plant == "native_transcendentals":
        set_policy(PrecisionPolicy.fp64(transcendentals="native"))
    elif plant == "reversed_tie_roles":
        monkeypatch.setattr(bl, "_maxsd", lambda d, s: jnp.where(s > d, s, d))
        monkeypatch.setattr(bl, "_minsd", lambda d, s: jnp.where(s < d, s, d))
    elif plant == "sst_from_t_bo":
        for c in cells:
            c["sst"] = c["t_bo"] - RT0
    jax.clear_caches()
    assert _diff(_model(cells, nml), oracle) != {}


def test_signed_zero_and_nan_sites_follow_object_code():
    # MAX(0, -qprec) with qprec = +0 is `maxsd 0, -0` -> +0; seen through hfx_spr = -0 + ze_s(0)*zh_s(0)*a*r1.
    nml = _deck()
    cells = _categories(seed=0)[0][:2]
    for c in cells:
        c["sprecip"], c["qprec_ice"], c["flux_hfx_spr"] = 1.0e-5, 0.0, -0.0
    rows = _oracle_leaves(_oracle_rows(cells, nml))
    model = _leaves(_model(cells, nml))
    assert np.array_equal(_bits(model["dh.fluxes.hfx_spr"]), _bits(rows["dh.fluxes.hfx_spr"]))
    assert np.all(_bits(rows["dh.fluxes.hfx_spr"]) == 0)  # +0, not -0
    assert not np.signbit(float(bl._maxsd(jnp.asarray(-0.0), 0.0)))
    assert np.isnan(float(bl._minsd(jnp.asarray(1.0), jnp.nan)))  # NaN: the source operand is returned
    assert float(bl._minsd(jnp.asarray(jnp.nan), 1.0)) == 1.0


def test_c1d_card_routes_its_dh_through_the_shared_core(monkeypatch):
    calls = []
    real = bl._si3_ice_thd_dh_core

    def spy(**kw):
        calls.append((kw["nn_icesal"], kw["e_i"].shape[-1]))
        return real(**kw)

    monkeypatch.setattr(bl, "_si3_ice_thd_dh_core", spy)
    z = jnp.zeros(1)
    st = bl.SI3ColumnArrays(z + 0.9, z + 1.0, z + 0.1, z + RT0 - 15.0, jnp.full((1, 3), 3.0e8),
                            jnp.full((1, 3), 1.1e8), z + 4.0, jnp.full((1, 3), 4.0), z)
    forcing = bl.SI3SurfaceForcing(*(z for _ in range(13)), frq_m=1.0)._replace(t_bottom=z + RT0 - 1.8, sss=z + 34.0)
    bl.si3_column_step_arrays(st, forcing, 3600.0, NEMO_SI3_CONSTANTS_CONFIG)
    assert calls == [(2, 3)]


def test_gradients_finite_and_central_difference():
    nml = _deck()
    cells = [c for c in _categories(seed=0)[0] if c["h_s"] > 0.1][:2]
    assert cells
    entry, fluxes = _entry(cells)

    def loss(qns):
        s = _stages(dict(entry, qns_ice=qns), fluxes, nml)
        return jnp.sum(s.t_i) + 100.0 * jnp.sum(s.dh.h_i) + jnp.sum(s.zdf.hfx_err_dif)

    g = jax.grad(loss)(entry["qns_ice"])
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.max(jnp.abs(g))) > 0.0
    eps = 1.0e-3
    for i in range(len(cells)):
        d = jnp.zeros_like(entry["qns_ice"]).at[i].set(eps)
        fd = (loss(entry["qns_ice"] + d) - loss(entry["qns_ice"] - d)) / (2 * eps)
        assert abs(float(fd) - float(g[i])) <= 1.0e-5 * max(1.0, abs(float(fd))), (i, float(fd), float(g[i]))


def test_no_defaults_and_refused_arms():
    nml = _deck()
    entry, fluxes = _entry(_categories()[3])
    for key in ("rn_cnd_s", "rn_kappa_i", "rn_kappa_s", "nn_icesal", "rn_sinew", "rn_snwblow", "ln_icedH"):
        with pytest.raises((ValueError, TypeError)):
            _stages(entry, fluxes, {k: v for k, v in nml.items() if k != key})
    for key, bad in (("nn_qtrice", 1), ("nn_snwfra", 1), ("ln_cndi_P07", False), ("ln_cndflx", True),
                     ("ln_virtual_itd", True), ("ln_zdf_chkcvg", True), ("nn_icesal", 1), ("ln_cndemulate", True),
                     ("ln_icedH", False)):
        with pytest.raises(ValueError):
            _stages(entry, fluxes, dict(nml, **{key: bad}))
    with pytest.raises(KeyError):
        _stages({k: v for k, v in entry.items() if k != "sst"}, fluxes, nml)
    with pytest.raises(ValueError):
        _stages(dict(entry, sst=None), fluxes, nml)
    with pytest.raises(ValueError):
        bl.si3_ice_thd_temp(h_i=None, e_i=entry["e_i"], sz_i=entry["sz_i"], constants=NEMO_SI3_CONSTANTS_CONFIG)
