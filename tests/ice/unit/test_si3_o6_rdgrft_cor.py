"""SI3 rung O6: five-category ridging/rafting and ice_cor(kt, 1), NEMO-literal.

The reference is an independent scalar transcription written here in plain
Python, in NEMO's statement order over NEMO's packed cell list
(``ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn_rdgrft.f90``, ``iceitd.f90``,
``icevar.f90``, ``icecor.f90``).  ``math.exp``/``math.tanh`` are glibc's,
the arm the record build links (DECISION 162).  MIN/MAX keep the operand the
compiled MINSD/MAXSD keeps on a tie where the object code was read.
"""

from __future__ import annotations

import hashlib
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice import itd as itd_mod
from legoesm.ice import ridging as rdg_mod
from legoesm.ice import si3_surface
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as K
from legoesm.ice.itd import SI3IceCorConfig, SI3IceCorState, si3_ice_cor_dyn
from legoesm.ice.ridging import (
    SI3RdgrftConfig,
    SI3RdgrftState,
    si3_ice_dyn_rdgrft,
    si3_rdgrft_config_after_init,
)

from legoesm import constants

NN_HLS = 2
JPI, JPJ, JPL, NLAY_I, NLAY_S = 9, 8, 5, 3, 2  # test-local layer counts; ORCA2 has 10/5
RDT = 10800.0  # ORCA2_SAS_ICE rn_Dt * nn_fsbc
RHOW = constants.rho_water  # phycst.f90:70 rhow = 1000
HI_MAX = np.array([0.0, 0.45, 1.13, 2.14, 3.67, 99.0])  # hi_max(0:jpl), hi_max(jpl)=rn_himax
HI_MEAN = 0.5 * (HI_MAX[:-1] + HI_MAX[1:])

# namelist_ice_ref:76-103 (ORCA2_SAS_ICE has no namdyn_rdgrft override); ponds LEV + lids, nn_icesal=4.
RDG_CFG = SI3RdgrftConfig(
    rn_csrdg=0.5, ln_partf_exp=True, rn_astar=0.03, ln_distf_lin=True, ln_distf_exp=False,
    rn_murdg=3.0, ln_ridging=True, rn_hstar=25.0, rn_porordg=0.3, rn_fsnwrdg=0.5, rn_fpndrdg=1.0,
    ln_rafting=True, rn_hraft=0.75, rn_craft=5.0, rn_fsnwrft=0.5, rn_fpndrft=1.0,
    ln_rhg_evp_or_vp=True, ln_pnd=True, ln_pnd_lids=True, nn_icesal=4, rdt_ice=RDT,
    rhoi=K.rho_ice, rhos=K.rho_snow, rhow=RHOW, rcp=K.c_ocean,
)
COR_CFG = SI3IceCorConfig(
    rn_himin=0.1, rn_simin=0.1, rn_sinew=0.75, nn_icesal=4, ln_pnd=True, ln_pnd_lids=True,
    rdt_ice=RDT, rhoi=K.rho_ice, rhos=K.rho_snow, rhow=RHOW, rt0=constants.T_freeze,
)


@pytest.fixture(autouse=True)
def _libm():
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(previous)


# ----------------------------------------------------------------------------- inputs
def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    sh, sh3 = (JPI, JPJ), (JPI, JPJ, JPL)
    a = rng.uniform(0.02, 0.3, sh3)
    h = np.sort(rng.uniform(0.05, 4.5, sh3), axis=-1)
    a[1:4, 2:5, 0] = 1e-12  # tiny category: participates, zhi forced 0
    a[4, 4, :] = 0.0  # ice-free cell
    a[5, 3] = [0.3, 0.3, 0.25, 0.1, 0.05]  # asum ~ 1: excessive-removal clamp
    h[6, 4] = [0.3, 0.8, 1.5, 3.0, 25.0]  # zvti = SUM(v_i) ~ 13 m, in (10, 20): porosity ramp
    a[6, 4] = [0.15, 0.1, 0.1, 0.1, 0.5]
    h[3, 5, 1] = 2.5  # too thick for category 2: rebinning up
    h[2, 3, 3] = 0.5  # too thin for category 4: rebinning down
    h[5, 5, 0] = 0.05  # below himin
    v = a * h
    s = dict(
        a_i=a, v_i=v, v_s=v * rng.uniform(0.05, 0.3, sh3), oa_i=a * rng.uniform(1e5, 1e7, sh3),
        a_ip=a * rng.uniform(0.0, 0.3, sh3), v_ip=None, v_il=None,
        e_s=rng.uniform(1e6, 1e8, (JPI, JPJ, NLAY_S, JPL)),
        e_i=rng.uniform(1e7, 3e8, (JPI, JPJ, NLAY_I, JPL)),
        szv_i=v[:, :, None, :] * rng.uniform(1.0, 9.0, (JPI, JPJ, NLAY_I, JPL)) / NLAY_I,
        at_i=rng.uniform(0, 1, sh), ato_i=None,
        wfx_dyn=rng.normal(0, 1e-4, sh), sfx_dyn=rng.normal(0, 1e-4, sh),
        hfx_dyn=rng.normal(0, 1.0, sh), wfx_snw_dyn=rng.normal(0, 1e-4, sh),
        wfx_pnd=rng.normal(0, 1e-4, sh),
    )
    s["v_ip"] = s["a_ip"] * rng.uniform(0.0, 0.3, sh3)
    s["v_il"] = s["v_ip"] * rng.uniform(0.0, 0.2, sh3)
    s["ato_i"] = np.maximum(0.0, 1.0 - a.sum(-1)) + rng.uniform(0, 0.05, sh)
    s["e_s"][s["v_s"][:, :, None, :].repeat(NLAY_S, 2) == 0] = 0.0
    divu = rng.normal(-2e-6, 3e-6, sh)
    delta = np.abs(divu) + rng.uniform(0, 4e-6, sh)
    divu[5, 3], delta[5, 3] = -3e-4, 3e-4
    sst = rng.uniform(-1.9, 0.5, sh)
    sss = rng.uniform(30.0, 35.0, sh)
    return s, divu, delta, sst, sss


def _cor_inputs(seed=1):
    s, *_rest, sst, sss = _inputs(seed)
    rng = np.random.default_rng(seed + 10)
    sh, sh3 = (JPI, JPJ), (JPI, JPJ, JPL)
    c = {k: s[k] for k in ("a_i", "v_i", "v_s", "oa_i", "a_ip", "v_ip", "v_il", "e_s", "e_i", "szv_i", "at_i", "ato_i")}
    c["a_i"] = c["a_i"].copy()
    c["a_i"][3, 3] *= 1.4  # at_i > amax
    c["v_i"] = c["v_i"].copy()
    c["v_i"][2, 2, 1] = 1e-11  # zapsmall
    c["szv_i"] = c["szv_i"].copy()
    c["szv_i"][4, 5, :, 2] = 0.0  # salinity below simin
    c["szv_i"][5, 2, :, 1] *= 40.0  # salinity above sinew*sss
    c.update(
        sz_i=rng.uniform(1, 9, (JPI, JPJ, NLAY_I, JPL)), t_su=rng.uniform(250, 273, sh3),
        t_i=rng.uniform(250, 273, (JPI, JPJ, NLAY_I, JPL)), t_s=rng.uniform(250, 273, (JPI, JPJ, NLAY_S, JPL)),
        h_i=rng.uniform(0, 3, sh3), h_s=rng.uniform(0, 1, sh3), h_ip=rng.uniform(0, 1, sh3),
        h_il=rng.uniform(0, 1, sh3), vt_i=rng.uniform(0, 3, sh),
        sfx_res=rng.normal(0, 1e-4, sh), hfx_res=rng.normal(0, 1.0, sh), wfx_res=rng.normal(0, 1e-4, sh),
    )
    amax = np.full(sh, 0.997)
    return c, amax, sst, sss


# ------------------------------------------------------- independent scalar transcription
def _maxk(keep, other):  # MAXSD returning the kept operand on a tie
    return other if other > keep else keep


def _mink(keep, other):
    return other if other < keep else keep


def _sum0(xs):
    acc = 0.0
    for x in xs:
        acc = acc + x
    return acc


def _prep_cell(a, v, ato, cn, op, cfg):
    jpl = len(a)
    r1 = 1.0 / cfg.rdt_ice
    z1_astar = 1.0 / cfg.rn_astar
    zhi = [v[j] / a[j] if a[j] > 1e-10 else 0.0 for j in range(jpl)]
    zasum = ato + _sum0(a)
    z1 = 1.0 / zasum if zasum > 1e-10 else 0.0
    zg = [0.0, ato * z1] + [(ato + _sum0(a[: j + 1])) * z1 for j in range(jpl)]
    zfac = 1.0 / (1.0 - math.exp(-z1_astar))
    ge = [math.exp(-(g * z1_astar)) * zfac for g in zg]
    ap = [ge[k] - ge[k + 1] for k in range(jpl + 1)]
    ar = [(1.0 + math.tanh(cfg.rn_craft * (zhi[j] - cfg.rn_hraft))) * 0.5 * ap[j + 1] for j in range(jpl)]
    af = [ap[j + 1] - ar[j] for j in range(jpl)]
    hrmin, hrmax, hraft, phi = [0.0] * jpl, [0.0] * jpl, [0.0] * jpl, [1.0] * jpl
    zak = ap[0]
    for j in range(jpl):
        if ap[j + 1] > 0.0:
            zhm = _maxk(math.sqrt(cfg.rn_hstar * zhi[j]), zhi[j] * 1.1)
            hrmin[j] = _mink(2.0 * zhi[j], 0.5 * (zhm + zhi[j]))
            hraft[j] = zhi[j] * 2.0
            hrmax[j] = 2.0 * zhm - hrmin[j]
            phi[j] = zhi[j] / max(zhm, 1e-20)
            zak = zak + ar[j] * (1.0 - phi[j]) + af[j] * (1.0 - 0.5)
    cg = cn / zak if zak > 1e-10 else 0.0
    for j in range(jpl):
        if ap[j + 1] * cg * cfg.rdt_ice > a[j] and ap[j + 1] != 0.0:
            cg = a[j] / ap[j + 1] * r1
    zf = ato + (op - ap[0] * cg) * cfg.rdt_ice
    if zf < 0.0:
        op = ap[0] * cg - ato * r1
    elif zf > zasum:
        op = ap[0] * cg + (zasum - ato) * r1
    return dict(ap=ap, ar=ar, af=af, phi=phi, hraft=hraft, hrmin=hrmin, hrmax=hrmax, cg=cg, op=op)


def _shift_cell(c, p, sst, sss, cfg, hi_max):
    jpl, nli = len(c["a_i"]), len(c["e_i"])
    nls = len(c["e_s"])
    r1, rdt, r1n = 1.0 / cfg.rdt_ice, cfg.rdt_ice, 1.0 / nli
    zvti = _sum0(c["v_i"])
    c["ato_i"] = _maxk(0.0, c["ato_i"] + (p["op"] - p["ap"][0] * p["cg"]) * rdt)
    for j1 in range(jpl):
        if not (p["ap"][j1 + 1] > 0.0 and p["cg"] > 0.0):
            continue
        z1ai = 1.0 / c["a_i"][j1] if c["a_i"][j1] > 1e-10 else 0.0
        ard1 = p["ar"][j1] * p["cg"] * rdt
        arf1 = p["af"][j1] * p["cg"] * rdt
        ard2, arf2 = ard1 * p["phi"][j1], arf1 * 0.5
        afr, aff = ard1 * z1ai, arf1 * z1ai
        v1 = c["v_i"][j1]
        if zvti <= 10.0:
            vsw = v1 * afr * cfg.rn_porordg
        elif zvti >= 20.0:
            vsw = 0.0
        else:
            vsw = v1 * afr * cfg.rn_porordg * _maxk(0.0, 2.0 - 0.1 * zvti)
        ersw = -(cfg.rhoi * vsw * cfg.rcp * sst)
        virdg, vsrdg = v1 * afr + vsw, c["v_s"][j1] * afr
        oirdg = c["oa_i"][j1] * afr * p["phi"][j1]
        virft, vsrft, oirft = v1 * aff, c["v_s"][j1] * aff, c["oa_i"][j1] * aff * 0.5
        aprdg, vprdg = c["a_ip"][j1] * afr * p["phi"][j1], c["v_ip"][j1] * afr
        aprft, vprft = c["a_ip"][j1] * aff * 0.5, c["v_ip"][j1] * aff
        vlrdg, vlrft = c["v_il"][j1] * afr, c["v_il"][j1] * aff
        esrdg = [c["e_s"][k][j1] * afr for k in range(nls)]
        esrft = [c["e_s"][k][j1] * aff for k in range(nls)]
        eirdg = [c["e_i"][k][j1] * afr + ersw * r1n for k in range(nli)]
        eirft = [c["e_i"][k][j1] * aff for k in range(nli)]
        sirdg = [c["szv_i"][k][j1] * afr + vsw * sss * r1n for k in range(nli)]
        sirft = [c["szv_i"][k][j1] * aff for k in range(nli)]
        c["wfx_dyn"] = c["wfx_dyn"] - vsw * cfg.rhoi * r1
        c["sfx_dyn"] = c["sfx_dyn"] - vsw * sss * cfg.rhoi * r1
        c["hfx_dyn"] = c["hfx_dyn"] + ersw * r1
        c["wfx_snw_dyn"] = c["wfx_snw_dyn"] + (cfg.rhos * vsrdg * (1.0 - cfg.rn_fsnwrdg)
                                               + cfg.rhos * vsrft * (1.0 - cfg.rn_fsnwrft)) * r1
        for k in range(nls):
            c["hfx_dyn"] = c["hfx_dyn"] + (-esrdg[k] * (1.0 - cfg.rn_fsnwrdg) - esrft[k] * (1.0 - cfg.rn_fsnwrft)) * r1
        c["wfx_pnd"] = c["wfx_pnd"] + (cfg.rhow * vprdg * (1.0 - cfg.rn_fpndrdg) + cfg.rhow * vprft * (1.0 - cfg.rn_fpndrft)) * r1
        c["wfx_pnd"] = c["wfx_pnd"] + (cfg.rhow * vlrdg * (1.0 - cfg.rn_fpndrdg) + cfg.rhow * vlrft * (1.0 - cfg.rn_fpndrft)) * r1
        keep = 1.0 - afr - aff
        c["a_i"][j1] = c["a_i"][j1] - ard1 - arf1
        for n in ("v_i", "v_s", "oa_i", "a_ip", "v_ip", "v_il"):
            c[n][j1] = c[n][j1] * keep
        for n in ("e_s", "e_i", "szv_i"):
            for k in range(len(c[n])):
                c[n][k][j1] = c[n][k][j1] * keep
        it_rdg = it_rft = 0
        for j2 in range(jpl):
            lo, hi = hi_max[j2], hi_max[j2 + 1]
            hmin, hmax = p["hrmin"][j1], p["hrmax"][j1]
            if hmin <= hi and hmax > lo:
                hL, hR = max(hmin, lo), min(hmax, hi)
                fa = (hR - hL) / (hmax - hmin)
                fv = (hR * hR - hL * hL) / (hmax * hmax - hmin * hmin)
                it_rdg = 1
            else:
                fa = fv = 0.0
            if lo < p["hraft"][j1] <= hi:
                zs = 1.0
                it_rft = 1
            else:
                zs = 0.0
            if it_rdg == 0 and j2 == jpl - 1:
                fa = fv = 1.0
            if it_rft == 0 and j2 == jpl - 1:
                zs = 1.0
            c["a_i"][j2] = c["a_i"][j2] + (ard2 * fa + arf2 * zs)
            c["oa_i"][j2] = c["oa_i"][j2] + (oirdg * fa + oirft * zs)
            c["v_i"][j2] = c["v_i"][j2] + (virdg * fv + virft * zs)
            c["v_s"][j2] = c["v_s"][j2] + (vsrdg * cfg.rn_fsnwrdg * fv + vsrft * cfg.rn_fsnwrft * zs)
            c["v_ip"][j2] = c["v_ip"][j2] + (vprdg * cfg.rn_fpndrdg * fv + vprft * cfg.rn_fpndrft * zs)
            c["a_ip"][j2] = c["a_ip"][j2] + (aprdg * cfg.rn_fpndrdg * fa + aprft * cfg.rn_fpndrft * zs)
            c["v_il"][j2] = c["v_il"][j2] + (vlrdg * cfg.rn_fpndrdg * fv + vlrft * cfg.rn_fpndrft * zs)
            for k in range(nls):
                c["e_s"][k][j2] = c["e_s"][k][j2] + (esrdg[k] * cfg.rn_fsnwrdg * fv + esrft[k] * cfg.rn_fsnwrft * zs)
            for k in range(nli):
                c["e_i"][k][j2] = c["e_i"][k][j2] + (eirdg[k] * fv + eirft[k] * zs)
                c["szv_i"][k][j2] = c["szv_i"][k][j2] + (sirdg[k] * fv + sirft[k] * zs)
    _roundoff_cell(c)


def _roundoff_cell(c):
    for n in ("a_i", "v_i", "v_s", "oa_i", "a_ip", "v_ip"):
        c[n] = [0.0 if x < 0.0 else x for x in c[n]]
    c["v_il"] = [0.0 if (x < 0.0 and x > -1e-10) else x for x in c["v_il"]]
    for n in ("e_i", "e_s", "szv_i"):
        c[n] = [[0.0 if x < 0.0 else x for x in row] for row in c[n]]


_CAT = ("a_i", "v_i", "v_s", "oa_i", "a_ip", "v_ip", "v_il")
_LAY = ("e_s", "e_i", "szv_i")
_FLX = ("wfx_dyn", "sfx_dyn", "hfx_dyn", "wfx_snw_dyn", "wfx_pnd")


def _cell(arrs, ji, jj, cat, lay, scal):
    c = {n: [float(x) for x in arrs[n][ji, jj]] for n in cat}
    c.update({n: [[float(x) for x in row] for row in arrs[n][ji, jj]] for n in lay})
    c.update({n: float(arrs[n][ji, jj]) for n in scal})
    return c


def _put(arrs, ji, jj, c, cat, lay, scal):
    for n in cat + lay:
        arrs[n][ji, jj] = np.array(c[n])
    for n in scal:
        arrs[n][ji, jj] = c[n]


def oracle_rdgrft(s, divu, delta, sst, sss, cfg, hi_max=HI_MAX):
    o = {k: np.array(v, dtype=np.float64, copy=True) for k, v in s.items()}
    cells = []
    for jj in range(NN_HLS, JPJ - NN_HLS):  # :271 DO jj ; DO ji
        for ji in range(NN_HLS, JPI - NN_HLS):
            o["at_i"][ji, jj] = _sum0(o["a_i"][ji, jj])
            if o["at_i"][ji, jj] > 1e-10:
                cells.append((ji, jj))
    st = {}
    for ji, jj in cells:
        dv, dl = float(divu[ji, jj]), float(delta[ji, jj])
        cn = cfg.rn_csrdg * 0.5 * (dl - abs(dv)) - _mink(0.0, dv)
        if dv < 0.0:
            cn = max(cn, -dv)
        c = _cell(o, ji, jj, _CAT, _LAY, ("ato_i",) + _FLX)
        p = _prep_cell(c["a_i"], c["v_i"], c["ato_i"], cn, cn + dv, cfg)
        if _sum0(p["ap"][1:]) > 0.0 and p["cg"] > 0.0:
            st[(ji, jj)] = dict(c=c, cn=cn, op=p["op"], dv=dv)
    it, iterate = 1, 1 if st else 0
    while iterate > 0 and it < 20:
        for key, d in st.items():
            p = _prep_cell(d["c"]["a_i"], d["c"]["v_i"], d["c"]["ato_i"], d["cn"], d["op"], cfg)
            d["op"] = p["op"]
            _shift_cell(d["c"], p, float(sst[key]), float(sss[key]), cfg, hi_max)
        iterate = 0
        for d in st.values():
            zfac = 1.0 - (d["c"]["ato_i"] + _sum0(d["c"]["a_i"]))
            if abs(zfac) < 1e-10:
                d["cn"] = d["op"] = 0.0
                d["c"]["ato_i"] = _maxk(0.0, 1.0 - _sum0(d["c"]["a_i"]))
            else:
                iterate = 1
                d["dv"] = zfac * (1.0 / cfg.rdt_ice)
                d["cn"], d["op"] = _maxk(0.0, -d["dv"]), _maxk(0.0, d["dv"])
        it += 1
    for (ji, jj), d in st.items():
        _put(o, ji, jj, d["c"], _CAT, _LAY, ("ato_i",) + _FLX)
    return o, it - 1, set(st)


def _shiftice_cell(c, jl, upper, dai, dvi, amax, cfg):
    j1, j2 = (jl + 1, jl) if upper else (jl, jl + 1)
    zat = [c["a_i"][j] * c["t_su"][j] for j in range(JPL)]
    zv = dvi / c["v_i"][j1] if c["v_i"][j1] >= 1e-10 else 0.0
    za = dai / c["a_i"][j1] if c["a_i"][j1] >= 1e-10 else 0.0
    c["a_i"][j1] -= dai
    c["a_i"][j2] += dai
    c["v_i"][j1] -= dvi
    c["v_i"][j2] += dvi
    for n, w in (("v_s", zv), ("oa_i", za), ("a_ip", za), ("v_ip", zv), ("v_il", zv)):
        t = c[n][j1] * w
        c[n][j1] -= t
        c[n][j2] += t
    t = zat[j1] * za
    zat[j1] -= t
    zat[j2] += t
    for n in _LAY:
        for k in range(len(c[n])):
            t = c[n][k][j1] * zv
            c[n][k][j1] -= t
            c[n][k][j2] += t
    _roundoff_cell(c)
    ztmp = _sum0(c["a_i"])
    if ztmp > amax:
        c["a_i"] = [x * amax / ztmp for x in c["a_i"]]
    for j in range(JPL):
        if c["a_i"][j] >= 1e-20:
            c["h_i"][j], c["t_su"][j] = c["v_i"][j] / c["a_i"][j], zat[j] / c["a_i"][j]
        else:
            c["h_i"][j], c["t_su"][j] = 0.0, cfg.rt0


def oracle_ice_cor(s, amax, sst, sss, cfg, hi_max=HI_MAX, hi_mean=HI_MEAN):
    o = {k: np.array(v, dtype=np.float64, copy=True) for k, v in s.items()}
    cat = _CAT + ("t_su", "h_i", "h_s", "h_ip", "h_il")
    lay = _LAY + ("sz_i", "t_i", "t_s")
    scal = ("at_i", "vt_i", "ato_i", "sfx_res", "hfx_res", "wfx_res")
    r1, r1n = 1.0 / cfg.rdt_ice, 1.0 / NLAY_I
    for jj in range(NN_HLS, JPJ - NN_HLS):
        for ji in range(NN_HLS, JPI - NN_HLS):
            c = _cell(o, ji, jj, cat, lay, scal)
            am = float(amax[ji, jj])
            c["h_i"] = [c["v_i"][j] / c["a_i"][j] if c["a_i"][j] >= 1e-20 else 0.0 for j in range(JPL)]
            for j in range(JPL):
                if c["h_i"][j] < cfg.rn_himin:
                    c["a_ip"][j] = c["a_ip"][j] * c["h_i"][j] / cfg.rn_himin
                    c["a_i"][j] = c["a_i"][j] * c["h_i"][j] / cfg.rn_himin
            c["at_i"] = _sum0(c["a_i"])
            if c["at_i"] > am:
                c["a_i"] = [x * am / c["at_i"] for x in c["a_i"]]
            for jl in range(JPL - 1):
                a, v = c["a_i"][jl], c["v_i"][jl]
                if a > 0.0 and v > a * hi_max[jl + 1]:
                    _shiftice_cell(c, jl, False, 0.5 * a, v - (1.0 - 0.5) * a * hi_mean[jl], am, cfg)
            for jl in range(JPL - 2, -1, -1):
                a, v = c["a_i"][jl + 1], c["v_i"][jl + 1]
                if a > 0.0 and v <= a * hi_max[jl + 1]:
                    _shiftice_cell(c, jl, True, a, v, am, cfg)
            for jl in range(JPL):
                for k in range(NLAY_I):
                    z = c["szv_i"][k][jl]
                    c["szv_i"][k][jl] = min(max(z, cfg.rn_simin * c["v_i"][jl] * r1n),
                                            cfg.rn_sinew * float(sss[ji, jj]) * c["v_i"][jl] * r1n)
                    c["sfx_res"] = c["sfx_res"] - (c["szv_i"][k][jl] - z) * cfg.rhoi * r1
            c["h_i"] = [c["v_i"][j] / c["a_i"][j] if c["a_i"][j] > 1e-10 else 0.0 for j in range(JPL)]
            small = [min(c["a_i"][j], c["v_i"][j], c["h_i"][j]) < 1e-10 for j in range(JPL)]
            for jl in range(JPL):
                for k in range(NLAY_I):
                    if small[jl]:
                        c["sfx_res"] = c["sfx_res"] + c["szv_i"][k][jl] * cfg.rhoi * r1
                        c["szv_i"][k][jl] = 0.0
                        c["sz_i"][k][jl] = min(cfg.rn_simin, cfg.rn_sinew * float(sss[ji, jj]))
            for n, tn in (("e_i", "t_i"), ("e_s", "t_s")):
                for jl in range(JPL):
                    for k in range(len(c[n])):
                        if small[jl]:
                            c["hfx_res"] = c["hfx_res"] - c[n][k][jl] * r1
                            c[n][k][jl], c[tn][k][jl] = 0.0, cfg.rt0
            for jl in range(JPL):
                if small[jl]:
                    c["wfx_res"] = c["wfx_res"] + c["v_i"][jl] * cfg.rhoi * r1
                    c["wfx_res"] = c["wfx_res"] + c["v_s"][jl] * cfg.rhos * r1
                    c["wfx_res"] = c["wfx_res"] + (c["v_ip"][jl] + c["v_il"][jl]) * cfg.rhow * r1
                    for n in ("a_i", "v_i", "v_s", "oa_i", "h_i", "h_s", "a_ip", "v_ip", "v_il", "h_ip", "h_il"):
                        c[n][jl] = 0.0
                    c["t_su"][jl] = float(sst[ji, jj]) + cfg.rt0
            c["at_i"], c["vt_i"] = _sum0(c["a_i"]), _sum0(c["v_i"])
            if c["at_i"] == 0.0:
                c["ato_i"] = 1.0
            _put(o, ji, jj, c, cat, lay, scal)
    return o


# ------------------------------------------------------------------------- helpers
def _run_rdgrft(s, divu, delta, sst, sss, cfg=RDG_CFG, jit=True):
    def f(st, dv, dl):
        return si3_ice_dyn_rdgrft(SI3RdgrftState(**st), divu_i=dv, delta_i=dl, sst_m=sst, sss_m=sss,
                                  hi_max=HI_MAX, nn_hls=NN_HLS, config=cfg)
    st = {k: jnp.asarray(s[k]) for k in SI3RdgrftState._fields}
    return (jax.jit(f) if jit else f)(st, jnp.asarray(divu), jnp.asarray(delta))


def _run_cor(c, amax, sst, sss, jit=True):
    def f(st):
        return si3_ice_cor_dyn(SI3IceCorState(**st), rn_amax_2d=amax, sss_m=sss, sst_m=sst,
                               hi_max=HI_MAX, hi_mean=HI_MEAN, nn_hls=NN_HLS, config=COR_CFG)
    st = {k: jnp.asarray(c[k]) for k in SI3IceCorState._fields}
    return (jax.jit(f) if jit else f)(st)


def _bits(x):
    return np.asarray(x, dtype=np.float64).view(np.uint64)


def _assert_bits(model_tuple, oracle, names, label):
    for n in names:
        got, want = _bits(getattr(model_tuple, n)), _bits(oracle[n])
        bad = np.argwhere(got != want)
        assert bad.size == 0, f"{label}:{n} differs at {bad[:3].tolist()} ({bad.shape[0]} words)"


def _sha(t):
    d = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(t):
        d.update(np.ascontiguousarray(np.asarray(leaf, dtype=np.float64)).tobytes())
    return d.hexdigest()


# --------------------------------------------------------------------------- tests
def test_rdgrft_equals_independent_scalar_transcription():
    s, divu, delta, sst, sss = _inputs()
    want, passes, sel = oracle_rdgrft(s, divu, delta, sst, sss, RDG_CFG)
    got = _run_rdgrft(s, divu, delta, sst, sss)
    _assert_bits(got.state, want, SI3RdgrftState._fields, "rdgrft")
    assert int(got.n_passes) == passes and passes >= 2  # a second pass executes
    assert set(map(tuple, np.argwhere(np.asarray(got.selected)).tolist())) == sel
    assert len(sel) == 19  # 20 interior cells, one ice-free
    zvti = s["v_i"][6, 4].sum()
    assert 10.0 < zvti < 20.0 and (6, 4) in sel
    # the fixture reaches the branches the oracle exercises
    assert np.asarray(got.state.wfx_dyn != jnp.asarray(s["wfx_dyn"])).any()  # porosity fluxes


def test_rdgrft_thermo_off_override_and_equality():
    cfg = si3_rdgrft_config_after_init(RDG_CFG, ln_icethd=False)
    assert (cfg.rn_porordg, cfg.rn_fsnwrdg, cfg.rn_fsnwrft, cfg.rn_fpndrdg, cfg.rn_fpndrft) == (0.0, 1.0, 1.0, 1.0, 1.0)
    assert si3_rdgrft_config_after_init(RDG_CFG, ln_icethd=True) == RDG_CFG
    s, divu, delta, sst, sss = _inputs(3)
    want, _, _ = oracle_rdgrft(s, divu, delta, sst, sss, cfg)
    _assert_bits(_run_rdgrft(s, divu, delta, sst, sss, cfg).state, want, SI3RdgrftState._fields, "thd-off")


def test_ice_cor_equals_independent_scalar_transcription():
    c, amax, sst, sss = _cor_inputs()
    want = oracle_ice_cor(c, amax, sst, sss, COR_CFG)
    got = _run_cor(c, amax, sst, sss)
    _assert_bits(got, want, SI3IceCorState._fields, "ice_cor")
    # every arm reached: rebin moved t_su by rounding, zapsmall fired, amax clamp
    assert np.asarray(got.wfx_res != jnp.asarray(c["wfx_res"])).any()
    assert np.asarray(got.sfx_res != jnp.asarray(c["sfx_res"])).any()


def _budget(st, flx_w, flx_s, flx_h, rhoi, rhos, rhow):
    """NEMO's ice_cons_hsm terms per cell (icectl.f90:110-121), e1e2t = 1."""
    m = (np.asarray(st.v_i) * rhoi + np.asarray(st.v_s) * rhos + (np.asarray(st.v_ip) + np.asarray(st.v_il)) * rhow).sum(-1)
    salt = np.asarray(st.szv_i).sum(-2).sum(-1) * rhoi
    heat = np.asarray(st.e_i).sum((-2, -1)) + np.asarray(st.e_s).sum((-2, -1))
    return m, salt, heat, flx_w, flx_s, flx_h


def test_rdgrft_conserves_mass_salt_heat_as_nemo_checks():
    s, divu, delta, sst, sss = _inputs()
    s0 = SI3RdgrftState(**{k: jnp.asarray(s[k]) for k in SI3RdgrftState._fields})
    s1 = _run_rdgrft(s, divu, delta, sst, sss).state

    def terms(st):
        fw = np.asarray(st.wfx_dyn) + np.asarray(st.wfx_snw_dyn) + np.asarray(st.wfx_pnd)
        return _budget(st, fw, np.asarray(st.sfx_dyn), -np.asarray(st.hfx_dyn), K.rho_ice, K.rho_snow, RHOW)

    t0, t1 = terms(s0), terms(s1)
    for name, k in (("mass", 0), ("salt", 1), ("heat", 2)):  # icectl.f90:176-178
        diag = (t1[k] - t0[k]) / RDT + (t1[k + 3] - t0[k + 3])
        scale = np.abs(t0[k]).max() / RDT
        assert np.abs(diag).max() <= 1e-12 * scale, name
        assert np.abs(t1[k] - t0[k]).max() > 1e-6 * np.abs(t0[k]).max() or name != "mass"
    for leaf in ("a_i", "v_i", "v_s", "v_ip", "e_i", "e_s", "szv_i"):  # icectl.f90:150-157 minima
        assert np.asarray(getattr(s1, leaf)).min() >= 0.0
    a_sum = np.asarray(s1.a_i).sum(-1) + np.asarray(s1.ato_i)
    sel = np.asarray(_run_rdgrft(s, divu, delta, sst, sss).selected)
    assert np.abs(a_sum[sel] - 1.0).max() < 1e-9  # converged: a_i + ato_i = 1


def test_ice_cor_conserves_mass_salt_heat_as_nemo_checks():
    c, amax, sst, sss = _cor_inputs()
    s0 = SI3IceCorState(**{k: jnp.asarray(c[k]) for k in SI3IceCorState._fields})
    s1 = _run_cor(c, amax, sst, sss)

    def terms(st):
        return _budget(st, np.asarray(st.wfx_res), np.asarray(st.sfx_res), -np.asarray(st.hfx_res),
                       K.rho_ice, K.rho_snow, RHOW)

    t0, t1 = terms(s0), terms(s1)
    for k in range(3):
        diag = (t1[k] - t0[k]) / RDT + (t1[k + 3] - t0[k + 3])
        assert np.abs(diag).max() <= 1e-12 * np.abs(t0[k]).max() / RDT


PLANTS = {
    "ridge_min_multiplier": (rdg_mod, "_RDG_HRDG_HI_MIN", 1.1000001),
    "porosity_ramp_slope": (rdg_mod, "_RDG_POROSITY_SLOPE", 0.1000001),
    "raft_area_ratio": (rdg_mod, "_RDG_HI_HRFT", 0.5000001),
}


@pytest.mark.parametrize("plant", sorted(PLANTS))
def test_rdgrft_production_path_plant_fires(plant, monkeypatch):
    s, divu, delta, sst, sss = _inputs()
    want, _, _ = oracle_rdgrft(s, divu, delta, sst, sss, RDG_CFG)
    module, name, value = PLANTS[plant]
    monkeypatch.setattr(module, name, value)
    got = _run_rdgrft(s, divu, delta, sst, sss)
    with pytest.raises(AssertionError):
        _assert_bits(got.state, want, SI3RdgrftState._fields, plant)


def test_sum_order_plant_fires(monkeypatch):
    """Fortran SUM is sequential from +0; a reversed sum must turn the score red."""
    s, divu, delta, sst, sss = _inputs()
    want, _, _ = oracle_rdgrft(s, divu, delta, sst, sss, RDG_CFG)

    def reversed_sum(terms):
        acc = jnp.zeros(terms.shape[:-1], terms.dtype)
        for jl in reversed(range(terms.shape[-1])):
            acc = si3_surface.sr(acc + terms[..., jl])
        return acc

    monkeypatch.setattr(si3_surface, "fortran_sum0", reversed_sum)
    with pytest.raises(AssertionError):
        _assert_bits(_run_rdgrft(s, divu, delta, sst, sss).state, want, SI3RdgrftState._fields, "sum")


def test_category_bound_and_lid_input_plants_fire():
    """The model reads hi_max and ln_pnd_lids: changing either alone turns the score red."""
    s, divu, delta, sst, sss = _inputs()
    want, _, _ = oracle_rdgrft(s, divu, delta, sst, sss, RDG_CFG)
    shifted = HI_MAX.copy()
    shifted[2] = np.nextafter(shifted[2], 9.0) * 1.01

    def run(cfg, hm):
        st = SI3RdgrftState(**{k: jnp.asarray(s[k]) for k in SI3RdgrftState._fields})
        return si3_ice_dyn_rdgrft(st, divu_i=divu, delta_i=delta, sst_m=sst, sss_m=sss,
                                  hi_max=hm, nn_hls=NN_HLS, config=cfg).state

    for cfg, hm in ((RDG_CFG, shifted), (RDG_CFG._replace(ln_pnd_lids=False), HI_MAX)):
        with pytest.raises(AssertionError):
            _assert_bits(run(cfg, hm), want, SI3RdgrftState._fields, "input plant")
    np.testing.assert_array_equal(_bits(oracle_rdgrft(s, divu, delta, sst, sss, RDG_CFG, shifted)[0]["v_i"]),
                                  _bits(run(RDG_CFG, shifted).v_i))

def test_ice_cor_rebin_plant_fires(monkeypatch):
    c, amax, sst, sss = _cor_inputs()
    want = oracle_ice_cor(c, amax, sst, sss, COR_CFG)
    monkeypatch.setattr(itd_mod, "_SI3_HALF", 0.5000001)
    with pytest.raises(AssertionError):
        _assert_bits(_run_cor(c, amax, sst, sss), want, SI3IceCorState._fields, "rebin")


def test_inexact_value_pins():
    s, divu, delta, sst, sss = _inputs()
    c, amax, csst, csss = _cor_inputs()
    assert _sha(_run_rdgrft(s, divu, delta, sst, sss).state) == RDG_PIN
    assert _sha(_run_cor(c, amax, csst, csss)) == COR_PIN


def test_eager_equals_jit():
    s, divu, delta, sst, sss = _inputs()
    c, amax, csst, csss = _cor_inputs()
    jit_r, jit_c = _run_rdgrft(s, divu, delta, sst, sss), _run_cor(c, amax, csst, csss)
    with jax.disable_jit():
        eag_r = _run_rdgrft(s, divu, delta, sst, sss, jit=False)
        eag_c = _run_cor(c, amax, csst, csss, jit=False)
    _assert_bits(jit_r.state, eag_r.state._asdict(), SI3RdgrftState._fields, "rdg eager")
    _assert_bits(jit_c, eag_c._asdict(), SI3IceCorState._fields, "cor eager")


def test_gradient_finite_and_matches_central_difference():
    s, divu, delta, sst, sss = _inputs()
    ji, jj = 3, 3

    def loss(x):
        st = {k: jnp.asarray(s[k]) for k in SI3RdgrftState._fields}
        st["v_i"] = st["v_i"].at[ji, jj, 2].set(x)
        out = si3_ice_dyn_rdgrft(SI3RdgrftState(**st), divu_i=divu, delta_i=delta, sst_m=sst, sss_m=sss,
                                 hi_max=HI_MAX, nn_hls=NN_HLS, config=RDG_CFG)
        return jnp.sum(out.state.v_i[ji, jj] * jnp.arange(1.0, JPL + 1.0)) + jnp.sum(out.state.hfx_dyn[ji, jj])

    x0 = float(s["v_i"][ji, jj, 2])
    g = float(jax.jit(jax.grad(loss))(x0))
    eps = 1e-7 * x0
    jloss = jax.jit(loss)
    fd = (float(jloss(x0 + eps)) - float(jloss(x0 - eps))) / (2 * eps)
    assert math.isfinite(g) and g != 0.0
    assert abs(g / fd - 1.0) < 1e-5
    full = jax.jit(jax.grad(lambda d: jnp.sum(_run_rdgrft(s, divu, d, sst, sss, jit=False).state.v_i)))(jnp.asarray(delta))
    assert np.isfinite(np.asarray(full)).all()


def test_physical_inputs_have_no_defaults():
    s, divu, delta, sst, sss = _inputs()
    st = SI3RdgrftState(**{k: jnp.asarray(s[k]) for k in SI3RdgrftState._fields})
    base = dict(divu_i=divu, delta_i=delta, sst_m=sst, sss_m=sss, hi_max=HI_MAX, nn_hls=NN_HLS, config=RDG_CFG)
    for name in ("divu_i", "delta_i", "sst_m", "sss_m", "hi_max"):
        with pytest.raises(ValueError, match=name):
            si3_ice_dyn_rdgrft(st, **{**base, name: None})
    with pytest.raises(ValueError):
        si3_ice_dyn_rdgrft(st._replace(ato_i=None), **base)
    with pytest.raises(TypeError):
        si3_ice_dyn_rdgrft(st, **{**base, "config": RDG_CFG._asdict()})
    for bad in (dict(ln_partf_exp=False), dict(ln_distf_exp=True), dict(nn_icesal=2), dict(ln_rhg_evp_or_vp=False)):
        with pytest.raises(ValueError):
            si3_ice_dyn_rdgrft(st, **{**base, "config": RDG_CFG._replace(**bad)})
    with pytest.raises(TypeError):
        SI3RdgrftConfig()
    with pytest.raises(TypeError):
        SI3IceCorConfig()
    c, amax, csst, csss = _cor_inputs()
    cs = SI3IceCorState(**{k: jnp.asarray(c[k]) for k in SI3IceCorState._fields})
    kw = dict(rn_amax_2d=amax, sss_m=csss, sst_m=csst, hi_max=HI_MAX, hi_mean=HI_MEAN, nn_hls=NN_HLS, config=COR_CFG)
    for name in ("rn_amax_2d", "sss_m", "sst_m", "hi_max", "hi_mean"):
        with pytest.raises(ValueError, match=name):
            si3_ice_cor_dyn(cs, **{**kw, name: None})
    with pytest.raises(ValueError):
        si3_ice_cor_dyn(cs._replace(t_su=None), **kw)


def test_halos_untouched():
    s, divu, delta, sst, sss = _inputs()
    got = _run_rdgrft(s, divu, delta, sst, sss).state
    c, amax, csst, csss = _cor_inputs()
    gc = _run_cor(c, amax, csst, csss)
    for st, src in ((got, s), (gc, c)):
        for n in st._fields:
            g, w = np.asarray(getattr(st, n)), np.asarray(src[n])
            for sl in (np.s_[:NN_HLS], np.s_[-NN_HLS:]):
                np.testing.assert_array_equal(g[sl], w[sl])
                np.testing.assert_array_equal(g[:, sl], w[:, sl])


RDG_PIN = "00e22bad91eb8c38750766bc6ba07f96642ea6f4cd214e06ea75c9d569e57415"
COR_PIN = "929051ba6c3889e31c92fa0e014810d45c99b5b5594ca566185d73ebde4cd0af"


def test_lane3_jpl1_arm_is_not_the_nemo_literal_kernel():
    """Measured reason lane 3's jpl=1 arm is not routed through the O6 kernel.

    Same jpl=1 inputs, lane 3's resolved selectors (exponential redistribution,
    zero porosity, unit retention): the two disagree in bits on interior cells,
    so delegating would move lane-3 card values.  Read off the code (PLAUSIBLE
    as the cause): lane 3 takes G(jpl) = 1 exactly where NEMO computes
    ``(ato + SUM(a)) * z1_asum`` (``icedyn_rdgrft.f90:481``), and it stops
    updating a cell once ``shift`` is false although NEMO still opens water
    there on the next pass (``:713``).
    """
    from legoesm.ice.ridging import SI3JPL1RidgingState, apply_si3_jpl1_ridging

    s, divu, delta, sst, sss = _inputs(5)
    one = {k: (v[..., :1] if np.ndim(v) in (3, 4) else v) for k, v in s.items()}
    one["e_s"], one["e_i"], one["szv_i"] = s["e_s"][..., :1], s["e_i"][..., :1], s["szv_i"][..., :1]
    one["a_i"] = s["a_i"].sum(-1, keepdims=True).clip(0, 0.99)
    one["v_i"] = s["v_i"].sum(-1, keepdims=True)
    one["ato_i"] = 1.0 - one["a_i"][..., 0]
    cfg = RDG_CFG._replace(ln_distf_lin=False, ln_distf_exp=True, rn_porordg=0.0, rn_fsnwrdg=1.0,
                           rn_fsnwrft=1.0, rn_fpndrdg=1.0, rn_fpndrft=1.0)
    st = SI3RdgrftState(**{k: jnp.asarray(one[k]) for k in SI3RdgrftState._fields})
    ours = si3_ice_dyn_rdgrft(st, divu_i=divu, delta_i=delta, sst_m=sst, sss_m=sss,
                              hi_max=np.array([0.0, 99.0]), nn_hls=NN_HLS, config=cfg).state
    lane3, _ = apply_si3_jpl1_ridging(
        SI3JPL1RidgingState(
            ice_area=jnp.asarray(one["a_i"][..., 0]), open_water_area=jnp.asarray(one["ato_i"]),
            ice_volume=jnp.asarray(one["v_i"][..., 0]), snow_volume=jnp.asarray(one["v_s"][..., 0]),
            age_content=jnp.asarray(one["oa_i"][..., 0]), pond_area=jnp.asarray(one["a_ip"][..., 0]),
            pond_volume=jnp.asarray(one["v_ip"][..., 0]), pond_lid_volume=jnp.asarray(one["v_il"][..., 0]),
            snow_enthalpy=jnp.asarray(one["e_s"][..., 0]), ice_enthalpy=jnp.asarray(one["e_i"][..., 0]),
            ice_salt_content=jnp.asarray(one["szv_i"][..., 0])),
        jnp.asarray(divu), jnp.asarray(delta), RDT)
    inner = np.s_[NN_HLS:-NN_HLS, NN_HLS:-NN_HLS]
    a_ours = np.asarray(ours.a_i)[..., 0][inner]
    a_l3 = np.asarray(lane3.ice_area)[inner]
    o_ours = np.asarray(ours.ato_i)[inner]
    o_l3 = np.asarray(lane3.open_water_area)[inner]
    n_diff = int((_bits(a_ours) != _bits(a_l3)).sum() + (_bits(o_ours) != _bits(o_l3)).sum())
    assert n_diff > 0
    np.testing.assert_allclose(a_ours, a_l3, rtol=1e-9)


# ------------------------------------------------- parent-vs-current controls
O6_PARENT_COMMIT = "e8ee89ece8cb"
_RHEO_ROOT = __import__("pathlib").Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/final")
_TOUCHED = (
    ("legoesm.ice.itd", "packages/ice/legoesm/ice/itd.py"),
    ("legoesm.ice.ridging", "packages/ice/legoesm/ice/ridging.py"),
    ("legoesm.ice.si3_surface", "packages/ice/legoesm/ice/si3_surface.py"),
)


def _load_parent(extra=()):
    import subprocess
    import sys
    import types
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    names = _TOUCHED + tuple(extra)
    saved = {n: sys.modules.get(n) for n, _ in names}
    loaded = {}
    try:
        for name, rel in names:
            src = subprocess.check_output(["git", "show", f"{O6_PARENT_COMMIT}:{rel}"], cwd=root, text=True)
            mod = types.ModuleType(name)
            mod.__file__ = f"<git:{O6_PARENT_COMMIT}:{rel}>"
            sys.modules[name] = mod
            exec(compile(src, mod.__file__, "exec"), mod.__dict__)
            loaded[name] = mod
    finally:
        for n, orig in saved.items():
            if orig is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = orig
    return loaded


def test_parent_vs_current_shared_ridging_and_itd_callers():
    """Lane 3's jpl=1 arm and the generic ridging/ITD entry points are bit-identical to the parent."""
    from legoesm.ice import ridging as cur

    par = _load_parent()["legoesm.ice.ridging"]
    rng = np.random.default_rng(7)
    sh = (6, 5)
    fields = dict(ice_area=rng.uniform(0.1, 0.95, sh), open_water_area=rng.uniform(0.0, 0.2, sh),
                  ice_volume=rng.uniform(0.1, 3.0, sh), snow_volume=rng.uniform(0, 0.3, sh),
                  age_content=rng.uniform(0, 1e6, sh), pond_area=rng.uniform(0, 0.1, sh),
                  pond_volume=rng.uniform(0, 0.02, sh), pond_lid_volume=rng.uniform(0, 0.002, sh),
                  snow_enthalpy=rng.uniform(1e6, 1e8, sh + (5,)), ice_enthalpy=rng.uniform(1e7, 3e8, sh + (10,)),
                  ice_salt_content=rng.uniform(0.01, 0.5, sh + (10,)))
    divu, delta = rng.normal(-3e-6, 3e-6, sh), rng.uniform(0, 6e-6, sh)
    out_c = cur.apply_si3_jpl1_ridging(cur.SI3JPL1RidgingState(**fields), divu, delta, RDT)
    out_p = par.apply_si3_jpl1_ridging(par.SI3JPL1RidgingState(**fields), divu, delta, RDT)
    for x, y in zip(jax.tree_util.tree_leaves(out_c), jax.tree_util.tree_leaves(out_p)):
        np.testing.assert_array_equal(_bits(x), _bits(y))
    a = rng.uniform(0.05, 0.3, (4, 3, 5))
    h = np.sort(rng.uniform(0.1, 4.0, (4, 3, 5)), -1)
    args = (a, h, a * h * 0.2, np.full((4, 3, 5), 6.0), np.full((4, 3), 2e-6), 5, RDT)
    gen_c, gen_p = cur.apply_ridging(*args, V_pond_cat=a * 0.01), par.apply_ridging(*args, V_pond_cat=a * 0.01)
    assert sorted(gen_c) == sorted(gen_p)
    for k in gen_c:
        np.testing.assert_array_equal(_bits(gen_c[k]), _bits(gen_p[k]))


@pytest.mark.slow
@pytest.mark.skipif(not _RHEO_ROOT.exists(), reason="lane-3 ICE_RHEO record absent")
def test_lane3_rheo_card_parent_vs_current_bit_identical():
    """One ICE_RHEO step (rhg -> adv -> rdgrft -> cor) with the parent's ridging/ITD/surface modules."""
    import importlib.util

    import netCDF4
    from legoesm.ice.fidelity import nemo_rheo_testcase_recipe as rheo_cur

    from legoesm import constants as C

    root = __import__("pathlib").Path(__file__).resolve().parents[3]
    par = _load_parent((("legoesm.ice.fidelity.nemo_rheo_testcase_recipe",
                         "packages/ice/legoesm/ice/fidelity/nemo_rheo_testcase_recipe.py"),))
    rheo_par = par["legoesm.ice.fidelity.nemo_rheo_testcase_recipe"]
    assert rheo_par.apply_si3_jpl1_ridging.__module__ != rheo_cur.apply_si3_jpl1_ridging.__module__ or (
        rheo_par.apply_si3_jpl1_ridging is not rheo_cur.apply_si3_jpl1_ridging)
    spec = importlib.util.spec_from_file_location(
        "o6_gate", root / "scripts/validate/ocean_fidelity/testcases/nemo_si3_oracle_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    _, frame = gate.read_frame(_RHEO_ROOT / "oracle_ice_step_entry_kt00000001.bin")
    names = ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f", "tmask", "umask", "vmask")
    with netCDF4.Dataset(_RHEO_ROOT / "mesh_mask.nc") as ds:
        mesh = {n: np.asarray(ds[n][:]) for n in names}
    with netCDF4.Dataset(_RHEO_ROOT / "output.init_ice.nc") as ds:
        sst_k = np.asarray(ds["sst"][0]) + C.T_freeze
    rc, rp = rheo_cur.build_ice_rheo_card(frame, mesh, sst_k), rheo_par.build_ice_rheo_card(frame, mesh, sst_k)
    sc = rheo_cur.step_ice_rheo_card(rc, rc.initial_state, completed_steps=0)
    sp = rheo_par.step_ice_rheo_card(rp, rp.initial_state, completed_steps=0)
    for x, y in zip(jax.tree_util.tree_leaves(sc), jax.tree_util.tree_leaves(sp)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
