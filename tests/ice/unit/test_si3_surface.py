"""SI3 ice-atmosphere surface statements (blk_ice_1, ice_alb, blk_ice_2, ice_flx_other).

The oracle here is an independent per-cell Python transcription of the
``ORCA2_SAS_ICE_O2_REF`` ppsrc statements, written from NEMO's own literals:
CPython float arithmetic is IEEE binary64 with no contraction, and
``math.exp/log/log10/pow`` call the same glibc as NEMO.  The model must equal
it bit for bit under production ``jit`` on EVERY output and statement
intermediate the model exposes (the key sets are asserted equal).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

jax.config.update("jax_enable_x64", True)

from legoesm import thermo  # noqa: E402
from legoesm.core import bulk_flux  # noqa: E402
from legoesm.ice import si3_surface as o7  # noqa: E402
from legoesm.ice.dynamics import SI3ORCA2FoldProgram  # noqa: E402
from legoesm.ice.si3_orca2_state import fortran_sum_categories, si3_lbc_lnk_orca2  # noqa: E402

DATA = Path(__file__).parent / "data"
DECK = DATA / "orca2_sas_ice_expref"
NEMO_CFGS = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs")
SHA = {
    # Admitted O2 run (si3_rounds/round7/oracle_o2_orca2_sas_ice/ref_2step/output.namelist.{dyn,ice}),
    # the ln_icedyn = ln_icethd = F diagnostic overlay (DECISION 159).
    "si3_o2_ref_output.namelist.dyn": "27a4ebeb6196546e1dae94fc09317ef38cf2712b1e4c3ddf31f4892cc5c4b16f",
    "si3_o2_ref_output.namelist.ice": "dfffdc445122bdcd8efc5f2791de67b784c79aa538d139f42fa5169e01caf985",
    # Lane-3 C1D record (nemo-testcases-l3/c1d_omip_l3_sasice_scalarmath_v2_a), a deck without ponds.
    "c1d_omip_l3_v2_output.namelist.dyn": "3f1e09f4eae0f70c4edd424e86f6349a9b36353d31f45c4f0800fb2e4bf2db9e",
    "c1d_omip_l3_v2_output.namelist.ice": "d2173437dfb9cfd9de8f82a996d4c26374c4bfce4fe66f0265c693af453d24e1",
    # Production ORCA2_SAS_ICE as shipped: cfgs/SHARED/namelist_{ref,ice_ref}, cfgs/ORCA2_SAS_ICE/EXPREF/namelist_{cfg,ice_cfg}.
    "orca2_sas_ice_expref/namelist_ref": "b04f2ce4d12aa247ae3b707483d25cf2d38d0b3cf8817d71d33e66918fd81cfd",
    "orca2_sas_ice_expref/namelist_cfg": "3636a977aec1bfaa6ca1948397b8fb3ee8de721bde8713be1990a1e9c29bda8a",
    "orca2_sas_ice_expref/namelist_ice_ref": "055ab5e6839c12dbae61bf1c3bcdda46a8f95cf275b7f294d68d7f8e73179307",
    "orca2_sas_ice_expref/namelist_ice_cfg": "9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8",
}
H, NI, NJ, JPL = 2, 8, 6, 5
TF = constants.T_freeze


@pytest.fixture(autouse=True)
def libm_policy():
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(previous)


def _path(name):
    path = DATA / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA[name]
    return path


def _o2_config():
    return o7.SI3SurfaceConfig.from_resolved_namelists(
        _path("si3_o2_ref_output.namelist.dyn"), _path("si3_o2_ref_output.namelist.ice"))


def _prod_config():
    """The production deck, read from its own shipped namelists merged as NEMO reads them."""
    return o7.SI3SurfaceConfig.from_input_namelists(
        _path("orca2_sas_ice_expref/namelist_ref"), _path("orca2_sas_ice_expref/namelist_cfg"),
        _path("orca2_sas_ice_expref/namelist_ice_ref"), _path("orca2_sas_ice_expref/namelist_ice_cfg"))


def _c1d_config():
    return o7.SI3SurfaceConfig.from_resolved_namelists(
        _path("c1d_omip_l3_v2_output.namelist.dyn"), _path("c1d_omip_l3_v2_output.namelist.ice"))


FOLD = SI3ORCA2FoldProgram(nn_hls=H, nfd_full_when_absent=True)


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    u = lambda lo, hi, shape=(NI, NJ): rng.uniform(lo, hi, shape)  # noqa: E731
    cat = (NI, NJ, JPL)
    full = (NI + 2 * H, NJ + 2 * H)
    x = dict(
        wndi=u(-15, 15), wndj=u(-15, 15), theta_air_zt=u(240, 280), q_air_zt=u(1e-4, 4e-3),
        slp=u(96000, 104000), tm_su=u(240, 273), rhoa=u(1.2, 1.45),
        t_su=u(240, 275, cat), h_i=u(0.0, 3.0, cat), h_s=u(0.0, 0.4, cat),
        a_ip_eff=u(0.0, 0.6, cat), h_ip=u(0.0, 0.3, cat), h_il=u(0.0, 0.04, cat),
        cloud_fra=u(0, 1), smask0=np.ones((NI, NJ)), qlw=u(150, 320), precip=u(0, 1e-4),
        snow=u(0, 5e-5), qsr=u(0, 300), sst_m=u(-1.9, 2.0), emp=u(-5e-5, 5e-5),
        a_i_b=u(0.0, 0.2, cat), qns_oce=u(-300, 50), qsr_oce=u(0, 250),
        u_ice=u(-0.3, 0.3, full), v_ice=u(-0.3, 0.3, full), ssu_m=u(-0.2, 0.2, full),
        ssv_m=u(-0.2, 0.2, full), utau=u(-0.3, 0.3), vtau=u(-0.3, 0.3), vt_i=u(0, 5),
        frq_m=u(0.3, 1.0), e3t_m=u(5, 15), t_bo=u(271.0, 271.4), rn_amax_2d=np.full((NI, NJ), 0.997),
    )
    # Branch coverage in fixed cells.
    x["smask0"][0, 0] = 0.0                                  # land
    x["h_i"][1, 0, :] = [0.02, 0.05, 0.5, 1.0, 2.5]           # thin / break / mid / pivot / thick
    x["h_s"][1, 1, :] = [0.0, 0.0, 1e-3, 0.0, 0.2]            # snow-free predicate both ways
    x["h_i"][1, 1, :] = [0.05, 0.08, 0.3, 0.09, 1.2]          # with h_i < 0.1 for transmission
    x["t_su"][1, 1, :] = [TF, TF + 0.1, TF + 0.1, 250.0, TF]   # no-pond melt arm: h_s == 0, t_su >= rt0, mid h_i
    x["t_su"][2, 0, :] = [TF, TF + 0.05, 260.0, TF, 250.0]    # melting / freezing
    x["h_il"][2, 1, :] = [0.0, 0.02, 0.0200001, 0.03, 0.01]   # lid threshold
    x["a_i_b"][3, 0, :] = 0.0                                 # open ocean: SUM of -0 terms
    x["q_air_zt"][3, 0] = 5e-2                                # deposition => evap_ice < 0
    x["theta_air_zt"][3, 2] = 280.0                           # MIN(ptair, rt0) both ways
    x["t_bo"][3, 1] = x["sst_m"][3, 1] + TF + 0.01            # supercooling
    x["vt_i"][3, 1] = 25.0
    x["vt_i"][4, 0] = 4.0                                     # landfast / full-cover stops
    x["u_ice"][H + 4, H + 0] = x["u_ice"][H + 3, H + 0] = 0.0
    x["v_ice"][H + 4, H + 0] = x["v_ice"][H + 4, H - 1] = 0.0
    x["sst_m"][5, 2] = -0.5                                   # warm leads (fhld branch)
    x["qns_oce"][5, 2] = 200.0
    # Thin dry snow: XLA's native exp(-h_s/0.02) differs from glibc here by an ULP that survives into alb
    # (searched; ponds, cloud and thick snow elsewhere round that ULP away), so the native plant binds.
    x["h_s"][6, 4, 0], x["t_su"][6, 4, 0], x["h_i"][6, 4, 0], x["a_ip_eff"][6, 4, 0] = 0.0066, 250.0, 1.0, 0.0
    x["cloud_fra"][6, 4] = 0.5
    for cell in ((5, 3), (4, 0), (4, 1)):                     # cooling leads at freezing SST
        x["sst_m"][cell] = x["t_bo"][cell] - TF
        x["qns_oce"][cell], x["qsr_oce"][cell] = -300.0, 0.0
    at_i_b = x["a_i_b"].sum(axis=2)
    x["at_i_b"] = at_i_b
    x["at_i"] = at_i_b.copy()
    x["at_i"][4, 0] = 0.9975
    x["at_i"][4, 1] = 1.0
    x["vt_i"][4, 1] = 3.5
    return x


def _step(x, config, fold=FOLD):
    return o7.si3_surface_step(**{k: jnp.asarray(v) for k, v in x.items()}, config=config, fold=fold)


# --------------------- independent scalar oracle (NEMO literals) ---------------------
RT0, RTT0, RPREF = 273.15, 273.16, 1.0e5  # const-ok: independent NEMO-literal oracle (phycst.f90:50, sbc_phy.f90:55,73)
RGAMMA = 8.314510 / (28.9647e-3 * 1005.0)  # sbc_phy.f90:50-54
REPS0 = 287.05 / 461.495  # const-ok: independent NEMO-literal oracle (sbc_phy.f90:44-46)
AG, BG, CG, DG = -9.09718, -3.56654, 0.876793, math.log10(6.1071)
RHO0, RCP, RCPI, RLFUS, RLSUB, RHOS = 1026.0, 3991.86795711963, 2096.7, 0.3333601e6, 2.8344e6, 330.0
RCP_AIR, EMISS, STEF, ALBO = 1000.5, 0.996, 5.67e-8, 0.066


def s_esat(t):
    zta = t if not (180.0 > t) else 180.0
    ztmp = RTT0 / zta
    zle = AG * (ztmp - 1.0) + BG * math.log10(ztmp) + CG * (1.0 - zta / RTT0) + DG
    return 100.0 * math.pow(10.0, zle)


def s_qsat(t, p):
    e = s_esat(t)
    return REPS0 * e / (p - (1.0 - REPS0) * e)


def s_dqsat(t, p):
    e = s_esat(t)
    zta = t if not (180.0 > t) else 180.0
    zde = -(AG * RTT0) / (zta * zta) - BG / (zta * math.log(10.0)) - CG / RTT0
    de = math.log(10.0) * zde * s_esat(zta)
    tmp = (REPS0 - 1.0) * e + p
    return REPS0 * p * de / (tmp * tmp)


def s_theta(t, p):
    return t * math.pow(RPREF / p, RGAMMA)


def s_alb(c, ts, hi, hs, ap, hp, hl, cf, m):
    fs = hs / (hs + 0.02)
    if c.ld_pnd_alb:
        fp = ap if not ((1.0 - fs) < ap) else (1.0 - fs)
        ai = c.rn_alb_idry
    else:
        fp = 0.0
        ai = c.rn_alb_imlt if (hs == 0.0 and ts >= RT0) else c.rn_alb_idry
    v = 1.0 - fp - fs
    fi = 0.0 if not (v > 0.0) else v
    z1c1 = 1.0 / (math.log(c.rn_alb_hpiv) - math.log(0.05))
    zh = math.log(c.rn_alb_hpiv)
    if 0.05 < hi and hi <= c.rn_alb_hpiv:
        ai = ai + (0.18 - ai) * z1c1 * (zh - math.log(hi))
    elif hi <= 0.05:
        ai = 0.066 + (0.18 - 0.066) * (1.0 / 0.05) * hi
    if ts < RT0:
        asn = c.rn_alb_sdry - (c.rn_alb_sdry - ai) * math.exp(-hs * (1.0 / 0.02))
    else:
        asn = c.rn_alb_smlt - (c.rn_alb_smlt - ai) * math.exp(-hs * (1.0 / 0.03))
    if hl <= 0.02:
        apn = c.rn_alb_dpnd - (c.rn_alb_dpnd - ai) * math.exp(-hp * (1.0 / 0.05))
    else:
        apn = c.rn_alb_lpnd - (c.rn_alb_lpnd - ai) * math.exp(-hp * (1.0 / 0.05))
    aos = (fs * asn + fp * apn + fi * ai) * m
    acs = aos - (-0.1010 * aos * aos + 0.1933 * aos - 0.0148) * m
    return (1.0 - cf) * acs + cf * aos


def _sum(terms):
    s = 0.0
    for t in terms:
        s = s + t
    return s


def s_cell(c, x, i, j):
    """Scalar oracle: every model output and exposed intermediate for interior cell (i, j)."""
    g = lambda k: float(x[k][i, j])  # noqa: E731
    gc = lambda k, l: float(x[k][i, j, l])  # noqa: E731
    o = {}
    # blk_ice_1 (sbcblk.f90:1082-1098,1137-1139)
    wi, wj, ta, qa, p, rho = g("wndi"), g("wndj"), g("theta_air_zt"), g("q_air_zt"), g("slp"), g("rhoa")
    wndm = math.sqrt(wi * wi + wj * wj)
    z = rho * c.rn_Cd_ia * wndm
    o.update({"b1.wndm_ice": wndm, "b1.zsipt": s_theta(g("tm_su"), p), "b1.Cd_ice": c.rn_Cd_ia,
              "b1.Ch_ice": c.rn_Ch_ia, "b1.Ce_ice": c.rn_Ce_ia, "b1.theta_zu_i": ta, "b1.q_zu_i": qa,
              "b1.utau_ice": z * wi, "b1.vtau_ice": z * wj})
    # ice_alb + blk_ice_2 (sbcblk.f90:1210-1331)
    m = g("smask0")
    zztmp = 1.0 / (1.0 - ALBO)
    crain = (ta - RT0) * RCP * m
    csnw = ((RT0 if RT0 < ta else ta) - RT0) * RCPI * m
    cptn = g("sst_m") * RCP * m
    o.update({"b2.zcptrain": crain, "b2.zcptsnw": csnw, "b2.zcptn": cptn})
    per = {k: [] for k in ("alb", "b2.qsr_ice", "b2.qns_ice", "b2.dqns_ice", "b2.qla_ice", "b2.dqla_ice",
                           "b2.evap_ice", "b2.devap_ice", "b2.zsq", "b2.zsipt", "b2.z_qlw", "b2.z_dqlw",
                           "b2.z_qsb", "b2.qtr_ice_top", "b2.qevap_ice")}
    blk = rho * wndm
    t1 = blk * RCP_AIR * c.rn_Ch_ia
    t2 = blk * RLSUB * c.rn_Ce_ia
    ztri = 0.18 * (1.0 - g("cloud_fra")) + 0.35 * g("cloud_fra")
    for l in range(JPL):
        ts = gc("t_su", l)
        a = s_alb(c, ts, gc("h_i", l), gc("h_s", l), gc("a_ip_eff", l), gc("h_ip", l), gc("h_il", l), g("cloud_fra"), m)
        zsq, zsipt = s_qsat(ts, p), s_theta(ts, p)
        q = zztmp * (1.0 - a) * g("qsr")
        st3 = ts * ts * ts
        qlw = EMISS * (g("qlw") - STEF * ts * st3) * m
        dqlw = (4.0 * EMISS * STEF) * st3
        qsb = t1 * (zsipt - ta)
        la = t2 * (zsq - qa)
        dla = t2 * s_dqsat(ts, p)
        if gc("h_s", l) <= 0.0:
            t = ztri
            if gc("h_i", l) < 0.1:
                t = t + (1.0 - t) * (1.0 - gc("h_i", l) * 10.0)
            qtr = q * t
        else:
            qtr = 0.0
        for k, v in (("alb", a), ("b2.qsr_ice", q), ("b2.qns_ice", qlw - qsb - la), ("b2.dqns_ice", -(dqlw + t1 + dla)),
                     ("b2.qla_ice", la), ("b2.dqla_ice", dla), ("b2.evap_ice", c.rn_efac * la * (1.0 / RLSUB)),
                     ("b2.devap_ice", c.rn_efac * dla * (1.0 / RLSUB)), ("b2.zsq", zsq), ("b2.zsipt", zsipt),
                     ("b2.z_qlw", qlw), ("b2.z_dqlw", dqlw), ("b2.z_qsb", qsb), ("b2.qtr_ice_top", qtr),
                     ("b2.qevap_ice", 0.0)):
            per[k].append(v)
    o.update(per)
    o["b2.z_dqsb"] = t1
    tp, sp = g("precip") * c.rn_pfac * m, g("snow") * c.rn_pfac * m
    zevap = g("emp") + tp
    atb = g("at_i_b")
    zsnw = 1.0 - math.pow(1.0 - atb, c.rn_snwblow)
    emp_oce = (1.0 - atb) * zevap - (tp - sp) - sp * (1.0 - zsnw)
    emp_ice = _sum(gc("a_i_b", l) * per["b2.evap_ice"][l] for l in range(JPL)) - sp * zsnw
    qemp_oce = -(1.0 - atb) * zevap * cptn + (tp - sp) * crain + sp * (1.0 - zsnw) * (csnw - RLFUS)
    qemp_ice = sp * zsnw * (csnw - RLFUS)
    o.update({
        "b2.tprecip": tp, "b2.sprecip": sp, "b2.zevap": zevap, "b2.zsnw": zsnw, "b2.emp_oce": emp_oce,
        "b2.emp_ice": emp_ice, "b2.emp_tot": emp_oce + emp_ice, "b2.qemp_oce": qemp_oce, "b2.qemp_ice": qemp_ice,
        "b2.qns_tot": (1.0 - atb) * g("qns_oce") + _sum(gc("a_i_b", l) * per["b2.qns_ice"][l] for l in range(JPL))
        + qemp_ice + qemp_oce,
        "b2.qsr_tot": (1.0 - atb) * g("qsr_oce") + _sum(gc("a_i_b", l) * per["b2.qsr_ice"][l] for l in range(JPL)),
        "b2.qprec_ice": RHOS * (csnw - RLFUS),
    })
    # ice_flx_other (icesbc.f90:385-482)
    if c.ln_icedyn:
        U = lambda k, di=0, dj=0: float(x[k][i + H + di, j + H + dj])  # noqa: E731
        zu, zum = U("u_ice") - U("ssu_m"), U("u_ice", -1) - U("ssu_m", -1)
        zv, zvm = U("v_ice") - U("ssv_m"), U("v_ice", 0, -1) - U("ssv_m", 0, -1)
        fric = c.rn_Cd_io * (0.5 * ((zu * zu + zum * zum) + (zv * zv + zvm * zvm))) * m
        a1, a2 = U("u_ice", -1) + U("u_ice"), U("v_ice", 0, -1) + U("v_ice")
        vel = 0.5 * math.sqrt(a1 * a1 + a2 * a2)
    else:
        fric = (1.0 / RHO0) * math.sqrt(g("utau") * g("utau") + g("vtau") * g("vtau")) * m
        vel = 0.0
    at, vt, dt = g("at_i"), g("vt_i"), c.rDt_ice
    r1 = 1.0 / dt
    sw = m if at >= 1e-10 else 0.0
    qld = m * dt * ((1.0 - atb) * g("qsr_oce") * g("frq_m") + (1.0 - atb) * g("qns_oce") + qemp_oce)
    sstk = g("sst_m") + RT0
    qfr = RHO0 * RCP * g("e3t_m") * (g("t_bo") - sstk) * m
    qneg = qfr if not (0.0 < qfr) else 0.0
    qpos = qfr if not (0.0 > qfr) else 0.0
    fu = math.sqrt(fric)
    fu = fu if not (0.0 > fu) else 0.0
    qsb = sw * RHO0 * RCP * 0.0057 * fu * (sstk - g("t_bo"))
    den = at if not (1e-10 > at) else 1e-10
    cap = -qneg * r1 / den
    qsb = sw * (cap if cap <= qsb else qsb)  # gfortran MINSD returns RHS on ties
    if (g("t_bo") - sstk) > 0.0 and vt >= 20.0:
        qfr = qpos = qsb = 0.0
    if (qld - qfr) < 0.0:
        fhld = 0.0
        e = qld - qfr
        qlead = 0.0 if not (e < 0.0) else e
    else:
        v = (qld - qpos) * r1 / den
        fhld = sw * (0.0 if not (v > 0.0) else v)
        qlead = 0.0
    if vel <= 5e-3 and at >= (g("rn_amax_2d") - 0.001) and vt >= 3.0:
        qlead = 0.0
    if at >= (1.0 - 1e-10) and vt >= 3.0:
        qlead = 0.0
    o.update({"fo.qsb_ice_bot": qsb, "fo.fhld": fhld, "fo.qlead": qlead})
    return o


def lbc_t_oracle(interior, psgn):
    """lbc_lnk on a T-point field, jpni = 1, ORCA2 (cyclic E-W, T-pivot fold, closed south), 1-based.

    Order and extents as lbc_lnk_pt2pt_generic.h90: E-W halos over the Nj_0 inner rows (step 4), self
    north fold lbc_nfd_generic.h90:22-75 (step 6), south halo = zland over all ipi (step 7). Halos start
    as NaN: every halo cell must be written.
    """
    ni0, nj0, hl = len(interior), len(interior[0]), H
    ipi, ipj = ni0 + 2 * hl, nj0 + 2 * hl
    t = {(i, j): math.nan for i in range(1, ipi + 1) for j in range(1, ipj + 1)}
    for i in range(ni0):
        for j in range(nj0):
            t[(i + hl + 1, j + hl + 1)] = interior[i][j]
    for jj in range(hl + 1, hl + nj0 + 1):
        for ji in range(1, hl + 1):
            t[(ji, jj)] = t[(ipi - 2 * hl + ji, jj)]                 # west <- im1i
            t[(ipi - hl + ji, jj)] = t[(hl + ji, jj)]                # east <- ip1i
    for jj in range(1, hl + 1):
        ij1, ij2 = ipj - jj + 1, ipj - 2 * hl + jj - 1
        pairs = [(ji, 2 * hl + 2 - ji) for ji in range(1, hl + 1)] + [(hl + 1, hl + 1)]
        pairs += [(2 + hl + ji - 1, ipi - hl - ji + 1) for ji in range(1, ni0)]
        pairs += [(ipi - hl + 1, hl + 1)] + [(ipi - hl + 1 + ji, ipi - hl + 1 - ji) for ji in range(1, hl)]
        for ii1, ii2 in pairs:
            t[(ii1, ij1)] = psgn * t[(ii2, ij2)]
    ij1 = ipj - hl
    for ii1, ii2 in [(ipi // 2 + ji + 1, ipi // 2 - ji + 1) for ji in range(1, ni0 // 2)] + \
            [(ji, 2 * hl + 2 - ji) for ji in range(1, hl + 1)]:
        t[(ii1, ij1)] = psgn * t[(ii2, ij1)]
    for jj in range(1, hl + 1):
        for ji in range(1, ipi + 1):
            t[(ji, jj)] = 0.0
    return np.array([[t[(i, j)] for j in range(1, ipj + 1)] for i in range(1, ipi + 1)])


def _bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def _model_view(res):
    view = {"alb": res.alb_ice}
    for prefix, tup in (("b1", res.blk_ice_1), ("b2", res.blk_ice_2), ("fo", res.flx_other)):
        for name, value in tup._asdict().items():
            value = np.asarray(value)
            view[f"{prefix}.{name}"] = value[..., 0] if value.shape == (NI, NJ, 1) else value
    view["utau_ice"], view["vtau_ice"] = res.utau_ice, res.vtau_ice  # full halo, after icesbc.f90:167
    return view


def _oracle_mismatches(res, x, c):
    view = {k: np.asarray(v) for k, v in _model_view(res).items()}
    bad = []
    cells = [[s_cell(c, x, i, j) for j in range(NJ)] for i in range(NI)]
    for i in range(NI):
        for j in range(NJ):
            cell = cells[i][j]
            assert set(cell) | {"utau_ice", "vtau_ice"} == set(view), set(cell) ^ set(view)
            for k, v in cell.items():
                if not np.array_equal(_bits(view[k][i, j]), _bits(np.asarray(v, dtype=np.float64))):
                    bad.append((k, i, j))
    for k in ("utau_ice", "vtau_ice"):   # icesbc.f90:167: lbc_lnk(..., 'T', -1.0_wp, ...)
        want = lbc_t_oracle([[cells[i][j][f"b1.{k}"] for j in range(NJ)] for i in range(NI)], -1.0)
        assert view[k].shape == want.shape and np.all(np.isfinite(want))
        bad += [(k, int(i), int(j)) for i, j in np.argwhere(_bits(view[k]) != _bits(want))]
    return bad


def _jit_step(x, c):
    return jax.jit(lambda xx: _step(xx, c))({k: jnp.asarray(v) for k, v in x.items()})


# ----------------------------------- tests -----------------------------------
def test_production_config_is_read_from_the_shipped_deck():
    c = _prod_config()
    assert c.rn_Cd_ia == c.rn_Ce_ia == c.rn_Ch_ia == float("1.4e-3")
    assert (c.ln_icedyn, c.ld_pnd_alb, c.rDt_ice, c.nn_hls, c.nn_snwfra) == (True, True, 10800.0, 2, 2)
    # The admitted O2 record differs from production only by its ln_icedyn = F overlay.
    o2 = _o2_config()
    assert {f for f in c._fields if getattr(c, f) != getattr(o2, f)} == {"ln_icedyn"}
    assert (o2.rn_alb_dpnd, o2.rn_alb_lpnd, o2.rn_snwblow) == (0.18, 0.3, 0.66)
    shipped = {
        "orca2_sas_ice_expref/namelist_ref": NEMO_CFGS / "SHARED/namelist_ref",
        "orca2_sas_ice_expref/namelist_ice_ref": NEMO_CFGS / "SHARED/namelist_ice_ref",
        "orca2_sas_ice_expref/namelist_cfg": NEMO_CFGS / "ORCA2_SAS_ICE/EXPREF/namelist_cfg",
        "orca2_sas_ice_expref/namelist_ice_cfg": NEMO_CFGS / "ORCA2_SAS_ICE/EXPREF/namelist_ice_cfg",
    }
    if all(p.exists() for p in shipped.values()):
        for name, src in shipped.items():
            assert (DATA / name).read_bytes() == src.read_bytes(), name


def test_config_has_no_defaults_and_fails_closed(tmp_path):
    bad = tmp_path / "dyn"
    bad.write_text(_path("si3_o2_ref_output.namelist.dyn").read_text().replace(" RN_CD_IA=", " XX="))
    with pytest.raises(ValueError, match="rn_cd_ia"):
        o7.SI3SurfaceConfig.from_resolved_namelists(bad, _path("si3_o2_ref_output.namelist.ice"))
    with pytest.raises(TypeError):
        o7.SI3SurfaceConfig()  # type: ignore[call-arg]


def test_folded_intrinsic_constants_are_correctly_rounded():
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.prec = 200
    assert o7._LOG_ALB_THIN_BREAK == float(mpmath.log(mpmath.mpf(0.05)))
    assert thermo._NEMO_LN10 == float(mpmath.log(mpmath.mpf(10.0)))
    assert thermo._NEMO_GOFF_ICE_D == float(mpmath.log10(mpmath.mpf(6.1071)))


def test_libm_pow_matches_glibc_and_differentiates():
    from legoesm.core import transcendentals as tr
    base = jnp.asarray([0.0, 0.3, 1.7, 10.0])
    got = np.asarray(jax.jit(lambda b: tr.pow(b, 0.66))(base))
    assert [float(v) for v in got] == [math.pow(float(b), 0.66) for b in base]
    grad = jax.grad(lambda b: tr.pow(b, 0.66).sum())(base)
    assert np.all(np.isfinite(np.asarray(grad))) and float(grad[0]) == 0.0
    assert abs(float(grad[2]) - 0.66 * 1.7 ** -0.34) < 1e-12


@pytest.mark.parametrize("deck", ["production", "o2_no_dynamics", "c1d_no_ponds"])
def test_jit_equals_independent_glibc_transcription_on_every_output(deck):
    c = {"production": _prod_config, "o2_no_dynamics": _o2_config, "c1d_no_ponds": _c1d_config}[deck]()
    x = _inputs()
    res = _jit_step(x, c)
    assert _oracle_mismatches(res, x, c) == []


def test_branches_bind_on_the_production_deck():
    c, x = _prod_config(), _inputs()
    res = _jit_step(x, c)
    b2, fo = res.blk_ice_2, res.flx_other
    assert float(fo.fhld[5, 2]) > 0.0 and float(fo.qlead[5, 3]) < 0.0
    assert float(fo.qsb_ice_bot[3, 1]) == 0.0
    assert float(fo.qlead[4, 0]) == 0.0 and float(fo.qlead[4, 1]) == 0.0
    thin = dict(x, vt_i=x["vt_i"].copy())
    thin["vt_i"][4, 0] = thin["vt_i"][4, 1] = 1.0                              # stops bind
    unstopped = _jit_step(thin, c).flx_other.qlead
    assert float(unstopped[4, 0]) < 0.0 and float(unstopped[4, 1]) < 0.0
    assert float(np.asarray(b2.qtr_ice_top)[1, 1, 0]) > 0.0 and float(np.asarray(b2.qtr_ice_top)[1, 1, 2]) == 0.0
    assert np.all(np.asarray(res.blk_ice_1.Cd_ice) == c.rn_Cd_ia)
    # The no-pond melt arm (h_s == 0 and t_su >= rt0) changes the albedo only on the no-pond deck.
    nopond = _jit_step(x, _c1d_config()).alb_ice
    assert c.ld_pnd_alb and not _c1d_config().ld_pnd_alb
    assert float(nopond[1, 1, 1]) != float(_jit_step(x, _c1d_config()._replace(rn_alb_imlt=0.64)).alb_ice[1, 1, 1])


def _zsipt_has_no_consumer(c, x):
    """Perturb only tm_su: zsipt must move and nothing else may (sbcblk.f90:1086 is read only by :1102,1107,1112)."""
    base = _jit_step(x, c)
    moved = dict(x, tm_su=x["tm_su"] + 7.0)
    other = _jit_step(moved, c)
    assert not np.array_equal(_bits(base.blk_ice_1.zsipt), _bits(other.blk_ice_1.zsipt))
    leaves = lambda r: [v for k, v in _model_view(r).items() if k != "b1.zsipt"]  # noqa: E731
    return all(np.array_equal(_bits(a), _bits(b)) for a, b in zip(leaves(base), leaves(other)))


def test_blk_ice_1_zsipt_is_transcribed_and_has_no_consumer_on_this_arm():
    c, x = _prod_config(), _inputs(4)
    assert _zsipt_has_no_consumer(c, x)


def test_zsipt_consumer_plant_is_caught(monkeypatch):
    real = bulk_flux.nemo_si3_blk_ice_1_cst

    def planted(*args):
        out = real(*args)
        return out._replace(theta_zu_i=out.zsipt)       # a consumer that wrongly reads zsipt

    monkeypatch.setattr(o7, "nemo_si3_blk_ice_1_cst", planted)
    assert not _zsipt_has_no_consumer(_prod_config(), _inputs(4))


def test_inexact_value_pins():
    c, x = _prod_config(), _inputs()
    res = _jit_step(x, c)
    got = {
        "alb": float(res.alb_ice[1, 0, 2]),
        "qns_tot": float(res.blk_ice_2.qns_tot[2, 3]),
        "qsb": float(res.flx_other.qsb_ice_bot[2, 2]),
        "zsipt": float(res.blk_ice_1.zsipt[2, 2]),
    }
    ref = {
        "alb": s_cell(c, x, 1, 0)["alb"][2],
        "qns_tot": s_cell(c, x, 2, 3)["b2.qns_tot"],
        "qsb": s_cell(c, x, 2, 2)["fo.qsb_ice_bot"],
        "zsipt": s_cell(c, x, 2, 2)["b1.zsipt"],
    }
    for key, v in got.items():
        assert v == ref[key] and v != round(v, 6), key


@pytest.mark.parametrize("plant", ["cd_ulp", "rtt0_ulp", "sum_first_category", "ln10_ulp", "snwfra", "native_exp", "fold_sign"])
def test_production_path_plants_fire(plant, monkeypatch):
    c, x = _prod_config(), _inputs()
    if plant == "cd_ulp":
        c = c._replace(rn_Cd_ia=math.nextafter(c.rn_Cd_ia, 1.0))
    elif plant == "rtt0_ulp":
        monkeypatch.setattr(constants, "T_triple_nemo", math.nextafter(constants.T_triple_nemo, 300.0))
    elif plant == "sum_first_category":
        monkeypatch.setattr(o7, "_sum0", lambda t: fortran_sum_categories(t, axis=-1))
    elif plant == "ln10_ulp":
        monkeypatch.setattr(thermo, "_NEMO_LN10", math.nextafter(thermo._NEMO_LN10, 3.0))
    elif plant == "snwfra":
        c = c._replace(nn_snwfra=1)
    elif plant == "native_exp":
        set_policy(PrecisionPolicy.fp64(transcendentals="native"))
    elif plant == "fold_sign":
        monkeypatch.setattr(o7, "si3_lbc_lnk_orca2", lambda *a, **k: si3_lbc_lnk_orca2(*a, **k, _plant_wrong_nfd_sign=True))
    mismatches = _oracle_mismatches(_jit_step(x, c), x, _prod_config())
    assert mismatches != []
    if plant == "native_exp":
        assert ("alb", 6, 4) in mismatches
    if plant == "fold_sign":
        assert {k for k, _, _ in mismatches} == {"utau_ice", "vtau_ice"}


def test_eager_disable_jit_equals_jit():
    c, x = _prod_config(), _inputs(1)
    xx = {k: jnp.asarray(v) for k, v in x.items()}
    jitted = jax.jit(lambda a: _step(a, c))(xx)
    with jax.disable_jit():
        eager = _step(xx, c)
    leaves = list(zip(jax.tree_util.tree_leaves(jitted), jax.tree_util.tree_leaves(eager)))
    assert len(leaves) == 9 + 2 + 1 + 30 + 3
    for a, b in leaves:
        assert np.array_equal(_bits(a), _bits(b))


def test_no_dynamics_arm_and_stress_exchange():
    c, x = _o2_config(), _inputs(2)
    res = _jit_step(x, c)
    sw = 1.0 if x["at_i"][2, 2] >= 1e-10 else 0.0
    assert sw == 1.0 and float(res.flx_other.qsb_ice_bot[2, 2]) != 0.0
    assert not np.array_equal(_bits(res.flx_other.qsb_ice_bot), _bits(_jit_step(x, _prod_config()).flx_other.qsb_ice_bot))
    full = np.asarray(res.utau_ice)
    assert full.shape == (NI + 2 * H, NJ + 2 * H)
    np.testing.assert_array_equal(full[:H, H:-H], full[NI:NI + H, H:-H])          # cyclic seam
    assert np.all(full[:, :H] == 0.0)                                              # closed south
    plant = np.asarray(si3_lbc_lnk_orca2(jnp.pad(res.blk_ice_1.utau_ice, H), "T", 1.0, nn_hls=H, ldfull=True))
    assert not np.array_equal(plant[:, -H:], full[:, -H:])                         # fold sign binds


def test_gradient_finite_and_central_difference():
    c, x = _prod_config(), _inputs(3)
    base = {k: jnp.asarray(v) for k, v in x.items()}

    def loss(t_cell):
        xx = dict(base)
        xx["t_su"] = base["t_su"].at[2, 3, 1].set(t_cell)
        return _step(xx, c).blk_ice_2.qns_tot.sum()

    t0 = float(base["t_su"][2, 3, 1])
    g = float(jax.grad(loss)(t0))
    eps = 1e-3
    fd = (float(loss(t0 + eps)) - float(loss(t0 - eps))) / (2 * eps)
    assert math.isfinite(g) and g != 0.0 and abs(g / fd - 1.0) < 1e-5
    grads = jax.grad(lambda xx: _step(xx, c).blk_ice_2.qns_tot.sum())(base)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in grads.values())


def test_no_array_defaults_and_refused_arms():
    c, x = _prod_config(), _inputs()
    xx = {k: jnp.asarray(v) for k, v in x.items()}
    for name in ("wndi", "rhoa", "a_ip_eff", "emp", "t_bo", "u_ice", "tm_su", "slp"):
        bad = dict(xx)
        bad[name] = None
        with pytest.raises(ValueError, match=name):
            o7.si3_surface_step(**bad, config=c, fold=FOLD)
    for field, value in (("nn_qtrice", 1), ("ln_Cx_ice_cst", False),
                         ("nn_flxdist", 0), ("ln_cndflx", True), ("ln_leadhfx", False)):
        with pytest.raises(ValueError, match="not transcribed"):
            _step(xx, c._replace(**{field: value}))
    with pytest.raises(TypeError):
        o7.si3_ice_sbc_tau_lbc(xx["wndi"], xx["wndj"], fold=(2, True))


def test_subnormal_snow_is_nonzero_eagerly():
    """NEMO's ``phs <= 0`` / ``ph_snw == 0`` are false for the smallest positive snow (``sbcblk.f90:1323``)."""
    c, x = _c1d_config(), _inputs()
    x["h_s"][1, 1, 0] = 5e-324
    with jax.disable_jit():
        res = _step({k: jnp.asarray(v) for k, v in x.items()}, c)
    assert float(res.blk_ice_2.qtr_ice_top[1, 1, 0]) == 0.0
    assert _oracle_mismatches(res, x, c) == []


@pytest.mark.xfail(strict=True, reason="D5_PLATFORM_FLOOR: XLA CPU jit flushes subnormal h_s (DECISION 158)")
def test_subnormal_snow_blocks_transmission_under_jit():
    c, x = _prod_config(), _inputs()
    x["h_s"][1, 1, 0] = 5e-324
    assert float(_jit_step(x, c).blk_ice_2.qtr_ice_top[1, 1, 0]) == 0.0


def test_production_surface_step_keeps_qsb_dry_negative_zero():
    """icesbc.f90:447 returns the cap operand on a signed-zero MIN tie."""
    c, x = _prod_config(), _inputs()
    x["smask0"][0, 0] = 0.0
    x["at_i"][0, 0] = x["at_i_b"][0, 0] = x["vt_i"][0, 0] = 0.0
    x["a_i_b"][0, 0, :] = 0.0
    x["sst_m"][0, 0], x["t_bo"][0, 0] = 0.0, TF
    got = np.asarray(_jit_step(x, c).flx_other.qsb_ice_bot)
    assert got.view(np.uint64)[0, 0] == np.uint64(0x8000000000000000)
