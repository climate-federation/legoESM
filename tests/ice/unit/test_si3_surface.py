"""Rung O7: SI3 ORCA2 ice-atmosphere surface step (blk_ice_1, ice_alb, blk_ice_2, ice_flx_other).

The oracle here is an independent per-cell Python transcription of the
``ORCA2_SAS_ICE_O2_REF`` ppsrc statements, written from NEMO's own literals:
CPython float arithmetic is IEEE binary64 with no contraction, and
``math.exp/log/log10/pow`` call the same glibc as NEMO.  The model must equal
it bit for bit under production ``jit``.
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

from legoesm.ice import si3_orca2_surface as o7  # noqa: E402
from legoesm.ice.dynamics import SI3ORCA2FoldProgram  # noqa: E402
from legoesm.ice.si3_orca2_state import fortran_sum_categories, si3_lbc_lnk_orca2  # noqa: E402

DATA = Path(__file__).parent / "data"
# Admitted O2 run, si3_rounds/round7/oracle_o2_orca2_sas_ice/ref_2step/output.namelist.{dyn,ice}
SHA = {
    "si3_o2_ref_output.namelist.dyn": "27a4ebeb6196546e1dae94fc09317ef38cf2712b1e4c3ddf31f4892cc5c4b16f",
    "si3_o2_ref_output.namelist.ice": "dfffdc445122bdcd8efc5f2791de67b784c79aa538d139f42fa5169e01caf985",
}
H, NI, NJ, JPL = 2, 8, 6, 5
TF = constants.T_freeze


@pytest.fixture(autouse=True)
def libm_policy():
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(previous)


def _text(name):
    raw = (DATA / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA[name]
    return raw.decode()


def _o2_config():
    return o7.SI3O7SurfaceConfig.from_resolved_namelists(
        _text("si3_o2_ref_output.namelist.dyn"), _text("si3_o2_ref_output.namelist.ice"))


def _prod_config():
    # The admitted O2 run is the ln_icedyn=F overlay; the production deck has T.
    return _o2_config()._replace(ln_icedyn=True)


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
    x["t_su"][2, 0, :] = [TF, TF + 0.05, 260.0, TF, 250.0]    # melting / freezing
    x["h_il"][2, 1, :] = [0.0, 0.02, 0.0200001, 0.03, 0.01]   # lid threshold
    x["a_i_b"][3, 0, :] = 0.0                                 # open ocean: SUM of -0 terms
    x["q_air_zt"][3, 0] = 5e-2                                # deposition => evap_ice < 0
    x["t_bo"][3, 1] = x["sst_m"][3, 1] + TF + 0.01            # supercooling
    x["vt_i"][3, 1] = 25.0
    x["vt_i"][4, 0] = 4.0                                     # landfast / full-cover stops
    x["u_ice"][H + 4, H + 0] = x["u_ice"][H + 3, H + 0] = 0.0
    x["v_ice"][H + 4, H + 0] = x["v_ice"][H + 4, H - 1] = 0.0
    x["sst_m"][5, 2] = -0.5                                   # warm leads (fhld branch)
    x["qns_oce"][5, 2] = 200.0
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
    return o7.si3_o7_surface_step(**{k: jnp.asarray(v) for k, v in x.items()}, config=config, fold=fold)


# --------------------- independent scalar oracle (NEMO literals) ---------------------
RT0, RTT0, RPREF = 273.15, 273.16, 1.0e5  # const-ok: independent NEMO-literal oracle (phycst.f90:50, sbc_phy.f90:55,73)
RGAMMA = 8.314510 / (28.9647e-3 * 1005.0)  # sbc_phy.f90:50-54
REPS0 = 287.05 / 461.495  # const-ok: independent NEMO-literal oracle (sbc_phy.f90:44-46)
AG, BG, CG, DG = -9.09718, -3.56654, 0.876793, math.log10(6.1071)
RHO0, RCP, RCPI, RLFUS, RLSUB, RHOS = 1026.0, 3991.86795711963, 2096.7, 0.3333601e6, 2.8344e6, 330.0


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


def s_alb(c, ts, hi, hs, ap, hp, hl, cf, m):
    fs = hs / (hs + 0.02)
    fp = ap if not ((1.0 - fs) < ap) else (1.0 - fs)
    v = 1.0 - fp - fs
    fi = 0.0 if not (v > 0.0) else v
    ai = c.rn_alb_idry
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


def s_cell(c, x, i, j):
    """Scalar oracle outputs for interior cell (i, j), production ln_icedyn=T arm."""
    g = lambda k: float(x[k][i, j])  # noqa: E731
    gc = lambda k, l: float(x[k][i, j, l])  # noqa: E731
    out = {}
    wi, wj = g("wndi"), g("wndj")
    wndm = math.sqrt(wi * wi + wj * wj)
    z = g("rhoa") * c.rn_Cd_ia * wndm
    out["utau_i"], out["vtau_i"] = z * wi, z * wj
    m, p, ta = g("smask0"), g("slp"), g("theta_air_zt")
    zztmp = 1.0 / (1.0 - 0.066)
    crain = (ta - RT0) * RCP * m
    csnw = ((RT0 if RT0 < ta else ta) - RT0) * RCPI * m
    cptn = g("sst_m") * RCP * m
    alb, qsr_i, qns_i, dqns, evap = [], [], [], [], []
    for l in range(JPL):
        ts = gc("t_su", l)
        a = s_alb(c, ts, gc("h_i", l), gc("h_s", l), gc("a_ip_eff", l), gc("h_ip", l), gc("h_il", l), g("cloud_fra"), m)
        alb.append(a)
        zsq = s_qsat(ts, p)
        zsipt = ts * math.pow(RPREF / p, RGAMMA)
        q = zztmp * (1.0 - a) * g("qsr")
        st3 = ts * ts * ts
        qlw = 0.996 * (g("qlw") - 5.67e-8 * ts * st3) * m
        dqlw = (4.0 * 0.996 * 5.67e-8) * st3
        blk = g("rhoa") * wndm
        t1 = blk * 1000.5 * c.rn_Ch_ia
        qsb = t1 * (zsipt - ta)
        t2 = blk * RLSUB * c.rn_Ce_ia
        la = t2 * (zsq - g("q_air_zt"))
        dla = t2 * s_dqsat(ts, p)
        qsr_i.append(q)
        qns_i.append(qlw - qsb - la)
        dqns.append(-(dqlw + t1 + dla))
        evap.append(c.rn_efac * la * (1.0 / RLSUB))
    out["alb"], out["qsr_ice"], out["qns_ice"], out["dqns_ice"] = alb, qsr_i, qns_i, dqns
    tp, sp = g("precip") * c.rn_pfac * m, g("snow") * c.rn_pfac * m
    zevap = g("emp") + tp
    atb = g("at_i_b")
    zsnw = 1.0 - math.pow(1.0 - atb, c.rn_snwblow)
    out["emp_oce"] = (1.0 - atb) * zevap - (tp - sp) - sp * (1.0 - zsnw)
    s = 0.0
    for l in range(JPL):
        s = s + gc("a_i_b", l) * evap[l]
    out["emp_ice"] = s - sp * zsnw
    qemp_oce = -(1.0 - atb) * zevap * cptn + (tp - sp) * crain + sp * (1.0 - zsnw) * (csnw - RLFUS)
    out["qemp_oce"] = qemp_oce
    qemp_ice = sp * zsnw * (csnw - RLFUS)
    s = 0.0
    for l in range(JPL):
        s = s + gc("a_i_b", l) * qns_i[l]
    out["qns_tot"] = (1.0 - atb) * g("qns_oce") + s + qemp_ice + qemp_oce
    ztri = 0.18 * (1.0 - g("cloud_fra")) + 0.35 * g("cloud_fra")
    qtr = []
    for l in range(JPL):
        if gc("h_s", l) <= 0.0:
            t = ztri
            if gc("h_i", l) < 0.1:
                t = t + (1.0 - t) * (1.0 - gc("h_i", l) * 10.0)
            qtr.append(qsr_i[l] * t)
        else:
            qtr.append(0.0)
    out["qtr_ice_top"] = qtr
    U = lambda k, di=0, dj=0: float(x[k][i + H + di, j + H + dj])  # noqa: E731
    zu, zum = U("u_ice") - U("ssu_m"), U("u_ice", -1) - U("ssu_m", -1)
    zv, zvm = U("v_ice") - U("ssv_m"), U("v_ice", 0, -1) - U("ssv_m", 0, -1)
    fric = c.rn_Cd_io * (0.5 * ((zu * zu + zum * zum) + (zv * zv + zvm * zvm))) * m
    a1, a2 = U("u_ice", -1) + U("u_ice"), U("v_ice", 0, -1) + U("v_ice")
    vel = 0.5 * math.sqrt(a1 * a1 + a2 * a2)
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
    qsb = sw * (qsb if not (cap < qsb) else cap)
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
    out["qsb_ice_bot"], out["fhld"], out["qlead"] = qsb, fhld, qlead
    return out


def _bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def _model_view(res):
    b2, fo = res.blk_ice_2, res.flx_other
    return {
        "utau_i": res.blk_ice_1.utau_ice, "vtau_i": res.blk_ice_1.vtau_ice, "alb": res.alb_ice,
        "qsr_ice": b2.qsr_ice, "qns_ice": b2.qns_ice, "dqns_ice": b2.dqns_ice, "emp_oce": b2.emp_oce,
        "emp_ice": b2.emp_ice, "qemp_oce": b2.qemp_oce, "qns_tot": b2.qns_tot,
        "qtr_ice_top": b2.qtr_ice_top, "qsb_ice_bot": fo.qsb_ice_bot, "fhld": fo.fhld, "qlead": fo.qlead,
    }


def _oracle_mismatches(res, x, c):
    view = {k: np.asarray(v) for k, v in _model_view(res).items()}
    bad = []
    for i in range(NI):
        for j in range(NJ):
            for k, v in s_cell(c, x, i, j).items():
                if not np.array_equal(_bits(view[k][i, j]), _bits(np.asarray(v, dtype=np.float64))):
                    bad.append((k, i, j))
    return bad


def _jit_step(x, c):
    return jax.jit(lambda xx: _step(xx, c))({k: jnp.asarray(v) for k, v in x.items()})


# ----------------------------------- tests -----------------------------------
def test_resolved_namelist_config_reads_deck_values():
    c = _o2_config()
    assert c.rn_Cd_ia == c.rn_Ce_ia == c.rn_Ch_ia == float("1.4000000000000000E-003")
    assert (c.rDt_ice, c.nn_hls, c.nn_snwfra, c.ld_pnd_alb, c.ln_icedyn) == (10800.0, 2, 2, True, False)
    assert (c.rn_alb_dpnd, c.rn_alb_lpnd, c.rn_snwblow) == (0.18, 0.3, 0.66)
    with pytest.raises(ValueError, match="RN_CD_IA"):
        o7.SI3O7SurfaceConfig.from_resolved_namelists(
            _text("si3_o2_ref_output.namelist.dyn").replace(" RN_CD_IA=", " XX="),
            _text("si3_o2_ref_output.namelist.ice"))


def test_folded_intrinsic_constants_are_correctly_rounded():
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.prec = 200
    assert o7._LOG_ALB_THIN_BREAK == float(mpmath.log(mpmath.mpf(0.05)))
    assert o7._LN10 == float(mpmath.log(mpmath.mpf(10.0)))
    assert o7._GOFF_ICE_D == float(mpmath.log10(mpmath.mpf(6.1071)))


def test_libm_pow_matches_glibc_and_differentiates():
    from legoesm.core import transcendentals as tr
    base = jnp.asarray([0.0, 0.3, 1.7, 10.0])
    got = np.asarray(jax.jit(lambda b: tr.pow(b, 0.66))(base))
    assert [float(v) for v in got] == [math.pow(float(b), 0.66) for b in base]
    grad = jax.grad(lambda b: tr.pow(b, 0.66).sum())(base)
    assert np.all(np.isfinite(np.asarray(grad))) and float(grad[0]) == 0.0
    assert abs(float(grad[2]) - 0.66 * 1.7 ** -0.34) < 1e-12


def test_jit_equals_independent_glibc_transcription_bit_for_bit():
    c, x = _prod_config(), _inputs()
    res = _jit_step(x, c)
    assert _oracle_mismatches(res, x, c) == []
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


def test_inexact_value_pins():
    c, x = _prod_config(), _inputs()
    res = _jit_step(x, c)
    got = {
        "alb": float(res.alb_ice[1, 0, 2]),
        "qns_tot": float(res.blk_ice_2.qns_tot[2, 3]),
        "qsb": float(res.flx_other.qsb_ice_bot[2, 2]),
    }
    ref = {
        "alb": s_cell(c, x, 1, 0)["alb"][2],
        "qns_tot": s_cell(c, x, 2, 3)["qns_tot"],
        "qsb": s_cell(c, x, 2, 2)["qsb_ice_bot"],
    }
    for key, v in got.items():
        assert v == ref[key] and v != round(v, 6), key


@pytest.mark.parametrize("plant", ["cd_ulp", "rtt0_ulp", "sum_first_category", "ln10_ulp", "snwfra"])
def test_production_path_plants_fire(plant, monkeypatch):
    c, x = _prod_config(), _inputs()
    if plant == "cd_ulp":
        c = c._replace(rn_Cd_ia=math.nextafter(c.rn_Cd_ia, 1.0))
    elif plant == "rtt0_ulp":
        monkeypatch.setattr(o7, "_RTT0", math.nextafter(o7._RTT0, 300.0))
    elif plant == "sum_first_category":
        monkeypatch.setattr(o7, "_sum0", lambda t: fortran_sum_categories(t, axis=-1))
    elif plant == "ln10_ulp":
        monkeypatch.setattr(o7, "_LN10", math.nextafter(o7._LN10, 3.0))
    elif plant == "snwfra":
        c = c._replace(nn_snwfra=1)
    assert _oracle_mismatches(_jit_step(x, c), x, _prod_config()) != []


def test_eager_disable_jit_equals_jit():
    c, x = _prod_config(), _inputs(1)
    xx = {k: jnp.asarray(v) for k, v in x.items()}
    jitted = jax.jit(lambda a: _step(a, c))(xx)
    with jax.disable_jit():
        eager = _step(xx, c)
    leaves = list(zip(jax.tree_util.tree_leaves(jitted), jax.tree_util.tree_leaves(eager)))
    assert len(leaves) == 33
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


def test_no_defaults_and_refused_arms():
    c, x = _prod_config(), _inputs()
    xx = {k: jnp.asarray(v) for k, v in x.items()}
    for name in ("wndi", "rhoa", "a_ip_eff", "emp", "t_bo", "u_ice"):
        bad = dict(xx)
        bad[name] = None
        with pytest.raises(ValueError, match=name):
            o7.si3_o7_surface_step(**bad, config=c, fold=FOLD)
    with pytest.raises(TypeError):
        o7.SI3O7SurfaceConfig()  # type: ignore[call-arg]
    for field, value in (("ld_pnd_alb", False), ("nn_qtrice", 1), ("ln_Cx_ice_cst", False),
                         ("nn_flxdist", 0), ("ln_cndflx", True), ("ln_leadhfx", False)):
        with pytest.raises(ValueError, match="not transcribed"):
            _step(xx, c._replace(**{field: value}))
    with pytest.raises(TypeError):
        o7.si3_o7_ice_sbc_tau_lbc(xx["wndi"], xx["wndj"], fold=(2, True))


@pytest.mark.xfail(strict=True, reason="D5_PLATFORM_FLOOR: XLA CPU jit flushes subnormal h_s (DECISION 158)")
def test_subnormal_snow_blocks_transmission():
    """NEMO's ``phs <= 0`` is false for the smallest positive snow (``sbcblk.f90:1323``)."""
    c, x = _prod_config(), _inputs()
    x["h_s"][1, 1, 0] = 5e-324
    assert float(_jit_step(x, c).blk_ice_2.qtr_ice_top[1, 1, 0]) == 0.0
