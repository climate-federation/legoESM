"""Two CAM6 MG2 cloud-ice sinks on the Morrison lane (CAM6 MG2 mode only):

A. ``MorrisonConfig.mg2_snow_accretes_ice``: snow collects cloud ice, MG2
   ``accrete_cloud_ice_snow`` (micro_mg_utils.F90:1356-1414, prai / nprai).
B. ``MorrisonConfig.mg2_ice_sed_clear_air_sublimation``: cloud ice falling
   into a layer with a smaller cloud fraction sublimates the clear-air share
   (micro_mg2_0.F90:2270-2296).

A is checked against the RUNNING MG2 (the compiled oracle's tap of prai /
nprai before the ice limiters, ``mg2_prognostic_nc_icelim.npz`` /
``mg2_prognostic_nc_frz.npz``, ESCOMP/CAM cam_cesm2_1_rel_60) and against a
line-by-line transcription of the routine; B against a transcription of the
MG2 sedimentation loop.  Both whole-call effects are checked for water and
heat conservation with the switch on and shown to vanish with it off.  x64.
"""
from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm import constants as C  # noqa: E402
from legoesm.atmosphere.physics.microphysics import _warm_rain as W  # noqa: E402
from legoesm.atmosphere.physics.microphysics import morrison as M  # noqa: E402
from legoesm.atmosphere.physics.microphysics.output import (  # noqa: E402
    HydrometeorState,
    sedimentation_tendency,
)

from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import (  # noqa: E402
    HARD_LIMIT,
    _production_morrison,
)

ROOT = Path(__file__).resolve().parents[4]
FX = ROOT / "tests/atmosphere/microphysics/oracle_fixtures"
J = jnp.asarray
HW = 1.0e-9                      # cold-step width [K] standing in for MG2's hard T <= tmelt
PROD_W = W.RAMP_WIDTH_PER_SHARPNESS / _production_morrison().melt_sharpness
QSMALL = 1.0e-18
# MG2 constants of accrete_cloud_ice_snow for the transcription
# (micro_mg_utils.F90 :186 eii, :139-140 as/bs, :127 rhosn; rhosu = 85000/(rd tmelt))
EII, AS, BS, RHOSN = 0.5, 11.72, 0.41, 250.0
RHOSU = 8.5e4 / (C.R_d * C.T_freeze)
# RULE 5 recorded gap: share of MG2's rate kept by the C2 cold step at the
# production width (2 K), at T_freeze - (30 K, 1, 0.5, 0.1 widths) and at T_freeze
GAP_COLD_STEP = [1.0, 1.0, 0.5, 0.00856, 0.0]


def _cfg(**kw):
    """Production deck's Morrison config, CAM6 MG2 mode, hard switches."""
    return M.resolve_morrison_flavor(_production_morrison()._replace(
        **HARD_LIMIT, warm_rain_scheme="kk2000_cam6", **kw))


def _prai_fortran(t, rho, qiic, niic, qsic, lams, n0s):
    """accrete_cloud_ice_snow, micro_mg_utils.F90:1356-1414, as written."""
    prai = np.zeros_like(t)
    nprai = np.zeros_like(t)
    gamma_bs_plus3 = math.gamma(3.0 + BS)
    for i in range(t.size):
        asn = AS * (RHOSU / rho.flat[i]) ** 0.54
        if qsic.flat[i] >= QSMALL and qiic.flat[i] >= QSMALL and t.flat[i] <= C.T_freeze:
            accrete_rate = (math.pi / 4.0 * EII * asn * rho.flat[i] * n0s.flat[i]
                            * gamma_bs_plus3 / lams.flat[i] ** (BS + 3.0))
            prai.flat[i] = accrete_rate * qiic.flat[i]
            nprai.flat[i] = accrete_rate * niic.flat[i]
    return prai, nprai


# ----------------------------------------------------------------- A: rate
def test_mg_flavour_carries_mg2_constants():
    cfg = _cfg()
    assert cfg.snow_aggregation_eii == EII and cfg.rho_snow == RHOSN
    assert cfg.fall_a_s == AS and cfg.fall_b_s == BS
    assert cfg.rho_su == pytest.approx(RHOSU, rel=1e-12)
    assert cfg.mg2_snow_accretes_ice is True
    assert cfg.mg2_ice_sed_clear_air_sublimation is True


def test_prai_matches_the_fortran_transcription_at_the_hard_limit():
    rng = np.random.default_rng(20261114)
    n = 400
    t = rng.uniform(200.0, 280.0, n)
    rho = rng.uniform(0.2, 1.3, n)
    qiic = 10.0 ** rng.uniform(-9, -3, n)
    niic = 10.0 ** rng.uniform(2, 6, n)
    qsic = 10.0 ** rng.uniform(-9, -3, n)
    lams = 10.0 ** rng.uniform(2.7, 5.0, n)
    n0s = 10.0 ** rng.uniform(3, 8, n)
    qiic[:20] = rng.choice([0.0, 0.5 * QSMALL], 20)        # below qsmall: no rate
    qsic[20:40] = rng.choice([0.0, 0.5 * QSMALL], 20)
    lams[20:40] = 0.0                                      # MG2's lams where there is no snow
    want_p, want_n = _prai_fortran(t, rho, qiic, niic, qsic, lams, n0s)
    got_p, got_n = (np.asarray(x) for x in W.mg2_accrete_cloud_ice_snow(
        J(t), J(rho), J(qiic), J(niic), J(qsic), J(lams), J(n0s), _cfg(), HW))
    assert (want_p > 0).sum() >= 150 and (want_p == 0).sum() >= 60
    assert np.all(np.isfinite(got_p)) and np.all(np.isfinite(got_n))
    np.testing.assert_allclose(got_p, want_p, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got_n, want_n, rtol=1e-12, atol=0)


@pytest.mark.parametrize("name", ["mg2_prognostic_nc_icelim", "mg2_prognostic_nc_frz"])
def test_prai_matches_the_running_mg2(name):
    """prai / nprai that the compiled MG2 hands to its ice limiters, from
    MG2's own state at that point (tap_ilim1: t, qi, ni, qs, icldm,
    precip_frac) through the port's in-cloud ice, ice slope and in-precip
    snow; snow number is the column input (MG2's instant snow melt changes it
    only above +2 C, where prai is 0)."""
    fx = dict(np.load(FX / f"{name}.npz"))
    cfg = _cfg()
    a = {k[10:]: J(fx[k]) for k in fx if k.startswith("ora_ilim1_")}
    rho, ns = J(fx["col_rho"]), J(fx["col_ns_kg"])
    has = a["qi"] >= W.MG2_QSMALL
    qiic = jnp.where(has, jnp.minimum(a["qi"] / a["icldm"], W.KK2000_CAM6_QC_MAX), 0.0)
    niic = jnp.where(has, jnp.maximum(a["ni"] / a["icldm"], 0.0), 0.0)
    nadj, _ = W.mg2_ice_slope(qiic, niic, cfg.rho_cloud_ice, cfg.ice_snow_d_auto, 0.0)
    qsic, _, lams, n0s = W.mg2_inprecip_snow(a["qs"], ns, a["precip_frac"], cfg,
                                             psd_sharpness=1.0e9)
    p, npr = (np.asarray(x) for x in W.mg2_accrete_cloud_ice_snow(
        a["t"], rho, qiic, nadj, qsic, lams, n0s, cfg, HW))
    wp, wn = fx["ora_ilim1_prai"], fx["ora_ilim1_nprai"]
    at0 = np.asarray(a["t"]) == C.T_freeze            # the recorded 0 C gap of the C2 step
    assert (wp > 0).sum() >= 150 and not at0.any()
    for g, w, k in ((p, wp, "prai"), (npr, wn, "nprai")):
        err = np.abs(g - w) / np.where(w > 0, w, 1.0)
        print(name, k, "max rel err", err[w > 0].max(), "points", (w > 0).sum())
        assert err[w > 0].max() <= 1.0e-9, k
        assert np.all(g[w == 0] == 0.0), k


def test_prai_cold_step_gap_and_gradient():
    """RULE 5: at the production width the rate is the hard-limit rate times
    the C2 step (exactly 0 at T_freeze, full from -width), recorded per case;
    the derivative through the step is finite and non-zero."""
    cfg = _cfg()
    T = J([C.T_freeze - 30.0, C.T_freeze - PROD_W, C.T_freeze - 0.5 * PROD_W,
           C.T_freeze - 0.1 * PROD_W, C.T_freeze])
    one = jnp.ones_like(T)
    args = (0.6 * one, 2.0e-4 * one, 5.0e4 * one, 3.0e-4 * one, 3.0e3 * one, 4.0e6 * one)
    hard = np.asarray(W.mg2_accrete_cloud_ice_snow(T, *args, cfg, HW)[0])
    soft = np.asarray(W.mg2_accrete_cloud_ice_snow(T, *args, cfg, PROD_W)[0])
    assert PROD_W == pytest.approx(2.0)
    gap = soft / np.where(hard > 0, hard, 1.0)
    # measured share of MG2's rate kept at T_freeze - (30 K, 1, 0.5, 0.1 widths), and at T_freeze
    print("cold-step share kept:", [float(x) for x in gap])
    np.testing.assert_allclose(gap, GAP_COLD_STEP, rtol=1e-6, atol=1e-12)
    assert hard[-1] == 0.0          # KNOWN DIFFERENCE: MG2 collects at exactly T_freeze

    def f(t, qi, qs):
        return W.mg2_accrete_cloud_ice_snow(
            t, 0.6 * one[:3], qi, 5.0e4 * one[:3], qs, 3.0e3 * one[:3], 4.0e6 * one[:3],
            cfg, PROD_W)[0]

    x = (C.T_freeze - J([0.75, 0.5, 0.15]) * PROD_W, 2.0e-4 * one[:3], 3.0e-4 * one[:3])
    check_grads(f, x, order=1, modes=("rev",), rtol=1e-6)
    g = jax.grad(lambda t: f(t, x[1], x[2]).sum())(x[0])
    assert np.all(np.isfinite(g)) and np.all(np.asarray(g) < 0)


# ------------------------------------------------- B: sedimentation kernel
def _ice_sed_fortran(q, n, fi, fni, icldm, pdel, dt):
    """micro_mg2_0.F90:2230-2300 for one column (do_cldice), as written:
    returns qitend, nitend, qvlat from sedimentation and the surface flux."""
    nlev = q.size
    dumi, dumni = q.copy(), n.copy()
    qitend, nitend, qvlat = np.zeros(nlev), np.zeros(nlev), np.zeros(nlev)
    sfc = 0.0
    nstep = 1 + int(max((fi / pdel).max(), (fni / pdel).max()) * dt)
    for _ in range(nstep):
        falouti = fi * dumi
        faloutni = fni * dumni
        k = 0
        faltndi = falouti[k] / pdel[k]
        faltndni = faloutni[k] / pdel[k]
        qitend[k] -= faltndi / nstep
        nitend[k] -= faltndni / nstep
        dumi[k] -= faltndi * dt / nstep
        dumni[k] -= faltndni * dt / nstep
        for k in range(1, nlev):
            dum1 = min(icldm[k] / icldm[k - 1], 1.0)
            faltndqie = (falouti[k] - falouti[k - 1]) / pdel[k]
            faltndi = (falouti[k] - dum1 * falouti[k - 1]) / pdel[k]
            faltndni = (faloutni[k] - dum1 * faloutni[k - 1]) / pdel[k]
            qitend[k] -= faltndi / nstep
            nitend[k] -= faltndni / nstep
            qvlat[k] -= (faltndqie - faltndi) / nstep
            dumi[k] -= faltndi * dt / nstep
            dumni[k] -= faltndni * dt / nstep
        sfc += falouti[-1] / C.g / nstep
    return qitend, nitend, qvlat, sfc, nstep


def _sed_case(seed=7, nlev=24, ncol=6):
    rng = np.random.default_rng(seed)
    p_half = np.linspace(1.5e4, 9.0e4, nlev + 1)[None].repeat(ncol, 0)
    pdel = np.diff(p_half, axis=1)
    rho = (0.5 * (p_half[:, 1:] + p_half[:, :-1])) / (C.R_d * rng.uniform(210, 265, (ncol, nlev)))
    dz = pdel / (C.g * rho)
    q = np.where(rng.random((ncol, nlev)) < 0.7, 10.0 ** rng.uniform(-7, -4, (ncol, nlev)), 0.0)
    n = q * 10.0 ** rng.uniform(8.5, 10.5, (ncol, nlev))
    V = rng.uniform(0.2, 1.1, (ncol, nlev))
    Vn = V * rng.uniform(0.2, 0.6, (ncol, nlev))
    cld = np.maximum(rng.random((ncol, nlev)) ** 2, W.MG2_MINCLD)
    cld[0] = 1.0                                  # one column overcast: nothing sublimates
    return dict(p_half=p_half, pdel=pdel, rho=rho, dz=dz, q=q, n=n, V=V, Vn=Vn, cld=cld)


@pytest.mark.parametrize("dt", [300.0, 1800.0])
def test_ice_sedimentation_matches_the_mg2_loop(dt):
    c = _sed_case()
    d = W.mg2_cloudy_inflow_fraction(J(c["cld"]))
    tq, sfc, loss, req = sedimentation_tendency(
        J(c["q"]), J(c["rho"]), J(c["V"]), J(c["dz"]), dt=dt, return_surface_flux=True,
        n_substeps_max=64, cfl_speed=J(c["Vn"]), return_substeps=True, inflow_fraction=d)
    tn, _ = sedimentation_tendency(
        J(c["n"]), J(c["rho"]), J(c["Vn"]), J(c["dz"]), dt=dt, n_substeps_max=64,
        cfl_speed=J(c["V"]), inflow_fraction=d)
    tot_loss = 0.0
    for i in range(c["q"].shape[0]):
        fi = C.g * c["rho"][i] * c["V"][i]
        fni = C.g * c["rho"][i] * c["Vn"][i]
        wq, wn, wv, wsfc, nstep = _ice_sed_fortran(
            c["q"][i], c["n"][i], fi, fni, c["cld"][i], c["pdel"][i], dt)
        assert int(req[i]) == nstep and nstep <= 64
        scale = np.abs(wq).max()
        assert np.abs(np.asarray(tq[i]) - wq).max() <= 1e-11 * scale
        assert np.abs(np.asarray(tn[i]) - wn).max() <= 1e-11 * np.abs(wn).max()
        assert np.abs(np.asarray(loss[i]) - wv).max() <= 1e-11 * scale
        assert abs(float(sfc[i]) - wsfc) <= 1e-11 * max(wsfc, 1e-300)
        tot_loss += wv.sum()
        if i == 0:
            assert np.all(np.asarray(loss[i]) == 0.0)      # overcast column
    assert tot_loss > 0.0
    assert (np.asarray(req) > 1).any() or dt < 1000.0       # the long step sub-cycles


@pytest.mark.parametrize("nsub", [1, 16])
def test_ice_sedimentation_conserves_and_reduces_to_the_old_form(nsub):
    c = _sed_case(seed=11)
    a = (J(c["q"]), J(c["rho"]), J(c["V"]), J(c["dz"]))
    kw = dict(dt=600.0, return_surface_flux=True, n_substeps_max=nsub)
    t0, s0 = sedimentation_tendency(*a, **kw)
    t1, s1, l1 = sedimentation_tendency(*a, **kw, inflow_fraction=jnp.ones_like(a[0]))
    assert np.array_equal(np.asarray(t0), np.asarray(t1)) and np.array_equal(s0, s1)
    assert np.all(np.asarray(l1) == 0.0)
    d = W.mg2_cloudy_inflow_fraction(J(c["cld"]))
    t2, s2, l2 = sedimentation_tendency(*a, **kw, inflow_fraction=d)
    m = c["rho"] * c["dz"]
    resid = ((np.asarray(t2) + np.asarray(l2)) * m).sum(1) + np.asarray(s2)
    assert np.abs(resid).max() <= 1e-13 * (np.abs(np.asarray(t2)) * m).sum(1).max()
    assert np.all(np.asarray(l2) >= 0.0) and np.asarray(l2).sum() > 0.0
    assert np.all(np.asarray(l2)[:, 0] == 0.0)             # nothing enters the top


def test_ice_sedimentation_gradients():
    c = _sed_case(seed=3, nlev=8, ncol=2)

    def f(q, cld):
        _, _, loss = sedimentation_tendency(
            q, J(c["rho"]), J(c["V"]), J(c["dz"]), dt=300.0, return_surface_flux=True,
            n_substeps_max=4, inflow_fraction=W.mg2_cloudy_inflow_fraction(cld))
        return loss.sum()

    q0, cld0 = J(np.maximum(c["q"], 1e-6)), J(c["cld"])
    # one scale per argument: q is O(1e-6), so it is perturbed through a factor
    check_grads(lambda s: f(q0 * s, cld0), (jnp.ones_like(q0),), order=1, modes=("rev",),
                eps=1e-6, rtol=1e-5)
    check_grads(lambda cl: f(q0, cl), (cld0,), order=1, modes=("rev",), eps=1e-7, rtol=1e-4)
    gq, gc = jax.grad(f, argnums=(0, 1))(q0, cld0)
    assert np.all(np.isfinite(gq)) and np.all(np.isfinite(gc))
    assert np.abs(np.asarray(gq)).max() > 0 and np.abs(np.asarray(gc)).max() > 0


# --------------------------------------------------------------- whole call
def _columns(ncol=8, nlev=20, seed=5):
    """Cold (-45..-5 C), liquid-free columns with cloud ice, snow and a
    varying cloud fraction: no melting, no homogeneous freezing, no warm rain,
    so the two sinks are the only thing the switches change."""
    rng = np.random.default_rng(seed)
    p_half = np.linspace(2.0e4, 7.0e4, nlev + 1)[None].repeat(ncol, 0)
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    T = np.linspace(C.T_freeze - 45.0, C.T_freeze - 5.0, nlev)[None] + rng.uniform(-1, 1, (ncol, nlev))
    rho = p_full / (C.R_d * T)
    dz = np.diff(p_half, axis=1) / (C.g * rho)
    qvi = np.asarray(M._qsat_ice_goff_gratch(J(T), J(p_full), cap_sharpness=math.inf))
    q_v = qvi * rng.uniform(0.85, 1.0, (ncol, nlev))
    r_i = 10.0 ** rng.uniform(-5.0, -4.3, (ncol, nlev))       # crystal radius [m]
    q_i = 10.0 ** rng.uniform(-6.0, -4.0, (ncol, nlev))
    N_i = q_i / (4.0 / 3.0 * math.pi * W._MG2_HOMOG_ICE_RHO * r_i ** 3)
    d_s = 10.0 ** rng.uniform(-3.7, -3.0, (ncol, nlev))       # flake diameter [m]
    q_s = 10.0 ** rng.uniform(-6.0, -4.0, (ncol, nlev))
    N_s = q_s / (math.pi / 6.0 * RHOSN * d_s ** 3)
    cf = np.clip(rng.random((ncol, nlev)), 0.05, 1.0)
    z = np.zeros((ncol, nlev))
    g = dict(T=T, q_v=q_v, p_full=p_full, p_half=p_half, rho=rho, dz=dz, cf=cf,
             q_i=q_i, N_i=N_i, q_s=q_s, N_s=N_s, z=z)
    return {k: J(v) for k, v in g.items()}


def _run(g, cfg, q_i=None, dt=600.0):
    hyd = HydrometeorState(q_c=g["z"], q_r=g["z"], q_i=g["q_i"] if q_i is None else q_i,
                           q_s=g["q_s"], q_g=g["z"], N_c=g["z"], N_r=g["z"], N_i=g["N_i"],
                           N_s=g["N_s"])
    return M.morrison_microphysics(
        g["T"], g["q_v"], hyd, g["p_full"], g["p_half"], g["rho"], g["dz"], dt, cfg,
        cloud_fraction=g["cf"])


def _col(g, x):
    return np.asarray((x * g["rho"] * g["dz"]).sum(1))


def _water(g, o):
    return _col(g, o.dq_v_dt + o.dq_c_dt + o.dq_r_dt + o.dq_i_dt + o.dq_s_dt + o.dq_g_dt)


OFF = dict(mg2_snow_accretes_ice=False, mg2_ice_sed_clear_air_sublimation=False)


def test_whole_call_snow_accretion_moves_ice_to_snow():
    """A alone: what cloud ice loses, snow gains (column sums with the surface
    fluxes and the snow that then sublimates: MG2's snow limiter sees the new
    source); crystals are removed."""
    g = _columns()
    off = _run(g, _cfg(**OFF))
    on = _run(g, _cfg(mg2_ice_sed_clear_air_sublimation=False))   # A at its default (True)
    dqi = _col(g, on.dq_i_dt - off.dq_i_dt)
    dqs = _col(g, on.dq_s_dt - off.dq_s_dt)
    dpr = np.asarray(on.precipitation - off.precipitation)
    scale = np.abs(_col(g, off.dq_i_dt)).max()
    assert (dqi < 0).all() and dqi.min() < -1e-3 * scale               # ice is lost in every column
    dqv = np.asarray(on.dq_v_dt - off.dq_v_dt)
    assert np.abs(dqi + dqs + _col(g, J(dqv)) + dpr).max() <= 1e-12 * scale   # ... to snow
    assert dqs.max() > 0 and np.abs(_col(g, J(dqv))).max() < 0.5 * np.abs(dqi).max()
    dT = np.asarray(on.dT_dt - off.dT_dt)                              # only snow sublimation moves heat
    np.testing.assert_allclose(dT, -C.L_s / C.c_pd * dqv, rtol=1e-9,
                               atol=1e-12 * np.abs(np.asarray(off.dT_dt)).max())
    assert _col(g, on.dN_i_dt - off.dN_i_dt).max() < 0                 # crystals are lost
    for o in (on, off):                                                # column water closes
        assert np.abs(_water(g, o) + np.asarray(o.precipitation)).max() <= 1e-12 * scale


def test_whole_call_clear_air_sublimation_conserves_water_and_heat():
    """B alone: the falling ice that enters clear air becomes vapour, with
    L_s/c_pd cooling cell by cell; snow is untouched."""
    g = _columns()
    off = _run(g, _cfg(**OFF))
    on = _run(g, _cfg(mg2_snow_accretes_ice=False))               # B at its default (True)
    dqv = np.asarray(on.dq_v_dt - off.dq_v_dt)
    dT = np.asarray(on.dT_dt - off.dT_dt)
    assert (dqv >= 0).all() and (dqv > 0).sum() >= 0.3 * dqv.size
    np.testing.assert_allclose(dT, -C.L_s / C.c_pd * dqv, rtol=1e-9,
                               atol=1e-12 * np.abs(np.asarray(off.dT_dt)).max())
    scale = np.abs(_col(g, off.dq_i_dt)).max()
    dqi = _col(g, on.dq_i_dt - off.dq_i_dt)
    dpr = np.asarray(on.precipitation - off.precipitation)
    assert np.abs(dqi + _col(g, J(dqv)) + dpr).max() <= 1e-12 * scale
    assert np.abs(np.asarray(on.dq_s_dt - off.dq_s_dt)).max() == 0.0
    assert (_col(g, on.dN_i_dt - off.dN_i_dt) < 0).all()               # those crystals are lost
    assert np.abs(_water(g, on) + np.asarray(on.precipitation)).max() <= 1e-12 * scale


def test_whole_call_gradient_through_both_sinks():
    g = _columns(ncol=2, nlev=8)
    cfg = M.resolve_morrison_flavor(_production_morrison()._replace(
        warm_rain_scheme="kk2000_cam6"))                     # smooth production defaults

    def col_tend(s, c):
        o = _run(g, c, q_i=g["q_i"] * s)
        return (o.dq_s_dt * g["rho"] * g["dz"]).sum(), (o.dq_v_dt * g["rho"] * g["dz"]).sum()

    for idx, flags in ((0, dict(mg2_ice_sed_clear_air_sublimation=False)),
                       (1, dict(mg2_snow_accretes_ice=False))):
        on = jax.grad(lambda s: col_tend(s, cfg._replace(**flags))[idx])(1.0)
        off = jax.grad(lambda s: col_tend(s, cfg._replace(**OFF))[idx])(1.0)
        assert np.isfinite(on) and np.isfinite(off)
        assert abs(float(on - off)) > 1e-3 * abs(float(on)), (idx, float(on), float(off))


# ------------------------------------------------------------ config wiring
FLAGS = (("morrison_mg2_snow_accretes_ice", "mg2_snow_accretes_ice",
          "--no-morrison-mg2-snow-accretes-ice"),
         ("morrison_mg2_ice_sed_clear_air_sublimation", "mg2_ice_sed_clear_air_sublimation",
          "--no-morrison-mg2-ice-sed-clear-air-sublimation"))


@pytest.mark.parametrize("flat,leaf,cli", FLAGS)
def test_switch_wiring(flat, leaf, cli):
    from scripts.run.run_amip import (
        _postprocess_args,
        build_arg_parser,
        build_config_from_args,
    )

    from legoesm.atmosphere.physics.microphysics.config import (
        MorrisonConfig,
        ThompsonConfig,
        apply_microphysics_experiment_flags,
    )
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import thread_morrison_scalars
    from legoesm.driver.run_config_yaml import load_yaml_config
    base = MorrisonConfig()
    assert ExperimentConfig._field_defaults[flat] is getattr(base, leaf) is True
    assert getattr(apply_microphysics_experiment_flags(
        base, "morrison", **{flat: False}), leaf) is False
    with pytest.raises(ValueError, match=flat):
        apply_microphysics_experiment_flags(ThompsonConfig(), "thompson", **{flat: False})
    with pytest.raises(TypeError, match="bool"):
        apply_microphysics_experiment_flags(base, "morrison", **{flat: "false"})
    ExperimentConfig(microphysics="morrison", **{flat: False}).validate_strict()
    with pytest.raises(ValueError, match=flat):
        ExperimentConfig(microphysics="thompson", **{flat: False}).validate_strict()
    assert thread_morrison_scalars(ExperimentConfig(microphysics="morrison"),
                                   "morrison", base) is base
    assert getattr(thread_morrison_scalars(
        ExperimentConfig(microphysics="morrison", **{flat: False}), "morrison", base),
        leaf) is False
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--microphysics", "morrison", cli]), parser))
    assert getattr(cfg, flat) is False
    assert getattr(build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser)), flat) is True
    keys = load_yaml_config(ROOT / "config/amip/amip_production.yaml", parser)
    assert keys[flat] is True
    parser.set_defaults(**keys)
    assert getattr(parser.parse_args([cli]), flat) is False
