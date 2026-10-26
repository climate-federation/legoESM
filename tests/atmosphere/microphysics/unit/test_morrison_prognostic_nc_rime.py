"""MorrisonConfig.prognostic_droplet_number, stage S10.5: MG2 snow riming of
cloud water (accrete_cloud_water_snow) on our snow size distribution and the
Hallett-Mossop splinters (secondary_ice_production), with the smooth cold
ramp and rounded splinter triangle (user decisions 2026-10-21, 1a-6a).

Fixture ``oracle_fixtures/mg2_prognostic_nc_rime.npz``: ``make_fixture.py
--prognostic-nc-rime`` through ``mg2_rime`` (unmodified micro_mg_utils
routines).  Our PSD is run at MG2's snow density 250.  Since S10.5b the snow
number is re-adjusted at the slope bounds as in MG2, so bound cells match at the
hard limit; the production rounding (sharpness 100) departs by <= 0.5 % next to
a bound.  x64.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm import constants as C  # noqa: E402
from legoesm.atmosphere.physics.microphysics import _warm_rain as W  # noqa: E402
from legoesm.atmosphere.physics.microphysics import morrison as M  # noqa: E402
from legoesm.core.smooth import smoothstep_quintic  # noqa: E402

from tests.atmosphere.microphysics.unit.test_morrison_prognostic_nc import _cfg  # noqa: E402
from tests.atmosphere.microphysics.unit import test_morrison_prognostic_nc_act as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/mg2_prognostic_nc_rime.npz"
HARD_W, HARD_S = 1.0e-9, 1.0e9
P = 8.0e4


def _fx():
    return dict(np.load(FIXTURE))


def _acc(q_c_ic, pgam, lamc, q_s, N_s, precip_frac, T, p, rho, cfg, w, s=HARD_S):
    """In-cloud psacws through main's MG2 snow helpers: in-precip snow PSD with
    the slope bounds rounded at ``s``, provisional number fall speed, air
    viscosity, then accrete_cloud_water_snow with cold width ``w``."""
    q_sic, _, lams, n0s = W.mg2_inprecip_snow(q_s, N_s, precip_frac, cfg, psd_sharpness=s)
    _, uns = W.mg2_snow_fall_speeds(lams, rho, cfg)
    _, mu = W.mg2_snow_vent_eps(n0s, lams, T, p, rho, cfg)
    return W.mg2_accrete_cloud_water_snow(T, rho, mu, uns, q_c_ic, q_c_ic, q_sic, pgam,
                                          lamc, lams, n0s, cfg, w)[0]


def _rime(fx, cfg, w=HARD_W, s=HARD_S):
    rho = jnp.asarray(fx["in_rho"])
    qc = jnp.asarray(fx["in_qcic"])
    nc, pgam, lamc = W.mg2_size_dist_liq(qc, jnp.asarray(fx["in_ncic_kg"]) * rho, rho)
    ps = _acc(
        qc, pgam, lamc, jnp.asarray(fx["in_qs"]),
        jnp.asarray(fx["in_ns_kg"]), jnp.asarray(fx["in_precip_frac"]),
        jnp.asarray(fx["in_T"]), P, rho, cfg, w, s)
    nps = jnp.where(qc >= W.MG2_QSMALL, ps * nc / jnp.where(qc > 0, qc, 1.0), 0.0)
    return np.asarray(ps), np.asarray(nps / rho)


def _cfg250():
    return _cfg()._replace(rho_snow=float(_fx()["rhosn"]))


def _in_bounds(fx, cfg):
    lam = fx["out_lams"]
    return (lam > cfg.lams_min * (1 + 1e-12)) & (lam < cfg.lams_max * (1 - 1e-12))


def test_fixture_coverage():
    fx = _fx()
    cfg = _cfg250()
    assert not any(np.isnan(v).any() for v in fx.values() if v.dtype.kind == "f")
    ins, T = _in_bounds(fx, cfg), fx["in_T"]
    raw = fx["out_psacws_raw"]
    assert ins.sum() >= 100 and (~ins & (raw > 0)).sum() >= 20
    assert (raw[T > C.T_freeze] == 0).all() and (raw[T < C.T_freeze] > 0).sum() >= 200
    assert (fx["out_nsacwi"] > 0).sum() >= 60
    assert (fx["in_qs"] / fx["in_precip_frac"] > 0.01).any()          # 0.01 cap
    assert (fx["in_precip_frac"] < 1).sum() >= 50
    assert (fx["out_ncic_adj"] != fx["in_ncic_kg"]).any()               # droplet size reset
    # MG2's 1.2*rhof fall-speed cap never binds on this PSD (uns/rhof <= 0.813)
    rhof = (cfg.rho_su / fx["in_rho"]) ** 0.54
    assert (fx["out_uns"] < 0.82 * rhof).all()


def _at_bounds(fx, cfg):
    lam = fx["out_lams"]
    return np.isclose(lam, cfg.lams_min, rtol=1e-12), np.isclose(lam, cfg.lams_max, rtol=1e-12)


def test_riming_matches_mg2_everywhere():
    """In bounds AND where MG2's snow slope sits on a bound (S10.5b: number
    re-adjusted to the held slope, as MG2), at the hard limit."""
    fx = _fx()
    cfg = _cfg250()
    act = fx["out_psacws_raw"] > 0
    lo, hi = _at_bounds(fx, cfg)
    assert (lo & act).sum() >= 30 and (hi & act).sum() >= 30 and (_in_bounds(fx, cfg) & act).sum() >= 200
    ps, nps = _rime(fx, cfg)
    want_p, want_n = fx["out_psacws_raw"], fx["out_npsacws"]
    for mk, nm in ((lo, "lo"), (hi, "hi"), (_in_bounds(fx, cfg), "in")):
        m = mk & act
        ep = (np.abs(ps - want_p)[m] / want_p[m]).max()
        en = (np.abs(nps - want_n)[m] / want_n[m]).max()
        print("S10.5b psacws/npsacws vs MG2", nm, m.sum(), ep, en)
        assert ep < 1e-12 and en < 1e-12, nm
    assert (ps[~act] == 0).all()


def test_rounded_bounds_at_production_sharpness():
    """At the production rounding (sharpness 100 on slope/bound) the rate departs
    from MG2 only next to a bound, by at most 0.5 %; elsewhere it is MG2."""
    fx = _fx()
    cfg = _cfg250()
    act = fx["out_psacws_raw"] > 0
    ps, _ = _rime(fx, cfg, s=W.MG2_CAP_SHARPNESS)
    rel = np.abs(ps - fx["out_psacws_raw"])[act] / fx["out_psacws_raw"][act]
    print("S10.5b production rounding: max rel", rel.max(), "median", np.median(rel))
    assert rel.max() < 5e-3 and np.median(rel) < 1e-12 and rel.max() > 1e-5
    lam = fx["out_lams"][act]
    far = (lam > 1.2 * cfg.lams_min) & (lam < cfg.lams_max / 1.2)
    assert rel[far].max() < 1e-6


def test_splinters_match_mg2():
    fx = _fx()
    left, m, n = (np.asarray(x) for x in W.mg2_secondary_ice_production(
        jnp.asarray(fx["out_psacws_raw"]), jnp.asarray(fx["in_T"]), HARD_S))
    for got, k in ((left, "psacws_left"), (m, "msacwi"), (n, "nsacwi")):
        assert np.allclose(got, fx[f"out_{k}"], rtol=1e-10, atol=1e-30), k
    assert (m <= 7.4e-7 * fx["out_psacws_raw"]).all()                   # min never binds


def test_smooth_cold_ramp_and_rounded_triangle():
    fx = _fx()
    cfg = _cfg250()
    w = W.RAMP_WIDTH_PER_SHARPNESS / 2.0                                 # production: 2 K
    hard, _ = _rime(fx, cfg)
    soft, _ = _rime(fx, cfg, w=w)
    x = C.T_freeze - fx["in_T"]
    want = hard * np.asarray(smoothstep_quintic(jnp.asarray(x / w)))
    assert np.allclose(soft, want, rtol=1e-12, atol=0)
    assert (soft[x <= 0] == 0).all() and (soft[x >= w] == hard[x >= w]).all()
    assert ((soft < hard) & (x > 0) & (x < w) & (hard > 0)).any()
    T = jnp.array([264.0, 265.16, 266.0, 268.16, 269.0, 270.16, 272.0])
    _, _, n = W.mg2_secondary_ice_production(jnp.ones_like(T), T, W.MG2_CAP_SHARPNESS)
    tri = np.maximum(0, np.minimum((270.16 - np.asarray(T)) / 2, (np.asarray(T) - 265.16) / 3))
    n = np.asarray(n) / W._MG2_SPLINTER_PER_KG
    assert np.abs(n - tri).max() < 0.008                                 # rounded by <= ln2/100
    assert n[3] < 1.0 and n[1] > 0 and n[5] > 0                          # corners rounded


def test_gradients():
    cfg = _cfg250()

    def f(qc_gkg, qs_gkg, T):
        rho = jnp.full_like(T, 0.8)
        _, pgam, lamc = W.mg2_size_dist_liq(qc_gkg * 1e-3, jnp.full_like(T, 1.0e8), rho)
        ps = _acc(
            qc_gkg * 1e-3, pgam, lamc, qs_gkg * 1e-3, jnp.full_like(T, 3.0e4),
            jnp.ones_like(T), T, P, rho, cfg, 2.0, W.MG2_CAP_SHARPNESS)
        return W.mg2_secondary_ice_production(ps * 1e6, T, W.MG2_CAP_SHARPNESS)

    T = jnp.array([272.0, 271.3, 270.16, 269.0, 268.16, 266.5, 265.16, 263.0])
    qc, qs = jnp.full_like(T, 0.4), jnp.full_like(T, 0.2)
    check_grads(f, (qc, qs, T), order=2, modes=["rev"], eps=1e-5, atol=1e-4, rtol=1e-4)


def test_gradients_through_the_snow_slope_bounds():
    """check_grads order 2 on the riming rate as the snow number sweeps the slope
    through lams_min and lams_max (rounded at the production sharpness)."""
    cfg = _cfg250()
    q_s = 1.0e-4
    lam = jnp.array([400.0, 480.0, 500.0, 520.0, 9.0e4, 1.0e5, 1.1e5, 2.0e5])
    ns0 = lam ** 3 * q_s / (np.pi * cfg.rho_snow)
    rho, T = jnp.full_like(lam, 0.8), jnp.full_like(lam, 266.0)
    _, pgam, lamc = W.mg2_size_dist_liq(jnp.full_like(lam, 4e-4), jnp.full_like(lam, 1e8), rho)

    def f(log_ns):
        return 1e9 * _acc(
            jnp.full_like(lam, 4e-4), pgam, lamc, jnp.full_like(lam, q_s), jnp.exp(log_ns),
            1.0, T, P, rho, cfg, 2.0, W.MG2_CAP_SHARPNESS)

    check_grads(f, (jnp.log(ns0),), order=2, modes=["rev"], eps=1e-4, atol=1e-4, rtol=1e-4)
    g = np.asarray(jax.vmap(jax.grad(lambda x, i: f(x)[i]), (None, 0))(jnp.log(ns0), jnp.arange(8)))
    d = np.diagonal(g)
    assert abs(d[0]) < 1e-3 * abs(d[3]) and abs(d[-1]) < 1e-3 * abs(d[4])   # held slope: no N dependence


def test_gate_jumps_are_negligible():
    """Decision 6a: MG2's hard qsmall gates kept.  Measured riming rate just
    ABOVE each gate (1 + 1e-6 times qsmall; at qsmall itself lamc = 0), for the
    largest number our own PSD bounds allow at that mass (droplets: MG2 size
    reset; snow: slope at its upper bound)."""
    cfg = _cfg250()
    rho, T = jnp.array([1.0]), jnp.array([268.16])
    q = W.MG2_QSMALL * (1.0 + 1e-6)
    nc_hi, pgam, lamc = W.mg2_size_dist_liq(jnp.array([q]), jnp.array([1e12]), rho)
    assert float(lamc[0]) > 0
    ns_hi = cfg.lams_max ** 3 * q / (np.pi * cfg.rho_snow)
    jc = _acc(jnp.array([q]), pgam, lamc, jnp.array([1e-3]),
                                        jnp.array([1e5]), 1.0, T, P, rho, cfg, 2.0, W.MG2_CAP_SHARPNESS)
    _, pg2, lc2 = W.mg2_size_dist_liq(jnp.array([1e-3]), jnp.array([1e8]), rho)
    js = _acc(jnp.array([1e-3]), pg2, lc2,
                                        jnp.array([q]), jnp.array([ns_hi]), 1.0, T, P,
                                        rho, cfg, 2.0, W.MG2_CAP_SHARPNESS)
    jnc = float(jc[0]) * float(nc_hi[0]) / q
    print("S10.5 gate jumps: droplet-gate psacws", float(jc[0]), "kg/kg/s, npsacws",
          jnc, "1/m3/s; snow-gate psacws", float(js[0]), "kg/kg/s")
    assert 0 < float(jc[0]) < 1e-18 and 0 < float(js[0]) < 1e-18 and jnc < 1e-3


def _col(fx, T_c):
    cell = np.asarray(fx["col_q_c"]) > 0
    T = np.where(cell, T_c, np.asarray(fx["col_T"]))
    from legoesm.thermo import saturation_specific_humidity_ice
    fx["col_q_v"] = np.asarray(saturation_specific_humidity_ice(
        jnp.asarray(T), jnp.asarray(fx["col_p_full"])))
    return jnp.asarray(cell), jnp.asarray(T)


def _wired(fx, monkeypatch, cfg, q_s_val, kill_splinter):
    cell, T = _col(fx, 268.16)
    q_s = jnp.where(cell, q_s_val, 0.0)
    N_s = q_s * 2000.0 ** 3 / (np.pi * cfg.rho_snow)
    q_i = jnp.where(cell, 1.0e-4, 0.0)
    N_i = jnp.where(cell, 1.0e6, 0.0)
    seen = {}
    real_s, real_a, real_g = (W.mg2_secondary_ice_production,
                              W.mg2_accrete_cloud_water_snow, W.snow_to_graupel_pgsacw)

    def spy_s(ps, T, s):
        out = real_s(ps, T, s)
        if kill_splinter:
            out = (ps, 0.0 * out[1], 0.0 * out[2])
        seen["split_in"], seen["split"] = ps, out
        return out

    def spy_a(*a):
        seen["acc_args"] = a
        seen["acc"] = real_a(*a)
        return seen["acc"]

    def spy_g(q_c, q_s, N_s, rime, *a, **k):
        seen["rime"] = rime
        return real_g(q_c, q_s, N_s, rime, *a, **k)

    real_d = W.mg2_size_dist_liq

    def spy_d(*a):
        seen["dist"] = real_d(*a)
        return seen["dist"]

    monkeypatch.setattr(M, "mg2_size_dist_liq", spy_d)
    monkeypatch.setattr(M, "mg2_secondary_ice_production", spy_s)
    monkeypatch.setattr(M, "mg2_accrete_cloud_water_snow", spy_a)
    monkeypatch.setattr(M, "snow_to_graupel_pgsacw", spy_g)
    o = A._run(fx, cfg=cfg, T=T, hyd_kw=dict(q_s=q_s, N_s=N_s, q_i=q_i, N_i=N_i))
    return cell, o, seen


def _split_diff(fx, monkeypatch, cfg, qs):
    cell, base, sk = _wired(dict(fx), monkeypatch, cfg, qs, True)
    cell, out, so = _wired(dict(fx), monkeypatch, cfg, qs, False)
    c = np.asarray(cell)
    d = {k: (np.asarray(getattr(out, k)) - np.asarray(getattr(base, k)))[c]
         for k in ("dq_i_dt", "dq_s_dt", "dq_g_dt", "dN_i_dt", "dq_c_dt", "dT_dt", "dN_c_dt")}
    d["ice_scale"] = np.abs(np.asarray(out.dq_i_dt)[c]).max()
    left, m, n = (np.asarray(x)[c] for x in so["split"])
    total = np.asarray(sk["split_in"])[c]
    s = np.asarray(sk["rime"])[c] / np.where(total > 0, total, 1.0)   # cloud-water limiter scale
    return d, m, n, left, s, np.asarray(so["rime"])[c], np.asarray(so["split_in"])[c]


def test_splinters_are_wired(monkeypatch):
    """Splinter mass moves from snow to ice, scaled by the cloud-water limiter
    scale s with the rest of riming; splinter number reaches ice UNSCALED (MG2
    :1579-1580); cloud water, heat and droplet number do not see the split.  At
    5 g/kg snow the limiter binds hard in some cells."""
    fx = A._fx()
    for qs in (2.0e-5, 5.0e-3):
        d, m, n, left, s, _, total = _split_diff(fx, monkeypatch, _cfg(), qs)
        assert (m > 0).all() and (n > 0).all() and (s > 0).all() and (s <= 1).all()
        # + 4 ulps of the whole ice tendency: the MG2 cloud-ice Bergeron (S10.6)
        # makes it ~1e7 x the splinter mass, so summation order shows
        tol = 1e-12 * np.abs(d["dq_i_dt"]).max() + 4 * np.finfo(float).eps * d["ice_scale"]
        assert np.abs(d["dq_i_dt"] - s * m).max() <= tol + 1e-9 * np.abs(s * m).max()
        assert np.abs(d["dq_i_dt"] + d["dq_s_dt"] + d["dq_g_dt"]).max() <= 1e-8 * np.abs(d["dq_i_dt"]).max()
        assert np.allclose(d["dN_i_dt"], n, rtol=1e-7)
        for k in ("dq_c_dt", "dT_dt", "dN_c_dt"):
            assert (d[k] == 0).all(), k
        # cells sit on the triangle's peak corner: the scheme passes the rounding sharpness
        tri = float(W.mg2_secondary_ice_production(
            jnp.array([1.0]), jnp.array([268.16]), W.MG2_CAP_SHARPNESS)[2][0]) / W._MG2_SPLINTER_PER_KG
        assert tri < 0.995 and np.allclose(n, W._MG2_SPLINTER_PER_KG * tri * total, rtol=1e-9)
        if qs > 1e-3:
            assert (s < 0.5).sum() >= 3                                    # limiter binds; number unscaled


def test_rimed_snow_to_graupel_reads_the_snow_share(monkeypatch):
    """Decision 3a: with graupel on, snow->graupel conversion reads s x (riming
    minus the splinter share); reading the total would be 7e-7 x tri larger."""
    d, m, _, left, s, rime, _ = _split_diff(A._fx(), monkeypatch, _cfg(), 2.0e-5)
    assert np.allclose(rime, s * left, rtol=1e-12, atol=0)
    assert (np.abs(rime - s * (left + m)) > 1e-10 * rime).all()


def test_riming_is_in_cloud_times_lcldm_and_off_switch(monkeypatch):
    """The riming call sees MG2's in-cloud droplets (5e-3 cap), the grid snow
    and MG2's precipitation fraction, and its rate is multiplied by lcldm;
    do_snow_riming=False with the switch on = no riming, no splinters (5a)."""
    fx = A._fx()
    cfg = _cfg()
    cell, _, so = _wired(dict(fx), monkeypatch, cfg, 2.0e-5, False)
    a = so["acc_args"]       # main's (T, rho, mu, uns, qcic, ncic, qsic, pgam, lamc, lams, n0s, cfg, w)
    qc_ic_arg, qs_arg, pgam_arg, lamc_arg = (np.asarray(a[i]) for i in (4, 6, 7, 8))
    q_c, cf = jnp.asarray(fx["col_q_c"]), jnp.asarray(fx["col_cf"])
    pf = np.asarray(W.mg2_precip_frac(cf, q_c, jnp.where(cell, 1.0e-4, 0.0)))
    qc_ic, _, _, lcldm = W.mg2_incloud_warm_rain_inputs(
        q_c, q_c, q_c, cf, rescale_nc=True, precip_frac=jnp.asarray(pf))
    assert np.allclose(qc_ic_arg, np.asarray(qc_ic), rtol=1e-14, atol=0)
    assert a[12] == W.RAMP_WIDTH_PER_SHARPNESS / cfg.melt_sharpness   # smooth cold ramp
    # droplet shape from the ONE size call, i.e. from the pre-reset number (MG2 :1296)
    assert np.array_equal(pgam_arg, np.asarray(so["dist"][1])) and np.array_equal(lamc_arg, np.asarray(so["dist"][2]))
    assert not np.allclose(pgam_arg, np.asarray(W.mg2_size_dist_liq(
        jnp.asarray(qc_ic_arg), so["dist"][0], jnp.asarray(fx["col_rho"]))[1]))
    # in-precip snow on MG2's precipitation fraction
    assert (pf < 1).any()
    assert np.allclose(qs_arg, np.where(np.asarray(cell), np.minimum(2.0e-5 / pf, W.MG2_QSIC_MAX),
                                        0.0), rtol=1e-14)
    want = np.asarray(so["acc"][0]) * np.asarray(lcldm)
    assert np.allclose(np.asarray(so["split_in"]), want, rtol=1e-14, atol=0)
    assert (want[np.asarray(cell)] > 0).all() and (np.asarray(lcldm)[np.asarray(cell)] < 1).any()
    cell, _, s0 = _wired(dict(fx), monkeypatch, cfg._replace(do_snow_riming=False), 2.0e-5, False)
    assert "acc" not in s0
    assert all((np.asarray(x) == 0).all() for x in s0["split"])


def test_whole_scheme_gradient_through_0C_and_splinter_corners():
    """Finite-difference check of the whole-scheme riming (cloud-water loss) and
    splinter number through 0 C and at the rounded triangle corners, production
    ramp width (melt_sharpness 2 /K)."""
    import jax as _jax
    fx = A._fx()
    cell = jnp.asarray(fx["col_q_c"]) > 0
    cfg = _cfg(do_snow_to_graupel=False)._replace(melt_sharpness=2.0)
    q_s = jnp.where(cell, 2.0e-5, 0.0)
    N_s = q_s * 2000.0 ** 3 / (np.pi * cfg.rho_snow)
    from legoesm.thermo import saturation_specific_humidity_ice

    def run(T0):
        T = jnp.where(cell, T0, jnp.asarray(fx["col_T"]))
        f2 = dict(fx, col_q_v=saturation_specific_humidity_ice(T, jnp.asarray(fx["col_p_full"])))
        return A._run(f2, cfg=cfg, T=T, hyd_kw=dict(q_s=q_s, N_s=N_s))

    def rime(T0):
        return -jnp.sum(jnp.where(cell, run(T0).dq_c_dt, 0.0))

    def splint(T0):
        return jnp.sum(jnp.where(cell, run(T0).dN_i_dt, 0.0))

    for f, x in ((rime, C.T_freeze - 0.5), (rime, C.T_freeze - 1e-3),
                 (splint, 270.16), (splint, 268.16), (splint, 265.16)):
        g = float(_jax.grad(f)(x))
        h = 1e-4
        fd = (float(f(x + h)) - float(f(x - h))) / (2 * h)
        assert np.isfinite(g) and np.isclose(g, fd, rtol=5e-3, atol=1e-6 * abs(fd)), (x, g, fd)


def test_splinters_are_sources_of_the_ice_limiters(monkeypatch):
    """S10.6: MG2's qi conservation counts the splinter mass (scaled with
    riming, x s) and its ni conservation the splinter number (unscaled) among
    the cloud-ice sources (micro_mg2_0.F90:1717-1761)."""
    fx = A._fx()

    def run(kill):
        calls = []
        real = M.mg2_qi_conservation_ratio

        def spy(*a, **k):
            calls.append([np.asarray(x) for x in a[:3]])
            return real(*a, **k)

        with monkeypatch.context() as mp:
            mp.setattr(M, "mg2_qi_conservation_ratio", spy)
            cell, _, seen = _wired(dict(fx), mp, _cfg(), 2.0e-5, kill)
        return np.asarray(cell), calls, seen

    c, base, sk = run(True)
    _, out, so = run(False)
    assert len(base) == len(out) == 2          # qi limiter, then ni limiter
    m, n = (np.asarray(x)[c] for x in so["split"][1:])
    total = np.asarray(sk["split_in"])[c]
    s = np.asarray(sk["rime"])[c] / np.where(total > 0, total, 1.0)
    dqi = (out[0][1] - base[0][1])[c]
    dni = (out[1][1] - base[1][1])[c]
    ulp = 4 * np.finfo(float).eps        # the sources sum ~1e7 x the splinter mass
    assert (m > 0).all() and np.allclose(dqi, s * m, rtol=1e-9, atol=ulp * np.abs(out[0][1]).max())
    assert (n > 0).all() and np.allclose(dni, n, rtol=1e-9, atol=ulp * np.abs(out[1][1]).max())
