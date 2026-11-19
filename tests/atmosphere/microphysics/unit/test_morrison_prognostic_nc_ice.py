"""MorrisonConfig.prognostic_droplet_number, stage S10.6 on main's MG2 snow
code: CAM6 MG2 cloud-ice deposition / sublimation / Bergeron, the 0 C smooth
steps on deposition, both Bergeron rates and Ferrier ice->snow, MG2's
deposition limiter and ice-number limiter (user decisions 2026-10-21 1a-13a,
S10.6b 2a-5a), against the RUNNING MG2.

Fixtures: ``mg2_prognostic_nc_ice.npz`` (``make_fixture.py --prognostic-nc-ice``,
``mg2_ice`` on unmodified micro_mg_utils routines, Goff-Gratch) and
``mg2_prognostic_nc_icelim.npz`` (``--prognostic-nc-icelim``, the limiter
inputs after MG2's cloud-water limiter, tap_ilim1, and the limited rates,
tap_ilim3).  A 1e-9 K ramp and hard slope rounding give CAM's forms.  x64.
"""
from __future__ import annotations

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
from legoesm.core.smooth import smoothstep_quintic  # noqa: E402
from legoesm.thermo import (  # noqa: E402
    saturation_specific_humidity_goff_gratch as qsl,
    saturation_specific_humidity_ice_goff_gratch as qsi,
)

from tests.atmosphere.microphysics.unit.test_morrison_prognostic_nc import _cfg  # noqa: E402
from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import _production_morrison  # noqa: E402
from tests.atmosphere.microphysics.unit import test_morrison_prognostic_nc_act as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FX = ROOT / "tests/atmosphere/microphysics/oracle_fixtures"
HW = 1.0e-9
OMSM = 1.0 - 1.0e-5
J = jnp.asarray
PROD_SHARP = _production_morrison().melt_sharpness


def _pt():
    return dict(np.load(FX / "mg2_prognostic_nc_ice.npz"))


def _lim():
    return dict(np.load(FX / "mg2_prognostic_nc_icelim.npz"))


def _sat(T, p):
    qvl = qsl(T, p)
    return qvl, jnp.where(T >= C.T_freeze, qvl, qsi(T, p))


def _points(fx, w=HW, log_width=0.0, psd=1.0e9):
    cfg = _cfg()._replace(rho_snow=float(fx["rhosn"]))
    dcs = float(fx["dcs"])
    g = {k[3:]: J(v) for k, v in fx.items() if k.startswith("in_")}
    T, p = g["T"], g["p"]
    rho = p / (C.R_d * T)
    qvl, qvi = _sat(T, p)
    vd, su, bg = W.mg2_ice_deposition_sublimation(
        g["q"], g["qi"], g["ni_kg"], g["icldm"], T, p, rho, qvl, qvi, cfg.rho_cloud_ice, dcs,
        log_width, w, presence_top=W.MG2_QSMALL)
    pf = g["precip_frac"]
    qsic, _, lams, n0s = W.mg2_inprecip_snow(g["qs"], g["ns_kg"], pf, cfg, psd_sharpness=psd)
    eps_s, _ = W.mg2_snow_vent_eps(n0s, lams, T, p, rho, cfg)
    bs = W.mg2_bergeron_snow(T, g["qcic"], qsic, qvl, qvi, eps_s, w)
    has = g["qi"] >= W.MG2_QSMALL
    qiic = jnp.where(has, jnp.minimum(g["qi"] / g["icldm"], 5e-3), 0.0)
    niic = jnp.where(has, jnp.maximum(g["ni_kg"] / g["icldm"], 0.0), 0.0)
    pr = W.mg2_snow_sublimation(g["q"], qvl, qvi, T, eps_s, qsic, g["lcldm"], pf, g["qcic"], qiic)
    nadj, lami = W.mg2_ice_slope(qiic, niic, cfg.rho_cloud_ice, dcs, log_width)
    pc, npc = W.ice_autoconversion_mg2(qiic, nadj, lami, dcs, T, w)
    out = dict(vap_dep=vd, ice_sublim=su, berg=bg, bergs=bs, prds=pr, prci=pc, nprci=npc,
               qvl=qvl, qvi=qvi)
    return {k: np.asarray(v) for k, v in out.items()}


def test_point_fixture_coverage():
    fx = _pt()
    T = fx["in_T"]
    act = fx["in_qi"] >= W.MG2_QSMALL
    assert (T > C.T_freeze).sum() >= 10 and (T < C.T_freeze - 30).sum() >= 10
    assert ((fx["in_qi"] > 0) & ~act).any()                                 # 0 < qi < qsmall
    assert (np.isclose(fx["out_lami"], 1.0 / (2 * fx["dcs"])) & act).sum() >= 10   # slope bounds
    assert (np.isclose(fx["out_lami"], 1.0e6) & act).sum() >= 10
    for k in ("vap_dep", "berg", "bergs", "prci"):
        assert (fx[f"out_{k}"] > 0).sum() >= 50, k
    assert (fx["out_ice_sublim"] < 0).sum() >= 50 and (fx["out_prds"] < 0).sum() >= 30
    assert ((fx["out_vap_dep"] == 0) & (fx["out_ice_sublim"] == 0) & act
            & (fx["in_q"] > fx["out_qvi"]) & (T >= C.T_freeze)).any()   # warm deposition dropped


def test_processes_match_mg2_at_the_hard_limit():
    fx = _pt()
    got = _points(fx)
    want = {k: fx[f"out_{k}"] for k in ("vap_dep", "ice_sublim", "berg", "bergs", "prci", "nprci",
                                       "qvl", "qvi", "prds")}
    for k, w in want.items():
        g = got[k]
        err = np.abs(g - w).max() / max(np.abs(w).max(), 1e-300)
        print("S10.6", k, err)
        assert err < 1e-12, k
        assert ((g == 0) == (w == 0)).all(), k


def test_smooth_zero_degree_steps_at_the_production_width():
    """Decision 5a / 6a: deposition, both Bergeron rates and ice->snow switch on
    with the smooth step, 0 at 0 C and full by -2 K (production melt sharpness);
    the gap to CAM per point is exactly (1 - step); sublimation is not gated."""
    fx = _pt()
    width = W.RAMP_WIDTH_PER_SHARPNESS / PROD_SHARP
    hard, soft = _points(fx), _points(fx, w=width)
    x = C.T_freeze - fx["in_T"]
    f = np.asarray(smoothstep_quintic(J(x / width)))
    for k in ("vap_dep", "berg", "bergs", "prci", "nprci"):
        assert np.allclose(soft[k], hard[k] * f, rtol=1e-12, atol=0), k
        assert ((soft[k] < hard[k]) & (x > 0) & (x < width)).any(), k
        assert (soft[k][x <= 0] == 0).all(), k
    assert np.array_equal(soft["ice_sublim"], hard["ice_sublim"])


def test_gradients_through_the_steps():
    T = J([250.0, 262.0, 271.8, 272.6, 268.0, 245.0])
    p = jnp.full_like(T, 6.0e4)
    rho = p / (C.R_d * T)
    cfg = _cfg()

    def f(qv_rel, qi_gkg, T):
        qvl, qvi = _sat(T, p)
        vd, su, bg = W.mg2_ice_deposition_sublimation(
            qvi * qv_rel, qi_gkg * 1e-3, jnp.full_like(T, 1e5), jnp.full_like(T, 0.6), T, p, rho,
            qvl, qvi, cfg.rho_cloud_ice, 5e-4, cfg.mg2_ice_slope_log_width, 2.0, presence_top=W.MG2_QSMALL)
        nadj, lami = W.mg2_ice_slope(qi_gkg * 1e-3, jnp.full_like(T, 1e5), cfg.rho_cloud_ice, 5e-4,
                                     cfg.mg2_ice_slope_log_width)
        pc, npc = W.ice_autoconversion_mg2(qi_gkg * 1e-3, nadj, lami, 5e-4, T, 2.0)
        bs = W.mg2_bergeron_snow(T, jnp.full_like(T, 3e-4), jnp.full_like(T, 3e-4), qvl, qvi,
                                 jnp.full_like(T, 1e-3), 2.0)
        return 1e9 * jnp.stack([vd, su, bg, pc, npc * 1e-9, bs])

    args = (J([1.1, 1.05, 0.9, 1.2, 0.95, 1.02]), jnp.full_like(T, 0.2), T)
    check_grads(f, args, order=2, modes=["rev"], eps=1e-4, atol=1e-3, rtol=1e-3)


def _ilim(fx):
    a = {k[10:]: J(fx[k]) for k in fx if k.startswith("ora_ilim1_")}
    c = {k[10:]: fx[k] for k in fx if k.startswith("ora_ilim3_")}
    return a, c, float(fx["dt"])


def test_deposition_limiter_matches_mg2_at_the_hard_limit():
    """MG2's deposition limits on MG2's own inputs; the only departure is our
    nucleated number scaled with its limited mass (10a)."""
    fx = _lim()
    a, c, dt = _ilim(fx)
    vd, mn, nn = (np.asarray(x) for x in W.mg2_deposition_limiter(
        a["q"], a["t"], a["qvi"], a["qc"], a["qcrat"], a["vap_dep"], a["mnuccd"], a["nnuccd"],
        dt, 1.0e-12, presence_top=W.MG2_QSMALL))
    for k, g in (("vap_dep", vd), ("mnuccd", mn)):
        w = c[k]
        err = np.abs(g - w).max() / max(np.abs(w).max(), 1e-300)
        moved = (~np.isclose(w, np.asarray(a[k]), rtol=1e-9, atol=0)).sum()
        print("S10.6 dep limiter", k, err, "moved", moved)
        assert err < 1e-12 and moved >= 5, k
    m0 = np.asarray(a["mnuccd"])
    share = np.where(m0 > 0, mn / np.where(m0 > 0, m0, 1), 1)
    assert np.allclose(nn, np.asarray(a["nnuccd"]) * share, rtol=1e-12, atol=0)
    assert (share < 1 - 1e-9).sum() >= 5
    assert np.array_equal(c["nnuccd"], np.asarray(a["nnuccd"]))   # MG2 leaves its number


def test_ice_number_limiter_formula_matches_mg2():
    """MG2 ni conservation (micro_mg2_0.F90:1749-1761) through
    mg2_qi_conservation_ratio at the hard limit, with MG2's sources
    nnuccd + nsacwi lcldm (+ nnuccri precip_frac, zero without rain on ice)."""
    fx = _lim()
    a, c, dt = _ilim(fx)
    nnuccri = J(fx["ora_ilim2_nnuccri"]) * a["precip_frac"]
    snk = (a["nprci"] + a["nprai"]) * a["icldm"]
    r = W.mg2_qi_conservation_ratio(a["ni"], a["nnuccd"] + a["nsacwi"] * a["lcldm"] + nnuccri,
                                    snk, dt, OMSM)
    g = np.asarray(a["nprci"] * r)
    w = c["nprci"]
    assert np.abs(g - w).max() <= 1e-12 * np.abs(w).max()
    assert (~np.isclose(w, np.asarray(a["nprci"]), rtol=1e-9, atol=0)).sum() >= 5


def test_limiter_gradients():
    fx = _lim()
    a, _, dt = _ilim(fx)
    keep = np.nonzero(np.asarray(a["vap_dep"] + a["mnuccd"]).ravel() > 0)[0][:16]
    s = {k: v.ravel()[keep] for k, v in a.items()}

    def f(r_qv, r_vd):
        vd, mn, _ = W.mg2_deposition_limiter(
            s["q"] * (1 + r_qv), s["t"], s["qvi"], s["qc"], s["qcrat"], s["vap_dep"] * (1 + r_vd),
            s["mnuccd"], s["nnuccd"], dt, 1.0 / W.MG2_CAP_SHARPNESS, presence_top=W.MG2_QSMALL)
        return jnp.stack([vd / (1e-30 + s["vap_dep"]), mn / (1e-30 + s["mnuccd"] + 1e-30)])

    z = jnp.zeros_like(s["q"])
    check_grads(f, (z, z), order=2, modes=["rev"], eps=1e-7, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------- wiring
def _cold(fx, T_c=258.0, q_s=1e-4, q_i=5e-5, rh_i=1.08):
    cell = np.asarray(fx["col_q_c"]) > 0
    T = np.where(cell, T_c, np.asarray(fx["col_T"]))
    p = np.asarray(fx["col_p_full"])
    fx["col_q_v"] = np.where(cell, rh_i * np.asarray(qsi(J(T), J(p))), fx["col_q_v"])
    c = J(cell)
    hyd = dict(q_s=jnp.where(c, q_s, 0.0), N_s=jnp.where(c, q_s * 2000.0 ** 3 / (np.pi * 100.0), 0.0),
               q_i=jnp.where(c, q_i, 0.0), N_i=jnp.where(c, 1e5, 0.0))
    return c, J(T), hyd


def _spy(mp, name, change=None):
    seen = {}
    real = getattr(M, name)

    def spy(*a, **k):
        out = real(*a, **k)
        if change is not None:
            out = change(out)
        seen["in"], seen["out"] = a, out
        return out

    mp.setattr(M, name, spy)
    return seen


@pytest.mark.parametrize("kw,msg", [
    ({"morrison_flavor": "sam", "ice_to_snow_scheme": "m2005_autoconv"}, "mg_ferrier"),
    ({"wbf_scheme": "bergeron_heuristic"}, "emergent"),
    ({"ice_deposition_scheme": "heuristic"}, "m2005"),
])
def test_start_up_guards(kw, msg):
    with pytest.raises(ValueError, match=msg):
        A._run(A._fx(), cfg=_cfg()._replace(**kw))


def test_cloud_ice_bergeron_and_deposition_are_wired(monkeypatch):
    """MG2's cloud-ice Bergeron is the scheme's cloud-water -> ice Bergeron
    (budget = -berg x qc limiter scale); the deposition limiter receives the raw
    deposition and the cloud-water limiter's qcrat; ice sublimation removes no
    ice number (4a)."""
    fx = A._fx()
    c, T, hyd = _cold(fx)
    s_dep = _spy(monkeypatch, "mg2_ice_deposition_sublimation")
    s_lim = _spy(monkeypatch, "mg2_deposition_limiter")
    out = A._run(fx, cfg=_cfg(do_snow_to_graupel=False), T=T, hyd_kw=hyd)
    qb = {k: np.asarray(v) for k, v in out.qc_budget.items()}
    cc = np.asarray(c)
    berg = np.asarray(s_dep["out"][2])
    qcrat = np.asarray(s_lim["in"][4])
    assert (berg[cc] > 0).all() and (qb["bergeron"][cc] < 0).all()
    assert np.allclose(-qb["bergeron"], berg * qcrat, rtol=1e-12, atol=0)
    assert np.array_equal(np.asarray(s_lim["in"][5]), np.maximum(np.asarray(s_dep["out"][0]), 0.0))
    assert ((qcrat < 1) & cc).any()


def test_no_ice_number_lost_to_sublimation(monkeypatch):
    fx = A._fx()
    c, T, hyd = _cold(fx, rh_i=0.7)
    s_dep = _spy(monkeypatch, "mg2_ice_deposition_sublimation")
    calls = []
    real_sed = M.sedimentation_tendency

    def spy_sed(x, *a, **k):
        calls.append((np.asarray(x), k.get("extra_sink")))
        return real_sed(x, *a, **k)

    monkeypatch.setattr(M, "sedimentation_tendency", spy_sed)
    zero_ac = _spy(monkeypatch, "ice_autoconversion_mg2",
                   lambda o: (jnp.zeros_like(o[0]), jnp.zeros_like(o[1])))
    A._run(fx, cfg=_cfg(do_snow_to_graupel=False), T=T, hyd_kw=hyd)
    assert zero_ac and (np.asarray(s_dep["out"][1])[np.asarray(c)] < 0).all()
    ni = [e for x, e in calls if e is not None
          and np.array_equal(x, np.clip(np.asarray(hyd["N_i"]), 0.0, None))]
    assert len(ni) == 1 and not np.asarray(ni[0]).any()


def test_snow_deposition_off_switch(monkeypatch):
    """Decision 3a: snow deposition off = no snow Bergeron and no snow sublimation."""
    fx = A._fx()
    c, T, hyd = _cold(fx)
    called = []
    monkeypatch.setattr(M, "mg2_bergeron_snow", lambda *a: called.append(1))
    monkeypatch.setattr(M, "mg2_snow_sublimation", lambda *a: called.append(1))
    out = A._run(fx, cfg=_cfg(do_snow_deposition=False), T=T, hyd_kw=hyd)
    assert not called and (np.asarray(out.qc_budget["bergeron_snow"]) == 0).all()


def test_whole_scheme_gradient_through_0C():
    """Finite-difference check of whole-scheme cloud-ice growth (deposition +
    Bergeron) through the smooth 0 C step at the production width."""
    fx = A._fx()
    cell = J(np.asarray(fx["col_q_c"]) > 0)
    cfg = _cfg(do_snow_to_graupel=False)._replace(melt_sharpness=PROD_SHARP)
    hyd = dict(q_i=jnp.where(cell, 5e-5, 0.0), N_i=jnp.where(cell, 1e5, 0.0))
    width = W.RAMP_WIDTH_PER_SHARPNESS / PROD_SHARP

    def grow(T0):
        T = jnp.where(cell, T0, J(fx["col_T"]))
        f2 = dict(fx, col_q_v=1.02 * qsi(T, J(fx["col_p_full"])))
        return jnp.sum(jnp.where(cell, A._run(f2, cfg=cfg, T=T, hyd_kw=hyd).dq_i_dt, 0.0))

    for x in (C.T_freeze - 0.5 * width, C.T_freeze - 1e-3, C.T_freeze - 1.25 * width):
        g = float(jax.grad(grow)(x))
        h = 1e-4
        fd = (float(grow(x + h)) - float(grow(x - h))) / (2 * h)
        assert np.isfinite(g) and np.isclose(g, fd, rtol=5e-3, atol=1e-6 * abs(fd)), (x, g, fd)


def test_limited_deposition_reaches_the_tendencies(monkeypatch):
    """The deposition limiter's output is the deposition the scheme applies:
    zeroing it inside the scheme returns exactly that vapour (dq_v) and heat."""
    fx = A._fx()
    c, T, hyd = _cold(fx)
    cfg = _cfg(do_snow_to_graupel=False)
    with monkeypatch.context() as mp:
        s0 = _spy(mp, "mg2_deposition_limiter")
        base = A._run(fx, cfg=cfg, T=T, hyd_kw=hyd)
    with monkeypatch.context() as mp:
        _spy(mp, "mg2_deposition_limiter", lambda o: (0.0 * o[0], o[1], o[2]))
        out = A._run(fx, cfg=cfg, T=T, hyd_kw=hyd)
    vd = np.asarray(s0["out"][0])
    m = np.asarray(c) & (vd > 0)
    dv = (np.asarray(out.dq_v_dt) - np.asarray(base.dq_v_dt))[m]
    dT = (np.asarray(out.dT_dt) - np.asarray(base.dT_dt))[m]
    assert m.sum() >= 5
    assert np.allclose(dv, vd[m], rtol=1e-6, atol=0)
    assert np.allclose(dT, -C.L_s * vd[m] / C.c_pd, rtol=1e-6, atol=0)
