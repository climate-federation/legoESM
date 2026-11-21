"""Route C (routec.py): Grenier and Bretherton (2001) subgrid inversion with
explicit entrainment inside CLUBB, Lock (2001) diagnosis; ported from Isma
Di Carlo's ENSO fork with every switch on traced state smoothed (RULE 5).

  V1 reconstruction of a known inversion (hard widths: the published form)
  V2 evaporative enhancement against a brute-force mixing line
  V3 entrainment velocity by hand
  V4 the blended constrained rows hold after the solve
  V5 CLUBB with route C conserves column thl and rt
  R5 the three RULE 5 tests: hard limit = reference (V1), stored per-case gap
     of the smooth default, gradients through the switches
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.turbulence import routec as rc  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    CLUBBConfig, CLUBBForcing, CLUBBMomentState, _batched_penta_solve, advance_clubb_core,
    make_clubb_grid, sat_mixrat_liq)

W0 = rc.RouteCWidths(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
WD = rc.widths_from_config(CLUBBConfig())


# ---------------------------------------------------------------- V1
def _known_column(delta=8.0, m_true=0.37, nz=10, kb=4, dp=2000.0, gam=2.0e-3, beta=-1.0e-7):
    """Ascending column, equal layer masses dp: well mixed below the inversion,
    linear in pi above it, the inversion at above-inversion mass fraction
    m_true of layer kb.  Returns inputs and the true jumps."""
    th0, q0 = 290.0, 9.0e-3
    p_half = 1.0e5 - dp * np.arange(nz + 1)
    pi_half = 1.0e5 - p_half
    pi_c = 0.5 * (pi_half[1:] + pi_half[:-1])
    pi_i = pi_half[kb + 1] - m_true * dp
    dq = -4.0e-3

    def above(base, jump, slope, a, b):     # mean of the linear profile over [a, b]
        return base + jump + slope * (0.5 * (a + b) - pi_i)
    thvl = np.full(nz, th0)
    qt = np.full(nz, q0)
    for j in range(kb + 1, nz):
        thvl[j] = above(th0, delta, gam, pi_half[j], pi_half[j + 1])
        qt[j] = above(q0, dq, beta, pi_half[j], pi_half[j + 1])
    thvl[kb] = (1 - m_true) * th0 + m_true * above(th0, delta, gam, pi_i, pi_half[kb + 1])
    qt[kb] = (1 - m_true) * q0 + m_true * above(q0, dq, beta, pi_i, pi_half[kb + 1])
    ql = np.zeros(nz)
    ql[kb - 1] = 2.0e-4
    z = 10.0 * pi_c       # any increasing height
    u = np.full(nz, 5.0)
    v = np.full(nz, -3.0)
    f = lambda a: jnp.asarray(a)[None, :]  # noqa: E731
    return dict(thvl=f(thvl), qt=f(qt), ql=f(ql), u=f(u), v=f(v), z=f(z), p_half=f(p_half),
                kb=kb, m=m_true, delta=delta, dq=dq, pi_c=pi_c, pi_half=pi_half)


def _cand(c, w=W0):
    return rc.diagnose_candidates(c["thvl"], c["ql"], c["p_half"], w)


def _pick(cand, x):
    """Weight-blended value of a per-candidate array (one-hot at hard widths)."""
    return float(jnp.sum(cand.weight * x, axis=1)[0])


def test_v1_reconstruction_returns_the_known_inversion():
    c = _known_column()
    cand = _cand(c)
    i = list(np.asarray(cand.kb)).index(c["kb"])
    np.testing.assert_array_equal(np.asarray(cand.weight[0]), np.eye(cand.kb.size)[i])
    np.testing.assert_allclose(float(cand.m[0, i]), c["m"], rtol=1e-12)
    np.testing.assert_allclose(float(cand.jump_thvl[0, i]), c["delta"], rtol=1e-12)
    np.testing.assert_allclose(_pick(cand, rc.inversion_jump(c["thvl"], cand, W0)), c["delta"], rtol=1e-12)
    np.testing.assert_allclose(_pick(cand, rc.inversion_jump(c["qt"], cand, W0)), c["dq"], rtol=1e-12)


def test_v1_cloud_top_inside_the_inversion_layer():
    c = _known_column()
    c["ql"] = c["ql"].at[0, c["kb"]].set(5.0e-5)     # cloud also in the ambiguous layer
    cand = _cand(c)
    i = list(np.asarray(cand.kb)).index(c["kb"])
    assert float(cand.weight[0, i]) == 1.0
    np.testing.assert_allclose(float(cand.m[0, i]), c["m"], rtol=1e-12)


def test_v1_small_m_uses_lock_double_grid_level_jump():
    c = _known_column(m_true=0.05)
    cand = _cand(c)
    np.testing.assert_allclose(_pick(cand, cand.m), 0.05, rtol=1e-10)
    q = np.asarray(c["qt"])[0]
    np.testing.assert_allclose(_pick(cand, rc.inversion_jump(c["qt"], cand, W0)),
                               q[c["kb"] + 1] - q[c["kb"] - 1], rtol=1e-14)


def test_v1_d0_branch_gives_the_minimum_jump():
    c = _known_column(delta=0.03, gam=0.0)
    cand = _cand(c)
    assert float(jnp.sum(cand.weight)) == 1.0
    np.testing.assert_allclose(_pick(cand, rc.inversion_jump(c["thvl"], cand, W0)), rc.D0, rtol=1e-12)


def test_v1_no_cloud_no_inversion():
    c = _known_column()
    c["ql"] = jnp.zeros_like(c["ql"])
    assert float(jnp.sum(_cand(c).weight)) == 0.0
    assert float(jnp.sum(_cand(c, WD).weight)) == 0.0


def test_cloud_top_weights_one_hot_and_bounded():
    c = jnp.asarray([[0, 1, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]], float)
    P = np.asarray(rc.cloud_top_weights(c))
    np.testing.assert_array_equal(P, [[0, 0, 1, 0, 0, 0], [0] * 6, [0, 0, 0, 0, 0, 1]])
    soft = np.asarray(rc.cloud_top_weights(jnp.asarray([[0.1, 0.9, 0.7, 0.2, 0.6, 0.3]])))
    assert np.all(soft >= 0.0) and soft.sum() <= 1.0 + 1e-15


# ---------------------------------------------------------------- V2
def test_v2_evaporative_enhancement_against_brute_force():
    p, ex = 90000.0, (90000.0 / constants.p_ref) ** constants.kappa
    thl_c = 288.3
    rs = float(sat_mixrat_liq(jnp.asarray(p), jnp.asarray(thl_c * ex)))
    rt_c = 1.05 * rs
    dthl, drt = 8.0, -4.0e-3
    a = lambda x: jnp.asarray([x])  # noqa: E731
    E, chi_s = rc.evaporative_enhancement(a(thl_c), a(rt_c), a(dthl), a(drt), a(p), a(ex), W0)
    E, chi_s = float(E[0]), float(chi_s[0])
    T_s = (thl_c + chi_s * dthl) * ex
    r_s = float(sat_mixrat_liq(jnp.asarray(p), jnp.asarray(T_s)))
    assert abs((rt_c + chi_s * drt) - r_s) / r_s < 1e-10
    chi = np.linspace(0.0, 1.0, 20001)
    thv = np.asarray(rc.theta_v_of(jnp.asarray(thl_c + chi * dthl), jnp.asarray(rt_c + chi * drt),
                                   jnp.full(chi.shape, p), jnp.full(chi.shape, ex), W0))
    b = thv - thv[0]
    mean_b = np.sum(0.5 * (b[1:] + b[:-1]) * np.diff(chi))
    E_bf = 1.0 - 2.0 * mean_b / b[-1]
    assert 0.0 < E < 1.0
    assert abs(E - E_bf) <= 0.02 * abs(E_bf), (E, E_bf)


def test_v2_no_enhancement_without_cloud():
    p, ex = 90000.0, (90000.0 / constants.p_ref) ** constants.kappa
    a = lambda x: jnp.asarray([x])  # noqa: E731
    rs = float(sat_mixrat_liq(jnp.asarray(p), jnp.asarray(288.3 * ex)))
    E, _ = rc.evaporative_enhancement(a(288.3), a(0.8 * rs), a(8.0), a(-3e-3), a(p), a(ex), W0)
    assert float(E[0]) == 0.0


# ---------------------------------------------------------------- V3
def test_v3_entrainment_velocity_by_hand():
    e, zi, d, th0, E = 0.3, 800.0, 8.0, 290.0, 0.3
    l_inf = 0.085 * zi                                   # GB01 (13)
    l_i = 0.4 * zi / (1.0 + 0.4 * zi / l_inf)            # (20), (12)
    A = 0.16 * (1.0 + 15.0 * E)                          # (21)
    w_hand = A * e * np.sqrt(e) / (l_i * constants.g * d / th0)   # (19)
    w = rc.entrainment_velocity(jnp.asarray([e]), jnp.asarray([zi]), jnp.asarray([d]),
                                jnp.asarray([th0]), jnp.asarray([E]))
    np.testing.assert_allclose(float(w[0]), w_hand, rtol=1e-14)




# ---------------------------------------------------------------- V4
def _rows_case():
    rng = np.random.default_rng(1)
    ng, nzm = 3, 9
    nd = 2 * nzm - 1
    lhs = rng.normal(size=(5, ng, nd)) * 0.1
    lhs[2] = 3.0 + rng.random((ng, nd))
    lhs[2, :, 0] = lhs[2, :, -1] = 1.0
    return jnp.asarray(lhs), jnp.asarray(rng.normal(size=(ng, nd))), nzm


def _rows_rc(weight, xi, kbs):
    z = jnp.zeros_like(weight)
    cand = rc.Candidates(jnp.asarray(kbs), weight, *([z] * 10))
    return {"cand": cand, "c_low": 1.0 - xi, "c_F": xi}


def test_v4_constrained_rows_hold_after_the_solve():
    lhs, rhs, nzm = _rows_case()
    kbs = list(range(1, nzm - 3))
    pick = {0: 2, 1: 4}                                # column -> kb; column 2 inactive
    weight = np.zeros((3, len(kbs)))
    for c, k in pick.items():
        weight[c, kbs.index(k)] = 1.0
    xi = jnp.full((3, len(kbs)), 0.6)
    F = jnp.full((3, len(kbs)), -2e-3)
    l2, r2 = rc.routec_constrain_rows(lhs, rhs, _rows_rc(jnp.asarray(weight), xi, kbs), F)
    wpxp = np.asarray(_batched_penta_solve([(l2, r2)])[0][:, 0::2])   # the production solver
    for c, k in pick.items():
        assert abs(wpxp[c, k + 1]) < 1e-12
        assert abs(wpxp[c, k] - 0.4 * wpxp[c, k - 1] - 0.6 * -2e-3) < 1e-12
    np.testing.assert_array_equal(np.asarray(l2[:, 2]), np.asarray(lhs[:, 2]))
    np.testing.assert_array_equal(np.asarray(l2[:, :, 1::2]), np.asarray(lhs[:, :, 1::2]))


def test_v4_split_weights_blend_the_two_rows():
    lhs, rhs, nzm = _rows_case()
    kbs = list(range(1, nzm - 3))
    weight = np.zeros((3, len(kbs)))
    weight[0, kbs.index(3)] = 0.3
    xi = jnp.full((3, len(kbs)), 0.6)
    F = jnp.full((3, len(kbs)), -2e-3)
    l2, r2 = rc.routec_constrain_rows(lhs, rhs, _rows_rc(jnp.asarray(weight), xi, kbs), F)
    r = 2 * 4                                          # top row of candidate kb=3
    np.testing.assert_allclose(np.asarray(l2[:, 0, r]),
                               0.7 * np.asarray(lhs[:, 0, r]) + 0.3 * np.eye(5)[2])
    np.testing.assert_allclose(float(r2[0, r]), 0.7 * float(rhs[0, r]))


# ---------------------------------------------------------------- V5 in CLUBB
def _sc_env(ng=2, nzt=12, dz=125.0):
    """A cloud-topped mixed layer under an inversion at about 790 m on a
    125 m CLUBB grid, with CLUBB moments of a convective boundary layer."""
    nzm = nzt + 1
    zm = np.arange(nzm) * dz
    zt = 0.5 * (zm[1:] + zm[:-1])
    gr = make_clubb_grid(jnp.asarray(np.tile(zm, (ng, 1))), jnp.asarray(np.tile(zt, (ng, 1))))
    zi = np.array([790.0, 830.0])[:ng, None]
    thl = np.where(zt[None] <= zi, 288.3, 295.0 + np.cbrt(np.clip(zt[None] - zi, 0, None)))
    rt = np.where(zt[None] <= zi, 9.45e-3, 1.5e-3 + 3.5e-3 * np.exp((zi - zt[None]) / 500.0))
    p = 101780.0 * np.exp(-zt / 8400.0)
    pm = 101780.0 * np.exp(-zm / 8400.0)
    ex = (p / constants.p_ref) ** constants.kappa
    T = thl * ex
    rho_t = p / (constants.R_d * T)
    rho_m = pm / (constants.R_d * np.interp(zm, zt, T[0]))
    inbl = (zm > 0) & (zm < 780.0)
    wp2 = np.where(inbl, 0.4, 0.02)[None].repeat(ng, 0)
    f = lambda a: jnp.asarray(np.broadcast_to(a, (ng,) + np.shape(a)[-1:]).copy())  # noqa: E731
    state = CLUBBMomentState(
        rtm=f(rt), thlm=f(thl), um=f(np.full(nzt, 5.0)), vm=f(np.full(nzt, -5.0)),
        wp2=f(wp2), wp3=f(np.zeros(nzt)), up2=f(wp2), vp2=f(wp2),
        wprtp=f(np.where(inbl, 3e-5, 0.0)), wpthlp=f(np.where(inbl, 5e-3, 0.0)),
        upwp=f(np.zeros(nzm)), vpwp=f(np.zeros(nzm)),
        rtp2=f(np.full(nzm, 1e-8)), thlp2=f(np.full(nzm, 1e-2)), rtpthlp=f(np.zeros(nzm)))
    zt0, zm0 = jnp.zeros((ng, nzt)), jnp.zeros((ng, nzm))
    forcing = CLUBBForcing(rtm=zt0, thlm=zt0, um=zt0, vm=zt0, wprtp=zm0, wpthlp=zm0,
                           rtp2=zm0, thlp2=zm0, rtpthlp=zm0)
    thv = T / ex * (1 + 0.61 * rt)
    env = dict(
        Lscale=f(np.where(zt < 790.0, 200.0, 20.0)),
        brunt_vaisala_freq_sqd=f(np.where(zm < 790.0, 1e-6, 1e-4)),
        exner_zt=f(ex), p_in_Pa_zt=f(p), p_sfc=jnp.full((ng,), 101780.0),
        thv_ds_zt=f(thv[0]), thv_ds_zm=f(np.interp(zm, zt, thv[0])),
        rho_ds_zm=f(rho_m), rho_ds_zt=f(rho_t[0]), invrs_rho_ds_zm=f(1 / rho_m),
        invrs_rho_ds_zt=f(1 / rho_t[0]), wm_zt=zt0, wm_zm=zm0, sfc_elevation=jnp.zeros((ng,)),
        fcor=jnp.full((ng,), 7.6e-5), ug=f(np.full(nzt, 5.0)), vg=f(np.full(nzt, -5.0)),
        dt=300.0, gr=gr, config=CLUBBConfig(routec_inversion=True), l_implemented=True)
    return state, forcing, env, gr


def test_v5_clubb_conserves_with_route_c():
    state, forcing, env, gr = _sc_env()
    state = state._replace(wprtp=state.wprtp.at[:, 0].set(0.0),
                           wpthlp=state.wpthlp.at[:, 0].set(0.0))
    w = np.asarray(env["rho_ds_zt"]) * np.asarray(gr.dzt)

    def ci(x):
        return np.sum(w * np.asarray(x), axis=1)
    thl0, rt0 = ci(state.thlm), ci(state.rtm)
    s, seen = state, False
    for _ in range(5):
        s, d = advance_clubb_core(s, forcing, **env)
        seen = seen or bool(np.any(np.asarray(d["routec"]["active"]) > 0.5))
        assert np.all(np.isfinite(np.asarray(s.thlm)))
    assert seen, "route C never found the inversion"
    assert np.max(np.abs(ci(s.thlm) - thl0) / thl0) < 1e-12
    assert np.max(np.abs(ci(s.rtm) - rt0) / rt0) < 1e-12


def test_v4_rows_hold_at_the_diagnosed_levels_inside_the_core(monkeypatch):
    """Hard widths: the flux CLUBB solves at the diagnosed kb obeys GB01 C12
    and is zero at M+3/2 (read before the limiter and the clip)."""
    from legoesm.atmosphere.physics.turbulence import clubb as C
    seen = []
    orig = C.xm_wpxp_clipping_and_stats

    def spy(solve_type, xm, wpxp_pre, *a, **k):
        seen.append((solve_type, np.asarray(wpxp_pre)))
        return orig(solve_type, xm, wpxp_pre, *a, **k)
    monkeypatch.setattr(C, "xm_wpxp_clipping_and_stats", spy)
    state, forcing, env, gr = _sc_env()
    hard = CLUBBConfig(routec_inversion=True, **{f: 0.0 for f in _WIDTH_FIELDS})
    _, d = advance_clubb_core(state, forcing, **{**env, "config": hard})
    r = {k: np.asarray(v) for k, v in d["routec"].items()}
    assert np.all(r["active"] == 1.0) and np.all(r["w_e"] > 0.0)
    assert seen
    for st, w in seen:
        F = r["F_rt_eff"] if st == C.MFL_RTM else r["F_thl_eff"]
        for c in range(w.shape[0]):
            k = int(round(float(r["kb_eff"][c])))
            assert abs(w[c, k + 1]) < 1e-14
            np.testing.assert_allclose(w[c, k] - r["c_low_eff"][c] * w[c, k - 1],
                                       r["c_F_eff"][c] * F[c], rtol=1e-10, atol=1e-16)
            # GB01 C12 on the mass flux: the kinematic row carries rho ratios
            rho = np.asarray(env["rho_ds_zm"])[c]
            np.testing.assert_allclose(r["c_low_eff"][c],
                                       (1 - r["xi_eff"][c]) * rho[k - 1] / rho[k], rtol=1e-12)
            assert abs(r["c_low_eff"][c] - (1 - r["xi_eff"][c])) > 1e-4


def test_route_c_changes_the_cloud_top_fluxes():
    state, forcing, env, gr = _sc_env()
    s_on, d_on = advance_clubb_core(state, forcing, **env)
    s_off, d_off = advance_clubb_core(state, forcing, **{**env, "config": CLUBBConfig()})
    assert "routec" not in d_off
    assert np.all(np.asarray(d_on["routec"]["active"]) > 0.5)
    assert np.max(np.abs(np.asarray(s_on.thlm) - np.asarray(s_off.thlm))) > 1e-6


# ---------------------------------------------------------------- RULE 5
_WIDTH_FIELDS = ("routec_ql_width_kg_kg", "routec_p_top_width_pa", "routec_jump_width_k",
                 "routec_m_width", "routec_lock_m_width", "routec_E_width")


def _blend_summary(c, w):
    cand = _cand(c, w)
    return (float(jnp.sum(cand.weight)), _pick(cand, cand.m),
            _pick(cand, rc.inversion_jump(c["thvl"], cand, w)),
            _pick(cand, rc.inversion_jump(c["qt"], cand, w)))


_CASES = {
    "known": dict(),
    "small_m": dict(m_true=0.05),
    "lock_edge": dict(m_true=0.1),
    "d0": dict(delta=0.03, gam=0.0),
    "thin": dict(m_true=0.995),
}
# per case: (sum of weights, blended m, D_i theta_vl, D_i q_t) at the default
# widths minus the hard form, relative where the hard value is non-zero.
# Re-measured 2026-11-21 after switching the logistics to compact C1 ramps
# (vote A/A/A: the logistic midpoints sat on admissible boundary states, e.g.
# a well-mixed layer, and gave a rejected candidate half weight).
_GAP = {
    "d0": [0.0, 0.0, 0.0, 0.0],
    "known": [0.0, 0.0, 0.0, 0.0],
    "lock_edge": [0.0, 0.0, -0.23076923076924002, -0.02912621359221919],  # m = 0.1 on Lock's ramp
    "small_m": [0.0, 0.0, 0.0, 0.0],
    "thin": [0.0, 0.0, 0.0, 0.0],
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_r5_default_gap_per_case(name):
    c = _known_column(**_CASES[name])
    hard, soft = _blend_summary(c, W0), _blend_summary(c, WD)
    got = [(s - h) / h if h != 0.0 else s - h for s, h in zip(soft, hard)]
    if name not in _GAP:
        pytest.fail(f"MEASURED {name!r}: {got!r}")
    np.testing.assert_allclose(got, _GAP[name], rtol=1e-6, atol=1e-12)


def test_r5_default_widths_are_the_config_defaults():
    d = CLUBBConfig()._asdict()
    assert tuple(d[f] for f in _WIDTH_FIELDS) == tuple(WD)
    assert CLUBBConfig().routec_inversion is False


def test_r5_gradients_through_the_switches():
    c = _known_column(m_true=0.12)                     # near Lock's switch
    th = c["thvl"]

    def f(thvl):
        cand = rc.diagnose_candidates(thvl, c["ql"], c["p_half"], WD)
        return jnp.sum(cand.weight * rc.inversion_jump(c["qt"], cand, WD))
    check_grads(f, (th,), order=1, modes=("rev",), eps=1e-4, rtol=1e-3)
    g = np.asarray(jax.grad(f)(th))
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)


def test_r5_enhancement_gradient_via_chi_s():
    p, ex = 90000.0, (90000.0 / constants.p_ref) ** constants.kappa
    rs = float(sat_mixrat_liq(jnp.asarray(p), jnp.asarray(288.3 * ex)))
    a = lambda x: jnp.asarray([x])  # noqa: E731

    def f(rt_c):
        return rc.evaporative_enhancement(a(288.3), rt_c, a(8.0), a(-4e-3), a(p), a(ex), WD)[0][0]
    x = a(1.05 * rs)
    check_grads(f, (x,), order=1, modes=("rev",), eps=1e-7, rtol=1e-3)
    assert float(jax.grad(f)(x)[0]) != 0.0


def test_no_eddy_diffusion_across_the_constrained_flux_levels(monkeypatch):
    """Hard widths: the explicit rtm/thlm diffusion gets Kh = 0 at the two flux
    levels where the entrainment flux is imposed, Kh unchanged elsewhere."""
    from legoesm.atmosphere.physics.turbulence import clubb as C
    got = []
    orig = C.expldiff_rtm_thlm

    def spy(rtm, thlm, Kh_zm, *a, **k):
        got.append(np.asarray(Kh_zm))
        return orig(rtm, thlm, Kh_zm, *a, **k)
    monkeypatch.setattr(C, "expldiff_rtm_thlm", spy)
    state, forcing, env, gr = _sc_env()
    hard = CLUBBConfig(routec_inversion=True, **{f: 0.0 for f in _WIDTH_FIELDS})
    _, d = advance_clubb_core(state, forcing, **{**env, "config": hard})
    Kh, Kh0 = got[-1], np.asarray(d["Kh_zm"])
    for c in range(Kh.shape[0]):
        k = int(round(float(d["routec"]["kb_eff"][c])))
        assert Kh0[c, k] > 0.0 and Kh0[c, k + 1] > 0.0
        assert Kh[c, k] == 0.0 and Kh[c, k + 1] == 0.0
        rest = np.delete(np.arange(Kh.shape[1]), [k, k + 1])
        np.testing.assert_array_equal(Kh[c, rest], Kh0[c, rest])


def test_routec_refused_on_the_diagnostic_clubb_path():
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence.integration import get_turbulence_fn
    cfg = TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(routec_inversion=True))
    with pytest.raises(ValueError, match="routec_inversion needs prognostic"):
        get_turbulence_fn(cfg)
    ok = TurbulenceConfig(scheme="clubb",
                          clubb=CLUBBConfig(routec_inversion=True, prognostic=True))
    assert get_turbulence_fn(ok)[0] == "clubb"


def _E_case(name):
    p, ex = 90000.0, (90000.0 / constants.p_ref) ** constants.kappa
    rs = float(sat_mixrat_liq(jnp.asarray(p), jnp.asarray(288.3 * ex)))
    rt = {"cloudy": 1.05 * rs, "just_saturated": 1.0000005 * rs,
          "unsaturated": 0.8 * rs}[name]
    a = lambda x: jnp.asarray([x])  # noqa: E731
    return a(288.3), a(rt), a(8.0), a(-4e-3), a(p), a(ex)


# per case: E at the default widths minus the hard form (absolute; E is in
# [0, 1]); measured 2026-11-21
_E_GAP = {"cloudy": 0.0001132809321338843, "just_saturated": 2.3604247784370417e-05,
          "unsaturated": 0.0}


@pytest.mark.parametrize("name", ["cloudy", "just_saturated", "unsaturated"])
def test_r5_enhancement_gap_per_case(name):
    args = _E_case(name)
    gap = float(rc.evaporative_enhancement(*args, WD)[0][0]
                - rc.evaporative_enhancement(*args, W0)[0][0])
    if _E_GAP.get(name) is None:
        pytest.fail(f"MEASURED {name!r}: {gap!r}")
    np.testing.assert_allclose(gap, _E_GAP[name], rtol=1e-6, atol=1e-12)


def test_r5_gradients_through_the_cloudy_and_pressure_switches():
    c = _known_column()
    ql = c["ql"].at[0, c["kb"] - 1].set(rc.QL_MIN - 5e-8)    # inside the ql ramp
    ph = c["p_half"]

    def f(ql, p_half):
        cand = rc.diagnose_candidates(c["thvl"], ql, p_half, WD)
        return jnp.sum(cand.weight * cand.m)
    check_grads(f, (ql, ph), order=1, modes=("rev",), eps=1e-9, rtol=1e-3, atol=1e-6)
    gq = np.asarray(jax.grad(f, argnums=0)(ql, ph))
    assert np.all(np.isfinite(gq)) and gq[0, c["kb"] - 1] != 0.0


def test_off_path_never_enters_route_c(monkeypatch):
    """routec_inversion=False: no route C code runs and the scalar diffusion
    gets the unmodified Kh (the default path is the base path)."""
    from legoesm.atmosphere.physics.turbulence import clubb as C

    def boom(*a, **k):
        raise AssertionError("route C entered on the off path")
    monkeypatch.setattr(rc, "routec_constrain_rows", boom)
    monkeypatch.setattr(rc, "clubb_routec_terms", boom)
    got = []
    orig = C.expldiff_rtm_thlm

    def spy(rtm, thlm, Kh_zm, *a, **k):
        got.append(np.asarray(Kh_zm))
        return orig(rtm, thlm, Kh_zm, *a, **k)
    monkeypatch.setattr(C, "expldiff_rtm_thlm", spy)
    state, forcing, env, gr = _sc_env()
    _, d = advance_clubb_core(state, forcing, **{**env, "config": CLUBBConfig()})
    assert "routec" not in d
    np.testing.assert_array_equal(got[-1], np.asarray(d["Kh_zm"]))


def test_gradient_finite_with_rejected_candidates():
    """Zero-weight candidates (any m, e.g. far below the inversion) leave no
    NaN in the forward step or its gradient."""
    state, forcing, env, gr = _sc_env()

    def f(thlm):
        s, _ = advance_clubb_core(state._replace(thlm=thlm), forcing, **env)
        return jnp.sum(s.thlm * s.rtm)
    g = np.asarray(jax.grad(f)(state.thlm))
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)


def _p_shift_column(p_inv):
    """The known column with the inversion top (top of the ambiguous layer)
    moved to pressure p_inv."""
    c = _known_column()
    shift = p_inv - float(c["p_half"][0, c["kb"] + 1])
    return {**c, "p_half": c["p_half"] + shift}


def test_r5_inversion_pressure_gate_gap_and_gradient():
    c = _p_shift_column(rc.P_TOP_PA + 250.0)                # inside the 500 Pa ramp
    hard = float(jnp.sum(_cand(c, W0).weight))
    soft = float(jnp.sum(_cand(c, WD).weight))
    assert hard == 1.0
    np.testing.assert_allclose(soft - hard, -0.5, rtol=1e-12)   # smoothstep(1/2) = 1/2

    def f(p_half):
        return jnp.sum(rc.diagnose_candidates(c["thvl"], c["ql"], p_half, WD).weight)
    check_grads(f, (c["p_half"],), order=1, modes=("rev",), eps=1e-3, rtol=1e-4, atol=1e-9)
    assert float(jnp.max(jnp.abs(jax.grad(f)(c["p_half"])))) > 0.0
    above = _p_shift_column(rc.P_TOP_PA - 1.0)              # inversion above 700 hPa
    assert float(jnp.sum(_cand(above, W0).weight)) == 0.0


def test_slope_choice_is_continuous_through_a_single_zero_slope():
    """One slope crossing zero while the other is small: the smooth choice of
    the slope pair moves continuously (no jump in m)."""
    c = _known_column(gam=0.0)                              # linear part flat above
    ms = []
    for eps in (-1e-9, 0.0, 1e-9):
        th = c["thvl"].at[0, c["kb"] + 2].add(eps * 2000.0)
        cand = rc.diagnose_candidates(th, c["ql"], c["p_half"], WD)
        ms.append(float(jnp.sum(cand.weight * cand.m)))
    assert abs(ms[0] - ms[1]) < 1e-6 and abs(ms[2] - ms[1]) < 1e-6


def test_r5_gradient_through_locks_switch():
    c = _known_column(m_true=0.0975)                    # inside Lock's ramp in m

    def f(qt):
        cand = rc.diagnose_candidates(c["thvl"], c["ql"], c["p_half"], WD)
        return jnp.sum(cand.weight * rc.inversion_jump(qt, cand, WD))
    check_grads(f, (c["qt"],), order=1, modes=("rev",), eps=1e-6, rtol=1e-5)
    cand = rc.diagnose_candidates(c["thvl"], c["ql"], c["p_half"], WD)
    lam = rc._gt(1.0 / rc.LOCK_FRAC_MAX - cand.m, WD.lock)
    assert np.any((np.asarray(lam) > 0.0) & (np.asarray(lam) < 1.0) & (np.asarray(cand.weight) > 0))
