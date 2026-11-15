"""Round-2 candidate r2_ptt_full (r2_ptt_r + radiatively driven top-down branch):
radiative jump, operator form, conservation, control, gradients (x64)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from legoesm.atmosphere.les_suite import candidate as cmod

from legoesm import constants
from tests.atmosphere.les_suite.test_r2_r2_ptt_r import _UW, _VW, N, _column

jax.config.update("jax_enable_x64", True)

_DIR = (Path(__file__).resolve().parents[3] / "scripts" / "validate" / "les_suite"
        / "cosci_candidates" / "round2")
_PATH = _DIR / "r2_ptt_full.py"
_BUDGET_RTOL = 1e-12      # voted D8 (2026-11-15): column budget residual / largest term
_STATE_RTOL = 1e-10       # as r2_ptt_r (voted V8 there): state rebuilt from the flux
_Q0 = -3.0e-5             # [K/s] cloud-top cooling (~ -2.6 K/day)


def _mod():
    spec = importlib.util.spec_from_file_location("_r2_ptt_full_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rad_column(q0=_Q0, law=True, cloudy=True, dt=600.0):
    inp = _column(_UW if law else jnp.nan, _VW if law else jnp.nan, cloudy=cloudy, dt=dt)
    top = (inp.z_full > 850.0) & (inp.z_full < 1000.0)      # upper part of the deck
    return inp._replace(rad_T_tendency=jnp.where(top, q0, 0.0) + 0.0 * inp.z_full)


def _p():
    return cmod.load_candidate(_PATH).default_params()


def test_declares_it_reads_radiation():
    assert cmod.load_candidate(_PATH).reads_rad is True


def test_radiative_jump_is_the_deck_cooling():
    """dF_rad = -c_pd sum sigma m Q: with sigma ~ 1 in the deck it is the deck's
    radiative cooling in W m-2 (positive for cooling)."""
    mod = _mod()
    inp = _rad_column()
    td = mod.top_down(_p(), inp)
    top = np.asarray((inp.z_full > 850.0) & (inp.z_full < 1000.0))
    expect = -constants.c_pd * _Q0 * float(np.sum(np.asarray(inp.mass)[top]))
    assert float(td["d_f"]) > 0.0
    np.testing.assert_allclose(float(td["d_f"]), expect, rtol=1e-3)
    assert 850.0 < float(td["z_t"]) < 1000.0
    assert float(td["w_t"]) > 0.1


def test_no_radiative_cooling_gives_the_reduced_variant_bit_for_bit():
    mod = _mod()
    base = cmod.load_candidate(_DIR / "r2_ptt_r.py")
    for q0 in (0.0, 1e-5):                       # none, and a net-HEATED deck
        inp = _rad_column(q0=q0)
        assert float(mod.top_down(_p(), inp)["w_t"]) == 0.0
        a = jax.tree_util.tree_leaves(mod.fluxes(_p(), inp, mod.init_aux(inp)))
        b = jax.tree_util.tree_leaves(base.fluxes(_p(), inp, base.init_aux(inp)))
        for x, y in zip(a, b):
            assert np.array_equal(np.asarray(x), np.asarray(y))


def test_cloud_top_cooling_sends_cloud_top_air_down():
    """Sources sit at the cooled cloud top (within one cell; on this weak-inversion
    fixture the cell just above the deck is denser than the cloudy air, so it sinks
    through the smallest barrier); the mass lands in or below the deck."""
    mod = _mod()
    inp = _rad_column()
    td = mod.top_down(_p(), inp)
    big_t = np.asarray(td["big_t"])
    assert np.all(big_t >= 0.0) and np.all(np.triu(big_t) == 0.0)
    src, recv = big_t.sum(axis=1), big_t.sum(axis=0)
    assert 850.0 < float(inp.z_full[int(np.argmax(src))]) < 1100.0
    in_or_below_deck = np.asarray(inp.z_full) < 1000.0     # deck top of the fixture
    assert recv[in_or_below_deck].sum() > 0.99 * recv.sum() > 0.0
    base = cmod.load_candidate(_DIR / "r2_ptt_r.py")
    (wt_f, _), _, _ = mod.fluxes(_p(), inp, mod.init_aux(inp))
    (wt_b, _), _, _ = base.fluxes(_p(), inp, base.init_aux(inp))
    assert float(jnp.max(jnp.abs(wt_f - wt_b))) > 1e-6


def test_top_down_operator_is_the_written_flux():
    """(B_t X)_K = -sum_{j<K<=i} M_ij X_i + U_K X_{K-1}, U_K = sum_{j<K<=i} M_ij."""
    mod = _mod()
    rng = np.random.default_rng(3)
    n = 7
    big = np.tril(rng.random((n, n)), -1)
    x = rng.random(n)
    want = np.zeros(n + 1)
    for k in range(1, n):
        u = 0.0
        for i in range(n):
            for j in range(n):
                if j < k <= i:
                    want[k] -= big[i, j] * x[i]
                    u += big[i, j]
        want[k] += u * x[k - 1]
    got = np.asarray(mod.top_down_matrix(jnp.asarray(big)) @ jnp.asarray(x))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-14)
    ones = np.asarray(mod.top_down_matrix(jnp.asarray(big)) @ jnp.ones(n))
    np.testing.assert_allclose(ones, 0.0, atol=1e-14)        # zero net mass per face
    gen = -(np.asarray(mod.top_down_matrix(jnp.asarray(big)))[1:]
            - np.asarray(mod.top_down_matrix(jnp.asarray(big)))[:-1])
    off = gen - np.diag(np.diag(gen))
    assert np.all(off >= -1e-15)                              # Metzler: q_t >= 0 kept


@pytest.mark.parametrize("law", [True, False])
def test_internal_state_closes_column_budgets_with_top_down(law):
    mod = _mod()
    inp = _rad_column(law=law)
    p = _p()
    aux = mod.init_aux(inp)
    bt = mod.top_down_matrix(mod.top_down(p, inp)["big_t"])
    st = mod._base.step(p, inp, aux, bt)
    (_, _), (uw, vw), _ = mod.fluxes(p, inp, aux)
    m, dt, r0 = inp.mass, inp.dt, inp.rho_half[0]
    rows = {
        "H": (m * inp.exner, st["theta_l"], inp.theta_l, r0 * inp.exner_half[0] * inp.w_theta_s),
        "q": (m, st["q_t"], inp.q_t, r0 * inp.w_qt_s),
        "u": (m, st["u"], inp.u, r0 * uw[0]),
        "v": (m, st["v"], inp.v, r0 * vw[0])}
    for name, (w, new, old, sfc) in rows.items():
        store = jnp.sum(w * (new - old)) / dt
        scale = float(jnp.max(jnp.abs(w * new)) / dt + jnp.abs(sfc))
        assert abs(float(store - sfc)) <= _BUDGET_RTOL * scale, name


def test_reported_flux_reproduces_the_internal_state():
    mod = _mod()
    inp = _rad_column()
    p = _p()
    aux = mod.init_aux(inp)
    bt = mod.top_down_matrix(mod.top_down(p, inp)["big_t"])
    st = mod._base.step(p, inp, aux, bt)
    (wt, wq), (uw, vw), _ = mod.fluxes(p, inp, aux)
    for f, x, new in ((wq, inp.q_t, st["q_t"]), (uw, inp.u, st["u"]),
                      (vw, inp.v, st["v"])):
        rebuilt = x + inp.dt * cmod.flux_to_tendency(f, inp.rho_half, inp.mass)
        np.testing.assert_allclose(rebuilt, new, rtol=_STATE_RTOL)
    th = inp.theta_l + inp.dt * cmod.heat_flux_to_theta_tendency(wt, inp, "enthalpy")
    np.testing.assert_allclose(th, st["theta_l"], rtol=_STATE_RTOL)


def test_q_t_stays_non_negative_under_strong_top_down():
    mod = _mod()
    inp = _rad_column(q0=-3e-4)
    p = {**_p(), "a_N": jnp.asarray(5.0)}
    st = mod._base.step(p, inp, mod.init_aux(inp),
                        mod.top_down_matrix(mod.top_down(p, inp)["big_t"]))
    assert float(jnp.min(st["q_t"])) >= 0.0


def test_gradients_through_the_top_down_branch():
    mod = _mod()
    inp = _rad_column()
    aux = mod.init_aux(inp)

    def loss_q(q):
        (wt, wq), (uw, vw), _ = mod.fluxes(_p(), inp._replace(rad_T_tendency=q), aux)
        return jnp.sum(wt ** 2) + jnp.sum(uw ** 2)

    g = jax.grad(loss_q)(inp.rad_T_tendency)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.max(jnp.abs(g))) > 0.0

    def loss_a(a):
        big = mod.top_down({**_p(), "a_N": a}, inp)["big_t"]
        return jnp.sum(mod.top_down_matrix(big) @ inp.theta_l)

    check_grads(loss_a, (jnp.asarray(1.0),), order=2, modes=["rev"])
    assert float(jax.grad(loss_a)(jnp.asarray(1.0))) != 0.0


def test_nan_radiation_propagates():
    mod = _mod()
    inp = _column(_UW, _VW, cloudy=True)                      # default: NaN radiation
    (wt, _), _, _ = mod.fluxes(_p(), inp, mod.init_aux(inp))
    assert not bool(jnp.all(jnp.isfinite(wt)))


@pytest.mark.parametrize("case", ["clear", "no_rad", "heated"])
def test_gradients_are_finite_where_the_branch_is_off(case):
    """Clear sky (h_c at its floor, cooling given), zero radiation, a net-heated deck: the branch is
    off or weak and every gradient, w.r.t. parameters and Q_rad, stays finite."""
    mod = _mod()
    inp = {"clear": lambda: _rad_column(cloudy=False),
           "no_rad": lambda: _rad_column(q0=0.0),
           "heated": lambda: _rad_column(q0=1e-5)}[case]()
    aux = mod.init_aux(inp)

    def loss(p, q):
        (wt, wq), (uw, vw), _ = mod.fluxes(p, inp._replace(rad_T_tendency=q), aux)
        return jnp.sum(wt ** 2) + jnp.sum(wq ** 2) + jnp.sum(uw ** 2) + jnp.sum(vw ** 2)

    gp, gq = jax.grad(loss, argnums=(0, 1))(_p(), inp.rad_T_tendency)
    assert all(bool(jnp.isfinite(v)) for v in gp.values())
    assert bool(jnp.all(jnp.isfinite(gq)))
