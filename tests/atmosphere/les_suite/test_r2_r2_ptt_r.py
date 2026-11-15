"""Round-2 candidate r2_ptt_r (no top-down): budgets, operator conservation pieces,
gradients and the CloudBench surface-law protocol (x64)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.les_suite import candidate as cmod
from legoesm.atmosphere.les_suite.scm_coupling import saturation_adjust
from legoesm.atmosphere.physics._shared import exner_function

from legoesm import constants

jax.config.update("jax_enable_x64", True)

_PATH = (Path(__file__).resolve().parents[3] / "scripts" / "validate" / "les_suite"
         / "cosci_candidates" / "round2" / "r2_ptt_r.py")
_UW, _VW = -0.09, 0.02
N = 16
_BUDGET_RTOL = 1e-10      # voted (V8): column budget residual / surface input
_STATE_RTOL = 1e-10       # voted (V8): expm state vs state rebuilt from reported flux


def _mod():
    spec = importlib.util.spec_from_file_location("_r2_ptt_r_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _column(uw_s, vw_s, z0_m=1e-3, sfc_drag_most=1.0, cloudy=False, dt=600.0):
    z_half = jnp.linspace(0.0, 1600.0, N + 1)
    z_full = 0.5 * (z_half[1:] + z_half[:-1])
    rho_full = 1.2 * jnp.exp(-z_full / 8000.0)
    p_half = 1.0e5 * jnp.exp(-z_half / 8000.0)
    p_full = 1.0e5 * jnp.exp(-z_full / 8000.0)
    exner = exner_function(p_full)
    theta_l = 300.0 + 0.004 * jnp.maximum(z_full - 600.0, 0.0)
    q_t = 0.01 - 2e-6 * z_full
    extra = {}
    if cloudy:   # saturated deck between 700 and 1000 m
        deck = (z_full > 700.0) & (z_full < 1000.0)
        # weakly unstable sublayer: an EXACTLY neutral one puts the shared PBL-height
        # helper's bulk-Ri pairs at roundoff level, where its interpolation fraction
        # flips 0 <-> 1 with the roundoff sign (h moves ~0.7 m between eager and jit;
        # helper defect, recorded in the round record, not fixable in this file)
        theta_l = jnp.where(z_full > 600.0, 296.0 + 0.004 * (z_full - 600.0),
                            296.0 + 5e-4 * (600.0 - z_full))
        q_t = jnp.where(deck, 0.0185, jnp.where(z_full < 700.0, 0.012, 0.006))
        theta, q_v, q_c = saturation_adjust(theta_l, q_t, exner, p_full)
        extra = dict(temp=theta * exner, q_v=q_v, q_c=q_c)
    return cmod.column_inputs(
        z_full=z_full, z_half=z_half, rho_full=rho_full,
        rho_half=cmod.half_level_average(rho_full, rho_full[0]),
        mass=(p_half[:-1] - p_half[1:]) / constants.g, exner=exner,
        exner_half=exner_function(p_half), p_full=p_full, p_half=p_half,
        theta_l=theta_l, q_t=q_t, u=5.0 + 0.002 * z_full, v=jnp.sin(z_full / 300.0),
        w_theta_s=0.05, w_qt_s=1e-4, uw_s=uw_s, vw_s=vw_s, dt=dt,
        z0_m=z0_m, z0h_m=z0_m, sfc_drag_most=sfc_drag_most, **extra)


def _cand():
    return cmod.load_candidate(_PATH)


def _run(inp, params=None, aux=None):
    cand = _cand()
    p = cand.default_params() if params is None else params
    return cand.fluxes(p, inp, cand.init_aux(inp) if aux is None else aux)


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


COLUMNS = {"law": dict(uw_s=_UW, vw_s=_VW), "own": dict(uw_s=jnp.nan, vw_s=jnp.nan),
           "law_cloudy": dict(uw_s=_UW, vw_s=_VW, cloudy=True),
           "own_cloudy": dict(uw_s=jnp.nan, vw_s=jnp.nan, cloudy=True)}


@pytest.mark.parametrize("col", list(COLUMNS))
def test_contract_ends_and_finite_at_600s(col):
    inp = _column(**COLUMNS[col])
    (wt, wq), (uw, vw), aux = _run(inp)
    assert float(wt[0]) == 0.05 and float(wq[0]) == 1e-4
    for f in (wt, wq, uw, vw, aux):
        assert f.shape == (N + 1,) and bool(jnp.all(jnp.isfinite(f)))
    for f in (wt, wq, uw, vw):
        assert float(f[-1]) == 0.0
    if col.startswith("law"):
        assert (float(uw[0]), float(vw[0])) == (_UW, _VW)


@pytest.mark.parametrize("col", list(COLUMNS))
def test_internal_expm_state_closes_column_budgets(col):
    """in - out - d(storage) = 0 for H (= c_pd Pi theta_l), q_t, u, v on the expm state
    (independent of the flux recovery): the operator's mass-weighted column sums vanish
    and the affine surface column delivers exactly the surface input."""
    mod = _mod()
    inp = _column(**COLUMNS[col])
    p = _cand().default_params()
    aux = mod.init_aux(inp)
    st = mod.step(p, inp, aux)
    (_, _), (uw, vw), _ = mod.fluxes(p, inp, aux)
    m, dt, r0 = inp.mass, inp.dt, inp.rho_half[0]
    checks = {
        "H": (jnp.sum(m * inp.exner * (st["theta_l"] - inp.theta_l)),
              dt * r0 * inp.exner_half[0] * inp.w_theta_s),
        "q_t": (jnp.sum(m * (st["q_t"] - inp.q_t)), dt * r0 * inp.w_qt_s),
        "u": (jnp.sum(m * (st["u"] - inp.u)), dt * r0 * uw[0]),
        "v": (jnp.sum(m * (st["v"] - inp.v)), dt * r0 * vw[0]),
    }
    for name, (dstore, inflow) in checks.items():
        res = abs(float(dstore - inflow))
        assert res <= _BUDGET_RTOL * abs(float(inflow)), (name, res, float(inflow))
    if col.startswith("law"):   # u, v: the PASSED stress is the column's input
        np.testing.assert_allclose(float(checks["u"][1]), dt * r0 * _UW, rtol=1e-14)


@pytest.mark.parametrize("col", list(COLUMNS))
def test_reported_flux_reproduces_the_expm_state_and_implied_surface_flux(col):
    """The harness's forward step with the reported fluxes (enthalpy heat form) lands on
    the expm state; the surface flux the state change implies equals the prescribed
    one (so overwriting index 0 hides nothing)."""
    mod = _mod()
    inp = _column(**COLUMNS[col])
    p = _cand().default_params()
    aux = mod.init_aux(inp)
    st = mod.step(p, inp, aux)
    (wt, wq), (uw, vw), _ = mod.fluxes(p, inp, aux)
    dt = inp.dt
    rebuilt = {
        "theta_l": inp.theta_l + dt * cmod.heat_flux_to_theta_tendency(wt, inp, "enthalpy"),
        "q_t": inp.q_t + dt * cmod.flux_to_tendency(wq, inp.rho_half, inp.mass),
        "u": inp.u + dt * cmod.flux_to_tendency(uw, inp.rho_half, inp.mass),
        "v": inp.v + dt * cmod.flux_to_tendency(vw, inp.rho_half, inp.mass),
    }
    for k, x in rebuilt.items():
        x0 = getattr(inp, k)
        rel = float(jnp.max(jnp.abs(x - st[k])) / jnp.max(jnp.abs(st[k] - x0)))
        assert rel <= _STATE_RTOL, (k, rel)
    imp = mod.implied_surface_fluxes(p, inp, aux)
    np.testing.assert_allclose(float(imp["wtheta"]), 0.05, rtol=_BUDGET_RTOL)
    np.testing.assert_allclose(float(imp["wqt"]), 1e-4, rtol=_BUDGET_RTOL)
    if col.startswith("law"):
        np.testing.assert_allclose([float(imp["uw"]), float(imp["vw"])], [_UW, _VW],
                                   rtol=_BUDGET_RTOL)


@pytest.mark.parametrize("cloudy", [False, True])
def test_exchange_operators_carry_zero_net_mass_through_every_face(cloudy):
    """Facewise mass balance: a uniform field gives zero flux through every interface
    (updraft mass returned exactly by the cascade); the nonlocal part is active and
    not symmetric (skewed: it is not a symmetric pair exchange)."""
    mod = _mod()
    inp = _column(_UW, _VW, cloudy=cloudy)
    st = mod.step(_cand().default_params(), inp, mod.init_aux(inp))
    ones = jnp.ones(N)
    for b in (st["b_nl"], st["b_loc"]):
        scale = float(jnp.max(jnp.sum(jnp.abs(b), axis=1)))
        assert scale > 0.0
        assert float(jnp.max(jnp.abs(b @ ones))) <= 1e-13 * scale
    big_m = st["op"]["big_m"]
    assert float(jnp.sum(big_m)) > 0.0
    sym = mod._crossing_matrix(big_m)       # symmetric pair exchange with the same M
    assert float(jnp.max(jnp.abs((st["b_nl"] - sym) @ inp.q_t))) > 1e-6 * float(
        jnp.max(jnp.abs(sym @ inp.q_t)))


def test_rotation_does_no_work():
    mod = _mod()
    inp = _column(_UW, _VW)
    st = mod.step(_cand().default_params(), inp, mod.init_aux(inp))
    bb = st["b_rot"]
    gu, gv = -(bb @ inp.v), bb @ inp.u
    work = gu[1:N] * (inp.u[1:] - inp.u[:-1]) + gv[1:N] * (inp.v[1:] - inp.v[:-1])
    assert float(jnp.sum(jnp.abs(work))) > 0.0
    assert float(jnp.abs(jnp.sum(work)) / jnp.sum(jnp.abs(work))) <= 1e-12


def test_q_t_propagator_is_nonnegative():
    mod = _mod()
    inp = _column(jnp.nan, jnp.nan, cloudy=True)
    st = mod.step(_cand().default_params(), inp, mod.init_aux(inp))
    b = st["b_loc"] + st["b_nl"]
    a = -mod._div(b) / inp.mass[:, None]
    off = a - jnp.diag(jnp.diag(a))
    assert float(jnp.min(off)) >= 0.0                     # Metzler
    prop = jax.scipy.linalg.expm(inp.dt * a, max_squarings=mod._MAX_SQUARINGS)
    assert float(jnp.min(prop)) >= -1e-12 * float(jnp.max(prop))


def test_total_energy_budget_closes_with_harness_ke_heating():
    inp = _column(jnp.nan, jnp.nan)
    fa, fb, _ = _run(inp)
    tend = cmod.candidate_tendencies("both", fa, fb, inp, heat_form="enthalpy",
                                     ke_heating=True)
    eb = cmod.column_energy_budget(tend, fa, fb, inp, "both")
    assert abs(float(eb.residual)) <= 1e-10 * abs(float(eb.boundary_input))


def test_cloud_weight_selects_saturated_stratification_in_the_deck():
    mod = _mod()
    inp = _column(_UW, _VW, cloudy=True)
    sig = mod._cloud_weight(inp)
    deck = np.asarray((inp.z_full > 700.0) & (inp.z_full < 1000.0))
    assert float(jnp.min(sig[deck])) > 0.99 and float(jnp.max(sig[~deck])) < 0.01
    o = mod._operator(_cand().default_params(), inp, mod.init_aux(inp))
    # an interior deck interface: N_e^2 = saturated (lifted-parcel) value
    b_par = mod._parcels(inp)
    k = int(np.argmax(deck)) + 1          # interface between the two lowest deck cells
    n2_s = -float(b_par[k - 1, k]) / float(inp.z_full[k] - inp.z_full[k - 1])
    np.testing.assert_allclose(float(o["n2_e"][k - 1]), n2_s, rtol=2e-2)


def test_memory_relaxes_exactly_and_is_not_inert():
    mod = _mod()
    inp = _column(_UW, _VW)
    p = _cand().default_params()
    aux0 = mod.init_aux(inp)
    (wt, wq), _fb, b1 = mod.fluxes(p, inp, aux0)
    sc = mod._scales(inp)
    th_h = cmod.half_level_average(inp.theta_l, inp.theta_l[0])
    qt_h = cmod.half_level_average(inp.q_t, inp.q_t[0])
    eps = mod._EPS_V
    b_now = constants.g * ((1 + eps * qt_h) * wt + eps * th_h * wq) / (th_h * (1 + eps * qt_h))
    decay = np.exp(-float(inp.dt) * float(sc["v0"]) / float(sc["h"]))
    np.testing.assert_allclose(np.asarray(b1), np.asarray(b_now) * (1 - decay), rtol=1e-12,
                               atol=1e-300)
    # carried memory is kept, not reset (codex r1 P1: from aux=0 alone the update
    # b_now*(1-decay) is indistinguishable from the exact relaxation)
    aux1 = aux0 + 0.05
    (wt1, wq1), _fb1, b2 = mod.fluxes(p, inp, aux1)
    b_now1 = constants.g * ((1 + eps * qt_h) * wt1 + eps * th_h * wq1) / (th_h * (1 + eps * qt_h))
    np.testing.assert_allclose(np.asarray(b2),
                               np.asarray(b_now1) + (np.asarray(aux1) - np.asarray(b_now1)) * decay,
                               rtol=1e-12, atol=1e-300)
    a = _leaves(mod.fluxes(p, inp, aux0)[:2])
    b = _leaves(mod.fluxes(p, inp, aux1)[:2])
    assert max(float(np.max(np.abs(x - y))) for x, y in zip(a, b, strict=True)) > 0.0


def _loss(cand, inp):
    def loss(p):
        (wt, wq), (uw, vw), _ = cand.fluxes(p, inp, cand.init_aux(inp))
        return (jnp.sum(wt ** 2) + 1e6 * jnp.sum(wq ** 2)
                + jnp.sum(uw[1:] ** 2 + vw[1:] ** 2))
    return loss


@pytest.mark.parametrize("col", ["law", "own"])
def test_every_parameter_has_a_finite_nonzero_gradient(col):
    cand = _cand()
    inp = _column(**COLUMNS[col])
    loss = _loss(cand, inp)
    p0 = cand.default_params()
    g = jax.grad(loss)(p0)
    for k, v in g.items():
        assert np.isfinite(float(v)) and abs(float(v)) > 0.0, (k, float(v))
    for k in ("a_N", "c_ez"):          # voted (V14): central FD check
        h = 1e-5 * float(p0[k])
        fd = (float(loss({**p0, k: p0[k] + h})) - float(loss({**p0, k: p0[k] - h}))) / (2 * h)
        np.testing.assert_allclose(float(g[k]), fd, rtol=1e-5)


@pytest.mark.parametrize("col", list(COLUMNS))
def test_jit_matches_eager(col):
    cand = _cand()
    inp = _column(**COLUMNS[col])
    p = cand.default_params()
    aux = cand.init_aux(inp) + 0.05     # GLM r1 P3-2: carried (nonzero) memory too
    a = _leaves((cand.fluxes(p, inp, cand.init_aux(inp)), cand.fluxes(p, inp, aux)))
    b = _leaves((jax.jit(cand.fluxes)(p, inp, cand.init_aux(inp)),
                 jax.jit(cand.fluxes)(p, inp, aux)))
    # voted (V15, codex/GLM/author all c): per-leaf normwise. The fluxes are recovered
    # from a difference of nearly equal expm states, so XLA's reordering of the
    # arithmetic gives a deterministic eager-jit gap bounded by eps * cond(expm) *
    # |X|/|dX| (theta_l bound ~1e-7, measured worst pointwise 1.9e-9); an elementwise
    # rtol blows up on near-zero flux elements.
    for x, y in zip(a, b, strict=True):
        assert np.max(np.abs(x - y)) <= 1e-9 * np.max(np.abs(y)), (
            float(np.max(np.abs(x - y))), float(np.max(np.abs(y))))


# --- CloudBench surface-law protocol (as test_r1_candidate_surface_law) -------------

def test_candidate_uses_the_passed_stress():
    _fa, (uw, vw), _aux = _run(_column(_UW, _VW))
    np.testing.assert_allclose([float(uw[0]), float(vw[0])], [_UW, _VW], rtol=1e-12, atol=0.0)


def test_law_column_ignores_the_candidates_own_surface_law(monkeypatch):
    """z0 also sets mixing lengths (z + z0), so the own-law-only knob is psi_m (as for
    the parent); every flux and the new aux must be bit-identical."""
    a = _leaves(_run(_column(_UW, _VW)))
    import legoesm.core.bulk_flux as bf
    monkeypatch.setattr(bf, "psi_m", lambda zeta, *a, **k: 0.0 * zeta + 0.3)
    b = _leaves(_run(_column(_UW, _VW)))
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    # control: the patch bites on an own-law column
    c = _leaves(_run(_column(jnp.nan, jnp.nan)))
    monkeypatch.undo()
    d = _leaves(_run(_column(jnp.nan, jnp.nan)))
    assert max(float(np.max(np.abs(x - y))) for x, y in zip(c, d, strict=True)) > 0.0


def test_nan_stress_keeps_the_own_law_bit_identically(monkeypatch):
    inp = _column(jnp.nan, jnp.nan)
    got = _leaves(_run(inp))
    monkeypatch.setattr(cmod, "surface_stress", lambda _x, uw, vw, us: (uw, vw, us))
    ref = _leaves(_run(inp))
    for x, y in zip(got, ref, strict=True):
        np.testing.assert_array_equal(x, y)
    _fa, (uw, _vw), _aux = _run(inp)
    assert abs(float(uw[0]) - _UW) > 1e-3 * abs(_UW)
    assert float(uw[0]) < 0.0             # own drag opposes u > 0


def test_gradient_through_a_law_column_is_finite():
    cand = _cand()
    inp = _column(_UW, _VW, sfc_drag_most=jnp.nan)
    g = jax.grad(_loss(cand, inp))(cand.default_params())
    assert all(np.isfinite(float(v)) for v in g.values())


def test_nan_roughness_on_a_law_column_uses_zero_in_the_mixing_lengths():
    """Voted V16 (a): z0_m = NaN (contract: column has none) with a passed stress gives
    the z0_m = 0 column's fluxes, not NaN (codex r1 P1)."""
    a = _leaves(_run(_column(_UW, _VW, z0_m=jnp.nan)))
    b = _leaves(_run(_column(_UW, _VW, z0_m=0.0)))
    for x, y in zip(a, b, strict=True):
        assert np.all(np.isfinite(x))
        np.testing.assert_array_equal(x, y)   # same arithmetic once z0 -> 0: no tolerance


@pytest.mark.parametrize("z0", [jnp.nan, 0.0])
def test_nan_roughness_on_a_law_column_has_finite_gradients(z0):
    """Voted V17 (double-where in the masked own-law drag; codex r2/r3 P1): with z0_m =
    NaN or 0 on a law column the wind gradient and a 2-step rollout's parameter
    gradients are finite (they were NaN: 0 * NaN through the masked drag)."""
    cand = _cand()
    p0 = cand.default_params()
    inp = _column(_UW, _VW, z0_m=z0)

    def f_u(u):
        (wt, _wq), (uw, vw), b = cand.fluxes(p0, inp._replace(u=u), cand.init_aux(inp))
        return jnp.sum(wt ** 2) + jnp.sum(uw ** 2) + jnp.sum(vw ** 2) + jnp.sum(b ** 2)

    def roll(p):
        (_wt, _wq), (uw, _vw), b = cand.fluxes(p, inp, cand.init_aux(inp))
        u1 = inp.u - inp.dt * (uw[1:] - uw[:-1]) / (inp.z_half[1:] - inp.z_half[:-1])
        (wt2, _), (uw2, _), _ = cand.fluxes(p, inp._replace(u=u1), b)
        return jnp.sum(wt2 ** 2) + jnp.sum(uw2 ** 2)

    assert np.all(np.isfinite(np.asarray(jax.grad(f_u)(inp.u))))
    for k, v in jax.grad(roll)(p0).items():
        assert np.isfinite(float(v)), k
