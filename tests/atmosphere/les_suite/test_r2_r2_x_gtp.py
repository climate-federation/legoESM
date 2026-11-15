"""Round-2 candidate X-GTP (plume-genealogy transilient matrix): contract, x64 budget
closure, genealogy mass budget, air-mass return, exact-update consistency, TKE
production identity, positivity, memory relaxation, gradients and the CloudBench
surface-law protocol (PI 2026-11-11)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.les_suite import candidate as cmod
from legoesm.atmosphere.les_suite.candidate_eval import e0_column_checks
from legoesm.atmosphere.les_suite.scm_coupling import saturation_adjust
from legoesm.atmosphere.physics._shared import exner_function

from legoesm import constants

jax.config.update("jax_enable_x64", True)

_PATH = (Path(__file__).resolve().parents[3] / "scripts" / "validate" / "les_suite"
         / "cosci_candidates" / "round2" / "r2_x_gtp.py")
_UW, _VW = -0.09, 0.02


def _module():
    spec = importlib.util.spec_from_file_location("_r2_x_gtp", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cand():
    return cmod.load_candidate(_PATH)


def _column(uw_s=jnp.nan, vw_s=jnp.nan, n=40, top=2400.0, z0_m=0.1, sfc_drag_most=1.0,
            dt=600.0, qt=None, wq_s=5e-5):
    """Moist convective column: mixed layer, cloud layer (q_c > 0), stable top."""
    zh = jnp.linspace(0.0, top, n + 1)
    zf = 0.5 * (zh[1:] + zh[:-1])
    H = constants.R_d * 290.0 / constants.g
    ph, pf = 1.0e5 * jnp.exp(-zh / H), 1.0e5 * jnp.exp(-zf / H)
    exf, exh = exner_function(pf), exner_function(ph)
    th = 298.0 + 0.003 * jnp.maximum(zf - 600.0, 0.0) + 0.006 * jnp.maximum(zf - 1500.0, 0.0)
    if qt is None:
        qt = 0.0175 - 0.0035 * jnp.clip((zf - 1500.0) / 400.0, 0.0, 1.0)
    theta, q_v, q_c = saturation_adjust(th, qt, exf, pf)
    rho_f = pf / (constants.R_d * theta * exf * (1.0 + (1.0 / constants.epsilon - 1.0) * q_v
                                                 - q_c))
    return cmod.column_inputs(
        z_full=zf, z_half=zh, rho_full=rho_f, rho_half=cmod.half_level_average(rho_f, rho_f[0]),
        mass=(ph[:-1] - ph[1:]) / constants.g, exner=exf, exner_half=exh, p_full=pf,
        p_half=ph, theta_l=th, q_t=qt, u=5.0 + 0.002 * zf, v=-1.0 + jnp.sin(zf / 400.0),
        w_theta_s=0.08, w_qt_s=wq_s, uw_s=uw_s, vw_s=vw_s, f_c=1e-4, dt=dt,
        temp=theta * exf, q_v=q_v, q_c=q_c, z0_m=z0_m, z0h_m=z0_m,
        sfc_drag_most=sfc_drag_most)


def _p0():
    return _cand().default_params()


def test_column_is_cloudy_and_plume_active():
    inp = _column()
    assert float(jnp.max(inp.q_c)) > 1e-4        # sigma and N_sat^2 exercised
    d = _module()._solve(_p0(), inp, _cand().init_aux(inp))[3]
    surv = jnp.sum(d["Pint"], axis=1) / d["M0"]
    assert float(d["M0"]) > 1e-3 and float(jnp.max(surv[2:])) > 0.1


@pytest.mark.parametrize("drag_most", [0.0, 1.0])
@pytest.mark.parametrize("heat_form", cmod.HEAT_FORMS)
def test_contract_and_column_budget_closure(drag_most, heat_form):
    cand = _cand()
    inp = _column(sfc_drag_most=drag_most)
    r = e0_column_checks(cand, cand.default_params(), inp, heat_form=heat_form,
                         ke_heating=True)
    assert r["finite"] and r["surface_exact"] and r["top_zero"], r
    assert max(r["closure_rel"].values()) < 1e-12, r["closure_rel"]
    assert r["jit_parity"] < 1e-12 and r["surface_stress_power"] < 0.0, r
    if heat_form == "enthalpy":
        assert r["energy_rel"] < 1e-12, r["energy_rel"]


@pytest.mark.parametrize("case", ["own", "law", "deep"])
def test_reported_fluxes_reproduce_the_exact_update(case):
    """Harness forward-Euler update with the reported fluxes == internal expm state."""
    if case == "deep":
        inp, prm = _deep_plume_case()
    else:
        inp, prm = (_column(_UW, _VW) if case == "law" else _column()), _p0()
    mod = _module()
    fs, fm, _aux, d = mod._solve(prm, inp, _cand().init_aux(inp))
    th1 = inp.theta_l + inp.dt * cmod.heat_flux_to_theta_tendency(fs[0], inp, "enthalpy")
    q1 = inp.q_t + inp.dt * cmod.flux_to_tendency(fs[1], inp.rho_half, inp.mass)
    u1 = inp.u + inp.dt * cmod.flux_to_tendency(fm[0], inp.rho_half, inp.mass)
    v1 = inp.v + inp.dt * cmod.flux_to_tendency(fm[1], inp.rho_half, inp.mass)
    X = d["X_new"]
    for got, ref, x0 in ((th1, X[0], inp.theta_l), (q1, X[1], inp.q_t),
                         (u1, X[2], inp.u), (v1, X[3], inp.v)):
        assert float(jnp.max(jnp.abs(got - ref))) <= 1e-10 * float(jnp.max(jnp.abs(x0)))
        assert float(jnp.max(jnp.abs(ref - x0))) > 1e-6 * float(jnp.max(jnp.abs(x0)))


def _deep_plume_case():
    """Dry, weakly unstable column with the weakest hazard: the plume reaches the absorbing
    layer, so the lid and the taper are exercised (the default case dies by ~700 m).
    Density and layer mass are kept from the moist column (slightly inconsistent with the
    replaced theta_l; irrelevant to the budget/structure properties tested with it)."""
    inp = _column(qt=jnp.full((40,), 0.004))
    th = 300.0 - 1e-3 * inp.z_full                      # weakly superadiabatic
    inp = inp._replace(theta_l=th, temp=th * inp.exner,
                       q_v=jnp.full((40,), 0.004), q_c=jnp.zeros((40,)))
    return inp, dict(_p0(), C_A=jnp.asarray(0.05))


@pytest.mark.parametrize("deep", [False, True])
def test_genealogy_mass_budget(deep):
    """Every origin's injected mass (launch, entrainment) is detrained inside the column;
    nothing leaves through the lid; sum_k phi_k = 1 is P's row sum = rho M."""
    inp, prm = _deep_plume_case() if deep else (_column(), _p0())
    d = _module()._solve(prm, inp, _cand().init_aux(inp))[3]
    P, E = d["Pint"], d["E"]
    n = inp.mass.shape[0]
    surv = jnp.sum(P, axis=1) / d["M0"]
    if deep:
        assert float(surv[n - 1]) > 1e-3        # control: the plume reaches the top cell
    assert float(jnp.max(jnp.abs(P[n]))) == 0.0
    injected = jnp.concatenate([d["M0"][None], P[jnp.arange(2, n + 1), jnp.arange(1, n)]])
    out = jnp.sum(E, axis=1)
    assert float(jnp.max(jnp.abs(out - injected))) <= 1e-12 * float(jnp.sum(injected))
    assert float(jnp.sum(E[:, 1:])) > 0.5 * float(d["M0"])
    assert bool(jnp.all(P >= 0.0)) and bool(jnp.all(E >= 0.0))
    assert bool(jnp.all(jnp.triu(P[1:n], 1) == 0.0))   # interface K holds origins < K only


def test_uniform_field_is_a_fixed_point_air_mass_is_returned():
    """Constant q_t: the plume moves air up, subsidence returns it, no net flux."""
    inp = _column(qt=jnp.full((40,), 0.012), wq_s=0.0)
    mod = _module()
    fs, _fm, _aux, d = mod._solve(_p0(), inp, _cand().init_aux(inp))
    assert float(jnp.max(jnp.sum(d["Pint"], axis=1))) > 1e-3     # control: plume active
    assert float(jnp.max(jnp.abs(fs[1]))) <= 1e-13 * 0.012 * float(jnp.max(jnp.sum(d["Pint"], 1)))
    np.testing.assert_allclose(np.asarray(d["X_new"][1]), 0.012, rtol=1e-13, atol=0.0)


def test_q_t_stays_non_negative_at_long_step():
    """Metzler generator: q_t >= 0 with no clipping, dry layer above a moist one."""
    qt = jnp.where(jnp.arange(40) < 12, 0.015, 0.0)
    inp = _column(qt=qt, dt=3600.0)
    d = _module()._solve(_p0(), inp, _cand().init_aux(inp))[3]
    q1 = d["X_new"][1]
    assert float(jnp.min(q1)) >= 0.0
    assert float(jnp.max(q1[12:])) > 1e-6                       # moisture did move up


@pytest.mark.parametrize("law", [False, True])
def test_tke_production_is_the_exact_mean_ke_loss(law):
    inp = _column(_UW, _VW) if law else _column()
    d = _module()._solve(_p0(), inp, _cand().init_aux(inp))[3]
    m = inp.mass
    dke = jnp.sum(m * 0.5 * (d["U1"] ** 2 + d["V1"] ** 2 - inp.u ** 2 - inp.v ** 2)) / inp.dt
    prod = jnp.sum(m * (d["Ps"] + d["Psfc"]))
    assert abs(float(prod + dke)) <= 1e-12 * float(jnp.sum(m * jnp.abs(d["Ps"])))
    assert float(prod) > 0.0


def test_memory_relaxes_exactly_toward_the_present_genealogy():
    mod = _module()
    cand = _cand()
    inp = _column(dt=60.0)       # tau_A ~ 3 min here: dt = 600 s would leave w ~ 0.03
    p0 = cand.default_params()
    aux0 = cand.init_aux(inp)
    _, _, aux1, d1 = mod._solve(p0, inp, aux0)
    np.testing.assert_array_equal(np.asarray(d1["P_use"]), np.asarray(d1["Pint"]))  # 1st call
    aux_z = dict(aux1, P_mem=jnp.zeros_like(aux1["P_mem"]))
    _, _, _, d2 = mod._solve(p0, inp, aux_z)
    w = jnp.exp(-inp.dt / d2["tau_A"])
    np.testing.assert_allclose(np.asarray(d2["P_use"]), np.asarray((1.0 - w) * d2["Pint"]),
                               rtol=1e-13, atol=1e-300)
    assert 0.05 < float(w) < 0.99 and float(d2["tau_A"]) > 10.0


def _jet_case():
    """Deep plume under a low-level jet over a wind minimum: the plume carries jet
    momentum up into increasing shear (counter-gradient nonlocal flux), weak TKE."""
    inp, prm = _deep_plume_case()
    z = inp.z_full
    u = 3.0 + 8.0 * jnp.exp(-((z - 200.0) / 150.0) ** 2) + 0.004 * jnp.maximum(z - 900.0, 0.0)
    inp = inp._replace(u=u)
    aux = dict(_cand().init_aux(inp), e=jnp.full((40,), 1e-3))
    return inp, prm, aux


def test_phi_R_acts_only_on_counter_gradient_flux_and_is_applied(monkeypatch):
    mod = _module()
    inp, prm, aux = _jet_case()
    d = mod._solve(prm, inp, aux)[3]
    phi, P_nl = d["phi_R"], d["P_nl"]
    assert bool(jnp.all((phi > 0.0) & (phi <= 1.0)))
    assert bool(jnp.all(jnp.where(P_nl >= 0.0, phi == 1.0, True)))   # zero demand: no damping
    assert float(jnp.min(phi)) < 0.99                                 # control: acts strongly
    monkeypatch.setattr(mod, "_realizability", lambda e_h, p_nl, dt: jnp.ones_like(e_h))
    d1 = mod._solve(prm, inp, aux)[3]
    assert float(jnp.max(jnp.abs(d1["U1"] - d["U1"]))) > 1e-6          # factor is applied
    np.testing.assert_array_equal(np.asarray(d1["X_new"][1]), np.asarray(d["X_new"][1]))


def test_phi_R_is_exactly_one_without_demand():
    """Where the nonlocal momentum flux vanishes (above the dead plume) and TKE is weak,
    phi_R must not damp anything (a softplus gate would give ~0.2 at e ~ 1e-6)."""
    inp = _column()
    aux = dict(_cand().init_aux(inp), e=jnp.full((40,), 1e-6))
    d = _module()._solve(_p0(), inp, aux)[3]
    none = jnp.abs(d["P_nl"]) < 1e-20                  # m2 s-3: no energy demand
    assert int(jnp.sum(none)) >= 10                                  # control
    assert bool(jnp.all(jnp.where(d["P_nl"] >= -1e-20, d["phi_R"] == 1.0, True)))


@pytest.mark.parametrize("profile", ["uniform", "layered"])
def test_rain_does_not_change_the_plume_thermodynamics(profile):
    """q_t carries rain; parcel buoyancy and the TKE buoyancy production must use
    q_v + q_c only (a nonuniform rain flux carries no vapour buoyancy)."""
    mod = _module()
    dry = _column()
    q_r = 2e-3 if profile == "uniform" else 2e-3 * jnp.exp(-((dry.z_full - 1500.0) / 300.0) ** 2)
    rainy = dry._replace(q_t=dry.q_t + q_r)           # + rain, q_v, q_c unchanged
    a = mod._solve(_p0(), dry, _cand().init_aux(dry))
    b = mod._solve(_p0(), rainy, _cand().init_aux(rainy))
    for k in ("Pint", "Km", "Kh", "tau_A"):
        np.testing.assert_allclose(np.asarray(b[3][k]), np.asarray(a[3][k]), rtol=1e-12,
                                   atol=1e-300)
    np.testing.assert_allclose(np.asarray(b[2]["e"]), np.asarray(a[2]["e"]), rtol=1e-11,
                               atol=0.0)
    if profile == "layered":                           # control: rain was transported
        assert float(jnp.max(jnp.abs(b[0][1] - a[0][1]))) > 1e-9


@pytest.mark.parametrize("case", ["own", "jet"])
def test_tke_residual_is_exactly_the_patankar_shortfall(case):
    mod = _module()
    if case == "jet":
        inp, prm, aux = _jet_case()
    else:
        inp, prm = _column(), _p0()
        aux = _cand().init_aux(inp)
    _, _, aux1, d = mod._solve(prm, inp, aux)
    ref = jnp.sum(inp.mass * d["snk"] * (1.0 - d["e1"] / (aux["e"] + mod.E_EPS)))
    scale = jnp.sum(inp.mass * (d["snk"] + jnp.abs(d["e1"] - aux["e"]) / inp.dt))
    assert abs(float(aux1["tke_residual"] - ref)) <= 1e-12 * float(scale)
    assert bool(jnp.all(d["e1"] > 0.0))


def _roll(cand, prm, inp, steps):
    col, aux = inp, cand.init_aux(inp)
    for _ in range(steps):
        fa, fb, aux = cand.fluxes(prm, col, aux)
        t = cmod.candidate_tendencies("both", fa, fb, col, heat_form="enthalpy",
                                      ke_heating=True)
        thl, q = col.theta_l + col.dt * t.dtheta, col.q_t + col.dt * t.dqt
        theta, q_v, q_c = saturation_adjust(thl, q, col.exner, col.p_full)
        col = col._replace(theta_l=thl, q_t=q, temp=theta * col.exner, q_v=q_v, q_c=q_c,
                           u=col.u + col.dt * t.du, v=col.v + col.dt * t.dv)
    return col, aux


@pytest.mark.parametrize("law", [False, True])
def test_gradient_finite_and_nonzero_for_every_parameter(law):
    cand = _cand()
    inp = (_column(_UW, _VW, sfc_drag_most=jnp.nan, z0_m=jnp.nan) if law else _column())

    def loss(p):
        col, aux = _roll(cand, p, inp, 3)
        return (jnp.sum(col.theta_l ** 2) + 1e3 * jnp.sum(col.q_t ** 2)
                + jnp.sum(col.u ** 2 + col.v ** 2) + jnp.sum(aux["e"] ** 2))
    g = jax.grad(loss)(cand.default_params())
    bad = [k for k, v in g.items() if not (np.isfinite(float(v)) and float(v) != 0.0)]
    assert not bad, {k: float(v) for k, v in g.items()}


def test_gradient_through_the_counter_gradient_gate():
    """phi_R < 1 path (P_delta active): finite, non-zero parameter gradient."""
    mod = _module()
    inp, prm, aux = _jet_case()

    def loss(p):
        _, (uw, vw), a1, _ = mod._solve(p, inp, aux)
        return jnp.sum(uw ** 2 + vw ** 2) + jnp.sum(a1["e"] ** 2)
    g = jax.grad(loss)(prm)
    assert all(np.isfinite(float(v)) and float(v) != 0.0 for v in g.values()), g


def test_uniform_wind_has_no_interior_momentum_flux(monkeypatch):
    """Momentum operator is constant-preserving for ANY interface profile of phi_R
    (uniform winds alone give phi_R = 1, so a varying profile is injected)."""
    inp, prm = _deep_plume_case()
    inp = inp._replace(u=jnp.full((40,), 7.0), v=jnp.full((40,), -2.0),
                       uw_s=jnp.asarray(0.0), vw_s=jnp.asarray(0.0))
    mod = _module()
    monkeypatch.setattr(mod, "_realizability",
                        lambda e_h, p_nl, dt: 0.5 + 0.4 * jnp.sin(jnp.arange(e_h.shape[0])))
    _, (uw, vw), _, d = mod._solve(prm, inp, _cand().init_aux(inp))
    assert float(jnp.max(d["phi_R"]) - jnp.min(d["phi_R"])) > 0.5     # control: varies
    assert float(jnp.max(jnp.sum(d["Pint"], 1))) > 1e-3              # control
    assert float(jnp.max(jnp.abs(uw)) + jnp.max(jnp.abs(vw))) <= 1e-13


# --- CloudBench surface-law protocol (PI 2026-11-11) ---------------------------------

def _leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


def test_initial_wind_gradient_on_a_roughless_law_column():
    """Cold start inside the differentiated function: z0 = NaN on a law column must not
    leak NaN into d/d(initial winds) (codex r2)."""
    cand = _cand()
    base = _column(_UW, _VW, z0_m=jnp.nan, sfc_drag_most=jnp.nan)

    def loss(u0):
        inp = base._replace(u=base.u + u0)
        _, (uw, vw), a1 = cand.fluxes(cand.default_params(), inp, cand.init_aux(inp))
        return jnp.sum(uw[1:] ** 2) + jnp.sum(a1["e"] ** 2)
    g = jax.grad(loss)(jnp.asarray(0.0))
    assert np.isfinite(float(g)) and float(g) != 0.0


def test_uses_the_passed_stress():
    cand = _cand()
    inp = _column(_UW, _VW)
    _fa, (uw, vw), _ = cand.fluxes(cand.default_params(), inp, cand.init_aux(inp))
    assert (float(uw[0]), float(vw[0])) == (_UW, _VW)


def test_law_column_ignores_the_own_surface_layer(monkeypatch):
    """Roughness, wall-drag record, lagged u* and the stability function enter only the
    own law: on a law column nothing may depend on them."""
    cand = _cand()
    p0 = cand.default_params()
    inp_a = _column(_UW, _VW, z0_m=1e-3, sfc_drag_most=1.0)
    a = _leaves(cand.fluxes(p0, inp_a, cand.init_aux(inp_a)))
    inp_b = _column(_UW, _VW, z0_m=0.3, sfc_drag_most=jnp.nan)
    aux_b = dict(cand.init_aux(inp_b), u_star=jnp.asarray(7.0), h_p=jnp.asarray(900.0))
    import legoesm.core.bulk_flux as bf
    monkeypatch.setattr(bf, "psi_m", lambda zeta, *a_, **k: 0.0 * zeta + 0.3)
    cand_b = _cand()
    b = _leaves(cand_b.fluxes(p0, inp_b, aux_b))
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_nan_stress_keeps_the_own_law_bit_identically(monkeypatch):
    inp = _column()
    cand = _cand()
    got = _leaves(cand.fluxes(cand.default_params(), inp, cand.init_aux(inp)))
    _fa, (uw, _vw), _ = cand.fluxes(cand.default_params(), inp, cand.init_aux(inp))
    assert abs(float(uw[0]) - _UW) > 1e-3 * abs(_UW) and float(uw[0]) < 0.0
    monkeypatch.setattr(cmod, "surface_stress", lambda _x, uw, vw, us: (uw, vw, us))
    cand_p = _cand()
    ref = _leaves(cand_p.fluxes(cand_p.default_params(), inp, cand_p.init_aux(inp)))
    for x, y in zip(got, ref, strict=True):
        np.testing.assert_array_equal(x, y)
    # control: the patch bites (a passed stress is then no longer used)
    law = _column(_UW, _VW)
    _fa, (uw, _vw), _ = cand_p.fluxes(cand_p.default_params(), law, cand_p.init_aux(law))
    assert abs(float(uw[0]) - _UW) > 1e-3 * abs(_UW)


def test_law_column_has_no_implicit_drag():
    """The applied surface momentum flux equals the passed stress also inside the update
    (implicit-drag leak would show as a cell-0 momentum change beyond the faces)."""
    cand = _cand()
    inp = _column(_UW, _VW)
    d = _module()._solve(cand.default_params(), inp, cand.init_aux(inp))[3]
    fu = cmod.tendency_to_flux((d["U1"] - inp.u) / inp.dt, inp.rho_half, inp.mass)
    fv = cmod.tendency_to_flux((d["V1"] - inp.v) / inp.dt, inp.rho_half, inp.mass)
    np.testing.assert_allclose([float(fu[0]), float(fv[0])], [_UW, _VW], rtol=1e-9, atol=0.0)
