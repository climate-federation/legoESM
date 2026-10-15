"""Tests for the co-scientist experimenter harness (candidate plug-in + stages)."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.les_suite import candidate as cmod
from legoesm.atmosphere.les_suite import candidate_eval as ev
from legoesm.atmosphere.les_suite.bridge import LESReferenceArtifact
from legoesm.atmosphere.physics._shared import exner_function

from legoesm import constants

jax.config.update("jax_enable_x64", True)

_REPO = Path(__file__).resolve().parents[3]
_CANDS = _REPO / "scripts" / "validate" / "les_suite" / "cosci_candidates"
sys.path.insert(0, str(_REPO / "scripts" / "run"))
sys.path.insert(0, str(_REPO / "scripts" / "validate" / "les_suite"))

N = 12
_P0 = 1.0e5        # Pa, column surface pressure for the synthetic test column
_H = 8000.0        # m, isothermal scale height of the synthetic test column


def _column(w_theta_s=0.05, uw_s=-0.09, vw_s=0.02):
    z_half = jnp.linspace(0.0, 1200.0, N + 1)
    z_full = 0.5 * (z_half[1:] + z_half[:-1])
    rho_full = 1.2 * jnp.exp(-z_full / _H)
    p_half = _P0 * jnp.exp(-z_half / _H)
    p_full = _P0 * jnp.exp(-z_full / _H)
    return cmod.column_inputs(
        z_full=z_full, z_half=z_half, rho_full=rho_full,
        rho_half=cmod.half_level_average(rho_full, rho_full[0]),
        mass=(p_half[:-1] - p_half[1:]) / constants.g, exner=exner_function(p_full),
        exner_half=exner_function(p_half), p_full=p_full, p_half=p_half,
        theta_l=300.0 + 0.004 * z_full, q_t=0.01 - 2e-6 * z_full,
        u=5.0 + 0.004 * z_full, v=jnp.sin(z_full / 300.0),
        w_theta_s=w_theta_s, w_qt_s=1e-4, uw_s=uw_s, vw_s=vw_s, dt=10.0)


def _write(tmp_path, name, body):
    p = tmp_path / f"{name}.py"
    p.write_text(textwrap.dedent(body))
    return cmod.load_candidate(p)


_GOOD_SCALAR = """
    import jax.numpy as jnp
    KIND = "scalar"
    PARAMS = {"K": (10.0, 1.0, 100.0, "m2/s")}
    def init_aux(inputs):
        return ()
    def fluxes(params, inputs, aux):
        def f(phi, s):
            g = (phi[1:] - phi[:-1]) / (inputs.z_full[1:] - inputs.z_full[:-1])
            return jnp.concatenate([s[None], -params["K"] * g, jnp.zeros(1)])
        return f(inputs.theta_l, inputs.w_theta_s), f(inputs.q_t, inputs.w_qt_s), aux
"""

_GOOD_MOMENTUM = """
    import jax.numpy as jnp
    KIND = "momentum"
    PARAMS = {"K": (10.0, 1.0, 100.0, "m2/s")}
    def init_aux(inputs):
        return ()
    def fluxes(params, inputs, aux):
        def f(phi, s):
            g = (phi[1:] - phi[:-1]) / (inputs.z_full[1:] - inputs.z_full[:-1])
            return jnp.concatenate([s[None], -params["K"] * g, jnp.zeros(1)])
        return f(inputs.u, inputs.uw_s), f(inputs.v, inputs.vw_s), aux
"""


# --- conversions ------------------------------------------------------------------

def test_flux_tendency_round_trip_and_telescoping():
    inp = _column()
    flux = jnp.concatenate([jnp.asarray([0.05]), jnp.linspace(0.04, -0.01, N - 1),
                            jnp.zeros(1)])
    tend = cmod.flux_to_tendency(flux, inp.rho_half, inp.mass)
    resid = jnp.sum(inp.mass * tend) - inp.rho_half[0] * flux[0]
    assert abs(float(resid)) < 1e-14 * float(jnp.sum(jnp.abs(inp.mass * tend)))
    np.testing.assert_allclose(cmod.tendency_to_flux(tend, inp.rho_half, inp.mass), flux,
                               rtol=0, atol=1e-15)


def test_absorb_top_conserves_column_momentum_and_zeroes_lid():
    inp = _column()
    wave = jnp.full((N + 1,), -0.02)  # a wave flux that reaches the lid
    tapered = cmod.absorb_top(wave, inp.z_half, z_base=800.0)
    assert float(tapered[-1]) == 0.0 and float(tapered[0]) == -0.02
    tend = cmod.flux_to_tendency(tapered, inp.rho_half, inp.mass)
    assert abs(float(jnp.sum(inp.mass * tend) - inp.rho_half[0] * tapered[0])) < 1e-16
    assert float(cmod.absorbed_mass_flux(wave, inp.rho_half, inp.z_half, 800.0)) < 0.0
    assert bool(jnp.all(jnp.isnan(cmod.absorb_top(wave, inp.z_half, z_base=1200.0))))


def test_prognostic_baseline_names_are_roster_schemes():
    from tune_scm_to_les import _base_turbulence
    for name in ev.PROGNOSTIC_TURBULENCE:
        assert _base_turbulence(name)[0].scheme == name


@pytest.mark.parametrize("heating, expect_closed", [(True, True), (False, False)])
def test_ke_dissipation_heating_closes_total_energy(tmp_path, heating, expect_closed):
    cand = _write(tmp_path, "mom", _GOOD_MOMENTUM)
    r = ev.e0_column_checks(cand, cand.default_params(), _column(),
                            heat_form="enthalpy", ke_heating=heating)
    assert (r["energy_rel"] < 1e-12) is expect_closed


@pytest.mark.parametrize("arm, expect_closed", [
    ("full_midpoint", True),       # heating at the whole step's midpoint winds
    ("isolated_midpoint", False),  # ignores the other accelerations of the step
    ("instantaneous", False),      # misses the dt^2 KE term
])
def test_ke_heating_closes_the_forced_forward_euler_step(tmp_path, monkeypatch, arm,
                                                         expect_closed):
    if arm == "instantaneous":
        monkeypatch.setattr(cmod, "step_midpoint_winds",
                            lambda du, dv, inputs: (inputs.u, inputs.v))
    cand = _write(tmp_path, f"mom_step_{arm}", _GOOD_MOMENTUM)
    inp = _column()
    # other accelerations of the step: Coriolis toward a sheared geostrophic wind
    f_c, ug = 1e-4, 8.0 + 0.003 * inp.z_full
    other = (f_c * inp.v, -f_c * (inp.u - ug))
    fa, fb, _ = cand.fluxes({"K": jnp.asarray(80.0)}, inp, ())
    ct = cmod.candidate_tendencies(
        "momentum", fa, fb, inp, heat_form="enthalpy", ke_heating=True,
        other_accel=None if arm == "isolated_midpoint" else other)
    dt = inp.dt
    du, dv = ct.du + other[0], ct.dv + other[1]
    u1, v1 = inp.u + dt * du, inp.v + dt * dv
    d_ke = 0.5 * ((u1 ** 2 - inp.u ** 2) + (v1 ** 2 - inp.v ** 2))
    d_enth = constants.c_pd * inp.exner * dt * ct.dtheta
    stepped = jnp.sum(inp.mass * (d_ke + d_enth)) / dt
    um, vm = inp.u + 0.5 * dt * du, inp.v + 0.5 * dt * dv
    other_work = jnp.sum(inp.mass * (um * other[0] + vm * other[1]))
    surf = inp.rho_half[0] * (fa[0] * um[0] + fb[0] * vm[0])
    rel = abs(float(stepped - other_work - surf)) / abs(float(surf))
    assert (rel < 1e-10) is expect_closed, rel
    eb = cmod.column_energy_budget(ct, fa, fb, inp, "momentum", other_accel=other)
    if arm == "full_midpoint":
        assert abs(float(eb.residual)) < 1e-12 * abs(float(eb.boundary_input))


def test_heat_forms_conserve_their_own_variable(tmp_path):
    cand = _write(tmp_path, "sc", _GOOD_SCALAR)
    theta = ev.e0_column_checks(cand, cand.default_params(), _column(),
                                heat_form="theta", ke_heating=False)
    enth = ev.e0_column_checks(cand, cand.default_params(), _column(),
                               heat_form="enthalpy", ke_heating=False)
    assert theta["closure_rel"]["heat"] < 1e-13 and enth["closure_rel"]["heat"] < 1e-13
    assert enth["energy_rel"] < 1e-12
    assert theta["energy_rel"] > 1e-6   # θ transport does not conserve enthalpy


# --- E0 checks, each shown to fail ------------------------------------------------

def test_e0_good_candidate_passes(tmp_path):
    cand = _write(tmp_path, "good", _GOOD_SCALAR)
    r = ev.e0_column_checks(cand, cand.default_params(), _column(),
                            heat_form="theta", ke_heating=False)
    assert r["finite"] and r["surface_exact"] and r["top_zero"]
    assert max(r["closure_rel"].values()) < 1e-13 and r["jit_parity"] < 1e-12


@pytest.mark.parametrize("edit, key", [
    ('f(inputs.theta_l, 1.01 * inputs.w_theta_s)', "surface_exact"),
    ('f(inputs.theta_l, inputs.w_theta_s).at[-1].set(1e-3)', "top_zero"),
    ('f(inputs.theta_l, inputs.w_theta_s) * jnp.nan', "finite"),
])
def test_e0_flags_broken_candidates(tmp_path, edit, key):
    body = _GOOD_SCALAR.replace("f(inputs.theta_l, inputs.w_theta_s)", edit)
    cand = _write(tmp_path, f"bad_{key}", body)
    r = ev.e0_column_checks(cand, cand.default_params(), _column(),
                            heat_form="theta", ke_heating=False)
    assert r[key] is False


def test_e0_closure_check_is_not_vacuous(tmp_path, monkeypatch):
    cand = _write(tmp_path, "good2", _GOOD_SCALAR)
    real = ev.candidate_tendencies

    def leaky(*a, **k):
        t = real(*a, **k)
        return t._replace(dtheta=t.dtheta * 1.001)
    monkeypatch.setattr(ev, "candidate_tendencies", leaky)
    r = ev.e0_column_checks(cand, cand.default_params(), _column(),
                            heat_form="theta", ke_heating=False)
    assert r["closure_rel"]["heat"] > 1e-5


def test_gradient_activity_flags_inert_parameter(tmp_path):
    body = _GOOD_SCALAR.replace('PARAMS = {"K": (10.0, 1.0, 100.0, "m2/s")}',
                                'PARAMS = {"K": (10.0, 1.0, 100.0, "m2/s"), '
                                '"unused": (1.0, 0.0, 2.0, "1")}')
    cand = _write(tmp_path, "inert", body)
    inp = _column()

    def loss(p):
        fa, fb, _ = cand.fluxes(p, inp, ())
        return jnp.sum(fa ** 2) + jnp.sum(fb ** 2)
    r = ev.gradient_activity(loss, cand.default_params())
    assert r["inert"] == ["unused"] and not r["nonfinite"]


def test_load_candidate_rejects_bad_contract(tmp_path):
    with pytest.raises(ValueError, match="KIND"):
        _write(tmp_path, "badkind", _GOOD_SCALAR.replace('"scalar"', '"heat"'))
    with pytest.raises(ValueError, match="lo < hi"):
        _write(tmp_path, "badbounds",
               _GOOD_SCALAR.replace("(10.0, 1.0, 100.0", "(10.0, 20.0, 100.0"))


# --- SCM integration --------------------------------------------------------------

def _cbl_artifact(nz=20, nt=3, seconds=600.0):
    z = np.linspace(10.0, 1500.0, nz)
    th = np.where(z > 800.0, 300.0 + 0.006 * (z - 800.0), 300.0)
    times = np.linspace(0.0, seconds, nt)
    return LESReferenceArtifact(
        case_name="cbl_harness", sgs="lasd", heights_m=z, times_s=times,
        theta=np.broadcast_to(th, (nt, nz)).copy(), u=np.full((nt, nz), 2.0),
        v=np.zeros((nt, nz)), wtheta_resolved=np.zeros((nt, nz)),
        wtheta_sgs=np.zeros((nt, nz)), prescribe="fluxes", w_theta_s=np.full(nt, 0.06),
        w_theta_s_cum=0.06 * times, f_c=0.0,
        uw_resolved=np.zeros((nt, nz)), uw_sgs=np.zeros((nt, nz)),
        vw_resolved=np.zeros((nt, nz)), vw_sgs=np.zeros((nt, nz)))


def test_pure_step_with_physics_matches_pure_step():
    from legoesm.atmosphere.les_suite.scm_runner import build_cbl_scm_from_artifact
    from tune_scm_to_les import _base_turbulence
    scm, _ = build_cbl_scm_from_artifact(_cbl_artifact(), _base_turbulence("louis")[0],
                                         nlev=16, dt=10.0)
    a, _ = scm.pure_step(scm.state, scm.phys_state, 0.0)
    b, _ = scm.pure_step_with_physics(scm.state, scm.phys_state, 0.0, scm.physics_fn)
    np.testing.assert_array_equal(np.asarray(a.T.data), np.asarray(b.T.data))
    np.testing.assert_array_equal(np.asarray(a.u.data), np.asarray(b.u.data))


def _hb_rollouts(candidate, nsteps=30):
    from cosci_eval import Arm
    from tune_scm_to_les import _base_turbulence
    base = _base_turbulence("holtslag_boville")[0]
    art = _cbl_artifact()
    rb, _ = Arm(base).run(art, nsteps, nlev=16, dt=10.0)
    rc, _ = Arm(base, candidate, "theta", False).run(
        art, nsteps, params=candidate.default_params(), nlev=16, dt=10.0)
    return rb, rc


def test_hb_equivalent_candidate_reproduces_hb_to_round_off():
    cand = cmod.load_candidate(_CANDS / "hb_equivalent.py")
    rb, rc = _hb_rollouts(cand)
    dth = float(jnp.max(jnp.abs(rb.final_inputs.theta_l - rc.final_inputs.theta_l)))
    du = float(jnp.max(jnp.abs(rb.final_inputs.u - rc.final_inputs.u)))
    assert dth < 1e-10 and du < 1e-12
    # the column did evolve under HB (a no-op harness would also "match")
    assert float(jnp.max(jnp.abs(rb.final_inputs.theta_l - rb.at(0).theta_l))) > 1e-3


def _moist_artifact():
    import dataclasses
    art = _cbl_artifact()
    nt, nz = np.asarray(art.theta).shape
    z = np.asarray(art.heights_m)
    # saturated in the upper mixed layer, so condensation physics acts
    qt = np.broadcast_to(np.where(z > 800.0, 0.008, 0.020), (nt, nz)).copy()
    qc = np.broadcast_to(np.where((z > 500.0) & (z <= 800.0), 5e-4, 0.0), (nt, nz)).copy()
    return dataclasses.replace(
        art, case_name="moist_harness", qt=qt, qc=qc, wqt_resolved=np.zeros((nt, nz)),
        wqt_sgs=np.zeros((nt, nz)), w_qv_s=np.full(nt, 1.5e-4))


def test_hb_equivalent_reproduces_hb_on_a_moist_column():
    """Moist path: a cloudy column with condensation acting, so the owned-part
    subtraction is tend_full - tend_turb with a nonzero remainder in q_v."""
    from cosci_eval import Arm
    from tune_scm_to_les import _base_turbulence
    cand = cmod.load_candidate(_CANDS / "hb_equivalent.py")
    base = _base_turbulence("holtslag_boville")[0]
    art = _moist_artifact()
    rb, _ = Arm(base).run(art, ev.e2_nsteps(art, 10.0), nlev=16, dt=10.0)
    rc, _ = Arm(base, cand, "theta", None).run(
        art, ev.e2_nsteps(art, 10.0), params=cand.default_params(), nlev=16, dt=10.0)
    for k in ("theta_l", "q_t", "u"):
        d = float(jnp.max(jnp.abs(getattr(rb.final_inputs, k) - getattr(rc.final_inputs, k))))
        assert d < 1e-10, (k, d)
    assert float(jnp.max(rb.at(0).q_c)) > 1e-5                  # cloud present
    assert float(jnp.max(jnp.abs(rb.final_inputs.q_c - rb.at(0).q_c))) > 1e-7
    s = ev.e2_score(art, rc, 10.0)
    assert {"qt", "wqt"} <= set(s.per_var)
    # non-vacuous: a zero-interior-flux candidate must change the moist column
    zero = cmod.load_candidate(_CANDS / "hb_equivalent.py")._replace(
        fluxes=lambda p, inp, aux: (
            jnp.zeros_like(inp.z_half).at[0].set(inp.w_theta_s),
            jnp.zeros_like(inp.z_half).at[0].set(inp.w_qt_s), aux))
    rz, _ = Arm(base, zero, "theta", None).run(
        art, ev.e2_nsteps(art, 10.0), params=zero.default_params(), nlev=16, dt=10.0)
    assert float(jnp.max(jnp.abs(rz.final_inputs.q_t - rb.final_inputs.q_t))) > 1e-6


def test_hb_self_check_is_not_vacuous(tmp_path):
    zero = _write(tmp_path, "zero_scalar", """
        import jax.numpy as jnp
        KIND = "scalar"
        PARAMS = {}
        def init_aux(inputs):
            return ()
        def fluxes(params, inputs, aux):
            z = jnp.zeros_like(inputs.z_half)
            return z.at[0].set(inputs.w_theta_s), z.at[0].set(inputs.w_qt_s), aux
    """)
    rb, rc = _hb_rollouts(zero)
    assert float(jnp.max(jnp.abs(rb.final_inputs.theta_l - rc.final_inputs.theta_l))) > 1e-3


def test_init_aux_sees_the_prescribed_surface_flux(tmp_path):
    from cosci_eval import Arm
    from tune_scm_to_les import _base_turbulence
    cand = _write(tmp_path, "aux_probe", """
        import jax.numpy as jnp
        KIND = "scalar"
        PARAMS = {}
        def init_aux(inputs):
            return inputs.w_theta_s
        def fluxes(params, inputs, aux):
            z = jnp.zeros_like(inputs.z_half)
            return z.at[0].set(inputs.w_theta_s), z.at[0].set(inputs.w_qt_s), aux
    """)
    _, aux = Arm(_base_turbulence("louis")[0], cand, "theta", False).run(
        _cbl_artifact(), 1, params={}, nlev=16, dt=10.0)
    assert abs(float(aux) - 0.06) < 1e-12


def test_e2_score_finite_and_gradient_reaches_candidate_param(tmp_path):
    from cosci_eval import Arm
    from tune_scm_to_les import _base_turbulence
    cand = _write(tmp_path, "good3", _GOOD_SCALAR)
    art = _cbl_artifact()
    arm = Arm(_base_turbulence("louis")[0], cand, "theta", False)

    def loss(p):
        rec, _ = arm.run(art, ev.e2_nsteps(art, 10.0), params=p, nlev=16, dt=10.0)
        return ev.e2_score(art, rec, 10.0).combined
    val = loss(cand.default_params())
    g = jax.grad(loss)(cand.default_params())
    assert np.isfinite(float(val)) and float(abs(g["K"])) > 0.0
    from cosci_eval import candidate_case_loss
    art60 = _cbl_artifact(seconds=600.0)
    jl = candidate_case_loss(arm, art60, nlev=16, dt=10.0)
    assert abs(float(jl(cand.default_params())) - float(val)) < 1e-12
    gj = jax.grad(jl)(cand.default_params())
    assert abs(float(gj["K"]) - float(g["K"])) <= 1e-9 * abs(float(g["K"]))
    short, _ = arm.run(art, ev.e2_nsteps(art, 10.0) - 1, params=cand.default_params(),
                       nlev=16, dt=10.0)
    with pytest.raises(ValueError, match="e2_nsteps"):
        ev.e2_score(art, short, 10.0)


def _forced_artifact():
    import dataclasses
    art = _cbl_artifact()
    z = np.asarray(art.heights_m)
    nt = np.asarray(art.times_s).shape[0]
    return dataclasses.replace(art, f_c=1e-4, u_geo=5.0 + 0.004 * z,
                               v_geo=np.zeros_like(z),
                               # t=0 stress differs: the floor must use frames after t=0
                               tau_x_s=np.r_[-9.0, np.full(nt - 1, -0.09)],
                               tau_y_s=np.r_[12.0, np.full(nt - 1, 0.12)])


def test_frames_take_the_nearest_step():
    import dataclasses
    art = dataclasses.replace(_cbl_artifact(nt=3, seconds=1309.0),
                              times_s=np.array([0.0, 654.5, 1309.0]))
    assert ev._frames(art, 10.0) == [65, 131]   # 654.5 s -> 650 s, 1309 s -> 1310 s
    with pytest.raises(ValueError, match="collide"):
        ev._frames(dataclasses.replace(art, times_s=np.array([0.0, 11.0, 12.0])), 10.0)


def test_momentum_scored_only_with_geostrophic_wind_and_floor_is_5pct_ustar2():
    import dataclasses
    assert ev.momentum_flux_floor(_cbl_artifact()) is None         # f_c=0, no u_geo
    zero_geo = dataclasses.replace(_cbl_artifact(), u_geo=np.zeros(20))
    assert ev.momentum_flux_floor(zero_geo) is None
    forced = _forced_artifact()
    assert abs(ev.momentum_flux_floor(forced) - 0.05 * 0.15) < 1e-15   # |(-.09,.12)|=.15
    with pytest.raises(ValueError, match="momentum data"):
        ev.momentum_flux_floor(dataclasses.replace(forced, uw_resolved=None))
    with pytest.raises(ValueError, match="momentum data"):
        ev.momentum_flux_floor(dataclasses.replace(forced, tau_x_s=None))


@pytest.mark.parametrize("plant", [False, True])
def test_rollout_other_accel_is_what_the_step_applied(tmp_path, monkeypatch, plant):
    """The dissipation heating's midpoint uses the step's real non-candidate wind
    tendency: u(s+1) = u(s) + dt (candidate du + recorded other du), to round-off."""
    from cosci_eval import Arm
    from tune_scm_to_les import _base_turbulence
    if plant:  # drop the SCM forcing's u-acceleration from the recorded ones
        real = ev.compute_forcing_tendencies

        def planted(st, sc, f, t):
            ft = real(st, sc, f, t)
            return ft._replace(du_dt=ft.du_dt.replace(data=0.0 * ft.du_dt.data))
        monkeypatch.setattr(ev, "compute_forcing_tendencies", planted)
    cand = _write(tmp_path, f"mom_roll_{plant}", _GOOD_MOMENTUM)
    rec, _ = Arm(_base_turbulence("louis")[0], cand, None, True).run(
        _forced_artifact(), 5, params=cand.default_params(), nlev=16, dt=10.0)
    worst = 0.0
    for s in range(5):
        a, b = rec.at(s), rec.at(s + 1)
        fl = rec.fluxes_of_step(s)
        du_c = cmod.flux_to_tendency(fl.uw, a.rho_half, a.mass)
        dv_c = cmod.flux_to_tendency(fl.vw, a.rho_half, a.mass)
        other = jax.tree_util.tree_map(lambda x, s=s: x[s], rec.other_accel)
        worst = max(worst, float(jnp.max(jnp.abs(b.u - a.u - 10.0 * (du_c + other[0])))),
                    float(jnp.max(jnp.abs(b.v - a.v - 10.0 * (dv_c + other[1])))))
    assert (worst < 1e-12) is (not plant), worst


def test_candidate_refuses_a_prognostic_baseline(tmp_path):
    from cosci_eval import Arm
    from tune_scm_to_les import _base_turbulence
    cand = _write(tmp_path, "sc_prog", _GOOD_SCALAR)
    with pytest.raises(ValueError, match="prognostic"):
        Arm(_base_turbulence("tke")[0], cand, "theta", None).run(
            _cbl_artifact(), 1, params=cand.default_params(), nlev=16, dt=10.0)


def test_adam_descend_reduces_a_quadratic():
    lo, hi = jnp.asarray([0.0]), jnp.asarray([10.0])
    best, best_loss, d_loss, hist = ev.adam_descend_normalized(
        lambda x: jnp.sum((x - 3.0) ** 2), lo, hi, jnp.asarray([8.0]), steps=40, lr=0.1)
    assert best_loss < d_loss and abs(float(best[0]) - 3.0) < 0.5 and len(hist) == 41


# --- L-stage instrument + CLI -------------------------------------------------------

def test_instrument_known_answers():
    from cosci_eval import instrument_checks
    r = instrument_checks([])
    sl = r["neutral_sl_known"]
    assert abs(sl["kappa_m_median"] - sl["expected_kappa"]) < 1e-12
    assert abs(sl["Pr_t_median"] - sl["expected_Pr_t"]) < 1e-12
    for k in ("stable_z_less_known", "stable_z_squared_known"):
        assert abs(r[k]["K_m_ratio"] / r[k]["expected_ratio"] - 1.0) < 1e-12
    ek = r["ekman_analytic_known"]
    assert abs(ek["turning_angle_deg"] - ek["expected_turning_deg"]) < 1e-9
    assert abs(ek["h_stress_5pct_m"] / ek["expected_h_m"] - 1.0) < 0.02
    fc = r["free_convection_known"]
    assert fc["z_i"] == fc["expected_z_i"]
    assert abs(fc["entrainment_ratio"] - fc["expected_entrainment_ratio"]) < 1e-12
    assert abs(fc["lapse_mid_ml_K_per_km"] - fc["expected_lapse_K_per_km"]) < 1e-9
    assert abs(fc["w_star"] - fc["expected_w_star"]) < 1e-12


def test_free_convection_instrument_detects_a_planted_shift():
    z = jnp.arange(10.0, 2000.0, 20.0)
    q0 = 0.06
    flux = jnp.where(z <= 1010.0, q0 * (1.0 - 1.2 * z / 1010.0), 0.0)
    shifted = jnp.where(z <= 1210.0, q0 * (1.0 - 1.2 * z / 1210.0), 0.0)
    a = ev.free_convection_metrics(300.0 + 0 * z, z, flux, z, q0=q0, theta0=300.0)
    b = ev.free_convection_metrics(300.0 + 0 * z, z, shifted, z, q0=q0, theta0=300.0)
    assert a["z_i"] == 1010.0 and b["z_i"] == 1210.0


def test_frame_artifact_slices_by_name_not_by_length():
    from cosci_eval import _frame_artifact
    nz = nt = 4  # a height profile the same length as the time axis
    z = np.linspace(10.0, 400.0, nz)
    th = 300.0 + np.arange(nt)[:, None] + 0.0 * z[None, :]
    art = LESReferenceArtifact(
        case_name="frame", sgs="lasd", heights_m=z, times_s=np.linspace(0.0, 300.0, nt),
        theta=th, u=np.zeros((nt, nz)), v=np.zeros((nt, nz)),
        wtheta_resolved=np.zeros((nt, nz)), wtheta_sgs=np.zeros((nt, nz)),
        prescribe="fluxes", w_theta_s=np.array([0.01, 0.02, 0.03, 0.04]),
        w_theta_s_cum=np.array([0.0, 1.5, 4.5, 9.0]), f_c=1e-4,
        u_geo=np.array([1.0, 2.0, 3.0, 4.0]), v_geo=np.zeros(nz))
    fa = _frame_artifact(art, 2)
    np.testing.assert_array_equal(np.asarray(fa.u_geo), art.u_geo)      # untouched
    np.testing.assert_array_equal(np.asarray(fa.theta)[0], th[2])
    assert float(np.asarray(fa.w_theta_s)[0]) == 0.03
    np.testing.assert_allclose(np.asarray(fa.w_theta_s_cum), [0.0, 0.03 * 3600.0])


def test_cli_requires_explicit_choices():
    import cosci_eval
    cand = str(_CANDS / "hb_equivalent.py")
    with pytest.raises(SystemExit):
        cosci_eval.parse_args(["--stage", "E0", "--candidate", cand, "--out", "x.json"])
    with pytest.raises(SystemExit):
        cosci_eval.parse_args(["--stage", "E0", "--baseline-only", "louis",
                               "--out", "x.json"])
    cosci_eval.check_kind_choices("scalar", "theta", None)
    cosci_eval.check_kind_choices("momentum", None, "on")
    for kind, hf, ke in (("scalar", None, None), ("scalar", "theta", "off"),
                         ("momentum", None, None), ("momentum", "theta", "on")):
        with pytest.raises(SystemExit):
            cosci_eval.check_kind_choices(kind, hf, ke)


def test_e2_tuning_refuses_an_inert_parameter(tmp_path):
    import cosci_eval
    from legoesm.atmosphere.les_suite.bridge import save_artifact
    art_path = tmp_path / "cbl_harness.npz"
    save_artifact(_cbl_artifact(seconds=60.0), art_path)
    cand = _write(tmp_path, "inert_e2", _GOOD_SCALAR.replace(
        '"K": (10.0, 1.0, 100.0, "m2/s")}',
        '"K": (10.0, 1.0, 100.0, "m2/s"), "unused": (1.0, 0.0, 2.0, "1")}'))
    with pytest.raises(SystemExit, match="unused"):
        cosci_eval.main([
            "--stage", "E2", "--candidate", cand.source_path, "--baseline", "louis",
            "--heat-form", "theta",
            "--artifacts", str(art_path), "--nlev", "16", "--ad-steps", "1",
            "--out", str(tmp_path / "e2.json")])


def test_cli_e2_writes_provenanced_json(tmp_path):
    import cosci_eval
    from legoesm.atmosphere.les_suite.bridge import save_artifact
    art_path = tmp_path / "cbl_harness.npz"
    save_artifact(_cbl_artifact(), art_path)
    out = tmp_path / "e2.json"
    cand = _write(tmp_path, "good4", _GOOD_SCALAR)
    rc = cosci_eval.main([
        "--stage", "E2", "--candidate", cand.source_path, "--baseline", "louis",
        "--heat-form", "theta",
        "--artifacts", str(art_path), "--baseline-only", "louis", "--no-tune",
        "--nlev", "16", "--out", str(out)])
    assert rc == 0
    rec = json.loads(out.read_text())
    assert rec["provenance"]["heat_form"] == "theta"
    assert rec["provenance"]["l_stage_constants"]["_L_CBL_Q0"] == 0.06
    assert rec["provenance"]["baseline_only"] == ["louis"]
    assert rec["provenance"]["candidate"]["sha256"]
    scores = rec["result"]["arms"]["candidate:good4"]["default"]["cbl_harness.npz"]
    # no geostrophic wind in this case -> momentum fluxes not scored
    assert np.isfinite(scores["combined"]) and "uw" not in scores["per_var"]
