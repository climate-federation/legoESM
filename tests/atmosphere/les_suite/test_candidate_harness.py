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
        assert _base_turbulence(name, **_RGH)[0].scheme == name


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

# the roughness every test column records (the SCM is built with the case's own)
_RGH = dict(z0_m=0.1, z0h_m=0.1)


def _cbl_artifact(nz=20, nt=3, seconds=600.0):
    z = np.linspace(10.0, 1500.0, nz)
    th = np.where(z > 800.0, 300.0 + 0.006 * (z - 800.0), 300.0)
    times = np.linspace(0.0, seconds, nt)
    return LESReferenceArtifact(
        case_name="cbl_nieuwstadt", sgs="lasd", heights_m=z, times_s=times,
        theta=np.broadcast_to(th, (nt, nz)).copy(), u=np.full((nt, nz), 2.0),
        v=np.zeros((nt, nz)), wtheta_resolved=np.zeros((nt, nz)),
        wtheta_sgs=np.zeros((nt, nz)), prescribe="fluxes", w_theta_s=np.full(nt, 0.06),
        w_theta_s_cum=0.06 * times, f_c=0.0,
        uw_resolved=np.zeros((nt, nz)), uw_sgs=np.zeros((nt, nz)),
        vw_resolved=np.zeros((nt, nz)), vw_sgs=np.zeros((nt, nz)),
        les_setup={"z0_m": _RGH["z0_m"], "sfc_flux_drag": "neutral"})


def test_pure_step_with_physics_matches_pure_step():
    from legoesm.atmosphere.les_suite.scm_runner import build_cbl_scm_from_artifact
    from tune_scm_to_les import _base_turbulence
    scm, _ = build_cbl_scm_from_artifact(_cbl_artifact(), _base_turbulence("louis", **_RGH)[0],
                                         nlev=16, dt=10.0)
    a, _ = scm.pure_step(scm.state, scm.phys_state, 0.0)
    b, _ = scm.pure_step_with_physics(scm.state, scm.phys_state, 0.0, scm.physics_fn)
    np.testing.assert_array_equal(np.asarray(a.T.data), np.asarray(b.T.data))
    np.testing.assert_array_equal(np.asarray(a.u.data), np.asarray(b.u.data))


def _hb_rollouts(candidate, nsteps=30):
    import dataclasses

    from cosci_eval import Arm
    from tune_scm_to_les import _base_turbulence
    base = _base_turbulence("holtslag_boville", **_RGH)[0]
    # the case's roughness differs from the base's, with z0h != z0, so the
    # candidate must take it from the column inputs
    art = dataclasses.replace(_cbl_artifact(), les_setup={
        "z0_m": 0.05, "z0h_m": 0.005, "sfc_flux_drag": "neutral"})
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
    from legoesm.atmosphere.les_suite.scm_runner import microphysics_record
    from legoesm.atmosphere.physics import MicrophysicsConfig
    return dataclasses.replace(
        art, case_name="cbl_nieuwstadt", qt=qt, qc=qc, wqt_resolved=np.zeros((nt, nz)),
        wqt_sgs=np.zeros((nt, nz)), w_qv_s=np.full(nt, 1.5e-4),
        les_setup=dict(art.les_setup, microphysics=microphysics_record(
            MicrophysicsConfig(scheme="kessler"))))


def test_harness_scores_moist_cases_with_recorded_kessler_only():
    """User 2026-10-19: a moist reference without a microphysics record, or with a
    scheme other than Kessler, is refused rather than run under a fallback."""
    import dataclasses

    from cosci_eval import Arm
    from legoesm.atmosphere.les_suite.scm_runner import microphysics_record
    from legoesm.atmosphere.physics import MicrophysicsConfig
    from tune_scm_to_les import _base_turbulence
    arm = Arm(_base_turbulence("louis", **_RGH)[0])
    art = _moist_artifact()
    no_record = dataclasses.replace(art, les_setup={"z0_m": _RGH["z0_m"],
                                                    "sfc_flux_drag": "neutral"})
    with pytest.raises(ValueError, match="records no LES microphysics"):
        arm.prepare(no_record, nlev=16, dt=10.0)
    sund = dataclasses.replace(art, les_setup=dict(
        art.les_setup, microphysics=microphysics_record(
            MicrophysicsConfig(scheme="sundqvist"))))
    with pytest.raises(ValueError, match="Kessler only"):
        arm.prepare(sund, nlev=16, dt=10.0)
    assert arm.prepare(art, nlev=16, dt=10.0)[1].microphysics.scheme == "kessler"


def test_hb_equivalent_reproduces_hb_on_a_moist_column():
    """Moist path: a cloudy column with condensation acting, so the owned-part
    subtraction is tend_full - tend_turb with a nonzero remainder in q_v."""
    from cosci_eval import Arm
    from tune_scm_to_les import _base_turbulence
    cand = cmod.load_candidate(_CANDS / "hb_equivalent.py")
    base = _base_turbulence("holtslag_boville", **_RGH)[0]
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
    _, aux = Arm(_base_turbulence("louis", **_RGH)[0], cand, "theta", False).run(
        _cbl_artifact(), 1, params={}, nlev=16, dt=10.0)
    assert abs(float(aux) - 0.06) < 1e-12


def test_e2_score_finite_and_gradient_reaches_candidate_param(tmp_path):
    from cosci_eval import Arm
    from tune_scm_to_les import _base_turbulence
    cand = _write(tmp_path, "good3", _GOOD_SCALAR)
    art = _cbl_artifact()
    arm = Arm(_base_turbulence("louis", **_RGH)[0], cand, "theta", False)

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


@pytest.mark.parametrize("kind", ["momentum", "both"])
@pytest.mark.parametrize("plant", [False, True])
def test_rollout_other_accel_is_what_the_step_applied(tmp_path, monkeypatch, plant, kind):
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
    body, hf = (_GOOD_MOMENTUM, None) if kind == "momentum" else (_GOOD_BOTH, "theta")
    cand = _write(tmp_path, f"{kind}_roll_{plant}", body)
    rec, _ = Arm(_base_turbulence("louis", **_RGH)[0], cand, hf, True).run(
        _forced_artifact(), 5, params=cand.default_params(), nlev=16, dt=10.0)
    assert rec.other_accel is not None
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
        Arm(_base_turbulence("tke", **_RGH)[0], cand, "theta", None).run(
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


def test_e2_tuning_freezes_an_inactive_parameter_and_tunes_the_rest(tmp_path):
    import cosci_eval
    from legoesm.atmosphere.les_suite.bridge import save_artifact
    art_path = tmp_path / "cbl_harness.npz"
    save_artifact(_cbl_artifact(seconds=60.0), art_path)
    cand = _write(tmp_path, "inert_e2", _GOOD_SCALAR.replace(
        '"K": (10.0, 1.0, 100.0, "m2/s")}',
        '"K": (10.0, 1.0, 100.0, "m2/s"), "unused": (1.0, 0.0, 2.0, "1")}'))
    out = tmp_path / "e2.json"
    cosci_eval.main([
        "--stage", "E2", "--candidate", cand.source_path, "--baseline", "louis",
        "--heat-form", "theta", "--artifacts", str(art_path), "--nlev", "16",
        "--ad-steps", "2", "--out", str(out)])
    ad = json.loads(out.read_text())["result"]["arms"]["candidate:inert_e2"]["ad"]
    assert ad["frozen_inactive"] == ["unused"] and ad["params"]["unused"] == 1.0
    assert ad["init_grad"]["K"] != 0.0 and len(ad["history"]) == 3


def test_tune_active_moves_active_keeps_frozen_and_flags_all_inactive():
    import cosci_eval
    lo, hi = jnp.zeros(2), jnp.full(2, 10.0)
    x0 = jnp.asarray([8.0, 4.0])
    best, rec = cosci_eval.tune_active(lambda x: (x[0] - 3.0) ** 2 + 0.0 * x[1], lo, hi,
                                       x0, ["a", "b"], steps=40, lr=0.1)
    assert rec["frozen_inactive"] == ["b"] and float(best[1]) == 4.0
    assert abs(float(best[0]) - 3.0) < 0.5
    assert rec["train_loss_best"] < rec["train_loss_default"]
    same, rec2 = cosci_eval.tune_active(lambda x: 0.0 * jnp.sum(x), lo, hi, x0,
                                        ["a", "b"], steps=5, lr=0.1)
    assert "not_tuned" in rec2 and bool(jnp.all(same == x0))


def test_baseline_loss_compiles_once_and_matches_eager(monkeypatch):
    import cosci_eval
    from tune_scm_to_les import (
        _base_turbulence,
        _scheme_tunable_metas,
        apply_overrides_to_base,
    )
    base, key = _base_turbulence("louis", **_RGH)
    fields = [m.field for m in _scheme_tunable_metas(key, (1,))]
    x0 = jnp.asarray([float(getattr(base.louis, f)) for f in fields])
    art = _cbl_artifact(seconds=60.0)
    train = [("cbl_harness.npz", art)]
    traces = []
    real_run = cosci_eval.Arm.run
    real_rwt = cosci_eval.Arm.run_with_turb

    def counting_rwt(self, *a, **k):
        traces.append(1)  # Python side effect: runs only while tracing
        return real_rwt(self, *a, **k)
    monkeypatch.setattr(cosci_eval.Arm, "run_with_turb", counting_rwt)
    arm = cosci_eval.Arm(base)
    jl = cosci_eval.baseline_loss(arm, base, fields, train, nlev=16, dt=10.0)
    xp = x0 * jnp.asarray([1.0, 0.5])            # perturb l_mix_max (active here)
    v0 = jl(x0)
    g = jax.grad(jl)(xp)
    vp = jl(xp)
    assert len(traces) <= 2   # one forward trace (+ one for the grad), not per call
    assert float(jnp.max(jnp.abs(g))) > 0.0
    for x, v in ((x0, v0), (xp, vp)):  # equals a FULL rebuild at that x
        turb = apply_overrides_to_base(base, {f: x[i] for i, f in enumerate(fields)})
        rec, _ = real_run(arm, art, ev.e2_nsteps(art, 10.0), nlev=16, dt=10.0, turb=turb)
        eager = float(ev.e2_score(art, rec, 10.0).combined)
        # jit reassociates float64 sums (measured 4e-11 relative), so parity is 1e-9
        assert abs(float(v) - eager) < 1e-9 * abs(eager)
    assert abs(float(vp) - float(v0)) > 1e-6 * abs(float(v0))  # the perturbation acts


def test_tune_active_refuses_a_non_finite_gradient():
    import cosci_eval
    with pytest.raises(SystemExit, match="non-finite"):
        cosci_eval.tune_active(lambda x: jnp.sqrt(x[0] - 1.0) + x[1], jnp.zeros(2),
                               jnp.full(2, 2.0), jnp.asarray([1.0, 1.0]), ["a", "b"],
                               steps=1, lr=0.1)


def test_les_tuned_arm_records_inapplicable_clubb_instead_of_crashing():
    import cosci_eval
    from tune_scm_to_les import _base_turbulence
    tuned, why = cosci_eval.les_tuned_arm(_base_turbulence("clubb", **_RGH)[0])
    assert tuned is None and "PROGNOSTIC" in why
    tuned, why = cosci_eval.les_tuned_arm(_base_turbulence("tke", **_RGH)[0])
    assert why is None and tuned is not None


def test_les_tuned_arm_propagates_a_malformed_tuned_file(monkeypatch):
    import cosci_eval

    def bad(base):
        raise ValueError("les_tuned: unknown field in tuned YAML")
    monkeypatch.setattr(cosci_eval, "apply_les_tuned_turbulence", bad)
    with pytest.raises(ValueError, match="unknown field"):
        cosci_eval.les_tuned_arm(None)


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


# --- merge with the GABLS3 lineage (user 2026-10-19) --------------------------------
def _geo_cum_artifact(nt=4, seconds=600.0):
    """A CBL column under a geostrophic wind that grows in time (u_g = 2 + 0.01 t),
    recorded as running integrals the way GABLS3 records its forcing."""
    import dataclasses
    art = _cbl_artifact(nt=nt, seconds=seconds)
    t = np.asarray(art.times_s)
    nz = np.asarray(art.heights_m).size
    ug_cum = (2.0 * t + 0.005 * t ** 2)[:, None] * np.ones(nz)
    return dataclasses.replace(art, f_c=1.0e-4, u_geo_cum=ug_cum,
                               v_geo_cum=np.zeros((nt, nz)),
                               tau_x_s=np.full(nt, -0.05), tau_y_s=np.zeros(nt))


def test_time_varying_geostrophic_wind_is_scored_and_seen_by_the_closure():
    from cosci_eval import Arm
    from tune_scm_to_les import _base_turbulence
    art = _geo_cum_artifact()
    assert ev.momentum_flux_floor(art) is not None     # u'w', v'w' are scored
    arm = Arm(_base_turbulence("louis", **_RGH)[0])
    prep = arm.prepare(art, nlev=16, dt=10.0)
    scm = prep[0]
    rec, _ = arm.run_prepared(prep, 30)
    for k in (0, 29):   # the closure's u_g is the one the SCM applies at that step
        want = np.asarray(scm.forcing_arg.u_geo(scm.t_seconds + 10.0 * k))[::-1]
        np.testing.assert_allclose(np.asarray(rec.inputs.u_geo[k]), want, rtol=1e-12)
    # interval means 3 m/s over [0, 200 s] and 5 m/s over [200, 400 s]
    assert float(rec.inputs.u_geo[29][0]) - float(rec.inputs.u_geo[0][0]) > 1.9
    # the closure also sees the case's roughness
    assert float(rec.inputs.z0_m[0]) == 0.1


def test_e1_frame_keeps_the_forcing_rate_applied_after_that_frame():
    import cosci_eval
    art = _geo_cum_artifact()
    t, c = np.asarray(art.times_s), np.asarray(art.u_geo_cum)
    for k, (a, b) in ((1, (1, 2)), (3, (2, 3))):        # the last frame: interval before
        fa = cosci_eval._frame_artifact(art, k)
        rate = (c[b] - c[a]) / (t[b] - t[a])
        np.testing.assert_allclose(np.asarray(fa.u_geo_cum)[1] / 3600.0, rate,
                                   rtol=1e-12)
        assert float(np.min(rate)) > 2.0
        assert np.asarray(fa.u_geo_cum)[0].max() == 0.0


def test_e1_frame_classifies_the_moist_surface_series():
    import dataclasses

    import cosci_eval
    art = _moist_artifact()
    nt = np.asarray(art.times_s).size
    art = dataclasses.replace(art, w_qv_s_cum=1.5e-4 * np.asarray(art.times_s),
                              precip_s=np.full(nt, 2e-5),
                              precip_s_cum=2e-5 * np.asarray(art.times_s))
    fa = cosci_eval._frame_artifact(art, 1)
    np.testing.assert_allclose(np.asarray(fa.w_qv_s_cum), [0.0, 1.5e-4 * 3600.0])
    np.testing.assert_allclose(np.asarray(fa.precip_s_cum), [0.0, 2e-5 * 3600.0])


def test_every_run_uses_the_les_cases_roughness():
    import dataclasses

    from cosci_eval import Arm
    from tune_scm_to_les import _base_turbulence
    art = dataclasses.replace(_cbl_artifact(),
                              les_setup={"z0_m": 0.05, "z0h_m": 0.005,
                                         "sfc_flux_drag": "neutral"})
    arm = Arm(_base_turbulence("louis", **_RGH)[0])       # built with z0 = 0.1
    prep = arm.prepare(art, nlev=16, dt=10.0)
    surf = prep[1].turbulence.louis.surface
    assert surf.z0 == 0.05 and surf.z0h_z0_ratio == pytest.approx(0.1, rel=1e-14)
    import cosci_eval
    seen = {}

    def _spy(col, cfg, nsteps, **kw):
        seen["surface"] = cfg.turbulence.louis.surface
        return "ran"
    orig = cosci_eval.rollout
    cosci_eval.rollout = _spy
    try:
        assert arm.run_with_turb(prep, art, 1, arm.base_turb) == "ran"
    finally:
        cosci_eval.rollout = orig
    assert seen["surface"].z0 == 0.05
    assert seen["surface"].z0h_z0_ratio == pytest.approx(0.1, rel=1e-14)


def test_total_water_includes_rain():
    from legoesm.atmosphere.les_suite.scm_runner import build_cbl_scm_from_artifact
    from legoesm.atmosphere.physics import MicrophysicsConfig
    from tune_scm_to_les import _base_turbulence
    scm, _ = build_cbl_scm_from_artifact(
        _moist_artifact(), _base_turbulence("louis", **_RGH)[0], nlev=16, dt=10.0,
        microphysics=MicrophysicsConfig(scheme="kessler"))
    tr = dict(scm.state.tracers)
    tr["q_r"] = tr["q_r"].replace(data=tr["q_r"].data + 1.0e-4)
    st = scm.state._replace(tracers=tr)
    zero = jnp.zeros(())
    inp = ev.scm_column_inputs(st, scm.sigma_coord, scm.phys_state, f_c=0.0,
                               u_geo=None, v_geo=None, dt=10.0, uw_s=zero, vw_s=zero)

    def col(name):
        return np.asarray(tr[name].data)[0, 0, 0][::-1]
    np.testing.assert_allclose(np.asarray(inp.q_t),
                               col("q_v") + col("q_c") + col("q_r"), rtol=1e-14)


def test_e1_and_e2_ignore_the_layer_above_the_case_sponge_base():
    """User 2026-10-19: every case is scored only below its catalog sponge base.
    Spoiling the reference above it (frames after the start, so the column is
    unchanged) must not move the E1 or E2 score."""
    import dataclasses

    import cosci_eval
    from cosci_eval import Arm
    from legoesm.atmosphere.les_suite.score import sponge_base_for_score
    from tune_scm_to_les import _base_turbulence
    art = dataclasses.replace(_cbl_artifact(), case_name="gabls3")
    z = np.asarray(art.heights_m)
    above = z > sponge_base_for_score("gabls3", z)
    assert 0 < above.sum() < z.size

    wt = np.asarray(art.wtheta_resolved).copy(); wt[1:, above] += 5.0
    th = np.asarray(art.theta).copy(); th[1:, above] += 30.0
    flux_spoiled = dataclasses.replace(art, wtheta_resolved=wt)
    spoiled = dataclasses.replace(flux_spoiled, theta=th)
    arm = Arm(_base_turbulence("louis", **_RGH)[0])
    rec, _ = arm.run(art, ev.e2_nsteps(art, 10.0), nlev=16, dt=10.0)
    a, b = ev.e2_score(art, rec, 10.0), ev.e2_score(spoiled, rec, 10.0)
    assert float(a.combined) == pytest.approx(float(b.combined), rel=1e-12)
    e1 = cosci_eval.stage_e1({"louis": arm}, [("a.npz", art)], 16, 10.0)
    e1b = cosci_eval.stage_e1({"louis": arm}, [("a.npz", flux_spoiled)], 16, 10.0)
    assert e1["a.npz"]["louis"]["wtheta"] == pytest.approx(
        e1b["a.npz"]["louis"]["wtheta"], rel=1e-12)


# --- "both" candidates: scalars and momentum, own surface stress -----------------

_GOOD_BOTH = """
    import jax.numpy as jnp
    KIND = "both"
    PARAMS = {"K": (10.0, 1.0, 100.0, "m2/s"), "C_d": (2e-3, 1e-4, 1e-2, "1")}
    def init_aux(inputs):
        return ()
    def fluxes(params, inputs, aux):
        def f(phi, s):
            g = (phi[1:] - phi[:-1]) / (inputs.z_full[1:] - inputs.z_full[:-1])
            return jnp.concatenate([jnp.reshape(s, (1,)), -params["K"] * g, jnp.zeros(1)])
        speed = jnp.sqrt(inputs.u[0] ** 2 + inputs.v[0] ** 2 + 1e-12)
        tau = -params["C_d"] * speed
        return ((f(inputs.theta_l, inputs.w_theta_s), f(inputs.q_t, inputs.w_qt_s)),
                (f(inputs.u, tau * inputs.u[0]), f(inputs.v, tau * inputs.v[0])), aux)
"""


def test_both_tendencies_are_the_scalar_and_momentum_kinds_summed(tmp_path):
    both = _write(tmp_path, "both_sum", _GOOD_BOTH)
    inp = _column()
    sa, mo = both.fluxes(both.default_params(), inp, ())[:2]
    ct = cmod.candidate_tendencies("both", sa, mo, inp, heat_form="enthalpy",
                                   ke_heating=True)
    cs = cmod.candidate_tendencies("scalar", *sa, inp, heat_form="enthalpy",
                                   ke_heating=None)
    cm = cmod.candidate_tendencies("momentum", *mo, inp, heat_form=None, ke_heating=True)
    np.testing.assert_array_equal(np.asarray(ct.dtheta), np.asarray(cs.dtheta + cm.dtheta))
    np.testing.assert_array_equal(np.asarray(ct.du), np.asarray(cm.du))
    np.testing.assert_array_equal(np.asarray(ct.dqt), np.asarray(cs.dqt))
    with pytest.raises(ValueError, match="both"):
        cmod.owned_fluxes("both", sa[0], mo[0])


@pytest.mark.parametrize("heating, expect_closed", [(True, True), (False, False)])
def test_both_e0_passes_and_total_energy_needs_dissipation_heating(tmp_path, heating,
                                                                   expect_closed):
    both = _write(tmp_path, "both_e0", _GOOD_BOTH)
    r = ev.e0_column_checks(both, both.default_params(), _column(),
                            heat_form="enthalpy", ke_heating=heating)
    assert r["finite"] and r["surface_exact"] and r["top_zero"]
    assert set(r["closure_rel"]) == {"heat", "qt", "u", "v"}
    assert max(r["closure_rel"].values()) < 1e-13 and r["jit_parity"] < 1e-12
    assert r["surface_stress_power"] < 0.0          # drag removes kinetic energy
    assert (r["energy_rel"] < 1e-12) is expect_closed, r["energy_rel"]


def test_both_e0_checks_scalar_surface_and_closure_are_not_vacuous(tmp_path, monkeypatch):
    bad = _GOOD_BOTH.replace("f(inputs.theta_l, inputs.w_theta_s)",
                             "f(inputs.theta_l, 1.01 * inputs.w_theta_s)")
    cand = _write(tmp_path, "both_badsfc", bad)
    r = ev.e0_column_checks(cand, cand.default_params(), _column(), heat_form="theta",
                            ke_heating=True)
    assert r["surface_exact"] is False
    both = _write(tmp_path, "both_leak", _GOOD_BOTH)
    real = ev.candidate_tendencies

    def leaky(*a, **k):
        t = real(*a, **k)
        return t._replace(dtheta=t.dtheta * 1.001)
    monkeypatch.setattr(ev, "candidate_tendencies", leaky)
    r = ev.e0_column_checks(both, both.default_params(), _column(), heat_form="theta",
                            ke_heating=True)
    assert r["closure_rel"]["heat"] > 1e-5


def test_both_e0_heat_closure_sees_a_composition_defect(tmp_path, monkeypatch):
    """A defect only in how "both" adds dissipation heating to the scalar heating
    (here: heating applied twice) must show in the heat closure, which is checked on
    the tendencies the candidate actually applies."""
    both = _write(tmp_path, "both_twice", _GOOD_BOTH)
    real = cmod.ke_dissipation_heating
    monkeypatch.setattr(cmod, "ke_dissipation_heating", lambda *a: 2.0 * real(*a))
    r = ev.e0_column_checks(both, both.default_params(), _column(), heat_form="enthalpy",
                            ke_heating=True)
    assert r["closure_rel"]["heat"] > 1e-8, r["closure_rel"]["heat"]


def test_both_e0_hides_the_baseline_surface_stress(tmp_path):
    peek = _GOOD_BOTH.replace("tau * inputs.u[0]", "inputs.uw_s")
    cand = _write(tmp_path, "both_peek", peek)
    r = ev.e0_column_checks(cand, cand.default_params(), _column(), heat_form="theta",
                            ke_heating=True)
    assert r["finite"] is False   # a candidate reading the absent stress gets NaN


def _rollout_final(cand, base_name, heat_form, ke_heating, nsteps=20):
    from cosci_eval import Arm
    from tune_scm_to_les import _base_turbulence
    arm = Arm(_base_turbulence(base_name, **_RGH)[0], cand, heat_form, ke_heating)
    rec, _ = arm.run(_forced_artifact(), nsteps, params=cand.default_params(), nlev=16,
                     dt=10.0)
    return rec


def test_both_rollout_does_not_depend_on_the_baseline(tmp_path):
    both = _write(tmp_path, "both_roll", _GOOD_BOTH)
    a = _rollout_final(both, "louis", "theta", True)
    b = _rollout_final(both, "holtslag_boville", "theta", True)
    for f in ("theta_l", "u", "v"):
        d = float(jnp.max(jnp.abs(getattr(a.final_inputs, f) - getattr(b.final_inputs, f))))
        assert d < 1e-12, (f, d)
    # the column evolved, and its surface stress is the candidate's own drag
    assert float(jnp.max(jnp.abs(a.final_inputs.u - a.at(0).u))) > 1e-3
    assert bool(jnp.all(jnp.isnan(a.inputs.uw_s)))
    col, fl = a.at(3), a.fluxes_of_step(3)
    speed = jnp.sqrt(col.u[0] ** 2 + col.v[0] ** 2 + 1e-12)
    np.testing.assert_allclose(float(fl.uw[0]), float(-2e-3 * speed * col.u[0]),
                               rtol=1e-13)
    # control: a momentum-only candidate keeps each baseline's scalar transport, so
    # the same comparison separates the two baselines
    mom = _write(tmp_path, "mom_ctrl", _GOOD_MOMENTUM)
    ma = _rollout_final(mom, "louis", None, True)
    mb = _rollout_final(mom, "holtslag_boville", None, True)
    assert float(jnp.max(jnp.abs(ma.final_inputs.theta_l - mb.final_inputs.theta_l))) > 1e-6


def test_both_rollout_on_a_cloudy_column_does_not_depend_on_the_baseline(tmp_path):
    """Moist path: condensation acts, so tend_full - tend_turb keeps a nonzero q_v
    remainder that the candidate's moisture transport must replace exactly."""
    from cosci_eval import Arm
    from tune_scm_to_les import _base_turbulence
    both = _write(tmp_path, "both_moist", _GOOD_BOTH)
    art = _moist_artifact()
    n = ev.e2_nsteps(art, 10.0)
    recs = [Arm(_base_turbulence(b, **_RGH)[0], both, "enthalpy", True).run(
        art, n, params=both.default_params(), nlev=16, dt=10.0)[0]
        for b in ("louis", "holtslag_boville")]
    for k in ("theta_l", "q_t", "u"):
        d = float(jnp.max(jnp.abs(getattr(recs[0].final_inputs, k)
                                  - getattr(recs[1].final_inputs, k))))
        assert d < 1e-12, (k, d)
    assert float(jnp.max(recs[0].at(0).q_c)) > 1e-5               # cloud present
    np.testing.assert_array_equal(np.asarray(recs[0].fluxes.wqt[:, 0]),
                                  np.asarray(recs[0].inputs.w_qt_s))
    assert float(jnp.max(jnp.abs(recs[0].final_inputs.q_t - recs[0].at(0).q_t))) > 1e-6


@pytest.mark.parametrize("plant", [False, True])
def test_both_owned_tendencies_keep_the_non_turbulent_remainder(monkeypatch, plant):
    """On a cloudy column the baseline total minus its turbulence leaves a nonzero
    microphysics remainder in T and q_v; a "both" candidate's tendency is added to
    exactly that remainder (planted: the q_v remainder dropped)."""
    from cosci_eval import Arm
    from legoesm.atmosphere.physics import MicrophysicsConfig
    from legoesm.atmosphere.physics.combined import make_physics
    from tune_scm_to_les import _base_turbulence
    scm, phys, *_ = Arm(_base_turbulence("holtslag_boville", **_RGH)[0]).prepare(
        _moist_artifact(), nlev=16, dt=10.0)
    args = (scm.state, scm.grid, scm.sigma_coord)
    full, _ = scm.physics_fn(*args, phys_state=scm.phys_state)
    turb_fn = make_physics(phys._replace(microphysics=MicrophysicsConfig(scheme="none")),
                           model_type="hydrostatic", dt=scm.dt)
    turb, _ = turb_fn(*args, phys_state=scm.phys_state)
    inp = ev.scm_column_inputs(scm.state, scm.sigma_coord, scm.phys_state, f_c=0.0,
                               u_geo=None, v_geo=None, dt=10.0, uw_s=0.0, vw_s=0.0)
    n = inp.mass.shape[0]
    r = jnp.linspace(-1.0, 1.0, n)
    cand = cmod.CandidateTendencies(dtheta=1e-4 * r, dqt=1e-7 * r, du=1e-3 * r,
                                    dv=-1e-3 * r)
    if plant:
        real = ev._swap
        monkeypatch.setattr(ev, "_swap", lambda f, new: real(
            f, cand.dqt if f is full.tracer_tendencies["q_v"] else new))
    out = ev._owned_tendencies(full, turb, cand, inp, "both")
    col = ev._col
    q_rem = col(full.tracer_tendencies["q_v"].data) - col(turb.tracer_tendencies["q_v"].data)
    t_rem = col(full.dT_dt.data) - col(turb.dT_dt.data)
    assert float(jnp.max(jnp.abs(q_rem))) > 0.0 and float(jnp.max(jnp.abs(t_rem))) > 0.0
    got_q = col(out.tracer_tendencies["q_v"].data)
    ok_q = float(jnp.max(jnp.abs(got_q - (q_rem + cand.dqt)))) <= 1e-15
    assert ok_q is (not plant)
    np.testing.assert_allclose(np.asarray(col(out.dT_dt.data)),
                               np.asarray(t_rem + cand.dtheta * inp.exner), rtol=0, atol=1e-15)
    for f, c in (("du_dt", cand.du), ("dv_dt", cand.dv)):
        rem = col(getattr(full, f).data) - col(getattr(turb, f).data)
        np.testing.assert_allclose(np.asarray(col(getattr(out, f).data)),
                                   np.asarray(rem + c), rtol=0, atol=1e-15)


def test_cli_both_needs_both_choices():
    import cosci_eval
    cosci_eval.check_kind_choices("both", "theta", "on")
    for hf, ke in ((None, "on"), ("theta", None), (None, None)):
        with pytest.raises(SystemExit):
            cosci_eval.check_kind_choices("both", hf, ke)


@pytest.mark.parametrize("law, want", [("neutral", 0.0), ("most", 1.0)])
def test_the_closure_sees_the_les_runs_wall_drag_law(tmp_path, law, want):
    """User 2026-10-28: each LES run's recorded wall-drag law reaches the closure as
    ColumnInputs.sfc_drag_most, on candidate and baseline rollouts alike."""
    import dataclasses

    from cosci_eval import Arm
    from tune_scm_to_les import _base_turbulence
    art = _forced_artifact()
    art = dataclasses.replace(art, les_setup=dict(art.les_setup, sfc_flux_drag=law))
    cand = _write(tmp_path, "both_drag", _GOOD_BOTH)
    base = _base_turbulence("louis", **_RGH)[0]
    for arm, params in ((Arm(base), None),
                        (Arm(base, cand, "theta", True), cand.default_params())):
        rec, _ = arm.run(art, 3, params=params, nlev=16, dt=10.0)
        got = np.asarray(rec.inputs.sfc_drag_most)
        assert got.shape == (3,) and np.all(got == want), got
        assert float(rec.final_inputs.sfc_drag_most) == want


def test_a_run_without_a_wall_drag_record_is_refused_and_idealized_columns_are_most():
    """User 2026-10-28: no LES run without a recorded wall-drag law; the idealized check
    columns hand the closure the stability-corrected law; a column built without a law
    carries NaN."""
    import dataclasses

    from cosci_eval import Arm
    from legoesm.atmosphere.les_suite.bridge import BridgeError
    from tune_scm_to_les import _base_turbulence
    art = _forced_artifact()
    bare = dataclasses.replace(art, les_setup={"z0_m": _RGH["z0_m"]})
    with pytest.raises(BridgeError, match="wall-drag law"):
        Arm(_base_turbulence("louis", **_RGH)[0]).prepare(bare, nlev=16, dt=10.0)
    z = jnp.arange(10.0, 500.0, 20.0)
    syn = ev.synthetic_artifact("syn", z, 300.0 + 0.0 * z, 5.0 + 0.0 * z, 0.0 * z,
                                w_theta_s=0.0, hours=1.0, nframes=2, z0_m=0.1)
    assert syn.les_setup["sfc_flux_drag"] == "most"
    assert np.isnan(float(_column().sfc_drag_most))
