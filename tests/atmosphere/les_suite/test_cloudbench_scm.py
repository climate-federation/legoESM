"""CloudBench column builder (les_suite/cloudbench_scm.py) and the conservation gate
(les_suite/conservation.py), on a synthetic CloudBench store."""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.les_suite import cloudbench as cb
from legoesm.atmosphere.les_suite.bridge import rad_tendency_fn
from legoesm.atmosphere.les_suite.candidate_eval import rollout, scm_column_inputs
from legoesm.atmosphere.les_suite.cloudbench_scm import build_cloudbench_scm
from legoesm.atmosphere.les_suite.conservation import check_conservation
from legoesm.atmosphere.les_suite.scm_coupling import interp_profile
from legoesm.atmosphere.les_suite.scm_runner import scm_physics_config
from legoesm.atmosphere.physics import TurbulenceConfig
from legoesm.atmosphere.physics._shared import exner_function
from legoesm.atmosphere.physics.turbulence.config import (
    SmagorinskyConfig,
    SurfaceLayerConfig,
    TurbulentEDMFConfig,
    YSUConfig,
)
from legoesm.atmosphere.physics.turbulence.swirl_most import SwirlMOSTConfig, swirl_most_fluxes

from legoesm import constants
from tests.atmosphere.les_suite.test_cloudbench import _sounding, _write_case

jax.config.update("jax_enable_x64", True)

_PARAMS = {"experiment": "amip", "month": 7, "sst": 298.33, "latitude": 16.5,
           "p_sfc": 100200.0, "theta_li_sfc": 297.6, "q_t_sfc": 0.0185}
_DT = 600.0
_NLEV = 12
_P_SCALE = 1.002     # the test sounding's pressure scaled so p_sfc != 1000 hPa


def _test_sounding():
    s = _sounding()
    return {**s, "p": s["p"] * _P_SCALE}


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cb")
    store, forcing, _, _ = _write_case(tmp, nz=16, nt=3)
    fdir = forcing / "0" / "7" / "amip"
    (fdir / "parameters.json").write_text(json.dumps(_PARAMS))
    lines = (fdir / "sounding.csv").read_text().splitlines()
    ip = lines[0].split(",").index("p")
    out = [lines[0]]
    for ln in lines[1:]:
        v = ln.split(",")
        v[ip] = f"{float(v[ip]) * _P_SCALE:.6e}"
        out.append(",".join(v))
    (fdir / "sounding.csv").write_text("\n".join(out) + "\n")
    art = cb.build_artifact(0, 7, "amip", forcing, store=str(store))
    return art, forcing


def _turb(name="smagorinsky"):
    surf = SurfaceLayerConfig(z0=1e-3, Ch_neutral=0.0)
    sub = {"smagorinsky": SmagorinskyConfig, "ysu": YSUConfig,
           "edmf": TurbulentEDMFConfig}[name](surface=surf)
    return TurbulenceConfig(scheme=name, **{name: sub})


def _build(case, name="smagorinsky"):
    art, forcing = case
    return build_cloudbench_scm(art, _turb(name), forcing_dir=forcing, nlev=_NLEV, dt=_DT)


def test_grid_spans_the_runs_surface_pressure_to_the_les_top_by_the_sounding(case):
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    old = get_policy()
    set_policy(PrecisionPolicy.fp32())      # the default policy: float32 compute
    try:
        scm, grid = _build(case)
    finally:
        set_policy(old)
    s = _test_sounding()
    assert scm.sigma_coord.sigma_half.dtype == jnp.float64
    assert float(scm.state.p_s.data.reshape(())) == _PARAMS["p_sfc"]
    p_top = np.exp(np.interp(cb.DOMAIN_TOP_M, s["z"], np.log(s["p"])))
    assert np.isclose(float(grid.p_half_pa[0]), p_top, rtol=1e-12)
    want_z = np.interp(-np.log(np.asarray(grid.p_full_pa)), -np.log(s["p"]), s["z"])
    assert np.allclose(np.asarray(grid.z_scm_m), want_z, rtol=1e-12)
    assert np.all(np.diff(np.asarray(grid.z_scm_m)) < 0.0)     # top-first


def test_start_is_frame_zero_at_the_column_levels(case):
    art, _ = case
    scm, grid = _build(case)
    z_les, z = jnp.asarray(art.heights_m), grid.z_scm_m
    ex = exner_function(grid.p_full_pa)
    q_c = jnp.clip(interp_profile(jnp.asarray(art.qc)[0], z_les, z), 0.0, None)
    theta = interp_profile(jnp.asarray(art.theta)[0], z_les, z) \
        + constants.L_v / (constants.c_pd * ex) * q_c
    T = np.asarray(scm.state.T.data).reshape(-1)
    q_v = np.asarray(scm.state.tracers["q_v"].data).reshape(-1)
    assert np.allclose(T, np.asarray(theta * ex), rtol=1e-13)
    assert np.allclose(q_v, np.asarray(interp_profile(jnp.asarray(art.qt)[0], z_les, z) - q_c),
                       rtol=1e-13)
    assert np.allclose(np.asarray(scm.state.u.data).reshape(-1),
                       np.asarray(interp_profile(jnp.asarray(art.u)[0], z_les, z)))


def test_forcing_channels_are_the_recipes(case):
    art, _ = case
    scm, grid = _build(case)
    f = scm.forcing_arg
    z_les, z = jnp.asarray(art.heights_m), grid.z_scm_m
    on = lambda p: np.asarray(interp_profile(jnp.asarray(p), z_les, z))  # noqa: E731
    assert f.subsidence_on == "theta_l_qt" and f.prescribe == "fluxes" and f.flux_to_closure
    assert f.f_c == 0.0 and f.w_th_s is None and f.surface_flux_fn is not None
    assert f.wind_relax_rate == 1.0 / art.les_setup["forcing_recipe"]["tau_wind_s"]
    assert np.allclose(np.asarray(f.subsidence_w(0.0)), on(art.subsidence_w))
    assert np.allclose(np.asarray(f.nudge_rate(0.0)), on(art.nudge_rate))
    assert np.allclose(np.asarray(f.theta_l_target(1e5)), on(np.asarray(art.theta)[0]))
    assert np.allclose(np.asarray(f.qt_target(1e5)), on(np.asarray(art.qt)[0]))
    assert np.allclose(np.asarray(f.u_target(1e5)), on(np.asarray(art.u)[0]))
    rad = rad_tendency_fn(art, dt_s=_DT)
    for t in (0.0, 96 * 3600.0 + 600.0):
        assert np.allclose(np.asarray(f.rad_T_tendency(t)), on(rad(t)), rtol=1e-13)


def test_surface_record_is_swirl_most_on_the_lowest_level(case):
    art, _ = case
    scm, _ = _build(case)
    rec = scm.forcing_arg.surface_flux_fn(scm.state, scm.sigma_coord, 0.0)
    zero = jnp.zeros(())
    ci = scm_column_inputs(scm.state, scm.sigma_coord, None, f_c=0.0, u_geo=None,
                           v_geo=None, dt=_DT, uw_s=zero, vw_s=zero)
    cfg = SwirlMOSTConfig(z0_m=1e-3, gustiness_m_s=1.0, g_m_s2=cb.SWIRL_G_M_S2,
                          density_free_drag=True)
    th_s = cb.surface_theta_l(_PARAMS["theta_li_sfc"], _PARAMS["q_t_sfc"], _PARAMS["p_sfc"])
    want = swirl_most_fluxes(ci.u[0], ci.v[0], ci.temp[0] / ci.exner[0], ci.theta_l[0],
                             ci.q_v[0] + ci.q_c[0], ci.rho_full[0], ci.z_full[0], th_s,
                             _PARAMS["q_t_sfc"], cfg)
    for a, b in zip((rec.w_theta_l, rec.w_q_t, rec.uw, rec.vw), want):
        assert np.isclose(float(a), float(b), rtol=1e-12)
    assert float(rec.rho_s) == pytest.approx(float(ci.rho_full[0]), rel=1e-14)
    assert float(rec.exner_s) == pytest.approx(float(ci.exner_half[0]), rel=1e-14)


def test_surface_theta_l_gives_the_run_surface_temperature():
    # cloudbench_s000_m07_amip parameters.json: the LES surface theta_li at p_sfc is
    # the SST it was derived from, to the 4 decimals the record carries
    th = cb.surface_theta_l(297.1260681152344, 0.019206369295716286, 101448.1015625)
    t_s = th * float(exner_function(jnp.asarray(101448.1015625)))
    assert t_s == pytest.approx(298.3300476074219, abs=2e-3)


def test_builder_refuses_another_sounding_and_non_cloudbench(case, tmp_path):
    art, forcing = case
    fdir = tmp_path / "0" / "7" / "amip"
    fdir.mkdir(parents=True)
    text = (forcing / "0" / "7" / "amip" / "sounding.csv").read_text()
    (fdir / "sounding.csv").write_text(text.replace("2.980000e+02", "2.981000e+02", 1))
    with pytest.raises(ValueError, match="not the sounding"):
        build_cloudbench_scm(art, _turb(), forcing_dir=tmp_path, nlev=_NLEV, dt=_DT)
    import dataclasses
    other = dataclasses.replace(art, les_setup={**art.les_setup, "source": "DEPHY"})
    with pytest.raises(ValueError, match="not a CloudBench"):
        build_cloudbench_scm(other, _turb(), forcing_dir=forcing, nlev=_NLEV, dt=_DT)
    renamed = dataclasses.replace(art, case_name="cloudbench_000_m07_amip")
    with pytest.raises(ValueError, match="not a CloudBench case name"):
        build_cloudbench_scm(renamed, _turb(), forcing_dir=forcing, nlev=_NLEV, dt=_DT)


# --- conservation gate ---------------------------------------------------------------

_NSTEPS = 6


def _run(case, name="smagorinsky", *, energy_form=True, leak=None):
    art, _ = case
    scm, _ = _build(case, name)
    if leak is not None:
        base = scm.physics_fn

        def leaky(st, grid, sc, phys_state=None):
            tend, out = base(st, grid, sc, phys_state=phys_state)
            return leak(tend), out
        scm.physics_fn = leaky
    rec, _ = rollout(scm, scm_physics_config(art, _turb(name)), _NSTEPS,
                     energy_form_baseline=energy_form)
    return rec


@pytest.mark.parametrize("name", ["smagorinsky", "ysu"])
def test_energy_form_benchmark_passes_the_gate(case, name):
    v = check_conservation(_run(case, name), _DT)
    assert v.passed, v.reason
    assert max(v.worst_step_rel.values()) < 1e-14


def test_native_benchmark_fails_on_energy_only(case):
    v = check_conservation(_run(case, "ysu", energy_form=False), _DT)
    assert not v.passed and "turbulence energy" in v.reason
    assert v.worst_step_rel["energy"] > 1e-6
    assert v.worst_process_rel["turbulence_energy"] > 1e-6
    for q in ("water", "u", "v"):
        assert v.worst_step_rel[q] < 1e-12, q
        assert v.worst_process_rel["turbulence_" + q] < 1e-12, q
        assert v.worst_process_rel["microphysics_" + q] < 1e-12, q


def _add(field, k, amount):
    d = np.zeros(field.data.shape)
    d.reshape(-1)[k] = amount
    return field.replace(data=field.data + d)


@pytest.mark.parametrize("what,quantity", [("q_v", "water"), ("T", "energy"), ("u", "u")])
def test_a_planted_source_fails_the_gate(case, what, quantity):
    def leak(tend):
        if what == "q_v":
            tr = dict(tend.tracer_tendencies)
            tr["q_v"] = _add(tr["q_v"], 5, 1.0e-9)
            return tend._replace(tracer_tendencies=tr)
        if what == "T":
            return tend._replace(dT_dt=_add(tend.dT_dt, 5, 1.0e-7))
        return tend._replace(du_dt=_add(tend.du_dt, 5, 1.0e-7))
    v = check_conservation(_run(case, leak=leak), _DT)
    # the leak rides on the non-turbulent physics, so that process and the total fail
    assert not v.passed
    assert v.worst_step_rel[quantity] > 1e-6
    assert v.worst_process_rel["microphysics_" + quantity] > 1e-6
    assert v.worst_process_rel["turbulence_" + quantity] < 1e-12


def test_opposite_errors_in_two_processes_do_not_cancel(case):
    rec = _run(case)
    b = dict(rec.budget)
    eps = 1e-3 * jnp.abs(b["sfc_water"])
    b["turbulence_water"] = b["turbulence_water"] + eps
    b["microphysics_water"] = b["microphysics_water"] - eps
    v = check_conservation(rec._replace(budget=b), _DT)
    assert v.worst_step_rel["water"] < 1e-12            # the total still closes
    assert not v.passed and "turbulence water" in v.reason and "microphysics water" in v.reason


def test_a_record_density_that_is_not_the_columns_fails_the_gate(case):
    rec = _run(case)
    b = dict(rec.budget)
    b["record_rho_rel"] = b["record_rho_rel"] + 1e-3
    v = check_conservation(rec._replace(budget=b), _DT)
    assert not v.passed and "record rho" in v.reason


def test_an_own_boundary_flux_against_a_zero_law_fails(case):
    rec = _run(case)
    b = dict(rec.budget)
    inputs = rec.inputs._replace(vw_s=jnp.zeros_like(rec.inputs.vw_s))
    b["sfc_v"] = jnp.zeros_like(b["sfc_v"])
    b["own_sfc_vw"] = jnp.full_like(b["own_sfc_vw"], 5e-3)
    v = check_conservation(rec._replace(budget=b, inputs=inputs), _DT)
    assert not v.passed and v.own_boundary_rel["vw"] > 1e-2


def test_round_off_against_a_vanishing_supply_passes_the_step_check(case):
    # supplies exactly zero while the turbulence still mixes: the step total is
    # pure round-off of the mixing and must be judged against it, not against nothing
    rec = _run(case)
    b = dict(rec.budget)
    zero = jnp.zeros_like(b["sfc_water"])
    noise = 1e-16 * b["gross_turbulence_water"]
    b.update(sfc_water=zero, forcing_water=zero, precip=zero, rate_water=noise,
             turbulence_water=noise, microphysics_water=zero)
    v = check_conservation(rec._replace(budget=b), _DT)
    assert v.worst_step_rel["water"] < 1e-12


def test_a_run_without_steps_fails():
    from legoesm.atmosphere.les_suite.candidate_eval import RolloutRecord
    v = check_conservation(RolloutRecord(inputs=None, fluxes=None, final_inputs=None,
                                         budget={"rate_water": jnp.zeros((0,))}), _DT)
    assert not v.passed and "no steps" in v.reason


def test_a_doubled_declared_supply_fails_the_gate(case):
    rec = _run(case)
    b = dict(rec.budget)
    b["sfc_water"] = 2.0 * b["sfc_water"]
    v = check_conservation(rec._replace(budget=b), _DT)
    assert not v.passed and "water" in v.reason


def test_a_non_finite_budget_fails_the_gate(case):
    rec = _run(case)
    b = dict(rec.budget)
    b["forcing_u"] = b["forcing_u"].at[2].set(jnp.nan)
    v = check_conservation(rec._replace(budget=b), _DT)
    assert not v.passed and "non-finite" in v.reason


@pytest.mark.parametrize("energy_form", [True, False])
def test_every_closure_receives_the_surface_law(case, energy_form):
    # the benchmark's own surface fluxes (before the harness sets them) are the law's:
    # water and stress exactly, heat up to the closure's own Exner conversion
    v = check_conservation(_run(case, energy_form=energy_form), _DT)
    assert v.own_boundary_rel["wqt"] < 1e-10 and v.own_boundary_rel["uw"] < 1e-10
    assert v.own_boundary_rel["vw"] < 1e-10
    assert v.own_boundary_rel["wtheta"] < 1e-2


def test_gate_needs_a_surface_law_budget(case):
    rec = _run(case)
    with pytest.raises(ValueError, match="no budget"):
        check_conservation(rec._replace(budget=None), _DT)


def _one_layer_record(precip_in_budget):
    """A one-layer, one-step column whose turbulence evaporates E and whose
    microphysics rains P out of the vapour (no other change)."""
    from legoesm.atmosphere.les_suite.candidate import column_inputs
    from legoesm.atmosphere.les_suite.candidate_eval import RolloutRecord
    m, E, P, q0 = 1.0e4, 4.0e-5, 1.5e-5, 0.012
    lv = constants.L_v  # latent-ok: test column

    def col(q):
        one = jnp.ones((1,))
        return column_inputs(z_full=50 * one, z_half=jnp.asarray([0.0, 100.0]),
                             rho_full=one, rho_half=jnp.ones((2,)), mass=m * one,
                             exner=one, exner_half=jnp.ones((2,)), p_full=1e5 * one,
                             p_half=jnp.asarray([1e5, 9.9e4]), theta_l=300 * one,
                             q_t=q * one, u=0 * one, v=0 * one, w_theta_s=0.0,
                             w_qt_s=E, uw_s=0.0, vw_s=0.0, dt=_DT, temp=300 * one,
                             q_v=q * one, q_c=0 * one)
    c0, c1 = col(q0), col(q0 + _DT * (E - P) / m)
    z, a = jnp.zeros((1,)), jnp.asarray
    # the fake rain leaves from the vapour, so its energy L_v P is booked as forcing
    budget = {"rate_water": a([E - P]), "rate_energy": a([lv * (E - P)]),
              "rate_u": z, "rate_v": z, "sfc_water": a([E]), "sfc_energy": a([lv * E]),
              "sfc_u": z, "sfc_v": z, "precip": a([precip_in_budget]),
              "turbulence_water": a([E]), "gross_turbulence_water": a([E]),
              "turbulence_energy": a([lv * E]), "gross_turbulence_energy": a([lv * E]),
              "microphysics_water": a([-P]), "gross_microphysics_water": a([P]),
              "microphysics_energy": z, "gross_microphysics_energy": z,
              "forcing_water": z, "gross_forcing_water": z,
              "forcing_energy": a([-lv * P]), "gross_forcing_energy": a([lv * P]),
              "record_rho_rel": z, "record_exner_rel": z,
              "own_sfc_wtheta": z, "own_sfc_wqt": a([E]), "own_sfc_uw": z, "own_sfc_vw": z}
    for p in ("turbulence", "microphysics", "forcing"):
        for c in ("u", "v"):
            budget[f"{p}_{c}"] = z
            budget[f"gross_{p}_{c}"] = z
    stack = jax.tree_util.tree_map(lambda x: x[None], c0)
    return RolloutRecord(inputs=stack, fluxes=None, final_inputs=c1, budget=budget)


def test_rain_reaching_the_ground_is_a_declared_water_sink():
    assert check_conservation(_one_layer_record(1.5e-5), _DT).passed
    v = check_conservation(_one_layer_record(-1.5e-5), _DT)
    assert not v.passed and "water" in v.reason


@pytest.mark.parametrize("change", [{"gross_microphysics_u": np.nan},
                                    {"own_sfc_uw": 1.0, "gross_turbulence_u": np.inf}])
def test_a_non_finite_budget_scale_fails_the_gate(change):
    rec = _one_layer_record(1.5e-5)
    b = dict(rec.budget)
    b.update({k: jnp.asarray([v]) for k, v in change.items()})
    v = check_conservation(rec._replace(budget=b), _DT)
    assert not v.passed and "non-finite" in v.reason


def test_a_closure_with_its_own_surface_flux_fails_the_gate(case):
    # EDMF adds a plume flux at the surface on top of the one it is given
    v = check_conservation(_run(case, "edmf"), _DT)
    assert not v.passed and "own surface flux" in v.reason
    assert v.own_boundary_rel["wtheta"] > 1e-2
    assert v.worst_step_rel["energy"] < 1e-12      # the budget itself closes


def test_the_harness_column_carries_the_surface_law(case):
    art, _ = case
    scm, _ = _build(case)
    rec = rollout(scm, scm_physics_config(art, _turb()), 2, energy_form_baseline=True)[0]
    law = scm.forcing_arg.surface_flux_fn(scm.state, scm.sigma_coord, 0.0)
    got = (rec.inputs.w_theta_s[0], rec.inputs.w_qt_s[0], rec.inputs.uw_s[0],
           rec.inputs.vw_s[0])
    for a, b in zip(got, (law.w_theta_l, law.w_q_t, law.uw, law.vw)):
        assert float(a) == pytest.approx(float(b), rel=1e-14)  # compiled vs eager: last bit


def test_swirl_most_gradient_in_roughness_under_jit():
    from legoesm.atmosphere.physics.turbulence.swirl_most import SwirlMOSTConfig, swirl_most_fluxes

    def heat(z0):
        f = swirl_most_fluxes(jnp.asarray(-5.0), jnp.asarray(2.0), jnp.asarray(296.5),
                              jnp.asarray(296.4), jnp.asarray(0.017), jnp.asarray(1.17),
                              jnp.asarray(70.0), jnp.asarray(297.126), jnp.asarray(0.0192),
                              SwirlMOSTConfig(z0_m=z0))
        return f.w_theta_l
    g = jax.jit(jax.grad(heat))(1e-3)
    assert bool(jnp.isfinite(g)) and float(g) > 0.0     # rougher: more heat flux
