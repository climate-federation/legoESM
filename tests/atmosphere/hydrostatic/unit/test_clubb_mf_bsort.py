"""Opt-in CAM buoyancy sorting, its smooth switches, and carried plume tops.

Pinned cec8c93ed2 oracle: poisson public + bsort=.true.; same fixture layout
as test_clubb_mf_oracle, including two calls with call 1's memory in call 2.
Run on a COMPUTE NODE only. To record default and combined HARD per-column gaps:
    python -m tests.atmosphere.hydrostatic.unit.test_clubb_mf_bsort --write-gaps
No values are invented when that measurement has not run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from legoesm.atmosphere.physics.turbulence.clubb_mf import (
    HARD, NUP, MFSmoothing, bsort_choose, bsort_clip, bsort_roots,
    buoyancy_sort_mixfrac, clubb_mf_forcing, integrate_mf, limit_bsort_area,
)
from legoesm.core.smooth import expm1_over_x

from tests.atmosphere.hydrostatic.unit.test_clubb_mf_oracle import (
    C_KEYS, FIELDS, FLUX, TOL, _rel,
)

jax.config.update("jax_enable_x64", True)
FIXTURE = Path(__file__).resolve().parents[2] / "oracle_fixtures" / "clubb_mf_bsort.npz"
GAPS = FIXTURE.with_name("clubb_mf_bsort_default_gaps.json")
COMBINED_FIXTURE = FIXTURE.with_name("clubb_mf_bsort_alphturb3.npz")
HARD_GAPS = FIXTURE.with_name("clubb_mf_bsort_alphturb3_hard_gaps.json")
REPO = Path(__file__).resolve().parents[4]
EXTRA = ("thl_zm", "qt_zm", "thv_zm")
SWITCHES = tuple(f for f in MFSmoothing._fields if f.startswith("bsort_"))
# Vendored verbatim from 097655038e^2 (origin/main merge parent, with alphturb).
PRE_BSORT = Path(__file__).with_name("_clubb_mf_pre_bsort.py.txt")
SOURCE = "packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb_mf.py"


@pytest.fixture(scope="module")
def fx():
    return dict(np.load(FIXTURE, allow_pickle=True))


def _cols(fx):
    return {k: jnp.asarray(fx[f"c_{k}"]) for k in C_KEYS + EXTRA + ("tke",)}


_RUN = {}


def _run(fx, sm, step, memory=None, alphturb=0.0):
    key = (sm, alphturb)
    if key not in _RUN:
        _RUN[key] = jax.jit(jax.vmap(lambda c, e, z: integrate_mf(
            c, e, z, sm, alphturb=alphturb, bsort=True)))
    return _RUN[key](_cols(fx), jnp.asarray(fx[f"ora{step}_enti"]),
                    jnp.asarray(fx[f"ora{step}_ztopm1_in"]) if memory is None else memory)


def _rel_columns(a, b):
    """One field normalised within each column; all-zero oracle uses absolute error."""
    return [_rel(ai, bi) for ai, bi in zip(a, b, strict=True)]


def test_fixture_provenance_and_nonvacuity(fx):
    meta = json.loads(str(fx["meta"]))
    assert meta["oracle"]["sha"] == "cec8c93ed22e6dd60bd676459d4fc0468826fe7d"
    # Includes the two requested source edits; do not accept the bsort-off fixture.
    assert meta["build"]["clubb_mf.F90"] == "3507ea5e52c764b3afa6b6e3e7d056e2db681d55cb9588443bce2a6ed2c91bf9"
    assert meta["nup"] == NUP and len(meta["names"]) == 20
    np.testing.assert_array_equal(fx["ora2_ztopm1_in"], fx["ora1_ztopm1"])
    assert np.any(fx["ora1_ztopm1"] != fx["ora1_ztopm1_in"])
    assert np.any(fx["ora2_upa"] > 0.0)
    assert np.any(fx["ora2_upqc"] > 0.0)
    ac = np.sum(fx["ora2_upa"][:, :-1], axis=-1)
    assert np.max(ac) <= 0.6 + TOL
    # Source fixture genuinely changes area with height (constant-area lane must fail).
    assert np.any(np.abs(np.diff(fx["ora2_upa"], axis=1)) > TOL)


@pytest.mark.parametrize("step", [1, 2])
def test_hard_limit_matches_cam_oracle(fx, step):
    got = _run(fx, HARD, step)
    errors = {k: _rel_columns(np.asarray(got[k]), fx[f"ora{step}_{k}"]) for k in FIELDS}
    assert max(max(values) for values in errors.values()) <= TOL, errors


@pytest.mark.parametrize("step", [1, 2])
def test_bsort_alphturb3_hard_limit_matches_cam_oracle(step):
    fx = dict(np.load(COMBINED_FIXTURE, allow_pickle=True))
    assert json.loads(str(fx["meta"]))["init_reals"][4] == 3.0
    assert HARD_GAPS.exists(), "Measure on a compute node: python -m tests.atmosphere.hydrostatic.unit.test_clubb_mf_bsort --write-gaps"
    stored = json.loads(HARD_GAPS.read_text())
    assert stored["fixture_sha256"] == hashlib.sha256(COMBINED_FIXTURE.read_bytes()).hexdigest()
    assert stored["gaps"].keys() == {"1", "2"}
    exceptions = stored["gaps"][str(step)]
    assert exceptions.keys() == set(FIELDS)
    got = _run(fx, HARD, step, alphturb=3.0)
    errors = {k: _rel_columns(np.asarray(got[k]), fx[f"ora{step}_{k}"]) for k in FIELDS}
    sensitivity = _hard_ulp_sensitivity(fx, step, got)
    for field, values in errors.items():
        assert exceptions[field].keys() <= {str(i) for i in range(len(values))}
        for column, value in enumerate(values):
            label = f"step{step}/{field}/{column}"
            if str(column) in exceptions[field]:
                expected = exceptions[field][str(column)]
                assert TOL < expected <= 10 * sensitivity[column], label
                np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-12,
                                           err_msg=label)
            else:
                assert value <= TOL, (label, value)


def _hard_ulp_sensitivity(fx, step, baseline):
    """Max per-column relative upa/flux change under separate thl/qt perturbations."""
    changes = []
    for field in ("thl", "qt"):
        perturbed = dict(fx)
        perturbed[f"c_{field}"] = fx[f"c_{field}"] * (1.0 + 2**-52)
        assert np.any(perturbed[f"c_{field}"] != fx[f"c_{field}"]), field
        got = _run(perturbed, HARD, step, alphturb=3.0)
        changes.extend(_rel_columns(np.asarray(got[k]), np.asarray(baseline[k]))
                       for k in ("upa",) + FLUX)
    return np.max(changes, axis=0)


def _hard_gaps(fx):
    """Only above-TOL exceptions; unlisted columns retain the strict oracle bar."""
    gaps = {}
    for step in (1, 2):
        got = _run(fx, HARD, step, alphturb=3.0)
        gaps[str(step)] = {
            k: {str(i): value for i, value in enumerate(
                _rel_columns(np.asarray(got[k]), fx[f"ora{step}_{k}"]))
                if value > TOL}
            for k in FIELDS
        }
    return gaps


@pytest.mark.parametrize("memory_m", [0.0, 500.0])
def test_bsort_alphturb3_smooth_values_and_reverse_gradients_are_finite(memory_m):
    """Measured near-dead-lane overflow: all 20 columns, both memory seeds.

    Differentiate through the plume velocities/root coefficients via the
    surface heat flux and environmental thl, rather than a detached output.
    """
    fx = dict(np.load(FIXTURE.with_name("clubb_mf_bsort_alphturb3.npz"), allow_pickle=True))
    memory = jnp.full((NUP,), memory_m)

    def evaluate(c, draws):
        def loss(thl, heat_flux):
            out = integrate_mf(dict(c, thl=thl, wthl_sfc=heat_flux), draws,
                               memory, MFSmoothing(), alphturb=3.0, bsort=True)
            return sum(jnp.sum(out[k]) for k in FIELDS), out
        return jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(c["thl"], c["wthl_sfc"])

    (values, outputs), grads = jax.jit(jax.vmap(evaluate))(
        _cols(fx), jnp.asarray(fx["ora1_enti"]))
    for name, value in dict(outputs, loss=values).items():
        assert np.all(np.isfinite(np.asarray(value))), (memory_m, name)
    for grad in grads:
        assert np.all(np.isfinite(np.asarray(grad))), memory_m
        assert np.any(np.asarray(grad) != 0.0), "detached regression observable"


def test_second_call_uses_first_calls_memory(fx):
    first = _run(fx, HARD, 1)
    second = _run(fx, HARD, 2, first["ztopm1"])
    for k in FIELDS:
        assert max(_rel_columns(np.asarray(second[k]), fx[f"ora2_{k}"])) <= TOL, k
    reset = _run(fx, HARD, 2, jnp.asarray(fx["ora1_ztopm1_in"]))
    assert np.any(np.asarray(second["upa"]) != np.asarray(reset["upa"]))


def _gaps(fx):
    gaps = {}
    # Existing switches also operate inside the extra condensation calls.
    for switch in MFSmoothing._fields:
        sm = HARD._replace(**{switch: getattr(MFSmoothing(), switch)})
        for step in (1, 2):
            got = _run(fx, sm, step)
            gaps[f"{switch}_step{step}"] = {
                k: _rel_columns(np.asarray(got[k]), fx[f"ora{step}_{k}"])
                for k in FIELDS
            }
    return gaps


def test_default_gaps_per_column_and_call(fx):
    assert GAPS.exists(), "Measure on a compute node: python -m tests.atmosphere.hydrostatic.unit.test_clubb_mf_bsort --write-gaps"
    stored = json.loads(GAPS.read_text())
    assert stored["fixture_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    got = _gaps(fx)
    assert got.keys() == stored["gaps"].keys()
    for case, fields in got.items():
        for k, values in fields.items():
            np.testing.assert_allclose(values, stored["gaps"][case][k], rtol=1e-6, atol=1e-12,
                                       err_msg=f"{case}/{k}")


@pytest.fixture(scope="module")
def main_module():
    mod = types.ModuleType("clubb_mf_pre_bsort_reference")
    exec(compile(PRE_BSORT.read_text(), str(PRE_BSORT), "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("alphturb", [0.0, 3.0])
@pytest.mark.parametrize("hard", [False, True])
@pytest.mark.parametrize("step", [1, 2])
def test_bsort_off_is_bitwise_main(fx, main_module, hard, step, alphturb):
    old_sm = main_module.HARD if hard else main_module.MFSmoothing()
    new_sm = HARD if hard else MFSmoothing()
    old = jax.jit(jax.vmap(lambda c, e, z: main_module.integrate_mf(c, e, z, old_sm, alphturb=alphturb)))
    new = jax.jit(jax.vmap(lambda c, e, z: integrate_mf(c, e, z, new_sm, alphturb=alphturb, bsort=False)))
    args = (_cols(fx), jnp.asarray(fx[f"ora{step}_enti"]), jnp.asarray(fx[f"ora{step}_ztopm1_in"]))
    before, after = old(*args), new(*args)
    assert before.keys() == after.keys()
    for k in before:
        a, b = np.asarray(before[k]), np.asarray(after[k])
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8), err_msg=k)


def test_bsort_area_cap_reaches_its_branch_and_preserves_plume_ratios():
    area = jnp.asarray([[0.1, 0.2], [0.4, 0.8], [0.4, 0.8]])
    got = np.asarray(limit_bsort_area(area, math.inf))
    np.testing.assert_allclose(got, [[0.1, 0.2], [0.2, 0.4], [0.4, 0.8]], rtol=TOL, atol=TOL)
    # Planted violation: removing the cap fails on row 1; the top is excluded by CAM.
    assert np.sum(np.asarray(area)[1]) > 0.6 and np.sum(got[1]) <= 0.6 + TOL


def test_stable_roots_near_stopping_plume_and_zero_b():
    small, _ = bsort_roots(jnp.asarray(1e-16), jnp.asarray(-1.0), jnp.asarray(0.5), HARD)
    np.testing.assert_allclose(small, 0.5, rtol=TOL, atol=TOL)
    # Planted naive quadratic subtraction misses this finite neutral fraction.
    naive = (1.0 - np.sqrt(1.0 - 2e-16)) / 2e-16
    assert abs(naive - 0.5) > TOL
    small, _ = bsort_roots(jnp.asarray(1.0), jnp.asarray(0.0), jnp.asarray(-1.0), HARD)
    assert float(small) == -1.0
    small, _ = bsort_roots(jnp.asarray(1.0), jnp.asarray(0.0), jnp.asarray(0.0), HARD)
    assert float(small) == 0.0


@pytest.mark.parametrize("sm", [HARD, MFSmoothing()])
@pytest.mark.parametrize("a", [1e-16, 1e-20, 1e-25, 1e-30, 1e-37])
def test_stable_roots_fp32_reverse_gradient_has_no_discarded_division_nan(sm, a):
    def root(aa):
        return bsort_roots(aa, jnp.float32(-1.0), jnp.float32(0.5), sm)[0]
    value, grad = jax.jit(jax.value_and_grad(root))(jnp.float32(a))
    assert value.dtype == grad.dtype == jnp.float32
    assert np.isfinite(np.asarray(value)) and np.isfinite(np.asarray(grad))
    # The finite root retains a nonzero derivative through the discriminant.
    assert float(grad) != 0.0
    expected = 0.5 if math.isinf(sm.bsort_root) else 0.5 / (1.0 + (1.0 / sm.bsort_root)**2)
    np.testing.assert_allclose(value, expected, rtol=4 * np.finfo(np.float32).eps)


@pytest.mark.parametrize("sm", [HARD, MFSmoothing()])
def test_dead_scan_lanes_do_not_evaluate_overflowing_mass_growth(fx, sm):
    c = {k: v[0].astype(jnp.float32) for k, v in _cols(fx).items()}
    c["thv"] = c["thv"] + 100.0  # force negative acceleration and immediate death
    draws = jnp.full((c["dzt"].shape[0], NUP), 10000, dtype=jnp.int32)
    out = integrate_mf(c, draws, jnp.zeros(NUP, jnp.float32), sm, bsort=True)
    for name, value in out.items():
        assert np.all(np.isfinite(np.asarray(value))), name
    # Planted unguarded exp(ent0*10000) cannot fit even in fp64.
    assert 0.2 * 10000 > np.log(np.finfo(np.float64).max)


def test_zero_entrainment_continuation_has_the_correct_value_and_slope():
    for x in (0.0, -1e-30):
        np.testing.assert_allclose(float(expm1_over_x(jnp.asarray(x))), 1.0, rtol=TOL)
        np.testing.assert_allclose(float(jax.grad(expm1_over_x)(jnp.asarray(x))), 0.5, rtol=TOL)
    check_grads(expm1_over_x, (jnp.asarray(0.0),), order=1, modes=("fwd", "rev"), eps=1e-3)


def _gradient_function(switch, sm):
    """Place production helpers on each transition; no clone of switch maths."""
    if switch == "bsort_xc":
        return lambda x: bsort_clip(0.005 + 0.01 * x, sm.bsort_xc)
    if switch == "bsort_area":
        return lambda x: jnp.sum(limit_bsort_area(
            jnp.stack([jnp.asarray([0.2, 0.4]) * (1.0 + 0.01 * x), jnp.zeros(2)]),
            sm.bsort_area))
    if switch == "bsort_equal":
        return lambda x: bsort_choose(0.495 + 0.01 * x, jnp.asarray(0.8), jnp.asarray(0.5), sm)
    if switch == "bsort_disc":
        return lambda x: bsort_roots(jnp.asarray(1.0), jnp.asarray(-1.0), 0.25 + 0.001 * x, sm)[0]
    if switch == "bsort_root":
        return lambda x: bsort_roots(jnp.asarray(0.01), -0.01 + 0.001 * x,
                                     jnp.asarray(0.001), sm)[0]
    # The case product crosses 0 at a saturated plume in a dry environment;
    # the excess regularisation uses the same small water scale as sm.qc.
    from legoesm.thermo import saturation_specific_humidity_cam_table
    p, iex, thl = jnp.asarray(1e5), jnp.asarray(1.0), jnp.asarray(300.0)
    qs = saturation_specific_humidity_cam_table(thl, p, clamp_sharpness=math.inf,
                                               cap_sharpness=math.inf)
    q_env = qs - (1e-5 if switch == "bsort_case" else 5e-6)
    scale, base = 1e-5, 5e-6
    from legoesm.atmosphere.physics.turbulence.clubb_mf import condensation_mf
    # Freeze the environment while perturbing plume water. This places the
    # neutral fraction inside the cumulus segment instead of at the 1 cap.
    thv_env = condensation_mf(qs + base, thl, p, iex, HARD)[0] + 0.25

    def f(x):
        ee, ud = buoyancy_sort_mixfrac(qs + base + scale * x, thl, jnp.asarray(1.0),
                                      thl, q_env, thv_env, iex, p,
                                      jnp.asarray(500.0), sm)
        return ee + 0.37 * ud
    return f


@pytest.mark.parametrize("switch", SWITCHES)
def test_check_grads_through_each_new_switch_at_default(switch):
    sm = MFSmoothing()
    f = _gradient_function(switch, sm)
    hard = _gradient_function(switch, sm._replace(**{switch: math.inf}))
    x = jnp.asarray(0.0)
    g, gh = float(jax.grad(f)(x)), float(jax.grad(hard)(x))
    assert math.isfinite(g) and g != 0.0
    assert abs(g - gh) > 1e-3 * max(abs(g), abs(gh)), (switch, g, gh)
    # Normalise only the test observable so the default check_grads absolute
    # tolerance cannot accidentally certify a detached tiny derivative.
    check_grads(lambda dx: f(dx) / abs(g), (x,), order=1, modes=("fwd", "rev"), eps=1e-3)


@pytest.mark.parametrize("sm", [HARD, MFSmoothing()])
@pytest.mark.parametrize("alphturb,memory_m", [(0.0, 0.0), (3.0, 0.0), (3.0, 500.0)])
def test_column_budgets_close_with_bsort(fx, sm, alphturb, memory_m):
    c = _cols(fx)
    rho = jnp.asarray(fx["c_rho_zt"])
    # Reuse the host divergence and CFL limiter, rather than reimplementing a budget.
    dt = float(json.loads(str(fx["meta"]))["dt"])
    heat, water, _ = jax.jit(jax.vmap(lambda col, rz, rzt, dz: clubb_mf_forcing(
        col, dt, rz, 1.0 / rzt, 1.0 / dz, sm, math.inf,
        alphturb=alphturb, bsort=True, ztopm1=jnp.full((NUP,), memory_m))))(
            c, c["rho_zm"], rho, c["dzt"])
    for tendency in (heat, water):
        terms = np.asarray(tendency * rho * c["dzt"])
        err = np.abs(np.sum(terms, axis=1))
        scale = np.sum(np.abs(terms), axis=1)
        assert np.all(np.isfinite(terms))
        assert np.all(err <= 32 * np.finfo(terms.dtype).eps * scale), (err, scale)
    # Planted nonzero boundary flux/tendency breaks the column budget.
    assert np.any(np.asarray(water) != 0.0)
    bad = np.asarray(water * rho * c["dzt"]).copy()
    bad[:, 0] += 1.0
    assert np.any(np.abs(bad.sum(axis=1)) > TOL)


@pytest.mark.parametrize("on", [False, True])
def test_carry_wiring_step_to_step_only_when_on(on):
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state, update_physics_state
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence.integration import (
        _carry_update_with_cloud_fraction, _clubb_mf_kwargs,
    )
    from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
    ps = init_physics_state(2, 6, PhysicsConfig())
    np.testing.assert_array_equal(ps.mf_ztopm1, np.zeros((2, NUP)))
    tc = TurbulenceConfig(clubb_mf=True, clubb_mf_bsort=on)
    top = jnp.arange(2 * NUP, dtype=ps.mf_ztopm1.dtype).reshape(2, NUP) + 100.0
    # Poisoned incoming memory must be unread and erased in an off run.
    ps = ps._replace(mf_ztopm1=top)
    for _ in range(2):
        kw = _clubb_mf_kwargs(tc, ps, 2)
        if on:
            np.testing.assert_array_equal(kw["mf_ztopm1"], top)
            top = top + 1.0
        else:
            assert "mf_ztopm1" not in kw
        z = jnp.zeros((2, 6)); s = jnp.zeros(2)
        out = TurbulenceOutput(du_dt=z, dv_dt=z, dT_dt=z, dq_v_dt=z, Km=z, Kh=z,
                               shflx=s, lhflx=s, ustar=s, h_pbl=s,
                               mf_ztopm1=top if on else None)
        ps = update_physics_state(ps, _carry_update_with_cloud_fraction(None, None, out))
        np.testing.assert_array_equal(ps.mf_ztopm1, top if on else np.zeros((2, NUP)))


@pytest.mark.parametrize("n_sub", [1, 2])
def test_real_bsort_bridge_returns_kernel_top_and_carries_memory(fx, monkeypatch, n_sub):
    """No fabricated plume outputs: real bridge, real forcing, real kernel.

    Explicitly requiring bsort=True is the omission control; the comparisons
    also require nonzero plumes and physical sensitivity to incoming memory.
    """
    from legoesm.atmosphere.physics.turbulence import clubb as cb
    from legoesm.atmosphere.physics.turbulence import clubb_mf as mf

    # One 32-level monsoon fixture column, including its production host grid.
    i = 11
    assert json.loads(str(fx["meta"]))["names"][i] == "monsoon_land"
    c = {k[3:]: jnp.asarray(v[i:i + 1]) for k, v in fx.items() if k.startswith("in_")}
    dt = 60.0
    cfg = cb.CLUBBConfig(prognostic=True, clubb_dt=dt / n_sub)
    carry = cb.pack_clubb_moments(cb.init_clubb_moments(1, c["T"].shape[1], cfg))
    seen = []
    original = mf.clubb_mf_forcing

    def spy(col, *args, **kw):
        assert kw.get("bsort") is True, "production bridge omitted bsort=True"
        result = original(col, *args, **kw)
        out = result[2]
        jax.debug.callback(
            lambda z, top, area, heat, water: seen.append(tuple(
                np.asarray(x).copy() for x in (z, top, area, heat, water))),
            kw["ztopm1"], out["ztopm1"], out["upa"], out["thlflx"], out["qtflx"],
            ordered=True)
        return result

    monkeypatch.setattr(mf, "clubb_mf_forcing", spy)
    runs, kernels = [], []
    for memory_m in (0.0, 500.0):
        seen.clear()
        z0 = jnp.full((1, NUP), memory_m)
        with jax.disable_jit():
            out, _ = cb.clubb_turbulence_prognostic(
                c["u"], c["v"], c["T"], c["q_v"], carry,
                c["p_full"], c["p_half"], c["z_full"], c["z_half"],
                c["T_sfc"], c["q_sfc"], c["rho"], dt, cfg,
                sfc_wpthlp=jnp.asarray(fx["c_wthl_sfc"][i:i + 1]),
                sfc_wprtp=jnp.asarray(fx["c_wqt_sfc"][i:i + 1]),
                lat=jnp.zeros(1), mf_pblh=jnp.asarray(fx["c_pblh"][i:i + 1]),
                clubb_mf=True, clubb_mf_bsort=True, mf_ztopm1=z0)
        jax.block_until_ready(out)
        assert len(seen) == n_sub
        np.testing.assert_array_equal(seen[0][0], np.asarray(z0)[0])
        for previous, current in zip(seen, seen[1:]):
            np.testing.assert_array_equal(current[0], previous[1])
        np.testing.assert_array_equal(np.asarray(out.mf_ztopm1)[0], seen[-1][1])
        assert np.any(seen[0][2] > 0.0), "all plumes dead: memory check is vacuous"
        for x in (out.mf_ztopm1, out.dT_dt, out.dq_v_dt):
            assert np.all(np.isfinite(np.asarray(x)))
        runs.append(out)
        kernels.append(list(seen))
    assert not np.array_equal(kernels[0][0][2], kernels[1][0][2]), "memory did not change kernel area"
    assert any(not np.array_equal(kernels[0][0][k], kernels[1][0][k]) for k in (3, 4)), "memory did not change fluxes"
    assert any(not np.array_equal(np.asarray(getattr(runs[0], k)), np.asarray(getattr(runs[1], k)))
               for k in ("dT_dt", "dq_v_dt")), "memory did not reach production tendencies"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-gaps", action="store_true", required=True)
    parser.parse_args()
    fixture = dict(np.load(FIXTURE, allow_pickle=True))
    GAPS.write_text(json.dumps({
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256((REPO / SOURCE).read_bytes()).hexdigest(),
        "names": json.loads(str(fixture["meta"]))["names"],
        "defaults": MFSmoothing()._asdict(),
        "gaps": _gaps(fixture),
    }, indent=1) + "\n")
    print("wrote", GAPS)
    combined = dict(np.load(COMBINED_FIXTURE, allow_pickle=True))
    HARD_GAPS.write_text(json.dumps({
        "fixture_sha256": hashlib.sha256(COMBINED_FIXTURE.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256((REPO / SOURCE).read_bytes()).hexdigest(),
        "names": json.loads(str(combined["meta"]))["names"],
        "gaps": _hard_gaps(combined),
    }, indent=1) + "\n")
    print("wrote", HARD_GAPS)
