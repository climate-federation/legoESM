"""Plane CRM mixed-precision mode (``--precision mixed``).

fp32 state and dynamics; fp64 for radiation inputs, microphysics tracer
inputs, acoustic substeps and dry-mass fixer sums; increment-form SSP-RK3
combinations. Tracers are stored fp32 and carried fp64 through every RK stage
of the mass-conserving transport (#2641), rounded to fp32 once per step.
Run with ``JAX_ENABLE_X64=1`` (the fp64 pieces need x64 on).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "run"))
sys.path.insert(0, str(_REPO / "tests" / "unit"))

jax = pytest.importorskip("jax")
if not jax.config.jax_enable_x64:
    pytest.skip("mixed precision needs JAX_ENABLE_X64=1", allow_module_level=True)

import jax.numpy as jnp  # noqa: E402

import run_rcemip_plane as rcp  # noqa: E402
import test_plane_tracer_mass_conservation as _tm  # noqa: E402
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (  # noqa: E402
    PlaneCompressibleEulerModel,
    make_flat_plane_terrain_metric,
)
from legoesm.grids.plane import create_plane_grid  # noqa: E402
from legoesm.timestepping.split_explicit import (  # noqa: E402
    SplitExplicitConfig,
    split_explicit_step,
)

NX, NLEV, STEPS = 16, 32, 30
TRACER_STEPS = 50
# Tolerances (measured on this setup; votes in the commit body).
DRY_MASS_TOL = 1e-12          # mixed 1.8e-13; fp32 fixer sums 6.6e-12 (voted 3-0)
WATER_VS_FP64_TOL = 1.5e-7    # mixed 3.0e-8 on the #2641 transport
MICRO_WATER_TOL = 1e-12        # mixed 4.2e-16; fp32 tracer inputs 3.2e-6 (pre-fix)
# Tracer mass, 50 steps: mixed max 2.4e-8 (van_leer); tracers rounded to fp32
# every stage with fp32 stage arithmetic 4.1e-7 on the fixer case (voted 3-0).
MIXED_TRACER_MASS_TOL = 1e-7
_FIXER_CASE = {"fix_mass": True, "anchor_mass_to_initial": True,
               "hyperdiff_rho_coeff": 1.0e7, "sponge_coeff": 0.01}
MICRO_FP32_TOL = 1e-7           # fp32 inputs after the satadj fix 5.7e-9; before it 3.2e-6 (voted 3-0)


def _build(precision, *, moisten=1.15, nx=NX, nlev=NLEV):
    argv = (f"--dx 4000 --dt 6 --hyperdiff 1.0e7 --n-acoustic-substeps 6 "
            f"--off-centering 0.1 --smag-cs 0.19 --semi-implicit "
            f"--microphysics morrison --nlev {nlev} --T-sfc 300.0 --nx {nx} "
            f"--ny {nx} --seed-kind band_noise --seed-kmax 3 "
            f"--theta-noise-amp 0.5 --precision {precision}").split()
    a = rcp.parse_args(argv)
    a.hyperdiff = rcp.dx_aware_hyperdiff(a.dx) if a.hyperdiff is None else a.hyperdiff
    dtype = jnp.float64 if precision == "float64" else jnp.float32
    grid = create_plane_grid(nx=a.nx, ny=a.ny, nlev=a.nlev, dx=a.dx, dy=a.dx,
                             dtype=dtype)
    hc, _, q_sfc = rcp.build_rcemip_height_coord(a, rcp.P_SFC_RCEMIP, dtype)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = rcp.build_dycore_config(a)
    state = rcp._build_rcemip_initial_state(
        grid, hc, dtype=dtype, theta_noise_amp=a.theta_noise_amp,
        n_tracers=rcp.microphysics_n_tracers(a.microphysics),
        seed_kind=a.seed_kind, seed_kmax=a.seed_kmax, q_sfc=q_sfc)
    # Near-saturated lower troposphere so the saturation adjustment acts.
    tr = state.tracers.data
    state = state._replace(tracers=state.tracers.replace(
        data=tr.at[..., 0].multiply(moisten)))
    mp = rcp._build_microphysics_config(
        a.microphysics, a.homogeneous_ice_nucleation,
        hard_saturation_adjustment=a.hard_saturation_adjustment,
        hard_sat_adjust_threshold=a.hard_sat_adjust_threshold,
        hard_sat_max_heating_K=a.hard_sat_max_heating_k,
        saturation_sharpness=a.saturation_sharpness)
    physics = rcp.make_rcemip_physics(
        grid, hc, tm, radiation_config=None, microphysics_config=mp, dt=a.dt,
        T_sfc=a.T_sfc, p_sfc=rcp.P_SFC_RCEMIP, surface_flux=False,
        mixed_precision=(precision == "mixed"))
    return a, grid, hc, tm, cfg, state, physics, mp


def _totals(state, hc, dx):
    rho = (np.asarray(hc.rho_ref, np.float64)
           + np.asarray(state.rho_prime.data, np.float64))
    dV = np.asarray(hc.dz, np.float64) * dx * dx
    tr = np.asarray(state.tracers.data, np.float64)
    water = tr[..., 0] + tr[..., 1:6].sum(-1)
    return float((rho * dV).sum()), float((rho * water * dV).sum())


def _run(precision, steps=STEPS, cfg_edit=None):
    a, grid, hc, tm, cfg, state, physics, _ = _build(precision)
    if cfg_edit is not None:
        cfg = cfg_edit(cfg)
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    m0, w0 = _totals(state, hc, a.dx)
    for _ in range(steps):
        state = model.step(state, dt=a.dt, physics_fn=physics)
    m1, w1 = _totals(state, hc, a.dx)
    return state, (m1 - m0) / m0, (w1 - w0) / w0


@pytest.fixture(scope="module")
def runs():
    return {p: _run(p) for p in ("float64", "mixed")}


def test_mixed_mode_selected_by_config_and_holds_fp32_storage(runs):
    a, *_ , cfg, _, _, _ = _build("mixed")
    assert cfg.mixed_precision is True
    state, _, _ = runs["mixed"]
    for name in ("u", "v", "w", "theta_prime", "rho_prime", "tracers"):
        assert getattr(state, name).data.dtype == jnp.float32, name


def test_mixed_dry_mass_conserved(runs):
    _, dm, _ = runs["mixed"]
    assert abs(dm) <= DRY_MASS_TOL, dm


def test_mixed_total_water_matches_fp64(runs):
    _, _, w64 = runs["float64"]
    _, _, wmx = runs["mixed"]
    assert abs(wmx - w64) <= WATER_VS_FP64_TOL, (wmx, w64)


def _mixed_transport_case(**overrides):
    """#2641's tracer-mass case (no physics, no sponge/hyperdiffusion) in
    mixed precision: fp32 grid, reference state and stored fields."""
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        CompressibleEulerConfig)
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        make_rest_state)
    from legoesm.grids.vertical import create_height_coordinate
    cfg = dict(sponge_coeff=0.0, hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
               hyperdiff_w_coeff=0.0, use_coriolis=False, fix_mass=False,
               semi_implicit_acoustic=True, substep_horizontal_acoustic=True,
               mixed_precision=True)
    cfg.update(overrides)
    n, nlev = 8, 10
    hc = create_height_coordinate(nlev, H=20.0e3)
    g64 = create_plane_grid(nx=n, ny=n, nlev=nlev, dx=1e3, dy=1e3,
                            dtype=jnp.float64)
    m64 = PlaneCompressibleEulerModel(
        g64, hc, make_flat_plane_terrain_metric(g64, hc),
        CompressibleEulerConfig(**{**cfg, "mixed_precision": False}))
    s64 = _tm._perturbed(m64, make_rest_state(g64, hc, dtype=jnp.float64),
                         _tm._varying_q((n, n, nlev)))
    hc32 = hc._replace(**{k: v.astype(jnp.float32) for k, v in hc._asdict().items()
                          if isinstance(v, jax.Array)
                          and jnp.issubdtype(v.dtype, jnp.floating)})
    g32 = create_plane_grid(nx=n, ny=n, nlev=nlev, dx=1e3, dy=1e3,
                            dtype=jnp.float32)
    model = PlaneCompressibleEulerModel(
        g32, hc32, make_flat_plane_terrain_metric(g32, hc32),
        CompressibleEulerConfig(**cfg))
    state = jax.tree.map(lambda a: a.astype(jnp.float32)
                         if jnp.issubdtype(a.dtype, jnp.floating) else a, s64)
    return model, state


def _fp64_masses(model, state):
    rho = (np.asarray(model.height_coord.rho_ref, np.float64)
           + np.asarray(state.rho_prime.data, np.float64))
    dz = np.asarray(model.height_coord.dz, np.float64)
    q = np.asarray(state.tracers.data, np.float64)
    return (float((rho * dz).sum()),
            np.array([(rho * q[..., i] * dz).sum() for i in range(q.shape[-1])]))


def _tracer_mass_drift(overrides=None, steps=TRACER_STEPS):
    model, state = _mixed_transport_case(**(overrides or {}))
    d0, m0 = _fp64_masses(model, state)
    for _ in range(steps):
        state = model.step(state, dt=2.0)
    assert state.tracers.data.dtype == jnp.float32
    d1, m1 = _fp64_masses(model, state)
    return float(np.max(np.abs(m1 - m0) / m0)), abs(d1 - d0) / d0


def _fp32_stage_tracers(monkeypatch):
    """The fix reverted: tracers stay fp32 through the step and the
    conservative stage update runs in fp32 (densities, fluxes, products),
    rounding the tracers at every RK stage."""
    import legoesm.atmosphere.dynamics.les.compressible_euler_plane as cep
    from legoesm.timestepping.split_explicit import split_explicit_step as _se

    def _split_step(self, state, slow_fn, ac_fn, dt, se_config, tracer_mass):
        tdt = state.tracers.data.dtype
        tm = tracer_mass

        def finish(base, ev, slow, out, tend, dte):
            if not isinstance(out, cep.PlaneNonHydrostaticState):
                out = (out[0], jax.tree.map(lambda a: a.astype(tdt), out[1]))
            r = tm.finish_stage(base, ev, slow, out, tend, dte)
            return r._replace(tracers=r.tracers.replace(
                data=r.tracers.data.astype(tdt)))

        hooks = tm._replace(
            with_tracers=lambda s, q: s._replace(
                tracers=s.tracers.replace(data=q.astype(tdt))),
            finish_stage=finish)
        return _se(state, lambda x: cep._cast_like(slow_fn(x), x),
                   cep._acoustic_fp64(ac_fn), dt, se_config, tracer_mass=hooks)

    monkeypatch.setattr(cep.PlaneCompressibleEulerModel, "_split_step",
                        _split_step)


@pytest.mark.parametrize("overrides", [
    {}, {"horizontal_advection_scheme": "van_leer",
         "vertical_tracer_advection": "van_leer"},
    _FIXER_CASE])
def test_mixed_tracer_mass_conserved(overrides):
    """RULE 0: tracer mass sum(rho q dV) of the fp32-stored state (summed in
    fp64) holds to fp32 storage round-off over the steps."""
    rel, _ = _tracer_mass_drift(overrides)
    assert rel <= MIXED_TRACER_MASS_TOL, rel


def test_mixed_tracer_mass_needs_fp64_stage_tracers(monkeypatch):
    """Non-vacuity: with tracers rounded to fp32 at every stage and fp32
    stage arithmetic (the fix reverted), the fixer case fails the budget.
    Without the fixer both forms sit at fp32 storage round-off (~1e-8)."""
    _fp32_stage_tracers(monkeypatch)
    rel, _ = _tracer_mass_drift(_FIXER_CASE)
    assert rel > MIXED_TRACER_MASS_TOL, rel


def _micro_water_tendency(precision, fp64_inputs):
    a, grid, hc, tm, cfg, state, _, mp = _build(precision, moisten=1.2)
    mic = rcp.make_microphysics_physics(mp, model_type="plane", dt=a.dt)
    if fp64_inputs:
        mic = rcp._with_fp64_tracers(mic)
    t = jax.jit(lambda s: mic(s, grid, hc, tm))(state)
    rho = (np.asarray(hc.rho_ref, np.float64)
           + np.asarray(state.rho_prime.data, np.float64))
    dq = np.asarray(t.dtracers_dt.data, np.float64)
    dV = np.asarray(hc.dz, np.float64)
    # Water tendency summed over the domain, and the vapour-exchange scale.
    return (float((rho * dq[..., :6].sum(-1) * dV).sum()),
            float((rho * np.abs(dq[..., 0]) * dV).sum()))


def test_microphysics_water_closure_with_fp64_tracer_inputs():
    """RULE 0: Morrison's saturation adjustment forms q_v = (q_v + q_c) - q_c_new;
    with fp32 tracer inputs that rounds at the vapour scale and creates water.
    The mixed mode feeds fp64 tracers; its domain water tendency must match
    fp64 and the fp32-input form must not."""
    ref, scale = _micro_water_tendency("float64", fp64_inputs=False)
    mixed, _ = _micro_water_tendency("mixed", fp64_inputs=True)
    fp32_in, _ = _micro_water_tendency("mixed", fp64_inputs=False)
    tol = MICRO_WATER_TOL * scale
    assert abs(mixed - ref) <= tol, (mixed, ref, scale)
    assert abs(fp32_in - ref) > tol, (fp32_in, ref, scale)


def test_microphysics_water_closure_with_fp32_inputs_after_satadj_fix():
    """RULE 0, fp32 closure: the saturation adjustment moves ONE amount between
    vapour and cloud, so fp32 tracer inputs no longer create water there
    (pre-fix: 3.2e-6 of the vapour-exchange scale)."""
    ref, scale = _micro_water_tendency("float64", fp64_inputs=False)
    fp32_in, _ = _micro_water_tendency("mixed", fp64_inputs=False)
    assert abs(fp32_in - ref) <= MICRO_FP32_TOL * scale, ((fp32_in - ref) / scale)


def test_rounding_safe_rk3_equals_plain_in_fp64():
    a, grid, hc, tm, cfg, state, physics, _ = _build("float64")
    plain = PlaneCompressibleEulerModel(grid, hc, tm, cfg).step(
        state, dt=a.dt, physics_fn=physics)
    safe = PlaneCompressibleEulerModel(
        grid, hc, tm, cfg._replace(mixed_precision=True)).step(
        state, dt=a.dt, physics_fn=physics)
    for name in ("u", "w", "theta_prime", "rho_prime", "tracers"):
        x = np.asarray(getattr(plain, name).data)
        y = np.asarray(getattr(safe, name).data)
        np.testing.assert_allclose(y, x, rtol=1e-11, atol=1e-12 * np.abs(x).max())


def test_rounding_safe_combinations_reject_other_integrators():
    a, grid, hc, tm, cfg, state, _, _ = _build("float64")
    with pytest.raises(ValueError, match="ssp_rk3 only"):
        split_explicit_step(
            state, lambda s: s, lambda s, *_: s, a.dt,
            SplitExplicitConfig(outer_integrator="ssp_rk54",
                                rounding_safe_combinations=True))


def test_mixed_step_gradient_flows_and_matches_fp64():
    """jax.grad through one mixed dynamics step (with its fp64 fixer sums) is
    finite, nonzero and close to fp64. Dynamics only: reverse mode through
    Morrison at this size does not fit a compile budget."""
    grads = {}
    for precision in ("float64", "mixed"):
        a, grid, hc, tm, cfg, state, _, _ = _build(precision, nx=8, nlev=16)
        model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
        model.precompute_target_mass(state)

        def loss(theta, model=model, state=state, a=a):
            s = state._replace(theta_prime=state.theta_prime.replace(data=theta))
            out = model.step(s, dt=a.dt)
            return jnp.sum(out.theta_prime.data.astype(jnp.float64) ** 2)

        grads[precision] = np.asarray(
            jax.grad(loss)(state.theta_prime.data), np.float64)
    g32, g64 = grads["mixed"], grads["float64"]
    assert np.all(np.isfinite(g32)) and np.abs(g32).max() > 0.0
    rel = np.linalg.norm(g32 - g64) / np.linalg.norm(g64)
    if os.environ.get("LEGOESM_MEASURE_MIXED"):
        print(f"\nGRAD rel={rel:.3e}")
    assert rel < 1e-5, rel


def test_mixed_refuses_fp32_env(monkeypatch):
    monkeypatch.setenv("LEGOESM_RCEMIP_PLANE_FP32", "1")
    monkeypatch.setattr(sys, "argv", [
        "run_rcemip_plane.py", "--precision", "mixed", "--nx", "4", "--ny", "4",
        "--nlev", "8", "--steps", "1", "--no-physics"])
    with pytest.raises(SystemExit, match="x64 ON"):
        rcp.main()


if os.environ.get("LEGOESM_MEASURE_MIXED"):
    def test_print_measurements():
        ref, scale = _micro_water_tendency("float64", fp64_inputs=False)
        mixed, _ = _micro_water_tendency("mixed", fp64_inputs=True)
        fp32_in, _ = _micro_water_tendency("mixed", fp64_inputs=False)
        print(f"\nMICRO ref={ref:.6e} scale={scale:.6e} "
              f"mixed-ref={(mixed - ref) / scale:.3e} fp32in-ref={(fp32_in - ref) / scale:.3e}")


def test_mixed_refuses_x64_off():
    import legoesm.atmosphere.dynamics.les.compressible_euler_plane as cep
    _, grid, hc, tm, cfg = _build("mixed", nx=8, nlev=16)[:5]
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="jax_enable_x64"):
            cep.PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    finally:
        jax.config.update("jax_enable_x64", True)
