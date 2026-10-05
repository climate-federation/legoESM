"""Physics cadence (``physics_update_steps``): the CAM-style held-physics
step on the MPAS lane.

Contract under test:
  * cadence "off" and cadence "write" produce byte-identical tendencies and
    carries (``physics_update_steps=1`` cannot change a trajectory);
  * the "held" variant returns the cache written by the last physics step,
    leaf-for-leaf, independent of the state it is handed, and refuses to run
    on an empty cache;
  * the seeded cache has the SAME pytree structure as a real physics step's
    output, so the carry never changes structure (no retrace); the held step
    is a second compiled ``model.step``, not a third or fourth;
  * on a held ``model.step`` the published surface diagnostics and ledger rows
    ARE the cached ones (accumulators integrate the applied rates, not zeros);
  * config validation and the CLI flag round trip.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.physics_state import (
    PHYSSTATE_INPUT_FIELDS,
    PhysicsState,
    init_physics_state,
    update_physics_state,
)


def _cfg():
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    return PhysicsConfig(turbulence=TurbulenceConfig(scheme="tke"))


def _setup(n=3, nlev=8, seed=42):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
        MPASPrimitiveEquationModel,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_mpas
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(n)
    sigma = create_sigma_coordinate(nlev)
    model = MPASPrimitiveEquationModel(mesh, sigma, MPASPrimitiveEquationConfig())
    state = held_suarez_init_mpas(mesh, sigma, seed=seed)
    # dtype-stable carry: the f32 idealized init is promoted to f64 by the
    # x64 step, which would retrace on step 2 and confound the compile count
    state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), state)
    return mesh, sigma, model, state


def _variants(cfg, dt=300.0, n_steps=1, **kw):
    from legoesm.atmosphere.physics.combined import held_physics_variant, make_physics
    fn_off = make_physics(cfg, model_type="mpas", dt=dt, **kw)
    fn_write = make_physics(cfg, model_type="mpas", dt=dt, physics_cadence="write",
                            physics_cadence_steps=n_steps, **kw)
    return fn_off, fn_write, held_physics_variant(fn_write)


_STATE_FIELDS = ("du_dt", "dv_dt", "dT_dt", "dp_s_dt", "dphis_dt", "tracer_tendencies")


def _diag_only(t):
    """The tendency with its state fields dropped (the diagnostic rates)."""
    return t._replace(**{f: None for f in _STATE_FIELDS})


def _state_only(t):
    return {f: getattr(t, f) for f in _STATE_FIELDS if getattr(t, f) is not None}


def _leaves_equal(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def _seeded(fn_write, state, mesh, sigma, cfg):
    from legoesm.atmosphere.physics.combined import physics_cache_seed
    ncol, nlev = state.T.data.shape
    ps = init_physics_state(int(ncol), int(nlev), cfg)
    return ps, ps._replace(held_physics=physics_cache_seed(
        fn_write, state, mesh, sigma, ps))


def test_held_physics_is_a_per_job_input_field():
    assert "held_physics" in PhysicsState._fields
    assert "held_physics" in PHYSSTATE_INPUT_FIELDS
    ps = init_physics_state(4, 3, _cfg())
    assert ps.held_physics is None
    tagged = ps._replace(held_physics=("x",))
    assert update_physics_state(tagged, {}).held_physics == ("x",)
    assert update_physics_state(tagged, {"held_physics": None}).held_physics is None


def test_off_and_write_are_byte_identical_and_seed_matches_structure():
    mesh, sigma, model, state = _setup()
    cfg = _cfg()
    fn_off, fn_write, fn_held = _variants(cfg, budget_ledger=True)
    ps, ps_seeded = _seeded(fn_write, state, mesh, sigma, cfg)

    t_off, ps_off = fn_off(state, mesh, sigma, phys_state=ps)
    t_w, ps_w = fn_write(state, mesh, sigma, phys_state=ps_seeded)
    assert _leaves_equal(t_off, t_w)
    assert _leaves_equal(ps_off._replace(held_physics=None),
                         ps_w._replace(held_physics=None))
    assert ps_off.held_physics is None
    # cache == this step's output, in the seed's structure (no retrace)
    assert _leaves_equal(ps_w.held_physics, t_w)
    assert (jax.tree_util.tree_structure(ps_w.held_physics)
            == jax.tree_util.tree_structure(ps_seeded.held_physics))
    for a, b in zip(jax.tree_util.tree_leaves(ps_w.held_physics),
                    jax.tree_util.tree_leaves(ps_seeded.held_physics)):
        assert a.shape == b.shape and a.dtype == b.dtype
    assert t_w.ledger_rows is not None  # the ledger rides the cache too

    # held: the cached DIAGNOSTIC rates leaf-for-leaf, whatever state it is
    # handed, and ZERO state tendency (the increment went in on the write step)
    other = state._replace(T=state.T.replace(data=state.T.data + 5.0))
    t_h, ps_h = fn_held(other, mesh, sigma, phys_state=ps_w)
    assert _leaves_equal(_diag_only(t_h), _diag_only(ps_w.held_physics))
    for f in jax.tree_util.tree_leaves(_state_only(t_h)):
        assert not np.any(np.asarray(f))
    assert np.any(np.asarray(ps_w.held_physics.dT_dt.data))  # non-vacuous
    assert _leaves_equal(ps_h, ps_w)
    assert not _leaves_equal(t_h, fn_off(other, mesh, sigma, phys_state=ps)[0])
    # non-vacuity: an empty cache is refused
    with pytest.raises(ValueError, match="held_physics"):
        fn_held(state, mesh, sigma, phys_state=ps)
    for fn in (fn_write, fn_held):
        assert fn._requires_phys_state is True
    assert fn_held._column_local is True


def test_merge_keeps_structure_and_fills_missing_from_previous():
    from legoesm.atmosphere.physics.combined import merge_physics_cache
    mesh, sigma, model, state = _setup()
    cfg = _cfg()
    _, fn_write, _ = _variants(cfg)
    ps, ps_seeded = _seeded(fn_write, state, mesh, sigma, cfg)
    t, _ = fn_write(state, mesh, sigma, phys_state=ps_seeded)
    assert t.sw_net_sfc is not None and t.shflx_sfc is not None
    partial = t._replace(sw_net_sfc=None, shflx_sfc=None)
    prev = ps_seeded.held_physics
    merged = merge_physics_cache(prev, partial)
    assert type(merged) is type(prev)
    assert _leaves_equal(merged.dT_dt, t.dT_dt)
    assert _leaves_equal(merged.tracer_tendencies, t.tracer_tendencies)
    assert merged.shflx_sfc is prev.shflx_sfc
    assert merged.sw_net_sfc is prev.sw_net_sfc
    with pytest.raises(ValueError, match="shape"):
        merge_physics_cache(prev, t._replace(dT_dt=t.dT_dt.replace(data=t.dT_dt.data[:1])))
    with pytest.raises(ValueError, match="tracer set"):
        merge_physics_cache(prev, t._replace(tracer_tendencies={}))
    # an unseeded write is refused (the carry structure must be fixed first)
    with pytest.raises(ValueError, match="unseeded"):
        fn_write(state, mesh, sigma, phys_state=ps)
    # writes are cast to the seed's dtype: the carry never changes structure
    f32 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), t)
    m32 = merge_physics_cache(prev, f32)
    assert all(a.dtype == b.dtype for a, b in zip(
        jax.tree_util.tree_leaves(m32), jax.tree_util.tree_leaves(prev)))
    # a suite with no carry of its own still gets one to hold the cache
    with pytest.raises(ValueError, match="phys_state=None"):
        fn_write(state, mesh, sigma, phys_state=None)
    with pytest.raises(ValueError, match="seed"):
        merge_physics_cache(prev._replace(dT_dt=None), t)


def test_model_step_held_publishes_cached_diagnostics_and_compiles_once():
    mesh, sigma, model, state = _setup()
    cfg = _cfg()
    dt = 300.0
    fn_off, fn_write, fn_held = _variants(cfg, dt=dt, budget_ledger=True)
    ps, ps_seeded = _seeded(fn_write, state, mesh, sigma, cfg)

    # "off" (what physics_update_steps=1 builds: the pre-existing function,
    # no wrapper, hence byte-identical by construction) vs "write" over 3
    # jitted model steps: same physics; the extra cache outputs change XLA's
    # fusion so the jitted trajectory agrees to rounding (1 ulp), not bits.
    s_off, s_w, p_off, p_w = state, state, ps, ps_seeded
    for _ in range(3):
        s_off = model.step(s_off, dt, physics_fn=fn_off, phys_state=p_off)
        p_off = model._phys_state
        s_w = model.step(s_w, dt, physics_fn=fn_write, phys_state=p_w)
        p_w = model._phys_state
    for a, b in zip(jax.tree_util.tree_leaves(s_off), jax.tree_util.tree_leaves(s_w)):
        b = np.asarray(b)
        np.testing.assert_allclose(np.asarray(a), b, rtol=0,
                                   atol=1e-12 * max(float(np.max(np.abs(b))), 1.0))
    assert _leaves_equal(p_off._replace(held_physics=None),
                         p_w._replace(held_physics=None))

    # cadence 2 over 4 steps: held steps re-publish the cached diagnostics
    fn_off, fn_write, fn_held = _variants(cfg, dt=dt, n_steps=2, budget_ledger=True)
    ps, ps_seeded = _seeded(fn_write, state, mesh, sigma, cfg)
    s, p = state, ps_seeded
    n0 = model._step_jit._cache_size()
    sizes = []
    for step in range(6):
        fn = fn_write if step % 2 == 0 else fn_held
        s = model.step(s, dt, physics_fn=fn, phys_state=p)
        p_prev, p = p, model._phys_state
        sizes.append(model._step_jit._cache_size() - n0)
        cache = p.held_physics
        if step % 2 == 1:
            assert _leaves_equal(p, p_prev)
            # the published step ledger = cached PHYSICS rows + the dycore's
            # own per-step dynamics / clip / closure rows
            from legoesm.diagnostics.process_ledger import (
                ROW_CLIPS,
                ROW_DYNAMICS,
                ROW_OTHER,
            )
            led = np.asarray(model._step_ledger)
            phys_rows = [r for r in range(led.shape[1])
                         if r not in (ROW_CLIPS, ROW_DYNAMICS, ROW_OTHER)]
            assert np.array_equal(led[:, phys_rows, :],
                                  np.asarray(cache.ledger_rows)[:, phys_rows, :])
            # published slots: 0 = sw_net_sfc, 6 = shflx_sfc (core.state map)
            def _arr(x):
                return np.asarray(getattr(x, "data", x))
            assert np.array_equal(_arr(model._sfc_diag[0]), _arr(cache.sw_net_sfc))
            assert np.array_equal(_arr(model._sfc_diag[6]), _arr(cache.shflx_sfc))
        else:
            assert not _leaves_equal(cache, p_prev.held_physics)
    # Compiles per step, pinned: write on the seeded carry (0), held (1),
    # write again on the post-step carry (2: ``init_physics_state`` seeds
    # ``tke`` weak-typed, the first step returns it strong-typed -- a
    # pre-existing one-off retrace, identical on the held-rate design), then
    # NO further compile: both variants are reused, never a per-step retrace.
    assert sizes == [1, 2, 3, 3, 3, 3]


def _mpas_cfg(**kw):
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )
    kw.setdefault("radiation", "gray")
    return ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=1, nlev=8,
                        vertical_coord="sigma"),
        dycore=DycoreConfig(dt=600.0, discretization="mpas"),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical", **kw,
    )


def test_validate_strict_physics_update_steps():
    _mpas_cfg().validate_strict()
    _mpas_cfg(physics_update_steps=4, rad_update_steps=8).validate_strict()
    _mpas_cfg(physics_update_steps=4, rad_update_steps=4).validate_strict()
    for bad in (dict(physics_update_steps=0),
                dict(physics_update_steps=2.0),
                dict(physics_update_steps=4, rad_update_steps=1),
                dict(physics_update_steps=4, rad_update_steps=6)):
        with pytest.raises(ValueError, match="physics_update_steps"):
            _mpas_cfg(**bad).validate_strict()
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )
    cube = ExperimentConfig(
        grid=GridConfig(resolution=8, nlev=8), dycore=DycoreConfig(dt=600.0),
        output=OutputConfig(diag_days=1), days=1, dataset="analytical",
        radiation="gray", physics_update_steps=4, rad_update_steps=4)
    with pytest.raises(ValueError, match="MPAS lane only"):
        cube.validate_strict()
    from legoesm.atmosphere.physics.combined import make_physics
    with pytest.raises(ValueError, match="physics_cadence"):
        make_physics(_cfg(), model_type="hydrostatic", dt=1.0, physics_cadence="write")
    with pytest.raises(ValueError, match="physics_cadence"):
        make_physics(_cfg(), model_type="mpas", dt=1.0, physics_cadence="held")
    # cadence "off" leaves every other lane's dispatch intact (codex round 1)
    assert callable(make_physics(_cfg(), model_type="hydrostatic", dt=1.0))
    assert make_physics(_cfg(), model_type="hydrostatic", dt=1.0)._requires_phys_state
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    gray_only = PhysicsConfig()
    assert not make_physics(gray_only, model_type="mpas", dt=1.0)._requires_phys_state
    assert make_physics(gray_only, model_type="mpas", dt=1.0,
                        physics_cadence="write")._requires_phys_state


def test_physics_update_steps_round_trips_cli_and_amip_config():
    from legoesm.driver.config import ExperimentConfig
    from legoesm.forcing.amip_config import AMIPExperimentConfig

    from scripts.run.run_amip import build_arg_parser
    parser = build_arg_parser()
    assert parser.parse_args([]).physics_update_steps == 1
    assert parser.parse_args(["--physics-update-steps", "16"]).physics_update_steps == 16
    amip = AMIPExperimentConfig(physics_update_steps=16, rad_update_steps=32)
    cfg = ExperimentConfig.from_amip_config(amip)
    assert cfg.physics_update_steps == 16
    assert cfg.to_amip_config().physics_update_steps == 16


def test_write_step_scales_state_tendencies_by_n_and_caches_the_rates():
    mesh, sigma, model, state = _setup()
    cfg = _cfg()
    fn_off, fn_w1, _ = _variants(cfg)
    _, fn_w4, fn_h4 = _variants(cfg, n_steps=4)
    ps, ps_seeded = _seeded(fn_w4, state, mesh, sigma, cfg)
    t_off, _ = fn_off(state, mesh, sigma, phys_state=ps)
    t_w1, _ = fn_w1(state, mesh, sigma, phys_state=ps_seeded)
    t_w4, ps_w4 = fn_w4(state, mesh, sigma, phys_state=ps_seeded)
    # n=1 is the off tendency bit for bit; n=4: state fields x4, diagnostics untouched
    assert _leaves_equal(t_w1, t_off)
    assert _leaves_equal(_diag_only(t_w4), _diag_only(t_off))
    for a, b in zip(jax.tree_util.tree_leaves(_state_only(t_w4)),
                    jax.tree_util.tree_leaves(_state_only(t_off))):
        assert np.array_equal(np.asarray(a), 4.0 * np.asarray(b))
    assert np.any(np.asarray(t_off.dT_dt.data))
    # the cache holds the UNSCALED rates (window-mean diagnostics)
    assert _leaves_equal(ps_w4.held_physics, t_off)
    # held applies zero, publishes the rates
    t_h, _ = fn_h4(state, mesh, sigma, phys_state=ps_w4)
    assert _leaves_equal(_diag_only(t_h), _diag_only(t_off))
    assert not np.any(np.asarray(t_h.dT_dt.data))
    from legoesm.atmosphere.physics.combined import make_physics
    with pytest.raises(ValueError, match="physics_cadence_steps"):
        make_physics(cfg, model_type="mpas", dt=1.0, physics_cadence_steps=4)
    with pytest.raises(ValueError, match="physics_cadence_steps"):
        make_physics(cfg, model_type="mpas", dt=1.0, physics_cadence="write",
                     physics_cadence_steps=0)


def test_window_increment_is_atomic_not_a_held_rate():
    """CAM-FV: the physics increment over a window is decided on the physics
    step alone.  Toy physics whose rate depends on the state it is handed:
    with the atomic increment, what the dynamics does to the state on the held
    steps cannot change the physics increment; the old held-RATE design
    re-applied the rate every step (same total only because the rate was
    frozen -- here we check the increment lands ONCE, on the write step)."""
    from legoesm.atmosphere.physics.combined import _scale_state_tendencies
    mesh, sigma, model, state = _setup()
    cfg = _cfg()
    n = 4
    _, fn_w, fn_h = _variants(cfg, n_steps=n)
    ps, ps_seeded = _seeded(fn_w, state, mesh, sigma, cfg)
    t_w, ps_w = fn_w(state, mesh, sigma, phys_state=ps_seeded)
    rate = ps_w.held_physics
    # per step: the write step carries the WHOLE window increment dt*(N*rate),
    # every held step exactly zero (a held-RATE design gives dt*rate each step
    # and passes a total-only check -- so each step is pinned separately)
    dt = 300.0
    r = np.asarray(rate.dT_dt.data)
    assert np.any(r)
    np.testing.assert_array_equal(dt * np.asarray(t_w.dT_dt.data), n * dt * r)
    total = dt * np.asarray(t_w.dT_dt.data)
    for _ in range(n - 1):
        t_h, ps_w = fn_h(state, mesh, sigma, phys_state=ps_w)
        assert not np.any(np.asarray(t_h.dT_dt.data))
        for f in jax.tree_util.tree_leaves(_state_only(t_h)):
            assert not np.any(np.asarray(f))
        total = total + dt * np.asarray(t_h.dT_dt.data)
    np.testing.assert_array_equal(total, n * dt * r)
    # the helper is exact: x0 gives zeros even where the rate is NaN
    nan_t = rate._replace(dT_dt=rate.dT_dt.replace(
        data=jnp.full_like(rate.dT_dt.data, jnp.nan)))
    assert not np.any(np.isnan(np.asarray(
        _scale_state_tendencies(nan_t, 0).dT_dt.data)))


def _mpas_cadence_driver(tmpdir, days, n_steps=2):
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )
    from legoesm.driver.model_driver import ModelDriver
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=2, nlev=8, vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=300.0),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=days, dataset="analytical", radiation="gray",
        convection="none", turbulence="tke", precision="fp64", distributed=False,
        physics_update_steps=n_steps, rad_update_steps=n_steps,
    )
    d = ModelDriver(cfg, output_dir=tmpdir)
    d.setup()
    return d


def test_mpas_run_refuses_a_restart_inside_a_physics_window(tmp_path):
    """The held cache is per-job and step 0 of a job applies a whole physics
    increment, so resuming from a step that is not a window boundary would
    apply an increment the uninterrupted run did not (codex, 2026-09-22)."""
    two_steps = 601.0 / 86400.0
    d = _mpas_cadence_driver(str(tmp_path / "mid"), two_steps)
    with pytest.raises(ValueError, match="physics-window boundary"):
        d.run(start_step=1, start_day=1.0)
    d = _mpas_cadence_driver(str(tmp_path / "boundary"), two_steps)
    assert d.run(start_step=2, start_day=1.0) == "COMPLETED"
    # a run length that would END mid-window (3 steps at cadence 2) is refused
    # up front: its final checkpoint could never be resumed
    d = _mpas_cadence_driver(str(tmp_path / "odd"), 901.0 / 86400.0)
    with pytest.raises(ValueError, match="window boundary"):
        d.run()
