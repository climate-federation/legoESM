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
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
        MPASPrimitiveEquationModel,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_mpas
    mesh = create_voronoi_mesh(n)
    sigma = create_sigma_coordinate(nlev)
    model = MPASPrimitiveEquationModel(mesh, sigma, MPASPrimitiveEquationConfig())
    state = held_suarez_init_mpas(mesh, sigma, seed=seed)
    # dtype-stable carry: the f32 idealized init is promoted to f64 by the
    # x64 step, which would retrace on step 2 and confound the compile count
    state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), state)
    return mesh, sigma, model, state


def _variants(cfg, dt=300.0, **kw):
    from legoesm.atmosphere.physics.combined import held_physics_variant, make_physics
    fn_off = make_physics(cfg, model_type="mpas", dt=dt, **kw)
    fn_write = make_physics(cfg, model_type="mpas", dt=dt, physics_cadence="write", **kw)
    return fn_off, fn_write, held_physics_variant(fn_write)


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

    # held: returns the cache leaf-for-leaf, whatever state it is handed
    other = state._replace(T=state.T.replace(data=state.T.data + 5.0))
    t_h, ps_h = fn_held(other, mesh, sigma, phys_state=ps_w)
    assert _leaves_equal(t_h, ps_w.held_physics)
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

    # cadence 2 over 4 steps: held steps re-publish the cache
    n_before = model._step_jit._cache_size()
    s, p = state, ps_seeded
    for step in range(4):
        fn = fn_write if step % 2 == 0 else fn_held
        s = model.step(s, dt, physics_fn=fn, phys_state=p)
        p_prev, p = p, model._phys_state
        cache = p.held_physics
        if step % 2 == 1:
            assert _leaves_equal(p, p_prev)
            # the published step ledger = cached PHYSICS rows + the dycore's
            # own per-step dynamics / clip / closure rows
            from legoesm.diagnostics.process_ledger import (
                ROW_CLIPS, ROW_DYNAMICS, ROW_OTHER,
            )
            led = np.asarray(model._step_ledger)
            phys_rows = [r for r in range(led.shape[1])
                         if r not in (ROW_CLIPS, ROW_DYNAMICS, ROW_OTHER)]
            assert np.array_equal(led[:, phys_rows, :],
                                  np.asarray(cache.ledger_rows)[:, phys_rows, :])
            # published slots: 0 = sw_net_sfc, 6 = shflx_sfc (core.state map)
            _arr = lambda x: np.asarray(getattr(x, "data", x))
            assert np.array_equal(_arr(model._sfc_diag[0]), _arr(cache.sw_net_sfc))
            assert np.array_equal(_arr(model._sfc_diag[6]), _arr(cache.shflx_sfc))
        else:
            assert not _leaves_equal(cache, p_prev.held_physics)
    # write + held = exactly two compiled steps (fn_write already compiled
    # above; the held variant adds ONE), never a per-step retrace
    assert model._step_jit._cache_size() == n_before + 1


def _mpas_cfg(**kw):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
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
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
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
