"""CAM6 macrophysics/microphysics sub-cycle (``cld_macmic_num_steps``) on the
hydrostatic/MPAS combined physics.

Contract under test (CAM ``tphysbc`` macmic loop, physpkg.F90):
  * N=1 is the pre-existing parallel split (default; byte-identical);
  * N>1 equals a hand-written sequential reference: convection applied to
    an intermediate state first, then N turbulence->microphysics sub-steps at
    dt/N on the running state (each increment applied before the next), the
    sub-cycled tendencies / precip / carries averaged (CAM's 1/N scaling),
    radiation and GWD once on the pre-physics state;
  * the sub-cycle is NOT a rate re-scaling: N=3 differs from N=1 (the
    intermediate state matters), jit parity, finite gradient;
  * the held-radiation variant, the cadence cache and the ledger rows all
    route through the same loop (rows still sum to the applied total);
  * config validation, CLI flag and AMIP-config round trip.
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, "tests/unit")
from legoesm.atmosphere.physics.physics_state import (  # noqa: E402
    PHYSSTATE_PER_CALL_INPUTS,
    PhysicsState,
    init_physics_state,
    update_physics_state,
)
from test_physics_cadence import _leaves_equal, _mpas_cfg, _setup  # noqa: E402

_X64 = bool(jax.config.jax_enable_x64)
# summation-order noise of the N-term averages at the running dtype
_RTOL = 1e-11 if _X64 else 3e-5


def _cfg(rad="none", conv="sbm"):
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.convection import ConvectionConfig
    from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
    from legoesm.atmosphere.physics.radiation import RadiationConfig
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    return PhysicsConfig(
        radiation=RadiationConfig(scheme=rad),
        convection=ConvectionConfig(scheme=conv),
        turbulence=TurbulenceConfig(scheme="tke"),
        microphysics=MicrophysicsConfig(scheme="kessler"),
    )


def _moist_setup():
    mesh, sigma, model, state = _setup()
    ncol, _ = state.T.data.shape
    sig = jnp.asarray(sigma.sigma_full)[None, :] * jnp.ones((ncol, 1))
    tr = {"q_v": 0.03 * sig ** 3, "q_c": 2.0e-4 * sig ** 2, "q_r": 1.0e-4 * sig ** 2}
    # a warm, moist boundary layer so the convection scheme is ACTIVE (the
    # dry Held-Suarez column leaves sbm at exactly zero)
    state = state._replace(
        T=state.T.replace(data=state.T.data + 20.0 * sig ** 8),
        tracers={k: state.T.replace(data=v, name=k, units="kg/kg") for k, v in tr.items()})
    return mesh, sigma, state


def _advance(state, t, dt):
    kw = dict(u=state.u.replace(data=state.u.data + dt * t.du_dt.data),
              T=state.T.replace(data=state.T.data + dt * t.dT_dt.data),
              p_s=state.p_s.replace(data=state.p_s.data + dt * t.dp_s_dt.data))
    if t.tracer_tendencies:
        kw["tracers"] = {k: (f.replace(data=f.data + dt * t.tracer_tendencies[k].data)
                             if k in t.tracer_tendencies else f)
                         for k, f in state.tracers.items()}
    return state._replace(**kw)


def _reference(cfg, state, mesh, sigma, ps, dt, n):
    """CAM tphysbc order from the standalone module factories."""
    from legoesm.atmosphere.physics.convection.integration import make_convection_physics
    from legoesm.atmosphere.physics.microphysics.integration import make_microphysics_physics
    from legoesm.atmosphere.physics.turbulence.integration import make_turbulence_physics
    conv = make_convection_physics(cfg.convection, "mpas", dt)
    turb = make_turbulence_physics(cfg.turbulence, "mpas", dt / n)
    micro = make_microphysics_physics(cfg.microphysics, "mpas", dt / n)
    t_c, upd_c = conv(state, mesh, sigma, phys_state=ps)
    assert float(jnp.abs(t_c.dT_dt.data).max()) > 0.0   # the "pre" increment is live
    state_m = _advance(state, t_c, dt)
    ps_m = update_physics_state(ps, (upd_c if isinstance(upd_c, dict)
                                     else {} if upd_c is None
                                     else {"conv_prog_profile": upd_c}))
    dT = t_c.dT_dt.data
    du = t_c.du_dt.data
    dq = {k: v.data for k, v in t_c.tracer_tendencies.items()}
    precip = 0.0
    for _ in range(n):
        t_t, tke = turb(state_m, mesh, sigma, phys_state=ps_m)
        state_m = _advance(state_m, t_t, dt / n)
        ps_m = update_physics_state(ps_m, {"tke": tke})
        t_m = micro(state_m, mesh, sigma)
        state_m = _advance(state_m, t_m, dt / n)
        dT = dT + (t_t.dT_dt.data + t_m.dT_dt.data) / n
        du = du + (t_t.du_dt.data + t_m.du_dt.data) / n
        for t in (t_t, t_m):
            for k, v in (t.tracer_tendencies or {}).items():
                dq[k] = dq.get(k, 0.0) + v.data / n
        precip = precip + t_m.precip.data / n
    return dT, du, dq, precip, ps_m


# Column sums of the N=1 (default) tendency on the moist fixture, computed at
# the PARENT commit 0870fdb52 (before the macmic loop existed), CPU x64:
# sum dT_dt, sum |dT_dt|, sum dq_v_dt, sum dq_r_dt.  The N=1 path must keep
# reproducing them (codex round 3: default-vs-explicit-1 was tautological).
_N1_PARENT_PINS = (-0.4682190773295994, 0.547585380929818,
                   0.00019821679962792423, -1.4502397831824815e-05)


def test_n1_default_is_the_parallel_split():
    from legoesm.atmosphere.physics.combined import make_physics
    mesh, sigma, state = _moist_setup()
    cfg = _cfg()
    ps = init_physics_state(*state.T.data.shape, cfg)
    t0, p0 = make_physics(cfg, model_type="mpas", dt=1800.0)(state, mesh, sigma, phys_state=ps)
    t1, p1 = make_physics(cfg, model_type="mpas", dt=1800.0, cld_macmic_num_steps=1)(
        state, mesh, sigma, phys_state=ps)
    assert _leaves_equal(t0, t1) and _leaves_equal(p0, p1)
    got = (float(jnp.sum(t0.dT_dt.data)), float(jnp.sum(jnp.abs(t0.dT_dt.data))),
           float(jnp.sum(t0.tracer_tendencies["q_v"].data)),
           float(jnp.sum(t0.tracer_tendencies["q_r"].data)))
    if _X64 and jax.default_backend() == "cpu":
        # CPU-x64 pins (summation order); f32 differs from them by ~2 %
        np.testing.assert_allclose(got, _N1_PARENT_PINS, rtol=1e-10)


def test_macmic_matches_the_sequential_reference_and_is_not_a_rescaling():
    from legoesm.atmosphere.physics.combined import make_physics
    mesh, sigma, state = _moist_setup()
    cfg = _cfg()
    dt, n = 1800.0, 3
    ps = init_physics_state(*state.T.data.shape, cfg)
    f1 = make_physics(cfg, model_type="mpas", dt=dt)
    f3 = make_physics(cfg, model_type="mpas", dt=dt, cld_macmic_num_steps=n)
    t1, _ = f1(state, mesh, sigma, phys_state=ps)
    t3, p3 = f3(state, mesh, sigma, phys_state=ps)
    dT, du, dq, precip, ps_ref = _reference(cfg, state, mesh, sigma, ps, dt, n)
    # exact up to summation order: a parallel turb+micro inside the sub-step
    # (codex round 1) missed this by 3e-6 relative at x64
    tol = dict(rtol=_RTOL, atol=_RTOL * float(jnp.abs(dT).max()))
    np.testing.assert_allclose(np.asarray(t3.dT_dt.data), np.asarray(dT), **tol)
    np.testing.assert_allclose(np.asarray(t3.du_dt.data), np.asarray(du), **tol)
    for k, v in dq.items():
        np.testing.assert_allclose(np.asarray(t3.tracer_tendencies[k].data),
                                   np.asarray(v), **tol)
    np.testing.assert_allclose(np.asarray(t3.precip.data), np.asarray(precip),
                               rtol=_RTOL, atol=_RTOL * float(jnp.abs(precip).max()))
    assert np.array_equal(np.asarray(p3.tke), np.asarray(ps_ref.tke))
    # the intermediate state matters: not the N=1 rates
    assert float(jnp.abs(t3.dT_dt.data - t1.dT_dt.data).max()) > 1e-6
    assert bool(jnp.isfinite(t3.dT_dt.data).all())


def test_jit_parity_and_finite_gradient():
    from legoesm.atmosphere.physics.combined import make_physics
    mesh, sigma, state = _moist_setup()
    cfg = _cfg()
    ps = init_physics_state(*state.T.data.shape, cfg)
    f1 = make_physics(cfg, model_type="mpas", dt=1800.0)
    f3 = make_physics(cfg, model_type="mpas", dt=1800.0, cld_macmic_num_steps=3)

    def spread(f):
        t, _ = f(state, mesh, sigma, phys_state=ps)
        tj, _ = jax.jit(lambda s, p: f(s, mesh, sigma, phys_state=p))(state, ps)
        return float(jnp.abs(tj.dT_dt.data - t.dT_dt.data).max())
    # the modules themselves have an eager-vs-jit spread on this fixture
    # (9e-8 K/s at N=1); the loop may not widen it by more than fusion noise
    assert spread(f3) <= 2.0 * spread(f1) + 1e-15

    def loss(T):
        return jnp.sum(f3(state._replace(T=state.T.replace(data=T)), mesh, sigma,
                          phys_state=ps)[0].dT_dt.data ** 2)
    g = jax.grad(loss)(state.T.data)
    assert bool(jnp.isfinite(g).all()) and float(jnp.abs(g).max()) > 0.0
    # no prognostic carry at all (SCM/hydrostatic route): the loop runs
    t_np, p_np = f3(state, mesh, sigma, phys_state=None)
    assert p_np is None and bool(jnp.isfinite(t_np.dT_dt.data).all())


def test_held_radiation_cache_and_ledger_route_through_the_loop():
    from legoesm.atmosphere.physics.combined import (
        held_physics_variant,
        make_physics,
        physics_cache_seed,
    )
    from legoesm.diagnostics.process_ledger import ROW_RADIATION
    mesh, sigma, state = _moist_setup()
    cfg = _cfg(rad="gray")
    ps = init_physics_state(*state.T.data.shape, cfg)
    kw = dict(model_type="mpas", dt=1800.0, cld_macmic_num_steps=3, budget_ledger=True)
    f_full = make_physics(cfg, **kw)
    f_norad = make_physics(cfg, need_rad=False, **kw)
    t_full, p_full = f_full(state, mesh, sigma, phys_state=ps)
    assert p_full.rad_heating is not None
    t_held, _ = f_norad(state, mesh, sigma, phys_state=p_full)
    # held: same physics except the cached heating replaces the fresh solve
    _scale = float(jnp.abs(t_full.dT_dt.data).max())
    np.testing.assert_allclose(np.asarray(t_held.dT_dt.data),
                               np.asarray(t_full.dT_dt.data), rtol=_RTOL, atol=_RTOL * _scale)
    # ledger rows sum to the applied column totals on both variants
    from legoesm.diagnostics.process_ledger import ledger_entry_column
    for t in (t_full, t_held):
        rows = t.ledger_rows
        assert rows is not None
        dq = sum(t.tracer_tendencies[k].data for k in ("q_v", "q_c", "q_r"))
        tot = ledger_entry_column(dq, t.dT_dt.data, state.p_s.data, sigma.dsigma)
        np.testing.assert_allclose(np.asarray(rows.sum(axis=1)), np.asarray(tot),
                                   rtol=1e-8 if _X64 else 1e-4,
                                   atol=(1e-8 if _X64 else 1e-4) * float(jnp.abs(tot).max()))
        assert float(jnp.abs(rows[:, ROW_RADIATION, :]).max()) > 0.0
    # cadence cache: the write variant caches exactly the macmic output
    f_w = make_physics(cfg, physics_cadence="write", physics_cadence_steps=2, **kw)
    seeded = ps._replace(held_physics=physics_cache_seed(f_w, state, mesh, sigma, ps))
    t_w, p_w = f_w(state, mesh, sigma, phys_state=seeded)
    np.testing.assert_allclose(np.asarray(t_w.dT_dt.data), 2.0 * np.asarray(t_full.dT_dt.data),
                               rtol=_RTOL, atol=_RTOL * _scale)
    t_h, _ = held_physics_variant(f_w)(state, mesh, sigma, phys_state=p_w)
    assert float(jnp.abs(t_h.dT_dt.data).max()) == 0.0


def test_per_call_inputs_survive_the_sub_cycle_and_clear_at_the_end(monkeypatch):
    """update_physics_state clears the per-call inputs; inside the loop every
    sub-module must still see them (codex round 2), the returned carry not."""
    from legoesm.atmosphere.physics.combined import make_physics
    from legoesm.atmosphere.physics.turbulence import integration as ti
    mesh, sigma, state = _moist_setup()
    cfg = _cfg()
    ps = init_physics_state(*state.T.data.shape, cfg)
    blank = update_physics_state(ps, {})
    assert all(getattr(blank, k) is None for k in PHYSSTATE_PER_CALL_INPUTS)
    assert set(PHYSSTATE_PER_CALL_INPUTS) == {
        k for k in PhysicsState._fields
        if getattr(update_physics_state(ps._replace(**{k: jnp.ones(3)}), {}), k) is None}
    ncol = state.T.data.shape[0]
    ps = ps._replace(surface_wth_override=jnp.full((ncol,), 0.05),
                     surface_wqv_override=jnp.full((ncol,), 1e-5))
    seen = []
    orig = ti.make_turbulence_physics

    def spying(*a, **kw):
        fn = orig(*a, **kw)

        def wrapped(st, grid, sc, phys_state=None, **k):
            seen.append(None if phys_state is None else phys_state.surface_wth_override)
            return fn(st, grid, sc, phys_state=phys_state, **k)
        for attr in ("_wants_forcing", "_wants_phys_state_ro"):
            if hasattr(fn, attr):
                setattr(wrapped, attr, getattr(fn, attr))
        return wrapped
    monkeypatch.setattr("legoesm.atmosphere.physics.combined.make_turbulence_physics", spying)
    f3 = make_physics(cfg, model_type="mpas", dt=1800.0, cld_macmic_num_steps=3)
    _, p_out = f3(state, mesh, sigma, phys_state=ps)
    assert len(seen) == 3 and all(v is not None for v in seen)
    assert p_out.surface_wth_override is None and p_out.surface_wqv_override is None


def test_validation_cli_and_amip_round_trip():
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    with pytest.raises(ValueError, match="cld_macmic_num_steps"):
        make_physics(_cfg(), model_type="mpas", dt=1.0, cld_macmic_num_steps=0)
    with pytest.raises(ValueError, match="cld_macmic_num_steps"):
        make_physics(_cfg(), model_type="spectral_pe", dt=1.0, cld_macmic_num_steps=2)
    with pytest.raises(ValueError, match="both are 'none'"):
        make_physics(PhysicsConfig(turbulence=TurbulenceConfig(scheme="none"),
                                   microphysics=MicrophysicsConfig(scheme="none")),
                     model_type="mpas", dt=1.0, cld_macmic_num_steps=2)
    assert callable(make_physics(PhysicsConfig(turbulence=TurbulenceConfig(scheme="tke")),
                                 model_type="hydrostatic", dt=1.0, cld_macmic_num_steps=2))
    _mpas_cfg(turbulence="tke", cld_macmic_num_steps=3).validate_strict()
    for bad, msg in ((dict(cld_macmic_num_steps=0), "cld_macmic_num_steps"),
                     (dict(cld_macmic_num_steps=3), "both are 'none'")):
        with pytest.raises(ValueError, match=msg):
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
        turbulence="tke", cld_macmic_num_steps=3)
    with pytest.raises(ValueError, match="MPAS lane"):
        cube.validate_strict()
    from legoesm.forcing.amip_config import AMIPExperimentConfig

    from scripts.run.run_amip import build_arg_parser
    parser = build_arg_parser()
    assert parser.parse_args([]).cld_macmic_num_steps == 1
    assert parser.parse_args(["--cld-macmic-num-steps", "3"]).cld_macmic_num_steps == 3
    cfg = ExperimentConfig.from_amip_config(AMIPExperimentConfig(cld_macmic_num_steps=3))
    assert cfg.cld_macmic_num_steps == 3
    assert cfg.to_amip_config().cld_macmic_num_steps == 3


def test_cam6_deck_reaches_the_config_through_the_yaml_route():
    """The flagship deck's two new rows must survive the --config route
    (YAML -> parser defaults -> build_config_from_args), which is NOT the
    AMIPExperimentConfig path (GLM round 2)."""
    import pathlib

    from legoesm.driver.run_config_yaml import load_yaml_config

    from scripts.run.run_amip import build_arg_parser
    deck = pathlib.Path(__file__).resolve().parents[2] / "config" / "amip" / "amip_production.yaml"
    parser = build_arg_parser()
    keys = load_yaml_config(str(deck), parser)
    assert keys["cld_macmic_num_steps"] == 3 and keys["morrison_sed_cfl_substeps"] is True
    parser.set_defaults(**keys)
    # the deck needs the real forcing paths to postprocess; the args ->
    # ExperimentConfig leg is covered by the CLI round trips above
    args = parser.parse_args([])
    assert args.cld_macmic_num_steps == 3 and args.morrison_sed_cfl_substeps is True
    assert args.microphysics == "morrison" and args.physics_update_steps == 16
