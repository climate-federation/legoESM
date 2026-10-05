"""MG2-style CFL sub-stepping of sedimentation (``micro_mg2_0.F90`` loop).

Contract under test:
  * ``n_substeps_max=1`` (default) is the one-pass flux-capped form, and the
    sub-stepped path reduces to it exactly whenever a column resolves to
    ``nstep=1`` (sub-CFL);
  * a super-CFL column advances the hydrometeor across MORE than one layer
    per call (the one-pass cap stops it at one), conserving column mass and
    positivity, with the static cap binding when it is smaller than the
    CFL count;
  * jit parity and a finite gradient through the masked loop;
  * the Morrison gate: ``sed_cfl_substeps`` is off by default (byte-identical)
    and on reaches the shared helper; the ExperimentConfig / CLI / applier
    route refuses the flag on any other scheme.
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.microphysics.output import sedimentation_tendency

# exact checks (array_equal) hold at either precision; the closure tolerance
# follows the dtype so the module never skips
_X64 = bool(jax.config.jax_enable_x64)
# Surface precipitation [kg/m^2/s] of the rain-shaft column in
# test_morrison_default_flip_..., CPU x64: sub-stepped (the new default) and
# the legacy one-pass arm.
_PRECIP_ON_PIN = 0.0026948362428024296
_PRECIP_OFF_PIN = 0.0006527683152764515
_RTOL = 1e-12 if _X64 else 1e-5


def _column(ncol=2, nlev=20, dz_m=1000.0, V=5.0):
    rho = jnp.linspace(0.3, 1.1, nlev)[None, :].repeat(ncol, 0)
    q = jnp.zeros((ncol, nlev)).at[:, 0].set(1.0e-4)   # all mass in the top layer
    dz = jnp.full((ncol, nlev), dz_m)
    Vt = jnp.full((ncol, nlev), V)
    return q, rho, Vt, dz


def _conserves(q, rho, Vt, dz, dt, sed, precip):
    col = jnp.sum(sed * rho * dz, axis=1)
    np.testing.assert_allclose(np.asarray(col), -np.asarray(precip), rtol=_RTOL,
                               atol=_RTOL * float(jnp.abs(sed * rho * dz).max()))
    assert bool(jnp.all(q + dt * sed >= -_RTOL * float(q.max())))


def test_sub_cfl_column_is_exactly_the_one_pass_form():
    q, rho, Vt, dz = _column()
    sink = jnp.full_like(q, 1.0e-9)
    dt = 100.0                                          # CFL = 0.5 -> nstep = 1
    for kw in ({}, {"extra_sink": sink}):
        one = sedimentation_tendency(q, rho, Vt, dz, dt=dt, return_surface_flux=True, **kw)
        sub = sedimentation_tendency(q, rho, Vt, dz, dt=dt, return_surface_flux=True,
                                     n_substeps_max=64, **kw)
        assert np.array_equal(np.asarray(one[0]), np.asarray(sub[0]))
        assert np.array_equal(np.asarray(one[1]), np.asarray(sub[1]))


def test_super_cfl_column_crosses_several_layers_and_conserves():
    q, rho, Vt, dz = _column()
    dt = 3000.0                                         # CFL = 15 -> nstep = 16
    one, p_one = sedimentation_tendency(q, rho, Vt, dz, dt=dt, return_surface_flux=True)
    sub, p_sub = sedimentation_tendency(q, rho, Vt, dz, dt=dt, return_surface_flux=True,
                                        n_substeps_max=64)
    q_one = q + dt * one
    q_sub = q + dt * sub
    # one pass: the cap lets the top layer drain at most into layer 1
    assert float(jnp.abs(q_one[:, 2:]).max()) == 0.0
    # sub-stepped: mass reaches well below layer 5 in one call
    assert float(jnp.abs(q_sub[:, 5:]).max()) > 0.0
    _conserves(q, rho, Vt, dz, dt, sub, p_sub)
    _conserves(q, rho, Vt, dz, dt, one, p_one)
    # the static cap binds: 4 sub-steps of 750 s cannot carry mass as deep
    cap, p_cap = sedimentation_tendency(q, rho, Vt, dz, dt=dt, return_surface_flux=True,
                                        n_substeps_max=4)
    q_cap = q + dt * cap
    assert float(jnp.abs(q_cap[:, 6:]).max()) == 0.0
    _conserves(q, rho, Vt, dz, dt, cap, p_cap)


def test_substepped_equals_nstep_sequential_one_pass_calls():
    """The loop IS nstep one-pass calls at dt/nstep on the running q with
    the tendency and surface flux averaged (MG2's loop, written out)."""
    q, rho, Vt, dz = _column()
    dt = 3000.0
    nstep = 1 + int(np.floor(5.0 * dt / 1000.0))          # 16, uniform column
    sub, p_sub = sedimentation_tendency(q, rho, Vt, dz, dt=dt, return_surface_flux=True,
                                        n_substeps_max=64)
    q_run, tend, sfc = q, 0.0, 0.0
    for _ in range(nstep):
        t_i, p_i = sedimentation_tendency(q_run, rho, Vt, dz, dt=dt / nstep,
                                          return_surface_flux=True)
        q_run = q_run + (dt / nstep) * t_i
        tend = tend + t_i / nstep
        sfc = sfc + p_i / nstep
    np.testing.assert_allclose(np.asarray(sub), np.asarray(tend), rtol=_RTOL,
                               atol=_RTOL * float(jnp.abs(tend).max()))
    np.testing.assert_allclose(np.asarray(p_sub), np.asarray(sfc), rtol=_RTOL, atol=0)
    assert float(jnp.abs(q + dt * sub)[:, 14:16].max()) > 0.0   # reached layer ~15


def test_pulse_reaches_the_surface_within_one_call():
    """A pulse 4 layers up with CFL 15 per call reaches the surface only with
    sub-stepping; the surface flux equals the column water lost (closure)
    and is a large fraction of the pulse (codex round 3: the earlier pulses
    never reached the surface, so a zeroed surface flux passed)."""
    q, rho, Vt, dz = _column(nlev=8)
    q = jnp.zeros_like(q).at[:, 3].set(1.0e-4)
    dt = 3000.0                                          # 15 layers of travel
    one, p_one = sedimentation_tendency(q, rho, Vt, dz, dt=dt, return_surface_flux=True)
    sub, p_sub, req = sedimentation_tendency(q, rho, Vt, dz, dt=dt, return_surface_flux=True,
                                             n_substeps_max=64, return_substeps=True)
    assert float(p_one.max()) == 0.0
    col0 = jnp.sum(q * rho * dz, axis=1)
    lost = -jnp.sum(sub * rho * dz, axis=1)
    np.testing.assert_allclose(np.asarray(p_sub * dt), np.asarray(lost * dt), rtol=_RTOL,
                               atol=_RTOL * float(col0.max()))
    assert float((p_sub * dt / col0).min()) > 0.95      # ~all of it lands
    assert np.array_equal(np.asarray(req), np.full(q.shape[0], 16, dtype=np.int32))
    # the one-pass path reports the SAME requirement (cap 1 is itself a cap
    # that can be exceeded; reporting 1 there hid the clamp -- codex round 4)
    _, _, req_one = sedimentation_tendency(q, rho, Vt, dz, dt=dt, return_surface_flux=True,
                                           return_substeps=True)
    assert np.array_equal(np.asarray(req_one), np.asarray(req))


def test_morrison_frozen_precipitation_reaches_the_surface_with_energy_closure():
    """Cold snow shaft at 600 s: sub-stepping delivers frozen precipitation to
    the surface, water closes column-wise against it, and the heating is
    IDENTICAL to the one-pass call -- sedimentation moves mass, not heat, and
    the process rates act on the same pre-source pools (documented
    departure from MG2)."""
    sys.path.insert(0, "tests/unit")
    from legoesm.atmosphere.physics.microphysics import morrison as mor
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from test_physics_microphysics import _make_column

    from legoesm import constants
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(T_sfc=255.0, q_c_val=0.0)
    hydro = hydro._replace(q_s=hydro.q_s.at[:, -6:].set(2.0e-3))
    q_v = 0.5 * q_v
    dt = 600.0
    off = mor.morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
                                    config=MorrisonConfig(sed_cfl_substeps=False))
    on = mor.morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
                                   config=MorrisonConfig())
    assert bool(jnp.all(T < constants.T_freeze))
    assert float(on.precipitation.min()) > float(off.precipitation.max()) > 0.0
    assert on.sed_substeps_required is not None and int(on.sed_substeps_required.max()) > 1
    assert off.sed_substeps_required is None
    for out in (off, on):
        dq = (out.dq_v_dt + out.dq_c_dt + out.dq_r_dt + out.dq_i_dt + out.dq_s_dt + out.dq_g_dt)
        col = jnp.sum(dq * rho * dz, axis=1)
        np.testing.assert_allclose(np.asarray(col), -np.asarray(out.precipitation),
                                   rtol=1e-6 if _X64 else 1e-4,
                                   atol=(1e-6 if _X64 else 1e-4) * float(out.precipitation.max()))
    np.testing.assert_allclose(np.asarray(on.dT_dt), np.asarray(off.dT_dt), rtol=0, atol=0)
    # column enthalpy: sedimentation adds nothing; every heating term is a
    # phase change of a water species (L_s for the vapour<->ice branches here)
    dh = constants.c_pd * on.dT_dt + constants.L_s * on.dq_v_dt
    assert float(jnp.abs(jnp.sum(dh * rho * dz, axis=1)).max()) <= 1e-6 * float(
        jnp.sum(jnp.abs(constants.c_pd * on.dT_dt) * rho * dz, axis=1).max() + 1e-30)


def test_strict_mode_raises_when_the_cap_binds():
    sys.path.insert(0, "tests/unit")
    from legoesm.atmosphere.physics.microphysics import morrison as mor
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from test_physics_microphysics import _make_column
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column()
    hydro = hydro._replace(q_r=hydro.q_r.at[:, -4:].set(1.0e-3),
                           N_r=hydro.N_r.at[:, -4:].set(1.0e3))
    out = mor.morrison_microphysics(
        T, q_v, hydro, p_full, p_half, rho, dz, 600.0,
        config=MorrisonConfig(sed_cfl_substeps=True, sed_cfl_substeps_max=2))
    assert int(out.sed_substeps_required.max()) > 2      # reported, not clamped
    with pytest.raises(Exception, match="sed_cfl_substeps_max"):
        mor.morrison_microphysics(
            T, q_v, hydro, p_full, p_half, rho, dz, 600.0,
            config=MorrisonConfig(sed_cfl_substeps=True, sed_cfl_substeps_max=2,
                                  sed_cfl_substeps_strict=True))


def test_reserve_holds_the_joint_positivity_guarantee():
    q, rho, Vt, dz = _column()
    q = jnp.full_like(q, 1.0e-4)
    dt = 3000.0
    sink = q / dt * 0.6                                 # the sink alone takes 60 %
    sub = sedimentation_tendency(q, rho, Vt, dz, dt=dt, extra_sink=sink, n_substeps_max=64)
    assert bool(jnp.all(q + dt * (sub - sink) >= -_RTOL * float(q.max())))


def test_jit_parity_and_finite_gradient():
    q, rho, Vt, dz = _column()
    dt = 3000.0
    f = lambda qq: sedimentation_tendency(qq, rho, Vt, dz, dt=dt, n_substeps_max=64)
    eager, jitted = f(q), jax.jit(f)(q)
    # tendency = flux_in - flux: the near-cancelled entries are roundoff
    np.testing.assert_allclose(np.asarray(eager), np.asarray(jitted), rtol=0,
                               atol=_RTOL * float(jnp.abs(eager).max()))
    g = jax.grad(lambda qq: jnp.sum(f(qq) ** 2))(q + 1.0e-6)
    assert bool(jnp.isfinite(g).all()) and float(jnp.abs(g).max()) > 0.0


def test_needs_dt():
    q, rho, Vt, dz = _column()
    with pytest.raises(ValueError, match="dt"):
        sedimentation_tendency(q, rho, Vt, dz, n_substeps_max=4)


# ---------------------------------------------------------------------------
# Morrison gate
# ---------------------------------------------------------------------------

def test_morrison_gate_reaches_the_helper_and_legacy_off_is_one_pass(monkeypatch):
    sys.path.insert(0, "tests/unit")
    from legoesm.atmosphere.physics.microphysics import morrison as mor
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from test_physics_microphysics import _make_column
    assert MorrisonConfig().sed_cfl_substeps is True     # user decision 2026-09-22
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column()
    # a rain shaft through the four ~470 m layers above the surface: V_r of
    # several m/s at 600 s is super-CFL there (the top layers are km thick and
    # would resolve nstep=1).  The fall speed is zero in rain-free layers (as
    # in MG2), so a shaft, not a single layer, is what sub-stepping moves.
    hydro = hydro._replace(q_r=hydro.q_r.at[:, -4:].set(1.0e-3),
                           N_r=hydro.N_r.at[:, -4:].set(1.0e3))
    seen = []
    orig = mor.sedimentation_tendency

    def spy(*a, **kw):
        seen.append(kw.get("n_substeps_max", 1))
        return orig(*a, **kw)
    monkeypatch.setattr(mor, "sedimentation_tendency", spy)
    off = mor.morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, 600.0,
                                    config=MorrisonConfig(sed_cfl_substeps=False))
    assert set(seen) == {1}
    seen.clear()
    on = mor.morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, 600.0,
                                   config=MorrisonConfig(sed_cfl_substeps=True))
    assert set(seen) == {MorrisonConfig().sed_cfl_substeps_max}
    # sub-stepped rain crosses more than one layer per call: the one-pass
    # form is not reproduced and more of it reaches the surface
    assert not np.array_equal(np.asarray(off.dq_r_dt), np.asarray(on.dq_r_dt))
    assert float(on.precipitation.min()) > float(off.precipitation.max())
    assert bool(jnp.isfinite(on.dq_r_dt).all())


def test_applier_and_config_refuse_the_flag_off_morrison():
    from legoesm.atmosphere.physics.microphysics.config import (
        MorrisonConfig,
        ThompsonConfig,
        apply_microphysics_experiment_flags,
    )
    off = apply_microphysics_experiment_flags(
        MorrisonConfig(), "morrison", morrison_sed_cfl_substeps=False)
    assert off.sed_cfl_substeps is False
    with pytest.raises(ValueError, match="morrison_sed_cfl_substeps"):
        apply_microphysics_experiment_flags(
            ThompsonConfig(), "thompson", morrison_sed_cfl_substeps=False)
    with pytest.raises(TypeError, match="bool"):
        apply_microphysics_experiment_flags(
            MorrisonConfig(), "morrison", morrison_sed_cfl_substeps="true")
    from legoesm.driver.config import ExperimentConfig
    # the flat default is LOCKED to the scheme leaf (both True since 2026-09-22)
    assert (ExperimentConfig._field_defaults["morrison_sed_cfl_substeps"]
            is MorrisonConfig().sed_cfl_substeps is True)
    assert (ExperimentConfig._field_defaults["morrison_sed_cfl_substeps_strict"]
            is MorrisonConfig().sed_cfl_substeps_strict is False)
    from legoesm.forcing.amip_config import AMIPExperimentConfig
    assert AMIPExperimentConfig().morrison_sed_cfl_substeps is True
    assert AMIPExperimentConfig().morrison_sed_cfl_substeps_strict is False
    # an untouched non-Morrison config is silent; an explicit deviation is not
    ExperimentConfig(microphysics="thompson").validate_strict()
    with pytest.raises(ValueError, match="must be a bool"):
        ExperimentConfig(microphysics="morrison",
                         morrison_sed_cfl_substeps="true").validate_strict()
    with pytest.raises(ValueError, match="needs"):
        ExperimentConfig(microphysics="morrison", morrison_sed_cfl_substeps=False,
                         morrison_sed_cfl_substeps_strict=True).validate_strict()
    ExperimentConfig(microphysics="morrison", morrison_sed_cfl_substeps=True,
                     morrison_sed_cfl_substeps_strict=True).validate_strict()
    from legoesm.driver.physics_pipeline import thread_morrison_scalars
    cfg = ExperimentConfig(microphysics="morrison", morrison_sed_cfl_substeps=True,
                           morrison_sed_cfl_substeps_strict=True)
    threaded = thread_morrison_scalars(cfg, "morrison", MorrisonConfig())
    assert threaded.sed_cfl_substeps and threaded.sed_cfl_substeps_strict
    legacy = thread_morrison_scalars(
        ExperimentConfig(microphysics="morrison", morrison_sed_cfl_substeps=False),
        "morrison", MorrisonConfig())
    assert legacy.sed_cfl_substeps is False
    with pytest.raises(ValueError, match="only supported by the morrison"):
        thread_morrison_scalars(
            ExperimentConfig(microphysics="thompson", morrison_sed_cfl_substeps=False),
            "thompson", ThompsonConfig())
    with pytest.raises(TypeError, match="bool"):
        thread_morrison_scalars(ExperimentConfig(microphysics="morrison",
                                                 morrison_sed_cfl_substeps="true"),
                                "morrison", MorrisonConfig())
    base = MorrisonConfig()
    assert thread_morrison_scalars(ExperimentConfig(microphysics="morrison"),
                                   "morrison", base) is base
    with pytest.raises(ValueError, match="morrison_sed_cfl_substeps"):
        ExperimentConfig(microphysics="thompson",
                         morrison_sed_cfl_substeps=False).validate_strict()


def test_cli_round_trip():
    from scripts.run.run_amip import (
        _postprocess_args,
        build_arg_parser,
        build_config_from_args,
    )
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--microphysics", "morrison",
        "--no-morrison-sed-cfl-substeps"]), parser))
    assert cfg.morrison_sed_cfl_substeps is False
    d = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert d.morrison_sed_cfl_substeps is True
    from legoesm.driver.config import ExperimentConfig
    rt = ExperimentConfig.from_amip_config(cfg.to_amip_config())
    assert rt.morrison_sed_cfl_substeps is False
    cfg2 = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--microphysics", "morrison",
        "--morrison-sed-cfl-substeps", "--morrison-sed-cfl-substeps-strict",
        "--morrison-sed-cfl-substeps-max", "96"]), parser))
    assert cfg2.morrison_sed_cfl_substeps_strict is True
    assert cfg2.morrison_sed_cfl_substeps_max == 96
    cfg2.validate_strict()
    assert ExperimentConfig.from_amip_config(
        cfg2.to_amip_config()).morrison_sed_cfl_substeps_max == 96
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig as _M
    from legoesm.driver.physics_pipeline import thread_morrison_scalars as _thr
    assert _thr(cfg2, "morrison", _M()).sed_cfl_substeps_max == 96
    with pytest.raises(ValueError, match="sed_cfl_substeps_max"):
        ExperimentConfig(microphysics="morrison",
                         morrison_sed_cfl_substeps_max=0).validate_strict()
    assert ExperimentConfig.from_amip_config(
        cfg2.to_amip_config()).morrison_sed_cfl_substeps_strict is True


def test_required_count_rides_the_combined_tendency_and_the_cadence_cache():
    """The int diagnostic Field survives the combined physics (max over the
    macmic window), the cadence cache seed (same structure, no retrace) and
    the held variant (re-published unscaled)."""
    sys.path.insert(0, "tests/unit")
    from legoesm.atmosphere.physics.combined import (
        PhysicsConfig,
        held_physics_variant,
        make_physics,
        physics_cache_seed,
    )
    from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    from test_physics_macmic import _moist_setup
    mesh, sigma, state = _moist_setup()
    zeros = state.T.replace(data=jnp.zeros_like(state.T.data))
    state = state._replace(tracers={**state.tracers,
                                    **{k: zeros.replace(name=k) for k in
                                       ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i")}})
    cfg = PhysicsConfig(
        turbulence=TurbulenceConfig(scheme="tke"),
        microphysics=MicrophysicsConfig(scheme="morrison",
                                        morrison=MorrisonConfig(sed_cfl_substeps=True)))
    ps = init_physics_state(*state.T.data.shape, cfg)
    f3 = make_physics(cfg, model_type="mpas", dt=1800.0, cld_macmic_num_steps=3,
                      physics_cadence="write", physics_cadence_steps=2)
    seeded = ps._replace(held_physics=physics_cache_seed(f3, state, mesh, sigma, ps))
    t, p = f3(state, mesh, sigma, phys_state=seeded)
    req = t.sed_substeps_required
    assert req is not None and req.data.dtype == jnp.int32 and int(req.data.max()) > 1
    assert (jax.tree_util.tree_structure(p.held_physics)
            == jax.tree_util.tree_structure(seeded.held_physics))
    t_h, _ = held_physics_variant(f3)(state, mesh, sigma, phys_state=p)
    assert np.array_equal(np.asarray(t_h.sed_substeps_required.data), np.asarray(req.data))
    assert float(jnp.abs(t_h.dT_dt.data).max()) == 0.0
    # the N=1 call at the sub-step length IS the window's first sub-cycle
    # (same state, same 600 s), so its count is a lower bound of the max --
    # and equals it whenever the later sub-cycles do not add hydrometeors
    f1 = make_physics(cfg, model_type="mpas", dt=600.0)
    t1, _ = f1(state, mesh, sigma, phys_state=ps)
    assert bool(jnp.all(t1.sed_substeps_required.data <= req.data))


def test_combined_count_is_the_max_over_the_macmic_window(monkeypatch):
    """The exported count is the element-wise MAX of the three sub-cycle
    counts, not their average (GLM round 3, #1/#7)."""
    sys.path.insert(0, "tests/unit")
    from legoesm.atmosphere.physics import combined as cmb
    from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    from test_physics_macmic import _moist_setup
    mesh, sigma, state = _moist_setup()
    zeros = state.T.replace(data=jnp.zeros_like(state.T.data))
    state = state._replace(tracers={**state.tracers,
                                    **{k: zeros.replace(name=k) for k in
                                       ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i")}})
    cfg = cmb.PhysicsConfig(
        turbulence=TurbulenceConfig(scheme="tke"),
        microphysics=MicrophysicsConfig(scheme="morrison",
                                        morrison=MorrisonConfig(sed_cfl_substeps=True)))
    seen = []
    orig = cmb.make_microphysics_physics

    def spying(*a, **kw):
        fn = orig(*a, **kw)

        def wrapped(*aa, **kk):
            t = fn(*aa, **kk)
            # non-monotone per-call bumps: max != last != mean are all distinct
            bump = (3000, 1000, 2000)[len(seen)]
            t = t._replace(sed_substeps_required=t.sed_substeps_required.replace(
                data=t.sed_substeps_required.data + bump))
            seen.append(np.asarray(t.sed_substeps_required.data))
            return t
        for attr in ("_wants_forcing", "_wants_phys_state_ro"):
            if hasattr(fn, attr):
                setattr(wrapped, attr, getattr(fn, attr))
        return wrapped
    monkeypatch.setattr(cmb, "make_microphysics_physics", spying)
    ps = init_physics_state(*state.T.data.shape, cfg)
    f3 = cmb.make_physics(cfg, model_type="mpas", dt=1800.0, cld_macmic_num_steps=3)
    t, _ = f3(state, mesh, sigma, phys_state=ps)
    assert len(seen) == 3
    expect = np.maximum(np.maximum(seen[0], seen[1]), seen[2])
    assert t.sed_substeps_required.data.dtype == seen[0].dtype
    assert np.array_equal(np.asarray(t.sed_substeps_required.data), expect)
    assert not np.array_equal(expect, seen[2])          # not last-wins
    assert not np.array_equal(expect, (seen[0] + seen[1] + seen[2]) / 3)


def _morrison_mpas(cap=256, strict=False, dt=112.5):
    """MPAS model + Morrison macmic-free physics on the moist fixture."""
    sys.path.insert(0, "tests/unit")
    from test_physics_macmic import _moist_setup

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
        MPASPrimitiveEquationModel,
    )
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    mesh, sigma, state = _moist_setup()
    zeros = state.T.replace(data=jnp.zeros_like(state.T.data))
    state = state._replace(tracers={**state.tracers,
                                    **{k: zeros.replace(name=k) for k in
                                       ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i")}})
    state = state._replace(tracers={**state.tracers,
                                    "q_r": state.tracers["q_r"].replace(
                                        data=state.tracers["q_r"].data + 5.0e-4)})
    cfg = PhysicsConfig(
        turbulence=TurbulenceConfig(scheme="tke"),
        microphysics=MicrophysicsConfig(
            scheme="morrison",
            morrison=MorrisonConfig(sed_cfl_substeps=True, sed_cfl_substeps_max=cap,
                                    sed_cfl_substeps_strict=strict)))
    model = MPASPrimitiveEquationModel(mesh, sigma, MPASPrimitiveEquationConfig())
    ps = init_physics_state(*state.T.data.shape, cfg)
    return model, state, mesh, sigma, make_physics(cfg, model_type="mpas", dt=dt), ps


def test_overflow_is_reported_through_the_mpas_run_output(caplog):
    """A planted overflow reaches the run's log through the lean-loop export
    slot, without the strict abort (codex round 4 P1: the count reached the
    tendency but no consumer read it)."""
    import logging

    from legoesm.core.state import MPAS_SFC_DIAG_EXTRA_KEYS
    from legoesm.driver.model_driver import (
        _sed_substeps_slot,
        report_sed_substep_overflow,
    )
    assert "sed_substeps_required" in MPAS_SFC_DIAG_EXTRA_KEYS
    # 3600 s on the coarse 8-level fixture puts rain above one layer per call
    model, state, mesh, sigma, phys, ps = _morrison_mpas(cap=2, dt=3600.0)
    model.step(state, 3600.0, phys, phys_state=ps)
    sfc = model._sfc_diag
    slot = _sed_substeps_slot()
    assert sfc[slot] is not None and int(jnp.max(sfc[slot].data)) > 2
    with caplog.at_level(logging.INFO, logger="legoesm.driver"):
        n = report_sed_substep_overflow(sfc, 2, step=432)
    assert n > 2
    assert any("CLAMPED" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="legoesm.driver"):
        report_sed_substep_overflow(sfc, 1024, step=432)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
    assert any("sub-steps required" in r.getMessage() for r in caplog.records)
    # nothing published (sub-stepping off / non-Morrison lane) is not an error
    assert report_sed_substep_overflow(None, 4, step=0) is None
    assert report_sed_substep_overflow((None,) * (slot + 1), 4, step=0) is None
    # an overflow BETWEEN two reports survives: the export slot holds only the
    # latest step, so the driver accumulates a window maximum (codex round 4)
    from legoesm.driver.model_driver import sed_substeps_window_max
    _quiet = tuple(
        (sfc[i].replace(data=jnp.ones_like(sfc[i].data)) if i == slot else sfc[i])
        for i in range(len(sfc)))
    win = None
    for _s in (_quiet, sfc, _quiet, _quiet):        # the spike is step 1 of 4
        win = sed_substeps_window_max(_s, win)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="legoesm.driver"):
        assert report_sed_substep_overflow(_quiet, 2, step=432, running=win) > 2
    assert any("CLAMPED" in r.getMessage() for r in caplog.records)


def test_strict_survives_a_caller_that_discards_the_diagnostics():
    """Under jit and under grad, a caller that keeps only precipitation must
    still hit the abort (codex round 4 P1: attaching error_if to the count
    alone let XLA eliminate it)."""
    sys.path.insert(0, "tests/unit")
    from test_physics_microphysics import _make_column

    from legoesm.atmosphere.physics.microphysics import morrison as mor
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column()
    hydro = hydro._replace(q_r=hydro.q_r.at[:, -4:].set(1.0e-3),
                           N_r=hydro.N_r.at[:, -4:].set(1.0e3))
    cfg = MorrisonConfig(sed_cfl_substeps=True, sed_cfl_substeps_max=2,
                         sed_cfl_substeps_strict=True)

    def precip_only(qv):
        return mor.morrison_microphysics(T, qv, hydro, p_full, p_half, rho, dz,
                                         600.0, config=cfg).precipitation
    for fn in (jax.jit(precip_only),
               jax.jit(lambda qv: jnp.sum(precip_only(qv))),
               jax.grad(lambda qv: jnp.sum(precip_only(qv)))):
        with pytest.raises(Exception, match="sed_cfl_substeps_max"):
            jax.block_until_ready(fn(q_v))

    def vapour_only(qv):
        return mor.morrison_microphysics(T, qv, hydro, p_full, p_half, rho, dz,
                                         600.0, config=cfg).dq_v_dt
    # Vapour does not descend from sedimentation, and jit(grad(...)) keeps the
    # gradient while dropping the error_if (its check sits outside AD), so in
    # those compositions the guard must POISON the values instead: NaN, never
    # a finite number a caller can use (codex round 4).
    def field_only(name):
        def f(qv):
            return getattr(mor.morrison_microphysics(
                T, qv, hydro, p_full, p_half, rho, dz, 600.0, config=cfg), name)
        return f
    for fn in (jax.jit(vapour_only),
               jax.jit(field_only("dT_dt")),          # no sedimentation term
               jax.jit(field_only("dN_r_dt")),        # number, computed later
               jax.jit(jax.grad(lambda qv: jnp.sum(field_only("dT_dt")(qv)))),
               jax.jit(jax.grad(lambda qv: jnp.sum(precip_only(qv)))),
               jax.jit(jax.grad(lambda qv: jnp.sum(vapour_only(qv))))):
        try:
            out = jax.block_until_ready(fn(q_v))
        except Exception as exc:                     # the abort is fine too
            assert "sed_cfl_substeps_max" in str(exc)
        else:
            assert not bool(jnp.isfinite(out).all())
    # cap 1 (the one-pass execution path) still detects the overflow
    with pytest.raises(Exception, match="sed_cfl_substeps_max"):
        mor.morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, 600.0,
                                  config=cfg._replace(sed_cfl_substeps_max=1))
    for bad in (1.9, True, "2"):
        with pytest.raises(ValueError, match="sed_cfl_substeps_max"):
            mor.morrison_microphysics(
                T, q_v, hydro, p_full, p_half, rho, dz, 600.0,
                config=MorrisonConfig(sed_cfl_substeps=True,
                                      sed_cfl_substeps_max=bad))
    # a cap that is not exceeded runs clean on all three
    cfg_ok = cfg._replace(sed_cfl_substeps_max=64)

    def precip_ok(qv):
        return mor.morrison_microphysics(T, qv, hydro, p_full, p_half, rho, dz,
                                         600.0, config=cfg_ok).precipitation
    assert bool(jnp.isfinite(jax.jit(precip_ok)(q_v)).all())
    g = jax.grad(lambda qv: jnp.sum(precip_ok(qv)))(q_v)
    assert bool(jnp.isfinite(g).all())
    assert bool(jnp.isfinite(
        jax.jit(jax.grad(lambda qv: jnp.sum(precip_ok(qv))))(q_v)).all())
    out_ok = mor.morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz,
                                       600.0, config=cfg_ok)
    assert bool(jnp.isfinite(out_ok.dq_v_dt).all())
    # Documented residual: a field that is STRUCTURALLY constant (dN_c_dt
    # with predict_Nc off) has no AD path, so jit(grad) of a loss reading
    # only it returns zeros -- but its forward value is NaN and the count
    # still reports the overflow (codex round 5).
    out_bad = mor.morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz,
                                        600.0, config=cfg._replace(
                                            sed_cfl_substeps_strict=False))
    assert int(out_bad.sed_substeps_required.max()) > 2
    nc = jax.jit(field_only("dN_c_dt"))(q_v)
    assert not bool(jnp.isfinite(nc).all())


def test_morrison_default_flip_changes_the_numbers_and_both_arms_are_pinned():
    """The default flip is WITNESSED: on the same column the default (ON) and
    the legacy arm (OFF) give different rain tendencies and surface
    precipitation, and each arm is pinned to its own value (GLM round 4: the
    macmic parent pins run Kessler, so they cannot see this)."""
    sys.path.insert(0, "tests/unit")
    from test_physics_microphysics import _make_column

    from legoesm.atmosphere.physics.microphysics import morrison as mor
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column()
    hydro = hydro._replace(q_r=hydro.q_r.at[:, -4:].set(1.0e-3),
                           N_r=hydro.N_r.at[:, -4:].set(1.0e3))
    on = mor.morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, 600.0,
                                   config=MorrisonConfig())
    off = mor.morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, 600.0,
                                    config=MorrisonConfig(sed_cfl_substeps=False))
    assert MorrisonConfig().sed_cfl_substeps is True            # the flip itself
    p_on, p_off = float(on.precipitation[0]), float(off.precipitation[0])
    assert p_on > 1.02 * p_off > 0.0                            # the arms differ
    # CPU x64 pins of BOTH arms on this fixture (recompute deliberately if the
    # scheme changes; a silent revert of the default would move the first one)
    if _X64:
        np.testing.assert_allclose([p_on, p_off], [_PRECIP_ON_PIN, _PRECIP_OFF_PIN],
                                   rtol=1e-9)


def test_new_config_fields_sit_at_the_tuple_end():
    """Positional-ABI drift guard (GLM round 4): the 2026-09-22 fields were
    inserted mid-tuple, which silently re-binds old positional constructions
    and pickles."""
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    from legoesm.forcing.amip_config import AMIPExperimentConfig
    # The sedimentation trio is no longer LAST: the in-run cloud-water budget
    # appended ``publish_qc_budget`` after it, which is the correct end-append,
    # so the trio is pinned where it now sits rather than at the tail.
    assert MorrisonConfig._fields[-9:-6] == (
        "sed_cfl_substeps", "sed_cfl_substeps_max", "sed_cfl_substeps_strict")
    assert MorrisonConfig._fields[-6] == "publish_qc_budget"
    # main end-appended its four warm-rain fields after it; the branch then
    # moved liquid_from_closure from mid-tuple to the very end.
    assert MorrisonConfig._fields[-5:] == (
        "autocon_fact", "accre_enhan_fact", "kk2000_cam6_relvar",
        "warm_rain_incloud", "liquid_from_closure")
    # ``morrison_do_graupel`` end-appended after the block (2026-09-24).
    # ExperimentConfig then end-appended the two land canopy smoothing widths
    # (2026-09-30); main appended Morrison warm-rain + ZM/CLUBB tunables
    # (ExperimentConfig) and zm_land_fraction (AMIP) after the block.
    for cls, tail in ((ExperimentConfig, 21), (AMIPExperimentConfig, 1)):
        f = cls._fields[:len(cls._fields) - tail]
        assert f[-5:-1] == (
            "cld_macmic_num_steps", "morrison_sed_cfl_substeps",
            "morrison_sed_cfl_substeps_max", "morrison_sed_cfl_substeps_strict")
        assert f[-1] == "morrison_do_graupel"
    assert ExperimentConfig._fields[-3:] == (
        "land_canopy_rh_cap_smoothing_width", "land_canopy_zeta_cap_smoothing_width",
        "land_canopy_most_n_iters")
    # ... AND the field before the block is pinned, so an insertion just
    # ahead of it (which re-binds every stored positional value) goes red
    # too (GLM round 4)
    assert MorrisonConfig._fields[-10] == "homogeneous_ice_supersaturation"
    assert ExperimentConfig._fields[-27] == "bechtold_rhebc_land_deep"
    assert AMIPExperimentConfig._fields[-7] == "physics_parameterization_seed"
    # full field ORDER, hashed: an insertion anywhere (not just before the
    # tail) re-binds every stored positional value, so pin the whole tuple
    # (recompute deliberately when a field is added AT THE END)
    import hashlib
    for cls, n, digest in (
            # 116 -> 117 when the in-run cloud-water budget appended
            # publish_qc_budget at the END, which is the convention this guard
            # protects rather than a violation of it.
            # 117 -> 118: liquid_from_closure END-appended (main had inserted
            # it mid-tuple; moved so the first 117 fields hash to the
            # pre-insertion 22176757db31d564 again).
            # 118 -> 122 at the 2026-10-05 merge of main: main END-appended
            # four warm-rain fields; liquid_from_closure stays last.
            (MorrisonConfig, 122, "90ebac6169c628f6"),
            # 283 -> 288 at the 2026-09-23 merge of main: main inserted five
            # cloud_cap_floor_* fields MID-tuple (idx ~65-69), which is exactly
            # what this guard is for.  Recomputed, not relaxed -- the audit
            # that accompanied it found no positional construction of this
            # tuple anywhere and its serialization is name-keyed (_asdict),
            # so nothing re-binds.
            # 288 -> 291 / 125 -> 126 at the 2026-09-26 merge: main added the
            # CLUBB liquid-partition fields; morrison_do_graupel end-appended.
            # 291 -> 293: land canopy smoothing widths END-appended; the first
            # 291 fields still hash to c023f86f71966a44.
            # 293 -> 294: land_canopy_most_n_iters END-appended (prefix still
            # b380eb4b2adef6b3).
            # 294 -> 309 / AMIP re-hashed at the 2026-10-05 merge: main appended
            # Morrison warm-rain + ZM/CLUBB tunables (ExperimentConfig) and
            # zm_land_fraction (AMIP); the canopy trio stays last.
            (ExperimentConfig, 309, "84c24671eba721a2"),
            (AMIPExperimentConfig, 126, "8924f13ccbc6880f")):
        assert len(cls._fields) == n, (cls.__name__, len(cls._fields))
        assert hashlib.sha256(",".join(cls._fields).encode()).hexdigest()[:16] \
            == digest, f"{cls.__name__} field ORDER changed (positional ABI)"
    # the OUTPUT tuples the count rides through are pinned the same way
    # (GLM round 7: ABI closure was 3/5) -- each new field is end-appended
    from legoesm.atmosphere.physics.microphysics.output import (
        MicrophysicsOutput,
    )
    from legoesm.core.physics_output import PhysicsOutput
    from legoesm.core.state import HydrostaticTendencies
    for cls, digest in ((MicrophysicsOutput, "7b2002c93c5b0f66"),  # +qc_budget
                        (HydrostaticTendencies, "660c90d9a469cba4"),
                        (PhysicsOutput, "90eef946c1e51a5f")):
        assert cls._fields[-1] == "sed_substeps_required", cls.__name__
        assert hashlib.sha256(",".join(cls._fields).encode()).hexdigest()[:16] \
            == digest, f"{cls.__name__} field ORDER changed (positional ABI)"
    # the flat default is the leaf default, in one place
    from legoesm.atmosphere.physics.microphysics import morrison as _mor
    assert (MorrisonConfig().sed_cfl_substeps_max
            == ExperimentConfig._field_defaults["morrison_sed_cfl_substeps_max"]
            == AMIPExperimentConfig().morrison_sed_cfl_substeps_max
            == _mor._SEDIMENTATION_SUBSTEPS_MAX)


def test_effective_cap_comes_from_the_resolved_scheme_config():
    """The reporter's cap must be the one the LANE runs: a deck's own scheme
    leaf wins over the flat field, and the flat field is the fallback
    (GLM round 5 -- reading the wrong one makes the clamp invisible)."""
    from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import effective_sed_substeps_cap
    from legoesm.driver.physics_pipeline import thread_morrison_scalars
    deck_leaf = MicrophysicsConfig(
        scheme="morrison", morrison=MorrisonConfig(sed_cfl_substeps_max=96))
    flat = ExperimentConfig(microphysics="morrison",
                            morrison_sed_cfl_substeps_max=90)
    assert effective_sed_substeps_cap(deck_leaf, "morrison", flat) == 96
    # ... and after threading (flat overrides the leaf) it follows the leaf
    threaded = deck_leaf._replace(
        morrison=thread_morrison_scalars(flat, "morrison", deck_leaf.morrison))
    assert threaded.morrison.sed_cfl_substeps_max == 90
    assert effective_sed_substeps_cap(threaded, "morrison", flat) == 90
    # no scheme leaf at all -> the flat field, then the scheme default
    assert effective_sed_substeps_cap(None, "morrison", flat) == 90
    assert effective_sed_substeps_cap(None, "morrison", None) == \
        MorrisonConfig().sed_cfl_substeps_max


def test_leaf_cap_rejects_an_out_of_range_value():
    sys.path.insert(0, "tests/unit")
    from test_physics_microphysics import _make_column

    from legoesm.atmosphere.physics.microphysics import morrison as mor
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column()
    for bad in (1025, 100000, 0):
        with pytest.raises(ValueError, match=r"\[1, 1024\]"):
            mor.morrison_microphysics(
                T, q_v, hydro, p_full, p_half, rho, dz, 600.0,
                config=MorrisonConfig(sed_cfl_substeps=True,
                                      sed_cfl_substeps_max=bad))


def test_the_flux_accumulator_tolerates_the_new_slot():
    """The CMOR flux accumulator must ignore the int32 count at slot 12."""
    from legoesm.driver import model_driver as md
    _Accum = md._MPASSfcFluxAccum
    model, state, mesh, sigma, phys, ps = _morrison_mpas(cap=256, dt=3600.0)
    model.step(state, 3600.0, phys, phys_state=ps)
    _sed_substeps_slot = md._sed_substeps_slot
    acc = _Accum()
    acc.add(model._sfc_diag)
    assert _sed_substeps_slot() not in _Accum.SLOTS
    assert model._sfc_diag[_sed_substeps_slot()].data.dtype == jnp.int32


def test_cap_has_an_upper_bound_and_the_report_names_the_effective_cap(caplog):
    """A typo'd cap costs linearly, so it is bounded; and the driver reports
    the cap the LANE ran, not the flat config field (GLM round 4)."""
    import logging

    from legoesm.atmosphere.physics.microphysics.config import (
        MorrisonConfig,
        apply_microphysics_experiment_flags,
    )
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import report_sed_substep_overflow
    with pytest.raises(ValueError, match=r"\[1, 1024\]"):
        apply_microphysics_experiment_flags(
            MorrisonConfig(), "morrison", morrison_sed_cfl_substeps_max=2560)
    with pytest.raises(ValueError, match=r"\[1, 1024\]"):
        ExperimentConfig(microphysics="morrison",
                         morrison_sed_cfl_substeps_max=2560).validate_strict()
    model, state, mesh, sigma, phys, ps = _morrison_mpas(cap=2, dt=3600.0)
    model.step(state, 3600.0, phys, phys_state=ps)
    from legoesm.driver.model_driver import effective_sed_substeps_cap
    # the lane's cap is the LEAF's 2, although the flat field says 256
    _cap = effective_sed_substeps_cap(
        MorrisonConfig(sed_cfl_substeps_max=2), "morrison",
        ExperimentConfig(microphysics="morrison"))
    assert _cap == 2
    with caplog.at_level(logging.INFO, logger="legoesm.driver"):
        report_sed_substep_overflow(model._sfc_diag, _cap, step=0)
    _msgs = [r.getMessage() for r in caplog.records]
    assert any("CLAMPED" in m for m in _msgs)
    # the number in the message is the cap the lane ran, not 256 (GLM round 7)
    assert any("capped at 2," in m for m in _msgs), _msgs
    assert not any("capped at 256" in m for m in _msgs), _msgs


def test_a_non_finite_fall_speed_reports_the_ceiling_not_one():
    """A NaN/inf fall speed is not a runnable column: the required count must
    saturate (so the cap-overflow gate fires) instead of reporting a quiet 1
    that looks like a sub-CFL column (GLM round 6)."""
    q, rho, Vt, dz = _column()
    for bad in (jnp.nan, jnp.inf):
        Vb = Vt.at[0, -1].set(bad)
        for kw in ({}, {"n_substeps_max": 8}):
            out = sedimentation_tendency(q, rho, Vb, dz, dt=100.0,
                                         return_substeps=True, **kw)
            n = out[-1]
            assert int(n[0]) > 2 ** 20            # the poisoned column saturates
            assert int(n[1]) == 1                 # its healthy neighbour does not


def test_f32_column_under_x64_keeps_its_dtype():
    """A float32 state with x64 enabled (the coupled lanes' mixed precision)
    must not be promoted by the loop: the fori_loop carry types would stop
    matching and the run dies at trace time."""
    q, rho, Vt, dz = _column()
    f32 = [jnp.asarray(a, dtype=jnp.float32) for a in (q, rho, Vt, dz)]
    sink = jnp.zeros_like(f32[0])
    for kw in ({}, {"extra_sink": sink}):
        # an f64 dt is what promoted the carry in the real lane, so the
        # weakly-typed python float will not do (GLM round 9)
        _dt = jnp.asarray(3000.0, dtype=jnp.float64) if _X64 else 3000.0
        sed, precip = sedimentation_tendency(
            *f32, dt=_dt, return_surface_flux=True, n_substeps_max=32, **kw)
        assert sed.dtype == jnp.float32 and precip.dtype == jnp.float32


def test_strict_startup_guard_reads_the_platform_env(monkeypatch):
    """The guard refuses a GPU-only lane from the DECLARED platform list, so
    it initializes no backend and every rank refuses alike; with nothing
    declared it falls back to a device query, and a query that raises is
    reported as the cause (GLM round 7)."""
    from legoesm.driver import model_driver as _mod

    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    with pytest.raises(SystemExit, match="needs a CPU device"):
        _mod.require_cpu_for_strict_sedimentation()
    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")
    _mod.require_cpu_for_strict_sedimentation()          # allowed

    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    import jax as _jax
    monkeypatch.setattr(_jax, "devices", lambda *_a: [])
    with pytest.raises(SystemExit, match="unset"):
        _mod.require_cpu_for_strict_sedimentation()

    def _boom(*_a):
        raise ValueError("no such backend")             # not RuntimeError
    monkeypatch.setattr(_jax, "devices", _boom)
    with pytest.raises(SystemExit, match="no such backend"):
        _mod.require_cpu_for_strict_sedimentation()


def test_a_lane_that_cannot_report_says_so(caplog):
    """The compiled-segment lane reduces inside lax.scan and publishes no
    count, so it announces the gap once at run start instead of being
    quietly blind to a clamped fall (GLM round 7)."""
    import logging

    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import warn_sed_substeps_unreported
    with caplog.at_level(logging.WARNING, logger="legoesm.driver"):
        assert warn_sed_substeps_unreported(
            ExperimentConfig(microphysics="morrison",
                             morrison_sed_cfl_substeps=True)) is True
        assert any("does not report" in r.getMessage() for r in caplog.records)
    # off, or strict (a clamp aborts): nothing to warn about
    assert warn_sed_substeps_unreported(
        ExperimentConfig(microphysics="morrison",
                         morrison_sed_cfl_substeps=False)) is False
    assert warn_sed_substeps_unreported(
        ExperimentConfig(microphysics="morrison",
                         morrison_sed_cfl_substeps=True,
                         morrison_sed_cfl_substeps_strict=True)) is False
    # a scheme that does not sediment must not be warned at all, and the
    # RESOLVED leaf decides, not the flat default (GLM round 8: the warning
    # fired on every kessler deck and on leaf overrides)
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    assert warn_sed_substeps_unreported(
        ExperimentConfig(microphysics="kessler")) is False
    _cfg = ExperimentConfig(microphysics="morrison")
    assert warn_sed_substeps_unreported(
        _cfg, "x", MorrisonConfig(sed_cfl_substeps=False)) is False
    assert warn_sed_substeps_unreported(
        _cfg, "x", MorrisonConfig(sed_cfl_substeps_strict=True)) is False
    assert warn_sed_substeps_unreported(_cfg, "x", MorrisonConfig()) is True


def test_the_per_step_lane_keeps_its_reporting_wiring():
    """Source ratchet for the finite-volume per-step loop (GLM round 7):
    its window accumulation, cadence report and post-loop flush have no
    cheap functional test (the lane is a full model run), so pin the call
    sites in the function that actually runs, and pin the attribute the cap
    resolver reads -- a silent rename would make the log name the wrong cap.
    """
    import inspect

    from legoesm.driver.model_driver import ModelDriver
    from legoesm.driver.physics_pipeline import PhysicsPipeline
    src = inspect.getsource(ModelDriver._run_per_step)
    assert src.count("sed_substeps_window_max(") == 2, (
        "the warm-up seed or the per-step accumulation is gone")
    assert src.count("report_sed_substep_overflow(") == 2, (
        "the cadence report or the post-loop flush is gone")
    assert "counts=phys_out.sed_substeps_required" in src
    assert "effective_sed_substeps_cap(" in src
    # the resolver reads this attribute off the pipeline; a rename would
    # silently fall back to the flat cap
    assert "micro_config" in inspect.signature(PhysicsPipeline.__init__).parameters


def test_a_leaf_only_strict_deck_still_needs_a_cpu(monkeypatch):
    """The startup refusal must see a deck that arms the strict abort on the
    scheme leaf alone, not only through the flat field (codex round 9)."""
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import strict_sed_abort_requested
    _cfg = ExperimentConfig(microphysics="morrison")
    assert strict_sed_abort_requested(_cfg) is False
    assert strict_sed_abort_requested(
        _cfg, MorrisonConfig(sed_cfl_substeps_strict=True)) is True
    assert strict_sed_abort_requested(
        ExperimentConfig(microphysics="morrison",
                         morrison_sed_cfl_substeps_strict=True)) is True
    # a non-sedimenting scheme never arms it
    assert strict_sed_abort_requested(
        ExperimentConfig(microphysics="kessler"),
        MorrisonConfig(sed_cfl_substeps_strict=True)) is False


def test_the_dispatcher_keeps_its_warning_and_guard():
    """Source ratchet for the lane dispatcher (GLM round 9): the per-lane
    warning and the strict-abort CPU refusal are called from one place, and
    deleting a call site would otherwise leave every suite green."""
    import inspect

    from legoesm.driver.model_driver import ModelDriver
    src = inspect.getsource(ModelDriver.run)
    assert src.count("warn_sed_substeps_unreported(") == 5, (
        "a non-reporting lane lost its warning (or a lane was added without "
        "one)")
    assert "strict_sed_abort_requested(" in src \
        and "require_cpu_for_strict_sedimentation()" in src, (
        "the strict-abort CPU refusal is no longer enforced for every lane")


@pytest.mark.parametrize("scale, counts, cap", [
    ((0.14, 0.54, 1.1), (3, 9, 17), 64),
    ((0.14, 0.54, 0.67), (3, 9, 11), 12),   # cap not a multiple of the chunk
])
def test_early_exit_columns_with_different_counts_match_their_own_sequence(scale, counts, cap):
    """Columns needing different pass counts (either side of the early-exit
    chunk edges) each equal their own sequence of one-pass calls, so the loop
    still runs every pass the neediest column requires."""
    q, rho, Vt, dz = _column(ncol=3)
    dt = 3000.0
    Vt = Vt * jnp.asarray(scale)[:, None]
    sub, p_sub, n = sedimentation_tendency(q, rho, Vt, dz, dt=dt, return_surface_flux=True,
                                           n_substeps_max=cap, return_substeps=True)
    assert [int(x) for x in n] == list(counts)
    for c, nstep in enumerate(counts):
        sl = slice(c, c + 1)
        q_run, tend, sfc = q[sl], 0.0, 0.0
        for _ in range(nstep):
            t_i, p_i = sedimentation_tendency(q_run, rho[sl], Vt[sl], dz[sl], dt=dt / nstep,
                                              return_surface_flux=True)
            q_run = q_run + (dt / nstep) * t_i
            tend = tend + t_i / nstep
            sfc = sfc + p_i / nstep
        np.testing.assert_allclose(np.asarray(sub[sl]), np.asarray(tend), rtol=_RTOL,
                                   atol=_RTOL * float(jnp.abs(tend).max()))
        np.testing.assert_allclose(np.asarray(p_sub[sl]), np.asarray(sfc), rtol=_RTOL, atol=0)


def test_empty_column_batch_returns_empty_outputs():
    """The early exit takes max(nstep) over columns; an empty batch must not
    raise (codex: jnp.max of an empty array has no identity)."""
    q = jnp.zeros((0, 20)); rho = jnp.ones((0, 20)); Vt = jnp.ones((0, 20)); dz = jnp.ones((0, 20))
    tend, sfc = sedimentation_tendency(q, rho, Vt, dz, dt=10.0, return_surface_flux=True,
                                       n_substeps_max=8)
    assert tend.shape == (0, 20) and sfc.shape == (0,)
