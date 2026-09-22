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
_RTOL = 1e-12 if jax.config.jax_enable_x64 else 1e-5


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

def test_morrison_gate_off_is_default_and_on_reaches_the_helper(monkeypatch):
    sys.path.insert(0, "tests/unit")
    from test_physics_microphysics import _make_column

    from legoesm.atmosphere.physics.microphysics import morrison as mor
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    assert MorrisonConfig().sed_cfl_substeps is False
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
                                    config=MorrisonConfig())
    assert set(seen) == {1}
    seen.clear()
    on = mor.morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, 600.0,
                                   config=MorrisonConfig(sed_cfl_substeps=True))
    assert set(seen) == {mor._SEDIMENTATION_SUBSTEPS_MAX}
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
    on = apply_microphysics_experiment_flags(
        MorrisonConfig(), "morrison", morrison_sed_cfl_substeps=True)
    assert on.sed_cfl_substeps is True
    with pytest.raises(ValueError, match="morrison_sed_cfl_substeps"):
        apply_microphysics_experiment_flags(
            ThompsonConfig(), "thompson", morrison_sed_cfl_substeps=True)
    from legoesm.driver.config import ExperimentConfig
    assert ExperimentConfig._field_defaults["morrison_sed_cfl_substeps"] is False
    from legoesm.driver.physics_pipeline import thread_morrison_scalars
    cfg = ExperimentConfig(microphysics="morrison", morrison_sed_cfl_substeps=True)
    assert thread_morrison_scalars(cfg, "morrison", MorrisonConfig()).sed_cfl_substeps
    base = MorrisonConfig()
    assert thread_morrison_scalars(ExperimentConfig(microphysics="morrison"),
                                   "morrison", base) is base
    with pytest.raises(ValueError, match="morrison_sed_cfl_substeps"):
        ExperimentConfig(microphysics="thompson",
                         morrison_sed_cfl_substeps=True).validate_strict()


def test_cli_round_trip():
    from scripts.run.run_amip import (
        _postprocess_args,
        build_arg_parser,
        build_config_from_args,
    )
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--microphysics", "morrison",
        "--morrison-sed-cfl-substeps"]), parser))
    assert cfg.morrison_sed_cfl_substeps is True
    d = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert d.morrison_sed_cfl_substeps is False
    from legoesm.driver.config import ExperimentConfig
    rt = ExperimentConfig.from_amip_config(cfg.to_amip_config())
    assert rt.morrison_sed_cfl_substeps is True
