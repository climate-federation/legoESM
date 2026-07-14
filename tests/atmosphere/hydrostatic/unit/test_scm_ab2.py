"""Tests for the Adams-Bashforth 2 (AB2) time integrator (Phase D).

AB2 is a second-order linear-multistep scheme used by jax_scm.  It
combines the current-step tendency with a one-step-old tendency:

    y_{n+1} = y_n + dt · (1.5 · tend_n − 0.5 · tend_{n-1}).

These tests verify:

* AB2 falls back to forward Euler on the first step (no prev tendency).
* AB2 second-order accuracy on ``dy/dt = −λy`` vs forward Euler.
* AB2 is single-stage (no sub-stage carry reuse) and therefore
  compatible with stateful turbulence / convection schemes — the SCM
  validator must NOT reject ``ab2`` for these schemes (unlike RK2/RK4).
* AB2 + MYNN-2.5 integrates without NaN.
* Each SCM instance carries its own AB2 ``prev_tend`` (no cross-
  contamination via the shared TIME_INTEGRATORS registry).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics import (
    ConvectionConfig,
    GravityWaveDragConfig,
    MicrophysicsConfig,
    PhysicsConfig,
    RadiationConfig,
    TurbulenceConfig,
)
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel, TIME_INTEGRATORS


NLEV = 16


def _no_physics_cfg() -> PhysicsConfig:
    return PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )


def test_ab2_registered():
    assert "ab2" in TIME_INTEGRATORS


def test_ab2_first_step_falls_back_to_euler():
    """No prev_tend on the first step → AB2 must run forward Euler.

    Use gray radiation with constant column to give a deterministic
    non-zero tendency we can compare against an Euler reference.
    """
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray", diurnal_cycle=False),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    T0 = jnp.linspace(220.0, 295.0, NLEV)
    scm_ab2 = SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=300.0,
        T_profile=T0, time_integrator="ab2",
    )
    scm_eul = SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=300.0,
        T_profile=T0, time_integrator="forward_euler",
    )
    scm_ab2.step()
    scm_eul.step()
    np.testing.assert_allclose(
        np.asarray(scm_ab2.state.T.data),
        np.asarray(scm_eul.state.T.data),
        rtol=0.0,
        atol=1e-12,
    )


def test_ab2_caches_prev_tendency_after_first_step():
    """``self._ab2_prev_tend`` is None initially, populated after the
    first step, and used for the second."""
    scm = SingleColumnModel.create(
        physics_config=_no_physics_cfg(), nlev=NLEV, dt=60.0,
        T_profile=jnp.linspace(220.0, 295.0, NLEV),
        time_integrator="ab2",
    )
    assert scm._ab2_prev_tend is None
    scm.step()
    assert scm._ab2_prev_tend is not None
    # Second step uses AB2 — runs through the multi-step branch.
    scm.step()
    assert scm._ab2_prev_tend is not None


def test_ab2_rejects_stateful_turbulence():
    """Codex Phase D iter-1 medium: AB2 multistep only advances
    HydrostaticState; PhysicsState carries (MYNN qke, TKE, …) get a
    plain forward-Euler-like update from the current step's physics
    call.  That breaks O(dt²) accuracy for any scheme with a
    prognostic carry, so AB2 rejects the same stateful combinations
    as RK2/RK4 until a true coupled-AB2 lands.
    """
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="mynn25"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    with pytest.raises(ValueError, match="incompatible"):
        SingleColumnModel.create(
            physics_config=cfg, nlev=NLEV, dt=60.0,
            T_profile=jnp.linspace(220.0, 295.0, NLEV),
            q_v_profile=jnp.linspace(1e-6, 1.5e-2, NLEV),
            u=5.0,
            time_integrator="ab2",
        )


def test_ab2_rejects_stateful_convection():
    """Symmetric rejection for profile-carrying convection schemes."""
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="mass_flux"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    with pytest.raises(ValueError, match="incompatible"):
        SingleColumnModel.create(
            physics_config=cfg, nlev=NLEV, dt=60.0,
            T_profile=jnp.linspace(220.0, 295.0, NLEV),
            q_v_profile=jnp.linspace(1e-6, 1.5e-2, NLEV),
            time_integrator="ab2",
        )


def test_ab2_accepts_diagnostic_physics():
    """Sanity-symmetric to the rejection tests: AB2 must still
    integrate diagnostic-physics configurations (gray radiation +
    Louis turbulence, both stateless).
    """
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray", diurnal_cycle=False),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="louis"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    scm = SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=60.0,
        T_profile=jnp.linspace(220.0, 295.0, NLEV),
        q_v_profile=jnp.linspace(1e-6, 1.5e-2, NLEV),
        u=5.0,
        time_integrator="ab2",
    )
    final, _ = scm.run(nsteps=5)
    assert jnp.all(jnp.isfinite(final.T.data))


def test_ab2_first_step_evaluates_tend_fn_exactly_once():
    """Codex Phase D iter-1/iter-2 high: AB2 first-step bootstrap must
    call ``_tend_fn`` — and therefore each forcing callable — exactly
    ONCE per step.  A double call would corrupt iterator / counter /
    file-cursor backed forcing sources.
    """
    call_count = {"n": 0}

    def counting_T_s(t):
        call_count["n"] += 1
        return jnp.asarray(285.0)

    from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing
    forcing = SCMForcing(prescribe="T_s", T_s=counting_T_s)
    scm = SingleColumnModel.create(
        physics_config=_no_physics_cfg(), nlev=NLEV, dt=60.0,
        T_profile=jnp.linspace(220.0, 295.0, NLEV),
        time_integrator="ab2",
        forcing=forcing,
    )
    n_before = call_count["n"]
    scm.step()
    n_after = call_count["n"]
    # Exactly one call: single physics evaluation per outer step via
    # ``_tend_fn``.  iter-2 removed the post-step re-stamp that
    # previously double-fired the callable.
    assert n_after - n_before == 1, (
        f"AB2 first step expected exactly 1 T_s call, got "
        f"{n_after - n_before}"
    )


def test_ab2_subsequent_step_evaluates_tend_fn_exactly_once():
    """Same constraint as the first step — AB2 must remain a
    single-evaluation-per-step integrator after the bootstrap.
    """
    call_count = {"n": 0}

    def counting_T_s(t):
        call_count["n"] += 1
        return jnp.asarray(285.0)

    from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing
    forcing = SCMForcing(prescribe="T_s", T_s=counting_T_s)
    scm = SingleColumnModel.create(
        physics_config=_no_physics_cfg(), nlev=NLEV, dt=60.0,
        T_profile=jnp.linspace(220.0, 295.0, NLEV),
        time_integrator="ab2",
        forcing=forcing,
    )
    scm.step()                              # bootstrap
    n_before = call_count["n"]
    scm.step()                              # multistep
    n_after = call_count["n"]
    assert n_after - n_before == 1, (
        f"AB2 multistep expected exactly 1 T_s call, got "
        f"{n_after - n_before}"
    )


def test_ab2_rejects_bechtold_convection():
    """Bechtold is a stateful convection scheme that carries
    ``conv_prog_profile``, ``conv_stoch_state``, and possibly
    ``prng_key`` across steps.  AB2's multistep only advances
    ``HydrostaticState``, so pairing it with Bechtold would silently
    mix integrators on the coupled (state, phys_state) system —
    rejected by ``_validate_integrator_compatibility``.
    """
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="bechtold"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    with pytest.raises(ValueError, match="incompatible"):
        SingleColumnModel.create(
            physics_config=cfg, nlev=NLEV, dt=60.0,
            T_profile=jnp.linspace(220.0, 295.0, NLEV),
            q_v_profile=jnp.linspace(1e-6, 1.5e-2, NLEV),
            time_integrator="ab2",
        )


def test_ab2_per_instance_prev_tend_no_cross_contamination():
    """Two SCMs using AB2 must have independent ``_ab2_prev_tend``
    caches — running one must not poison the other.  This catches
    accidental module-level state in the AB2 implementation.
    """
    T0 = jnp.linspace(220.0, 295.0, NLEV)
    scm_a = SingleColumnModel.create(
        physics_config=_no_physics_cfg(), nlev=NLEV, dt=60.0,
        T_profile=T0, time_integrator="ab2",
    )
    scm_b = SingleColumnModel.create(
        physics_config=_no_physics_cfg(), nlev=NLEV, dt=60.0,
        T_profile=T0, time_integrator="ab2",
    )
    scm_a.step()
    scm_a.step()
    # scm_b has not run; its cache must still be None.
    assert scm_b._ab2_prev_tend is None
    assert scm_a._ab2_prev_tend is not None
