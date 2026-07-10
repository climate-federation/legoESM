"""Tests for the SCM external-forcing API (Phase A of issue-XYZ).

Covers:
  * ``default_forcing`` round-trips through SCM without changing state.
  * Pure-Coriolis inertial oscillation matches the analytical period
    ``T = 2*pi / |f_c|`` under RK4 (the leading-order error of RK4 on a
    pure rotation is O(dt^4) so a 5e-3 fractional tolerance is loose
    enough to absorb any per-step drift over a quarter-period).
  * Large-scale subsidence on a stably stratified column produces
    warming everywhere except the surface end-cell (correct sign).
  * Prescribed horizontal-advection tendency on ``theta`` produces a
    cooling rate equal to the prescribed value (modulo the Exner
    conversion to ``T``).
  * ``validate_forcing`` rejects every documented misconfiguration.
  * Codex iter-1 fixes: legacy 4-arg integrator shim, qv_adv state-
    consistency check, subsidence inflow-endpoint masking, and the
    flat-static pytree registration of :class:`SCMForcing`.

Each test runs with the *no-physics* PhysicsConfig so the forcing
contribution is the only signal in the state delta.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics import (
    ConvectionConfig,
    GravityWaveDragConfig,
    MicrophysicsConfig,
    PhysicsConfig,
    RadiationConfig,
    TurbulenceConfig,
)
from legoesm.atmosphere.scm import (
    SingleColumnModel,
    TIME_INTEGRATORS,
    _apply_tendencies,
    register_time_integrator,
)
from legoesm.atmosphere.scm_forcing import (
    SCMForcing,
    add_tendencies,
    compute_forcing_tendencies,
    default_forcing,
    validate_forcing,
)


NLEV = 16


def _no_physics_cfg() -> PhysicsConfig:
    return PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )


def _make_scm(
    *,
    forcing=None,
    integrator: str = "forward_euler",
    u: float = 0.0,
    v: float = 0.0,
    T_profile=None,
    q_v_profile=None,
    dt: float = 60.0,
) -> SingleColumnModel:
    if T_profile is None:
        T_profile = jnp.linspace(220.0, 295.0, NLEV)
    return SingleColumnModel.create(
        physics_config=_no_physics_cfg(),
        nlev=NLEV,
        dt=dt,
        T_profile=T_profile,
        q_v_profile=q_v_profile,
        u=u,
        v=v,
        time_integrator=integrator,
        forcing=forcing,
    )


# ----------------------------------------------------------------------
# Identity / backward-compat
# ----------------------------------------------------------------------


def test_default_forcing_is_identity_with_no_physics():
    """``forcing=None`` ≡ ``forcing=default_forcing()`` ≡ baseline SCM."""
    T0 = jnp.linspace(220.0, 295.0, NLEV)
    scm_none = _make_scm(forcing=None, T_profile=T0)
    scm_default = _make_scm(forcing=default_forcing(), T_profile=T0)

    final_none, _ = scm_none.run(nsteps=5)
    final_default, _ = scm_default.run(nsteps=5)

    np.testing.assert_array_equal(final_none.T.data, final_default.T.data)
    np.testing.assert_array_equal(final_none.u.data, final_default.u.data)
    np.testing.assert_array_equal(final_none.v.data, final_default.v.data)


def test_step_advances_t_seconds():
    """``self.t_seconds`` increments by ``dt`` per ``step()`` call."""
    scm = _make_scm(forcing=default_forcing(), dt=120.0)
    assert scm.t_seconds == 0.0
    scm.step()
    assert scm.t_seconds == 120.0
    scm.step()
    assert scm.t_seconds == 240.0


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_validate_forcing_rejects_unknown_prescribe():
    with pytest.raises(ValueError, match="prescribe="):
        validate_forcing(SCMForcing(prescribe="bogus"))


def test_validate_forcing_T_s_requires_callable():
    with pytest.raises(ValueError, match="T_s"):
        validate_forcing(SCMForcing(prescribe="T_s"))


def test_validate_forcing_fluxes_requires_at_least_one_flux():
    with pytest.raises(ValueError, match="fluxes"):
        validate_forcing(SCMForcing(prescribe="fluxes"))


def test_scm_create_propagates_validation():
    with pytest.raises(ValueError, match="prescribe="):
        _make_scm(forcing=SCMForcing(prescribe="bogus"))


# ----------------------------------------------------------------------
# Coriolis — inertial oscillation
# ----------------------------------------------------------------------


def test_coriolis_inertial_oscillation_quarter_period():
    """Pure Coriolis with no friction, no geostrophic wind: the
    horizontal wind vector rotates clockwise (NH) with period
    ``T = 2*pi / f_c``.  After T/4, (u, v) should rotate from
    ``(u0, 0)`` to approximately ``(0, -u0)``.  RK4 is fourth-order on
    this linear ODE so we use a loose 1 % tolerance to absorb the
    O(dt^4) error over a quarter-period.
    """
    f_c = 1e-4
    u0 = 10.0
    T_period = 2.0 * float(jnp.pi) / f_c
    dt = 30.0
    nsteps = int(round(T_period / 4.0 / dt))

    forcing = SCMForcing(f_c=f_c)
    scm = _make_scm(
        forcing=forcing, integrator="rk4", u=u0, v=0.0, dt=dt,
    )
    scm.run(nsteps=nsteps)

    # Pure Coriolis on (u, v) — closed-form: u = u0 cos(f*t),
    # v = -u0 sin(f*t).  After t = T/4: cos = 0, sin = 1, so
    # (u, v) → (0, -u0).
    u_final = float(scm.state.u.data[0, 0, 0, 0])
    v_final = float(scm.state.v.data[0, 0, 0, 0])
    assert abs(u_final) < 0.01 * u0, f"u not near 0 at T/4: {u_final}"
    assert abs(v_final - (-u0)) < 0.01 * u0, (
        f"v not near -u0 at T/4: {v_final}"
    )


def test_coriolis_geostrophic_steady_state_no_tendency():
    """When ``(u, v) == (u_g, v_g)`` the Coriolis tendency vanishes,
    so a single Euler step preserves the wind."""
    f_c = 1e-4
    u_g = 7.0
    v_g = 0.0
    forcing = SCMForcing(
        f_c=f_c,
        u_geo=lambda t: jnp.full(NLEV, u_g),
        v_geo=lambda t: jnp.full(NLEV, v_g),
    )
    scm = _make_scm(forcing=forcing, u=u_g, v=v_g, dt=300.0)
    scm.step()
    np.testing.assert_allclose(
        np.asarray(scm.state.u.data[0, 0, 0]), u_g, atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(scm.state.v.data[0, 0, 0]), v_g, atol=1e-10,
    )


# ----------------------------------------------------------------------
# Subsidence — sign convention on stable column
# ----------------------------------------------------------------------


def test_subsidence_warms_stable_column():
    """Stable potential-temperature gradient (dθ/dz > 0) under
    subsidence (w_ls < 0) yields warming via -w · dθ/dz > 0 at every
    interior level.

    The test inspects the *tendency* (not an integrated state) so we
    can assert sign at every level without integrator-induced drift
    from the interpolation of the lowest cell where ``q_v`` happens to
    be absent.
    """
    # Stable theta: T decreases with height in K but theta increases
    # with height (because exner decreases faster).  Hand-construct a
    # large stable profile.
    T_profile = jnp.linspace(220.0, 295.0, NLEV)  # top→bottom
    forcing = SCMForcing(
        subsidence_w=lambda t: jnp.full(NLEV, -0.01),  # 1 cm/s subsidence
    )
    scm = _make_scm(forcing=forcing, T_profile=T_profile)
    tend = compute_forcing_tendencies(
        scm.state, scm.sigma_coord, forcing, t_seconds=0.0,
    )
    dT_dt = np.asarray(tend.dT_dt.data[0, 0, 0])
    # Interior levels (skip the two endpoints because the one-sided
    # stencil's truncation can flip sign at the boundary if the
    # initial T profile happens to be linear; the test is about the
    # interior physics).
    assert np.all(dT_dt[1:-1] > 0.0), (
        f"subsidence on a stable column did not warm interior: {dT_dt}"
    )


def test_subsidence_zero_w_is_zero_tendency():
    """``w_ls ≡ 0`` produces an identically-zero tendency contribution."""
    forcing = SCMForcing(subsidence_w=lambda t: jnp.zeros(NLEV))
    scm = _make_scm(forcing=forcing)
    tend = compute_forcing_tendencies(
        scm.state, scm.sigma_coord, forcing, t_seconds=0.0,
    )
    np.testing.assert_allclose(tend.dT_dt.data, 0.0, atol=1e-30)


# ----------------------------------------------------------------------
# Prescribed advective tendency
# ----------------------------------------------------------------------


def test_advective_theta_tendency_matches_prescription():
    """``theta_adv = const`` produces ``dT/dt = exner * const`` exactly
    (no other forcing channels active, no physics).
    """
    theta_rate = -1e-4  # K/s
    forcing = SCMForcing(theta_adv=lambda t: jnp.full(NLEV, theta_rate))
    scm = _make_scm(forcing=forcing)
    tend = compute_forcing_tendencies(
        scm.state, scm.sigma_coord, forcing, t_seconds=0.0,
    )
    p_full = scm.sigma_coord.pressure_at_full(scm.state.p_s.data)
    exner = (p_full / constants.p_ref) ** constants.kappa
    expected = exner[0, 0, 0] * theta_rate
    np.testing.assert_allclose(
        np.asarray(tend.dT_dt.data[0, 0, 0]),
        np.asarray(expected),
        rtol=1e-12,
        atol=0.0,
    )


def test_qv_advective_tendency_routes_to_tracer_dict():
    """``qv_adv`` populates the ``q_v`` slot of ``tracer_tendencies``
    when ``q_v`` is in the state, and the resulting tendency equals
    the prescription element-wise."""
    rate = 1.5e-7
    qv_init = jnp.linspace(1e-6, 1.5e-2, NLEV)
    forcing = SCMForcing(qv_adv=lambda t: jnp.full(NLEV, rate))
    scm = _make_scm(forcing=forcing, q_v_profile=qv_init)
    tend = compute_forcing_tendencies(
        scm.state, scm.sigma_coord, forcing, t_seconds=0.0,
    )
    assert tend.tracer_tendencies is not None
    assert "q_v" in tend.tracer_tendencies
    np.testing.assert_allclose(
        np.asarray(tend.tracer_tendencies["q_v"].data[0, 0, 0]),
        rate,
        rtol=1e-12,
    )


# ----------------------------------------------------------------------
# Time-varying forcing reaches RK stages
# ----------------------------------------------------------------------


def test_rk4_samples_forcing_at_stage_times():
    """Time-varying ``theta_adv(t)`` evaluated by RK4 must use stage
    abscissae ``t, t+dt/2, t+dt/2, t+dt``.  Using a *linear-in-time*
    forcing makes the integrated update analytically tractable: the
    mean tendency over one RK4 step equals exactly the value at
    ``t + dt/2`` (because RK4 is exact on cubic polynomials in t for
    linear ODEs, and a + b·t is cubic-or-less).  Thus the temperature
    change over one step is ``exner * (a + b*(t0 + dt/2)) * dt``.
    """
    a = -1e-4   # K/s
    b = 1e-7    # K/s^2
    dt = 60.0
    t0 = 100.0
    forcing = SCMForcing(theta_adv=lambda t: jnp.full(NLEV, a + b * t))
    scm = _make_scm(
        forcing=forcing, integrator="rk4", T_profile=jnp.full(NLEV, 290.0),
        dt=dt,
    )
    scm.t_seconds = t0
    T_before = np.asarray(scm.state.T.data[0, 0, 0]).copy()
    scm.step()
    T_after = np.asarray(scm.state.T.data[0, 0, 0])
    p_full = scm.sigma_coord.pressure_at_full(scm.state.p_s.data)
    exner = (p_full / constants.p_ref) ** constants.kappa
    expected_dT = (
        np.asarray(exner[0, 0, 0]) * (a + b * (t0 + 0.5 * dt)) * dt
    )
    np.testing.assert_allclose(T_after - T_before, expected_dT, rtol=1e-9)


# ----------------------------------------------------------------------
# add_tendencies utility
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Codex iter-1 regression fixes
# ----------------------------------------------------------------------


def test_legacy_4arg_integrator_registers_and_runs_with_no_forcing():
    """Pre-Phase-A integrators (4-arg signature) must still register and
    drive a baseline run when forcing is absent.  The compatibility
    shim emits a DeprecationWarning at registration; the integrator
    itself behaves like forward Euler when the closure is called twice.
    """
    def legacy_double_euler(state, phys, f, dt):
        tend, phys_out = f(state, phys)
        mid = _apply_tendencies(state, tend, 0.5 * dt)
        tend2, phys_out2 = f(mid, phys_out)
        return _apply_tendencies(mid, tend2, 0.5 * dt), phys_out2

    with pytest.warns(DeprecationWarning, match="legacy 4-arg signature"):
        register_time_integrator("legacy_test", legacy_double_euler)
    assert "legacy_test" in TIME_INTEGRATORS

    cfg = _no_physics_cfg()
    scm = SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=300.0,
        T_profile=jnp.linspace(220.0, 295.0, NLEV),
        time_integrator="legacy_test",
    )
    final, _ = scm.run(nsteps=3)
    assert jnp.all(jnp.isfinite(final.T.data))


def test_register_time_integrator_rejects_bad_arity():
    """Anything other than 4-or-5 positional args is rejected at
    registration time (TypeError), not at first call."""
    def too_few(state, phys):
        return state, phys

    with pytest.raises(TypeError, match="4 or 5 positional"):
        register_time_integrator("too_few_test", too_few)


def test_register_time_integrator_rejects_keyword_only_t():
    """Codex iter-2 medium: a 5-parameter callable where ``t`` is
    keyword-only would *count* as 5 by ``len(sig.parameters)`` but
    the driver invokes it positionally, so it must be rejected at
    registration time rather than failing on first ``step()``.
    """
    def kwonly_t(state, phys, f, dt, *, t):
        return state, phys

    with pytest.raises(TypeError, match="keyword-only"):
        register_time_integrator("kwonly_t_test", kwonly_t)


def test_register_time_integrator_rejects_varargs():
    """Variadic positional signatures can't have their arity verified;
    reject explicitly rather than silently accepting."""
    def variadic(*args, **kwargs):
        return args[0], args[1]

    with pytest.raises(TypeError, match="variadic positional"):
        register_time_integrator("variadic_test", variadic)


def test_register_time_integrator_accepts_pos_only():
    """Positional-only parameters (PEP 570) satisfy the contract."""
    def pos_only(state, phys, f, dt, t, /):
        from legoesm.atmosphere.scm import _apply_tendencies
        tend, phys_out = f(state, phys, t)
        return _apply_tendencies(state, tend, dt), phys_out

    register_time_integrator("pos_only_test", pos_only)
    assert "pos_only_test" in TIME_INTEGRATORS


def test_register_time_integrator_rejects_non_callable():
    """Codex iter-3 medium: a non-callable object cannot be silently
    accepted just because ``inspect.signature`` happens to raise on it.
    """
    with pytest.raises(TypeError, match="must be callable"):
        register_time_integrator("not_callable_test", 42)


def test_register_time_integrator_rejects_uninspectable():
    """Codex iter-3 medium: any callable whose signature cannot be
    inspected (raw ``classmethod`` descriptors, certain C wrappers)
    must be rejected at registration rather than crashing on first
    ``step()``.
    """
    class _StepDescriptor:
        @classmethod
        def __call__(cls, *args, **kwargs):     # pragma: no cover
            raise NotImplementedError

    # ``classmethod`` descriptors are callable but ``inspect.signature``
    # raises ValueError on the raw descriptor; if Python's behaviour
    # changes and the call succeeds, the test still confirms that we
    # don't silently accept an invalid signature.
    cm = classmethod(lambda cls, a, b, c, d, e: None)
    with pytest.raises(TypeError):
        register_time_integrator("descriptor_test", cm)


def test_legacy_4arg_integrator_rejects_forcing():
    """Codex iter-3 medium: pairing a legacy 4-arg integrator with a
    non-None forcing must hard-error at construction.  The legacy
    shim samples forcing at the outer-step time only, which would
    silently corrupt time-dependent forcing if allowed.
    """
    def legacy_euler(state, phys, f, dt):
        tend, phys_out = f(state, phys)
        return _apply_tendencies(state, tend, dt), phys_out

    with pytest.warns(DeprecationWarning):
        register_time_integrator("legacy_for_forcing_test", legacy_euler)

    forcing = SCMForcing(f_c=1e-4, u_geo=lambda t: jnp.zeros(NLEV))
    with pytest.raises(ValueError, match="legacy 4-arg"):
        SingleColumnModel.create(
            physics_config=_no_physics_cfg(),
            nlev=NLEV, dt=60.0,
            T_profile=jnp.linspace(220.0, 295.0, NLEV),
            time_integrator="legacy_for_forcing_test",
            forcing=forcing,
        )


def test_coriolis_requires_v_field():
    """Codex iter-4 medium: Coriolis forcing on a state with no
    meridional wind (MPAS-style ``state.v is None``) must hard-error
    at construction rather than silently producing zero wind forcing.
    """
    state = SingleColumnModel.create(
        physics_config=_no_physics_cfg(),
        nlev=NLEV, dt=60.0,
        T_profile=jnp.linspace(220.0, 295.0, NLEV),
    ).state
    state_no_v = state._replace(v=None)

    forcing = SCMForcing(f_c=1e-4, u_geo=lambda t: jnp.zeros(NLEV))
    from legoesm.atmosphere.scm_forcing import validate_forcing_against_state
    with pytest.raises(ValueError, match="f_c != 0"):
        validate_forcing_against_state(forcing, state_no_v)


def test_compute_forcing_tendencies_rejects_invalid_prescribe():
    """Codex Phase D iter-4 medium: ``compute_forcing_tendencies`` is
    a public assembler.  Direct callers must hit ``validate_forcing``
    (state-blind) *and* ``validate_forcing_against_state``, so dispatch
    typos and empty prescribed-flux configs raise at the boundary
    instead of silently returning zero tendencies."""
    scm = _make_scm()
    # Invalid prescribe spelling.
    with pytest.raises(ValueError, match="prescribe="):
        compute_forcing_tendencies(
            scm.state, scm.sigma_coord,
            SCMForcing(prescribe="flxes"),
            t_seconds=0.0,
        )
    # prescribe='fluxes' with no flux callables.
    with pytest.raises(ValueError, match="prescribe='fluxes' requires"):
        compute_forcing_tendencies(
            scm.state, scm.sigma_coord,
            SCMForcing(prescribe="fluxes"),
            t_seconds=0.0,
        )


@pytest.mark.parametrize(
    "channel,kwargs",
    [
        ("subsidence_w", dict(subsidence_w=lambda t: jnp.zeros((4, 4)))),
        ("theta_adv", dict(theta_adv=lambda t: jnp.zeros((4, 4)))),
        ("qv_adv", dict(qv_adv=lambda t: jnp.zeros((4, 4)))),
        ("u_geo", dict(f_c=1e-4, u_geo=lambda t: jnp.zeros((4, 4)))),
        ("v_geo", dict(f_c=1e-4, v_geo=lambda t: jnp.zeros((4, 4)))),
    ],
)
def test_profile_callable_wrong_shape_raises(channel, kwargs):
    """Codex iter-6 medium: every profile-valued forcing channel must
    reject same-size non-(nlev,) returns rather than reshaping them as
    a 1-D profile (silent scientific corruption of vertical structure).
    The test uses NLEV=16 so a ``(4, 4)`` table has size==NLEV but the
    wrong rank — exactly the bug the channel-uniform validation is
    paid to catch.
    """
    qv_profile = jnp.linspace(1e-6, 1.5e-2, NLEV)
    forcing = SCMForcing(**kwargs)
    scm = _make_scm(forcing=forcing, q_v_profile=qv_profile)
    with pytest.raises(ValueError, match="expected"):
        compute_forcing_tendencies(
            scm.state, scm.sigma_coord, forcing, t_seconds=0.0,
        )


def test_compute_forcing_tendencies_public_api_validates_state():
    """Codex iter-5 medium: ``compute_forcing_tendencies`` is a public
    assembler; direct callers must hit the same state-dependent
    validation as :class:`SingleColumnModel.__init__`.
    """
    scm = _make_scm()                        # state with v present
    state_no_v = scm.state._replace(v=None)
    bad_coriolis = SCMForcing(
        f_c=1e-4, u_geo=lambda t: jnp.zeros(NLEV),
    )
    with pytest.raises(ValueError, match="f_c != 0"):
        compute_forcing_tendencies(
            state_no_v, scm.sigma_coord, bad_coriolis, t_seconds=0.0,
        )

    # ... and likewise for qv_adv on a dry state.
    dry_scm = _make_scm(q_v_profile=None)
    bad_qv = SCMForcing(qv_adv=lambda t: jnp.zeros(NLEV))
    with pytest.raises(ValueError, match="qv_adv"):
        compute_forcing_tendencies(
            dry_scm.state, dry_scm.sigma_coord, bad_qv, t_seconds=0.0,
        )


def test_legacy_4arg_integrator_allowed_when_forcing_is_none():
    """Sanity-symmetric to the previous test: a legacy integrator must
    *still* work when forcing is omitted, preserving the documented
    backwards-compatible path."""
    def legacy_euler(state, phys, f, dt):
        tend, phys_out = f(state, phys)
        return _apply_tendencies(state, tend, dt), phys_out

    with pytest.warns(DeprecationWarning):
        register_time_integrator("legacy_no_forcing_test", legacy_euler)
    scm = SingleColumnModel.create(
        physics_config=_no_physics_cfg(),
        nlev=NLEV, dt=60.0,
        T_profile=jnp.linspace(220.0, 295.0, NLEV),
        time_integrator="legacy_no_forcing_test",
        forcing=None,
    )
    final, _ = scm.run(nsteps=2)
    assert jnp.all(jnp.isfinite(final.T.data))


def test_qv_adv_without_qv_tracer_raises_at_construction():
    """Codex high-severity: configuring ``qv_adv`` on a state that
    carries no ``q_v`` tracer must raise at construction time rather
    than silently dropping the moisture tendency on every step.
    """
    forcing = SCMForcing(qv_adv=lambda t: jnp.zeros(NLEV))
    with pytest.raises(ValueError, match="qv_adv"):
        SingleColumnModel.create(
            physics_config=_no_physics_cfg(),
            nlev=NLEV, dt=300.0,
            T_profile=jnp.linspace(220.0, 295.0, NLEV),
            q_v_profile=None,                 # <-- no q_v tracer
            forcing=forcing,
        )


def test_qv_adv_with_qv_tracer_passes_construction():
    """Sanity: the symmetric case (qv_adv + q_v tracer present) must
    *not* raise."""
    forcing = SCMForcing(qv_adv=lambda t: jnp.full(NLEV, 1e-8))
    scm = SingleColumnModel.create(
        physics_config=_no_physics_cfg(),
        nlev=NLEV, dt=300.0,
        T_profile=jnp.linspace(220.0, 295.0, NLEV),
        q_v_profile=jnp.linspace(1e-6, 1.5e-2, NLEV),
        forcing=forcing,
    )
    final, _ = scm.run(nsteps=2)
    assert jnp.all(jnp.isfinite(final.tracers["q_v"].data))


def test_subsidence_top_inflow_tendency_is_masked_to_zero():
    """Top-of-column inflow (``w[0] < 0``) has no upstream donor outside
    the SCM domain.  The tendency at the top level must be exactly
    zero, regardless of whatever the duplicated interior stencil would
    have produced.
    """
    # Uniform downward velocity throughout the column — top cell is
    # therefore an inflow boundary on every step.
    forcing = SCMForcing(
        subsidence_w=lambda t: jnp.full(NLEV, -0.05),  # 5 cm/s downward
    )
    scm = _make_scm(forcing=forcing)
    tend = compute_forcing_tendencies(
        scm.state, scm.sigma_coord, forcing, t_seconds=0.0,
    )
    dT_dt = np.asarray(tend.dT_dt.data[0, 0, 0])
    assert dT_dt[0] == 0.0, (
        f"top-of-column inflow tendency not masked: dT_dt[0]={dT_dt[0]}"
    )
    # Interior levels still warm under stable subsidence — sanity check
    # that the mask did not zero everything.
    assert np.any(dT_dt[1:-1] != 0.0)


def test_subsidence_surface_inflow_tendency_is_masked_to_zero():
    """Surface inflow (``w[-1] > 0``) has no upstream donor below the
    surface — the surface tendency must be zero.
    """
    forcing = SCMForcing(
        subsidence_w=lambda t: jnp.full(NLEV, +0.05),  # 5 cm/s upward
    )
    scm = _make_scm(forcing=forcing)
    tend = compute_forcing_tendencies(
        scm.state, scm.sigma_coord, forcing, t_seconds=0.0,
    )
    dT_dt = np.asarray(tend.dT_dt.data[0, 0, 0])
    assert dT_dt[-1] == 0.0, (
        f"surface inflow tendency not masked: dT_dt[-1]={dT_dt[-1]}"
    )


def test_subsidence_qv_inflow_endpoints_masked():
    """Same inflow-endpoint contract for the q_v subsidence channel."""
    qv = jnp.linspace(1e-6, 1.5e-2, NLEV)
    forcing = SCMForcing(
        subsidence_w=lambda t: jnp.full(NLEV, -0.05),
    )
    scm = _make_scm(forcing=forcing, q_v_profile=qv)
    tend = compute_forcing_tendencies(
        scm.state, scm.sigma_coord, forcing, t_seconds=0.0,
    )
    assert tend.tracer_tendencies is not None
    dqv = np.asarray(tend.tracer_tendencies["q_v"].data[0, 0, 0])
    assert dqv[0] == 0.0, f"q_v top inflow not masked: {dqv[0]}"


def test_scm_forcing_is_flat_static_pytree():
    """``SCMForcing`` is registered as a flat-static pytree so neither
    callables nor the ``prescribe`` string appear as JAX tree leaves —
    generic ``jax.tree_util`` consumers (loss reducers, optimizer state
    walkers) see an empty leaf list.
    """
    forcing = SCMForcing(
        f_c=1e-4,
        u_geo=lambda t: jnp.zeros(NLEV),
        theta_adv=lambda t: jnp.zeros(NLEV),
        prescribe="T_s",
        T_s=lambda t: jnp.asarray(288.0),
    )
    leaves = jax.tree_util.tree_leaves(forcing)
    assert leaves == [], f"SCMForcing leaked non-static leaves: {leaves}"
    # Round-trip through tree_flatten / tree_unflatten preserves identity.
    flat, treedef = jax.tree_util.tree_flatten(forcing)
    restored = jax.tree_util.tree_unflatten(treedef, flat)
    assert restored is forcing


# ----------------------------------------------------------------------
# Phase B v2: prescribed-surface forcing via PhysicsState override
# ----------------------------------------------------------------------


def test_prescribe_T_s_with_zero_physics_does_not_modify_state():
    """With no turbulence consuming the prescribed surface T, the
    column must evolve under whatever else the forcing contributes —
    in this case nothing.  The integration is therefore the identity
    on every cell.  This catches any regression where the old
    state.T[-1] mutation path silently came back.
    """
    T0 = jnp.linspace(220.0, 295.0, NLEV)
    T_s_fn = lambda t: 320.0
    forcing = SCMForcing(prescribe="T_s", T_s=T_s_fn)
    scm = _make_scm(forcing=forcing, T_profile=T0)
    scm.run(nsteps=3)
    np.testing.assert_allclose(
        np.asarray(scm.state.T.data[0, 0, 0]), np.asarray(T0), atol=1e-12,
    )


def test_prescribe_T_s_drives_sensible_flux_with_turbulence():
    """The whole point of ``prescribe='T_s'`` is to drive the
    turbulence scheme's bulk sensible heat flux with a prescribed
    surface temperature *distinct* from the lowest air temperature.

    Run with the Louis turbulence scheme, ``T_s`` warmer than the air
    column by 20 K.  The bulk-flux gradient ``T_s - T[..., -1]`` is
    positive, so the lowest air cell must warm over the run.  If the
    legacy ``state.T[-1] = T_s`` collapse-the-gradient bug returned
    (Phase B iter-1 high finding), this test fails because no
    sensible flux is produced.
    """
    T0 = jnp.full(NLEV, 280.0)
    T_s_fn = lambda t: jnp.asarray(300.0)           # 20 K warmer than column
    forcing = SCMForcing(prescribe="T_s", T_s=T_s_fn)
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="louis"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    scm = SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=60.0,
        T_profile=T0, q_v_profile=jnp.full(NLEV, 5e-3),
        u=5.0,  # need wind for bulk flux
        forcing=forcing,
    )
    scm.run(nsteps=20)
    T_lowest = float(scm.state.T.data[0, 0, 0, -1])
    # The bulk-flux gradient ``T_s - T[..., -1] = 20 K`` drives a
    # kinematic sensible flux of ``Ch * |V| * ΔT ~ 1.5e-3 * 5 * 20 =
    # 0.15 K m/s``.  Distributed over the lowest layer's dz_bot
    # (~tens of metres) and 1200 s of integration, the lowest cell
    # warms by ~0.1 K under the Louis scheme.  A buggy zero-gradient
    # implementation gives exactly 0 K of warming, so any positive
    # delta of >= 0.05 K validates the fix.
    assert T_lowest - 280.0 > 0.05, (
        f"prescribed T_s=300 with turbulence did not warm lowest cell "
        f"(stayed at {T_lowest}) — likely the T[-1]-overwrite bug "
        "from Phase B v1 returned, collapsing the bulk-flux gradient."
    )


def test_phys_state_default_override_is_nan():
    """The default ``PhysicsState.surface_T_sfc_override`` must be the finite
    ``NO_SFC_T_OVERRIDE`` sentinel (#911, was NaN) so that turbulence's
    ``_resolve_T_sfc`` falls back to the lowest air temperature for every 3-D
    run — while keeping the state finite."""
    from legoesm.atmosphere.physics.physics_state import NO_SFC_T_OVERRIDE
    scm = _make_scm()
    override = np.asarray(scm.phys_state.surface_T_sfc_override)
    assert override.shape == (1,)
    assert np.all(np.isfinite(override))
    assert np.allclose(override, NO_SFC_T_OVERRIDE)


def test_prescribe_T_s_populates_phys_state_override():
    """After a ``step()`` with ``prescribe='T_s'`` *and* active
    physics, the persistent ``phys_state.surface_T_sfc_override`` must
    hold the most recent injected T_s value (the tendency closure
    writes it each stage and ``physics_fn``'s ``update_physics_state``
    preserves it on return).

    Active physics is required because Phase D codex iter-2 removed
    the end-of-step re-stamp to avoid double-firing the T_s callable;
    zero-physics runs have ``new_phys=None`` so the persistent override
    is not refreshed (observationally invisible — no module reads it
    between stages).
    """
    T_s_fn = lambda t: jnp.asarray(285.0 + 0.01 * t)
    forcing = SCMForcing(prescribe="T_s", T_s=T_s_fn)
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray", diurnal_cycle=False),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    scm = SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=60.0,
        T_profile=jnp.linspace(220.0, 295.0, NLEV),
        forcing=forcing,
    )
    scm.step()
    override = float(scm.phys_state.surface_T_sfc_override[0])
    # ``_tend_fn`` injects T_s at the outer-step time (forward Euler
    # has a single stage at t=0), so the persistent override is
    # ``T_s(t=0)`` after one step.
    expected = float(T_s_fn(0.0))
    np.testing.assert_allclose(override, expected, atol=1e-12)


def test_prescribe_fluxes_warms_lowest_cell():
    """``prescribe='fluxes'`` with positive ``w_th_s`` (upward kinematic
    sensible heat flux) injects ``w_th_s / dz_bot`` into ``dT/dt`` at
    the lowest cell.  Over ``nsteps``, the lowest cell warms by the
    flux divergence times total elapsed time.
    """
    w_th_value = 0.05  # K m/s — strong surface heating
    forcing = SCMForcing(
        prescribe="fluxes",
        w_th_s=lambda t: jnp.asarray(w_th_value),
        w_qv_s=lambda t: jnp.asarray(0.0),
    )
    T0 = jnp.full(NLEV, 280.0)
    qv0 = jnp.full(NLEV, 5e-3)
    nsteps = 10
    dt = 60.0
    scm = _make_scm(
        forcing=forcing, T_profile=T0, q_v_profile=qv0, dt=dt,
    )
    scm.run(nsteps=nsteps)
    T_lowest = float(scm.state.T.data[0, 0, 0, -1])
    assert T_lowest > 280.0, f"prescribed +w_th did not warm: {T_lowest}"
    assert T_lowest - 280.0 < 50.0


def test_shared_physics_fn_radiation_hook_isolated_between_scms():
    """Codex Phase B v2 iter-5 high: if two SCM instances share one
    ``physics_fn`` (passed explicitly to the constructor rather than
    built via ``create()``), the prescribed-T_s radiation hook must not
    leak from the forced SCM into the unforced SCM.  try/finally cleanup
    ensures the override cell is cleared after every physics call.
    """
    from legoesm.atmosphere.physics.combined import make_physics
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.scm import (
        make_column_state, make_scm_grid,
    )
    from legoesm.grids.vertical import create_sigma_coordinate

    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray", diurnal_cycle=False),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    shared_physics_fn = make_physics(cfg, model_type="hydrostatic", dt=60.0)
    grid = make_scm_grid(latitude_deg=0.0)
    sigma_coord = create_sigma_coordinate(NLEV, sigma_top=0.01)
    T0 = jnp.full(NLEV, 280.0)
    state_a = make_column_state(NLEV, T_profile=T0)
    state_b = make_column_state(NLEV, T_profile=T0)
    phys_state_a = init_physics_state(ncol=1, nlev=NLEV, physics_config=cfg)
    phys_state_b = init_physics_state(ncol=1, nlev=NLEV, physics_config=cfg)

    forcing_warm = SCMForcing(
        prescribe="T_s", T_s=lambda t: jnp.asarray(350.0),
    )
    scm_warm = SingleColumnModel(
        physics_fn=shared_physics_fn,
        state=state_a,
        phys_state=phys_state_a,
        grid=grid,
        sigma_coord=sigma_coord,
        dt=60.0,
        forcing=forcing_warm,
    )
    scm_plain = SingleColumnModel(
        physics_fn=shared_physics_fn,
        state=state_b,
        phys_state=phys_state_b,
        grid=grid,
        sigma_coord=sigma_coord,
        dt=60.0,
        forcing=None,  # no forcing
    )

    scm_warm.step()
    # After scm_warm.step(), the radiation override cell on
    # shared_physics_fn must have been cleared by the try/finally
    # in _tendency_fn.  Step scm_plain and verify its lowest cell
    # was driven by T[..., -1] = 280, not by the warm override.
    scm_plain.step()
    T_plain = float(scm_plain.state.T.data[0, 0, 0, -1])
    # Sanity bound: gray radiation with T_sfc=280 gives a small
    # cooling tendency; T_plain should be within a few K of 280.
    # If the override leaked (T_sfc=350), radiation would cool the
    # surface much faster (sigma * T^4 is highly nonlinear).
    assert abs(T_plain - 280.0) < 1.0, (
        f"Override leaked from scm_warm to scm_plain: "
        f"T_plain={T_plain} (expected near 280 K)"
    )


def test_apply_T_sfc_override_rejects_broadcasting_overrides():
    """Codex Phase B v2 iter-4 medium: a scalar or shape-(1,) override
    against multi-column T_sfc must raise rather than silently
    broadcast one value to every column.
    """
    from legoesm.atmosphere.physics.radiation.integration import (
        _apply_T_sfc_override,
    )
    T_sfc = jnp.array([280.0, 285.0, 290.0])
    bad_scalar = jnp.asarray(300.0)
    bad_one = jnp.array([300.0])
    with pytest.raises(ValueError, match="override shape"):
        _apply_T_sfc_override(T_sfc, bad_scalar)
    with pytest.raises(ValueError, match="override shape"):
        _apply_T_sfc_override(T_sfc, bad_one)
    # Correct shape (3,) passes through.
    good = jnp.array([301.0, 302.0, 303.0])
    out = _apply_T_sfc_override(T_sfc, good)
    np.testing.assert_array_equal(out, good)


def test_prescribe_T_s_drives_radiative_surface_boundary():
    """Codex Phase B v2 iter-2 high: prescribed T_s must also propagate
    to the radiation surface boundary so surface longwave emission is
    consistent with turbulence's bulk-flux boundary.  Differential
    test: run two SCMs with the same atmospheric column and gray
    radiation, differing only in ``T_s`` (warm vs cold).  The lowest
    cell T after a few steps must differ — radiation must respond to
    the prescribed skin temperature, not just to ``T[..., -1]``.
    """
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray", diurnal_cycle=False),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    T0 = jnp.full(NLEV, 280.0)

    forcing_warm = SCMForcing(
        prescribe="T_s", T_s=lambda t: jnp.asarray(320.0),
    )
    scm_warm = SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=300.0,
        T_profile=T0, latitude_deg=0.0,
        forcing=forcing_warm,
    )
    scm_warm.run(nsteps=5)
    T_warm = float(scm_warm.state.T.data[0, 0, 0, -1])

    forcing_cold = SCMForcing(
        prescribe="T_s", T_s=lambda t: jnp.asarray(240.0),
    )
    scm_cold = SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=300.0,
        T_profile=T0, latitude_deg=0.0,
        forcing=forcing_cold,
    )
    scm_cold.run(nsteps=5)
    T_cold = float(scm_cold.state.T.data[0, 0, 0, -1])

    # With turbulence disabled, the only path by which T_s affects
    # the lowest cell is through the radiation surface boundary.
    # A 80 K spread in T_s must produce a measurable spread in the
    # lowest-cell tendency-integrated temperature.  If the radiation
    # hook is missing, both runs see the same T_sfc = T[..., -1] = 280
    # and T_warm == T_cold.
    assert abs(T_warm - T_cold) > 0.01, (
        "Prescribed T_s did not propagate to radiation surface "
        f"boundary: T_warm={T_warm}, T_cold={T_cold} (should differ "
        "for an 80 K spread in T_s)."
    )


def test_prescribe_T_s_rejects_non_finite():
    """Codex Phase B v2 iter-1 high: the NaN sentinel inside
    ``_resolve_T_sfc`` means a missing value in the user's ``T_s(t)``
    time series would silently revert to ``T_col[:, -1]``.  Reject at
    injection so the failure is loud, not silent.
    """
    forcing = SCMForcing(prescribe="T_s", T_s=lambda t: jnp.asarray(jnp.nan))
    scm = _make_scm(forcing=forcing)
    with pytest.raises(ValueError, match="non-finite"):
        scm.step()

    forcing_inf = SCMForcing(prescribe="T_s", T_s=lambda t: jnp.asarray(jnp.inf))
    scm_inf = _make_scm(forcing=forcing_inf)
    with pytest.raises(ValueError, match="non-finite"):
        scm_inf.step()


def test_prescribe_fluxes_with_nonzero_Ch_neutral_raises_double_count():
    """Phase F deferred-item #2: ``prescribe='fluxes'`` paired with a
    turbulence scheme that has a non-zero ``surface.Ch_neutral``
    double-counts the surface sensible/latent flux (once via the
    prescribed-flux tendency, once via the bulk-formula bottom BC of
    the implicit-diffusion solver).  SCM construction must
    fail-fast.
    """
    from legoesm.atmosphere.physics.turbulence.config import (
        MYNN25Config, SurfaceLayerConfig,
    )
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(
            scheme="mynn25",
            mynn25=MYNN25Config(
                surface=SurfaceLayerConfig(
                    Cd_neutral=1.5e-3, Ch_neutral=1.5e-3,
                ),
            ),
        ),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    forcing = SCMForcing(
        prescribe="fluxes",
        w_th_s=lambda t: jnp.asarray(0.1),
    )
    with pytest.raises(ValueError, match="double-count"):
        SingleColumnModel.create(
            physics_config=cfg, nlev=NLEV, dt=10.0,
            T_profile=jnp.linspace(270.0, 295.0, NLEV),
            q_v_profile=jnp.zeros(NLEV),
            forcing=forcing,
        )


def test_prescribe_fluxes_with_MOST_bulk_scheme_raises():
    """Phase F fix #2 codex iter-1 high: ``bulk_scheme='coare3'`` and
    ``'large_yeager'`` solve MOST iteratively and ignore
    ``Ch_neutral``, so setting ``Ch_neutral=0`` does NOT suppress the
    bulk-formula heat flux.  SCM must reject these bulk schemes when
    paired with ``prescribe='fluxes'``.
    """
    from legoesm.atmosphere.physics.turbulence.config import (
        MYNN25Config, SurfaceLayerConfig,
    )
    for bs in ("coare3", "large_yeager"):
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(
                scheme="mynn25",
                mynn25=MYNN25Config(
                    surface=SurfaceLayerConfig(
                        Cd_neutral=1.5e-3, Ch_neutral=0.0,
                        bulk_scheme=bs,
                    ),
                ),
            ),
            microphysics=MicrophysicsConfig(scheme="none"),
            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
        )
        forcing = SCMForcing(
            prescribe="fluxes", w_th_s=lambda t: jnp.asarray(0.1),
        )
        with pytest.raises(ValueError, match="bulk_scheme"):
            SingleColumnModel.create(
                physics_config=cfg, nlev=NLEV, dt=10.0,
                T_profile=jnp.linspace(270.0, 295.0, NLEV),
                q_v_profile=jnp.zeros(NLEV),
                forcing=forcing,
            )


def test_direct_constructor_requires_physics_config_for_prescribed_fluxes():
    """Phase F fix #2 codex iter-1 medium: direct
    ``SingleColumnModel(...)`` construction bypasses the validation
    that ``create()`` runs.  With ``prescribe='fluxes'`` the
    constructor must demand ``physics_config`` so the no-double-count
    check can fire (or refuse construction)."""
    from legoesm.atmosphere.physics.combined import make_physics
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.scm import make_column_state, make_scm_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    cfg = _no_physics_cfg()
    fn = make_physics(cfg, model_type="hydrostatic", dt=10.0)
    state = make_column_state(
        NLEV, T_profile=jnp.full(NLEV, 280.0),
        q_v_profile=jnp.zeros(NLEV),
    )
    grid = make_scm_grid(latitude_deg=0.0)
    sigma_coord = create_sigma_coordinate(NLEV, sigma_top=0.01)
    phys_state = init_physics_state(ncol=1, nlev=NLEV, physics_config=cfg)
    forcing = SCMForcing(
        prescribe="fluxes", w_th_s=lambda t: jnp.asarray(0.1),
    )
    # Phase F fix #2 codex iter-2 medium: direct construction is
    # rejected outright because the constructor cannot verify that
    # ``physics_fn`` was built from the supplied (or any) config.
    with pytest.raises(ValueError, match="Route through|create"):
        SingleColumnModel(
            physics_fn=fn, state=state, phys_state=phys_state,
            grid=grid, sigma_coord=sigma_coord, dt=10.0,
            forcing=forcing,
        )
    # Even passing physics_config explicitly does not unlock direct
    # construction — only ``create()`` can pair the validated
    # ``physics_config`` with a config-derived ``physics_fn``.
    with pytest.raises(ValueError, match="Route through|create"):
        SingleColumnModel(
            physics_fn=fn, state=state, phys_state=phys_state,
            grid=grid, sigma_coord=sigma_coord, dt=10.0,
            forcing=forcing,
            physics_config=cfg,   # offered but ignored on the direct path
        )


def test_prescribe_fluxes_with_zero_Ch_neutral_passes():
    """Symmetric to the rejection: the documented Phase B v2 workaround
    (``Ch_neutral=0``) constructs successfully and runs without
    double-counting."""
    from legoesm.atmosphere.physics.turbulence.config import (
        MYNN25Config, SurfaceLayerConfig,
    )
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(
            scheme="mynn25",
            mynn25=MYNN25Config(
                surface=SurfaceLayerConfig(
                    Cd_neutral=1.5e-3, Ch_neutral=0.0,
                ),
            ),
        ),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    forcing = SCMForcing(
        prescribe="fluxes",
        w_th_s=lambda t: jnp.asarray(0.1),
    )
    scm = SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=10.0,
        T_profile=jnp.linspace(270.0, 295.0, NLEV),
        q_v_profile=jnp.zeros(NLEV),
        forcing=forcing,
    )
    assert scm.forcing.prescribe == "fluxes"


def test_prescribe_fluxes_w_qv_without_qv_tracer_raises_at_init():
    """Codex iter-1 medium #2: prescribe='fluxes' with ``w_qv_s`` but
    no ``q_v`` tracer must raise at SCM construction time (not at
    first step).  The state-dependent validator now catches this
    symmetric to the qv_adv check.
    """
    forcing = SCMForcing(
        prescribe="fluxes",
        w_qv_s=lambda t: jnp.asarray(1e-5),
    )
    with pytest.raises(ValueError, match="prescribe='fluxes' with w_qv_s"):
        SingleColumnModel.create(
            physics_config=_no_physics_cfg(),
            nlev=NLEV, dt=60.0,
            T_profile=jnp.linspace(220.0, 295.0, NLEV),
            q_v_profile=None,
            forcing=forcing,
        )


def test_add_tendencies_is_field_wise_sum():
    """``add_tendencies`` returns a field-wise sum on the dense
    channels and a key-union sum on tracers."""
    scm = _make_scm()
    f1 = compute_forcing_tendencies(
        scm.state, scm.sigma_coord,
        SCMForcing(theta_adv=lambda t: jnp.full(NLEV, 1.0)),
        t_seconds=0.0,
    )
    f2 = compute_forcing_tendencies(
        scm.state, scm.sigma_coord,
        SCMForcing(theta_adv=lambda t: jnp.full(NLEV, 2.0)),
        t_seconds=0.0,
    )
    summed = add_tendencies(f1, f2)
    np.testing.assert_allclose(
        np.asarray(summed.dT_dt.data),
        np.asarray(f1.dT_dt.data) + np.asarray(f2.dT_dt.data),
        rtol=1e-12,
    )
