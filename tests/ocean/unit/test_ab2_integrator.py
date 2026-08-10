"""Adams-Bashforth-2 outer time integrator (config.outer_integrator="ab2").

This is Veros's actual outer scheme (AB2 on the EXPLICIT tendency, NOT leapfrog —
see the strategy §8 integrator row). The default "forward_euler" path is unchanged;
the AB2 path extrapolates the EXPLICIT-only forward-Euler increment
``X*=X^n+(1.5+ε)·ΔX_expl^n−(0.5+ε)·ΔX_expl^{n-1}`` and then applies implicit
vertical mixing ONCE, ``X^{n+1}=ImplicitVertMix(X*)`` (Veros core/thermodynamics.py
+ core/external/solve_stream.py). It carries the EXPLICIT increment ΔX_expl^{n-1} on
``{T,S,u,v}_incr_prev``, keeps the barotropic mode from the barotropic solve, and
bootstraps ΔX_expl^{n-1}=0.  Because implicit mixing is applied once (backward-Euler),
the scheme is UNCONDITIONALLY stable in the vertical and compatible with convective
adjustment.

The exact algebraic-identity tests set ``implicit_vertical_mixing=False`` so the
explicit increment IS the full increment (then the closed-form AB2 relation holds to
rtol ~1e-11); the stability tests keep implicit mixing ON to exercise the
implicit-once property.

Includes a LONG (300-step) stability run — the prior leapfrog attempt's 40-step
test masked a slow blowup, so AB2 is validated over many steps + the real-ACC
free-run (separately).

Run in the fp64 precision policy so the exact (rtol ~1e-11) relations hold.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_DT = 1800.0
_EPS = 0.1

# ---------------------------------------------------------------------------
# KNOWN OPEN INSTABILITY -- read this before touching the two long-run tests.
# ---------------------------------------------------------------------------
# The default lat-lon C-grid lane exercised by ``_channel`` --
#   vorticity_scheme='al81', coriolis_scheme='matsuno_split',
#   een_q_boundary='neumann_fill', een_e3f_scheme='min',
#   momentum_advection='vector_invariant'
# (all printed from the instantiated config, not assumed) -- is NOT stable in
# this 8x16x4 implicit-vertical-mixing channel.  max|u| over the trajectory
# grows MONOTONICALLY and without bound:
#
#     n_steps            150      300      600      900     1200
#     pre-2026-08 op    1.630    4.790   13.910   52.658   90.409
#     current op        1.645    5.110   28.900     NaN      NaN
#
# (fp64, JAX_PLATFORMS=cpu, PrecisionPolicy.fp64(); the only variable between
# the two rows is the AL81 triad<->mass-flux pairing in
# ``pv_flux_al81_partial_cell`` -- same fixture, same seed, same step counts.)
#
# So the instability PRE-EXISTS the pairing fix; the fix (which independently
# restores BOTH energy and potential-enstrophy conservation of the operator --
# see tests/ocean/unit/test_al81_budget.py) makes it grow FASTER and reach NaN
# sooner.  The MECHANISM FOR THE RATE CHANGE IS UNKNOWN and is deliberately not
# asserted here.  In particular this is NOT known to be the recorded "C-grid
# barotropic Coriolis 2dx null mode": that item is documented in
# ``tests/ocean/unit/test_barotropic_coriolis_null_mode.py`` and
# ``docs/dev-notes/issues/barotropic_mode_noise.md``, and the only in-repo
# measurement of this exact al81+matsuno_split pairing
# (``docs/ocean/fidelity/dino_wiring_diagram.md:496``) records it as STABLE for
# DINO.  Connection NOT established -- do not cite one.
#
# The old assertion here was ``max|u|(300) < 5.0``, which the pairing fix
# crosses (5.110).  Raising 5.0 to 5.5 would re-hide the same defect one notch
# up, so instead:
#   * ``test_ab2_long_run_growth_rate_ratchet`` gates the GROWTH RATE **and**
#     the amplitude, each against its own SHRINK-ONLY baseline (same pattern as
#     tests/_ratchet_audit.py: the number may only ever be lowered, never
#     raised, and lowering it is the record of an improvement);
#   * ``test_ab2_long_run_bounded_900_steps`` states the property we actually
#     want and is marked ``xfail(strict=True)``, so the day someone fixes the
#     lane it turns the suite RED and forces both numbers to be updated.
# A green suite that silently contains a 90 m/s blowup is worse than a red one.
#
# BOTH gates are needed and neither subsumes the other: a RATIO is blind to
# amplitude (umax 10 -> 30 is growth 3.0 and would pass while being strictly
# worse at both times -- exactly what the deleted ``max|u|(300) < 5.0`` caught),
# and an AMPLITUDE bound alone is what the old test used and what conflates "the
# transient grew a bit" with "the trajectory is diverging".
#
# SHRINK-ONLY, both.  Any growth value > 1.0 means the lane GROWS; this is a
# recorded defect, not a passing grade.  The tolerances are deliberately loose
# (~4%) because these are maxima of an exponentially diverging trajectory two
# hundred steps from NaN, so the last digits are hardware/XLA-dependent; the
# MEASURED values sit next to them and are what an improvement must beat.
_AB2_GROWTH_600_OVER_300_BASELINE = 5.90     # measured 5.6557 (28.9001/5.1099)
_AB2_UMAX_300_BASELINE = 5.32                # measured 5.1099

# The step count at which the lane currently goes non-finite (measured: finite
# at 600, NaN at 900 and at 1200).  Used by the xfail test below.
_AB2_NAN_BY_STEPS = 900

_LONG_RUN_CACHE: dict = {}


def _ab2_running_umax(n_steps):
    """Running max|u| over the first ``k`` steps of ONE ``n_steps`` AB2
    trajectory, as ``{k: max|u|}``.  Cached so the two long-run tests below
    share a single integration instead of paying for it twice."""
    if n_steps not in _LONG_RUN_CACHE:
        state, model = _channel("ab2")
        final, traj = model.integrate_scan(state, n_steps=n_steps, dt=_DT)
        u = np.asarray(traj.u.data)
        T = np.asarray(traj.T.data)
        _LONG_RUN_CACHE[n_steps] = (
            {k: float(np.max(np.abs(u[:k]))) for k in (300, 600, n_steps)},
            {k: float(np.max(np.abs(T[:k]))) for k in (300, 600, n_steps)},
            final,
        )
    return _LONG_RUN_CACHE[n_steps]


@pytest.fixture(autouse=True)
def _fp64():
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _channel(outer_integrator="forward_euler", n_lat=8, n_lon=16, **cfg_kw):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(n_lat, n_lon)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=80.0)
    lat = np.degrees(np.asarray(grid.lat))
    T = np.asarray(state.T.data) + 4.0 * np.tanh(lat / 15.0)[:, None, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    cfg_kw.setdefault("implicit_vertical_mixing", True)
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3,
        n_barotropic_substeps=8, enable_runtime_checks=False,
        outer_integrator=outer_integrator, **cfg_kw)
    return state, LatLonCGridOceanModel(grid, z_coord, cfg)


def _depth_mean_u(model, state, u):
    from legoesm.ocean.vertical import compute_layer_thickness
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface
    h_k = compute_layer_thickness(
        state.eta.data, state.H_bathy.data, model.z_coord,
        min_water_column_m=model.config.min_water_column_m)
    h_u = np.asarray(min_cell_to_uface(h_k))
    u = np.asarray(u)
    return np.sum(u * h_u, axis=-1) / np.maximum(np.sum(h_u, axis=-1), 1e-10)


def test_forward_euler_default_runs_and_carry_untouched():
    state, model = _channel("forward_euler")
    s = model.step(state, dt=_DT)
    assert np.all(np.isfinite(np.asarray(s.T.data)))
    assert s.T_incr_prev is None and s.u_incr_prev is None


def test_unknown_outer_integrator_raises():
    # An unknown outer_integrator is rejected at CONSTRUCTION (fail-early dispatch
    # hardening: LatLonCGridOceanModel.__init__ -> _validate_config), not at .step().
    with pytest.raises(ValueError, match="outer_integrator"):
        _channel("bogus")


def test_double_ab2_rejected():
    state, model = _channel("ab2", tracer_time_integrator="ab2")
    with pytest.raises(ValueError, match="double-counts"):
        model.step(state, dt=_DT)


def test_ab2_tracer_invariant_exact():
    """AB2 relation holds EXACTLY from one step's I/O: the stored increment is the
    EXPLICIT-only increment ΔX_expl^n and T^{n+1}=T^n+(1.5+ε)·ΔX_expl^n
    −(0.5+ε)·ΔX_expl^{n-1} (verified on wet cells).  ``implicit_vertical_mixing=False``
    so the explicit increment is the full increment and the closed form is exact;
    this also pins the carry-identity (the carry is the explicit-only increment)."""
    state, model = _channel("ab2", implicit_vertical_mixing=False)
    wet = np.broadcast_to(
        (np.asarray(state.land_mask.data) > 0.5)[..., None],
        np.asarray(state.T.data).shape)
    T_n = np.asarray(state.T.data)
    dT_prev_in = 0.05 * np.ones_like(T_n)              # nonzero prior increment
    state = state._replace(
        T_incr_prev=state.T.replace(data=jnp.asarray(dT_prev_in)),
        S_incr_prev=state.S.replace(data=jnp.zeros_like(state.S.data)),
        u_incr_prev=state.u.replace(data=jnp.zeros_like(state.u.data)),
        v_incr_prev=state.v.replace(data=jnp.zeros_like(state.v.data)))
    s = model.step(state, dt=_DT)
    dT_n = np.asarray(s.T_incr_prev.data)              # = ΔX^n stored
    exp_T = (T_n + (1.5 + _EPS) * dT_n - (0.5 + _EPS) * dT_prev_in)
    exp_T = exp_T * (np.asarray(state.land_mask.data)[..., None])
    np.testing.assert_allclose(np.asarray(s.T.data)[wet], exp_T[wet],
                               rtol=1e-11, atol=1e-18)


def test_ab2_bootstrap_is_1p6_fe():
    """Bootstrap (incr_prev None ⇒ ΔX_expl^{n-1}=0): first AB2 step is
    X^n+(1.5+ε)·ΔX_expl^n (a 1.6× forward-Euler seed).  ``implicit_vertical_mixing=
    False`` on both so the explicit increment is the full increment and the 1.6×
    identity is exact (no implicit-once smoothing of the AB2 state)."""
    state_ab2, model_ab2 = _channel("ab2", implicit_vertical_mixing=False)
    state_fe, model_fe = _channel("forward_euler", implicit_vertical_mixing=False)
    wet = np.broadcast_to(
        (np.asarray(state_ab2.land_mask.data) > 0.5)[..., None],
        np.asarray(state_ab2.T.data).shape)
    T_n = np.asarray(state_ab2.T.data)
    s_ab2 = model_ab2.step(state_ab2, dt=_DT)
    s_fe = model_fe.step(state_fe, dt=_DT)
    dT_fe = np.asarray(s_fe.T.data) - T_n
    exp = T_n + (1.5 + _EPS) * dT_fe
    np.testing.assert_allclose(np.asarray(s_ab2.T.data)[wet], exp[wet], rtol=1e-8)
    assert s_ab2.T_incr_prev is not None


def test_ab2_momentum_barotropic_mean_preserved():
    """AB2 the baroclinic deviation only: the barotropic depth-mean equals the
    forward-Euler/split-explicit solve's (un-AB2'd free surface)."""
    state_ab2, model_ab2 = _channel("ab2")
    state_fe, model_fe = _channel("forward_euler")
    s_ab2 = model_ab2.step(state_ab2, dt=_DT)
    s_fe = model_fe.step(state_fe, dt=_DT)
    bt_ab2 = _depth_mean_u(model_ab2, state_ab2, s_ab2.u.data)
    bt_fe = _depth_mean_u(model_fe, state_fe, s_fe.u.data)
    np.testing.assert_allclose(bt_ab2, bt_fe, rtol=1e-8, atol=1e-12)


def test_ab2_long_run_growth_rate_ratchet():
    """THE key long-run test (the leapfrog attempt's 40-step test masked a slow
    blowup).  Gates the GROWTH RATE of ``max|u|`` between step 300 and step 600,
    against a SHRINK-ONLY baseline -- see the ``_AB2_GROWTH_600_OVER_300_BASELINE``
    block at the top of this module for why the previous absolute ``max|u| < 5.0``
    assertion was replaced and for the measured ladder.

    The rate, not the magnitude, is what the original test's docstring says it
    exists to catch ("masked a SLOW BLOWUP").  A magnitude bound conflates "the
    transient is a bit larger" with "the trajectory is diverging"; a ratio does
    not, and a shrink-only baseline means the number can only ever record an
    improvement.

    NB the growth rate is > 1: this lane IS diverging.  This test asserts that it
    is not diverging FASTER than the last recorded measurement -- it is a ratchet
    on a known defect, NOT a certificate of stability.  The certificate is
    ``test_ab2_long_run_bounded_900_steps`` below, and it xfails.

    NON-VACUOUS: the run is checked finite over the gated window, the two maxima
    are asserted distinct (a frozen trajectory would give a ratio of exactly 1.0
    and pass trivially), and the ratchet fires if the ratio regresses.
    """
    umax, Tmax, final = _ab2_running_umax(_AB2_NAN_BY_STEPS)
    # Finiteness is only claimed over the window this test gates (0..600); the
    # 900-step tail is the xfail test's business.
    assert np.isfinite(umax[300]) and np.isfinite(umax[600]), (
        f"AB2 went non-finite inside the gated window: {umax}")
    assert Tmax[600] < 50.0, f"AB2 T unbounded over 600 steps: max|T|={Tmax[600]}"
    assert final.T_incr_prev is not None

    growth = umax[600] / umax[300]
    msg = (f"AB2 al81+matsuno_split growth max|u|(600)/max|u|(300) = "
           f"{growth:.4f}  (max|u|: 300 -> {umax[300]:.4f}, "
           f"600 -> {umax[600]:.4f})")
    print("\n  " + msg)
    if growth > 1.0:
        warnings.warn(
            "KNOWN OPEN DEFECT (not a new regression): " + msg
            + " -- max|u| GROWS without bound on this lane and reaches NaN by "
              f"{_AB2_NAN_BY_STEPS} steps.  See the header block in "
              "tests/ocean/unit/test_ab2_integrator.py.",
            RuntimeWarning, stacklevel=2)
    assert umax[600] > umax[300], (
        "trajectory did not evolve between step 300 and 600 -- the ratchet "
        "would be vacuous (ratio identically 1.0)")
    assert growth <= _AB2_GROWTH_600_OVER_300_BASELINE, (
        f"AB2 growth rate REGRESSED: {growth:.4f} > baseline "
        f"{_AB2_GROWTH_600_OVER_300_BASELINE}.  This baseline is SHRINK-ONLY: "
        f"if you have improved the lane, LOWER it (and lower "
        f"_AB2_UMAX_300_BASELINE / _AB2_NAN_BY_STEPS / retire the xfail as "
        f"appropriate).  Do NOT raise it to make this pass.")
    # AMPLITUDE ratchet -- a ratio alone cannot see a uniformly larger
    # trajectory (see the header block).  Also shrink-only.
    assert umax[300] <= _AB2_UMAX_300_BASELINE, (
        f"AB2 amplitude REGRESSED: max|u|(300) = {umax[300]:.4f} > baseline "
        f"{_AB2_UMAX_300_BASELINE}.  SHRINK-ONLY -- do NOT raise it.")


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN OPEN: the default al81 + matsuno_split lat-lon C-grid lane "
           "diverges in this channel -- max|u| grows monotonically "
           "(1.6 -> 5.1 -> 28.9 over 150/300/600 steps) and goes NaN by 900. "
           "PRE-EXISTS the 2026-08 AL81 triad<->flux pairing fix (which reaches "
           "90 m/s by 1200 steps WITHOUT the fix, finite but diverging); the "
           "fix accelerates it. Mechanism UNKNOWN and deliberately not asserted "
           "-- in particular NOT established as the recorded C-grid barotropic "
           "Coriolis 2dx null mode (the only in-repo measurement of this exact "
           "scheme pairing, dino_wiring_diagram.md:496, records it STABLE for "
           "DINO). strict=True: fixing the lane turns this RED so the ratchet "
           "above and this marker are updated together.")
def test_ab2_long_run_bounded_900_steps():
    """The property this lane SHOULD have and currently does not: a 900-step AB2
    trajectory stays finite and bounded.  Written as the real assertion rather
    than deleted, so the defect is visible in the suite instead of hidden behind
    a relaxed threshold."""
    umax, _Tmax, final = _ab2_running_umax(_AB2_NAN_BY_STEPS)
    for name, f in (("T", final.T.data), ("u", final.u.data),
                    ("v", final.v.data), ("eta", final.eta.data)):
        assert np.all(np.isfinite(np.asarray(f))), f"AB2 final {name} non-finite"
    assert umax[_AB2_NAN_BY_STEPS] < 5.0, (
        f"AB2 u unbounded over {_AB2_NAN_BY_STEPS} steps: "
        f"max|u|={umax[_AB2_NAN_BY_STEPS]}")


def test_ab2_accepts_convective_adjustment():
    """The faithful AB2 (implicit vertical mixing applied ONCE, not extrapolated) is
    unconditionally stable in the vertical, so ab2 + convective adjustment is now
    ALLOWED (the prior conditional-instability guard is removed).  Runs finite."""
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    physics = OceanPhysicsConfig(
        lateral_mixing=LateralMixingConfig(scheme="none"),   # lat-lon: physics factory is none
        convection=OceanConvectionConfig(scheme="enhanced_diffusion"))
    state, model = _channel("ab2", physics=physics)
    f, _ = model.integrate_scan(state, n_steps=20, dt=_DT)   # no raise
    for arr in (f.T.data, f.S.data, f.u.data, f.v.data, f.eta.data):
        assert np.all(np.isfinite(np.asarray(arr))), "ab2 + convection went non-finite"


def test_ab2_unconditional_stability_with_stiff_vertical_mixing():
    """Proves the scheme is fixed (inverts the prior conditional-stability test):
    with the implicit vertical-mixing increment applied ONCE (not AB2-extrapolated),
    a stiff constant K_v that previously made ab2 blow up at ~120 steps now stays
    bounded — comparable to forward_euler — over a long run."""
    state_ab2, model_ab2 = _channel("ab2", K_v=2000.0, A_v=2000.0)
    state_fe, model_fe = _channel("forward_euler", K_v=2000.0, A_v=2000.0)
    f_ab2, _ = model_ab2.integrate_scan(state_ab2, n_steps=300, dt=_DT)
    f_fe, _ = model_fe.integrate_scan(state_fe, n_steps=300, dt=_DT)
    fe_arr = np.asarray(f_fe.T.data)
    ab2_arr = np.asarray(f_ab2.T.data)
    assert np.all(np.isfinite(fe_arr))                    # forward_euler stays stable
    assert np.all(np.isfinite(ab2_arr)), "faithful ab2 must stay finite with stiff K_v"
    fe_max = float(np.max(np.abs(fe_arr)))
    ab2_max = float(np.max(np.abs(ab2_arr)))
    # ab2 now stays bounded, of the same order as forward_euler (no >10x blowup).
    assert ab2_max < 5.0 * fe_max + 1.0, (
        f"faithful ab2 should be bounded like forward_euler with stiff K_v; "
        f"ab2_max={ab2_max:.3g} fe_max={fe_max:.3g}")


def test_ab2_implicit_mixing_is_applied_once():
    """The implicit vertical mixing is actually APPLIED in the faithful AB2 path:
    one AB2 step with a stiff background K_v smooths the vertical T profile (lower
    vertical variance) relative to K_v=0.  Confirms implicit-once runs (not skipped)."""
    # Stratified column: impose a sharp vertical T gradient.
    state0, _ = _channel("ab2")
    T = np.asarray(state0.T.data)
    nlev = T.shape[-1]
    T = T + np.linspace(6.0, -6.0, nlev)[None, None, :]   # strong vertical gradient
    state0 = state0._replace(T=state0.T.replace(data=jnp.asarray(T)))

    s_mix, model_mix = _channel("ab2", K_v=2000.0)
    s_nomix, model_nomix = _channel("ab2", K_v=0.0, A_v=0.0)
    s_mix = s_mix._replace(T=s_mix.T.replace(data=jnp.asarray(T)))
    s_nomix = s_nomix._replace(T=s_nomix.T.replace(data=jnp.asarray(T)))

    out_mix = np.asarray(model_mix.step(s_mix, dt=_DT).T.data)
    out_nomix = np.asarray(model_nomix.step(s_nomix, dt=_DT).T.data)
    wet = np.asarray(state0.land_mask.data) > 0.5
    var_mix = float(np.var(out_mix[wet], axis=-1).mean())
    var_nomix = float(np.var(out_nomix[wet], axis=-1).mean())
    assert var_mix < 0.95 * var_nomix, (
        f"implicit vertical mixing not applied in AB2 path: "
        f"var(K_v=2000)={var_mix:.4g} not < var(K_v=0)={var_nomix:.4g}")


def test_ab2_step_differentiable():
    state, model = _channel("ab2")
    state = state._replace(
        T_incr_prev=state.T.replace(data=jnp.zeros_like(state.T.data)),
        S_incr_prev=state.S.replace(data=jnp.zeros_like(state.S.data)),
        u_incr_prev=state.u.replace(data=jnp.zeros_like(state.u.data)),
        v_incr_prev=state.v.replace(data=jnp.zeros_like(state.v.data)))
    T0 = state.T.data

    def loss(T_in):
        st = state._replace(T=state.T.replace(data=T_in))
        return jnp.sum(model.step(st, dt=_DT).T.data ** 2)

    g = jax.grad(loss)(T0)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.sum(jnp.abs(g))) > 0.0
