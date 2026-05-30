"""Adams-Bashforth-2 outer time integrator (config.outer_integrator="ab2").

This is Veros's actual outer scheme (AB2 on the explicit tendency, NOT leapfrog —
see the strategy §8 integrator row). The default "forward_euler" path is unchanged;
the AB2 path extrapolates the forward-Euler increment
``X^{n+1}=X^n+(1.5+ε)·ΔX^n−(0.5+ε)·ΔX^{n-1}``, carrying ΔX^{n-1} on
``{T,S,u,v}_incr_prev``, keeping the barotropic free-surface mode from the
split-explicit solve, and bootstrapping ΔX^{n-1}=0.

Includes a LONG (300-step) stability run — the prior leapfrog attempt's 40-step
test masked a slow blowup, so AB2 is validated over many steps + the real-ACC
free-run (separately).

Run in the fp64 precision policy so the exact (rtol ~1e-11) relations hold.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_DT = 1800.0
_EPS = 0.1


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
    cfg = LatLonCGridOceanConfig(
        A_h=2.0e4, bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
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
    state, model = _channel("bogus")
    with pytest.raises(ValueError, match="outer_integrator"):
        model.step(state, dt=_DT)


def test_double_ab2_rejected():
    state, model = _channel("ab2", tracer_time_integrator="ab2")
    with pytest.raises(ValueError, match="double-counts"):
        model.step(state, dt=_DT)


def test_ab2_tracer_invariant_exact():
    """AB2 relation holds EXACTLY from one step's I/O: the stored increment equals
    ΔX^n and T^{n+1}=T^n+(1.5+ε)·ΔX^n−(0.5+ε)·ΔX^{n-1} (verified on wet cells)."""
    state, model = _channel("ab2")
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
    """Bootstrap (incr_prev None ⇒ ΔX^{n-1}=0): first AB2 step is X^n+(1.5+ε)·ΔX^n
    (a 1.6× forward-Euler seed, as the inner tracer AB2 does)."""
    state_ab2, model_ab2 = _channel("ab2")
    state_fe, model_fe = _channel("forward_euler")
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


def test_ab2_long_run_stable_300_steps():
    """THE key test (the leapfrog attempt's 40-step test masked a slow blowup):
    a 300-step AB2 trajectory in the implicit-vertical-mixing channel stays finite
    + bounded — AB2 applies the implicit-mixing increment ~once, not doubled."""
    state, model = _channel("ab2")
    final, traj = model.integrate_scan(state, n_steps=300, dt=_DT)
    for f in (final.T.data, final.u.data, final.v.data, final.eta.data):
        assert np.all(np.isfinite(np.asarray(f))), "AB2 went non-finite"
    umax = float(np.max(np.abs(np.asarray(traj.u.data))))
    Tmax = float(np.max(np.abs(np.asarray(traj.T.data))))
    assert umax < 5.0, f"AB2 u unbounded over 300 steps: max|u|={umax}"
    assert Tmax < 50.0, f"AB2 T unbounded over 300 steps: max|T|={Tmax}"
    assert final.T_incr_prev is not None


def test_ab2_rejects_convective_adjustment():
    """The step guard rejects ab2 + convective adjustment — ab2 AB2-extrapolates the
    implicit vertical-mixing increment, which is conditionally unstable for stiff
    mixing (adversarial-review #2). The ACC recipe (convection="none") is unaffected."""
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    physics = OceanPhysicsConfig(
        lateral_mixing=LateralMixingConfig(scheme="none"),   # lat-lon: physics factory is none
        convection=OceanConvectionConfig(scheme="enhanced_diffusion"))
    state, model = _channel("ab2", physics=physics)
    with pytest.raises(ValueError, match="convective adjustment"):
        model.step(state, dt=_DT)


def test_ab2_conditional_stability_with_stiff_vertical_mixing():
    """Locks the review-found regression: AB2-extrapolating the implicit
    vertical-mixing increment is CONDITIONALLY stable — a stiff constant K_v makes
    ab2 grow unboundedly while forward_euler (implicit applied once) stays bounded.
    This is WHY the guard + the not-Veros-faithful caveat exist."""
    state_ab2, model_ab2 = _channel("ab2", K_v=2000.0, A_v=2000.0)
    state_fe, model_fe = _channel("forward_euler", K_v=2000.0, A_v=2000.0)
    f_ab2, _ = model_ab2.integrate_scan(state_ab2, n_steps=120, dt=_DT)
    f_fe, _ = model_fe.integrate_scan(state_fe, n_steps=120, dt=_DT)
    fe_arr = np.asarray(f_fe.T.data)
    ab2_arr = np.asarray(f_ab2.T.data)
    assert np.all(np.isfinite(fe_arr))                    # forward_euler stays stable
    fe_max = float(np.max(np.abs(fe_arr)))
    ab2_max = float(np.max(np.abs(ab2_arr)))
    # ab2 either blew up (non-finite) or grew >> forward_euler.
    assert (not np.all(np.isfinite(ab2_arr))) or ab2_max > 10.0 * fe_max, (
        f"expected ab2 conditional-stability blowup with stiff K_v; "
        f"ab2_max={ab2_max:.3g} fe_max={fe_max:.3g}")


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
