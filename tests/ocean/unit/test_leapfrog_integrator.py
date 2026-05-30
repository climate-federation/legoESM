"""Leapfrog + Robert-Asselin outer integrator (build gate I1 — tracers).

Locks the ``config.outer_integrator="leapfrog_ab2"`` wiring on the lat-lon C-grid
ocean model: the default ``"forward_euler"`` path is unchanged (covered by the
broader ocean suite); the leapfrog path advances the fully-explicit tracers by the
centred-in-time leapfrog (``X^{n+1}=X^{n-1}+2(X_FE-X^n)``) + the Robert-Asselin filter,
carries the τ-1 state on ``T_prev``/``S_prev``, bootstraps from ``X^{-1}=X^0`` on the
first step, runs stably in a ``jax.lax.scan``, leaves the momentum/free surface at the
forward-Euler result (I1: tracers only), and is differentiable.

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
_NU = 0.05


@pytest.fixture(autouse=True)
def _fp64():
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _channel(outer_integrator="forward_euler", asselin_nu=_NU, n_lat=8, n_lon=16):
    """Small stratified re-entrant channel + model with the chosen integrator."""
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
        outer_integrator=outer_integrator, asselin_nu=asselin_nu)
    return state, LatLonCGridOceanModel(grid, z_coord, cfg)


def _wet3(state):
    m = np.asarray(state.land_mask.data) > 0.5          # (n_lat, n_lon)
    nlev = np.asarray(state.T.data).shape[-1]
    return np.broadcast_to(m[..., None], m.shape + (nlev,))


def test_forward_euler_default_runs_and_carry_untouched():
    """Default integrator runs + never touches the leapfrog τ-1 carry (the broader
    ocean suite covers default bit-for-bit behaviour)."""
    state, model = _channel("forward_euler")
    s = model.step(state, dt=_DT)
    assert np.all(np.isfinite(np.asarray(s.T.data)))
    assert s.T_prev is None and s.S_prev is None


def test_unknown_outer_integrator_raises():
    state, model = _channel("bogus")
    with pytest.raises(ValueError, match="outer_integrator"):
        model.step(state, dt=_DT)


def test_leapfrog_robert_asselin_invariant_exact():
    """The Robert-Asselin filter relation holds EXACTLY between one step's I/O:
    ``T_prev_out = T^n + ν(T^{n+1} - 2 T^n + T^{n-1})`` (and same for S). Verified on
    wet cells from the step's own output T (=T^{n+1}) + the seeded τ-1 + input T^n."""
    state, model = _channel("leapfrog_ab2", asselin_nu=_NU)
    wet = _wet3(state)
    T_n = np.asarray(state.T.data)
    # Non-trivial τ-1 (X^{-1} != X^0) so the filter term is exercised.
    T_prev_in = T_n - 0.1
    state = state._replace(
        T_prev=state.T.replace(data=jnp.asarray(T_prev_in)),
        S_prev=state.S.replace(data=state.S.data))
    s = model.step(state, dt=_DT)
    T_out = np.asarray(s.T.data)          # = T^{n+1}
    exp_prev = T_n + _NU * (T_out - 2.0 * T_n + T_prev_in)
    np.testing.assert_allclose(
        np.asarray(s.T_prev.data)[wet], exp_prev[wet], rtol=1e-11, atol=1e-20)
    assert np.all(np.isfinite(T_out))


def test_leapfrog_bootstrap_is_2fe_minus_initial():
    """Bootstrap (τ-1 None ⇒ X^{-1}=X^0): the leapfrog first step gives
    X^{1} = 2·X_FE − X^0 (the standard 2dt forward-Euler seed). Compared against the
    forward-Euler model's own step on the same state."""
    state_lf, model_lf = _channel("leapfrog_ab2")
    state_fe, model_fe = _channel("forward_euler")
    wet = _wet3(state_lf)
    T_n = np.asarray(state_lf.T.data)
    s_lf = model_lf.step(state_lf, dt=_DT)
    s_fe = model_fe.step(state_fe, dt=_DT)
    exp = 2.0 * np.asarray(s_fe.T.data) - T_n
    np.testing.assert_allclose(np.asarray(s_lf.T.data)[wet], exp[wet], rtol=1e-8)
    assert s_lf.T_prev is not None  # carry now seeded


def test_leapfrog_i1_leaves_momentum_at_forward_euler():
    """I1 leapfrogs tracers only — the momentum + free surface match the
    forward-Euler step (Veros leaves the barotropic free surface un-leapfrogged)."""
    state_lf, model_lf = _channel("leapfrog_ab2")
    state_fe, model_fe = _channel("forward_euler")
    s_lf = model_lf.step(state_lf, dt=_DT)
    s_fe = model_fe.step(state_fe, dt=_DT)
    np.testing.assert_allclose(
        np.asarray(s_lf.u.data), np.asarray(s_fe.u.data), rtol=1e-8, atol=1e-15)
    np.testing.assert_allclose(
        np.asarray(s_lf.eta.data), np.asarray(s_fe.eta.data), rtol=1e-8, atol=1e-15)


def test_leapfrog_scan_runs_stable():
    """integrate_scan seeds the τ-1 carry (constant scan-carry pytree) and runs a
    multi-step trajectory that stays finite."""
    state, model = _channel("leapfrog_ab2")
    final, traj = model.integrate_scan(state, n_steps=12, dt=_DT)
    assert np.all(np.isfinite(np.asarray(final.T.data)))
    assert np.all(np.isfinite(np.asarray(final.u.data)))
    assert final.T_prev is not None
    assert np.all(np.isfinite(np.asarray(traj.T.data)))


def test_leapfrog_step_differentiable():
    """Leapfrog + Robert-Asselin is linear in the time levels ⇒ grad of a scalar of
    the stepped state w.r.t. the input T is finite + nonzero."""
    state, model = _channel("leapfrog_ab2")
    state = state._replace(
        T_prev=state.T.replace(data=state.T.data),
        S_prev=state.S.replace(data=state.S.data))
    T0 = state.T.data

    def loss(T_in):
        st = state._replace(T=state.T.replace(data=T_in))
        return jnp.sum(model.step(st, dt=_DT).T.data ** 2)

    g = jax.grad(loss)(T0)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.sum(jnp.abs(g))) > 0.0
