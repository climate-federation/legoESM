"""Tracer transport tests for the plane NH dycore (PR3d).

PR3d lifts the PR2b ``n_tracers == 0`` restriction. Tracer slow
tendency is upwind flux form (advective form via the same
``_upwind_advection_x/_y`` and ``_vertical_advection_plane`` helpers
used for momentum / theta), vmapped over the trailing tracer axis.

Tests:

1. ``n_tracers = 0`` path unchanged (zero-shape output).
2. Passive tracer with non-zero initial condition + zero velocity
   stays at rest (no tracer drift).
3. Passive tracer total mass conserved over a multi-step
   integration (no source / sink in the dycore).
4. ``physics_fn`` argument plumbing: a zero-tendency physics_fn is
   bit-exact equal to ``physics_fn=None``; a non-zero physics_fn
   contributes to the result.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.core.field import Field
from legoesm.core.state import PlaneNonHydrostaticTendencies
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


jax.config.update("jax_enable_x64", True)


def _setup_with_tracers(n_tracers=2):
    grid = create_plane_grid(
        nx=8, ny=8, nlev=4, dx=200.0, dy=200.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(grid.nlev, H=2_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
        semi_implicit_acoustic=False, use_coriolis=False, fix_mass=False,
        smagorinsky_cs=0.0,
        # These tests exercise PASSIVE tracer transport. With the default
        # moist_buoyancy=True, slot-0 acts as q_v and a horizontally-varying
        # tracer becomes buoyant → w → drift; disable so the tracer stays
        # passive (moist buoyancy is covered by test_plane_nh_buoyancy).
        moist_buoyancy=False,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    rest = make_rest_state(grid, hc, dtype=jnp.float64)
    # Replace tracers with non-empty axis.
    tracer_data = jnp.zeros(
        rest.tracers.data.shape[:-1] + (n_tracers,), dtype=jnp.float64,
    )
    state = rest._replace(
        tracers=rest.tracers.replace(data=tracer_data),
    )
    return model, state, grid, hc, tm


def test_n_tracers_zero_path_unchanged():
    """PR2b/c rest path with ``n_tracers == 0`` must keep zero-axis
    output and stay at rest."""
    model, state, _, _, _ = _setup_with_tracers(n_tracers=0)
    assert state.tracers.data.shape[-1] == 0
    next_state = model.step(state, dt=1.0)
    assert next_state.tracers.data.shape[-1] == 0


def test_passive_tracer_rest_state_stays_at_rest():
    """Zero velocity + non-zero tracer IC: tracer must not drift."""
    model, state, _, _, _ = _setup_with_tracers(n_tracers=2)
    rng = np.random.default_rng(0)
    init_tracers = jnp.asarray(
        rng.standard_normal(state.tracers.data.shape) * 1.0e-3,
    )
    state = state._replace(
        tracers=state.tracers.replace(data=init_tracers),
    )
    next_state = model.step(state, dt=1.0)
    # u = v = w = 0 throughout → advective tendency = 0 → tracer
    # unchanged to round-off.
    assert jnp.allclose(
        next_state.tracers.data, init_tracers, atol=1.0e-12,
    )


def test_passive_tracer_total_mass_conserved_under_advection():
    """Sum of tracer over the periodic plane stays constant up to
    round-off when there are no sources / sinks. Use non-zero u to
    actually advect."""
    model, state, _, _, _ = _setup_with_tracers(n_tracers=1)
    rng = np.random.default_rng(1)
    init_tracers = jnp.asarray(
        rng.standard_normal(state.tracers.data.shape) * 1.0e-3,
    )
    # Uniform u-shear in x; upwind advection is bounded but not
    # strictly TVD under varying u — the integral over the periodic
    # plane stays conserved regardless (advective form: ∑ -u ∂q/∂x
    # cancels to zero on periodic domain only when div(u) = 0). For
    # constant u, ∑ -u dq/dx = -u ∑ dq/dx = 0 trivially.
    u_uniform = jnp.full(state.u.data.shape, 0.1)
    state = state._replace(
        tracers=state.tracers.replace(data=init_tracers),
        u=state.u.replace(data=u_uniform),
    )
    total0 = float(jnp.sum(init_tracers))
    for _ in range(20):
        state = model.step(state, dt=0.5)
    total_f = float(jnp.sum(state.tracers.data))
    rel = abs(total_f - total0) / max(abs(total0), 1.0e-30)
    assert rel < 1.0e-8, f"Tracer mass drift = {rel:.3e}"


def test_physics_fn_none_is_default():
    """``model.step(state, dt)`` without a ``physics_fn`` should equal
    ``model.step(state, dt, physics_fn=None)``."""
    model, state, _, _, _ = _setup_with_tracers(n_tracers=0)
    s_no_arg = model.step(state, dt=1.0)
    s_none = model.step(state, dt=1.0, physics_fn=None)
    assert jnp.array_equal(s_no_arg.u.data, s_none.u.data)


def test_physics_fn_zero_tendency_is_no_op():
    """A physics_fn that returns all zeros must give the same step
    result as ``physics_fn=None`` to round-off."""
    model, state, _, _, _ = _setup_with_tracers(n_tracers=0)

    def zero_phys(s, grid, hc, tm):
        return PlaneNonHydrostaticTendencies(
            du_dt=s.u.replace(data=jnp.zeros_like(s.u.data)),
            dv_dt=s.v.replace(data=jnp.zeros_like(s.v.data)),
            dw_dt=s.w.replace(data=jnp.zeros_like(s.w.data)),
            dtheta_prime_dt=s.theta_prime.replace(
                data=jnp.zeros_like(s.theta_prime.data)),
            drho_prime_dt=s.rho_prime.replace(
                data=jnp.zeros_like(s.rho_prime.data)),
            dphis_dt=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
            dtracers_dt=s.tracers.replace(
                data=jnp.zeros_like(s.tracers.data)),
        )

    s_none = model.step(state, dt=1.0, physics_fn=None)
    s_zero = model.step(state, dt=1.0, physics_fn=zero_phys)
    # Bit-exact: same compiled path, same arithmetic.
    assert jnp.allclose(s_zero.u.data, s_none.u.data, atol=1.0e-15)
    assert jnp.allclose(s_zero.theta_prime.data, s_none.theta_prime.data, atol=1.0e-15)


def test_physics_fn_nonzero_tendency_contributes():
    """A physics_fn that returns a constant heating must perturb the
    theta_prime evolution away from the no-physics result."""
    model, state, _, _, _ = _setup_with_tracers(n_tracers=0)

    def heating_phys(s, grid, hc, tm):
        heat = jnp.full_like(s.theta_prime.data, 1.0e-3)  # 1 mK/s
        zeros_u = jnp.zeros_like(s.u.data)
        zeros_v = jnp.zeros_like(s.v.data)
        zeros_w = jnp.zeros_like(s.w.data)
        zeros_rho = jnp.zeros_like(s.rho_prime.data)
        zeros_phis = jnp.zeros_like(s.phis.data)
        zeros_tracers = jnp.zeros_like(s.tracers.data)
        return PlaneNonHydrostaticTendencies(
            du_dt=s.u.replace(data=zeros_u),
            dv_dt=s.v.replace(data=zeros_v),
            dw_dt=s.w.replace(data=zeros_w),
            dtheta_prime_dt=s.theta_prime.replace(data=heat),
            drho_prime_dt=s.rho_prime.replace(data=zeros_rho),
            dphis_dt=s.phis.replace(data=zeros_phis),
            dtracers_dt=s.tracers.replace(data=zeros_tracers),
        )

    s_none = model.step(state, dt=1.0, physics_fn=None)
    s_heat = model.step(state, dt=1.0, physics_fn=heating_phys)
    diff = float(jnp.max(jnp.abs(
        s_heat.theta_prime.data - s_none.theta_prime.data
    )))
    # Expected diff ~ 1e-3 K (heating rate * dt = 1e-3 * 1).
    assert diff > 1.0e-5, (
        f"physics_fn heating did not propagate: diff = {diff:.3e}"
    )
