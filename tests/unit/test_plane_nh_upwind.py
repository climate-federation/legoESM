"""Upwind advection tests for the plane NH dycore.

PR3b replaces the PR2b/PR2d centred-difference advection of u, v,
theta', and w with first-order upwind reconstruction. Tests cover:

1. Upwind helpers sign convention (positive u flows west→east, etc).
2. Pure-advection monotonicity preservation (no overshoot/undershoot
   on a square wave).
3. Rest state preservation (zero advection on zero state).
4. Upwind contribution is differentiable through ``jax.grad``.
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
    _upwind_advection_x,
    _upwind_advection_y,
    make_flat_plane_terrain_metric,
    make_rest_state,
    plane_compressible_euler_slow_tendencies,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


jax.config.update("jax_enable_x64", True)


def _setup():
    grid = create_plane_grid(
        nx=16, ny=8, nlev=4, dx=200.0, dy=200.0, dtype=jnp.float64,
    )
    height_coord = create_height_coordinate(grid.nlev, H=4_000.0)
    terrain = make_flat_plane_terrain_metric(grid, height_coord)
    config = CompressibleEulerConfig(
        sponge_coeff=0.0,
        hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0,
        hyperdiff_w_coeff=0.0,
        semi_implicit_acoustic=False,
        use_coriolis=False,
        fix_mass=False,
    )
    return PlaneCompressibleEulerModel(grid, height_coord, terrain, config), \
        grid, height_coord, terrain, config


# --------------------------------------------------------------------- #
# ADV-SPLIT #86: 2nd-order centred momentum advection (= gSAM advect2_mom) #
# --------------------------------------------------------------------- #


def test_centered_advection_x_reduces_to_centered_difference():
    """For CONSTANT advecting velocity, the flux-form centred scheme reduces
    to the classic 2nd-order centred difference ``-u·(f[i+1]-f[i-1])/(2dx)``
    (non-diffusive, dispersive — = gSAM `advect2_mom_xy.f90`). Guards the
    momentum-leg scheme used by the ADV-SPLIT #86 per-field split."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        _centered_advection_x,
    )
    nx = 8
    f = jnp.asarray(
        np.sin(2.0 * np.pi * np.arange(nx) / nx)
    ).reshape(1, nx, 1)
    u = jnp.full_like(f, 2.0)            # constant velocity
    dx = 0.5
    tend = _centered_advection_x(f, u, dx)
    expected = -2.0 * (
        jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dx)
    assert np.allclose(np.asarray(tend), np.asarray(expected), atol=1e-12), (
        "centred advection != centred difference for constant velocity"
    )


def test_momentum_advection_split_is_wired_and_active():
    """ADV-SPLIT #86: setting `horizontal_momentum_advection_scheme="centered"`
    (van_leer scalars) must change the MOMENTUM tendency vs the all-van_leer
    default on a sheared state — proving the split is wired to u/v/w, not a
    no-op. (Default None ⇒ identical to all-van_leer, covered by every other
    test passing unchanged.)"""
    grid = create_plane_grid(
        nx=12, ny=12, nlev=6, dx=500.0, dy=500.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(grid.nlev, H=6_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    rng = np.random.default_rng(3)
    base = make_rest_state(grid, hc, dtype=jnp.float64)
    import equinox as eqx
    u_pert = jnp.asarray(rng.standard_normal(base.u.data.shape))
    state = eqx.tree_at(lambda s: s.u.data, base, u_pert)

    def du_dt(mscheme):
        cfg = CompressibleEulerConfig(
            sponge_coeff=0.0, hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
            hyperdiff_w_coeff=0.0, semi_implicit_acoustic=False,
            use_coriolis=False, fix_mass=False,
            horizontal_advection_scheme="van_leer",
            horizontal_momentum_advection_scheme=mscheme,
        )
        tend = plane_compressible_euler_slow_tendencies(state, grid, hc, tm, cfg)
        return np.asarray(tend.du_dt.data)

    diff = np.max(np.abs(du_dt("centered") - du_dt(None)))
    assert diff > 1.0e-8, (
        f"momentum split inactive: centred vs van_leer du_dt identical "
        f"(max diff {diff:.2e})"
    )


def test_centered_momentum_is_non_dissipative_vs_van_leer():
    """ADV-SPLIT #86 (codex iter-68 [S1] KE-budget): the QUANTITATIVE mechanism
    behind the split. SAM's `advect2_mom` is non-dissipative; van_leer's TVD
    flux limiter DISSIPATES kinetic energy — which is WHY it suppresses
    convective updraft cores / w-variance tails. For self-advection ``-u·∂u/∂x``
    on a periodic field:

    * BOTH schemes conserve momentum (``Σ tend ≈ 0``, flux-form).
    * van_leer's KE budget ``Σ u·tend`` is large + NEGATIVE (strongly
      dissipative); centered's is TINY (≈ the advective-form residual, NOT
      systematic dissipation — so it PRESERVES convective KE/extremes).

    This is why centered restores the updraft extremes van_leer damps."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        _centered_advection_x, _van_leer_advection_x,
    )
    rng = np.random.default_rng(5)
    u = jnp.asarray(rng.standard_normal((2, 32, 3)))
    dx = 1.0
    tc = _centered_advection_x(u, u, dx)
    tv = _van_leer_advection_x(u, u, dx)
    norm = float(jnp.sum(u * u))
    # momentum conservation (flux-form): Σ tend ≈ 0 for both
    assert abs(float(jnp.sum(tc))) < 1.0e-10 * norm
    assert abs(float(jnp.sum(tv))) < 1.0e-10 * norm
    ke_c = abs(float(jnp.sum(u * tc))) / norm     # centered: near-conserving
    ke_v = float(jnp.sum(u * tv)) / norm          # van_leer: dissipative (<0)
    assert ke_c < 0.05, (
        f"centered KE budget {ke_c:.3f} too large — should be near-conserving"
    )
    assert ke_v < -0.1, (
        f"van_leer should be KE-DISSIPATIVE (negative budget); got {ke_v:.3f}"
    )
    # centered preserves ~10x+ more KE than van_leer dissipates (the #86 effect):
    # van_leer ~-0.4·||u||² per step, centered ~+0.015 — the dissipation that
    # damps van_leer's updraft cores is absent in centered.
    assert ke_c < abs(ke_v) / 5.0


# --------------------------------------------------------------------- #
# Sign convention                                                       #
# --------------------------------------------------------------------- #


def test_upwind_x_uses_backward_diff_for_positive_u():
    """``u > 0`` should pick backward difference (information from
    upstream cell i-1). Set ``u = +1``, ``f = [0, 1, 0, ...]`` and
    check ``out[0, 1] = -1 * (f[1] - f[0]) / dx = -1/dx``."""
    nx, ny, nlev = 4, 1, 1
    dx = 10.0
    f = jnp.zeros((ny, nx, nlev)).at[0, 1, 0].set(1.0)
    u = jnp.ones((ny, nx, nlev))
    out = _upwind_advection_x(f, u, dx)
    # out = -u * (f - f[..., i-1]) / dx for u > 0.
    # At i=1: -(1) * (1 - 0)/10 = -0.1
    assert float(out[0, 1, 0]) == pytest.approx(-0.1)
    # At i=2: -(1) * (0 - 1)/10 = +0.1 (downstream tip)
    assert float(out[0, 2, 0]) == pytest.approx(0.1)


def test_upwind_x_uses_forward_diff_for_negative_u():
    """``u < 0`` should pick forward difference (information from
    upstream cell i+1)."""
    nx, ny, nlev = 4, 1, 1
    dx = 10.0
    f = jnp.zeros((ny, nx, nlev)).at[0, 1, 0].set(1.0)
    u = -jnp.ones((ny, nx, nlev))
    out = _upwind_advection_x(f, u, dx)
    # out = -u * (f[..., i+1] - f) / dx for u < 0.
    # At i=0: -(-1) * (1 - 0)/10 = +0.1
    assert float(out[0, 0, 0]) == pytest.approx(0.1)
    # At i=1: -(-1) * (0 - 1)/10 = -0.1
    assert float(out[0, 1, 0]) == pytest.approx(-0.1)


def test_upwind_y_uses_backward_diff_for_positive_v():
    nx, ny, nlev = 1, 4, 1
    dy = 5.0
    f = jnp.zeros((ny, nx, nlev)).at[1, 0, 0].set(1.0)
    v = jnp.ones((ny, nx, nlev))
    out = _upwind_advection_y(f, v, dy)
    assert float(out[1, 0, 0]) == pytest.approx(-0.2)
    assert float(out[2, 0, 0]) == pytest.approx(0.2)


# --------------------------------------------------------------------- #
# Monotonicity                                                          #
# --------------------------------------------------------------------- #


def test_upwind_constant_velocity_collocated_is_tvd_one_step():
    """A square wave advected one step under **constant collocated**
    velocity at Courant ``u dt/dx < 1`` with first-order upwind in
    **advective form** stays within the original [min, max] range.

    Caveat (Codex iter-1 finding): the TVD guarantee holds only for
    this restricted regime — constant velocity, collocated layout,
    CFL-limited single forward-Euler step. Under spatially varying
    velocity advective-form upwind is not conservative and may show
    bounded but non-monotone behaviour. The dycore wraps each
    advective-form call inside SSP-RK3, so the single-step bound is
    a necessary but not sufficient condition for the runtime
    behaviour."""
    nx = 16
    dx = 100.0
    dt = 1.0
    u_const = 5.0  # m/s — Courant = u*dt/dx = 0.05, safely under 1
    f0 = jnp.zeros((1, nx, 1))
    f0 = f0.at[0, 4:8, 0].set(1.0)  # square wave on cells 4..7
    u = jnp.full_like(f0, u_const)
    df_dt = _upwind_advection_x(f0, u, dx)
    f1 = f0 + dt * df_dt
    # No overshoot, no undershoot in the constant-velocity case.
    assert float(jnp.max(f1)) <= 1.0 + 1.0e-12, (
        f"Upwind overshoot: max={float(jnp.max(f1)):.6f}"
    )
    assert float(jnp.min(f1)) >= 0.0 - 1.0e-12, (
        f"Upwind undershoot: min={float(jnp.min(f1)):.6f}"
    )


def test_upwind_varying_velocity_stays_bounded_but_not_monotone():
    """Sanity check on advective-form upwind under spatially varying
    velocity: the field is bounded above by ``max(f) + |u_max|*dt/dx``
    and below by ``min(f) - |u_max|*dt/dx`` (loose bound),
    documenting that strict TVD does not hold in this regime."""
    nx = 16
    dx = 100.0
    dt = 1.0
    f0 = jnp.zeros((1, nx, 1)).at[0, 4:8, 0].set(1.0)
    # Linearly varying u: -1 to +1 m/s across the domain.
    u = jnp.linspace(-1.0, 1.0, nx).reshape((1, nx, 1))
    df_dt = _upwind_advection_x(f0, u, dx)
    f1 = f0 + dt * df_dt
    # Bound: f1 stays within [-1, 2] roughly (loose; advective form
    # is not conservative so we can exceed [0, 1] slightly).
    assert float(jnp.max(f1)) < 2.0
    assert float(jnp.min(f1)) > -1.0
    assert bool(jnp.all(jnp.isfinite(f1)))


# --------------------------------------------------------------------- #
# Rest preservation in full dycore                                      #
# --------------------------------------------------------------------- #


def test_full_dycore_with_upwind_preserves_rest_state():
    """The dycore was already proven to preserve rest in PR2b; this
    pins that PR3b's upwind term doesn't break it."""
    model, grid, hc, _, _ = _setup()
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    next_state = model.step(state, dt=1.0)
    for name, field in (
        ("u", next_state.u.data), ("v", next_state.v.data),
        ("w", next_state.w.data),
        ("theta_p", next_state.theta_prime.data),
        ("rho_p", next_state.rho_prime.data),
    ):
        assert float(jnp.max(jnp.abs(field))) == 0.0, name


def test_upwind_is_differentiable_through_jax_grad():
    """``jax.grad`` should flow through both upwind branches."""
    model, grid, hc, tm, cfg = _setup()
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    rng = np.random.default_rng(0)
    u0 = jnp.asarray(rng.standard_normal(state.u.data.shape) * 0.1)

    def loss(u_data):
        s = state._replace(u=state.u.replace(data=u_data))
        tend = plane_compressible_euler_slow_tendencies(
            s, grid, hc, tm, cfg,
        )
        return jnp.sum(tend.du_dt.data ** 2)

    grad = jax.grad(loss)(u0)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.max(jnp.abs(grad))) > 0.0


def test_upwind_grad_through_zero_velocity_is_finite():
    """``jnp.maximum/minimum`` against ``0.0`` are non-smooth at the
    branch point but JAX returns a finite subgradient.
    ``jax.grad`` of a scalar loss involving the upwind tendency
    must stay finite (not NaN) even when ``u`` contains exact
    zeros — that is the property this test pins."""
    model, grid, hc, tm, cfg = _setup()
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    # Mix of negative, zero, and positive values to exercise the
    # branch point.
    u_with_zeros = jnp.zeros_like(state.u.data).at[0, 4:8, 0].set(0.1)
    u_with_zeros = u_with_zeros.at[0, 8:12, 0].set(-0.1)

    def loss(u_data):
        s = state._replace(u=state.u.replace(data=u_data))
        tend = plane_compressible_euler_slow_tendencies(
            s, grid, hc, tm, cfg,
        )
        return jnp.sum(tend.du_dt.data ** 2)

    grad = jax.grad(loss)(u_with_zeros)
    assert bool(jnp.all(jnp.isfinite(grad))), (
        "jax.grad produced non-finite values at u=0 kink — check "
        "jnp.maximum/minimum subgradient behaviour."
    )
