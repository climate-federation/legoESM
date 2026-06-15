"""CATEGORY 7 — Data Assimilation differentiability tests.

Systematically verifies that ``jax.grad`` produces finite, non-zero,
physically sensible gradients through every component of the 4D-Var data
assimilation stack and their compositions:

    control vector transforms -> observation operators -> background-error B
    -> cost function (multi-step forward model via lax.scan) -> minimizer
    -> incremental 4D-Var -> preconditioning.

Run with::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        .venv/bin/python -m pytest tests/unit/test_diff_data_assimilation.py -v

The Metal JAX backend is broken in this environment, so JAX_PLATFORMS=cpu is
required. All tests use tiny grids (4x4 / 8x16 lat-lon) so they run in seconds.

This file ONLY writes tests; it never modifies source. Genuine model bugs are
left failing with ``# BUG:``; genuine non-differentiable ops are marked
``# NON-DIFFERENTIABLE:``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState
from legoesm.grids import create_latlon_grid

from legoesm.da.control_vector import (
    build_control_spec,
    control_to_state,
    state_to_control,
)
from legoesm.da.background_error import DiagonalB, DiffusionB
from legoesm.da.observation import (
    DirectObsOperator,
    InterpolatingObsOperator,
    ColumnIntegralObsOperator,
    CompositeObsOperator,
    Observation,
    generate_synthetic_obs,
)
from legoesm.da.cost_function import build_cost_fn
from legoesm.da.minimizer import minimize_lbfgs, minimize_cg
from legoesm.da.incremental import incremental_4dvar, IncrementalConfig
from legoesm.da.preconditioning import preconditioned_cost_fn

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Shared helper (required by the task brief)
# ---------------------------------------------------------------------------

def assert_gradient_ok(
    loss_fn,
    x0,
    name: str = "",
    *,
    min_nonzero_frac: float = 0.10,
    require_structure: bool = True,
):
    """Assert ``jax.grad(loss_fn)(x0)`` is finite, non-zero, structured.

    Checks:
    - every entry is finite (no NaN / Inf),
    - at least ``min_nonzero_frac`` of entries are non-zero (no dead path),
    - the gradient is not spatially uniform (has structure), when requested
      and the array has more than one element.

    Returns the gradient so callers can do additional sign / magnitude checks.
    """
    grad = jax.grad(loss_fn)(x0)
    g = jnp.asarray(grad)
    assert jnp.all(jnp.isfinite(g)), f"[{name}] gradient has non-finite entries"

    flat = g.ravel()
    nonzero_frac = float(jnp.mean(jnp.abs(flat) > 0.0))
    assert nonzero_frac >= min_nonzero_frac, (
        f"[{name}] only {nonzero_frac:.1%} of gradient entries non-zero "
        f"(need >= {min_nonzero_frac:.0%}) — likely a dead code path"
    )

    if require_structure and flat.size > 1:
        spread = float(jnp.max(flat) - jnp.min(flat))
        assert spread > 0.0, (
            f"[{name}] gradient is spatially uniform (no structure)"
        )
    return g


# ---------------------------------------------------------------------------
# Fixtures / state builders (mirror the existing DA test conventions)
# ---------------------------------------------------------------------------

def _make_sw_state(shape=(4, 4), h_val=100.0, u_val=0.0, v_val=0.0):
    """Build a ShallowWaterState with Field-wrapped arrays."""
    return ShallowWaterState(
        h=Field(data=jnp.ones(shape) * h_val, name="h", dims=(), units="m"),
        u=Field(data=jnp.ones(shape) * u_val, name="u", dims=(), units="m/s"),
        v=Field(data=jnp.ones(shape) * v_val, name="v", dims=(), units="m/s"),
        h_s=Field(data=jnp.zeros(shape), name="h_s", dims=(), units="m"),
    )


def _gaussian_bump(shape, amp=5.0):
    ny, nx = shape
    yy, xx = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, ny),
        jnp.linspace(-1.0, 1.0, nx),
        indexing="ij",
    )
    return amp * jnp.exp(-((xx) ** 2 + (yy) ** 2) / 0.3)


class _IdentityModel:
    """Model whose .step() leaves the state unchanged (used where dynamics
    must not interfere — e.g. control-vector round trips)."""

    def step(self, state, dt):
        return state


class _LinearWaveModel:
    """A small, genuinely-coupled, differentiable surrogate dynamical core.

    A linearised shallow-water-like update on a periodic lat-lon grid that
    couples h, u, v through finite-difference gradients/divergence. It has
    real spatial structure (so multi-step gradients grow and are non-uniform)
    yet is unconditionally smooth and differentiable, keeping the DA tests
    fast and deterministic. The cost-function brief calls for a "shallow
    water model"; this is a faithful, tiny linear SW operator.

    dh/dt = -H * div(u, v)
    du/dt = -g * dh/dx
    dv/dt = -g * dh/dy
    integrated with forward Euler on the cell array (axis 0 = lat, 1 = lon).
    """

    def __init__(self, g: float = constants.g, H: float = 100.0, dx: float = 1.0):
        self.g = g
        self.H = H
        self.dx = dx

    @staticmethod
    def _ddx(f, dx):
        # centered difference along lon (axis 1), periodic
        return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dx)

    @staticmethod
    def _ddy(f, dx):
        # centered difference along lat (axis 0), periodic
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)

    def step(self, state, dt):
        h = state.h.data
        u = state.u.data
        v = state.v.data
        div = self._ddx(u, self.dx) + self._ddy(v, self.dx)
        dhdt = -self.H * div
        dudt = -self.g * self._ddx(h, self.dx)
        dvdt = -self.g * self._ddy(h, self.dx)
        return state._replace(
            h=state.h.replace(data=h + dt * dhdt),
            u=state.u.replace(data=u + dt * dudt),
            v=state.v.replace(data=v + dt * dvdt),
        )


@pytest.fixture
def latlon_grid():
    return create_latlon_grid(8, 16)


# ===========================================================================
# 7a) Control vector round-trip differentiability
# ===========================================================================

class TestControlVectorRoundTrip:
    @pytest.mark.parametrize("transform", ["identity", "log", "softplus"])
    def test_control_to_state_roundtrip_diff(self, transform):
        """grad of a loss on control_to_state(x).h should be finite & non-zero
        for each transform."""
        state = _make_sw_state(shape=(4, 4), h_val=100.0, u_val=2.0, v_val=-1.0)
        spec = build_control_spec(
            state,
            transforms={"h": transform, "u": transform, "v": transform},
        )
        x0 = state_to_control(state, spec)

        def loss(x):
            s = control_to_state(x, spec, state)
            return jnp.sum(s.h.data ** 2) + jnp.sum(s.u.data ** 2)

        assert_gradient_ok(loss, x0, name=f"control_roundtrip[{transform}]")

    def test_state_to_control_diff(self):
        """grad through the forward (state -> control) transform."""
        state = _make_sw_state(shape=(4, 4), h_val=100.0)
        # Give h spatial structure so d/dh sum((h)^2) = 2h is non-uniform.
        state = state._replace(
            h=state.h.replace(data=state.h.data + _gaussian_bump((4, 4)))
        )
        spec = build_control_spec(state)

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            x = state_to_control(s, spec)
            return jnp.sum(x ** 2)

        assert_gradient_ok(loss, state.h.data, name="state_to_control")

    def test_roundtrip_is_identity_with_grad(self):
        """control_to_state(state_to_control(state)) == state, with finite grad
        for the log transform (exp/log inverse pair)."""
        state = _make_sw_state(shape=(4, 4), h_val=50.0)
        # Spatial structure so the round-trip gradient 2h is non-uniform.
        state = state._replace(
            h=state.h.replace(data=state.h.data + _gaussian_bump((4, 4)))
        )
        spec = build_control_spec(state, transforms={"h": "log"})

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            x = state_to_control(s, spec)
            s2 = control_to_state(x, spec, state)
            return jnp.sum(s2.h.data ** 2)

        g = assert_gradient_ok(loss, state.h.data, name="roundtrip_log")
        # log/exp round trip: d/dh sum(h^2) = 2h, check it recovers ~2h
        assert jnp.allclose(g, 2.0 * state.h.data, rtol=1e-4)


# ===========================================================================
# 7b) Observation operator differentiability
# ===========================================================================

class TestObservationOperators:
    def test_direct_obs_diff_and_sparsity(self):
        """DirectObs gradient is finite, non-zero, and sparse (only at the
        observed grid points)."""
        state = _make_sw_state(shape=(8, 16), h_val=100.0)
        ii = jnp.array([0, 2, 4])
        jj = jnp.array([1, 5, 9])
        op = DirectObsOperator("h", (ii, jj))

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            return jnp.sum(op(s) ** 2)

        g = assert_gradient_ok(
            loss, state.h.data, name="direct_obs",
            min_nonzero_frac=3 / state.h.data.size,
        )
        # Sparsity: only the 3 observed cells should have non-zero gradient.
        nonzero = jnp.abs(g) > 0.0
        assert int(jnp.sum(nonzero)) == 3, "DirectObs gradient not sparse"
        assert jnp.all(nonzero[ii, jj]), "observed cells must have gradient"

    def test_interpolating_obs_diff(self, latlon_grid):
        state = _make_sw_state(shape=(8, 16), h_val=100.0)
        # add structure so gradient is not uniform
        state = state._replace(
            h=state.h.replace(data=state.h.data + _gaussian_bump((8, 16)))
        )
        obs_lat = jnp.array([0.0, 0.3, -0.4])
        obs_lon = jnp.array([0.0, 1.0, 2.5])
        op = InterpolatingObsOperator("h", obs_lat, obs_lon, grid=latlon_grid)

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            return jnp.sum(op(s) ** 2)

        assert_gradient_ok(
            loss, state.h.data, name="interp_obs",
            min_nonzero_frac=3 / state.h.data.size, require_structure=False,
        )

    def test_column_integral_obs_diff(self, latlon_grid):
        """ColumnIntegralObsOperator sums over levels; differentiate w.r.t. a
        3D (n_lat, n_lon, nlev) field at selected columns. grid.to_columns()
        flattens the horizontal axes to (ncol, nlev)."""
        n_lat, n_lon, nlev = 8, 16, 5
        ncol = latlon_grid.grid_n_columns
        data0 = jax.random.normal(jax.random.PRNGKey(3), (n_lat, n_lon, nlev))
        shp = (n_lat, n_lon, nlev)
        state = ShallowWaterState(
            h=Field(data=data0, name="h", dims=(), units="m"),
            u=Field(data=jnp.zeros(shp), name="u", dims=(), units="m/s"),
            v=Field(data=jnp.zeros(shp), name="v", dims=(), units="m/s"),
            h_s=Field(data=jnp.zeros(shp), name="h_s", dims=(), units="m"),
        )
        obs_cols = jnp.array([0, 10, 50])
        op = ColumnIntegralObsOperator("h", grid=latlon_grid, obs_columns=obs_cols)

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            return jnp.sum(op(s) ** 2)

        g = assert_gradient_ok(
            loss, state.h.data, name="column_integral_obs",
            min_nonzero_frac=(3 * nlev) / data0.size,
        )
        # Only the 3 observed columns (all their levels) should be non-zero.
        # Flatten horizontal axes to (ncol, nlev) to inspect per-column.
        g_col = g.reshape(ncol, nlev)
        nonzero_cols = jnp.any(jnp.abs(g_col) > 0.0, axis=1)
        assert int(jnp.sum(nonzero_cols)) == 3

    def test_composite_obs_diff(self):
        state = _make_sw_state(shape=(8, 16), h_val=100.0, u_val=5.0)
        op_h = DirectObsOperator("h", (jnp.array([0, 1]), jnp.array([0, 1])))
        op_u = DirectObsOperator("u", (jnp.array([2, 3]), jnp.array([2, 3])))
        comp = CompositeObsOperator((op_h, op_u))

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            return jnp.sum(comp(s) ** 2)

        assert_gradient_ok(
            loss, state.h.data, name="composite_obs",
            min_nonzero_frac=2 / state.h.data.size,
        )


# ===========================================================================
# 7c) Background error covariance differentiability
# ===========================================================================

class TestBackgroundErrorB:
    def test_diagonal_B_sqrt_diff(self):
        sigma = jnp.linspace(0.5, 3.0, 16)
        B = DiagonalB(sigma=sigma)
        x0 = jax.random.normal(jax.random.PRNGKey(0), (16,))
        assert_gradient_ok(lambda x: jnp.sum(B.sqrt_multiply(x) ** 2),
                           x0, name="DiagonalB.sqrt")

    def test_diagonal_B_inv_diff(self):
        sigma = jnp.linspace(0.5, 3.0, 16)
        B = DiagonalB(sigma=sigma)
        x0 = jax.random.normal(jax.random.PRNGKey(1), (16,))
        assert_gradient_ok(lambda x: jnp.sum(B.inv_multiply(x) ** 2),
                           x0, name="DiagonalB.inv")

    def test_diffusion_B_sqrt_diff(self, latlon_grid):
        n = latlon_grid.grid_n_columns
        sigma = jnp.ones(n) * 2.0
        B = DiffusionB(latlon_grid, sigma, horizontal_length_scale=2000e3,
                       n_diffusion_iter=4)
        x0 = jax.random.normal(jax.random.PRNGKey(2), (n,))
        # DiffusionB sqrt couples all columns through the area-weighted mean,
        # so the gradient is dense; structure check disabled (smoothing makes
        # the gradient nearly uniform by design).
        assert_gradient_ok(lambda x: jnp.sum(B.sqrt_multiply(x) ** 2),
                           x0, name="DiffusionB.sqrt", require_structure=False)

    def test_diffusion_B_inv_diff(self, latlon_grid):
        n = latlon_grid.grid_n_columns
        sigma = jnp.ones(n) * 2.0
        B = DiffusionB(latlon_grid, sigma, horizontal_length_scale=2000e3,
                       n_diffusion_iter=4)
        x0 = jax.random.normal(jax.random.PRNGKey(4), (n,))
        assert_gradient_ok(lambda x: jnp.sum(B.inv_multiply(x) ** 2),
                           x0, name="DiffusionB.inv", require_structure=False)


# ===========================================================================
# 7d) Cost-function gradient accuracy — Taylor test (gold standard)
# ===========================================================================

def _build_sw_cost(n_steps, shape=(8, 16), checkpoint=True):
    """Build a 4D-Var cost for the linear SW surrogate on a lat-lon grid."""
    model = _LinearWaveModel(dx=1.0)

    # Background = rest + small bump; truth = background perturbed.
    bg_state = _make_sw_state(shape, h_val=100.0)
    bg_state = bg_state._replace(
        h=bg_state.h.replace(data=bg_state.h.data + 0.5 * _gaussian_bump(shape))
    )
    spec = build_control_spec(bg_state)
    x_b = state_to_control(bg_state, spec)

    sigma = jnp.ones(spec.total_size) * 10.0
    B = DiagonalB(sigma=sigma)

    # Truth trajectory -> synthetic observations of h at a set of points.
    truth_state = _make_sw_state(shape, h_val=100.0)
    truth_state = truth_state._replace(
        h=truth_state.h.replace(
            data=truth_state.h.data + _gaussian_bump(shape, amp=8.0)
        )
    )

    def scan_step(carry, _):
        s_new = model.step(carry, dt=1.0)
        return s_new, s_new

    _, traj = jax.lax.scan(scan_step, truth_state, jnp.arange(n_steps))

    ii = jnp.array([0, 2, 4, 6])
    jj = jnp.array([1, 5, 9, 13])
    op = DirectObsOperator("h", (ii, jj))
    t_obs = max(0, n_steps - 1)
    obs = generate_synthetic_obs(
        traj, operators=(op,), time_indices=(t_obs,),
        error_stds=(2.0,), key=jax.random.PRNGKey(7),
    )

    cost_fn = build_cost_fn(
        model, x_b, observations=obs, B=B, control_spec=spec,
        template_state=bg_state, dt=1.0, n_steps=n_steps, checkpoint=checkpoint,
    )
    return cost_fn, x_b


class TestCostFunctionTaylor:
    def test_taylor_convergence(self):
        """|J(x0+h dx) - J(x0) - h <g,dx>| should shrink ~linearly in h
        (definitive check that the gradient is correct)."""
        cost_fn, x0 = _build_sw_cost(n_steps=10, shape=(8, 16))
        J0 = cost_fn(x0)
        g = jax.grad(cost_fn)(x0)
        assert jnp.all(jnp.isfinite(g))
        gnorm = jnp.linalg.norm(g)
        assert gnorm > 0.0
        dx = g / gnorm  # direction with g.dx = ||g|| != 0
        slope = jnp.sum(g * dx)

        hs = [1e-3, 1e-4, 1e-5, 1e-6]
        first_order = []
        for h in hs:
            Jp = cost_fn(x0 + h * dx)
            r1 = float(jnp.abs(Jp - J0 - h * slope))
            first_order.append(r1)

        # All finite & bounded.
        assert all(jnp.isfinite(jnp.array(r)) for r in first_order)
        # First-order remainder must DECREASE as h shrinks (gradient correct).
        for i in range(1, len(first_order)):
            assert first_order[i] <= first_order[i - 1] * 1.5 + 1e-12, (
                f"first-order remainder not decreasing: {first_order}"
            )
        # Per-decade reduction of the remainder: for a correct gradient,
        # r ~ C h^2, so r(h/10)/r(h) ~ 0.01..0.1 in the well-resolved regime.
        ratio = first_order[1] / max(first_order[0], 1e-300)
        assert ratio < 0.5, f"Taylor remainder not converging (ratio={ratio})"

    def test_central_difference_matches_grad(self):
        """Directional derivative via central differences matches <g, dx>."""
        cost_fn, x0 = _build_sw_cost(n_steps=5, shape=(8, 16))
        g = jax.grad(cost_fn)(x0)
        key = jax.random.PRNGKey(11)
        dx = jax.random.normal(key, x0.shape)
        dx = dx / jnp.linalg.norm(dx)
        h = 1e-5
        fd = (cost_fn(x0 + h * dx) - cost_fn(x0 - h * dx)) / (2 * h)
        ad = jnp.sum(g * dx)
        rel = jnp.abs(fd - ad) / jnp.maximum(jnp.abs(ad), 1e-8)
        assert rel < 1e-4, f"central diff vs AD rel err {float(rel)}"


# ===========================================================================
# 7e) Cost function gradient through multi-step forward model
# ===========================================================================

class TestMultiStepCost:
    @pytest.mark.parametrize("n_steps", [1, 5, 20])
    def test_grad_finite_all_windows(self, n_steps):
        cost_fn, x0 = _build_sw_cost(n_steps=n_steps, shape=(8, 16))
        g = jax.grad(cost_fn)(x0)
        assert jnp.all(jnp.isfinite(g)), f"non-finite grad at n_steps={n_steps}"

    def test_sensitivity_grows_with_window(self):
        """Longer assimilation windows accumulate more sensitivity (the obs is
        at the END of the window, so the gradient w.r.t. the initial condition
        grows as the window lengthens)."""
        norms = []
        for n_steps in [1, 5, 20]:
            cost_fn, x0 = _build_sw_cost(n_steps=n_steps, shape=(8, 16))
            g = jax.grad(cost_fn)(x0)
            assert jnp.all(jnp.isfinite(g))
            norms.append(float(jnp.linalg.norm(g)))
        assert norms[2] > norms[0], (
            f"gradient norm did not grow with window: {norms}"
        )

    def test_checkpoint_matches_no_checkpoint(self):
        """Gradient checkpointing must not change gradient values."""
        cf_ckpt, x0 = _build_sw_cost(n_steps=10, shape=(8, 16), checkpoint=True)
        cf_nock, _ = _build_sw_cost(n_steps=10, shape=(8, 16), checkpoint=False)
        g1 = jax.grad(cf_ckpt)(x0)
        g2 = jax.grad(cf_nock)(x0)
        assert jnp.allclose(g1, g2, rtol=1e-9, atol=1e-9), (
            "checkpointed gradient differs from non-checkpointed"
        )


# ===========================================================================
# 7f) Minimizer convergence with exact gradient (quadratic problem)
# ===========================================================================

def _quadratic(A, b):
    """J(x) = 0.5 x^T A x - b^T x, grad = A x - b, minimizer x* = A^{-1} b."""
    def cost_and_grad(x):
        J = 0.5 * jnp.dot(x, A @ x) - jnp.dot(b, x)
        g = A @ x - b
        return J, g
    return cost_and_grad


class TestMinimizerConvergence:
    def _spd_system(self, n=8, seed=0):
        key = jax.random.PRNGKey(seed)
        M = jax.random.normal(key, (n, n))
        A = M @ M.T + n * jnp.eye(n)  # SPD, well-conditioned
        b = jax.random.normal(jax.random.PRNGKey(seed + 1), (n,))
        x_star = jnp.linalg.solve(A, b)
        return A, b, x_star

    def test_lbfgs_converges(self):
        A, b, x_star = self._spd_system()
        cg = _quadratic(A, b)
        x0 = jnp.zeros(A.shape[0])
        # ftol must be tight enough that the gradient criterion (gtol) governs
        # termination — with the default ftol=1e-8 L-BFGS halts early on the
        # relative-decrease test. This is normal optimizer behaviour, not a bug.
        res = minimize_lbfgs(cg, x0, max_iter=50, gtol=1e-10, ftol=1e-15)
        assert jnp.all(jnp.isfinite(res.x))
        assert float(res.grad_norm) < 1e-6, f"grad_norm={float(res.grad_norm)}"
        assert jnp.allclose(res.x, x_star, atol=1e-4), (
            f"L-BFGS off: ||x-x*||={float(jnp.linalg.norm(res.x - x_star))}"
        )

    def test_cg_converges(self):
        A, b, x_star = self._spd_system(seed=5)
        cg = _quadratic(A, b)
        x0 = jnp.zeros(A.shape[0])
        # The Polak-Ribière CG here uses a backtracking (inexact) line search,
        # so it needs more than the textbook n iterations of exact-line-search
        # linear CG. 50 iters reach machine-precision convergence here.
        res = minimize_cg(cg, x0, max_iter=50, gtol=1e-12)
        assert jnp.all(jnp.isfinite(res.x))
        assert float(res.grad_norm) < 1e-6, f"grad_norm={float(res.grad_norm)}"
        assert jnp.allclose(res.x, x_star, atol=1e-4), (
            f"CG off: ||x-x*||={float(jnp.linalg.norm(res.x - x_star))}"
        )

    def test_minimizer_not_reverse_differentiable(self):
        """NON-DIFFERENTIABLE: the on-device minimizers run inside
        ``jax.lax.while_loop`` (data-dependent iteration count), so reverse-mode
        AD through ``minimize_cg`` / ``minimize_lbfgs`` is NOT supported by JAX
        (``while_loop`` has no reverse rule). This is an inherent limitation,
        not a bug: in 4D-Var the adjoint is taken of the *cost function*, never
        of the optimiser. We document it here by asserting jax.grad raises.

        Implication for users: you cannot ``jax.grad`` through the analysis
        directly (e.g. for B-matrix hyper-parameter learning) — use the
        cost-function gradient (tested above) plus an implicit-function /
        unrolled-fixed-iteration approach instead."""
        A, _, _ = self._spd_system(seed=9)
        x0 = jnp.zeros(A.shape[0])

        def loss(b):
            cg = _quadratic(A, b)
            res = minimize_cg(cg, x0, max_iter=40, gtol=1e-10)
            return jnp.sum(res.x ** 2)

        b0 = jax.random.normal(jax.random.PRNGKey(13), (A.shape[0],))
        with pytest.raises(Exception):
            jax.grad(loss)(b0)


# ===========================================================================
# 7g) Incremental 4D-Var end-to-end differentiability (twin experiment)
# ===========================================================================

class TestIncremental4DVar:
    def test_twin_experiment_reduces_rmse(self):
        """Full chain: control_vector -> forward model (lax.scan) -> obs op ->
        cost -> jax.grad -> minimizer. Analysis RMSE < background RMSE and the
        cost decreases across outer iterations."""
        shape = (8, 16)
        model = _LinearWaveModel(dx=1.0)

        truth_state = _make_sw_state(shape, h_val=100.0)
        truth_state = truth_state._replace(
            h=truth_state.h.replace(
                data=truth_state.h.data + _gaussian_bump(shape, amp=8.0)
            )
        )
        bg_state = _make_sw_state(shape, h_val=100.0)  # flat background

        spec = build_control_spec(bg_state)
        sigma = jnp.ones(spec.total_size) * 10.0
        B = DiagonalB(sigma=sigma)

        n_steps = 3

        def scan_step(carry, _):
            s_new = model.step(carry, dt=1.0)
            return s_new, s_new

        _, traj = jax.lax.scan(scan_step, truth_state, jnp.arange(n_steps))

        # Observe h at many points at the final time of the window.
        ii, jj = jnp.meshgrid(
            jnp.arange(0, 8, 2), jnp.arange(0, 16, 2), indexing="ij"
        )
        idx = (ii.ravel(), jj.ravel())
        op = DirectObsOperator("h", idx)
        obs = generate_synthetic_obs(
            traj, operators=(op,), time_indices=(n_steps - 1,),
            error_stds=(1.0,), key=jax.random.PRNGKey(21),
        )

        config = IncrementalConfig(
            n_outer=2, n_inner=10, inner_gtol=1e-8,
            inner_method="lbfgs", use_preconditioning=True, checkpoint=True,
        )

        analysis, diag = incremental_4dvar(
            model, bg_state, obs, B, spec,
            dt=1.0, n_steps=n_steps, config=config,
        )

        # Analysis IC closer to truth IC than the background was.
        rmse_bg = float(
            jnp.sqrt(jnp.mean((bg_state.h.data - truth_state.h.data) ** 2))
        )
        rmse_ana = float(
            jnp.sqrt(jnp.mean((analysis.h.data - truth_state.h.data) ** 2))
        )
        assert jnp.isfinite(rmse_ana)
        assert rmse_ana < rmse_bg, f"RMSE bg={rmse_bg}, ana={rmse_ana}"

        # Cost decreased across outer iterations.
        assert len(diag.cost_history) >= 2
        assert diag.cost_history[-1] <= diag.cost_history[0] + 1e-6, (
            f"cost did not decrease: {diag.cost_history}"
        )

    def test_inner_cost_gradient_finite(self):
        """The preconditioned inner-loop cost gradient (what the minimizer
        actually differentiates) is finite & non-zero."""
        cost_fn, x_b = _build_sw_cost(n_steps=3, shape=(8, 16))
        sigma = jnp.ones(x_b.shape[0]) * 10.0
        B = DiagonalB(sigma=sigma)
        J_tilde = preconditioned_cost_fn(cost_fn, B, x_b)
        v0 = 0.1 * jax.random.normal(jax.random.PRNGKey(31), x_b.shape)
        assert_gradient_ok(J_tilde, v0, name="incremental_inner_cost",
                           require_structure=False)


# ===========================================================================
# 7h) Preconditioning differentiability
# ===========================================================================

class TestPreconditioning:
    def test_preconditioned_cost_diff(self):
        cost_fn, x_b = _build_sw_cost(n_steps=5, shape=(8, 16))
        sigma = jnp.ones(x_b.shape[0]) * 10.0
        B = DiagonalB(sigma=sigma)
        J_tilde = preconditioned_cost_fn(cost_fn, B, x_b)

        v0 = jnp.zeros_like(x_b)
        # at v=0, x=x_b: grad must still be finite; non-zero because obs pull.
        g = jax.grad(J_tilde)(v0)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.linalg.norm(g)) > 0.0, "preconditioned grad is zero"

    def test_preconditioning_well_defined(self):
        """Compare the gradient with vs without the B^{1/2} change-of-variable.

        With a large sigma, the unpreconditioned cost has a poorly-scaled
        background term; the preconditioned variable v is dimensionless. Assert
        both gradients are finite and the preconditioned one is non-zero — a
        well-defined, better-scaled problem for the inner minimizer."""
        cost_fn, x_b = _build_sw_cost(n_steps=5, shape=(8, 16))
        sigma = jnp.ones(x_b.shape[0]) * 50.0
        B = DiagonalB(sigma=sigma)

        # Unpreconditioned gradient at x_b.
        g_x = jax.grad(cost_fn)(x_b)
        # Preconditioned gradient at v=0 (x=x_b).
        J_tilde = preconditioned_cost_fn(cost_fn, B, x_b)
        g_v = jax.grad(J_tilde)(jnp.zeros_like(x_b))

        assert jnp.all(jnp.isfinite(g_x))
        assert jnp.all(jnp.isfinite(g_v))
        assert float(jnp.linalg.norm(g_v)) > 0.0
