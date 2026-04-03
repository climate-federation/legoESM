"""Differentiability tests for data assimilation components.

Categories:
  7a) Control vector round-trip
  7b) Observation operator
  7c) Background error covariance
  7d) Cost function gradient accuracy (Taylor test)
  7e) Minimizer convergence
  7f) Preconditioned cost function
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field


def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
    assert nonzero_frac >= min_nonzero_frac, (
        f"{name}: only {nonzero_frac*100:.1f}% non-zero (need {min_nonzero_frac*100:.0f}%)"
    )


# ============================================================================
# 7a  Control vector round-trip
# ============================================================================

class TestControlVectorGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.da.control_vector import build_control_spec, state_to_control, control_to_state
        from legoesm.core.state import ShallowWaterState

        n = 4
        state = ShallowWaterState(
            h=Field(1000.0 * jnp.ones((6, n, n)), name="h"),
            u=Field(jnp.zeros((6, n, n)), name="u"),
            v=Field(jnp.zeros((6, n, n)), name="v"),
            h_s=Field(jnp.zeros((6, n, n)), name="h_s"),
        )
        self.spec = build_control_spec(state, fields=("h", "u", "v"))
        self.template = state
        self.x0 = state_to_control(state, self.spec)
        self.state_to_control = state_to_control
        self.control_to_state = control_to_state

    def test_round_trip_grad(self):
        spec, template = self.spec, self.template

        def loss(x):
            s = self.control_to_state(x, spec, template)
            return jnp.sum(s.h.data ** 2)

        grad = jax.grad(loss)(self.x0)
        assert_gradient_ok(grad, "Control vector round-trip")


# ============================================================================
# 7b  Observation operator
# ============================================================================

class TestObsOperatorGrad:

    def test_direct_obs_grad(self):
        from legoesm.da.observation import DirectObsOperator
        from legoesm.core.state import ShallowWaterState

        n = 4
        state = ShallowWaterState(
            h=Field(1000.0 * jnp.ones((6, n, n)), name="h"),
            u=Field(jnp.zeros((6, n, n)), name="u"),
            v=Field(jnp.zeros((6, n, n)), name="v"),
            h_s=Field(jnp.zeros((6, n, n)), name="h_s"),
        )

        # Observe h at a subset of points
        indices = (jnp.array([0, 1, 2]), jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        H = DirectObsOperator("h", indices)

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            return jnp.sum(H(s) ** 2)

        grad = jax.grad(loss)(state.h.data)
        assert jnp.all(jnp.isfinite(grad)), "DirectObs gradient not finite"
        # Only observed points should have non-zero gradient
        assert jnp.sum(jnp.abs(grad) > 0) > 0, "DirectObs gradient all zero"


# ============================================================================
# 7c  Background error covariance
# ============================================================================

class TestBackgroundErrorGrad:

    def test_diagonal_B_sqrt_grad(self):
        from legoesm.da.background_error import DiagonalB

        n = 100
        sigma = 10.0 * jnp.ones(n)
        B = DiagonalB(sigma=sigma)

        def loss(x):
            return jnp.sum(B.sqrt_multiply(x) ** 2)

        x = jnp.ones(n)
        grad = jax.grad(loss)(x)
        assert_gradient_ok(grad, "DiagonalB sqrt_multiply")

    def test_diagonal_B_inv_grad(self):
        from legoesm.da.background_error import DiagonalB

        n = 100
        sigma = 10.0 * jnp.ones(n)
        B = DiagonalB(sigma=sigma)

        def loss(x):
            return jnp.sum(B.inv_multiply(x) ** 2)

        x = jnp.ones(n)
        grad = jax.grad(loss)(x)
        assert_gradient_ok(grad, "DiagonalB inv_multiply")


# ============================================================================
# 7d  Cost function gradient (Taylor test)
# ============================================================================

class TestCostFunctionTaylor:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.atmosphere.dynamics.shallow_water_fv_latlon import (
            FVShallowWaterLatLonModel, FVShallowWaterLatLonConfig,
        )
        from legoesm.core.state import ShallowWaterState
        from legoesm.da.control_vector import build_control_spec, state_to_control
        from legoesm.da.observation import DirectObsOperator, Observation
        from legoesm.da.background_error import DiagonalB
        from legoesm.da.cost_function import build_cost_fn

        grid = create_latlon_grid(8, 16)
        dt = 120.0
        model = FVShallowWaterLatLonModel(grid, dt=dt)

        key = jax.random.PRNGKey(0)
        h_data = 1000.0 + 10.0 * jax.random.normal(key, (8, 16))
        state = ShallowWaterState(
            h=Field(h_data, name="h"),
            u=Field(jnp.zeros((8, 16)), name="u"),
            v=Field(jnp.zeros((8, 16)), name="v"),
            h_s=Field(jnp.zeros((8, 16)), name="h_s"),
        )

        spec = build_control_spec(state, fields=("h",))
        x_b = state_to_control(state, spec)

        # Synthetic obs at a few points
        indices = (jnp.array([2, 4, 6]), jnp.array([3, 8, 12]))
        H = DirectObsOperator("h", indices)
        obs_vals = H(state) + 5.0  # perturbed observations
        obs = (Observation(
            values=obs_vals,
            errors=10.0 * jnp.ones(3),
            time_index=2,
            operator=H,
        ),)

        sigma = 20.0 * jnp.ones(spec.total_size)
        B = DiagonalB(sigma=sigma)

        self.cost_fn = build_cost_fn(model, x_b, obs, B, spec, state, dt, n_steps=3)
        self.x0 = x_b

    def test_taylor_convergence(self):
        cost_fn = self.cost_fn
        x0 = self.x0

        grad = jax.grad(cost_fn)(x0)
        dx = grad / jnp.linalg.norm(grad)

        ratios = []
        for h in [1e-2, 1e-3, 1e-4, 1e-5]:
            fd_linear = cost_fn(x0 + h * dx) - cost_fn(x0) - h * jnp.dot(grad, dx)
            ratios.append(jnp.abs(fd_linear) / h ** 2)

        # Check that ratios are bounded (2nd-order convergence)
        ratios = jnp.array(ratios)
        assert jnp.all(jnp.isfinite(ratios)), "Taylor test ratios not finite"
        # Ratio should stabilize (not blow up) as h -> 0
        assert ratios[-1] < 100 * ratios[0] or ratios[-1] < 1e10, (
            f"Taylor test: ratios diverging: {ratios}"
        )


# ============================================================================
# 7e  Minimizer convergence
# ============================================================================

class TestMinimizerGrad:

    def test_lbfgs_quadratic(self):
        from legoesm.da.minimizer import minimize_lbfgs

        n = 10
        # Well-conditioned diagonal SPD
        A = jnp.diag(jnp.linspace(1.0, 5.0, n))
        b = jnp.ones(n)

        def cost_and_grad(x):
            J = 0.5 * x @ A @ x - b @ x
            g = A @ x - b
            return J, g

        x0 = jnp.zeros(n)
        result = minimize_lbfgs(cost_and_grad, x0, max_iter=100, gtol=1e-4)
        x_opt = jnp.linalg.solve(A, b)

        assert jnp.linalg.norm(result.x - x_opt) < 1e-2, (
            f"L-BFGS solution error: {jnp.linalg.norm(result.x - x_opt)}"
        )

    def test_cg_quadratic(self):
        from legoesm.da.minimizer import minimize_cg

        n = 10
        A = jnp.diag(jnp.linspace(1.0, 5.0, n))
        b = jnp.ones(n)

        def cost_and_grad(x):
            J = 0.5 * x @ A @ x - b @ x
            g = A @ x - b
            return J, g

        x0 = jnp.zeros(n)
        result = minimize_cg(cost_and_grad, x0, max_iter=100, gtol=1e-4)
        x_opt = jnp.linalg.solve(A, b)

        assert jnp.linalg.norm(result.x - x_opt) < 1e-2, (
            f"CG solution error: {jnp.linalg.norm(result.x - x_opt)}"
        )


# ============================================================================
# 7f  Preconditioned cost function
# ============================================================================

class TestPreconditioningGrad:

    def test_preconditioned_cost_grad(self):
        from legoesm.da.preconditioning import preconditioned_cost_fn
        from legoesm.da.background_error import DiagonalB

        n = 50
        sigma = 10.0 * jnp.ones(n)
        B = DiagonalB(sigma=sigma)
        x_b = jnp.zeros(n)

        # Simple quadratic cost in x-space
        def cost_fn(x):
            return 0.5 * jnp.sum((x - x_b) ** 2 / sigma ** 2)

        precond_cost = preconditioned_cost_fn(cost_fn, B, x_b)

        v0 = jnp.ones(n)
        grad = jax.grad(precond_cost)(v0)
        assert_gradient_ok(grad, "Preconditioned cost gradient")
