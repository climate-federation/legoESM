"""Tests for change-of-variable preconditioning."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.da.background_error import DiagonalB
from legoesm.da.preconditioning import preconditioned_cost_fn


class TestPreconditioning:
    def test_identity_at_background(self):
        """J~(0) == J(x_b)."""
        sigma = jnp.array([1.0, 2.0, 3.0])
        B = DiagonalB(sigma=sigma)
        x_b = jnp.array([10.0, 20.0, 30.0])

        def J(x):
            dx = x - x_b
            return 0.5 * jnp.sum(dx * B.inv_multiply(dx))

        J_tilde = preconditioned_cost_fn(J, B, x_b)
        v_zero = jnp.zeros(3)
        assert jnp.allclose(J_tilde(v_zero), J(x_b), atol=1e-6)

    def test_same_minimum(self):
        """Minimum of J~ maps to same x* as minimum of J."""
        sigma = jnp.array([1.0, 2.0])
        B = DiagonalB(sigma=sigma)
        x_b = jnp.array([5.0, 10.0])
        x_target = jnp.array([6.0, 11.0])

        def J(x):
            dx = x - x_b
            J_b = 0.5 * jnp.sum(dx * B.inv_multiply(dx))
            J_o = 0.5 * jnp.sum((x - x_target) ** 2 / 4.0)
            return J_b + J_o

        J_tilde = preconditioned_cost_fn(J, B, x_b)

        # Find minimizers (using JAX grad)
        from legoesm.da.minimizer import minimize_lbfgs

        # x-space
        J_vg = jax.value_and_grad(J)
        result_x = minimize_lbfgs(J_vg, x_b, max_iter=100, gtol=1e-8)

        # v-space
        Jt_vg = jax.value_and_grad(J_tilde)
        result_v = minimize_lbfgs(Jt_vg, jnp.zeros(2), max_iter=100, gtol=1e-8)
        x_from_v = x_b + B.sqrt_multiply(result_v.x)

        assert jnp.allclose(result_x.x, x_from_v, atol=1e-3)

    def test_differentiable(self):
        """Preconditioned cost should be differentiable."""
        sigma = jnp.array([1.0, 2.0, 3.0])
        B = DiagonalB(sigma=sigma)
        x_b = jnp.array([1.0, 2.0, 3.0])

        def J(x):
            return 0.5 * jnp.sum(x ** 2)

        J_tilde = preconditioned_cost_fn(J, B, x_b)
        v = jnp.array([0.1, -0.2, 0.3])
        grad = jax.grad(J_tilde)(v)
        assert jnp.all(jnp.isfinite(grad))
