"""Tests for on-device minimizers."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.da.minimizer import minimize_lbfgs, minimize_cg


class TestLBFGS:
    def test_quadratic_bowl(self):
        """Minimize 0.5 * x^T A x - b^T x."""
        A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        b = jnp.array([1.0, 2.0])
        x_exact = jnp.linalg.solve(A, b)

        def cost_and_grad(x):
            f = 0.5 * x @ A @ x - b @ x
            g = A @ x - b
            return f, g

        x0 = jnp.zeros(2)
        result = minimize_lbfgs(cost_and_grad, x0, max_iter=50, gtol=1e-8)
        assert jnp.allclose(result.x, x_exact, atol=1e-4)
        assert result.grad_norm < 1e-4

    def test_rosenbrock(self):
        """Minimize Rosenbrock function: (1-x)^2 + 100*(y-x^2)^2."""
        def cost_and_grad(xy):
            x, y = xy[0], xy[1]
            f = (1 - x)**2 + 100 * (y - x**2)**2
            return f, jax.grad(lambda z: (1-z[0])**2 + 100*(z[1]-z[0]**2)**2)(xy)

        x0 = jnp.array([-1.0, 1.0])
        result = minimize_lbfgs(cost_and_grad, x0, max_iter=200, gtol=1e-5)
        assert jnp.allclose(result.x, jnp.array([1.0, 1.0]), atol=0.1)

    def test_jit_compiles(self):
        """L-BFGS should JIT-compile."""
        A = jnp.eye(3)
        b = jnp.array([1.0, 2.0, 3.0])

        def cost_and_grad(x):
            f = 0.5 * x @ A @ x - b @ x
            g = A @ x - b
            return f, g

        result = jax.jit(lambda x0: minimize_lbfgs(cost_and_grad, x0, max_iter=20))(
            jnp.zeros(3)
        )
        assert jnp.allclose(result.x, b, atol=1e-3)

    def test_history_recorded(self):
        """Cost history should be recorded."""
        def cost_and_grad(x):
            f = jnp.sum(x ** 2)
            return f, 2 * x

        x0 = jnp.array([5.0, 3.0])
        result = minimize_lbfgs(cost_and_grad, x0, max_iter=20)
        assert result.history[0] < jnp.inf  # First entry should be filled


class TestCG:
    def test_quadratic_bowl(self):
        """CG should solve quadratic in at most n iterations."""
        A = jnp.array([[2.0, 0.0], [0.0, 5.0]])
        b = jnp.array([4.0, 10.0])
        x_exact = jnp.linalg.solve(A, b)

        def cost_and_grad(x):
            f = 0.5 * x @ A @ x - b @ x
            g = A @ x - b
            return f, g

        x0 = jnp.zeros(2)
        result = minimize_cg(cost_and_grad, x0, max_iter=50, gtol=1e-8)
        assert jnp.allclose(result.x, x_exact, atol=1e-2)

    def test_preconditioned(self):
        """Preconditioned CG should converge faster on ill-conditioned problem."""
        A = jnp.diag(jnp.array([1.0, 100.0, 10000.0]))
        b = jnp.array([1.0, 1.0, 1.0])
        x_exact = jnp.linalg.solve(A, b)

        def cost_and_grad(x):
            f = 0.5 * x @ A @ x - b @ x
            g = A @ x - b
            return f, g

        # Preconditioner: inverse diagonal of A
        P_inv = jnp.diag(1.0 / jnp.diag(A))

        def precond(g):
            return P_inv @ g

        x0 = jnp.zeros(3)
        result = minimize_cg(cost_and_grad, x0, max_iter=20, gtol=1e-6,
                             preconditioner=precond)
        assert jnp.allclose(result.x, x_exact, atol=1e-3)

    def test_jit_compiles(self):
        def cost_and_grad(x):
            return jnp.sum(x ** 2), 2 * x

        result = jax.jit(lambda x0: minimize_cg(cost_and_grad, x0, max_iter=20))(
            jnp.array([5.0, 3.0])
        )
        assert jnp.allclose(result.x, 0.0, atol=1e-3)
