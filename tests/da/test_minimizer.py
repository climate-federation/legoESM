"""Tests for on-device minimizers."""

import jax
import jax.numpy as jnp
import numpy as np
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


class TestLBFGSDescentSafeguards:
    def test_history_monotone_on_nonconvex_cost(self):
        """Negative-curvature pairs must not enter the memory and a failed
        line search must not be accepted, so the cost never increases."""
        def f(x):
            return jnp.sum(jnp.cos(3 * x) + 0.1 * x ** 2) + 0.5 * x[0] * x[1]

        vg = jax.value_and_grad(f)
        for seed in range(20):
            x0 = 2.0 * jax.random.normal(jax.random.PRNGKey(seed), (4,))
            r = minimize_lbfgs(vg, x0, max_iter=30, gtol=1e-12, ftol=0.0)
            h = np.asarray(r.history)
            h = h[np.isfinite(h)]
            assert np.all(np.diff(h) <= 1e-12), (seed, h)
            assert float(r.fun) <= float(f(x0)) + 1e-12, seed

    def test_failed_line_search_keeps_current_point(self):
        """Inconsistent gradient (wrong sign): no step satisfies Armijo, so
        the minimizer must stop at x0 instead of accepting an ascent step."""
        def vg(x):
            return jnp.sum(x ** 2), -2.0 * x

        x0 = jnp.array([1.0, -2.0, 0.5])
        r = minimize_lbfgs(vg, x0, max_iter=5)
        np.testing.assert_allclose(np.asarray(r.x), np.asarray(x0))
        assert float(r.fun) == float(jnp.sum(x0 ** 2))
        assert not bool(r.converged)
        assert bool(r.line_search_failed)


class TestCGDescentSafeguards:
    def test_failed_line_search_keeps_current_point(self):
        """Wrong-sign gradient: no step meets Armijo, so CG must stay at x0
        and report the failure instead of accepting an ascent step."""
        def vg(x):
            return jnp.sum(x ** 2), -2.0 * x

        x0 = jnp.array([1.0, -2.0, 0.5])
        r = minimize_cg(vg, x0, max_iter=5)
        np.testing.assert_allclose(np.asarray(r.x), np.asarray(x0))
        assert float(r.fun) == float(jnp.sum(x0 ** 2))
        assert bool(r.line_search_failed)

    def test_history_monotone_on_nonconvex_cost(self):
        def f(x):
            return jnp.sum(jnp.cos(3 * x) + 0.1 * x ** 2) + 0.5 * x[0] * x[1]

        vg = jax.value_and_grad(f)
        for seed in range(20):
            x0 = 2.0 * jax.random.normal(jax.random.PRNGKey(seed), (4,))
            r = minimize_cg(vg, x0, max_iter=30, gtol=1e-12)
            h = np.asarray(r.history)
            h = h[np.isfinite(h)]
            assert np.all(np.diff(h) <= 1e-12), (seed, h)


class TestLineSearchScaling:
    def test_lbfgs_small_gradient_not_enlarged(self):
        """f = x^2 from x0 = 1e-8: a unit-length first step overshoots every
        acceptable point even at 2^-20; the first step must not be enlarged
        beyond -g (a half-step of -g reaches the minimum)."""
        r = minimize_lbfgs(lambda x: (jnp.sum(x ** 2), 2.0 * x),
                           jnp.array([1e-8]), gtol=1e-12)
        assert not bool(r.line_search_failed)
        assert float(r.fun) < 1e-20

    def test_cg_recurrence_uses_the_retried_direction(self, monkeypatch):
        """f = 0.5e8 x^2 from x0 = 0.75: the first search fails and the
        capped retry lands at x = -0.25. Every direction CG searches must be
        a descent direction (g . d < 0); building the next direction from
        the unscaled pre-retry d made it point uphill."""
        import legoesm.da.minimizer as mz

        slopes = []
        real = mz._backtracking_line_search

        def spy(fn, x, f, g, d, *a, **k):
            jax.debug.callback(lambda v: slopes.append(float(v)), jnp.sum(g * d))
            return real(fn, x, f, g, d, *a, **k)

        monkeypatch.setattr(mz, "_backtracking_line_search", spy)

        def vg(x):
            return 0.5e8 * jnp.sum(x ** 2), 1e8 * x

        mz.minimize_cg(vg, jnp.array([0.75]), max_iter=4, gtol=1e-12)
        assert len(slopes) >= 3, slopes
        assert all(v < 0.0 for v in slopes), slopes
