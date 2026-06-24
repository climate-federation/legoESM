"""Correctness tests for matrix-free second-order AD (curvature.py).

Every operator is checked against an analytic/dense ground truth: HVP vs the
exact quadratic Hessian and vs a finite difference of the gradient; the
Gauss-Newton product vs J^T R^-1 J; Lanczos vs dense eigenvalues; the Laplace
covariance vs an explicit inverse.
"""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.da import curvature as cv  # noqa: E402


def _spd(n, seed=0):
    M = jax.random.normal(jax.random.PRNGKey(seed), (n, n))
    return M @ M.T + n * jnp.eye(n)


def test_hvp_exact_on_quadratic():
    A = _spd(6, seed=1)
    f = lambda x: 0.5 * x @ A @ x
    x = jax.random.normal(jax.random.PRNGKey(2), (6,))
    v = jax.random.normal(jax.random.PRNGKey(3), (6,))
    assert jnp.allclose(cv.hvp(f, x, v), A @ v, rtol=1e-9, atol=1e-10)


def test_hvp_matches_finite_difference():
    f = lambda x: jnp.sum(jnp.sin(x) * x ** 2)
    x = jnp.linspace(-1.0, 1.0, 7)
    v = jax.random.normal(jax.random.PRNGKey(4), (7,))
    eps = 1e-5
    fd = (jax.grad(f)(x + eps * v) - jax.grad(f)(x - eps * v)) / (2 * eps)
    assert jnp.allclose(cv.hvp(f, x, v), fd, rtol=1e-5, atol=1e-6)


def test_gauss_newton_value_and_psd():
    m, n = 8, 5
    J = jax.random.normal(jax.random.PRNGKey(5), (m, n))
    y0 = jax.random.normal(jax.random.PRNGKey(6), (m,))
    residual_fn = lambda x: J @ x - y0
    x = jax.random.normal(jax.random.PRNGKey(7), (n,))
    v = jax.random.normal(jax.random.PRNGKey(8), (n,))
    # r_inv = None  =>  J^T J v
    assert jnp.allclose(cv.gauss_newton_hvp(residual_fn, x, v), J.T @ (J @ v),
                        rtol=1e-9, atol=1e-10)
    # PSD: v^T (J^T J) v >= 0
    assert float(v @ cv.gauss_newton_hvp(residual_fn, x, v)) >= -1e-9
    # diagonal R^-1
    r_inv = jnp.abs(jax.random.normal(jax.random.PRNGKey(9), (m,))) + 0.1
    assert jnp.allclose(
        cv.gauss_newton_hvp(residual_fn, x, v, r_inv=r_inv),
        J.T @ (r_inv * (J @ v)), rtol=1e-9, atol=1e-10,
    )


def test_dense_hessian_quadratic():
    A = _spd(4, seed=10)
    f = lambda x: 0.5 * x @ A @ x
    x = jnp.ones(4)
    assert jnp.allclose(cv.dense_hessian(f, x), A, rtol=1e-9, atol=1e-10)


def test_dense_hessian_rejects_pytree_shape():
    with pytest.raises(ValueError, match="1-D array"):
        cv.dense_hessian(lambda x: jnp.sum(x ** 2), jnp.ones((2, 2)))


def test_dense_from_matvec():
    M = _spd(5, seed=11)
    matvec = lambda v: M @ v
    assert jnp.allclose(cv.dense_from_matvec(matvec, 5), M, rtol=1e-9, atol=1e-10)


def test_lanczos_recovers_extremal_eigenvalues():
    M = _spd(8, seed=12)
    matvec = lambda v: M @ v
    ritz = cv.lanczos_eigvalsh(matvec, 8, jax.random.PRNGKey(13), n_iter=8)
    true = jnp.linalg.eigvalsh(M)
    assert jnp.allclose(ritz[-1], true[-1], rtol=1e-6, atol=1e-6)
    assert jnp.allclose(ritz[0], true[0], rtol=1e-6, atol=1e-6)


def test_laplace_posterior_cov():
    H = _spd(4, seed=14)
    p = 2.5
    cov = cv.laplace_posterior_cov(H, p)
    assert jnp.allclose(cov, jnp.linalg.inv(H + p * jnp.eye(4)), rtol=1e-8, atol=1e-9)
    # dense prior precision
    P = _spd(4, seed=15)
    assert jnp.allclose(cv.laplace_posterior_cov(H, P), jnp.linalg.inv(H + P),
                        rtol=1e-8, atol=1e-9)


def test_flat_operator_on_pytree_objective():
    A = _spd(3, seed=16)
    c = 4.0
    f = lambda p: 0.5 * p["a"] @ A @ p["a"] + 0.5 * c * jnp.sum(p["b"] ** 2)
    x = {"a": jnp.ones(3), "b": jnp.ones(2)}
    op = lambda v: cv.hvp(f, x, v)
    matvec, dim, unravel = cv.flat_operator(op, x)
    assert dim == 5
    dense = cv.dense_from_matvec(matvec, dim)
    expected = jnp.zeros((5, 5))
    expected = expected.at[:3, :3].set(A).at[3:, 3:].set(c * jnp.eye(2))
    assert jnp.allclose(dense, expected, rtol=1e-8, atol=1e-9)
