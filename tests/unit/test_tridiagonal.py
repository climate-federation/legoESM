"""Direct unit tests for legoesm.timestepping.tridiagonal.thomas_solve.

tridiagonal.py is on the CLAUDE.md untested-but-live list; this gives the Thomas
solver a direct test: correctness vs a dense solve, the MIXED-DTYPE path (f32 RHS
/ f64 coefficients -- the implicit-vertical-mixing case under x64) that previously
emitted an implicit f64->f32 scatter FutureWarning, batched columns, and
jax.grad-safety.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.timestepping.tridiagonal import thomas_solve


def _dense_solve(a, b, c, d):
    """Reference dense tridiagonal solve (NumPy)."""
    n = b.shape[-1]
    M = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        M[k, k] = b[k]
        if k > 0:
            M[k, k - 1] = a[k]
        if k < n - 1:
            M[k, k + 1] = c[k]
    return np.linalg.solve(M, d)


def _random_dd_system(n, rng, dtype_a=np.float64, dtype_d=np.float64):
    """A diagonally dominant (stable) tridiagonal system."""
    a = np.r_[0.0, rng.uniform(-1.0, 0.0, n - 1)].astype(dtype_a)
    c = np.r_[rng.uniform(-1.0, 0.0, n - 1), 0.0].astype(dtype_a)
    b = (3.0 + rng.uniform(0.0, 1.0, n)).astype(dtype_a)
    d = rng.uniform(-1.0, 1.0, n).astype(dtype_d)
    return a, b, c, d


def test_thomas_solve_matches_dense():
    rng = np.random.default_rng(0)
    a, b, c, d = _random_dd_system(8, rng)
    x = np.asarray(thomas_solve(jnp.asarray(a), jnp.asarray(b),
                                jnp.asarray(c), jnp.asarray(d)))
    np.testing.assert_allclose(x, _dense_solve(a, b, c, d), rtol=1e-10, atol=1e-12)


def test_thomas_solve_mixed_dtype_no_downcast_warning():
    """f32 RHS + f64 coefficients (implicit vertical mixing under x64): must NOT
    emit the f64->f32 scatter FutureWarning, must stay correct, and must return
    the solution in the RHS dtype (so callers' f32 state stores are dtype-stable)."""
    rng = np.random.default_rng(1)
    a, b, c, d = _random_dd_system(6, rng, dtype_a=np.float64, dtype_d=np.float32)
    aj, bj, cj = jnp.asarray(a), jnp.asarray(b), jnp.asarray(c)
    dj = jnp.asarray(d)  # float32
    assert dj.dtype == jnp.float32 and bj.dtype == jnp.float64
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        x = thomas_solve(aj, bj, cj, dj)
    assert x.dtype == jnp.float32, "solution should come back in the RHS dtype"
    np.testing.assert_allclose(
        np.asarray(x, np.float64),
        _dense_solve(a.astype(np.float64), b.astype(np.float64),
                     c.astype(np.float64), d.astype(np.float64)),
        rtol=1e-5, atol=1e-6)


def test_thomas_solve_batched_columns():
    """Leading dims are batched (column-parallel)."""
    rng = np.random.default_rng(2)
    B, n = 4, 6
    a = np.concatenate([np.zeros((B, 1)), rng.uniform(-1, 0, (B, n - 1))], axis=1)
    c = np.concatenate([rng.uniform(-1, 0, (B, n - 1)), np.zeros((B, 1))], axis=1)
    b = 3.0 + rng.uniform(0, 1, (B, n))
    d = rng.uniform(-1, 1, (B, n))
    x = np.asarray(thomas_solve(jnp.asarray(a), jnp.asarray(b),
                                jnp.asarray(c), jnp.asarray(d)))
    for k in range(B):
        np.testing.assert_allclose(
            x[k], _dense_solve(a[k], b[k], c[k], d[k]), rtol=1e-9, atol=1e-11)


def test_thomas_solve_grad_safe():
    """jax.grad flows through the solver and matches finite differences."""
    n = 5
    a = jnp.asarray(np.r_[0.0, -0.2 * np.ones(n - 1)])
    c = jnp.asarray(np.r_[-0.2 * np.ones(n - 1), 0.0])
    b = jnp.asarray(2.0 * np.ones(n))
    d = jnp.asarray(np.linspace(-1.0, 1.0, n))

    def loss(dd):
        return jnp.sum(thomas_solve(a, b, c, dd) ** 2)

    g = np.asarray(jax.grad(loss)(d))
    assert np.all(np.isfinite(g))
    eps = 1e-6
    fd = np.array([
        float((loss(d.at[i].add(eps)) - loss(d.at[i].add(-eps))) / (2 * eps))
        for i in range(n)
    ])
    np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)


def _thomas_loop_np(a, b, c, d):
    """Reference sequential Thomas sweep in NumPy (same recurrence)."""
    n = b.shape[-1]
    cs = np.zeros(np.broadcast_shapes(a.shape, b.shape, c.shape, d.shape))
    ds = np.zeros_like(cs)
    cs[..., 0] = c[..., 0] / b[..., 0]
    ds[..., 0] = d[..., 0] / b[..., 0]
    for k in range(1, n):
        den = b[..., k] - a[..., k] * cs[..., k - 1]
        cs[..., k] = c[..., k] / den
        ds[..., k] = (d[..., k] - a[..., k] * ds[..., k - 1]) / den
    x = np.zeros_like(ds)
    x[..., -1] = ds[..., -1]
    for k in range(n - 2, -1, -1):
        x[..., k] = ds[..., k] - cs[..., k] * x[..., k + 1]
    return x


def test_thomas_solve_short_systems_and_broadcast_shapes():
    """n in {1, 2}, and a/b/c/d of different broadcast-compatible shapes
    (one operator, many right-hand sides; one RHS, many diagonals): the
    output is the full broadcast shape and matches the sequential sweep."""
    rng = np.random.default_rng(3)
    for n in (1, 2, 5):
        a, b, c, d = _random_dd_system(n, rng)
        x = np.asarray(thomas_solve(*map(jnp.asarray, (a, b, c, d))))
        np.testing.assert_allclose(x, _thomas_loop_np(a, b, c, d), rtol=1e-13, atol=1e-15)
    a, b, c, _ = _random_dd_system(6, rng)
    D = rng.uniform(-1, 1, (4, 6))
    x = np.asarray(thomas_solve(*map(jnp.asarray, (a, b, c, D))))
    assert x.shape == (4, 6)
    np.testing.assert_allclose(x, _thomas_loop_np(a, b, c, D), rtol=1e-13, atol=1e-15)
    B = 3.0 + rng.uniform(0, 1, (3, 6))
    d = rng.uniform(-1, 1, 6)
    x = np.asarray(thomas_solve(*map(jnp.asarray, (a, B, c, d))))
    assert x.shape == (3, 6)
    np.testing.assert_allclose(x, _thomas_loop_np(a, B, c, d), rtol=1e-13, atol=1e-15)


def test_thomas_solve_batched_matches_sequential_sweep_bitwise_close():
    """A (cells, levels) batch -- the ocean's implicit-mixing shape -- against
    the sequential NumPy sweep, float64."""
    rng = np.random.default_rng(4)
    shp = (257, 40)
    a = rng.uniform(-1, 0, shp); a[:, 0] = 0
    c = rng.uniform(-1, 0, shp); c[:, -1] = 0
    b = 2.5 + rng.uniform(0, 1, shp); d = rng.normal(size=shp)
    x = np.asarray(thomas_solve(*map(jnp.asarray, (a, b, c, d))))
    np.testing.assert_allclose(x, _thomas_loop_np(a, b, c, d), rtol=1e-13, atol=1e-15)
