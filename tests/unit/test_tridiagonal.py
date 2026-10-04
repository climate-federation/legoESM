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

from legoesm.timestepping.tridiagonal import thomas_solve, thomas_solve_shared


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


def test_thomas_sweeps_emit_levels_instead_of_writing_into_a_buffer():
    """Each sweep scans over the level axis and stacks its outputs; a per-level
    ``x.at[..., k].set`` inside a loop lowers to a scatter that XLA's CPU
    backend turns into a whole-buffer rewrite every iteration (~30% of an
    MPAS dycore step before it was removed)."""
    import jax
    import jax.numpy as jnp

    a = b = c = d = jnp.ones((7, 32))
    jaxpr = str(jax.make_jaxpr(thomas_solve)(a, b + 3.0, c, d))
    assert "scan" in jaxpr  # the solve body is printed, so the check bites
    assert "scatter" not in jaxpr and "dynamic_update_slice" not in jaxpr
    # the one-matrix, several-RHS variant used by ocean vertical mixing
    jaxpr = str(jax.make_jaxpr(
        lambda a, b, c, d: thomas_solve_shared(a, b, c, (d, 2.0 * d)))(a, b + 3.0, c, d))
    assert "scan" in jaxpr
    assert "scatter" not in jaxpr and "dynamic_update_slice" not in jaxpr


def test_thomas_single_level_without_jit():
    """n == 1 is just d/b; it must also work eagerly (zero-length scans are
    rejected outside jit)."""
    import jax
    import jax.numpy as jnp
    b = jnp.array([[2.0], [4.0]]); d = jnp.array([[1.0], [3.0]])
    z = jnp.zeros_like(b)
    with jax.disable_jit():
        x = thomas_solve(z, b, z, d)
    np.testing.assert_allclose(np.asarray(x), np.asarray(d / b), rtol=1e-6)
    with jax.disable_jit():
        x1, x2 = thomas_solve_shared(z, b, z, (d, 2.0 * d))
    np.testing.assert_allclose(np.asarray(x1), np.asarray(d / b), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(x2), np.asarray(2.0 * d / b), rtol=1e-6)


def test_thomas_shared_matches_single_rhs_bitwise():
    rng = np.random.default_rng(0)
    for dtype in (np.float32, np.float64):
        a = jnp.asarray(rng.uniform(-1, 0, (5, 16)), dtype)
        c = jnp.asarray(rng.uniform(-1, 0, (5, 16)), dtype)
        b = jnp.asarray(3.0 + rng.uniform(0, 1, (5, 16)), dtype)
        d1 = jnp.asarray(rng.normal(size=(5, 16)), dtype)
        d2 = jnp.asarray(rng.normal(size=(5, 16)), dtype)
        for wrap in (lambda f: f, jax.jit):
            x1, x2 = wrap(lambda a, b, c, d1, d2: thomas_solve_shared(a, b, c, (d1, d2)))(a, b, c, d1, d2)
            single = wrap(thomas_solve)
            np.testing.assert_array_equal(np.asarray(x1), np.asarray(single(a, b, c, d1)))
            np.testing.assert_array_equal(np.asarray(x2), np.asarray(single(a, b, c, d2)))
            y1, y2 = wrap(lambda b, d1, d2: thomas_solve_shared(0 * b, b, 0 * b, (d1, d2)))(b[:, :1], d1[:, :1], d2[:, :1])
            np.testing.assert_array_equal(np.asarray(y1), np.asarray(single(0 * b[:, :1], b[:, :1], 0 * b[:, :1], d1[:, :1])))
