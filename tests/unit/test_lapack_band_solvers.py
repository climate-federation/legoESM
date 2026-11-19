"""LAPACK-transcribing solvers: ``thomas_solve(..., "lapack_dgtsv")`` and
``banded.lapack_band_solve`` (reference ``dgtsv`` / ``dgbsv``).

The references below are scalar Python transcriptions of reference LAPACK
3.x ``dgtsv`` and ``dgbtf2``+``dgbtrs`` (with reference BLAS ``idamax``,
``dswap``, ``dscal``, ``dger``, ``dtbsv``), the routines CLUBB links from its
bundled ``src/Lapack``.  Python floats are IEEE doubles with no fused
multiply-add, as gfortran emits them without ``-march``.

Bit-for-bit equality holds op by op (``jax.disable_jit``).  Compiled, XLA's
CPU code generator contracts ``a - b*c`` into fused multiply-adds on an FMA
machine (measured: bitwise again under ``--xla_cpu_max_isa=AVX``), so the
compiled solve is held to the campaign bar, ``max|Δ|/max|ref| <= 1e-12``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm.timestepping.banded import lapack_band_solve  # noqa: E402
from legoesm.timestepping.tridiagonal import thomas_solve  # noqa: E402


def _ref_dgtsv(dl, d, du, b):
    dl, d, du, b = (list(map(float, v)) for v in (dl, d, du, b))
    n = len(d)
    for i in range(n - 1):
        if abs(d[i]) >= abs(dl[i]):
            fact = dl[i] / d[i]
            d[i + 1] = d[i + 1] - fact * du[i]
            b[i + 1] = b[i + 1] - fact * b[i]
            if i < n - 2:
                dl[i] = 0.0
        else:
            fact = d[i] / dl[i]
            d[i] = dl[i]
            temp = d[i + 1]
            d[i + 1] = du[i] - fact * temp
            if i < n - 2:
                dl[i] = du[i + 1]
                du[i + 1] = -fact * dl[i]
            du[i] = temp
            temp = b[i]
            b[i] = b[i + 1]
            b[i + 1] = temp - fact * b[i + 1]
    b[n - 1] = b[n - 1] / d[n - 1]
    if n > 1:
        b[n - 2] = (b[n - 2] - du[n - 2] * b[n - 1]) / d[n - 2]
    for i in range(n - 3, -1, -1):
        b[i] = (b[i] - du[i] * b[i + 1] - dl[i] * b[i + 2]) / d[i]
    return np.array(b)


def _ref_dgbsv(A, b, kl, ku):
    """dgbtf2 + dgbtrs on a dense copy, touching only the band entries LAPACK touches."""
    n = len(b)
    a = [list(map(float, r)) for r in A]
    b = list(map(float, b))
    ipiv = [0] * n
    ju = 0
    for j in range(n):
        km = min(kl, n - 1 - j)
        col = [abs(a[j + r][j]) for r in range(km + 1)]
        jp = col.index(max(col))
        ipiv[j] = j + jp
        ju = max(ju, min(j + ku + jp, n - 1))
        if jp != 0:
            for c in range(j, ju + 1):
                a[j][c], a[j + jp][c] = a[j + jp][c], a[j][c]
        if km > 0:
            r = 1.0 / a[j][j]
            for q in range(1, km + 1):
                a[j + q][j] = r * a[j + q][j]
            for c in range(j + 1, ju + 1):
                if a[j][c] != 0.0:
                    temp = -1.0 * a[j][c]
                    for q in range(1, km + 1):
                        a[j + q][c] = a[j + q][c] + a[j + q][j] * temp
    for j in range(n - 1):
        lm = min(kl, n - 1 - j)
        p = ipiv[j]
        b[p], b[j] = b[j], b[p]
        if b[j] != 0.0:
            temp = -1.0 * b[j]
            for q in range(1, lm + 1):
                b[j + q] = b[j + q] + a[j + q][j] * temp
    for j in range(n - 1, -1, -1):
        if b[j] != 0.0:
            b[j] = b[j] / a[j][j]
            temp = b[j]
            for i in range(j - 1, max(0, j - kl - ku) - 1, -1):
                b[i] = b[i] - temp * a[i][j]
    return np.array(b)


_BAR = 1.0e-12


def _check(run, ref):
    with jax.disable_jit():
        np.testing.assert_array_equal(np.asarray(run()), ref)
    x = np.asarray(run())
    assert np.max(np.abs(x - ref)) <= _BAR * np.max(np.abs(ref))


def _bands(rng, ng, n, kl, ku, dominant):
    bands = rng.standard_normal((kl + ku + 1, ng, n))
    if dominant:
        bands[ku] = 1.0 + np.abs(bands).sum(axis=0)
    return bands


def _dense(bands_col, kl, ku):
    n = bands_col.shape[-1]
    A = np.zeros((n, n))
    for b in range(kl + ku + 1):
        s = ku - b
        for k in range(n):
            if 0 <= k + s < n:
                A[k, k + s] = bands_col[b, k]
    return A


@pytest.mark.parametrize("dominant", [True, False])
@pytest.mark.parametrize("n", [1, 2, 3, 9])
def test_dgtsv_mode_is_reference_lapack_bitwise(n, dominant):
    rng = np.random.default_rng(n + 10 * dominant)
    sub, mid, sup = _bands(rng, 3, n, 1, 1, dominant)[::-1]
    sub[:, 0] = 0.0
    sup[:, -1] = 0.0
    rhs = rng.standard_normal((3, n))
    ref = np.stack([_ref_dgtsv(sub[c, 1:], mid[c], sup[c, :-1], rhs[c]) for c in range(3)])
    _check(lambda: thomas_solve(jnp.asarray(sub), jnp.asarray(mid), jnp.asarray(sup),
                                jnp.asarray(rhs), "lapack_dgtsv"), ref)


def test_dgtsv_mode_pivots_where_normalised_thomas_cannot():
    """A zero leading diagonal: dgtsv swaps rows, the unpivoted sweep cannot."""
    sub = jnp.array([[0.0, 2.0, 1.0]])
    mid = jnp.array([[0.0, 1.0, 3.0]])
    sup = jnp.array([[1.0, 1.0, 0.0]])
    rhs = jnp.array([[1.0, 2.0, 3.0]])
    x = np.asarray(thomas_solve(sub, mid, sup, rhs, "lapack_dgtsv"))
    A = _dense(np.stack([np.asarray(sup), np.asarray(mid), np.asarray(sub)])[:, 0], 1, 1)
    np.testing.assert_allclose(A @ x[0], np.asarray(rhs[0]), rtol=0, atol=1e-14)


@pytest.mark.parametrize("dominant", [True, False])
@pytest.mark.parametrize("kl,ku,n", [(2, 2, 11), (2, 2, 3), (1, 2, 7), (2, 1, 6)])
def test_band_solve_is_reference_lapack_bitwise(kl, ku, n, dominant):
    rng = np.random.default_rng(100 * kl + 10 * ku + n + dominant)
    bands = _bands(rng, 3, n, kl, ku, dominant)
    rhs = rng.standard_normal((3, n))
    ref = np.stack([_ref_dgbsv(_dense(bands[:, c], kl, ku), rhs[c], kl, ku) for c in range(3)])
    _check(lambda: lapack_band_solve(jnp.asarray(bands), jnp.asarray(rhs), kl, ku), ref)
    for c in range(3):
        A = _dense(bands[:, c], kl, ku)
        np.testing.assert_allclose(A @ ref[c], rhs[c], rtol=0, atol=1e-10 * np.abs(A).sum())


def test_band_solve_pivots_on_a_zero_leading_diagonal():
    """A[0, 0] = 0: elimination without a row swap divides by zero."""
    rng = np.random.default_rng(5)
    bands = _bands(rng, 1, 8, 2, 2, True)
    bands[2, 0, 0] = 0.0
    rhs = rng.standard_normal((1, 8))
    x = np.asarray(lapack_band_solve(jnp.asarray(bands), jnp.asarray(rhs), 2, 2))
    A = _dense(bands[:, 0], 2, 2)
    assert np.all(np.isfinite(x))
    np.testing.assert_allclose(A @ x[0], rhs[0], rtol=0, atol=1e-12 * np.abs(A).sum())


@pytest.mark.parametrize("n", [1, 2])
def test_band_solve_ignores_bands_outside_a_small_matrix(n):
    rng = np.random.default_rng(n)
    bands = _bands(rng, 1, n, 2, 2, True)
    rhs = rng.standard_normal((1, n))
    x = np.asarray(lapack_band_solve(jnp.asarray(bands), jnp.asarray(rhs), 2, 2))
    np.testing.assert_allclose(_dense(bands[:, 0], 2, 2) @ x[0], rhs[0], rtol=1e-14)


def test_band_solve_gradient_is_finite_at_a_tiny_pivot():
    """diag(1, 1, 1e-170), rhs (1, 1, 0): x3 = 0, so d(sum x)/d diag3 = 0;
    raw autodiff of the elimination forms 1/pivot^2 and returns NaN."""
    bands = jnp.zeros((5, 1, 3)).at[2].set(jnp.array([[1.0, 1.0, 1e-170]]))
    rhs = jnp.array([[1.0, 1.0, 0.0]])
    g = jax.grad(lambda bd: jnp.sum(lapack_band_solve(bd, rhs, 2, 2)))(bands)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(g[2, 0, 2]) == 0.0


def test_band_solve_jit_and_grad():
    rng = np.random.default_rng(7)
    bands = jnp.asarray(_bands(rng, 2, 9, 2, 2, False))
    rhs = jnp.asarray(rng.standard_normal((2, 9)))
    eager = lapack_band_solve(bands, rhs, 2, 2)
    jitted = jax.jit(lapack_band_solve, static_argnums=(2, 3))(bands, rhs, 2, 2)
    assert np.max(np.abs(np.asarray(jitted - eager))) <= _BAR * np.max(np.abs(np.asarray(eager)))

    def loss(bd, r):
        return jnp.sum(lapack_band_solve(bd, r, 2, 2) ** 2)

    check_grads(loss, (bands, rhs), order=1, modes=["rev"], eps=1e-5, atol=1e-6, rtol=1e-6)


def test_dgtsv_mode_grad_matches_finite_difference():
    rng = np.random.default_rng(3)
    sub, mid, sup = (jnp.asarray(v) for v in _bands(rng, 2, 6, 1, 1, False)[::-1])
    rhs = jnp.asarray(rng.standard_normal((2, 6)))

    sub = sub.at[:, 0].set(0.0)
    sup = sup.at[:, -1].set(0.0)

    def loss(a, b, c, d):
        return jnp.sum(thomas_solve(a, b, c, d, "lapack_dgtsv") ** 2)

    check_grads(loss, (sub, mid, sup, rhs), order=1, modes=["rev"], eps=1e-5,
                atol=1e-6, rtol=1e-6)


def test_band_solve_rhs_cotangent_keeps_the_rhs_dtype():
    rng = np.random.default_rng(9)
    bands = jnp.asarray(_bands(rng, 1, 6, 2, 2, True))
    rhs = jnp.asarray(rng.standard_normal((1, 6)), jnp.float32)
    g = jax.grad(lambda r: jnp.sum(lapack_band_solve(bands, r, 2, 2)))(rhs)
    assert g.dtype == jnp.float32


def test_band_solve_overflowing_multiplier_meets_no_zero():
    """dgbtf2 touches columns j..JU only and dger skips y == 0: an infinite
    multiplier must not turn a structural zero into NaN (codex counterexample)."""
    h = 1.0e308
    A = np.array([[1, -h, 0, 0], [1, h, 0, 0], [1, h, 1, 0], [0, 0, 0, 1.0]])
    bd = np.zeros((5, 1, 4))
    for b in range(5):
        for i in range(4):
            c = i + 2 - b
            if 0 <= c < 4:
                bd[b, 0, i] = A[i, c]
    r = np.array([[0.0, 0.0, 0.0, 1.0]])
    ref = _ref_dgbsv(A, r[0], 2, 2)
    _check(lambda: lapack_band_solve(jnp.asarray(bd), jnp.asarray(r), 2, 2), ref[None])


def test_band_solve_second_derivative_at_zero_rhs():
    """A = 2I, rhs = 0, loss = sum(x^2): Hessian in rhs is 2*A^-T A^-1 = 0.5 I
    (codex: zero-skips inside the adjoint solve halved nothing / broke it)."""
    n = 4
    bands = jnp.zeros((5, 1, n)).at[2].set(2.0)
    rhs = jnp.zeros((1, n))
    loss = lambda r: jnp.sum(lapack_band_solve(bands, r, 2, 2) ** 2)  # noqa: E731
    for H in (jax.hessian(loss)(rhs), jax.jacrev(jax.jacrev(loss))(rhs)):
        np.testing.assert_allclose(np.asarray(H).reshape(n, n), 0.5 * np.eye(n), rtol=1e-14)


def test_band_solve_second_derivative_finite_after_overflow():
    """Overflowing elimination (codex r6): d2/dt2 of x3^2 with rhs (0,0,0,t) is 2."""
    h = 1.0e308
    A = np.array([[1, -h, 0, 0], [1, h, 0, 0], [1, h, 1, 0], [0, 0, 0, 1.0]])
    bd = np.zeros((5, 1, 4))
    for b in range(5):
        for i in range(4):
            c = i + 2 - b
            if 0 <= c < 4:
                bd[b, 0, i] = A[i, c]
    bd = jnp.asarray(bd)

    def loss(t):
        r = jnp.zeros((1, 4)).at[0, 3].set(t)
        return lapack_band_solve(bd, r, 2, 2)[0, 3] ** 2

    assert float(jax.grad(jax.grad(loss))(1.0)) == 2.0
