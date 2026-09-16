"""Stable adjoint of the (non-batched) ``thomas_solve``.

The reverse mode of the raw Thomas recursion divides by ``denom**2`` per pivot, so a
pivot clamped to ``_TINY`` (a near-singular system — e.g. an implicit soil-thermal
matrix whose heat capacity collapses for a stiff-clay, near-saturated land cell) gives
a FINITE forward but a NaN backward.  ``thomas_solve`` now carries a ``custom_vjp`` that
solves the transposed tridiagonal system instead (the analytic VJP of a linear solve).

Pins: forward bit-identical to the raw sweep, gradient equal to finite-difference AND to
the raw element-wise autodiff on a well-conditioned system (no regression), and a FINITE
gradient on a near-singular pivot (the bug this fixes).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.timestepping.tridiagonal import (  # noqa: E402
    thomas_solve,
    _thomas_solve_impl,
)


def _random_dd_system(shape, seed=0):
    """Strongly diagonally-dominant tridiagonal (well-conditioned)."""
    rng = np.random.default_rng(seed)
    a = jnp.asarray(rng.normal(size=shape) * 0.3).at[..., 0].set(0.0)
    c = jnp.asarray(rng.normal(size=shape) * 0.3).at[..., -1].set(0.0)
    b = jnp.asarray(2.0 + rng.uniform(size=shape))
    d = jnp.asarray(rng.normal(size=shape))
    return a, b, c, d


def _loss(fn):
    return lambda abcd: jnp.sum(fn(*abcd) ** 2)


def test_forward_bit_identical_to_raw_sweep():
    for shape in [(6,), (3, 6), (2, 4, 5)]:
        a, b, c, d = _random_dd_system(shape)
        assert jnp.array_equal(thomas_solve(a, b, c, d), _thomas_solve_impl(a, b, c, d))


def test_solve_residual():
    a, b, c, d = _random_dd_system((3, 7))
    x = thomas_solve(a, b, c, d)
    Ax = b * x
    Ax = Ax.at[..., 1:].add(a[..., 1:] * x[..., :-1])
    Ax = Ax.at[..., :-1].add(c[..., :-1] * x[..., 1:])
    assert float(jnp.max(jnp.abs(Ax - d))) < 1e-12


def test_grad_matches_finite_difference():
    n = 6
    a, b, c, d = _random_dd_system((3, n))
    g = jax.grad(_loss(thomas_solve))((a, b, c, d))
    eps = 1e-6
    for gi, arr, which in [(g[1], b, 1), (g[3], d, 3), (g[0], a, 0), (g[2], c, 2)]:
        flat = arr.flatten()
        for fi in (2, 9, 14):
            base = [a, b, c, d]
            pp = list(base); pp[which] = flat.at[fi].add(eps).reshape(arr.shape)
            pm = list(base); pm[which] = flat.at[fi].add(-eps).reshape(arr.shape)
            fd = float((_loss(thomas_solve)(tuple(pp)) - _loss(thomas_solve)(tuple(pm))) / (2 * eps))
            assert abs(float(gi.flatten()[fi]) - fd) < 1e-5


def test_grad_no_regression_vs_raw_autodiff():
    """On a well-conditioned system the analytic adjoint must equal the raw
    element-wise autodiff to machine precision (the existing behaviour)."""
    a, b, c, d = _random_dd_system((4, 8))
    g_new = jax.grad(_loss(thomas_solve))((a, b, c, d))
    g_raw = jax.grad(_loss(_thomas_solve_impl))((a, b, c, d))
    for i in range(4):
        assert float(jnp.max(jnp.abs(g_new[i] - g_raw[i]))) < 1e-10


def test_stiff_pivot_adjoint_finite_and_tracks_true_gradient():
    """A STIFF pivot (near-cancellation, denom ~ 1e-6) is the regime that — compounded
    over the seasonal land scan with large incoming cotangents — overflowed the raw
    1/denom**2 reverse mode to NaN.  Here the analytic transposed-solve adjoint stays
    finite and equals BOTH the raw element-wise autodiff (so no production gradient
    moved) and finite-difference (so it is the true gradient), into the stiff regime."""
    a = jnp.array([0.0, 2.0, 0.3])
    b = jnp.array([2.0, 2.0 + 1e-6, 3.0])   # row-1 pivot ~ 1e-6 after elimination
    c = jnp.array([2.0, 0.3, 0.0])
    d = jnp.array([1.0, 1.0, 1.0])
    x = thomas_solve(a, b, c, d)
    assert bool(jnp.all(jnp.isfinite(x)))
    Lb = lambda fn, B: jnp.sum(fn(a, B, c, d) ** 2)
    g_new = jax.grad(lambda B: Lb(thomas_solve, B))(b)
    g_raw = jax.grad(lambda B: Lb(_thomas_solve_impl, B))(b)
    assert bool(jnp.all(jnp.isfinite(g_new)))
    assert float(jnp.max(jnp.abs(g_new - g_raw))) < 1e-6          # no regression
    eps = 1e-3
    fd = jnp.array([(Lb(thomas_solve, b.at[i].add(eps))
                     - Lb(thomas_solve, b.at[i].add(-eps))) / (2 * eps) for i in range(3)])
    assert float(jnp.max(jnp.abs((g_new - fd) / (jnp.abs(fd) + 1.0)))) < 1e-2  # ~ true grad


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")


# --- #1736: which AD modes this solver supports, pinned rather than prose ----
# Every bullet below was unpinned before this: grepping tests/, scripts/ and
# docs/ found ZERO assertions on the solver's forward-mode behaviour, so the
# docstring's claims could drift (and had -- it asserted that no path
# forward-differentiates the solver while three operators in da/curvature.py
# did exactly that, and raised).


def _banded_objective(x):
    """Scalar objective whose tridiagonal BANDS depend on the control.

    The control enters the DIAGONAL, so the solve is nonlinear in ``x`` and the
    Hessian is not the trivial one a linear-in-RHS objective would give.
    """
    n = 6
    a = jnp.concatenate([jnp.zeros(1), -jnp.ones(n - 1)])
    c = jnp.concatenate([-jnp.ones(n - 1), jnp.zeros(1)])
    b = 4.0 + x
    d = jnp.ones(n) * 2.0
    return jnp.sum(thomas_solve(a, b, c, d) ** 2)


def _dense_objective(x):
    """The same objective with the band assembled and solved densely."""
    n = 6
    A = (jnp.diag(4.0 + x)
         + jnp.diag(-jnp.ones(n - 1), -1)
         + jnp.diag(-jnp.ones(n - 1), 1))
    return jnp.sum(jnp.linalg.solve(A, jnp.ones(n) * 2.0) ** 2)


def test_forward_over_reverse_hvp_matches_a_dense_hessian():
    """FAILS on the pre-#1736 solver with TypeError; passes after.

    This is the cheap stand-in for the land-column integration test, which
    costs ~5 minutes per evaluation.  The dense reference is an independent
    implementation, not a rearrangement of the same sweep.
    """
    x0 = jnp.linspace(0.5, 1.5, 6)
    v = jnp.ones_like(x0) / jnp.sqrt(6.0)
    got = jax.jvp(jax.grad(_banded_objective), (x0,), (v,))[1]
    want = jax.jvp(jax.grad(_dense_objective), (x0,), (v,))[1]
    assert jnp.all(jnp.isfinite(got))
    assert jnp.allclose(got, want, rtol=1e-9, atol=1e-11), (
        f"forward-over-reverse HVP differs from the dense Hessian by "
        f"{float(jnp.max(jnp.abs(got - want))):.3e}")


def test_first_order_gradient_is_unchanged_by_the_1736_fix():
    """The #1736 change touches only the bwd rule's internal lambda solve, so
    the gradient itself must be bit-identical to the dense reference's."""
    x0 = jnp.linspace(0.5, 1.5, 6)
    g_band = jax.grad(_banded_objective)(x0)
    g_dense = jax.grad(_dense_objective)(x0)
    assert jnp.allclose(g_band, g_dense, rtol=1e-10, atol=1e-12)


def test_plain_forward_mode_through_the_solver_still_raises():
    """The custom_vjp defines no JVP, and that is still true and still by
    design -- #1736 fixed forward-OVER-REVERSE, not plain forward mode.

    Pins the first docstring bullet so a future 'fix' cannot quietly make
    jacfwd work by deleting the custom adjoint and its clamp protection.
    """
    n = 4
    a = jnp.concatenate([jnp.zeros(1), -jnp.ones(n - 1)])
    c = jnp.concatenate([-jnp.ones(n - 1), jnp.zeros(1)])
    b, d = 4.0 * jnp.ones(n), jnp.ones(n)
    with pytest.raises(TypeError, match="forward-mode autodiff"):
        jax.jvp(lambda bb: thomas_solve(a, bb, c, d), (b,), (jnp.ones(n),))


def test_second_order_at_a_clamped_pivot_is_the_price_of_1736():
    """Records the trade #1736 made, so it cannot be discovered by surprise.

    Higher-order reverse no longer recurses through the custom rule, so a
    second derivative taken where a pivot is CLAMPED differentiates the raw
    ``1/denom**2``.  In float64 -- what production runs -- this stays finite;
    the test asserts that, and asserts the first-order adjoint is finite there
    too, which is the property the custom_vjp exists to provide and which
    #1736 did NOT trade away.
    """
    n = 4

    def near_singular(x):
        a = jnp.concatenate([jnp.zeros(1), -jnp.ones(n - 1)])
        c = jnp.concatenate([-jnp.ones(n - 1), jnp.zeros(1)])
        b = jnp.full(n, 1e-30) * x          # pivots driven towards the clamp
        return jnp.sum(thomas_solve(a, b, c, jnp.ones(n)) ** 2)

    x0 = jnp.ones(())
    g = jax.grad(near_singular)(x0)
    assert jnp.isfinite(g), "first-order adjoint must stay clamp-protected"
    hv = jax.jvp(jax.grad(near_singular), (x0,), (jnp.ones(()),))[1]
    assert jnp.isfinite(hv), (
        "float64 second order must stay finite at a clamped pivot; if this "
        "goes NaN the #1736 trade is worse than measured and the clamp floor "
        "needs raising")
