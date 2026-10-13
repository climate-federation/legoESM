"""Hand-derived block-transposed adjoint of the CLM-ml coupled T/q tridiagonal
solve (``MLMathToolsMod.tridiag_2eq`` custom VJP; review F7).

Pins (a) the forward against a dense ``jnp.linalg.solve`` of the same 2n x 2n
system, (b) all ten band cotangents against plain AD through that dense solve,
and (c) ``jax.test_util.check_grads`` (reverse mode, order 2 — the custom VJP
has no forward-mode rule by construction).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLMathToolsMod import (  # noqa: E402
    tridiag_2eq,
)

N = 4
BANDS = ("a1", "b11", "b12", "c1", "d1", "a2", "b21", "b22", "c2", "d2")


def _coeffs(seed=0):
    rng = np.random.default_rng(seed)
    off = lambda: jnp.asarray(rng.uniform(-0.6, 0.6, N))        # noqa: E731
    return dict(
        a1=off(), b11=jnp.asarray(rng.uniform(3.0, 4.0, N)), b12=off(), c1=off(),
        d1=jnp.asarray(rng.normal(size=N)),
        a2=off(), b21=off(), b22=jnp.asarray(rng.uniform(3.0, 4.0, N)), c2=off(),
        d2=jnp.asarray(rng.normal(size=N)))


def _dense(a1, b11, b12, c1, d1, a2, b21, b22, c2, d2):
    """Same system, unknowns interleaved [T0, q0, T1, q1, ...]."""
    A = jnp.zeros((2 * N, 2 * N))
    for i in range(N):
        t, q = 2 * i, 2 * i + 1
        A = A.at[t, t].set(b11[i]).at[t, q].set(b12[i])
        A = A.at[q, t].set(b21[i]).at[q, q].set(b22[i])
        if i > 0:
            A = A.at[t, t - 2].set(a1[i]).at[q, q - 2].set(a2[i])
        if i < N - 1:
            A = A.at[t, t + 2].set(c1[i]).at[q, q + 2].set(c2[i])
    rhs = jnp.stack([d1, d2], axis=1).reshape(-1)
    x = jnp.linalg.solve(A, rhs).reshape(N, 2)
    return x[:, 0], x[:, 1]


def _solve(*bands):
    t, q = tridiag_2eq(*bands, N)
    return jnp.stack(t), jnp.stack(q)


def test_forward_matches_dense_solve():
    c = _coeffs()
    t, q = _solve(*(c[k] for k in BANDS))
    t_ref, q_ref = _dense(*(c[k] for k in BANDS))
    np.testing.assert_allclose(t, t_ref, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(q, q_ref, rtol=1e-12, atol=1e-13)


def test_all_ten_band_cotangents_match_dense_ad():
    c = _coeffs(1)
    rng = np.random.default_rng(7)
    wt, wq = jnp.asarray(rng.normal(size=N)), jnp.asarray(rng.normal(size=N))

    def loss(fn):
        def f(*bands):
            t, q = fn(*bands)
            return jnp.sum(wt * t) + jnp.sum(wq * q)
        return f

    args = tuple(c[k] for k in BANDS)
    g = jax.grad(loss(_solve), argnums=tuple(range(10)))(*args)
    g_ref = jax.grad(loss(_dense), argnums=tuple(range(10)))(*args)
    for name, gi, ri in zip(BANDS, g, g_ref):
        np.testing.assert_allclose(gi, ri, rtol=1e-10, atol=1e-12,
                                   err_msg=f"cotangent of {name}")
    # the coupling bands are live (a transpose slip would show here first)
    assert float(jnp.abs(g[2]).max()) > 0.0 and float(jnp.abs(g[6]).max()) > 0.0


def test_check_grads_reverse_order_2():
    c = _coeffs(2)
    check_grads(_solve, tuple(c[k] for k in BANDS), order=2, modes=["rev"])
