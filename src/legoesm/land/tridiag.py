"""Shared tridiagonal (Thomas) solver for soil column models.

Used by both the Richards equation solver and the soil thermal solver.
All operations are JAX-differentiable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def thomas_solve_batch(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    d: jnp.ndarray,
) -> jnp.ndarray:
    """Solve a batch of tridiagonal systems via the Thomas algorithm.

    Parameters
    ----------
    a, b, c, d : jnp.ndarray
        Sub-diagonal, main diagonal, super-diagonal, and RHS.
        All have shape ``(ncol, nlayers)``.  ``a[:, 0]`` and ``c[:, -1]``
        are ignored (boundary padding).

    Returns
    -------
    x : jnp.ndarray
        Solution, shape ``(ncol, nlayers)``.
    """
    # Promote all inputs to a common dtype so that lax.scan carry types
    # are consistent (avoids float32-init / float64-body mismatches).
    common = jnp.result_type(a, b, c, d)
    a, b, c, d = (x.astype(common) for x in (a, b, c, d))
    _tiny = jnp.finfo(common).tiny

    def solve_single(a_col, b_col, c_col, d_col):
        n = b_col.shape[0]

        def fwd(carry, k):
            c_p, d_p = carry
            denom = b_col[k] - a_col[k] * c_p
            denom = jnp.where(jnp.abs(denom) < _tiny,
                              jnp.sign(denom) * _tiny + _tiny, denom)
            c_new = c_col[k] / denom
            d_new = (d_col[k] - a_col[k] * d_p) / denom
            return (c_new, d_new), (c_new, d_new)

        denom0 = jnp.where(jnp.abs(b_col[0]) < _tiny, _tiny, b_col[0])
        init = (c_col[0] / denom0, d_col[0] / denom0)
        _, (c_primes, d_primes) = jax.lax.scan(fwd, init, jnp.arange(1, n))
        c_all = jnp.concatenate([jnp.array([init[0]]), c_primes])
        d_all = jnp.concatenate([jnp.array([init[1]]), d_primes])

        def bwd(x_next, k):
            x_k = d_all[k] - c_all[k] * x_next
            return x_k, x_k

        x_last = d_all[-1]
        _, x_rev = jax.lax.scan(bwd, x_last, jnp.arange(n - 2, -1, -1))
        return jnp.concatenate([jnp.flip(x_rev), jnp.array([x_last])])

    return jax.vmap(solve_single)(a, b, c, d)
