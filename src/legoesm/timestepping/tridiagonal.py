"""Thomas algorithm for tridiagonal systems, JIT-friendly.

Solves the tridiagonal system:

    a[k] * x[k-1] + b[k] * x[k] + c[k] * x[k+1] = d[k]

for k = 0, ..., n-1, where a[0] = 0 and c[n-1] = 0.

Uses jax.lax.scan for the forward/backward sweeps, enabling
efficient JIT compilation and automatic differentiation.

Used by the semi-implicit acoustic substeps in the compressible
Euler equations to solve for vertical velocity w implicitly.

References
----------
- Thomas, L. H. (1949). Elliptic problems in linear difference equations
  over a network. Watson Sci. Comput. Lab. Report, Columbia University.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_TINY = float(jnp.finfo(jnp.float32).tiny)  # Smallest normal float32 (~1.18e-38)


def thomas_solve(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: jax.Array,
) -> jax.Array:
    """Solve a tridiagonal system via the Thomas algorithm.

    Parameters
    ----------
    a : jax.Array, shape (..., n)
        Sub-diagonal coefficients. ``a[..., 0]`` is unused (set to 0).
    b : jax.Array, shape (..., n)
        Main diagonal coefficients (must be nonzero).
    c : jax.Array, shape (..., n)
        Super-diagonal coefficients. ``c[..., -1]`` is unused (set to 0).
    d : jax.Array, shape (..., n)
        Right-hand side vector.

    Returns
    -------
    x : jax.Array, shape (..., n)
        Solution vector.

    Notes
    -----
    The leading dimensions ``...`` are batched over (column-parallel).
    The last dimension is the tridiagonal system size.
    """
    n = b.shape[-1]

    # Forward sweep: eliminate sub-diagonal
    def forward_step(carry, k):
        c_prev, d_prev = carry  # Modified c and d from previous row
        ak = a[..., k]
        bk = b[..., k]
        ck = c[..., k]
        dk = d[..., k]

        # For k=0, a[0]=0 so m=0; c_star=c/b, d_star=d/b
        m = jnp.where(k > 0, ak / (bk - ak * c_prev), 0.0)
        c_star = ck / (bk - ak * c_prev + _TINY)
        d_star = (dk - ak * d_prev) / (bk - ak * c_prev + _TINY)

        return (c_star, d_star), (c_star, d_star)

    # Initialize: for k=0, c_star = c[0]/b[0], d_star = d[0]/b[0]
    c0_star = c[..., 0] / (b[..., 0] + _TINY)
    d0_star = d[..., 0] / (b[..., 0] + _TINY)

    # We'll do the sweep manually with lax.scan over k=1..n-1
    # But lax.scan needs fixed-size arrays. Instead, build vectorized.
    # Use the stable sequential approach with fori_loop.

    # Allocate modified arrays
    c_star = jnp.zeros_like(c)
    d_star = jnp.zeros_like(d)
    c_star = c_star.at[..., 0].set(c0_star)
    d_star = d_star.at[..., 0].set(d0_star)

    def forward_body(k, carry):
        c_star_c, d_star_c = carry
        ak = a[..., k]
        bk = b[..., k]
        ck = c[..., k]
        dk = d[..., k]

        c_prev = c_star_c[..., k - 1]
        d_prev = d_star_c[..., k - 1]

        denom = bk - ak * c_prev
        denom = jnp.where(jnp.abs(denom) < _TINY, _TINY, denom)

        c_star_k = ck / denom
        d_star_k = (dk - ak * d_prev) / denom

        c_star_c = c_star_c.at[..., k].set(c_star_k)
        d_star_c = d_star_c.at[..., k].set(d_star_k)

        return (c_star_c, d_star_c)

    c_star, d_star = jax.lax.fori_loop(
        1, n, forward_body, (c_star, d_star),
    )

    # Backward substitution: x[n-1] = d_star[n-1]
    x = jnp.zeros_like(d)
    x = x.at[..., -1].set(d_star[..., -1])

    def backward_body(k_rev, x_c):
        # k_rev counts 0, 1, ..., n-2; actual index k = n-2-k_rev
        k = n - 2 - k_rev
        x_c = x_c.at[..., k].set(
            d_star[..., k] - c_star[..., k] * x_c[..., k + 1]
        )
        return x_c

    x = jax.lax.fori_loop(0, n - 1, backward_body, x)

    return x


def thomas_solve_batched(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: jax.Array,
) -> jax.Array:
    """Batched Thomas solve over columns.

    Same as :func:`thomas_solve` but uses vmap over the first two
    dimensions for cubed-sphere fields.

    Parameters
    ----------
    a, b, c, d : jax.Array, shape (6, n, n, nlev)
        Tridiagonal system for each grid column.

    Returns
    -------
    x : jax.Array, shape (6, n, n, nlev)
    """
    # Flatten spatial dims, solve, reshape
    orig_shape = a.shape
    spatial_shape = orig_shape[:-1]
    n_sys = orig_shape[-1]

    # Reshape to (N_columns, n_sys) for batch processing
    n_cols = 1
    for s in spatial_shape:
        n_cols *= s

    a_flat = a.reshape(n_cols, n_sys)
    b_flat = b.reshape(n_cols, n_sys)
    c_flat = c.reshape(n_cols, n_sys)
    d_flat = d.reshape(n_cols, n_sys)

    # vmap thomas_solve over columns
    x_flat = jax.vmap(thomas_solve)(a_flat, b_flat, c_flat, d_flat)

    return x_flat.reshape(orig_shape)
