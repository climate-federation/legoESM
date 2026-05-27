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


def pcr_solve_batched(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: jax.Array,
) -> jax.Array:
    """Parallel cyclic reduction for batched tridiagonal systems.

    Pure-JAX. Unlike :func:`thomas_solve_batched` which calls cuSPARSE
    via a custom_call (fusion barrier), PCR is a sequence of elementwise
    ops that XLA can fuse into surrounding compute. For the small
    nlev (~30) used in the SI acoustic substep this is comparable in
    raw speed to cuSPARSE but enables single-kernel substep bodies.

    Algorithm: at level k (stride = 2^k), each row eliminates its
    sub/super-diagonal contributions from the row stride positions
    away. After log2(n_pad) levels, every row is decoupled and
    ``x = d / b`` solves the system trivially.

    Shape ``(..., n)`` for a/b/c/d (n is the tridiag system size, last
    axis); leading axes are batched over. n need not be a power of 2 —
    the system is padded to the next power of 2 with identity rows
    (``b=1, a=c=d=0``) which decouple from the original system.
    """
    if a.shape != b.shape or a.shape != c.shape or a.shape != d.shape:
        raise ValueError(
            f"pcr_solve_batched expects matching shapes; got a={a.shape}, "
            f"b={b.shape}, c={c.shape}, d={d.shape}"
        )
    if a.shape[-1] < 2:
        raise ValueError(
            f"pcr_solve_batched requires n>=2 on trailing axis; got {a.shape[-1]}"
        )

    import math
    n = a.shape[-1]
    pad_axes = ((0, 0),) * (a.ndim - 1)
    n_pad = 1 << max(1, (n - 1).bit_length())
    pad_n = n_pad - n

    if pad_n > 0:
        a = jnp.pad(a, (*pad_axes, (0, pad_n)))
        b = jnp.pad(b, (*pad_axes, (0, pad_n)), constant_values=1.0)
        c = jnp.pad(c, (*pad_axes, (0, pad_n)))
        d = jnp.pad(d, (*pad_axes, (0, pad_n)))

    idx = jnp.arange(n_pad)
    n_levels = int(math.log2(n_pad))

    for k in range(n_levels):
        stride = 1 << k
        has_above = idx >= stride
        has_below = idx < n_pad - stride

        a_up = jnp.pad(a[..., :-stride], (*pad_axes, (stride, 0)))
        b_up = jnp.pad(
            b[..., :-stride], (*pad_axes, (stride, 0)), constant_values=1.0,
        )
        c_up = jnp.pad(c[..., :-stride], (*pad_axes, (stride, 0)))
        d_up = jnp.pad(d[..., :-stride], (*pad_axes, (stride, 0)))

        a_dn = jnp.pad(a[..., stride:], (*pad_axes, (0, stride)))
        b_dn = jnp.pad(
            b[..., stride:], (*pad_axes, (0, stride)), constant_values=1.0,
        )
        c_dn = jnp.pad(c[..., stride:], (*pad_axes, (0, stride)))
        d_dn = jnp.pad(d[..., stride:], (*pad_axes, (0, stride)))

        alpha = jnp.where(has_above, -a / b_up, 0.0)
        beta = jnp.where(has_below, -c / b_dn, 0.0)

        a = alpha * a_up
        c = beta * c_dn
        b_new = b + alpha * c_up + beta * a_dn
        d = d + alpha * d_up + beta * d_dn
        b = b_new

    x = d / b
    if pad_n > 0:
        x = x[..., :n]
    return x


def thomas_solve_batched(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: jax.Array,
) -> jax.Array:
    """Batched Thomas solve over columns.

    On GPU (CUDA): wraps cuSPARSE's batched tridiagonal solver via
    ``jax.lax.linalg.tridiagonal_solve``. On other backends (CPU/Metal/TPU)
    where the GPU primitive may be missing/slow: falls back to the legacy
    fori_loop Thomas.

    Verified correctness (iter-39/40 PR #320):
    - fp64 max residual 9.99e-16 (machine precision)
    - fp32 max residual ~5e-7 (machine precision)
    - AD via `jax.grad` works; gradients finite with mean ~1.0 for
      sum-of-output test
    - 6 acoustic-substep tests PASS

    Vertical axis convention: ``a, b, c, d`` MUST have the tridiagonal
    system as the LAST axis. Leading axes are batched over.

    Parameters
    ----------
    a : jax.Array, shape (..., n_sys)
        Sub-diagonal coefficients. ``a[..., 0]`` is unused (set to 0).
    b : jax.Array, shape (..., n_sys)
        Main diagonal coefficients (must be nonzero).
    c : jax.Array, shape (..., n_sys)
        Super-diagonal coefficients. ``c[..., -1]`` is unused (set to 0).
    d : jax.Array, shape (..., n_sys)
        Right-hand side.

    Returns
    -------
    x : jax.Array, shape (..., n_sys)
    """
    # Stronger argument validation per codex iter-57 review:
    if a.shape != b.shape or a.shape != c.shape or a.shape != d.shape:
        raise ValueError(
            f"thomas_solve_batched expects a/b/c/d to share shape; "
            f"got a={a.shape}, b={b.shape}, c={c.shape}, d={d.shape}"
        )
    if a.dtype != b.dtype or a.dtype != c.dtype or a.dtype != d.dtype:
        raise ValueError(
            f"thomas_solve_batched expects a/b/c/d to share dtype; "
            f"got a={a.dtype}, b={b.dtype}, c={c.dtype}, d={d.dtype}"
        )
    if a.ndim < 1:
        raise ValueError(
            f"thomas_solve_batched requires at least 1 axis (the tridiag "
            f"system axis as the last dim); got shape {a.shape}"
        )
    if a.shape[-1] < 2:
        raise ValueError(
            f"thomas_solve_batched requires n_sys >= 2 on the trailing "
            f"axis; got n_sys={a.shape[-1]}"
        )

    # Backend selection priority:
    #   1. env LEGOESM_TRIDIAG=pcr   -> pure-JAX PCR (fusable with surrounding ops)
    #   2. env LEGOESM_TRIDIAG=legacy-> fori_loop Thomas (debug / CPU fallback)
    #   3. CUDA backend                 -> cuSPARSE via tridiagonal_solve
    #   4. Else                        -> legacy fori_loop
    import os
    forced = os.environ.get("LEGOESM_TRIDIAG", "").lower()
    if forced == "pcr":
        return pcr_solve_batched(a, b, c, d)
    if forced == "legacy":
        return _thomas_solve_batched_legacy(a, b, c, d)

    try:
        from jax.lax.linalg import tridiagonal_solve  # noqa: F401
        default_platform = jax.default_backend()
        use_cusparse = default_platform in ("gpu", "cuda")
    except (ImportError, AttributeError):
        use_cusparse = False

    if not use_cusparse:
        return _thomas_solve_batched_legacy(a, b, c, d)

    from jax.lax.linalg import tridiagonal_solve
    import math

    orig_shape = a.shape
    spatial_shape = orig_shape[:-1]
    n_sys = orig_shape[-1]
    n_cols = max(1, math.prod(spatial_shape))

    a_flat = a.reshape(n_cols, n_sys)
    b_flat = b.reshape(n_cols, n_sys)
    c_flat = c.reshape(n_cols, n_sys)
    d_flat = d.reshape(n_cols, n_sys)

    # jax.lax.linalg.tridiagonal_solve signature: (dl, d, du, b)
    # natively accepts a leading batch axis: dl/d/du shape (B, n) and
    # b shape (B, n, nrhs). Pass batched directly — XLA lowers to a
    # single batched cuSPARSE invocation, skipping the vmap-induced
    # per-column launch loop.
    x_flat = tridiagonal_solve(
        a_flat, b_flat, c_flat, d_flat[..., None],
    )[..., 0]
    return x_flat.reshape(orig_shape)


def _thomas_solve_batched_legacy(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: jax.Array,
) -> jax.Array:
    """Legacy fori_loop-based batched Thomas. Kept for CPU fallback /
    regression testing. ~2000× slower than the cuSPARSE-backed default."""
    orig_shape = a.shape
    spatial_shape = orig_shape[:-1]
    n_sys = orig_shape[-1]
    n_cols = 1
    for s in spatial_shape:
        n_cols *= s

    a_flat = a.reshape(n_cols, n_sys)
    b_flat = b.reshape(n_cols, n_sys)
    c_flat = c.reshape(n_cols, n_sys)
    d_flat = d.reshape(n_cols, n_sys)
    x_flat = jax.vmap(thomas_solve)(a_flat, b_flat, c_flat, d_flat)
    return x_flat.reshape(orig_shape)
