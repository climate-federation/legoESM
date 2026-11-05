"""Banded linear solve with partial pivoting, transcribing reference LAPACK.

``lapack_band_solve`` reproduces the arithmetic of reference LAPACK ``dgbsv``
(``dgbtrf`` -> unblocked ``dgbtf2`` for ``kl`` below the ``ilaenv`` block size,
then ``dgbtrs``) with the reference BLAS kernels it calls (``idamax``,
``dswap``, ``dscal``, ``dger``, ``dtbsv``): the same pivot choice (first
maximum of ``|a|``), multipliers formed as ``a * (1/pivot)``, rank-1 updates as
``a + x*(-y)`` and the upper-triangular back solve dividing by the diagonal.
It exists so a port whose oracle solves its band systems with LAPACK (CLUBB
``penta_solve_method = lapack``) can solve them with the same operations.

The column range ``j..JU`` of ``dgbtf2`` and the reference BLAS skips of an
exactly-zero ``y`` (``dger``) or ``x(j)`` (``dtbsv``) are reproduced, so an
overflowing multiplier never meets a ``0`` it would turn into ``NaN``; the
reverse pass is the analytic adjoint below, so these selections never reach a
derivative.  The pivot choice is a selection between two exact eliminations of
the same system, not a limit on the solution, so it is not smoothed.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def lapack_band_solve(bands: jax.Array, rhs: jax.Array, kl: int, ku: int) -> jax.Array:
    """Solve ``A x = rhs`` for a banded ``A`` as reference LAPACK ``dgbsv``.

    Parameters
    ----------
    bands : jax.Array, shape exactly ``(ku + kl + 1, *rhs.shape)``
        Band ``b`` holds ``A[k, k + ku - b]`` at position ``k`` (row-indexed):
        ``bands[0]`` the outermost super-diagonal, ``bands[ku]`` the main
        diagonal, ``bands[ku + kl]`` the outermost sub-diagonal.  Entries that
        fall outside the matrix are ignored.
    rhs : jax.Array, shape ``(*batch, n)``
    kl, ku : int
        Number of sub- and super-diagonals (static).

    Returns
    -------
    jax.Array, shape ``(*batch, n)``

    Differentiability: a ``custom_vjp`` gives the adjoint of the linear solve
    (``λ = A⁻ᵀ x̄`` by the same routine on the transposed bands, ``Ā = −λ xᵀ``
    on the band), not of the elimination, so a tiny pivot cannot overflow the
    reverse pass the way ``1/pivot²`` in raw autodiff does.  Reverse mode only.
    """
    return _band_solve_impl(bands, rhs, kl, ku)


def _band_solve_impl(bands, rhs, kl, ku, skips=True):
    nb = ku + kl + 1
    if bands.shape != (nb, *rhs.shape):
        raise ValueError(f"bands shape {bands.shape} != {(nb, *rhs.shape)}")
    n = bands.shape[-1]
    batch = rhs.shape[:-1]
    dtype = jnp.result_type(bands, rhs)
    bands = jnp.asarray(bands, dtype).reshape(nb, -1, n)
    x = jnp.asarray(rhs, dtype).reshape(-1, n).T          # (n, B)
    nbatch = x.shape[1]
    kv = ku + kl
    ldab = 2 * kl + ku + 1

    # LAPACK band storage AB[kv + i - j, j] = A[i, j]; band b sits on AB row
    # kl + b at column j = i + ku - b.  Columns padded by kv for the windows.
    ab = jnp.zeros((ldab, n + kv, nbatch), dtype)
    for b in range(nb):
        s = ku - b                                         # column offset j - i
        row = bands[b].T                                   # (n, B), indexed by i
        if abs(s) >= n:
            continue                                       # band outside the matrix
        if s >= 0:
            ab = ab.at[kl + b, s:n].set(row[:n - s])
        else:
            ab = ab.at[kl + b, :n + s].set(row[-s:])

    a_idx = jnp.arange(kl + 1)
    r_idx = kv + a_idx[:, None] - jnp.arange(kv + 1)[None, :]   # (kl+1, kv+1)
    c_idx = jnp.broadcast_to(jnp.arange(kv + 1)[None, :], r_idx.shape)

    # ---- dgbtf2: LU with partial pivoting, column by column ----
    b_idx = jnp.arange(kv + 1)

    def factor(j, carry):
        ab, ipiv, ju = carry
        win = lax.dynamic_slice_in_dim(ab, j, kv + 1, axis=1)   # (ldab, kv+1, B)
        blk = win[r_idx, c_idx]                                 # rows j..j+kl
        km = jnp.minimum(kl, n - 1 - j)
        live = a_idx <= km
        cand = jnp.where(live[:, None], jnp.abs(blk[:, 0]), -1.0)
        jp = jnp.argmax(cand, axis=0)                           # idamax: first max
        prow = jnp.take_along_axis(blk, jp[None, None, :], axis=0)[0]
        nz = (prow[0] != 0.0) | (not skips)                     # else singular: skip
        jp = jp.astype(ju.dtype)
        ju = jnp.where(nz, jnp.maximum(ju, jnp.minimum(j + ku + jp, n - 1)).astype(ju.dtype), ju)
        cols = ((j + b_idx)[:, None] <= ju[None, :]) | (not skips)  # (kv+1, B): j..JU
        is_p = a_idx[:, None] == jp[None, :]
        swap = nz[None, None, :] & cols[None]
        blk = jnp.where(swap & (a_idx == 0)[:, None, None], prow[None],
                        jnp.where(swap & is_p[:, None, :], blk[0][None], blk))
        piv = jnp.where(nz, blk[0, 0], 1.0)
        rows = live[1:, None] & nz[None, :]
        mult = jnp.where(rows, blk[1:, 0] * (1.0 / piv), blk[1:, 0])   # dscal by 1/pivot
        y = blk[0, 1:]
        upd = blk[1:, 1:] + mult[:, None] * (-y)[None]          # dger, alpha = -1
        keep = rows[:, None, :] & (cols[1:] & ((y != 0.0) | (not skips)))[None]  # dger skips y == 0
        blk = blk.at[1:, 0].set(mult).at[1:, 1:].set(jnp.where(keep, upd, blk[1:, 1:]))
        win = win.at[r_idx, c_idx].set(blk)
        ab = lax.dynamic_update_slice_in_dim(ab, win, j, axis=1)
        return ab, ipiv.at[j].set(jp.astype(ipiv.dtype)), ju

    ab, ipiv, _ = lax.fori_loop(
        0, n, factor, (ab, jnp.zeros((n, nbatch), jnp.int32),
                       jnp.zeros((nbatch,), jnp.int32)))

    # ---- dgbtrs: apply P and L (forward), then dtbsv upper (backward) ----
    xp = jnp.concatenate([x, jnp.zeros((kl, nbatch), dtype)], axis=0)

    def forward(j, xp):
        w = lax.dynamic_slice_in_dim(xp, j, kl + 1, axis=0)     # (kl+1, B)
        jp = ipiv[j]
        prow = jnp.take_along_axis(w, jp[None, :], axis=0)[0]
        is_p = a_idx[:, None] == jp[None, :]
        w = jnp.where((a_idx == 0)[:, None], prow[None],
                      jnp.where(is_p, w[0][None], w))
        lcol = lax.dynamic_index_in_dim(ab, j, axis=1, keepdims=False)[kv + 1:]
        lm = jnp.minimum(kl, n - 1 - j)
        upd = w[1:] + lcol * (-w[0])[None]
        keep = (a_idx[1:] <= lm)[:, None] & ((w[0] != 0.0) | (not skips))[None]  # dger skips y == 0
        w = w.at[1:].set(jnp.where(keep, upd, w[1:]))
        return lax.dynamic_update_slice_in_dim(xp, w, j, axis=0)

    xp = lax.fori_loop(0, n - 1, forward, xp)

    xb = jnp.concatenate([jnp.zeros((kv, nbatch), dtype), xp[:n]], axis=0)

    def backward(t, xb):
        j = n - 1 - t
        w = lax.dynamic_slice_in_dim(xb, j, kv + 1, axis=0)     # x[j-kv .. j]
        col = lax.dynamic_index_in_dim(ab, j, axis=1, keepdims=False)[:kv + 1]
        go = (w[kv] != 0.0) | (not skips)                      # dtbsv skips x(j) == 0
        xj = jnp.where(go, w[kv] / jnp.where(go, col[kv], 1.0), w[kv])
        w = w.at[kv].set(xj).at[:kv].set(
            jnp.where(go[None], w[:kv] - xj[None] * col[:kv], w[:kv]))
        return lax.dynamic_update_slice_in_dim(xb, w, j, axis=0)

    xb = lax.fori_loop(0, n, backward, xb)
    return xb[kv:].T.reshape(*batch, n)


def _shift(v, s):
    """``out[..., k] = v[..., k + s]``, zero where ``k + s`` leaves ``[0, n)``."""
    n = v.shape[-1]
    if abs(s) >= n:
        return jnp.zeros_like(v)
    z = jnp.zeros_like(v[..., :abs(s)])
    if s > 0:
        return jnp.concatenate([v[..., s:], z], axis=-1)
    if s < 0:
        return jnp.concatenate([z, v[..., :n + s]], axis=-1)
    return v


def _band_solve_fwd(bands, rhs, kl, ku):
    # Under differentiation the primal is the skip-free elimination (equal to
    # the skipping one up to the sign of a zero when finite), so a higher-order
    # pass differentiating this rule sees no zero-selection; the skipping solve
    # is used only when the skip-free one overflowed.
    x = _finite_or_skipping(bands, rhs, kl, ku)
    return x, (bands, x, jnp.zeros((), rhs.dtype))


def _finite_or_skipping(bands, rhs, kl, ku):
    ok = jnp.all(jnp.isfinite(lax.stop_gradient(
        _band_solve_impl(bands, rhs, kl, ku, skips=False))))
    # Whole-solve switch (not per entry): an overflow inside the skip-free
    # elimination poisons its derivatives everywhere, finite entries included.
    # Both solves sit INSIDE the branches so the overflowed one never enters
    # the derivative.
    return lax.cond(ok, lambda: _band_solve_impl(bands, rhs, kl, ku, skips=False),
                    lambda: _band_solve_impl(bands, rhs, kl, ku))


def _band_solve_bwd(kl, ku, res, x_bar):
    bands, x, rhs_dtype_tag = res
    nb = kl + ku + 1
    # Aᵀ is banded with kl' = ku, ku' = kl: band b' holds Aᵀ[k, k+kl-b'] =
    # A[k+kl-b', k] = bands[nb-1-b'][k+kl-b'].
    bands_t = jnp.stack([_shift(bands[nb - 1 - b], kl - b) for b in range(nb)])
    # Skip-free adjoint solve (overflow aside): BLAS zero-skips would cut the
    # derivative a second-order reverse pass takes here (A = 2I, rhs = 0).
    lam = _finite_or_skipping(bands_t, x_bar, ku, kl)
    bands_bar = jnp.stack([-lam * _shift(x, ku - b) for b in range(nb)])
    # entries outside the matrix carry no derivative
    n = x.shape[-1]
    k = jnp.arange(n)
    valid = jnp.stack([(k + ku - b >= 0) & (k + ku - b < n) for b in range(nb)])
    bands_bar = jnp.where(valid.reshape((nb,) + (1,) * (x.ndim - 1) + (n,)), bands_bar, 0.0)
    return bands_bar.astype(bands.dtype), lam.astype(rhs_dtype_tag.dtype)


lapack_band_solve.defvjp(_band_solve_fwd, _band_solve_bwd)
