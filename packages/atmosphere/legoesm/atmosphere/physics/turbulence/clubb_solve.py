"""Tridiagonal solve adapter for the CLUBB moment-advance LHS band storage.

The CLUBB ``advance_*`` modules assemble their implicit left-hand sides in a
3-row band layout (``advance_windm_edsclrm`` and ``advance_xp2_xpyp`` use the
tridiagonal solver; ``advance_wp2_wp3`` and ``advance_xm_wpxp`` use a separate
pentadiagonal solver, ported later):

    lhs[0, :, k] = superdiagonal  (couples level k to k+1)   [Fortran lhs(-1)]
    lhs[1, :, k] = main diagonal                              [Fortran lhs( 0)]
    lhs[2, :, k] = subdiagonal    (couples level k to k-1)   [Fortran lhs( 1)]

so the system is ``sub[k]*x[k-1] + main[k]*x[k] + super[k]*x[k+1] = rhs[k]``.

Rather than re-derive a tridiagonal solver (CLAUDE.md: reuse shared utilities),
this wraps legoESM's batched Thomas solver
``legoesm.timestepping.tridiagonal.thomas_solve`` (``a*x[k-1] + b*x[k] +
c*x[k+1] = d``), which is JIT/``jax.grad``-clean. The Thomas algorithm is the LU
forward/back substitution for a tridiagonal matrix — mathematically identical to
CLUBB's ``tridiag_lu_solver``; results agree with the reference to round-off.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from legoesm.timestepping.tridiagonal import thomas_solve


def tridiag_solve(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solve the CLUBB-band tridiagonal system ``lhs @ x = rhs`` per column.

    Parameters
    ----------
    lhs : jax.Array
        Band-stored LHS, shape ``(3, ngrdcol, ndim)`` with rows
        ``[super, main, sub]`` (see module docstring).
    rhs : jax.Array
        Right-hand side, shape ``(ngrdcol, ndim)``.

    Returns
    -------
    jax.Array
        Solution, shape ``(ngrdcol, ndim)``.
    """
    sup = lhs[0]   # super (couples to k+1) -> Thomas c
    mid = lhs[1]   # main diagonal          -> Thomas b
    sub = lhs[2]   # sub (couples to k-1)   -> Thomas a
    return thomas_solve(sub, mid, sup, rhs)


def penta_solve(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solve a pentadiagonal system ``lhs @ x = rhs`` via LU (CLUBB band storage).

    Faithful port of ``penta_lu_solver.F90`` /
    ``CLUBB-JAX/.../penta_lu_solver.py`` (``penta_lu_solve``). legoESM has no
    pentadiagonal solver; this is the one used by the coupled moment advances
    (``advance_wp2_wp3``, ``advance_xm_wpxp``), whose interleaved 2-field systems
    are pentadiagonal of size ``2*nzm-1``. Pure ``lax.scan`` LU (no module-scope
    JIT — the caller JITs the physics step); reverse-mode differentiable.

    Parameters
    ----------
    lhs : jax.Array
        Band-stored LHS, shape ``(5, ngrdcol, ndim)`` with rows
        ``[super2, super1, diag, sub1, sub2]`` (Fortran ``lhs(-2:2)``):
        ``super2``/``super1`` couple level ``k`` to ``k+2``/``k+1``;
        ``sub1``/``sub2`` couple to ``k-1``/``k-2``. Requires ``ndim >= 3``.
    rhs : jax.Array
        Right-hand side, shape ``(ngrdcol, ndim)``.

    Returns
    -------
    jax.Array
        Solution, shape ``(ngrdcol, ndim)``.
    """
    super2_t = lhs[0].T   # (ndim, ngrdcol)
    super1_t = lhs[1].T
    diag_t = lhs[2].T
    sub1_t = lhs[3].T
    sub2_t = lhs[4].T
    rhs_t = rhs.T
    ndim = lhs.shape[2]

    # ---- LU decomposition ----
    ldi_0 = 1.0 / diag_t[0]
    u1_0 = ldi_0 * super1_t[0]
    u2_0 = ldi_0 * super2_t[0]
    l1_0 = jnp.zeros_like(ldi_0)
    l2_0 = jnp.zeros_like(ldi_0)

    l1_1 = sub1_t[1]
    l2_1 = jnp.zeros_like(ldi_0)
    ldi_1 = 1.0 / (diag_t[1] - l1_1 * u1_0)
    u1_1 = ldi_1 * (super1_t[1] - l1_1 * u2_0)
    u2_1 = ldi_1 * super2_t[1]

    def lu_scan_step(carry, x):
        u1_km1, u1_km2, u2_km1, u2_km2 = carry
        s2, s1, d, sb1, sb2 = x
        l2 = sb2
        l1 = sb1 - l2 * u1_km2
        ldi = 1.0 / (d - l2 * u2_km2 - l1 * u1_km1)
        u1 = ldi * (s1 - l1 * u2_km1)
        u2 = ldi * s2
        return (u1, u1_km1, u2, u2_km1), (ldi, l1, l2, u1, u2)

    _, (ldi_rest, l1_rest, l2_rest, u1_rest, u2_rest) = lax.scan(
        lu_scan_step, (u1_1, u1_0, u2_1, u2_0),
        (super2_t[2:], super1_t[2:], diag_t[2:], sub1_t[2:], sub2_t[2:]))

    ldi_t = jnp.concatenate([ldi_0[None], ldi_1[None], ldi_rest], axis=0)
    l1_t = jnp.concatenate([l1_0[None], l1_1[None], l1_rest], axis=0)
    l2_t = jnp.concatenate([l2_0[None], l2_1[None], l2_rest], axis=0)
    u1_t = jnp.concatenate([u1_0[None], u1_1[None], u1_rest], axis=0)
    u2_t = jnp.concatenate([u2_0[None], u2_1[None], u2_rest], axis=0)

    # ---- Forward substitution: L y = rhs ----
    soln_0 = ldi_t[0] * rhs_t[0]
    soln_1 = ldi_t[1] * (rhs_t[1] - l1_t[1] * soln_0)

    def fwd_scan_step(carry, x):
        soln_km2, soln_km1 = carry
        rhs_k, ldi_k, l1_k, l2_k = x
        soln_k = ldi_k * (rhs_k - l2_k * soln_km2 - l1_k * soln_km1)
        return (soln_km1, soln_k), soln_k

    _, soln_rest = lax.scan(
        fwd_scan_step, (soln_0, soln_1),
        (rhs_t[2:], ldi_t[2:], l1_t[2:], l2_t[2:]))
    soln_t = jnp.concatenate([soln_0[None], soln_1[None], soln_rest], axis=0)

    # ---- Backward substitution: U x = y ----
    soln_nm2 = soln_t[ndim - 2] - u1_t[ndim - 2] * soln_t[ndim - 1]

    def bwd_scan_step(carry, x):
        soln_kp1, soln_kp2 = carry
        soln_k_fwd, u1_k, u2_k = x
        soln_k = soln_k_fwd - u1_k * soln_kp1 - u2_k * soln_kp2
        return (soln_k, soln_kp1), soln_k

    _, soln_bwd_rev = lax.scan(
        bwd_scan_step, (soln_nm2, soln_t[ndim - 1]),
        (soln_t[:ndim - 2][::-1], u1_t[:ndim - 2][::-1], u2_t[:ndim - 2][::-1]))

    soln_final_t = jnp.concatenate(
        [soln_bwd_rev[::-1], soln_nm2[None], soln_t[ndim - 1:ndim]], axis=0)
    return soln_final_t.T


__all__ = ["tridiag_solve", "penta_solve"]
