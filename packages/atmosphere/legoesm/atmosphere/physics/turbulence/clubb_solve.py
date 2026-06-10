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


__all__ = ["tridiag_solve"]
