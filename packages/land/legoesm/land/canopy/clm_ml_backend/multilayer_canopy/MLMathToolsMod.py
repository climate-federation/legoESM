"""
JAX translation of MLMathToolsMod Fortran module.

Mathematical tools for the multilayer canopy model:
root-finding (hybrid secant/Brent, Brent, bisection),
quadratic solver, tridiagonal solvers (scalar and 2-equation
coupled), log-gamma function, beta function, beta distribution
PDF and CDF, and the incomplete beta continued fraction.

Original Fortran module: MLMathToolsMod
Fortran lines 1-330
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varctl import iulog  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varpar import nlevmlcan  # noqa: F401

# Type alias for the function signature shared by all solvers.
# Fortran abstract interface `func(p, ic, il, mlcanopy_inst, x, val)`
# → Python callable returning (val, updated_mlcanopy_inst).
FuncType = Callable[
    [int, int, int, mlcanopy_type, float],
    Tuple[float, mlcanopy_type],
]

# Stateless variant: pure Python scalar function x -> residual.
# Used by the refactored photosynthesis/stomatal solver paths.
ScalarFuncType = Callable[[float], float]


# ---------------------------------------------------------------------------
# Public: hybrid secant / Brent root finder
# ---------------------------------------------------------------------------


def hybrid(
    msg: str,
    p: int,
    ic: int,
    il: int,
    mlcanopy_inst: mlcanopy_type,
    func: FuncType,
    xa: float,
    xb: float,
    tol: float,
) -> Tuple[float, mlcanopy_type]:
    """
    Find the root of ``func`` using the secant method with Brent's
    method as a robust fallback.

    Mirrors Fortran function ``hybrid`` (lines 47-100).

    Starting from initial estimates ``xa`` and ``xb``, the secant method
    iterates until ``|dx| < tol``.  If the iterates bracket a root
    (``f1 * f0 < 0``), :func:`zbrent` is called for the final result.
    If convergence is not achieved within ``itmax = 40`` iterations,
    the iterate with the smallest absolute function value is returned.

    Args:
        msg: Diagnostic string passed to :func:`zbrent` if called.
        p: Patch index.
        ic: Canopy layer index.
        il: Sunlit (1) or shaded (2) leaf index.
        mlcanopy_inst: Canopy container threaded through ``func``.
        func: Callable ``(p, ic, il, mlcanopy_inst, x) → (val, mlcanopy_inst)``.
        xa: First initial estimate.
        xb: Second initial estimate.
        tol: Convergence tolerance.

    Returns:
        Tuple ``(root, mlcanopy_inst)``.
    """
    itmax: int = 40  # Fortran: parameter itmax = 40

    x0 = xa
    f0, mlcanopy_inst = func(p, ic, il, mlcanopy_inst, x0)
    if f0 == 0.0:
        return x0, mlcanopy_inst

    x1 = xb
    f1, mlcanopy_inst = func(p, ic, il, mlcanopy_inst, x1)
    if f1 == 0.0:
        return x1, mlcanopy_inst

    # Track the iterate with the smallest |f| — Fortran lines 71-75
    if abs(f1) < abs(f0):
        minx, minf = x1, f1
    else:
        minx, minf = x0, f0

    # Secant iteration — Fortran lines 77-97
    _iter = 0
    while True:
        _iter += 1
        dx = -f1 * (x1 - x0) / (f1 - f0)
        x = x1 + dx
        if abs(dx) < tol:
            x0 = x
            break
        x0 = x1
        f0 = f1
        x1 = x
        f1, mlcanopy_inst = func(p, ic, il, mlcanopy_inst, x1)
        if abs(f1) < abs(minf):
            minx, minf = x1, f1

        # Root bracketed — switch to Brent — Fortran lines 86-89
        if f1 * f0 < 0.0:
            x0, mlcanopy_inst = zbrent(msg, p, ic, il, mlcanopy_inst, func, x0, x1, tol)
            break

        # Fallback to minimum — Fortran lines 91-95
        if _iter > itmax:
            f1, mlcanopy_inst = func(p, ic, il, mlcanopy_inst, minx)
            x0 = minx
            break

    return x0, mlcanopy_inst


# ---------------------------------------------------------------------------
# Public: hybrid secant / Brent root finder (stateless scalar variant)
# ---------------------------------------------------------------------------


def hybrid_scalar(
    msg: str,
    func: ScalarFuncType,
    xa: float,
    xb: float,
    tol: float,
) -> float:
    """
    Stateless version of :func:`hybrid`.

    ``func`` is a pure Python callable ``func(x) -> float`` with no
    JAX state threading.  Mirrors the same algorithm as :func:`hybrid`
    but returns only the root as a float.
    """
    itmax: int = 40

    x0 = xa
    f0 = func(x0)
    if f0 == 0.0:
        return x0

    x1 = xb
    f1 = func(x1)
    if f1 == 0.0:
        return x1

    if abs(f1) < abs(f0):
        minx, minf = x1, f1
    else:
        minx, minf = x0, f0

    _iter = 0
    while True:
        _iter += 1
        dx = -f1 * (x1 - x0) / (f1 - f0)
        x = x1 + dx
        if abs(dx) < tol:
            x0 = x
            break
        x0 = x1
        f0 = f1
        x1 = x
        f1 = func(x1)
        if abs(f1) < abs(minf):
            minx, minf = x1, f1

        if f1 * f0 < 0.0:
            x0 = zbrent_scalar(msg, func, x0, x1, tol)
            break

        if _iter > itmax:
            func(minx)
            x0 = minx
            break

    return x0


# ---------------------------------------------------------------------------
# Public: Brent's method
# ---------------------------------------------------------------------------


def zbrent(
    msg: str,
    p: int,
    ic: int,
    il: int,
    mlcanopy_inst: mlcanopy_type,
    func: FuncType,
    xa: float,
    xb: float,
    tol: float,
) -> Tuple[float, mlcanopy_type]:
    """
    Find the root of ``func`` bracketed by ``[xa, xb]`` using
    Brent's method.

    Mirrors Fortran function ``zbrent`` (lines 102-160).

    Args:
        msg: Diagnostic string printed on error.
        p: Patch index.
        ic: Canopy layer index.
        il: Sunlit (1) or shaded (2) leaf index.
        mlcanopy_inst: Canopy container threaded through ``func``.
        func: Callable ``(p, ic, il, mlcanopy_inst, x) → (val, mlcanopy_inst)``.
        xa: Lower bracket.
        xb: Upper bracket.
        tol: Convergence tolerance.

    Returns:
        Tuple ``(root, mlcanopy_inst)``.
    """
    itmax: int = 50  # Fortran: parameter itmax = 50
    eps: float = 1.0e-8  # Fortran: parameter eps = 1.e-08_r8

    a = xa
    b = xb
    fa, mlcanopy_inst = func(p, ic, il, mlcanopy_inst, a)
    fb, mlcanopy_inst = func(p, ic, il, mlcanopy_inst, b)

    # Bracket check — Fortran lines 124-131
    if (fa > 0.0 and fb > 0.0) or (fa < 0.0 and fb < 0.0):
        print(f"{iulog}: zbrent: Root must be bracketed")
        print(f"{iulog}: called from: {msg}")
        print(f"{iulog}: {xa} {fa}")
        print(f"{iulog}: {xb} {fb}")
        endrun(msg=" ERROR: zbrent error")

    c = b
    fc = fb
    d = 0.0
    e = 0.0  # Initialised before use in first iteration

    _iter = 0
    while True:
        if _iter == itmax:
            break
        _iter += 1

        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
            c = a
            fc = fa
            d = b - a
            e = d

        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        tol1 = 2.0 * eps * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)

        if abs(xm) <= tol1 or fb == 0.0:
            break

        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                pp = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                pp = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if pp > 0.0:
                q = -q
            pp = abs(pp)
            if 2.0 * pp < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = pp / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d

        a = b
        fa = fb
        if abs(d) > tol1:
            b = b + d
        else:
            b = b + jnp.copysign(tol1, xm)  # Fortran: b + sign(tol1, xm)

        fb, mlcanopy_inst = func(p, ic, il, mlcanopy_inst, b)
        if fb == 0.0:
            break

    if _iter == itmax:
        print(f"{iulog}: zbrent: Maximum number of iterations exceeded")
        print(f"{iulog}: called from: {msg}")
        endrun(msg=" ERROR: zbrent error")

    return b, mlcanopy_inst


# ---------------------------------------------------------------------------
# Public: Brent's method (stateless scalar variant)
# ---------------------------------------------------------------------------


def zbrent_scalar(
    msg: str,
    func: ScalarFuncType,
    xa: float,
    xb: float,
    tol: float,
) -> float:
    """
    Stateless version of :func:`zbrent`.

    ``func`` is a pure Python callable ``func(x) -> float``.
    Returns the root as a float.
    """
    itmax: int = 50
    eps: float = 1.0e-8

    a = xa
    b = xb
    fa = func(a)
    fb = func(b)

    if (fa > 0.0 and fb > 0.0) or (fa < 0.0 and fb < 0.0):
        print(f"{iulog}: zbrent_scalar: Root must be bracketed")
        print(f"{iulog}: called from: {msg}")
        print(f"{iulog}: {xa} {fa}")
        print(f"{iulog}: {xb} {fb}")
        endrun(msg=" ERROR: zbrent_scalar error")

    c = b
    fc = fb
    d = 0.0
    e = 0.0

    _iter = 0
    while True:
        if _iter == itmax:
            break
        _iter += 1

        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
            c = a
            fc = fa
            d = b - a
            e = d

        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        tol1 = 2.0 * eps * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)

        if abs(xm) <= tol1 or fb == 0.0:
            break

        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                pp = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                pp = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if pp > 0.0:
                q = -q
            pp = abs(pp)
            if 2.0 * pp < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = pp / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d

        a = b
        fa = fb
        if abs(d) > tol1:
            b = b + d
        else:
            b = b + math.copysign(tol1, xm)

        fb = func(b)
        if fb == 0.0:
            break

    if _iter == itmax:
        print(f"{iulog}: zbrent_scalar: Maximum number of iterations exceeded")
        print(f"{iulog}: called from: {msg}")
        endrun(msg=" ERROR: zbrent_scalar error")

    return b


# ---------------------------------------------------------------------------
# Public: bisection root finder
# ---------------------------------------------------------------------------


def bisection(
    msg: str,
    p: int,
    ic: int,
    il: int,
    mlcanopy_inst: mlcanopy_type,
    func: FuncType,
    xa: float,
    xb: float,
    tol: float,
) -> Tuple[float, mlcanopy_type]:
    """
    Find the root of ``func`` bracketed by ``[xa, xb]`` using
    bisection.

    Mirrors Fortran function ``bisection`` (lines 162-210).

    Args:
        msg: Diagnostic string printed on error.
        p: Patch index.
        ic: Canopy layer index.
        il: Sunlit (1) or shaded (2) leaf index.
        mlcanopy_inst: Canopy container threaded through ``func``.
        func: Callable ``(p, ic, il, mlcanopy_inst, x) → (val, mlcanopy_inst)``.
        xa: Lower bracket.
        xb: Upper bracket.
        tol: Convergence tolerance.

    Returns:
        Tuple ``(root, mlcanopy_inst)``.
    """
    itmax: int = 100  # Fortran: parameter itmax = 100

    a = xa
    b = xb
    fa, mlcanopy_inst = func(p, ic, il, mlcanopy_inst, a)
    fb, mlcanopy_inst = func(p, ic, il, mlcanopy_inst, b)

    # Bracket check — Fortran lines 184-190
    if fa * fb > 0.0:
        print(f"{iulog}: bisection error: Root must be bracketed")
        print(f"{iulog}: called from: {msg}")
        print(f"{iulog}: {xa} {fa}")
        print(f"{iulog}: {xb} {fb}")
        endrun(msg=" ERROR: bisection error")

    c = a  # Initialise before while-loop to satisfy type checker
    _iter = 1
    while abs(b - a) > tol and _iter <= itmax:  # Fortran: do while
        c = (a + b) / 2.0
        fc, mlcanopy_inst = func(p, ic, il, mlcanopy_inst, c)
        if fa * fc < 0.0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        _iter += 1

    if _iter > itmax:
        print(f"{iulog}: bisection error: Maximum number of iterations exceeded")
        print(f"{iulog}: called from: {msg}")
        endrun(msg=" ERROR: bisection error")

    return c, mlcanopy_inst


# ---------------------------------------------------------------------------
# Public: bisection root finder (stateless scalar variant)
# ---------------------------------------------------------------------------


def bisection_scalar(
    msg: str,
    func: ScalarFuncType,
    xa: float,
    xb: float,
    tol: float,
) -> float:
    """
    Stateless version of :func:`bisection`.

    ``func`` is a pure Python callable ``func(x) -> float``.
    Returns the root as a float.
    """
    itmax: int = 100

    a = xa
    b = xb
    fa = func(a)
    fb = func(b)

    if fa * fb > 0.0:
        print(f"{iulog}: bisection_scalar error: Root must be bracketed")
        print(f"{iulog}: called from: {msg}")
        print(f"{iulog}: {xa} {fa}")
        print(f"{iulog}: {xb} {fb}")
        endrun(msg=" ERROR: bisection_scalar error")

    c = a
    _iter = 1
    while abs(b - a) > tol and _iter <= itmax:
        c = (a + b) / 2.0
        fc = func(c)
        if fa * fc < 0.0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        _iter += 1

    if _iter > itmax:
        print(f"{iulog}: bisection_scalar error: Maximum number of iterations exceeded")
        print(f"{iulog}: called from: {msg}")
        endrun(msg=" ERROR: bisection_scalar error")

    return c


# ---------------------------------------------------------------------------
# Public: quadratic solver
# ---------------------------------------------------------------------------


def quadratic(a: float, b: float, c: float) -> Tuple[float, float]:
    """
    Solve the quadratic equation ``a*x^2 + b*x + c = 0`` for its
    two roots using a numerically stable form.

    Mirrors Fortran subroutine ``quadratic`` (lines 212-240).

    Uses the sign of ``b`` to choose the branch that avoids
    catastrophic cancellation:

    .. code-block:: none

        q  = -0.5*(b + sqrt(b^2 - 4ac))   if b >= 0
        q  = -0.5*(b - sqrt(b^2 - 4ac))   if b < 0
        r1 = q/a
        r2 = c/q   (or 1e36 if q == 0)

    Args:
        a: Quadratic coefficient (must be non-zero).
        b: Linear coefficient.
        c: Constant term.

    Returns:
        Tuple ``(r1, r2)`` of the two roots.
    """
    # Safe sqrt: clamp discriminant to ≥0 to avoid NaN when JAX traces
    # both branches unconditionally (e.g. inside jnp.where callers).
    discriminant = jnp.sqrt(jnp.maximum(b * b - 4.0 * a * c, 1.0e-30))

    # Numerically stable branch selection via jnp.where — avoids
    # Python `if` on potentially JAX-traced `b`.
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug.
    # All callers pass a > 0 (curvature parameters, conductances); using
    # maximum is numerically equivalent for all physically valid inputs.
    a_safe = jnp.maximum(a, 1.0e-30)
    q = jnp.where(
        b >= 0.0,
        -0.5 * (b + discriminant),
        -0.5 * (b - discriminant),
    )
    # q can be signed; guard against exact 0 by adding a tiny offset
    # when q == 0 (avoids select-as-denominator pattern).
    q_safe = q + jnp.asarray(q == 0.0, dtype=q.dtype)  # adds 1.0 when q == 0
    r1 = q / a_safe
    r2 = jnp.where(q != 0.0, c / q_safe, jnp.asarray(1.0e36))

    return r1, r2


def quadratic_py(a: float, b: float, c: float):
    """Pure-Python version of :func:`quadratic` using ``math.sqrt``.

    No JAX dispatch overhead — safe for use inside per-layer Python loops
    where ``a``, ``b``, ``c`` are plain Python floats.
    """
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        disc = 0.0
    sq = math.sqrt(disc)
    if b >= 0.0:
        q = -0.5 * (b + sq)
    else:
        q = -0.5 * (b - sq)
    if abs(a) > 0.0:
        r1 = q / a
    else:
        r1 = 1.0e36
    if q != 0.0:
        r2 = c / q
    else:
        r2 = 1.0e36
    return r1, r2


# ---------------------------------------------------------------------------
# Public: scalar tridiagonal solver
# ---------------------------------------------------------------------------


def tridiag(a: Array, b: Array, c: Array, r: Array, n: int) -> Array:
    """
    Solve tridiagonal system of equations using Thomas algorithm.

    Translated from Fortran TridiagonalMod.F90.

    Args:
        a: Lower diagonal coefficients [1:n]
        b: Main diagonal coefficients [1:n]
        c: Upper diagonal coefficients [1:n]
        r: Right-hand side [1:n]
        n: System size

    Returns:
        Solution vector [1:n]

    Note:
        Arrays use 1-based indexing in the algorithm (index 0 unused).
        Input arrays should have shape (n+1,) to accommodate this.
    """
    # Allocate arrays (1-based indexing, so size n+1)
    gam = jnp.zeros(n + 1, dtype=jnp.float64)
    u = jnp.zeros(n + 1, dtype=jnp.float64)

    # Forward elimination (Fortran lines similar to original)
    # Guard pivot (bet) away from zero to keep ratio coefficients bounded.
    # Physical systems have bet > 0 (positive definite diagonal dominance).
    _eps_bet = jnp.asarray(1.0e-10)
    bet = b[1]
    bet_safe = jnp.maximum(jnp.abs(bet), _eps_bet)
    u = u.at[1].set(r[1] / bet_safe)

    for j in range(2, n + 1):
        gam = gam.at[j].set(c[j - 1] / bet_safe)
        bet = b[j] - a[j] * gam[j]
        bet_safe = jnp.maximum(jnp.abs(bet), _eps_bet)
        u = u.at[j].set((r[j] - a[j] * u[j - 1]) / bet_safe)

    # Backward substitution
    for j in range(n - 1, 0, -1):
        u = u.at[j].set(u[j] - gam[j + 1] * u[j + 1])

    return u


# ---------------------------------------------------------------------------
# Public: coupled 2-equation tridiagonal solver
# ---------------------------------------------------------------------------


def _tridiag_2eq_fwd(
    a1: list[float],
    b11: list[float],
    b12: list[float],
    c1: list[float],
    d1: list[float],
    a2: list[float],
    b21: list[float],
    b22: list[float],
    c2: list[float],
    d2: list[float],
    n: int,
) -> Tuple[list[float], list[float]]:
    """Forward-pass implementation of the coupled 2-equation tridiagonal solver.

    All arrays are **0-based** (indices 0..n-1).  Returns ``(t, q)`` solution
    vectors.  Also used as the plain (non-differentiable) implementation via
    the custom-VJP trampoline below.

    Mirrors Fortran subroutine ``tridiag_2eq`` (lines 287-350).
    """
    t, q = _tridiag_2eq_solve(a1, b11, b12, c1, d1, a2, b21, b22, c2, d2, n)
    return t, q


@jax.custom_vjp
def tridiag_2eq(
    a1: list[float],
    b11: list[float],
    b12: list[float],
    c1: list[float],
    d1: list[float],
    a2: list[float],
    b21: list[float],
    b22: list[float],
    c2: list[float],
    d2: list[float],
    n: int,
) -> Tuple[list[float], list[float]]:
    """
    Solve the coupled tridiagonal system for air temperature ``t`` and
    water vapour mole fraction ``q`` at each canopy layer.

    Mirrors Fortran subroutine ``tridiag_2eq`` (lines 287-350).

    The system at each layer ``i = 0, ..., n-1`` (0-based) is:

    .. code-block:: none

        a1(i)*T(i-1) + b11(i)*T(i) + b12(i)*q(i) + c1(i)*T(i+1) = d1(i)
        a2(i)*q(i-1) + b21(i)*T(i) + b22(i)*q(i) + c2(i)*q(i+1) = d2(i)

    Solved by forward elimination to express each layer in terms of the
    layer above, followed by back substitution from the top layer downward.

    A **custom VJP** is registered so that the backward pass solves the
    transpose system ``A^T λ = g`` directly rather than differentiating
    through the potentially ill-conditioned forward elimination, which can
    produce huge intermediate ``e``-coefficients and NaN gradients.

    All arrays are **0-based** (indices 0..n-1), matching the calling
    convention in ``ImplicitFluxProfileSolution``.

    Args:
        a1, b11, b12, c1, d1: Coefficients for the temperature equation,
            each length at least ``n``, 0-indexed.
        a2, b21, b22, c2, d2: Coefficients for the water vapour equation,
            each length at least ``n``, 0-indexed.
        n: Number of layers.

    Returns:
        Tuple ``(t, q)`` of solution vectors, each of length ``n``, 0-indexed.
    """
    t, q = _tridiag_2eq_fwd(a1, b11, b12, c1, d1, a2, b21, b22, c2, d2, n)
    return t, q


def _tridiag_2eq_fwd_rule(a1, b11, b12, c1, d1, a2, b21, b22, c2, d2, n):
    """Forward rule for custom VJP: runs the solver and saves residuals."""
    t, q = _tridiag_2eq_fwd(a1, b11, b12, c1, d1, a2, b21, b22, c2, d2, n)
    # Save the inputs and solution for the backward pass.
    residuals = (a1, b11, b12, c1, d1, a2, b21, b22, c2, d2, t, q)
    return (t, q), residuals


def _tridiag_2eq_bwd_rule(residuals, g):
    """Backward rule: solve the transpose system A^T λ = g.

    For the block-tridiagonal system A x = d with x = [t; q]:

      Row i:  a1[i]*T[i-1] + [[b11,b12],[b21,b22]][i] * [T;q][i]
                            + c1[i]*T[i+1] + c2[i]*q[i+1] = [d1;d2][i]

    where lower-diag block at row i is diag(a1[i], a2[i]),
    main block is [[b11, b12],[b21, b22]], upper block is diag(c1[i], c2[i]).

    The transpose system A^T λ = g (same block size n) has:
      - lower-diag block at row i  = diag(c1[i-1], c2[i-1])  (was upper of A)
      - main block at row i         = [[b11[i], b21[i]], [b12[i], b22[i]]]  (transposed B)
      - upper-diag block at row i   = diag(a1[i+1], a2[i+1])  (was lower of A)

    We solve A^T λ = g using the same block Thomas algorithm applied to
    this transposed system.  The solution λ = [z1; z2] satisfies:

        dL/d(d1[i]) = z1[i],    dL/d(d2[i]) = z2[i]

    Gradients w.r.t. matrix coefficients follow from the implicit function
    theorem.  For A(θ)y = d(θ), the full VJP is λᵀ(∂d/∂θ − ∂A/∂θ · y),
    so for each matrix entry:

        dL/dA[i,j] = -λ[i] * y[j]

    Expanded for the block structure (0-based index i, solution (t, q)):

        dL/da1[i]  = -z1[i] * t[i-1]   (t[-1] = 0 by boundary)
        dL/db11[i] = -z1[i] * t[i]
        dL/db12[i] = -z1[i] * q[i]
        dL/dc1[i]  = -z1[i] * t[i+1]   (t[n]  = 0 by boundary)
        dL/da2[i]  = -z2[i] * q[i-1]   (q[-1] = 0 by boundary)
        dL/db21[i] = -z2[i] * t[i]
        dL/db22[i] = -z2[i] * q[i]
        dL/dc2[i]  = -z2[i] * q[i+1]   (q[n]  = 0 by boundary)
    """
    a1, b11, b12, c1, d1, a2, b21, b22, c2, d2, _t, _q = residuals
    g_t, g_q = g  # cotangents for t and q outputs (JAX arrays of length n)

    n = len(d1)

    # Build explicit arrays for A^T coefficients.
    # A^T block structure (0-based, row i):
    #   lower  (a^T): at row i it's the (i, i-1) block of A^T
    #                 = (i-1, i) block of A = diag(c1[i-1], c2[i-1])
    #                 → a1T[i] = c1[i-1],  a2T[i] = c2[i-1],  with a1T[0]=a2T[0]=0
    #   main   (b^T): B_i^T = [[b11[i], b21[i]], [b12[i], b22[i]]]
    #   upper  (c^T): at row i it's the (i, i+1) block of A^T
    #                 = (i+1, i) block of A = diag(a1[i+1], a2[i+1])
    #                 → c1T[i] = a1[i+1], c2T[i] = a2[i+1],  with c1T[n-1]=c2T[n-1]=0

    # lower diagonal of A^T
    a1T = [jnp.zeros(())] * n
    a2T = [jnp.zeros(())] * n
    for i in range(1, n):
        a1T[i] = c1[i - 1]
        a2T[i] = c2[i - 1]

    # main diagonal of A^T (transposed 2x2 blocks)
    # b11T[i] = b11[i], b12T[i] = b21[i], b21T[i] = b12[i], b22T[i] = b22[i]
    b11T = [b11[i] for i in range(n)]
    b12T = [b21[i] for i in range(n)]
    b21T = [b12[i] for i in range(n)]
    b22T = [b22[i] for i in range(n)]

    # upper diagonal of A^T
    c1T = [jnp.zeros(())] * n
    c2T = [jnp.zeros(())] * n
    for i in range(n - 1):
        c1T[i] = a1[i + 1]
        c2T[i] = a2[i + 1]

    # RHS of A^T system = g = [g_t; g_q]
    g1 = [g_t[i] for i in range(n)]
    g2 = [g_q[i] for i in range(n)]

    # Solve A^T λ = g using the same Thomas algorithm as _tridiag_2eq_fwd.
    z1_list, z2_list = _tridiag_2eq_solve(a1T, b11T, b12T, c1T, g1, a2T, b21T, b22T, c2T, g2, n)

    # Gradient w.r.t. d1 and d2 is the adjoint λ = [z1; z2].
    g_d1 = jnp.stack(z1_list)
    g_d2 = jnp.stack(z2_list)

    # Gradients w.r.t. matrix coefficients: dL/dA[i,j] = -λ[i] * y[j]
    # where y = (t, q) is the primal solution.
    # Boundary ghost values: t[-1] = t[n] = q[-1] = q[n] = 0.
    _zero = jnp.zeros(())
    g_a1_list = []
    g_b11_list = []
    g_b12_list = []
    g_c1_list = []
    g_a2_list = []
    g_b21_list = []
    g_b22_list = []
    g_c2_list = []
    for i in range(n):
        t_prev = _t[i - 1] if i > 0 else _zero
        t_curr = _t[i]
        t_next = _t[i + 1] if i < n - 1 else _zero
        q_prev = _q[i - 1] if i > 0 else _zero
        q_curr = _q[i]
        q_next = _q[i + 1] if i < n - 1 else _zero

        g_a1_list.append(-z1_list[i] * t_prev)
        g_b11_list.append(-z1_list[i] * t_curr)
        g_b12_list.append(-z1_list[i] * q_curr)
        g_c1_list.append(-z1_list[i] * t_next)
        g_a2_list.append(-z2_list[i] * q_prev)
        g_b21_list.append(-z2_list[i] * t_curr)
        g_b22_list.append(-z2_list[i] * q_curr)
        g_c2_list.append(-z2_list[i] * q_next)

    # Cotangents must match the primal input pytree structure.
    return (
        jnp.stack(g_a1_list),  # grad a1
        jnp.stack(g_b11_list),  # grad b11
        jnp.stack(g_b12_list),  # grad b12
        jnp.stack(g_c1_list),  # grad c1
        g_d1,  # grad d1  =  λ_t
        jnp.stack(g_a2_list),  # grad a2
        jnp.stack(g_b21_list),  # grad b21
        jnp.stack(g_b22_list),  # grad b22
        jnp.stack(g_c2_list),  # grad c2
        g_d2,  # grad d2  =  λ_q
        None,  # grad n (static int)
    )


def _tridiag_2eq_solve(
    a1,
    b11,
    b12,
    c1,
    d1,
    a2,
    b21,
    b22,
    c2,
    d2,
    n,
):
    """Core Thomas algorithm for the block-2x2 tridiagonal system.

    Accepts arbitrary list-like inputs (Python lists or slices thereof).
    Returns (t_list, q_list) as Python lists of JAX scalars.
    """
    # n may be a JAX tracer if called inside a traced context; use len(d1)
    # to always get a concrete Python int for list allocation and range().
    n = len(d1)
    e11 = [jnp.zeros(())] * n
    e12 = [jnp.zeros(())] * n
    e21 = [jnp.zeros(())] * n
    e22 = [jnp.zeros(())] * n
    f1 = [jnp.zeros(())] * n
    f2 = [jnp.zeros(())] * n

    e11_prev = jnp.zeros(())
    e12_prev = jnp.zeros(())
    e21_prev = jnp.zeros(())
    e22_prev = jnp.zeros(())
    f1_prev = jnp.zeros(())
    f2_prev = jnp.zeros(())

    _eps_det = jnp.asarray(1.0e-10)
    for i in range(n):
        ainv = b11[i] - a1[i] * e11_prev
        binv = b12[i] - a1[i] * e12_prev
        cinv = b21[i] - a2[i] * e21_prev
        dinv = b22[i] - a2[i] * e22_prev
        det = ainv * dinv - binv * cinv
        _abs_det = jnp.abs(det)
        det_safe = jnp.where(_abs_det > _eps_det, det, _eps_det)

        e11[i] = dinv * c1[i] / det_safe
        e12[i] = -binv * c2[i] / det_safe
        e21[i] = -cinv * c1[i] / det_safe
        e22[i] = ainv * c2[i] / det_safe

        f1[i] = (dinv * (d1[i] - a1[i] * f1_prev) - binv * (d2[i] - a2[i] * f2_prev)) / det_safe
        f2[i] = (-cinv * (d1[i] - a1[i] * f1_prev) + ainv * (d2[i] - a2[i] * f2_prev)) / det_safe

        e11_prev, e12_prev = e11[i], e12[i]
        e21_prev, e22_prev = e21[i], e22[i]
        f1_prev, f2_prev = f1[i], f2[i]

    t = [jnp.zeros(())] * n
    q = [jnp.zeros(())] * n
    t[n - 1] = f1[n - 1]
    q[n - 1] = f2[n - 1]
    for i in range(n - 2, -1, -1):
        t[i] = f1[i] - e11[i] * t[i + 1] - e12[i] * q[i + 1]
        q[i] = f2[i] - e21[i] * t[i + 1] - e22[i] * q[i + 1]

    return t, q


tridiag_2eq.defvjp(_tridiag_2eq_fwd_rule, _tridiag_2eq_bwd_rule)


# ---------------------------------------------------------------------------
# Public: log-gamma function
# ---------------------------------------------------------------------------


def log_gamma_function(x: float) -> float:
    """
    Return the natural logarithm of the gamma function, ``ln(Γ(x))``,
    using Lanczos's approximation.

    Mirrors Fortran function ``log_gamma_function`` (lines 362-390).

    Coefficients follow the Numerical Recipes implementation
    (Fortran lines 377-379).

    Args:
        x: Input argument (x > 0).

    Returns:
        ``ln(Γ(x))``.
    """
    # Fortran: parameter coef(6), stp
    coef = (
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    )
    stp: float = 2.5066282746310005

    y = x
    tmp = x + 5.5
    tmp = (x + 0.5) * jnp.log(tmp) - tmp
    ser = 1.000000000190015
    for j in range(6):  # Fortran: do j = 1, 6; y = y + 1; ser += coef(j)/y
        y += 1.0
        ser += coef[j] / y
    return tmp + jnp.log(stp * ser / x)


# ---------------------------------------------------------------------------
# Public: beta function
# ---------------------------------------------------------------------------


def beta_function(a: float, b: float) -> float:
    """
    Return the beta function ``B(a, b) = Γ(a)Γ(b)/Γ(a+b)``.

    Mirrors Fortran function ``beta_function`` (lines 392-405).

    Args:
        a: First parameter (a > 0).
        b: Second parameter (b > 0).

    Returns:
        ``B(a, b)``.
    """
    return jnp.exp(log_gamma_function(a) + log_gamma_function(b) - log_gamma_function(a + b))


# ---------------------------------------------------------------------------
# Public: beta distribution PDF
# ---------------------------------------------------------------------------


def beta_distribution_pdf(a: float, b: float, x: float) -> float:
    """
    Return the beta distribution probability density function,
    ``f(x; a, b)``.

    Mirrors Fortran function ``beta_distribution_pdf`` (lines 407-425).

    .. code-block:: none

        f(x; a, b) = x^(a-1) * (1-x)^(b-1) / B(a, b)

    Args:
        a: First shape parameter (a > 0).
        b: Second shape parameter (b > 0).
        x: Evaluation point (0 ≤ x ≤ 1).

    Returns:
        ``f(x; a, b)``.
    """
    return (1.0 / beta_function(a, b)) * x ** (a - 1.0) * (1.0 - x) ** (b - 1.0)


# ---------------------------------------------------------------------------
# Public: beta distribution CDF
# ---------------------------------------------------------------------------


def beta_distribution_cdf(a: float, b: float, x: float) -> float:
    """
    Return the beta distribution cumulative distribution function,
    ``F(x; a, b) = I_x(a, b)``, using the incomplete beta function.

    Mirrors Fortran function ``beta_distribution_cdf`` (lines 427-455).

    Uses the continued fraction representation via
    :func:`_beta_function_incomplete_cf`.  When
    ``x < (a+1)/(a+b+2)`` the direct form is used; otherwise the
    symmetry relation ``I_x(a,b) = 1 - I_{1-x}(b,a)`` is applied
    (Fortran lines 446-450).

    Args:
        a: First shape parameter.
        b: Second shape parameter.
        x: Evaluation point (0 ≤ x ≤ 1).

    Returns:
        ``F(x; a, b)``.
    """
    # Fortran lines 440-444
    if x == 0.0 or x == 1.0:
        bt = 0.0
    else:
        bt = jnp.exp(
            log_gamma_function(a + b)
            - log_gamma_function(a)
            - log_gamma_function(b)
            + a * jnp.log(x)
            + b * jnp.log(1.0 - x)
        )

    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_function_incomplete_cf(a, b, x) / a
    else:
        return 1.0 - bt * _beta_function_incomplete_cf(b, a, 1.0 - x) / b


# ---------------------------------------------------------------------------
# Private: incomplete beta continued fraction
# ---------------------------------------------------------------------------


def _beta_function_incomplete_cf(a: float, b: float, x: float) -> float:
    """
    Evaluate the continued fraction representation of the incomplete
    beta function.

    Mirrors Fortran function ``beta_function_incomplete_cf`` (private,
    lines 457-510).

    Uses the modified Lentz algorithm with ``maxit = 100``,
    ``eps = 3e-7``, and ``fpmin = 1e-30``.

    Args:
        a: First shape parameter.
        b: Second shape parameter.
        x: Evaluation point.

    Returns:
        The continued fraction value ``betacf``.
    """
    maxit: int = 100  # Fortran: parameter maxit = 100
    eps: float = 3.0e-7  # Fortran: parameter eps = 3.e-07_r8
    fpmin: float = 1.0e-30  # Fortran: parameter fpmin = 1.e-30_r8

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, maxit + 1):  # Fortran: do m = 1, maxit
        m2 = 2 * m
        # Even step — Fortran lines 487-494
        aa = float(m) * (b - float(m)) * x / ((qam + float(m2)) * (a + float(m2)))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h = h * d * c
        # Odd step — Fortran lines 495-503
        aa = -(a + float(m)) * (qab + float(m)) * x / ((qap + float(m2)) * (a + float(m2)))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h = h * delta
        if abs(delta - 1.0) < eps:
            return h

    endrun(msg=" ERROR: beta_function_incomplete_cf error")
    return h  # Unreachable; satisfies type checker
