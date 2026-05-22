"""Hoskins–Simmons (1975) semi-implicit FV3 D-grid scaffolding.

Phase 2 of the issue #273 throughput work.  Phase 1 (already merged
in commit ``f957a8be``) added the ``CDGridPrimitiveEquationConfig.
implicit_grav_wave_use_pcg`` flag that currently raises
``NotImplementedError``.  Phase 2 lands the building blocks that
the production wiring needs:

1. ``cdgrid_scalar_laplacian(p_s, cdgrid)`` — single canonical
   FV ``div(grad)`` composition on the cubed-sphere D-grid.
2. ``adjoint_residual_norm(L, cdgrid, n_samples)`` — measures how
   far the operator is from a discrete FV adjoint pair under the
   area-weighted inner product.  Phase 2 should drive this towards
   machine zero before wiring the CG solve.
3. ``make_helmholtz_op(coeff, cdgrid)`` — returns
   ``A(p) = p − coeff · ∇²p`` for use with
   ``jax.scipy.sparse.linalg.cg`` once the underlying ``∇²`` is
   symmetric.

Current ``cdgrid_scalar_laplacian`` is *not* symmetric across cube
panel seams — empirical asymmetry under the area-weighted inner
product is ~13 % on a random IC at n = 8.  The blocker is the halo
interpolation: the discrete identity
``⟨grad p, u⟩_face = −⟨p, div u⟩_cell``
holds in the interior of a face but breaks at the corner-stitched
seams because the halo on either side of a seam is constructed by
interpolation that has no transposed counterpart in the divergence
stencil.  Two paths to a proper FV adjoint pair:

* Re-derive the halo interpolation as the transpose of the
  divergence stencil; rebuild ``cgrid_gradient_2d`` against it.
* Construct the gradient + divergence directly from the dual-mesh
  stencils of the FV cubed-sphere connectivity (the same approach
  ``barotropic_implicit_latlon_cgrid._make_helmholtz`` uses on lat-
  lon C-grid: gradient at u/v faces × face length, divergence by
  net face-flux ÷ cell area).

This module is opt-in: nothing in production imports it yet.  Tests
under ``tests/atmosphere/hydrostatic/unit/test_semi_implicit_cdgrid.py``
exercise the building blocks and assert the current adjoint-residual
is finite (Phase 2 lowers the threshold once the operator is fixed).

References
----------
* Hoskins B., Simmons A. (1975) "A multi-layer spectral model and
  the semi-implicit method", Quart. J. Roy. Met. Soc., 101, 637-655.
* ``legoesm.timestepping.semi_implicit`` — spectral analogue we are
  porting from.
* ``legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid`` —
  reference C-grid FV adjoint-pair Helmholtz template.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.operators_cdgrid import (
    cgrid_divergence,
    cgrid_gradient_2d,
)


__all__ = (
    "cdgrid_scalar_laplacian",
    "adjoint_residual_norm",
    "make_helmholtz_op",
    "richardson_helmholtz_solve",
)


def cdgrid_scalar_laplacian(p, cdgrid):
    """``∇² p = div(grad p)`` on the cubed-sphere D-grid.

    Composes the existing FV gradient ``cgrid_gradient_2d`` (cell-
    centre → edge midpoints) with the FV divergence
    ``cgrid_divergence`` (edge midpoints → cell-centre).

    Parameters
    ----------
    p : (6, n, n) or (6, n, n, nlev) jax.Array
        Scalar field at cell centres.
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    jax.Array, same shape as ``p``.

    Notes
    -----
    NOT symmetric across cube panel seams under the area-weighted
    inner product — see module docstring for context.  Use only
    where the FV adjoint property is not required (e.g. legacy
    explicit forward-Euler diffusion at small ``α dt / dx²``).
    """
    grad_x, grad_y = cgrid_gradient_2d(p, cdgrid)
    return cgrid_divergence(grad_x, grad_y, cdgrid)


def adjoint_residual_norm(
    L: Callable[[jnp.ndarray], jnp.ndarray],
    cdgrid,
    n_samples: int = 4,
    seed: int = 0,
) -> float:
    """Quantify how far ``L`` is from a symmetric operator under the
    area-weighted inner product ``⟨p, q⟩ = Σ area · p · q``.

    Returns the maximum relative asymmetry
    ``max_i |⟨x_i, L y_i⟩ − ⟨L x_i, y_i⟩|
           / max(|⟨x_i, L y_i⟩|, |⟨L x_i, y_i⟩|)``
    over ``n_samples`` random (x, y) pairs.

    A symmetric operator (under this inner product) yields a value
    near machine precision (~1e-14 in fp64).  The current
    ``cdgrid_scalar_laplacian`` yields ~0.13 on a (6, 8, 8) grid —
    well above machine zero — because the halo interpolation breaks
    the FV adjoint identity at cube panel seams.

    .. warning::
       This is an **FV-adjoint diagnostic**, not an SPD-readiness
       check for ``jax.scipy.sparse.linalg.cg``.  ``cg`` uses the
       *Euclidean* dot product on the flattened array, not the
       area-weighted inner product.  An operator that is self-
       adjoint only under ``⟨p, q⟩ = Σ area · p · q`` is not
       Euclidean-symmetric unless the cell areas are uniform.
       Phase 2 readiness for ``cg`` therefore needs either
       (a) a mass-weighted transform ``M^{1/2} A M^{-1/2}`` (where
       ``M = diag(area)``) that yields a Euclidean-SPD operator, or
       (b) a re-derivation of the FV stencils on a uniform-area
       dual mesh so the two inner products coincide.

    Phase 2 success criteria (CG-readiness gate):
    * area-weighted adjoint residual < 1e-10, AND
    * Euclidean adjoint residual < 1e-10 on the operator actually
      passed to ``cg`` (or a documented mass-transform shim that
      makes the Euclidean-symmetric operator available), AND
    * **positive-definiteness sanity** — the minimum sampled
      Rayleigh quotient
      ``min_i ⟨x_i, A x_i⟩ / ⟨x_i, x_i⟩``
      over random ``x_i`` is strictly positive on the operator
      passed to ``cg``.  Symmetry alone is not enough — a sign
      error in the Laplacian, a wrong-sign coefficient, or an
      over-aggressive mass transform can leave the operator
      indefinite while still passing both residual thresholds.
      ``cg`` on an indefinite operator can stagnate or return
      meaningless iterates with no warning.
    """
    n = cdgrid.base.n
    area = cdgrid.base.area  # (6, n, n)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_samples):
        x = jnp.asarray(rng.standard_normal((6, n, n)))
        y = jnp.asarray(rng.standard_normal((6, n, n)))
        Lx = L(x)
        Ly = L(y)
        xLy = float(jnp.sum(area * x * Ly))
        yLx = float(jnp.sum(area * y * Lx))
        denom = max(abs(xLy), abs(yLx), 1.0e-30)
        worst = max(worst, abs(xLy - yLx) / denom)
    return worst


def _power_iteration_spectral_radius(
    op: Callable[[jnp.ndarray], jnp.ndarray],
    shape,
    n_iter: int = 20,
    seed: int = 0,
) -> jnp.ndarray:
    """Estimate spectral radius of ``op`` via power iteration.

    Returns ``|λ_max(op)|`` as a traced scalar so the result can be
    used inside JIT to size the Richardson damping factor without
    triggering a recompile per ``coeff``.
    """
    # Deterministic seed via host random; the iteration itself is
    # traced so the result is JIT-friendly.
    rng = np.random.default_rng(seed)
    v = jnp.asarray(rng.standard_normal(shape))
    v = v / (jnp.linalg.norm(v.ravel()) + 1.0e-30)

    def _step(carry, _):
        u = op(carry)
        norm = jnp.linalg.norm(u.ravel()) + 1.0e-30
        return u / norm, None

    v, _ = jax.lax.scan(_step, v, None, length=n_iter)
    return jnp.linalg.norm(op(v).ravel())


def richardson_helmholtz_solve(
    p_explicit,
    coeff,
    grid,
    *,
    n_iter_max: int = 60,
    tol: float = 1.0e-6,
    tau: float | None = None,
    return_residual: bool = False,
):
    """Solve ``(I − coeff · ∇²) p_new = p_explicit`` via damped
    Richardson iteration with residual-monitored early stop.

    Phase 2 first-cut backend for the issue #273 Hoskins–Simmons
    FV3 D-grid port.  Bypasses both blockers identified in Phase 1:

    1. **No SPD requirement** — the ``cdgrid_scalar_laplacian`` is
       non-symmetric (~13 % area-weighted asymmetry), so
       ``jax.scipy.sparse.linalg.cg`` is undefined.  Richardson is
       a fixed-point method.
    2. **No transpose-solve** — Richardson uses only forward
       matrix-vector products, so the
       ``NotImplementedError: scatter transpose is only implemented
       where unique_indices=True`` blocker inside
       ``lax.custom_linear_solve`` does not apply.

    Iteration:

      ``p^{k+1} = p^k + τ · (p_explicit − (I − coeff · ∇²) p^k)``

    .. warning::
       The cubed-sphere ``laplacian_compact`` is *non-normal* (the
       halo interpolation breaks the FV adjoint identity at cube
       panel seams).  For non-normal ``A``, the eigenvalue bound
       ``|λ_max|`` is not sufficient to guarantee
       ``ρ(I − τ A) < 1`` — pseudospectral / transient growth can
       still inflate the iteration norm.  We therefore (a) use a
       conservative damping ``τ ≈ 1 / (1 + 1.5 · coeff · |λ_max|)``
       (safety factor over the optimal-spectral choice
       ``2 / (2 + coeff · |λ_max|)``), and (b) monitor the actual
       residual ``‖p_explicit − A p^k‖`` inside a
       ``lax.while_loop`` that stops as soon as the configured
       ``tol`` is reached or ``n_iter_max`` is hit.  Callers
       requesting ``return_residual=True`` get the final relative
       residual back and can decide whether to skip the damping
       step (the policy used by ``_step_fv3``), retry with a
       smaller coefficient, or use any other stable fallback —
       falling back to the *legacy explicit forward-Euler* update
       at the same ``coeff`` is NOT safe, since that path is
       unstable exactly in the regime where Richardson fails.

    Parameters
    ----------
    p_explicit : jax.Array
        Right-hand side ``b = p_explicit``.  Used as warm start.
    coeff : float | jax.Array
        ``α · dt`` in the diffusion form.  May be a traced JAX
        scalar so ``α · dt`` can change between JIT'd calls
        without re-tracing.
    grid : CubedSphereGrid
        Used by ``laplacian_compact``.
    n_iter_max : int, keyword-only
        Hard cap on Richardson iterations.  Default 60.
    tol : float, keyword-only
        Stop early when ``‖b − A p^k‖ / ‖b‖`` falls below this
        value.  Default 1e-6 (production-grade implicit-solve
        tolerance).
    tau : float, keyword-only
        Override the auto-computed damping.  When ``None``
        (default), the damping is
        ``τ = 1 / (1 + 1.5 · coeff · |λ_max(∇²)|)``
        with ``|λ_max(∇²)|`` estimated via 20-step power
        iteration on ``laplacian_compact``.  The ``1.5`` factor
        is a conservative safety margin over the optimal spectral
        choice ``2 / (2 + coeff · |λ_max|)`` to absorb the non-
        normal pseudospectral inflation.
    return_residual : bool, keyword-only
        When True, returns ``(p_new, rel_res)`` instead of just
        ``p_new``.  ``rel_res`` is the final relative L2 residual,
        a traced scalar suitable for JIT-safe downstream branching.

    Returns
    -------
    p_new : jax.Array
        Approximate solution.
    rel_res : jax.Array, optional
        Final relative residual ``‖b − A p_new‖ / ‖b‖``.  Returned
        only when ``return_residual=True``.

    Notes
    -----
    Phase-3 upgrade path: replace this iteration with
    ``jax.scipy.sparse.linalg.cg`` once
    ``cdgrid_scalar_laplacian`` is made FV-adjoint-symmetric (the
    canary regression test in
    ``tests/atmosphere/hydrostatic/unit/test_semi_implicit_cdgrid.py``
    will fail and force the upgrade to be visible in CI).
    """
    from legoesm.core.operators import laplacian_compact

    def L(p):
        return laplacian_compact(p, grid)

    if tau is None:
        # Conservative safety factor over the optimal spectral
        # choice ``τ_opt = 2 / (2 + coeff · |λ_max(L)|)`` for the
        # diffusion-dominated regime — absorbs pseudospectral
        # inflation from the operator's non-normality without
        # relying on a Hermitian-part bound (which would need the
        # transpose-solve infrastructure that is blocked on this
        # operator; see module docstring).
        lam_max = _power_iteration_spectral_radius(L, p_explicit.shape)
        _tau = 1.0 / (1.0 + 1.5 * coeff * lam_max)
    else:
        _tau = jnp.asarray(tau, dtype=p_explicit.dtype)

    _rhs_norm = jnp.maximum(
        jnp.linalg.norm(p_explicit.ravel()), 1.0e-30,
    )

    def _residual_norm(p):
        return jnp.linalg.norm(
            (p_explicit - (p - coeff * L(p))).ravel(),
        )

    # Initial state: warm start at p_explicit, residual ‖b − A b‖.
    init_residual = _residual_norm(p_explicit)
    init_state = (
        p_explicit,                  # p
        init_residual,               # absolute residual
        jnp.int32(0),                # iteration count
    )

    def _cond(state):
        _p, abs_res, k = state
        return jnp.logical_and(
            abs_res > tol * _rhs_norm,
            k < n_iter_max,
        )

    def _body(state):
        p, _abs_res, k = state
        Ap = p - coeff * L(p)
        residual_field = p_explicit - Ap
        p_new = p + _tau * residual_field
        abs_res_new = _residual_norm(p_new)
        return (p_new, abs_res_new, k + 1)

    p_final, abs_res_final, _k_final = jax.lax.while_loop(
        _cond, _body, init_state,
    )

    if return_residual:
        return p_final, abs_res_final / _rhs_norm
    return p_final


def make_helmholtz_op(
    coeff,
    cdgrid,
    *,
    allow_nonsymmetric_for_testing: bool = False,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return ``A(p) = p − coeff · ∇² p`` on the cubed-sphere D-grid.

    Intended *future* use:
      ``p_new, _ = jax.scipy.sparse.linalg.cg(A, p_explicit, ...)``

    By default this factory **fails closed** with
    ``NotImplementedError`` because the underlying
    ``cdgrid_scalar_laplacian`` is neither FV-adjoint-symmetric (~13
    % area-weighted asymmetry) nor Euclidean-SPD as required by
    ``jax.scipy.sparse.linalg.cg``.  Use the legacy explicit
    forward-Euler diffusion in
    ``CDGridPrimitiveEquationModel._step_fv3`` until Phase 2 (issue
    #273) replaces the operator with a true FV adjoint pair *and*
    documents the Euclidean-SPD shim.

    Parameters
    ----------
    coeff : float | jax.Array
        ``α · dt`` in the diffusion form, or the equivalent
        scaling in the full Hoskins–Simmons Helmholtz.
    cdgrid : CubedSphereCDGrid
    allow_nonsymmetric_for_testing : bool, keyword-only
        Internal escape hatch for unit tests that exercise the
        linearity / coeff=0 identity properties of the factory
        without exercising the (broken) CG path.  Production
        callers must leave this ``False``.

    Returns
    -------
    A : callable
        ``A(p) = p − coeff · cdgrid_scalar_laplacian(p, cdgrid)``.
    """
    if not allow_nonsymmetric_for_testing:
        raise NotImplementedError(
            "make_helmholtz_op returns a Phase 2 scaffold whose "
            "underlying ``cdgrid_scalar_laplacian`` is neither "
            "FV-adjoint-symmetric (~13 % area-weighted asymmetry) "
            "nor Euclidean-SPD as required by "
            "jax.scipy.sparse.linalg.cg.  Production use of CG on "
            "this operator can stagnate or return misleading "
            "iterates with no warning.  See "
            "src/legoesm/atmosphere/dynamics/semi_implicit_cdgrid.py "
            "module docstring for the Phase-2 plan, and the "
            "matching CDGridPrimitiveEquationConfig."
            "implicit_grav_wave_use_pcg field documentation.  Pass "
            "allow_nonsymmetric_for_testing=True only inside the "
            "scaffolding unit tests."
        )

    def A(p):
        return p - coeff * cdgrid_scalar_laplacian(p, cdgrid)

    return A
