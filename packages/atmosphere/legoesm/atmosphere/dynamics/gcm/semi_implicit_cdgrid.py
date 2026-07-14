"""Hoskins–Simmons (1975) semi-implicit FV3 D-grid solver core.

Phase 3 of the issue #273 throughput work.  Builds an FV-adjoint-
symmetric cubed-sphere Laplacian by restricting the halo exchange to a
nearest-index copy (no Lagrange interpolation, no duogrid remap), so
that the discrete identity

    ⟨grad p, u⟩_face = −⟨p, div u⟩_cell

holds across cube panel seams.  Composing this gradient with the
existing FV ``cgrid_divergence`` then yields a Laplacian that is

* **symmetric** under the area-weighted inner product
  ``⟨p, q⟩_M = Σ area · p · q``     (residual ≤ 1e-12 in fp64), and
* **negative semi-definite** under that inner product, so
  ``A(p) = p − coeff · ∇² p`` with ``coeff ≥ 0`` is strictly
  M-positive-definite.

For ``jax.scipy.sparse.linalg.cg`` (which uses the Euclidean inner
product) we apply the standard mass-weighted shim

    B = M^{1/2} · A · M^{−1/2},      tilde_p = M^{1/2} · p,

so that ``B`` is Euclidean-SPD and ``cg`` returns the correct
``p = M^{−1/2} tilde_p``.  The shim is wrapped inside
``cg_helmholtz_solve`` — callers do not see the transformation.

Module API
----------
* ``cdgrid_scalar_laplacian(p, cdgrid)`` — FV-adjoint-symmetric
  ``div(grad p)`` on the cubed-sphere D-grid.
* ``adjoint_residual_norm(L, cdgrid, n_samples)`` — area-weighted
  adjoint-residual diagnostic.  Drops to machine zero on the new
  operator; the canary regression test fails if the asymmetric halo
  interpolation ever creeps back in.
* ``make_helmholtz_op(coeff, cdgrid)`` — returns
  ``A(p) = p − coeff · ∇² p`` (M-SPD; CG-ready via
  ``cg_helmholtz_solve``).
* ``cg_helmholtz_solve(rhs, coeff, cdgrid, ...)`` — Preconditioned-
  free CG with the mass-weighted Euclidean shim; differentiable via
  ``jax.scipy.sparse.linalg.cg``'s built-in implicit-function-
  theorem VJP.
* ``richardson_helmholtz_solve`` — retained as a fallback / reference
  backend; the production wiring in ``_step_fv3`` uses CG.

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

from legoesm.core.operators_cdgrid import cgrid_divergence
from legoesm.grids.halo import pad_halo


__all__ = (
    "cdgrid_scalar_laplacian",
    "adjoint_residual_norm",
    "make_helmholtz_op",
    "cg_helmholtz_solve",
    "richardson_helmholtz_solve",
)


def _pad_halo_nearest_copy(field):
    """Halo exchange via nearest-index copy only.

    The standard ``pad_halo_auto`` path in ``operators_cdgrid``
    optionally applies a 3-point Lagrange interpolation
    (``interp_offsets``) or a duogrid remap when the source panel
    is not aligned with the destination halo strip.  Both options
    introduce a non-transposable linear combination of source-panel
    cells, which breaks the FV adjoint identity
    ``⟨grad p, u⟩_face = −⟨p, div u⟩_cell`` at cube panel seams.

    For the symmetric Helmholtz operator we want the halo to be a
    *pure transpose-friendly gather*: each halo cell receives a single
    source-panel cell value.  ``pad_halo(..., interp_offsets=None,
    duogrid=None)`` already implements that nearest-index gather, so
    we just call it directly.
    """
    return pad_halo(field, interp_offsets=None, duogrid=None)


def cdgrid_scalar_laplacian(p, cdgrid):
    """FV-adjoint-symmetric ``∇² p = div(grad p)`` on the cubed-sphere
    D-grid.

    Composes a nearest-copy-halo FV gradient (cell-centre → edge
    midpoints) with the standard FV ``cgrid_divergence`` (edge
    midpoints → cell-centre).  The composition is

    * **symmetric** under ``⟨p, q⟩_M = Σ area · p · q`` to machine
      precision (~1e-12 in fp64), and
    * **negative semi-definite** under the same inner product.

    Parameters
    ----------
    p : (6, n, n) jax.Array
        Scalar field at cell centres.
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    jax.Array, same shape as ``p``.

    Notes
    -----
    The metric arrays (``cdgrid.rdxc``, ``cdgrid.rdyc``,
    ``cdgrid.dy_edge_x``, ``cdgrid.dx_edge_y``) inherit their dtype
    from the active ``PrecisionPolicy``.  Running this operator at
    f32 cell-edge metrics floors the adjoint-residual diagnostic at
    ~1e-7 and the CG residual at the same scale — fine for f32
    state, but call sites that need a tight implicit solve should
    set ``PrecisionPolicy.fp64()`` before constructing the grid.
    """
    eta_pad = _pad_halo_nearest_copy(p)
    grad_x = (eta_pad[:, 1:, 1:-1] - eta_pad[:, :-1, 1:-1]) * cdgrid.rdxc
    grad_y = (eta_pad[:, 1:-1, 1:] - eta_pad[:, 1:-1, :-1]) * cdgrid.rdyc
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

    On the FV-adjoint-symmetric ``cdgrid_scalar_laplacian`` this
    drops to ~1e-14 in fp64 (machine zero).  Used in two ways:

    1. **Operator regression canary** — the Phase-3 unit test
       asserts ``residual < 1e-10`` so any reintroduction of an
       interpolating halo on the symmetric Laplacian path trips CI.
    2. **CG-readiness diagnostic** — measured on the operator
       *actually passed to* ``cg`` (after the mass-weighted shim
       in ``cg_helmholtz_solve``); see warning below.

    .. warning::
       This is an **FV-adjoint diagnostic**, not an SPD-readiness
       check for ``jax.scipy.sparse.linalg.cg``.  ``cg`` uses the
       *Euclidean* dot product on the flattened array, not the
       area-weighted inner product.  An operator that is M-symmetric
       only is not Euclidean-symmetric unless cell areas are
       uniform.  ``cg_helmholtz_solve`` therefore applies a
       ``M^{1/2}·A·M^{-1/2}`` shim that yields a Euclidean-SPD
       operator; the shim is verified Euclidean-symmetric and
       positive-definite by dedicated unit tests.
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
    rng = np.random.default_rng(seed)
    v = jnp.asarray(rng.standard_normal(shape))
    v = v / (jnp.linalg.norm(v.ravel()) + 1.0e-30)

    def _step(carry, _):
        u = op(carry)
        norm = jnp.linalg.norm(u.ravel()) + 1.0e-30
        return u / norm, None

    v, _ = jax.lax.scan(_step, v, None, length=n_iter)
    return jnp.linalg.norm(op(v).ravel())


def make_helmholtz_op(coeff, cdgrid) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return ``A(p) = p − coeff · ∇² p`` on the cubed-sphere D-grid.

    The underlying ``cdgrid_scalar_laplacian`` is FV-adjoint-symmetric
    + negative semi-definite under the area-weighted inner product
    ``⟨p, q⟩_M = Σ area · p · q``, so ``A`` is **M-positive-definite**
    for ``coeff ≥ 0``.

    For direct use with ``jax.scipy.sparse.linalg.cg`` (Euclidean
    inner product), wrap with the mass-weighted shim in
    ``cg_helmholtz_solve`` rather than passing ``A`` directly.

    Parameters
    ----------
    coeff : float | jax.Array
        ``α · dt`` in the diffusion form, or the equivalent
        scaling in the full Hoskins–Simmons Helmholtz.  May be a
        traced JAX scalar.
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    A : callable
        ``A(p) = p − coeff · cdgrid_scalar_laplacian(p, cdgrid)``.
    """

    def A(p):
        return p - coeff * cdgrid_scalar_laplacian(p, cdgrid)

    return A


def cg_helmholtz_solve(
    rhs,
    coeff,
    cdgrid,
    *,
    tol: float = 1.0e-10,
    maxiter: int = 200,
    return_residual: bool = False,
):
    """Solve ``(I − coeff · ∇²) p = rhs`` on the cubed-sphere D-grid
    via ``jax.scipy.sparse.linalg.cg`` with a mass-weighted Euclidean
    shim.

    Phase 3 production backend for the issue #273 Hoskins–Simmons
    FV3 D-grid port.  Supersedes ``richardson_helmholtz_solve``:

    * **Quadratic convergence** (vs Richardson's linear) — reaches
      ``rel_res < 1e-10`` in O(10) iterations at α dt / dx² ≤ 5,
      vs Richardson's O(100) at tol=1e-6.
    * **Reverse-mode AD compatible** — ``jax.scipy.sparse.linalg.cg``
      installs an implicit-function-theorem VJP, so ``jax.grad``
      flows cleanly through the solve without needing a manual
      ``custom_vjp`` wrapper.

    Mass-weighted shim
    ------------------
    ``cdgrid_scalar_laplacian`` is symmetric under
    ``⟨p, q⟩_M = Σ area · p · q``, not under the Euclidean dot
    product that ``cg`` uses.  We solve the transformed system

        B · tilde_p = sqrt(area) · rhs,   with
        B(tilde_p)  = tilde_p − coeff · sqrt(area) · ∇²(tilde_p / sqrt(area))
                    = M^{1/2} · A · M^{-1/2} · tilde_p,

    which is Euclidean-symmetric *and* positive-definite for
    ``coeff ≥ 0``.  The returned solution is
    ``p = tilde_p / sqrt(area)``, which exactly satisfies
    ``A p = rhs`` (up to CG convergence tolerance).

    Parameters
    ----------
    rhs : (6, n, n) jax.Array
        Right-hand side ``b``.  Used as warm start.
    coeff : float | jax.Array
        ``α · dt`` in the diffusion form.
    cdgrid : CubedSphereCDGrid
    tol : float, keyword-only
        ``cg`` relative-residual tolerance.  Default 1e-10.
    maxiter : int, keyword-only
        Hard cap on CG iterations.  Default 200.
    return_residual : bool, keyword-only
        When True, returns ``(p, rel_res)`` instead of just ``p``;
        ``rel_res`` is the externally verified relative residual
        ``‖rhs − A p‖ / ‖rhs‖`` (a traced scalar).  Useful for
        downstream JIT-safe convergence-aware fallbacks.

    Returns
    -------
    p : jax.Array
        Approximate solution.
    rel_res : jax.Array, optional
        Final externally verified relative residual.
    """
    area = cdgrid.base.area.astype(rhs.dtype)
    sqrt_area = jnp.sqrt(area)
    inv_sqrt_area = 1.0 / sqrt_area

    def B_op(tilde_p):
        p = inv_sqrt_area * tilde_p
        return tilde_p - coeff * sqrt_area * cdgrid_scalar_laplacian(
            p, cdgrid,
        )

    tilde_rhs = sqrt_area * rhs
    tilde_x0 = sqrt_area * rhs  # warm start at rhs in physical space
    # NOTE (pinned jax==0.10.1): ``cg`` takes ``tol=`` (verified honored by
    # test_residual_meets_production_tolerance).  A LATER jax renames it to
    # ``rtol=`` and emits a DeprecationWarning; when bumping jax, switch this
    # keyword to ``rtol=tol`` (``rtol=`` raises TypeError on 0.10.1 today).
    tilde_sol, _info = jax.scipy.sparse.linalg.cg(
        B_op, tilde_rhs, x0=tilde_x0, tol=tol, maxiter=maxiter,
    )
    sol = inv_sqrt_area * tilde_sol

    if return_residual:
        # External verification in *physical* space so the reported
        # residual is the quantity the caller cares about, not the
        # M^{1/2}-transformed one.
        def A_phys(p):
            return p - coeff * cdgrid_scalar_laplacian(p, cdgrid)

        rhs_norm = jnp.maximum(
            jnp.linalg.norm(rhs.ravel()), 1.0e-30,
        )
        rel_res = jnp.linalg.norm(
            (rhs - A_phys(sol)).ravel(),
        ) / rhs_norm
        return sol, rel_res
    return sol


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

    Retained from Phase 2.  Operates on the **base** ``CubedSphereGrid``
    via ``laplacian_compact`` (which is *non-symmetric*) — see
    ``cg_helmholtz_solve`` for the Phase-3 production backend that
    uses the FV-adjoint-symmetric ``cdgrid_scalar_laplacian`` and
    ``jax.scipy.sparse.linalg.cg``.

    Iteration:

      ``p^{k+1} = p^k + τ · (p_explicit − (I − coeff · ∇²) p^k)``

    Parameters
    ----------
    p_explicit : jax.Array
        Right-hand side ``b = p_explicit``.  Used as warm start.
    coeff : float | jax.Array
        ``α · dt`` in the diffusion form.
    grid : CubedSphereGrid
    n_iter_max : int, keyword-only
        Hard cap on Richardson iterations.  Default 60.
    tol : float, keyword-only
        Stop early when ``‖b − A p^k‖ / ‖b‖`` falls below this
        value.  Default 1e-6.
    tau : float, keyword-only
        Override the auto-computed damping.  When ``None``
        (default), the damping is
        ``τ = 1 / (1 + 1.5 · coeff · |λ_max(∇²)|)``.
    return_residual : bool, keyword-only
        When True, returns ``(p_new, rel_res)``.

    Returns
    -------
    p_new : jax.Array
    rel_res : jax.Array, optional
    """
    from legoesm.core.operators import laplacian_compact

    def L(p):
        return laplacian_compact(p, grid)

    if tau is None:
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

    init_residual = _residual_norm(p_explicit)
    init_state = (
        p_explicit,
        init_residual,
        jnp.int32(0),
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
