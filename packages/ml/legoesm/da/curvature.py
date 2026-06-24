"""Matrix-free second-order AD: Hessian / Gauss-Newton products, spectra, Laplace covariance.

For derivative-based uncertainty quantification and Gauss-Newton inner solves
on any differentiable scalar objective (4D-Var cost, training loss, physics-
parameter misfit).  The products are matrix-free (forward-over-reverse AD) so
they scale to large control/state vectors without ever forming the Hessian;
dense helpers are provided for small parameter vectors (the physics-calibration
case, ~10-100 params) where forming H is cheap and an exact spectrum/covariance
is wanted.

All operators work on arbitrary pytrees (flat control arrays *or* structured
parameter pytrees) via :func:`flat_operator`, which is the bridge to the
flat-array Lanczos / dense routines.
"""
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def hvp(f: Callable, x, v):
    """Hessian-vector product ``H(x) v`` via forward-over-reverse AD.

    Pytree in / pytree out; ``v`` must match the structure of ``x``.  Never
    forms H, so the cost is ~2 gradient evaluations regardless of dimension.
    """
    return jax.jvp(jax.grad(f), (x,), (v,))[1]


def gauss_newton_hvp(residual_fn: Callable, x, v, r_inv=None):
    """Gauss-Newton Hessian-vector product ``J^T R^{-1} J v`` (symmetric PSD), matrix-free.

    ``J = d residual_fn / d x`` at ``x``.  This is the Hessian of
    ``1/2 r^T R^{-1} r`` with the second-derivative-of-residual term dropped —
    the standard Gauss-Newton/Fisher approximation, always PSD, ideal for an
    inner CG solve or a Laplace posterior precision.

    Parameters
    ----------
    residual_fn : callable
        ``x -> residual`` (array or pytree).
    x, v : pytree
        Linearisation point and tangent (same structure).
    r_inv : None | array | callable
        Observation precision ``R^{-1}``.  ``None`` => identity; an array is
        multiplied elementwise into ``J v`` (diagonal R for an array residual);
        a callable is applied to the ``J v`` residual pytree.
    """
    r, jv = jax.jvp(residual_fn, (x,), (v,))
    if r_inv is None:
        w = jv
    elif callable(r_inv):
        w = r_inv(jv)
    else:
        w = jax.tree.map(lambda a: r_inv * a, jv)
    _, vjp_fn = jax.vjp(residual_fn, x)
    return vjp_fn(w)[0]


def flat_operator(op_at_x: Callable, x):
    """Bridge a pytree matrix-free operator to a flat ``(dim,)->(dim,)`` matvec.

    ``op_at_x`` is a 1-argument operator already linearised at ``x`` (e.g.
    ``lambda v: hvp(f, x, v)`` or ``lambda v: gauss_newton_hvp(res, x, v)``).
    Returns ``(matvec, dim, unravel)`` so :func:`lanczos_eigvalsh` /
    :func:`dense_from_matvec` can run on a structured control pytree.
    """
    x_flat, unravel = ravel_pytree(x)

    def matvec(v_flat):
        hv = op_at_x(unravel(v_flat))
        hv_flat, _ = ravel_pytree(hv)
        return hv_flat

    return matvec, int(x_flat.shape[0]), unravel


def dense_hessian(f: Callable, x):
    """Dense Hessian ``(n, n)`` of scalar ``f`` at a 1-D array ``x``.

    For small parameter vectors only (forming H is O(n^2) memory); ravel a
    pytree with :func:`jax.flatten_util.ravel_pytree` first.
    """
    x = jnp.asarray(x)
    if x.ndim != 1:
        raise ValueError("dense_hessian requires a 1-D array; ravel pytrees first.")
    return jax.hessian(f)(x)


def dense_from_matvec(matvec: Callable, dim: int):
    """Materialise a ``(dim, dim)`` matrix from a flat symmetric matvec (small dim only)."""
    return jax.vmap(matvec)(jnp.eye(dim, dtype=jnp.result_type(float)))


def lanczos_eigvalsh(matvec: Callable, dim: int, key, *, n_iter: int = 20):
    """Approximate extremal eigenvalues of a symmetric matrix-free operator via Lanczos.

    ``matvec`` maps ``(dim,)->(dim,)`` (e.g. a flattened ``hvp`` or
    ``gauss_newton_hvp``).  Runs ``m = min(n_iter, dim)`` Lanczos iterations
    with full reorthogonalisation and returns the ``m`` Ritz values (ascending).
    The largest approximate the stiff (well-constrained) curvature directions,
    the smallest the sloppy ones — the spectrum used for identifiability / UQ.
    """
    m = int(min(n_iter, dim))
    dtype = jnp.result_type(float)
    v = jax.random.normal(key, (dim,), dtype=dtype)
    v = v / jnp.linalg.norm(v)

    V = [v]
    alphas: list = []
    betas: list = []
    w = matvec(v)
    alpha = jnp.vdot(w, v).real
    alphas.append(alpha)
    w = w - alpha * v
    for _ in range(1, m):
        # full reorthogonalisation (m is small) for numerical stability
        for u in V:
            w = w - jnp.vdot(u, w) * u
        beta = jnp.linalg.norm(w)
        # guard against a (near-)zero residual / invariant subspace
        v_next = jnp.where(beta > 1e-30, w / jnp.where(beta > 1e-30, beta, 1.0), v)
        betas.append(beta)
        V.append(v_next)
        w = matvec(v_next)
        alpha = jnp.vdot(w, v_next).real
        alphas.append(alpha)
        w = w - alpha * v_next - beta * V[-2]

    a = jnp.stack(alphas)
    if betas:
        b = jnp.stack(betas)
        T = jnp.diag(a) + jnp.diag(b, 1) + jnp.diag(b, -1)
    else:
        T = jnp.diag(a)
    return jnp.linalg.eigvalsh(T)


def laplace_posterior_cov(hessian, prior_precision=None):
    """Laplace posterior covariance ``(H + P)^{-1}`` for a dense Hessian/GN matrix.

    ``H`` is the (Gauss-Newton) Hessian of the negative log-likelihood at the
    optimum; ``prior_precision`` ``P`` is ``None`` (flat prior), a scalar
    (isotropic ``P = p I``), or a dense ``(n, n)`` precision (e.g. ``B^{-1}``).
    Small-dimension only — inverts an ``(n, n)`` matrix.
    """
    H = jnp.asarray(hessian)
    n = H.shape[0]
    if prior_precision is None:
        P = jnp.zeros_like(H)
    elif jnp.ndim(prior_precision) == 0:
        P = prior_precision * jnp.eye(n, dtype=H.dtype)
    else:
        P = jnp.asarray(prior_precision)
    return jnp.linalg.inv(H + P)
