"""Variable-coefficient matrix-free pressure Poisson for the pseudo-incompressible
plane LES — the GPU/TPU-scalable alternative to the spectral core's all-to-all FFT.

The pseudo-incompressible constraint (Durran 1989; LEX, GMD 2026, ``solver_opt=1``)
enforces ``∇·(ρ0θ0 u) = 0`` each RK stage by solving an elliptic equation for the
Exner-function perturbation ``π'``::

    Cp · ∇·( C ∇π' ) = R                                              (LHS = L(π'))

with the variable coefficient ``C = rtt = ρ0θ0·θ·(1+reps·q_v)/(1+q_v)`` (the moist
density potential-temperature weighting). Horizontal is periodic; the vertical has
rigid-lid / ground walls (``w=0``).

Why matrix-free + BiCGSTAB (vs the spectral ``project()``)
---------------------------------------------------------
The operator :func:`laplace_pi` is a **nearest-neighbour 7-point stencil** — its only
communication is a one-cell halo exchange per matvec, so it weak-scales on a GPU/TPU
device mesh. The spectral projection inverts the Laplacian with an FFT, i.e. a global
all-to-all every step, which is the mesh-scaling bottleneck this core removes. We solve
the (symmetric, negative-definite) system with ``jax.scipy.sparse.linalg.bicgstab`` and a
**Jacobi (diagonal) preconditioner** :func:`poisson_diag` — matrix-free, AD-safe, and a
large iteration-count cut on the variable-``C`` (moist) operator that plain BiCGSTAB
(LEX's choice) leaves slow on large/stretched grids.

Fidelity note (deliberate, disclosed)
-------------------------------------
LEX's ``laplace_of_pressure`` writes the *prescribed* wall buoyancy-flux into the
operator's wall faces, which makes the operator **affine** (``L(π')+const``) — unsound
for a Krylov method, which assumes linearity. Here the operator is a **proper linear**
variable-coefficient Laplacian with **homogeneous-Neumann** walls (zero π'-flux at the
ground/lid); the prescribed buoyancy/forcing Neumann data belongs on the RHS ``R`` (the
caller assembles it). This is faithful to the *scheme* (the elliptic projection that makes
the ρ-weighted velocity divergence-free) and correct for the solver. With periodic-x/y +
Neumann-z the operator has a one-dimensional constant null space, so the gauge is pinned
(``π'`` mean removed) and the RHS is made compatible (its mean removed) before the solve.

Layout: physical ``(ny, nx, nz)`` (y axis 0, x axis 1, z axis 2), matching
``spectral_les_plane`` / ``compressible_euler_plane``. Uniform ``Δx, Δy, Δz``. ``π'`` and
``C`` are cell-centred ``(ny, nx, nz)`` arrays carrying NO ghost layers (periodic
neighbours are taken with ``jnp.roll``; wall neighbours never enter, by the zero-flux BC).

Pure-pytree, JIT- and ``jax.grad``-safe (no Python control flow on traced values, no host
callbacks). MPI: replace the ``jnp.roll`` halos with ``parallel.halo_exchange`` strips
when distributing — the stencil and diagonal are unchanged (follow-up; see runbook
``docs/physics-notes/pseudo_incompressible_les.md``).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants

# Axis convention for the physical (ny, nx, nz) plane layout.
_AY, _AX, _AZ = 0, 1, 2


def _face_coeff_x(c):
    """C at the i+½ x-faces, ``0.5·(C_i + C_{i+1})`` (periodic in x)."""
    return 0.5 * (c + jnp.roll(c, -1, axis=_AX))


def _face_coeff_y(c):
    """C at the j+½ y-faces, ``0.5·(C_j + C_{j+1})`` (periodic in y)."""
    return 0.5 * (c + jnp.roll(c, -1, axis=_AY))


def _face_coeff_z(c):
    """C at the INTERIOR k+½ z-faces (length nz-1 along z); walls excluded."""
    return 0.5 * (c[..., :-1] + c[..., 1:])


def laplace_pi(pi, c, dx, dy, dz, cp=constants.c_pd):
    """Matrix-free LHS ``Cp · ∇·( C ∇π' )`` (negative-definite), 7-point C-grid stencil.

    Periodic in x, y (``jnp.roll`` neighbours); homogeneous-Neumann (zero-flux) walls in
    z. ``pi`` and ``c`` are ``(ny, nx, nz)`` with no ghosts. Returns ``(ny, nx, nz)``.
    """
    # x: flux F^x_{i+½} = C_{i+½}·(π_{i+1} − π_i)/Δx ; ∂_x F at centre = (F_{i+½}−F_{i−½})/Δx
    cx = _face_coeff_x(c)
    fx = cx * (jnp.roll(pi, -1, axis=_AX) - pi) / dx
    lap_x = (fx - jnp.roll(fx, 1, axis=_AX)) / dx
    # y
    cy = _face_coeff_y(c)
    fy = cy * (jnp.roll(pi, -1, axis=_AY) - pi) / dy
    lap_y = (fy - jnp.roll(fy, 1, axis=_AY)) / dy
    # z with zero-flux walls: interior faces only; wall faces contribute 0.
    cz = _face_coeff_z(c)                                   # (ny, nx, nz-1)
    fz_int = cz * (pi[..., 1:] - pi[..., :-1]) / dz          # interior k+½ faces
    # Pad a zero flux at the bottom (below k=0) and top (above k=nz-1) walls, then
    # difference: lap_z[k] = (F_{k+½} − F_{k−½})/Δz with F_wall = 0.
    fz = jnp.pad(fz_int, [(0, 0), (0, 0), (1, 1)])          # (ny, nx, nz+1)
    lap_z = (fz[..., 1:] - fz[..., :-1]) / dz
    return cp * (lap_x + lap_y + lap_z)


def poisson_diag(c, dx, dy, dz, cp=constants.c_pd):
    """Diagonal ``∂L(π')_m/∂π'_m`` of :func:`laplace_pi` for Jacobi preconditioning.

    For the centred face stencil the self-coupling at cell ``m`` is
    ``−Cp·[ (C_{i+½}+C_{i−½})/Δx² + (C_{j+½}+C_{j−½})/Δy² + (C_{k+½}+C_{k−½})/Δz² ]``
    with the wall faces (zero-flux) dropped from the z sum at k=0, nz-1.

    Run in float64 (``JAX_ENABLE_X64=1``; the LES default). On high-aspect-ratio LES
    grids (Δz ≪ Δx) the z term dominates this sum by orders of magnitude, so the
    ``1/diag`` Jacobi factor loses accuracy under float32 cancellation and weakens the
    preconditioner.
    """
    cx_p = _face_coeff_x(c)                                  # C_{i+½}
    cx_m = jnp.roll(cx_p, 1, axis=_AX)                       # C_{i−½}
    cy_p = _face_coeff_y(c)
    cy_m = jnp.roll(cy_p, 1, axis=_AY)
    cz_int = _face_coeff_z(c)                                # interior k+½ (nz-1)
    cz_full = jnp.pad(cz_int, [(0, 0), (0, 0), (1, 1)])      # 0 at the two walls
    cz_p = cz_full[..., 1:]                                  # C_{k+½}, 0 at top wall
    cz_m = cz_full[..., :-1]                                 # C_{k−½}, 0 at bottom wall
    diag = -((cx_p + cx_m) / dx ** 2
             + (cy_p + cy_m) / dy ** 2
             + (cz_p + cz_m) / dz ** 2)
    return cp * diag


def _zero_mean(f):
    """Remove the global mean (gauge fix / RHS compatibility for the singular system)."""
    return f - jnp.mean(f)


def solve_pressure(rhs, c, dx, dy, dz, x0=None, cp=constants.c_pd,
                   tol=1e-6, atol=1e-10, maxiter=200, precondition=True):
    """Solve ``Cp·∇·(C ∇π') = rhs`` for ``π'`` (zero-mean gauge) with Jacobi-BiCGSTAB.

    Returns ``(pi, info)`` like ``jax.scipy.sparse.linalg.bicgstab`` (``info==0`` ⇒
    converged). The RHS mean is removed for solvability; pass ``x0`` (the previous
    step's ``π'``) to warm-start. Set ``precondition=False`` for the unconditioned
    baseline (used by the test that proves the Jacobi speedup).
    """
    rhs_c = _zero_mean(rhs)

    def A(p):
        return laplace_pi(p, c, dx, dy, dz, cp=cp)

    M = None
    if precondition:
        inv_diag = 1.0 / poisson_diag(c, dx, dy, dz, cp=cp)

        def M(r):                                           # Jacobi: M⁻¹ r = r/diag
            return inv_diag * r

    pi, info = jax.scipy.sparse.linalg.bicgstab(
        A, rhs_c, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)
    return _zero_mean(pi), info
