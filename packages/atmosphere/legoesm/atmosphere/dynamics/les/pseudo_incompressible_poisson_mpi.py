"""MPI-distributed (y-slab) variable-coefficient pressure Poisson for the
pseudo-incompressible plane LES — the nearest-neighbour, mesh-scalable elliptic solve.

This is the distributed counterpart of :mod:`pseudo_incompressible_poisson`. It proves the
core GPU/TPU-scaling thesis: the matrix-free 7-point Laplacian communicates ONLY a 1-cell
halo with its two y-neighbours per matvec (NOT the spectral core's global all-to-all FFT),
and the BiCGSTAB Krylov inner products are the only global reductions (one ``allreduce`` per
dot, as any distributed Krylov method needs).

Decomposition: y-slab (1D over the y axis, axis 0). Each rank owns ``(ny_local, nx, nz)``;
``ny_global = ny_local · n_ranks``. x is periodic and undecomposed (local ``jnp.roll``); z is
local (vertical). The y direction is periodic across the rank ring (``mpi4jax.sendrecv``).

The serial module is UNCHANGED (no regression). A serial==MPI parity test
(``tests/distributed/test_pseudo_incompressible_poisson_mpi.py``) gates correctness: the
gathered MPI Laplacian and solve must match the single-process result to tolerance.

Requires the ``mpi4jax`` stack (``.venv-mpi``; the main ``.venv`` jax 0.10 cannot host
mpi4jax — run under ``mpirun`` with ``.venv-mpi``).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.parallel.reductions import global_sum_mpi

_AY, _AX, _AZ = 0, 1, 2


def _halo_y(field, halo, comm):
    """Periodic y-ring halo exchange: pad ``field`` (ny_loc,nx,nz) to
    (ny_loc+2·halo, nx, nz) with ``halo`` rows from the up/down rank neighbours."""
    import mpi4jax
    rank, nranks = comm.Get_rank(), comm.Get_size()
    up = (rank + 1) % nranks            # neighbour at +y
    down = (rank - 1) % nranks          # neighbour at −y
    top_strip = field[-halo:]           # last rows
    bot_strip = field[:halo]            # first rows
    # bottom ghost (row below my first) = my DOWN-neighbour's last row, which it sends
    # UPWARD: so SEND top_strip to `up`, RECEIVE from `down`. Both ranks run this call
    # first, then the next — call-order pairing (no tags needed; correct incl. nranks==2).
    bot_ghost = mpi4jax.sendrecv(top_strip, top_strip, source=down, dest=up, comm=comm)
    # top ghost (row above my last) = my UP-neighbour's first row, sent DOWNWARD:
    # SEND bot_strip to `down`, RECEIVE from `up`.
    top_ghost = mpi4jax.sendrecv(bot_strip, bot_strip, source=up, dest=down, comm=comm)
    bot_ghost = bot_ghost[0] if isinstance(bot_ghost, tuple) else bot_ghost
    top_ghost = top_ghost[0] if isinstance(top_ghost, tuple) else top_ghost
    return jnp.concatenate([bot_ghost, field, top_ghost], axis=_AY)


def laplace_pi_mpi(pi, c, dx, dy, dz, comm, cp=constants.c_pd):
    """Distributed ``Cp·∇·(C∇π')`` on a y-slab. x periodic-local, y halo-exchanged,
    z zero-flux walls. ``pi, c`` are local ``(ny_loc, nx, nz)``; returns same shape."""
    # x: identical to the serial compact stencil (x undecomposed)
    cx = 0.5 * (c + jnp.roll(c, -1, axis=_AX))
    fx = cx * (jnp.roll(pi, -1, axis=_AX) - pi) / dx
    lap_x = (fx - jnp.roll(fx, 1, axis=_AX)) / dx
    # y: 1-cell halo, then face stencil on the padded array (interior result)
    pih = _halo_y(pi, 1, comm)          # (ny_loc+2, nx, nz)
    ch = _halo_y(c, 1, comm)
    cyf = 0.5 * (ch[:-1] + ch[1:])      # C at y-faces, length ny_loc+1
    fyf = cyf * (pih[1:] - pih[:-1]) / dy
    lap_y = (fyf[1:] - fyf[:-1]) / dy   # interior (ny_loc, nx, nz)
    # z: zero-flux walls (local, identical to serial)
    cz = 0.5 * (c[..., :-1] + c[..., 1:])
    fz_int = cz * (pi[..., 1:] - pi[..., :-1]) / dz
    fz = jnp.pad(fz_int, [(0, 0), (0, 0), (1, 1)])
    lap_z = (fz[..., 1:] - fz[..., :-1]) / dz
    return cp * (lap_x + lap_y + lap_z)


def poisson_diag_mpi(c, dx, dy, dz, comm, cp=constants.c_pd):
    """Diagonal of :func:`laplace_pi_mpi` (for Jacobi preconditioning)."""
    cx_p = 0.5 * (c + jnp.roll(c, -1, axis=_AX))
    cx_m = jnp.roll(cx_p, 1, axis=_AX)
    ch = _halo_y(c, 1, comm)
    cyf = 0.5 * (ch[:-1] + ch[1:])      # length ny_loc+1
    cy_p = cyf[1:]; cy_m = cyf[:-1]     # C_{j+½}, C_{j−½} interior
    cz_int = 0.5 * (c[..., :-1] + c[..., 1:])
    cz_full = jnp.pad(cz_int, [(0, 0), (0, 0), (1, 1)])
    cz_p = cz_full[..., 1:]; cz_m = cz_full[..., :-1]
    diag = -((cx_p + cx_m) / dx ** 2 + (cy_p + cy_m) / dy ** 2
             + (cz_p + cz_m) / dz ** 2)
    return cp * diag


def _gdot(a, b, comm):
    """Global inner product ⟨a,b⟩ = allreduce(Σ_local a·b)."""
    return global_sum_mpi(jnp.sum(a * b), comm=comm)


def _gmean(f, n_global, comm):
    return global_sum_mpi(jnp.sum(f), comm=comm) / n_global


def _bicgstab_mpi(matvec, b, x0, M, comm, tol, atol, maxiter):
    """Matrix-free BiCGSTAB with GLOBAL (allreduced) inner products — the distributed
    analogue of ``jax.scipy.sparse.linalg.bicgstab`` (which sums dot-products LOCALLY
    only and is therefore wrong under domain decomposition). Jacobi M = 1/diag.

    The Krylov loop runs EAGER (Python ``while``): mpi4jax collectives execute in
    program/rank order, avoiding the token-threading hazard of collectives inside
    ``lax.while_loop``. Each matvec is one 1-cell y-halo exchange; each dot is one
    allreduce. The convergence test syncs the (already-global) residual norm to host.

    NOT differentiable: the eager Python loop with a data-dependent break does not
    support ``jax.grad`` (it would silently unroll to a fixed count and give wrong
    gradients). The distributed solve is a FORWARD-only path; AD through the distributed
    pressure projection is a separate effort (the serial solver IS differentiable).

    Breakdown-guarded: the BiCGSTAB scalar denominators (⟨r̂,v⟩ for α, ⟨t,t⟩ for ω, and
    ⟨s,s⟩→exact-convergence on the half-step) are checked and the iteration exits cleanly
    rather than propagating a 0/0 NaN into the solution."""
    x = x0
    r = b - matvec(x)
    r_hat = r
    one = jnp.asarray(1.0, b.dtype)
    rho = alpha = omega = one
    v = p = jnp.zeros_like(b)
    bnorm = float(jnp.sqrt(_gdot(b, b, comm)))
    tol_eff = max(tol * bnorm, atol)
    brk = (bnorm + 1.0) * 1e-30          # absolute breakdown floor for denominators
    for _ in range(maxiter):
        rho_new = _gdot(r_hat, r, comm)
        if float(jnp.abs(rho_new)) < brk:           # ⟨r̂,r⟩ breakdown
            break
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        phat = M(p)
        v = matvec(phat)
        rhat_v = _gdot(r_hat, v, comm)
        if float(jnp.abs(rhat_v)) < brk:            # α breakdown
            break
        alpha = rho_new / rhat_v
        s = r - alpha * v
        x_half = x + alpha * phat
        if float(jnp.sqrt(_gdot(s, s, comm))) <= tol_eff:   # exact half-step convergence
            x = x_half
            break
        shat = M(s)
        t = matvec(shat)
        tt = _gdot(t, t, comm)
        if float(tt) < brk:                         # ω breakdown
            x = x_half
            break
        omega = _gdot(t, s, comm) / tt
        x = x_half + omega * shat
        r = s - omega * t
        rho = rho_new
        if float(jnp.sqrt(_gdot(r, r, comm))) <= tol_eff:
            break
    return x


def solve_pressure_mpi(rhs, c, dx, dy, dz, comm, n_global, x0=None,
                       cp=constants.c_pd, tol=1e-6, atol=1e-10, maxiter=200,
                       precondition=True):
    """Distributed solve ``Cp·∇·(C∇π')=rhs`` (zero-mean gauge) with Jacobi-BiCGSTAB.

    ``n_global = ny_global·nx·nz`` (for the global zero-mean). Returns the local
    ``π'`` slab. Mirrors the serial :func:`pseudo_incompressible_poisson.solve_pressure`
    but every reduction is an MPI allreduce over ``comm``."""
    n_local = rhs.shape[0] * rhs.shape[1] * rhs.shape[2]
    expected = n_local * comm.Get_size()
    if n_global != expected:
        raise ValueError(
            f"n_global={n_global} must equal ny_global·nx·nz = n_local·n_ranks = "
            f"{expected}; passing the LOCAL slab count under-normalises the zero-mean "
            f"gauge by a factor of n_ranks.")
    rhs_c = rhs - _gmean(rhs, n_global, comm)
    x0 = jnp.zeros_like(rhs_c) if x0 is None else x0

    def matvec(p):
        return laplace_pi_mpi(p, c, dx, dy, dz, comm, cp=cp)

    if precondition:
        inv_diag = 1.0 / poisson_diag_mpi(c, dx, dy, dz, comm, cp=cp)
        M = lambda r: inv_diag * r       # noqa: E731
    else:
        M = lambda r: r                  # noqa: E731

    pi = _bicgstab_mpi(matvec, rhs_c, x0, M, comm, tol, atol, maxiter)
    return pi - _gmean(pi, n_global, comm)
