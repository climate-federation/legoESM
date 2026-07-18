"""MPI-distributed (y-slab) variable-coefficient pressure Poisson for the
pseudo-incompressible plane LES — the nearest-neighbour, mesh-scalable elliptic solve.

This is the distributed counterpart of :mod:`pseudo_incompressible_poisson`. It proves the
core GPU/TPU-scaling thesis: the matrix-free 7-point Laplacian communicates ONLY a 1-cell
halo with its two y-neighbours per matvec (NOT the spectral core's global all-to-all FFT),
and the BiCGSTAB Krylov inner products are the only global reductions.

Decomposition: y-slab (1D over the y axis, axis 0). Each rank owns ``(ny_local, nx, nz)``;
``ny_global = ny_local · n_ranks``. x is periodic and undecomposed (local ``jnp.roll``); z is
local (vertical). The y direction is periodic across the rank ring, exchanged through the
shared AD-safe ``sendrecv`` wrapper (:func:`legoesm.parallel.halo_exchange.get_sendrecv_vjp`
— ``custom_vjp``, GPU-transport preflight included). At ``n_ranks == 1`` the ring
degenerates to a local periodic pad, so every entry point also runs serially (and under
``jax.grad``) without an MPI launcher.

Communication budget per BiCGSTAB iteration (the strong-scaling currency):

- 2 matvecs × 1 packed 1-cell y-halo message pair (the coefficient halo is hoisted out of
  the Krylov loop — see ``solve_pressure_mpi``),
- 3 batched ``allreduce(SUM)`` (α batch, ω/half-step batch, ρ/residual batch) instead of
  the textbook 6 sequential scalar allreduces.

The Krylov loop itself is a fixed-trip-count ``lax.fori_loop`` with masked ("frozen")
updates after convergence/breakdown — the same static-schedule doctrine as the ocean
fixed-iteration barotropic PCG (``barotropic_common``): every rank executes the identical
collective schedule, so there is no data-dependent trip count to desynchronise mpi4jax.
The forward driver runs the loop in host-checked CHUNKS (one ``float()`` sync per chunk,
not per iteration); :func:`solve_pressure_mpi_fixed_iters` runs a single static-length
loop with NO host syncs and is reverse-mode differentiable end to end (halo cotangents via
the ``custom_vjp`` sendrecv, dot cotangents via ``allreduce(SUM)``).

A serial==MPI parity test (``tests/distributed/test_pseudo_incompressible_poisson_mpi.py``)
gates correctness: the gathered MPI Laplacian and solve must match the single-process
result to tolerance.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
from jax import lax
from legoesm.parallel.reductions import global_sum_mpi

from legoesm import constants

_AY, _AX, _AZ = 0, 1, 2

# MPI tags for the two directional ring exchanges. Distinct tags keep the
# up/down messages unambiguous even at n_ranks == 2 (where both neighbours
# are the same rank); the range sits above the cubed-sphere halo tag scheme
# (max 399_999 in ``halo_exchange``) so a shared communicator can never
# collide.
_TAG_SEND_UP = 610_001
_TAG_SEND_DOWN = 610_002

# Host-convergence check cadence of the forward solver: one device->host
# sync every ``_CHECK_EVERY`` Krylov iterations (a latency/wasted-iteration
# trade; iterations after convergence are masked no-ops). Loop cadence is a
# numerics constant, not a tunable physics parameter.
_CHECK_EVERY = 8


def _halo_y(field, halo, comm):
    """Periodic y-ring halo exchange: pad ``field`` (ny_loc, ...) to
    (ny_loc+2·halo, ...) with ``halo`` rows from the up/down rank neighbours.

    Routes through the shared AD-safe ``custom_vjp`` sendrecv (the single
    mandated halo choke point — raw ``mpi4jax.sendrecv`` would break
    ``jax.grad`` and skip the GPU-transport preflight). ``n_ranks == 1``
    reduces to a local periodic pad (serial + AD-testable without mpirun).
    """
    nranks = comm.Get_size()
    if nranks == 1:
        return jnp.pad(
            field, [(halo, halo)] + [(0, 0)] * (field.ndim - 1), mode="wrap",
        )
    import mpi4jax
    from legoesm.parallel.halo_exchange import get_sendrecv_vjp

    sendrecv = get_sendrecv_vjp(mpi4jax)
    rank = comm.Get_rank()
    up = (rank + 1) % nranks            # neighbour at +y
    down = (rank - 1) % nranks          # neighbour at −y
    top_strip = field[-halo:]           # last rows
    bot_strip = field[:halo]            # first rows
    # bottom ghost (rows below my first) = my DOWN-neighbour's last rows,
    # which it sends UPWARD: SEND top_strip to `up`, RECEIVE from `down`.
    bot_ghost = sendrecv(
        top_strip, top_strip, down, up, _TAG_SEND_UP, _TAG_SEND_UP, comm,
    )
    # top ghost (rows above my last) = my UP-neighbour's first rows, sent
    # DOWNWARD: SEND bot_strip to `down`, RECEIVE from `up`.
    top_ghost = sendrecv(
        bot_strip, bot_strip, up, down, _TAG_SEND_DOWN, _TAG_SEND_DOWN, comm,
    )
    return jnp.concatenate([bot_ghost, field, top_ghost], axis=_AY)


def _halo_y_packed(fields, halo, comm):
    """ONE periodic y-ring halo exchange for MULTIPLE fields.

    Flattens each ``(ny_loc, ...)`` field to ``(ny_loc, -1)``, concatenates
    along the flattened axis, runs a single up/down sendrecv pair (2 MPI
    messages instead of ``2·len(fields)``), then splits + reshapes each
    padded field back to ``(ny_loc + 2·halo, ...)``. Trailing shapes may
    differ per field (u/v/w/θ and the 4-D tracer block pack together).
    All sizes are static Python ints → jit-safe."""
    shapes = [f.shape for f in fields]
    flat = jnp.concatenate(
        [f.reshape(f.shape[0], -1) for f in fields], axis=1,
    )
    flat_pad = _halo_y(flat, halo, comm)
    out, col = [], 0
    for shp in shapes:
        n = math.prod(shp[1:])
        out.append(
            flat_pad[:, col:col + n].reshape((flat_pad.shape[0],) + shp[1:])
        )
        col += n
    return out


def laplace_pi_mpi(pi, c, dx, dy, dz, comm, cp=constants.c_pd, c_pad=None):
    """Distributed ``Cp·∇·(C∇π')`` on a y-slab. x periodic-local, y halo-exchanged,
    z zero-flux walls. ``pi, c`` are local ``(ny_loc, nx, nz)``; returns same shape.

    ``c_pad``: optional pre-exchanged ``(ny_loc+2, nx, nz)`` halo of ``c``.
    The coefficient is CONSTANT across a Krylov solve, so the solver hoists
    its halo out of the per-iteration matvec (halving matvec message count);
    when ``None`` (direct callers, tests) π' and C share ONE packed exchange."""
    # x: identical to the serial compact stencil (x undecomposed)
    cx = 0.5 * (c + jnp.roll(c, -1, axis=_AX))
    fx = cx * (jnp.roll(pi, -1, axis=_AX) - pi) / dx
    lap_x = (fx - jnp.roll(fx, 1, axis=_AX)) / dx
    # y: 1-cell halo, then face stencil on the padded array (interior result)
    if c_pad is None:
        pih, ch = _halo_y_packed((pi, c), 1, comm)      # one exchange
    else:
        pih = _halo_y(pi, 1, comm)      # (ny_loc+2, nx, nz)
        ch = c_pad
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
    cy_p, cy_m = cyf[1:], cyf[:-1]      # C_{j+½}, C_{j−½} interior
    cz_int = 0.5 * (c[..., :-1] + c[..., 1:])
    cz_full = jnp.pad(cz_int, [(0, 0), (0, 0), (1, 1)])
    cz_p, cz_m = cz_full[..., 1:], cz_full[..., :-1]
    diag = -((cx_p + cx_m) / dx ** 2 + (cy_p + cy_m) / dy ** 2
             + (cz_p + cz_m) / dz ** 2)
    return cp * diag


def _gdot(a, b, comm):
    """Global inner product ⟨a,b⟩ = allreduce(Σ_local a·b)."""
    return global_sum_mpi(jnp.sum(a * b), comm=comm)


def _gdots(pairs, comm):
    """Batched global inner products: ONE allreduce for k dot-products.

    Stacks the k local partial sums into a length-k vector and allreduces
    it in a single collective — on GPU strong scaling each avoided
    allreduce is a full network latency, so batching the Krylov dots is
    the difference between 6 and 3 collectives per BiCGSTAB iteration."""
    local = jnp.stack([jnp.sum(a * b) for a, b in pairs])
    return global_sum_mpi(local, comm=comm)


def _gmean(f, n_global, comm):
    return global_sum_mpi(jnp.sum(f), comm=comm) / n_global


def _bicgstab_iteration(carry, c, c_pad, inv_diag, dx, dy, dz, cp, comm,
                        tol_eff, brk):
    """One masked BiCGSTAB iteration (the ``lax.fori_loop`` body).

    Every rank executes the SAME operation schedule every iteration — 2
    matvecs (1 packed halo message pair each) + 3 batched allreduces —
    regardless of convergence/breakdown state; once ``active`` drops, all
    state updates are frozen through ``jnp.where`` masks (static schedule,
    no data-dependent collectives — the ocean fixed-iteration PCG
    doctrine). Divisions are double-``where`` guarded so the frozen branch
    can neither produce NaN forward values nor NaN cotangents under
    reverse-mode AD.

    Exit semantics match the classic eager loop exactly:

    - ρ breakdown (|⟨r̂,r⟩| < brk)  → freeze at ``x``
    - α breakdown (|⟨r̂,v⟩| < brk)  → freeze at ``x``
    - half-step convergence (true allreduced ‖s‖ ≤ tol) → ``x = x_half``
    - ω breakdown (⟨t,t⟩ < brk)     → ``x = x_half``
    - full-step convergence (‖r‖ ≤ tol) after the update.

    The half-step norm is the exact ⟨s,s⟩ dot (batched with the ω dots),
    NOT a recurrence expansion — no fp32 cancellation can trigger a
    premature exit.
    """
    (x, r, r_hat, p, v, alpha, omega, rho, rho_new, active) = carry

    def matvec(q):
        return laplace_pi_mpi(q, c, dx, dy, dz, comm, cp=cp, c_pad=c_pad)

    def precond(q):
        return q if inv_diag is None else inv_diag * q

    one = jnp.asarray(1.0, x.dtype)

    # -- β and the direction update (ρ of THIS iteration was measured by the
    #    tail batch of the previous one — a fresh dot, no recurrence drift).
    live = active & (jnp.abs(rho_new) >= brk)           # ρ breakdown → freeze
    rho_safe = jnp.where(jnp.abs(rho) >= brk, rho, one)
    omega_safe = jnp.where(jnp.abs(omega) >= brk, omega, one)
    beta = (rho_new / rho_safe) * (alpha / omega_safe)
    p_try = r + beta * (p - omega * v)
    phat = precond(p_try)
    v_try = matvec(phat)

    # -- batch A: α denominator.
    rhat_v = _gdots([(r_hat, v_try)], comm)[0]
    live_a = live & (jnp.abs(rhat_v) >= brk)            # α breakdown → freeze
    alpha_try = rho_new / jnp.where(jnp.abs(rhat_v) >= brk, rhat_v, one)
    x_half = x + alpha_try * phat
    s = r - alpha_try * v_try

    # -- batch B: ω dots + the TRUE half-step norm (one collective).
    shat = precond(s)
    t = matvec(shat)
    dots_b = _gdots([(t, t), (t, s), (s, s)], comm)
    tt, ts, ss = dots_b[0], dots_b[1], dots_b[2]
    half_conv = jnp.sqrt(jnp.maximum(ss, 0.0)) <= tol_eff
    tt_brk = tt < brk
    exit_half = live_a & (half_conv | tt_brk)           # → x_half, freeze
    live_b = live_a & ~half_conv & ~tt_brk

    # -- full update.
    omega_try = ts / jnp.where(tt >= brk, tt, one)
    x_full = x_half + omega_try * shat
    r_try = s - omega_try * t

    # -- batch C: ρ of the NEXT iteration + the residual norm.
    dots_c = _gdots([(r_hat, r_try), (r_try, r_try)], comm)
    rho_next, rr = dots_c[0], dots_c[1]
    full_conv = jnp.sqrt(jnp.maximum(rr, 0.0)) <= tol_eff

    # -- masked commit.
    x_out = jnp.where(live_b, x_full, jnp.where(exit_half, x_half, x))
    r_out = jnp.where(live_b, r_try, r)
    p_out = jnp.where(live_b, p_try, p)
    v_out = jnp.where(live_b, v_try, v)
    alpha_out = jnp.where(live_b, alpha_try, alpha)
    omega_out = jnp.where(live_b, omega_try, omega)
    rho_out = jnp.where(live_b, rho_new, rho)
    rho_new_out = jnp.where(live_b, rho_next, rho_new)
    active_out = live_b & ~full_conv
    return (x_out, r_out, r_hat, p_out, v_out, alpha_out, omega_out,
            rho_out, rho_new_out, active_out)


def _bicgstab_carry_init(rhs_c, x0, c, c_pad, dx, dy, dz, cp, comm):
    """Initial Krylov carry + the (‖b‖², ‖r₀‖²) batch (one allreduce)."""
    r = rhs_c - laplace_pi_mpi(x0, c, dx, dy, dz, comm, cp=cp, c_pad=c_pad)
    init = _gdots([(rhs_c, rhs_c), (r, r)], comm)
    one = jnp.asarray(1.0, rhs_c.dtype)
    carry = (
        x0, r, r,                       # x, r, r_hat (r̂ = r₀, constant)
        jnp.zeros_like(rhs_c), jnp.zeros_like(rhs_c),   # p, v
        one, one, one,                  # α, ω, ρ
        init[1],                        # ρ_new = ⟨r̂,r⟩ = ‖r₀‖² at entry
        jnp.asarray(True),              # active
    )
    return carry, init[0], init[1]


def _run_bicgstab_chunks(rhs_c, x0, c, c_pad, inv_diag, dx, dy, dz, cp, comm,
                         tol, atol, maxiter, check_every):
    """Forward driver: fixed-schedule chunks + ONE host sync per chunk."""
    carry, bb, rr0 = _bicgstab_carry_init(
        rhs_c, x0, c, c_pad, dx, dy, dz, cp, comm,
    )
    bnorm = math.sqrt(max(float(bb), 0.0))
    tol_eff = jnp.asarray(max(tol * bnorm, atol), rhs_c.dtype)
    brk = jnp.asarray((bnorm + 1.0) * 1e-30, rhs_c.dtype)
    if math.sqrt(max(float(rr0), 0.0)) <= float(tol_eff):
        return x0                       # x0 already converged

    def body(_, cy):
        return _bicgstab_iteration(
            cy, c, c_pad, inv_diag, dx, dy, dz, cp, comm, tol_eff, brk,
        )

    done = 0
    while done < maxiter:
        k = min(check_every, maxiter - done)
        carry = lax.fori_loop(0, k, body, carry)
        done += k
        if not bool(carry[-1]):         # ONE host sync per chunk; the
            break                       # active flag is allreduce-derived,
    return carry[0]                     # so every rank breaks together.


def solve_pressure_mpi(rhs, c, dx, dy, dz, comm, n_global, x0=None,
                       cp=constants.c_pd, tol=1e-6, atol=1e-10, maxiter=200,
                       precondition=True, check_every=_CHECK_EVERY):
    """Distributed solve ``Cp·∇·(C∇π')=rhs`` (zero-mean gauge) with Jacobi-BiCGSTAB.

    ``n_global = ny_global·nx·nz`` (for the global zero-mean). Returns the local
    ``π'`` slab. Mirrors the serial :func:`pseudo_incompressible_poisson.solve_pressure`
    but every reduction is a batched MPI allreduce over ``comm``.

    ``check_every``: Krylov iterations per host convergence check (the only
    device→host syncs in the solve). Forward-only driver — for a
    reverse-differentiable solve use :func:`solve_pressure_mpi_fixed_iters`.
    """
    n_local = rhs.shape[0] * rhs.shape[1] * rhs.shape[2]
    expected = n_local * comm.Get_size()
    if n_global != expected:
        raise ValueError(
            f"n_global={n_global} must equal ny_global·nx·nz = n_local·n_ranks = "
            f"{expected}; passing the LOCAL slab count under-normalises the zero-mean "
            f"gauge by a factor of n_ranks.")
    rhs_c = rhs - _gmean(rhs, n_global, comm)
    x0 = jnp.zeros_like(rhs_c) if x0 is None else x0

    # C is constant across the solve: exchange its halo ONCE and hoist it
    # out of the per-iteration matvec (1 message pair per matvec, not 2).
    c_pad = _halo_y(c, 1, comm)
    inv_diag = (
        1.0 / poisson_diag_mpi(c, dx, dy, dz, comm, cp=cp)
        if precondition else None
    )

    pi = _run_bicgstab_chunks(
        rhs_c, x0, c, c_pad, inv_diag, dx, dy, dz, cp, comm,
        tol, atol, maxiter, int(check_every),
    )
    return pi - _gmean(pi, n_global, comm)


def solve_pressure_mpi_fixed_iters(rhs, c, dx, dy, dz, comm, n_global,
                                   n_iters, x0=None, cp=constants.c_pd,
                                   tol=1e-6, atol=1e-10):
    """Reverse-mode-differentiable distributed pressure solve (static length).

    Runs EXACTLY ``n_iters`` masked BiCGSTAB iterations in one static-trip
    ``lax.fori_loop`` (→ ``scan``, reverse-differentiable) with NO host
    syncs: after convergence/breakdown the remaining iterations are frozen
    no-ops with the identical collective schedule on every rank. Gradients
    flow through the halo exchanges via the shared ``custom_vjp`` sendrecv
    and through the batched dots via ``allreduce(SUM)`` — the distributed
    twin of the ocean fixed-iteration barotropic PCG.

    ``tol``/``atol`` only arm the masked early-freeze (they do not change
    the trip count); pick ``n_iters`` from the forward solver's typical
    iteration count for the resolution at hand.
    """
    n_local = rhs.shape[0] * rhs.shape[1] * rhs.shape[2]
    expected = n_local * comm.Get_size()
    if n_global != expected:
        raise ValueError(
            f"n_global={n_global} must equal ny_global·nx·nz = n_local·n_ranks = "
            f"{expected}; passing the LOCAL slab count under-normalises the zero-mean "
            f"gauge by a factor of n_ranks.")
    rhs_c = rhs - _gmean(rhs, n_global, comm)
    x0 = jnp.zeros_like(rhs_c) if x0 is None else x0
    c_pad = _halo_y(c, 1, comm)
    inv_diag = 1.0 / poisson_diag_mpi(c, dx, dy, dz, comm, cp=cp)

    carry, bb, _rr0 = _bicgstab_carry_init(
        rhs_c, x0, c, c_pad, dx, dy, dz, cp, comm,
    )
    # tol_eff from the allreduced ‖b‖ WITHOUT a host sync (‖b‖ is a traced
    # scalar here — the fixed-length loop needs no Python-level decisions).
    bnorm = jnp.sqrt(jnp.maximum(bb, 0.0))
    tol_eff = jnp.maximum(tol * bnorm, jnp.asarray(atol, rhs_c.dtype))
    brk = (bnorm + 1.0) * 1e-30

    def body(_, cy):
        return _bicgstab_iteration(
            cy, c, c_pad, inv_diag, dx, dy, dz, cp, comm, tol_eff, brk,
        )

    carry = lax.fori_loop(0, int(n_iters), body, carry)
    pi = carry[0]
    return pi - _gmean(pi, n_global, comm)
