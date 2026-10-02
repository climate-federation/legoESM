"""MPI-distributed (y-slab) full time step for the pseudo-incompressible plane LES.

PARKED in ``_future/`` (ponytail #23, user-approved 2026-10-02): not wired —
no production driver, factory or registry imports this module, and its tests
are skipped.  Wire it into production (moving it back) or delete it.

Domain decomposition: y-slab (axis 0). Each rank owns ``(ny_local, nx, nz)``;
``ny_global = ny_local · n_ranks``. The serial dycore (:mod:`pseudo_incompressible_plane`)
is REUSED verbatim on halo-padded slabs — every horizontal stencil has reach ≤3, so a
3-cell y-halo exchange before a tendency evaluation makes the serial code produce the
correct INTERIOR tendencies (the wrapped halo rows are discarded). The only non-stencil
operation, the MOST surface PLANAR mean, is computed GLOBALLY (allreduce over the unpadded
interior) and injected via ``tendencies(..., sfc_means=...)``. The pressure projection uses
the distributed Poisson (:mod:`pseudo_incompressible_poisson_mpi`), whose matvec halos
internally and whose Krylov dots are allreduced.

A full-step serial==MPI parity test
(``tests/distributed/test_pseudo_incompressible_step_mpi.py``) gates correctness.

Run under ``mpirun`` with an mpi4jax-capable venv. ``step_mpi`` is FORWARD-only (its
chunked distributed solve host-checks convergence); a reverse-differentiable distributed
solve exists as :func:`pseudo_incompressible_poisson_mpi.solve_pressure_mpi_fixed_iters`.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.parallel.reductions import global_sum_mpi
from legoesm.atmosphere.dynamics.les import pseudo_incompressible_plane as _ser
from legoesm.atmosphere.dynamics.les import pseudo_incompressible_poisson_mpi as _pmpi

_AY, _AX, _AZ = 0, 1, 2
# y-halo width per advection scheme = its horizontal stencil reach.
_STENCIL_REACH = {"upwind": 1, "central": 1, "van_leer": 2, "weno5": 3, "weno7": 4, "weno9": 5}


def _halo_width(cfg):
    """Max y-halo the serial stencils need for THIS config — the larger reach of the
    scalar ``scheme`` and the ``momentum_scheme`` (weno5=3, weno7=4, weno9=5,
    central/upwind=1). Static Python int (schemes are static), so it sizes the halo
    exchange + interior slice consistently."""
    return max(_STENCIL_REACH[cfg.scheme], _STENCIL_REACH[cfg.momentum_scheme or cfg.scheme])


def _global_surface_means(u, v, theta, comm, n_surf):
    """GLOBAL planar means ⟨|u₁|⟩, ⟨θ₁⟩ over the lowest level (unpadded interior)."""
    uc0 = 0.5 * (u[..., 0] + jnp.roll(u[..., 0], 1, axis=_AX))
    vc0 = 0.5 * (v[..., 0] + jnp.roll(v[..., 0], 1, axis=_AY))
    spd0 = jnp.sqrt(uc0 ** 2 + vc0 ** 2 + 1e-12)
    spd_mean = global_sum_mpi(jnp.sum(spd0), comm=comm) / n_surf
    th1_mean = global_sum_mpi(jnp.sum(theta[..., 0]), comm=comm) / n_surf
    return spd_mean, th1_mean


def tendencies_mpi(u, v, w, theta, tracers, g, comm, ny_global, forcing=None):
    """Distributed tendencies: scheme-width y-halo (``_halo_width``) → serial
    ``tendencies`` on the padded slab (global surface means injected) → keep the interior."""
    n_surf = ny_global * g.cfg.nx
    sfc_means = _global_surface_means(u, v, theta, comm, n_surf)
    h = _halo_width(g.cfg)      # 3 for weno5 (default), 4/5 for weno7/weno9
    if comm.Get_size() > 1 and u.shape[0] < h:
        raise ValueError(
            f"ny_local={u.shape[0]} is narrower than the advection stencil "
            f"reach h={h} ({g.cfg.scheme}/{g.cfg.momentum_scheme or g.cfg.scheme}): "
            f"the single-hop ring exchange cannot fill a {h}-row halo from a "
            f"{u.shape[0]}-row slab. Use fewer ranks or a narrower scheme."
        )
    # All prognostics (u, v, w, θ + the tracer block) share ONE packed halo
    # round (2 MPI messages instead of 2·n_fields).
    if tracers is None:
        uH, vH, wH, thH = _pmpi.halo_y_packed((u, v, w, theta), h, comm)
        trH = None
    else:
        uH, vH, wH, thH, trH = _pmpi.halo_y_packed(
            (u, v, w, theta, tracers), h, comm)
    au, av, aw, ath, atr = _ser.tendencies(uH, vH, wH, thH, trH, g, forcing,
                                           sfc_means=sfc_means)
    sl = slice(h, -h)
    return (au[sl], av[sl], aw[sl], ath[sl],
            None if atr is None else atr[sl])


def _rho_weighted_divergence_mpi(u, v, w, g, comm):
    """``∇·(ρ0θ0 u)`` (C-grid compact), y-decomposed (v needs a 1-cell down halo)."""
    rt = g.rho0_theta0[None, None, :]
    du = (u - jnp.roll(u, 1, axis=_AX)) / g.dx
    vH = _pmpi.halo_y(v, 1, comm)                     # ghost row below + above
    dv = (vH[1:-1] - vH[:-2]) / g.dy                   # (v_j − v_{j−1})/Δy, interior
    rt_f = _ser._adv._centre_to_face_z(g.rho0_theta0[None, None, :])
    flux_w = rt_f * w
    dw = (flux_w[..., 1:] - flux_w[..., :-1]) / g.dz
    return rt * (du + dv) + dw


def project_mpi(u, v, w, theta, tracers, pi_prev, dt, g, comm, ny_global):
    """EXACT distributed projection (mirrors serial ``project``): solve the distributed
    Poisson then apply the C-grid pressure-gradient correction (y-halo for ∂/∂y)."""
    cfg = g.cfg
    cp = constants.c_pd
    c = _ser._rtt(theta, tracers, g)
    n_global = ny_global * cfg.nx * cfg.nz
    rhs = _rho_weighted_divergence_mpi(u, v, w, g, comm) / dt
    # Same precision-aware tolerance floors as the serial projection —
    # raw fp32 targets below O(eps) drive the Krylov recurrences into
    # breakdown/NaN (shared helper, so serial and MPI stay identical).
    tol, atol = _ser.precision_floored_poisson_tols(cfg, rhs.dtype)
    pi = _pmpi.solve_pressure_mpi(rhs, c, g.dx, g.dy, g.dz, comm, n_global,
                                  x0=pi_prev, tol=tol, atol=atol,
                                  maxiter=cfg.poisson_maxiter)
    th_rho = _ser._theta_rho(theta, tracers, cfg)
    # x (local): same as serial
    thr_xf = 0.5 * (th_rho + jnp.roll(th_rho, -1, axis=_AX))
    dpi_dx_f = (jnp.roll(pi, -1, axis=_AX) - pi) / g.dx
    u_new = u - dt * cp * thr_xf * dpi_dx_f
    # y (decomposed): θ_ρ and π need the +1 (up) neighbour for the j+½ face
    # (one packed round for both fields)
    thrH, piH = _pmpi.halo_y_packed((th_rho, pi), 1, comm)
    thr_yf = 0.5 * (thrH[1:-1] + thrH[2:])             # 0.5(θρ_j + θρ_{j+1})
    dpi_dy_f = (piH[2:] - piH[1:-1]) / g.dy            # (π_{j+1} − π_j)/Δy
    v_new = v - dt * cp * thr_yf * dpi_dy_f
    # z (local): identical to serial
    rtt_zf = 0.5 * (c[..., :-1] + c[..., 1:])
    rho0t0 = g.rho0_theta0[None, None, :]
    rho0t0_zf = 0.5 * (rho0t0[..., :-1] + rho0t0[..., 1:])
    coef_w = cp * rtt_zf / rho0t0_zf
    dpi_dz_int = (pi[..., 1:] - pi[..., :-1]) / g.dz
    w_corr = jnp.pad(coef_w * dpi_dz_int, [(0, 0), (0, 0), (1, 1)])
    w_new = (w - dt * w_corr).at[..., 0].set(0.0).at[..., -1].set(0.0)
    return u_new, v_new, w_new, pi


def _euler_then_project_mpi(su, sv, sw, sth, str_, thp, trp, pi_prev, dt, g, comm, nyg,
                            forcing):
    au, av, aw, ath, atr = tendencies_mpi(su, sv, sw, sth, str_, g, comm, nyg, forcing)
    u_s = su + dt * au
    v_s = sv + dt * av
    w_s = (sw + dt * aw).at[..., 0].set(0.0).at[..., -1].set(0.0)
    th_n = sth + dt * ath
    tr_n = None if str_ is None else str_ + dt * atr
    un, vn, wn, pi = project_mpi(u_s, v_s, w_s, thp, trp, pi_prev, dt, g, comm, nyg)
    return un, vn, wn, th_n, tr_n, pi


def step_mpi(state, g, dt, comm, ny_global, forcing=None):
    """SSP-RK3 distributed step (mirrors serial :func:`pseudo_incompressible_plane.step`),
    projection per stage. ``state`` carries local y-slabs. Returns a new local state."""
    if g.cfg.shapiro_coeff > 0.0 or g.cfg.momentum_shapiro_coeff > 0.0:
        # The serial post-projection [1,2,1] de-noisers (scalar shapiro_coeff and the
        # velocity momentum_shapiro_coeff) roll in y; under the y-slab decomposition
        # that crosses ranks → needs a 1-cell y-halo exchange (not yet wired). Guard
        # so a distributed run can't SILENTLY diverge from serial (same doctrine as
        # the once-serial LASD/dealias MPI guards).
        raise NotImplementedError(
            "shapiro_coeff / momentum_shapiro_coeff > 0 is not supported under MPI yet "
            "(step_mpi): the [1,2,1] y-filter needs a haloed exchange. Run serial, or 0.")
    u0, v0, w0, th0, tr0 = state.u, state.v, state.w, state.theta, state.tracers
    pi0 = state.pi_prev
    nyg = ny_global
    u1, v1, w1, th1, tr1, pi1 = _euler_then_project_mpi(
        u0, v0, w0, th0, tr0, th0, tr0, pi0, dt, g, comm, nyg, forcing)
    eu, ev, ew, eth, etr, _ = _euler_then_project_mpi(
        u1, v1, w1, th1, tr1, th1, tr1, pi1, dt, g, comm, nyg, forcing)
    a, b = 0.75, 0.25
    u2 = a * u0 + b * eu; v2 = a * v0 + b * ev
    w2 = (a * w0 + b * ew).at[..., 0].set(0.0).at[..., -1].set(0.0)
    th2 = a * th0 + b * eth
    tr2 = None if tr0 is None else a * tr0 + b * etr
    u2, v2, w2, pi2 = project_mpi(u2, v2, w2, th2, tr2, pi1, dt, g, comm, nyg)
    eu, ev, ew, eth, etr, _ = _euler_then_project_mpi(
        u2, v2, w2, th2, tr2, th2, tr2, pi2, dt, g, comm, nyg, forcing)
    a, b = 1.0 / 3.0, 2.0 / 3.0
    u3 = a * u0 + b * eu; v3 = a * v0 + b * ev
    w3 = (a * w0 + b * ew).at[..., 0].set(0.0).at[..., -1].set(0.0)
    th3 = a * th0 + b * eth
    tr3 = None if tr0 is None else a * tr0 + b * etr
    u3, v3, w3, pi3 = project_mpi(u3, v3, w3, th3, tr3, pi2, dt, g, comm, nyg)
    return _ser.PseudoIncompressibleState(u=u3, v=v3, w=w3, theta=th3, pi_prev=pi3,
                                          tracers=tr3)
