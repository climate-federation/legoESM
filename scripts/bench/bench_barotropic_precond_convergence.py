#!/usr/bin/env python
"""Barotropic-PCG preconditioner convergence: M-to-tol + distributed cost.

The DECISIVE measurement for the last open scaling lever (codex remaining-
headroom audit 2026-06-14: CPU-multinode ocean Chebyshev M-cut). The fixed-M
distributed PCG's weak-scaling wall is its GLOBAL allreduce latency = 2*M
reductions/step. Chebyshev preconditioning cuts the OUTER iteration count M
(lower residual per iter, NO per-iter reduction) at the cost of `degree`
extra LOCAL matvec-halos per iter. On a latency-bound multinode regime
(allreduce >> halo) trading reductions for halos WINS — IF Chebyshev cuts M
enough.

This sweeps rel_residual vs M for jacobi and chebyshev(deg 2/4/8) on a
realistic stiff (pole+coastal, variable-depth) lat-lon Helmholtz (single
device, no MPI — the *algorithm* is identical with/without ranks), and
reports, per preconditioner, the M to reach a tolerance and the resulting
DISTRIBUTED cost:
  * reductions = 2*M (standard PCG; the multinode latency wall)
  * halos      = M*(1 + degree)  (cheaper neighbor exchange)
If chebyshev's M-to-tol gives 2*M_cheb << 2*M_jac, the multinode A/B is
worth a 2-node allocation; if not, the lever is dead (CPU-multinode ocean
weak is at its practical limit too).

Usage (compute node): JAX_ENABLE_X64=1 python scripts/bench/\
bench_barotropic_precond_convergence.py [--n-lat 90 --n-lon 180]
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-lat", type=int, default=90)
    ap.add_argument("--n-lon", type=int, default=180)
    ap.add_argument("--coeff", type=float, default=5.0e7)
    ap.add_argument("--degrees", default="2,4,8")
    ap.add_argument("--m-list", default="4,8,12,16,20,24,32,40,48,60")
    ap.add_argument("--tols", default="1e-4,1e-6,1e-8")
    args = ap.parse_args()

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
    from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
        _make_helmholtz, _helmholtz_inv_diag, _make_chebyshev_preconditioner,
    )
    from legoesm.ocean.dynamics.barotropic_common import solve_helmholtz_implicit

    n_lat, n_lon = args.n_lat, args.n_lon
    grid = ensure_geometry(create_latlon_grid(n_lat, n_lon))
    rng = np.random.default_rng(13)
    # Stiff, realistic: variable depth + pole rows + meridional coast + basin.
    H = 1000.0 + 500.0 * rng.random((n_lat, n_lon))
    mask = np.ones((n_lat, n_lon))
    mask[0, :] = 0.0
    mask[-1, :] = 0.0
    mask[:, n_lon // 8:n_lon // 8 + 4] = 0.0
    mask[n_lat // 3:n_lat // 3 + 3, n_lon // 2:n_lon // 2 + 10] = 0.0
    H_u = np.zeros((n_lat, n_lon + 1))
    H_u[:, 1:-1] = 0.5 * (H[:, 1:] + H[:, :-1])
    H_u[:, 0] = H_u[:, -1] = 0.5 * (H[:, 0] + H[:, -1])
    u_wet = np.zeros_like(H_u)
    u_wet[:, 1:-1] = mask[:, 1:] * mask[:, :-1]
    u_wet[:, 0] = u_wet[:, -1] = mask[:, 0] * mask[:, -1]
    H_u = jnp.asarray(H_u * u_wet)
    H_v = np.zeros((n_lat + 1, n_lon))
    H_v[1:-1, :] = 0.5 * (H[1:, :] + H[:-1, :])
    v_wet = np.zeros_like(H_v)
    v_wet[1:-1, :] = mask[1:, :] * mask[:-1, :]
    H_v = jnp.asarray(H_v * v_wet)
    mask = jnp.asarray(mask)
    u_wet = jnp.asarray(u_wet)
    v_wet = jnp.asarray(v_wet)
    coeff = jnp.asarray(args.coeff)

    A_op = _make_helmholtz(H_u, H_v, coeff, grid, mask, u_wet, v_wet)
    inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid, mask)
    w = grid.area * mask
    rhs = jnp.asarray(rng.standard_normal((n_lat, n_lon))) * mask
    x0 = jnp.zeros_like(rhs)

    def jacobi(r):
        return r * inv_diag.astype(r.dtype)

    preconds = [("jacobi", 0, jacobi)]
    for d in (int(x) for x in args.degrees.split(",") if x.strip()):
        preconds.append((f"cheby{d}", d,
                         _make_chebyshev_preconditioner(A_op, inv_diag, mask, d)))

    m_list = [int(x) for x in args.m_list.split(",") if x.strip()]
    tols = [float(x) for x in args.tols.split(",") if x.strip()]

    print(f"[conv] grid={n_lat}x{n_lon} coeff={args.coeff:g} "
          f"wet={float(jnp.sum(mask)):.0f}/{n_lat*n_lon}", flush=True)
    # residual(M) table
    resid = {}   # (name) -> {M: rel_res}
    for (name, deg, M_inv) in preconds:
        resid[name] = {}
        for M in m_list:
            _, diag = solve_helmholtz_implicit(
                A_op, rhs, M_inv, x0, distributed=True, fixed_iters=M,
                residual_tol=1e-30, stock_cg_tol=1e-12, stock_cg_maxiter=200,
                pcg_variant="standard", dot_weight=w)
            resid[name][M] = float(diag.rel_residual)
        row = "  ".join(f"M{M}={resid[name][M]:.2e}" for M in m_list)
        print(f"[conv] {name:8s} deg={deg}: {row}", flush=True)

    # M-to-tol + distributed cost (reductions=2M; halos=M*(1+deg)).
    def m_to_tol(name, tol):
        for M in m_list:
            if resid[name][M] <= tol:
                return M
        return None

    print("\n[cost] M-to-tol + distributed cost (reductions=2M, halos=M*(1+deg)):",
          flush=True)
    for tol in tols:
        print(f"  tol={tol:g}:", flush=True)
        jac_M = m_to_tol("jacobi", tol)
        for (name, deg, _) in preconds:
            M = m_to_tol(name, tol)
            if M is None:
                print(f"    {name:8s}: not reached within M<={m_list[-1]}",
                      flush=True)
                continue
            red, hal = 2 * M, M * (1 + deg)
            tag = ""
            if name != "jacobi" and jac_M is not None:
                tag = (f"  reductions x{(2*jac_M)/red:.2f} vs jacobi "
                       f"(halos x{hal/(jac_M*1):.2f})")
            print(f"    {name:8s}: M={M:3d}  reductions={red:3d}  "
                  f"halos={hal:3d}{tag}", flush=True)
    print("CONV_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
