#!/usr/bin/env python
"""POC: does a 2-grid V-cycle preconditioner cut the barotropic-PCG M cheaply?

The last open scaling lever (campaign at practical limit otherwise). The
reduction-cut preconditioners (zonal_line/chebyshev) MEASURED as non-wins:
their per-iter LOCAL cost (cyclic-Thomas / extra matvecs) exceeded the
global-reduction latency saved. Multigrid is structurally different — it cuts
the OUTER iteration count via a coarse-grid correction whose per-iter cost is
cheap stencil smoothing + cheap restriction/prolongation halos (same price as
existing halos, NOT a heavy local solve). The chebyshev docstring says MG was
deliberately skipped as "too expensive (restriction/prolongation/coarse-grid-
under-MPI)" — but with the cheap alternatives now ruled out, this POC measures
whether MG's M-cut would justify the full MPI build.

DECISIVE METRIC: M-to-tol for a 2-grid-preconditioned fixed-M PCG vs jacobi,
on the same stiff operator as the convergence char (bench_barotropic_precond_
convergence.py). If 2-grid reaches jacobi-M60 accuracy at ~M<=8-12 with a
per-iter cost of O(few smooths + 1 restrict + 1 coarse-solve + 1 prolong),
the distributed reduction count drops ~5-8x and MG is worth building. If not,
the campaign is DEFINITIVELY at its practical limit.

Single-device, all-ocean (POC: no coastal mask — proves the mechanism; mask +
pole + MPI are the production build's job). Periodic lon, pole rows zeroed.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
    _make_helmholtz, _helmholtz_inv_diag,
)
from legoesm.ocean.dynamics.barotropic_common import (
    _fixed_iteration_pcg, solve_helmholtz_implicit,
)


def _build_level(n_lat, n_lon, H_cell, coeff):
    """(A_op, inv_diag, grid, mask, area, H_cell) for one grid level.

    All-ocean except the two pole rows (v-faces zeroed there). Face depths via
    the production min-rule (periodic in lon)."""
    grid = ensure_geometry(create_latlon_grid(n_lat=n_lat, n_lon=n_lon))
    mask = np.ones((n_lat, n_lon))
    mask[0, :] = 0.0
    mask[-1, :] = 0.0
    mask = jnp.asarray(mask)
    Hu_inner = jnp.minimum(jnp.roll(H_cell, 1, axis=1), H_cell)
    H_u = Hu_inner  # (n_lat, n_lon); _make_helmholtz expects (n_lat, n_lon+1)?
    # _make_helmholtz uses H_u (n_lat, n_lon+1), H_v (n_lat+1, n_lon).
    H_u = jnp.concatenate([Hu_inner, Hu_inner[:, 0:1]], axis=1)
    Hv_inner = jnp.minimum(H_cell[:-1], H_cell[1:])
    H_v = jnp.concatenate(
        [jnp.zeros((1, n_lon)), Hv_inner, jnp.zeros((1, n_lon))], axis=0)
    u_mask = jnp.ones((n_lat, n_lon + 1))
    v_mask = jnp.ones((n_lat + 1, n_lon))
    A_op = _make_helmholtz(H_u, H_v, coeff, grid, mask, u_mask, v_mask)
    inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid, mask)
    # Zonal-LINE preconditioner (exact periodic-tridiagonal solve per lat row =
    # implicit ZONAL solve) — the anisotropy-aware smoother: near the poles
    # dx<<dy so the Helmholtz couples strongly in lon, which pointwise jacobi
    # can't smooth but a zonal line solve handles directly.
    from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
        _make_zonal_line_preconditioner,
    )
    zl_minv = _make_zonal_line_preconditioner(H_u, H_v, coeff, grid, mask)
    return A_op, inv_diag, grid, mask, H_u, H_v, zl_minv


def _restrict(rf, nlc, nloc):
    """Full-weighting fine->coarse: 2x2 block average. rf (2*nlc, 2*nloc)."""
    return 0.25 * (rf[0::2, 0::2] + rf[1::2, 0::2] + rf[0::2, 1::2] + rf[1::2, 1::2])


def _prolong(xc):
    """Piecewise-constant coarse->fine (POC). xc (nlc, nloc) -> (2nlc, 2nloc)."""
    return jnp.repeat(jnp.repeat(xc, 2, axis=0), 2, axis=1)


def _wjacobi(A_op, inv_diag, b, x, omega, sweeps):
    """omega-weighted Jacobi smoother: x <- x + omega*Dinv*(b - A x)."""
    for _ in range(sweeps):
        x = x + omega * inv_diag * (b - A_op(x))
    return x


def _line_smooth(A_op, zl_minv, b, x, omega, sweeps):
    """Zonal-LINE-preconditioned Richardson smoother: x <- x + w*M_zl(b - A x).
    M_zl exactly inverts the zonal tridiagonal per row -> smooths the strong
    zonal coupling that defeats pointwise jacobi on the anisotropic polar grid.
    """
    for _ in range(sweeps):
        x = x + omega * zl_minv(b - A_op(x))
    return x


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-lat", type=int, default=96)
    ap.add_argument("--n-lon", type=int, default=192)
    ap.add_argument("--coeff", type=float, default=5.0e7)
    ap.add_argument("--omega", type=float, default=0.8)
    ap.add_argument("--pre", type=int, default=2)
    ap.add_argument("--post", type=int, default=2)
    ap.add_argument("--coarse-sweeps", type=int, default=10)
    ap.add_argument("--smoother", choices=("jacobi", "zonal_line"),
                    default="jacobi")
    ap.add_argument("--m-list", default="2,4,6,8,12,16,24,40,60")
    args = ap.parse_args()

    nlf, nlof = args.n_lat, args.n_lon
    coeff = jnp.asarray(args.coeff)
    rng = np.random.default_rng(13)
    H_fine = jnp.asarray(1000.0 + 500.0 * rng.random((nlf, nlof)))

    # Build the RECURSIVE level hierarchy by 2x coarsening until the grid is
    # small (>=8 rows). Each level re-discretizes the Helmholtz on the
    # coarsened grid with 2x2-averaged depth (geometric multigrid).
    levels = []   # (A_op, inv_diag_f64, mask, n_lat, n_lon, zl_minv)
    H = H_fine
    nl, nlo = nlf, nlof
    while True:
        A_op, inv_diag, g, mask, _, _, zl_minv = _build_level(nl, nlo, H, coeff)
        levels.append((A_op, inv_diag.astype(jnp.float64), mask, nl, nlo, zl_minv))
        if nl // 2 < 8 or nl % 2 or nlo % 2:
            break
        H = _restrict(H, nl // 2, nlo // 2)
        nl, nlo = nl // 2, nlo // 2
    n_levels = len(levels)
    Af, invf_d, maskf = levels[0][0], levels[0][1], levels[0][2]
    gf = ensure_geometry(create_latlon_grid(n_lat=nlf, n_lon=nlof))
    _use_line = args.smoother == "zonal_line"

    def _smooth(A_op, inv_d, zl_minv, b, x, sweeps):
        if _use_line:
            return _line_smooth(A_op, zl_minv, b, x, args.omega, sweeps)
        return _wjacobi(A_op, inv_d, b, x, args.omega, sweeps)

    def v_cycle(lvl, b, x):
        A_op, inv_d, mask, nl, nlo, zl_minv = levels[lvl]
        if lvl == n_levels - 1:                       # coarsest: solve hard
            return _smooth(A_op, inv_d, zl_minv, b, x, args.coarse_sweeps)
        x = _smooth(A_op, inv_d, zl_minv, b, x, args.pre)            # pre
        r = (b - A_op(x)) * mask                                     # residual
        nlc, nloc = nl // 2, nlo // 2
        rc = _restrict(r, nlc, nloc) * levels[lvl + 1][2]           # restrict
        ec = v_cycle(lvl + 1, rc, jnp.zeros_like(rc))               # recurse
        x = x + _prolong(ec) * mask                                  # correct
        x = _smooth(A_op, inv_d, zl_minv, b, x, args.post)           # post
        return x * mask

    def mg_vcycle(b):
        """One recursive V-cycle as a preconditioner apply M_inv(b)~A^-1 b."""
        return v_cycle(0, b, jnp.zeros_like(b))

    def jac(r):
        return r * invf_d

    w = gf.area * maskf
    rhs = jnp.asarray(rng.standard_normal((nlf, nlof))) * maskf
    x0 = jnp.zeros_like(rhs)
    m_list = [int(x) for x in args.m_list.split(",") if x.strip()]

    grid_chain = " -> ".join(f"{L[3]}x{L[4]}" for L in levels)
    print(f"[mgpoc] {n_levels}-level V-cycle: {grid_chain} | coeff={args.coeff:g} "
          f"omega={args.omega} pre/post={args.pre}/{args.post} "
          f"coarse_sweeps={args.coarse_sweeps} SMOOTHER={args.smoother}",
          flush=True)

    def resid_of(M_inv, M):
        _, diag = solve_helmholtz_implicit(
            Af, rhs, M_inv, x0, distributed=True, fixed_iters=M,
            residual_tol=1e-30, stock_cg_tol=1e-12, stock_cg_maxiter=200,
            pcg_variant="standard", dot_weight=w)
        return float(diag.rel_residual)

    for name, M_inv in (("jacobi", jac), ("2grid-MG", mg_vcycle)):
        row = "  ".join(f"M{M}={resid_of(M_inv, M):.2e}" for M in m_list)
        print(f"[mgpoc] {name:9s}: {row}", flush=True)

    # Distributed cost proxy (latency-bound: each halo/reduction ~1 message):
    #   jacobi PCG: per outer iter = 1 halo (A matvec) + 2 reductions.
    #   MG PCG:     per outer iter = [(n_levels-1)*(pre+post+1) + coarse_sweeps]
    #               halos + 2 reductions.  WIN = fewer OUTER iters M => fewer
    #               2*M reductions; halos rise but are the SAME price (no heavy
    #               local solve, unlike zonal_line cyclic-Thomas).
    mg_halos_per_iter = (n_levels - 1) * (args.pre + args.post + 1) + args.coarse_sweeps
    print(f"[mgpoc] per-outer-iter: jacobi=1 halo+2 red ; "
          f"MG={mg_halos_per_iter} halos+2 red", flush=True)
    print(f"[mgpoc] COST at jacobi-M60-accuracy (read M from the table above): "
          f"jacobi M60 = 60 halos + 120 red ; MG M_mg = "
          f"M_mg*{mg_halos_per_iter} halos + M_mg*2 red", flush=True)
    print("MGPOC_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
