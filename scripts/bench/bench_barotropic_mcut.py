"""Distributed barotropic-PCG iteration-count (M) cut: jacobi vs banded multigrid.

The barotropic implicit solve is a fixed-M preconditioned-CG; each outer
iteration issues a batched ``allreduce(SUM)`` for its two dot products, so the
per-step GLOBAL-reduction count is ``2*M`` — the multinode weak-scaling
reduction-latency wall.  The banded geometric-multigrid preconditioner
(``_make_multigrid_preconditioner_banded``) has NO reduction inside its V-cycle
(only neighbour halos), so if it cuts M from ~60 (jacobi) to ~4-8 it cuts the
per-step reduction count ~120 -> ~16.

This sweeps M for each preconditioner under MPI and records the relative
residual, finding the M each needs to reach a target — the iteration-count cut.
The V-cycle is reduction-free, so the cut holds DISTRIBUTED (this is what makes
it a weak-scaling win, not just a per-device one).

Run::

    mpirun -np 2 python scripts/bench/bench_barotropic_mcut.py --n-lat 96 --n-lon 192

Writes ``docs/scaling/barotropic_mcut.csv`` on rank 0 for the plotter.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from mpi4py import MPI

from legoesm.grids.halo import set_halo_backend
from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.parallel.latlon_mpi import (
    make_latlon_band_layout, slice_latlon_grid_to_band,
)
from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
    _make_helmholtz, _helmholtz_inv_diag, _faces_from_cell_depth,
    _select_preconditioner,
)
from legoesm.ocean.dynamics.barotropic_common import solve_helmholtz_implicit


def _coastal_mask(n_lat, n_lon):
    m = np.ones((n_lat, n_lon))
    m[0, :] = 0.0
    m[-1, :] = 0.0
    m[:, n_lon // 8: n_lon // 8 + 4] = 0.0           # meridional coast
    m[2 * n_lat // 5: 2 * n_lat // 5 + 6,
      4 * n_lon // 9: 4 * n_lon // 9 + 15] = 0.0      # interior basin
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-lat", type=int, default=96)
    p.add_argument("--n-lon", type=int, default=192)
    p.add_argument("--coeff", type=float, default=5.0e7)
    p.add_argument("--m-sweep", type=int, nargs="+",
                   default=[2, 4, 6, 8, 12, 20, 40, 60])
    p.add_argument("--target", type=float, default=1e-6)
    p.add_argument("--out", default="docs/scaling/barotropic_mcut.csv")
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    rank, n_ranks = comm.Get_rank(), comm.Get_size()
    n_lat, n_lon = args.n_lat, args.n_lon
    if n_lat % n_ranks != 0:
        if rank == 0:
            print(f"SKIP: n_lat {n_lat} not divisible by n_ranks {n_ranks}")
        return

    coeff = jnp.asarray(args.coeff)
    grid_raw = create_latlon_grid(n_lat, n_lon)
    mask = jnp.asarray(_coastal_mask(n_lat, n_lon))
    rng = np.random.default_rng(13)
    H_cell = jnp.asarray(1000.0 + 500.0 * rng.random((n_lat, n_lon))) * mask
    rhs_g = jnp.asarray(rng.standard_normal((n_lat, n_lon))) * mask

    layout = make_latlon_band_layout(rank, n_ranks, n_lat, n_lon)
    s, e = layout.lat_start, layout.lat_end
    set_halo_backend("mpi", layout)

    grid_l = ensure_geometry(slice_latlon_grid_to_band(grid_raw, layout))
    Hc_l, m_l = H_cell[s:e], mask[s:e]
    rhs_l = rhs_g[s:e]
    H_u, H_v, u_mask, v_mask = _faces_from_cell_depth(Hc_l, m_l,
                                                      layout.n_lat_local, n_lon)
    A_op = _make_helmholtz(H_u, H_v, coeff, grid_l, m_l, u_mask, v_mask)
    inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid_l, m_l)
    w = grid_l.area * m_l
    x0 = jnp.zeros_like(rhs_l)

    precs = {
        "jacobi": _select_preconditioner(
            "jacobi", inv_diag, H_u, H_v, coeff, grid_l, m_l, A_op=A_op),
        "multigrid": _select_preconditioner(
            "multigrid", inv_diag, H_u, H_v, coeff, grid_l, m_l,
            A_op=A_op, H_cell=Hc_l, layout=layout),
    }

    def resid(M_inv, M):
        _, diag = solve_helmholtz_implicit(
            A_op, rhs_l, M_inv, x0, distributed=True, fixed_iters=M,
            residual_tol=1e-30, stock_cg_tol=1e-12, stock_cg_maxiter=200,
            pcg_variant="standard", dot_weight=w)
        return float(diag.rel_residual)

    rows = []
    for name, M_inv in precs.items():
        for M in args.m_sweep:
            r = resid(M_inv, M)
            rows.append((name, M, r))
            if rank == 0:
                print(f"  {name:10s} M={M:3d}  rel_residual={r:.3e}  "
                      f"reductions/step={2 * M}")

    if rank == 0:
        def m_to_target(name):
            ms = [M for (nm, M, r) in rows if nm == name and r <= args.target]
            return min(ms) if ms else None
        mj = m_to_target("jacobi")
        mg = m_to_target("multigrid")
        print(f"\n=== M to reach rel_residual<={args.target:.0e} "
              f"(grid {n_lat}x{n_lon}, np{n_ranks}) ===")
        print(f"  jacobi:    M={mj}  reductions/step={2 * mj if mj else 'NA'}")
        print(f"  multigrid: M={mg}  reductions/step={2 * mg if mg else 'NA'}")
        if mj and mg:
            print(f"  M-cut: {mj}/{mg} = {mj / mg:.1f}x  "
                  f"reduction-count cut {2 * mj} -> {2 * mg}")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write("preconditioner,M,rel_residual,reductions_per_step,n_ranks,n_lat,n_lon\n")
            for (nm, M, r) in rows:
                f.write(f"{nm},{M},{r:.6e},{2 * M},{n_ranks},{n_lat},{n_lon}\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
