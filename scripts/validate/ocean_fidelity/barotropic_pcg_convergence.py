#!/usr/bin/env python3
"""How many PCG iterations the MPAS implicit barotropic solve actually needs.

Captures the REAL Helmholtz systems (operator, right-hand side, warm start)
produced by ``MPASOceanModel.step`` on a single process, then replays each
captured system through the same fixed-iteration PCG the distributed path
runs (``_fixed_iteration_pcg``, area-weighted dots) at several iteration
counts M.

Reported per (step, M):
  rel_res   true area-weighted relative residual  ||b - A x||_w / ||b||_w
            over wet cells, recomputed from the returned x (never the
            solver's own running estimate);
  d_eta_max max |x_M - x_ref| in metres against a tightly converged
            reference (M_ref iterations, IN THE RUN'S OWN PRECISION — an
            f32 run's reference is f32 and stops at the f32 floor);
  cont      the implied continuity defect |r|_max / dt in mm/day.

The halo exchange is the identity on one process, so this is the same
recurrence the multi-rank solve runs; only the collective is absent — this
measures CONVERGENCE, and cannot see a distributed-only defect (stale halo,
wrong ownership weights, partition-dependent coefficients).

Scored on the raw solve output, before the mass projection and the floor
clamp the caller applies.  The projection adds a spatially uniform constant,
which shifts the residual by that same constant over wet cells; the clamp is
a no-op whenever the free surface stays above the floor.

Provenance: every run stamps git SHA, flags and mesh into the JSON receipt.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subdivision", type=int, default=8)
    p.add_argument("--nlev", type=int, default=40)
    p.add_argument("--dt", type=float, default=300.0)
    p.add_argument("--steps", type=int, default=3,
                   help="model steps to capture systems from")
    p.add_argument("--spinup", type=int, default=0,
                   help="jitted model steps to run BEFORE capturing, so the "
                        "captured systems come from an evolved state rather "
                        "than the initial smooth perturbation.  These run "
                        "with the config's OWN iteration count, so the "
                        "trajectory is self-consistent with the default "
                        "under test.")
    p.add_argument("--iters", type=str, default="5,10,15,20,30,45,60",
                   help="comma-separated M values to score")
    p.add_argument("--ref-iters", type=int, default=600,
                   help="iteration count of the converged reference")
    p.add_argument("--x64", action="store_true",
                   help="run the model state in float64 (default float32)")
    p.add_argument("--lloyd", type=int, default=0)
    p.add_argument("--emulate-devices", type=int, default=1,
                   help="reorder+pad the mesh for this many devices (same "
                        "call the SPMD bench makes) so a LOCAL preconditioner "
                        "can be emulated: edges crossing device blocks are "
                        "dropped from the preconditioner's operator only")
    p.add_argument("--precond", type=str, default="jacobi",
                   help="jacobi | poly:K  (K damped-Jacobi sweeps of the "
                        "device-LOCAL operator, no communication; the "
                        "preconditioner stays SPD, the CG operator is the "
                        "full A)")
    p.add_argument("--save-systems", type=str, default="",
                   help="write the captured systems to this .npz")
    p.add_argument("--load-systems", type=str, default="",
                   help="replay systems from this .npz instead of stepping")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    import jax
    if args.x64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64() if args.x64 else PrecisionPolicy.fp32())

    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.dynamics import barotropic_implicit_mpas as bim
    from legoesm.ocean.dynamics.barotropic_common import _fixed_iteration_pcg
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
    from bench_ocean_mpas_scaling import (  # noqa: E402
        build_problem_config, perturbed_rest_state,
    )

    z_coord, config = build_problem_config(
        args.nlev, barotropic_solver="implicit_cn")
    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
                               lloyd_iterations=args.lloyd)
    nd = int(args.emulate_devices)
    n_real = int(mesh.nCells)
    if nd > 1:
        from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
        from legoesm.parallel.voronoi_spmd_ocean import n_real_cells
        mesh = reorder_voronoi_for_sharding(mesh, nd, method="sfc")
        n_real = n_real_cells(mesh)
    # Device block of each cell under the production contiguous ownership.
    cells_per = int(mesh.nCells) // nd
    owner = np.minimum(np.arange(int(mesh.nCells)) // cells_per, nd - 1)
    c_on_e = np.asarray(mesh.cellsOnEdge)
    same_owner_edge = (owner[c_on_e[0]] == owner[c_on_e[1]]).astype(np.float64)
    if nd > 1:
        print(f"# emulating {nd} devices: {int((1 - same_owner_edge).sum())} "
              f"of {c_on_e.shape[1]} edges cross a device boundary")

    # Capture the real systems: the single-rank leg calls this with exactly
    # the arguments the distributed leg feeds its own solver.
    captured: list[dict] = []
    if args.load_systems:
        z = np.load(args.load_systems)
        if int(z["subdivision"]) != args.subdivision or int(z["nd"]) != nd:
            raise SystemExit("saved systems are for a different mesh/order")
        for k in range(int(z["n"])):
            captured.append({f: jnp.asarray(z[f"{k}_{f}"]) for f in
                             ("rhs", "x0", "H_e", "coeff", "mask",
                              "edge_mask", "inv_diag")})
        spinup_s = capture_s = 0.0
    stock = bim.solve_helmholtz_freesurface_mpas

    def capturing(rhs, x0, H_e, coeff, mask, edge_mask, inv_diag, msh,
                  **kw):
        captured.append(dict(rhs=rhs, x0=x0, H_e=H_e, coeff=coeff,
                             mask=mask, edge_mask=edge_mask,
                             inv_diag=inv_diag))
        return stock(rhs, x0, H_e, coeff, mask, edge_mask, inv_diag, msh,
                     **kw)

    # ``MPASOceanModel.step`` jits internally, so the captured arguments
    # would be tracers; step eagerly for the capture only.
    if not args.load_systems:
        state = perturbed_rest_state(mesh, z_coord, n_cells_real=n_real)
        model = MPASOceanModel(mesh, z_coord, config)
        t0 = time.perf_counter()
        for _ in range(args.spinup):
            state = model.step(state, args.dt)  # jitted, own iteration count
        jax.block_until_ready(jax.tree.leaves(state))
        spinup_s = time.perf_counter() - t0

        bim.solve_helmholtz_freesurface_mpas = capturing
        try:
            t0 = time.perf_counter()
            with jax.disable_jit():
                for _ in range(args.steps):
                    state = model.step(state, args.dt)
            capture_s = time.perf_counter() - t0
        finally:
            bim.solve_helmholtz_freesurface_mpas = stock
        if args.save_systems:
            np.savez(args.save_systems, subdivision=args.subdivision, nd=nd,
                     n=len(captured),
                     **{f"{k}_{f}": np.asarray(v) for k, sysm in
                        enumerate(captured) for f, v in sysm.items()})

    if not captured:
        raise SystemExit("no Helmholtz system captured — the implicit "
                         "barotropic leg did not run")

    area = jnp.asarray(mesh.areaCell)
    m_list = [int(v) for v in args.iters.split(",") if v.strip()]
    rows = []
    for k, sysm in enumerate(captured):
        A_op = bim._make_helmholtz(sysm["H_e"], sysm["coeff"], mesh,
                                   sysm["mask"], sysm["edge_mask"])
        inv_diag = sysm["inv_diag"]
        rhs, x0, mask = sysm["rhs"], sysm["x0"], sysm["mask"]
        w = area.astype(rhs.dtype) * mask

        if args.precond == "jacobi":
            def M_inv(r):
                return r * inv_diag
        elif args.precond.startswith("poly:"):
            K = int(args.precond.split(":")[1])
            em_loc = sysm["edge_mask"] * jnp.asarray(same_owner_edge,
                                                     dtype=rhs.dtype)
            A_loc = bim._make_helmholtz(sysm["H_e"], sysm["coeff"], mesh,
                                        mask, em_loc)
            inv_diag_loc = bim._helmholtz_inv_diag_mpas(
                sysm["H_e"], sysm["coeff"], mesh, mask, em_loc)
            omega = 2.0 / 3.0
            # The local operator keeps the TRUE diagonal of A (every term
            # of a cell's diagonal lives on its own edges, so a device has
            # it) and drops only the off-diagonal couplings that cross a
            # device boundary.  Using the local diagonal instead scaled
            # boundary cells by up to 18x and made the preconditioner
            # worse than Jacobi (measured, 2026-09-20).
            diag_gap = (1.0 / jnp.where(inv_diag > 0, inv_diag, 1.0)
                        - 1.0 / jnp.where(inv_diag_loc > 0, inv_diag_loc, 1.0)
                        ) * mask

            def A_loc_full_diag(z):
                return A_loc(z) + diag_gap * z

            def M_inv(r):
                # K sweeps of damped Jacobi on the device-local operator,
                # from a zero start: z = sum_j (I - w D^-1 A_loc)^j w D^-1 r,
                # a symmetric positive-definite polynomial in A_loc.  No
                # halo exchange: A_loc has no cross-device edges.
                z = omega * inv_diag * r
                for _ in range(K - 1):
                    z = z + omega * inv_diag * (r - A_loc_full_diag(z))
                return z
        else:
            raise ValueError(f"unknown --precond {args.precond!r}")

        def wnorm(v):
            return float(jnp.sqrt(jnp.sum(w * v * v)))

        def solve(m):
            f = jax.jit(lambda b, g: _fixed_iteration_pcg(
                A_op, b, M_inv, g, max_iter=int(m), dot_weight=w)[0])
            return f(rhs, x0)

        x_ref = np.asarray(solve(args.ref_iters))
        nb = wnorm(rhs)
        res_ref = wnorm(rhs - A_op(jnp.asarray(x_ref))) / max(nb, 1e-300)
        for m in m_list:
            x = solve(m)
            r = rhs - A_op(x)
            d = np.abs(np.asarray(x) - x_ref) * np.asarray(mask)
            rows.append(dict(
                step=k, iters=m,
                rel_res=wnorm(r) / max(nb, 1e-300),
                d_eta_max_m=float(d.max()),
                d_eta_rms_m=float(np.sqrt((d * d).mean())),
                cont_defect_mm_day=float(np.abs(np.asarray(r)).max()
                                         / args.dt * 1e3 * 86400.0),
            ))
        rows.append(dict(step=k, iters=args.ref_iters, rel_res=res_ref,
                         d_eta_max_m=0.0, d_eta_rms_m=0.0,
                         cont_defect_mm_day=float(
                             np.abs(np.asarray(rhs - A_op(jnp.asarray(x_ref))))
                             .max() / args.dt * 1e3 * 86400.0)))

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    rec = dict(git_sha=sha, argv=sys.argv[1:], subdivision=args.subdivision,
               emulate_devices=nd, precond=args.precond,
               n_cells=int(mesh.nCells), nlev=args.nlev, dt=args.dt,
               precision="float64" if args.x64 else "float32",
               backend=jax.default_backend(), capture_s=round(capture_s, 1),
               spinup_steps=args.spinup, spinup_s=round(spinup_s, 1),
               ref_iters=args.ref_iters, rows=rows)
    print(f"# mpas barotropic PCG convergence  L{args.subdivision} "
          f"spinup={args.spinup} precond={args.precond} nd={nd} "
          f"nCells={int(mesh.nCells)} {rec['precision']} "
          f"{rec['backend']} dt={args.dt}")
    print(f"{'step':>4} {'M':>5} {'rel_res':>11} {'dEta_max[m]':>12} "
          f"{'dEta_rms[m]':>12} {'cont[mm/day]':>13}")
    for r in rows:
        print(f"{r['step']:>4} {r['iters']:>5} {r['rel_res']:>11.3e} "
              f"{r['d_eta_max_m']:>12.3e} {r['d_eta_rms_m']:>12.3e} "
              f"{r['cont_defect_mm_day']:>13.3e}")
    if args.out:
        Path(args.out).write_text(json.dumps(rec) + "\n")
        print(f"receipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
