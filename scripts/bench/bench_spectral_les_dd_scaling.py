"""MPI strong + weak scaling benchmark for the spectral plane LES (y-slab FFT).

Config, timestep and per-step map are the LES-suite emission driver's
(``scripts/run/run_les_suite.py``) for a dry-convective case (default
``cbl_nieuwstadt``): ``cbl_les_config`` (de-aliasing, SSP-RK3, buoyancy, the
case's first SGS variant, ``sgs_buoyancy=False``), ``select_dt`` and
``_make_emit_step``. Every argument this bench does not own goes to that
driver's ``build_arg_parser`` (``--theta0``, ``--pr-sgs``, ``--cfl``, ...).

The case's cell size (dx, dy, dz) is kept; the bench sets the cell COUNTS:

* ``--mode strong``: ``--nx/--ny`` are the GLOBAL grid.
* ``--mode weak``: ``--nx/--ny`` are cells per rank; the global grid is
  ``(ny * n_y, nx * n_x)`` with ``(n_y, n_x)`` the most square factorisation
  of N, so the physical domain grows at fixed dy, dx. Cells per rank are
  constant, but under the y-slab layout the slab SHAPE is
  ``(ny * n_y / N) x (nx * n_x)`` and so changes with N; part of the weak
  "efficiency" is that shape change, not communication.

N = 1 runs the serial spectral core (no layout); N > 1 runs the y-slab layout
(global ny must divide by N). Each rank builds its own slab of the initial
state; the theta noise is drawn per GLOBAL row, so the IC does not depend on N.

    JAX_PLATFORMS=cpu mpiexec -n 4 python scripts/bench/bench_spectral_les_dd_scaling.py \\
        --mode strong --precision float32 --nx 512 --ny 512 --nz 128 \\
        --output results/crm_les_scaling/les.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_plane_crm_dd_scaling import (  # noqa: E402
    _timed, global_grid_shape, most_square_factors, provenance,
)


def bench_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
    p.add_argument("--precision", choices=["float32", "float64"],
                   default="float64")
    p.add_argument("--case", default="cbl_nieuwstadt")
    p.add_argument("--nx", type=int, required=True)
    p.add_argument("--ny", type=int, required=True)
    p.add_argument("--nz", type=int, required=True)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--time-steps", type=int, default=10)
    p.add_argument("--windows", type=int, default=3)
    p.add_argument("--output", type=str, default=None)
    return p


def scaled_grid(case_grid, nx: int, ny: int, nz: int):
    """``case_grid`` with new cell counts at the case's cell size."""
    dx = case_grid.Lx_m / case_grid.nx
    dy = case_grid.Ly_m / case_grid.ny
    dz = case_grid.Lz_m / case_grid.nz
    return dataclasses.replace(case_grid, nx=nx, ny=ny, nz=nz,
                               Lx_m=nx * dx, Ly_m=ny * dy, Lz_m=nz * dz)


def local_cbl_state(sl, g, args, iy_start: int, ny_local: int, u_geo: float,
                    dtype):
    """The driver's dry-CBL IC on rows ``iy_start .. iy_start+ny_local``.

    Same profile as ``run_les_suite._build_cbl`` (mixed layer below
    ``--zi0``, ``--gamma`` lapse above, 0.1 K Gaussian noise in the mixed
    layer); the noise uses one key per GLOBAL row, so it is independent of
    the rank count (it is a different draw from the driver's single key).
    """
    import jax
    import jax.numpy as jnp

    cfg = g.cfg
    nx, nz = cfg.nx, cfg.nz
    z = g.z_c
    th = jnp.where(z > args.zi0, args.theta0 + args.gamma * (z - args.zi0),
                   args.theta0).astype(dtype)
    key = jax.random.PRNGKey(0)
    rows = jnp.arange(iy_start, iy_start + ny_local)
    noise = jax.vmap(lambda r: jax.random.normal(
        jax.random.fold_in(key, r), (nx, nz), dtype=dtype))(rows)
    th3 = (jnp.broadcast_to(th, (ny_local, nx, nz)).astype(dtype)
           + 0.1 * noise * (z < args.zi0).astype(dtype))
    u = jnp.full((ny_local, nx, nz), dtype(u_geo), dtype)
    v = jnp.zeros((ny_local, nx, nz), dtype)
    w = jnp.zeros((ny_local, nx, nz + 1), dtype)
    return sl.SpectralLESState(
        u=u, v=v, w=w, rhs_u_prev=jnp.zeros_like(u),
        rhs_v_prev=jnp.zeros_like(v), rhs_w_prev=jnp.zeros_like(w),
        theta=th3, rhs_theta_prev=jnp.zeros_like(th3))


def main(argv=None):
    bargs, rest = bench_parser().parse_known_args(argv)
    if bargs.precision == "float32" and "--f32" not in sys.argv:
        sys.argv.append("--f32")  # the driver toggles x64 at import from argv
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))
    import jax
    import jax.numpy as jnp
    from mpi4py import MPI

    import run_les_suite as les
    from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl
    from legoesm.atmosphere.les_suite import (
        get_case, list_cases, register_default_catalog,
    )
    from legoesm.timestepping.split_explicit import select_dt

    jax.config.update("jax_enable_x64", bargs.precision == "float64")
    dtype = jnp.float32 if bargs.precision == "float32" else jnp.float64
    args = les.build_arg_parser().parse_args(["--case", bargs.case] + rest)
    if not list_cases():
        register_default_catalog()
    case = get_case(args.case)
    if case.regime != "dry_convective":
        raise SystemExit(f"{case.name}: bench wires dry_convective only")
    if args.theta0 is None:
        args.theta0 = les._CBL_THETA0_K
    u_geo = args.Ug if args.Ug is not None else (case.geostrophic_wind_m_s or 0.0)
    lat = args.lat if args.lat is not None else les._CBL_LAT_DEG
    f_cor = les._coriolis_f(lat) if u_geo > 0.0 else 0.0
    if not math.isfinite(u_geo) or u_geo < 0.0:
        raise SystemExit(f"--Ug must be finite and >= 0, got {u_geo}")
    if u_geo > 0.0 and not (math.isfinite(lat) and -90.0 <= lat <= 90.0):
        raise SystemExit(f"--lat must be finite in [-90, 90], got {lat}")
    sgs = args.sgs if args.sgs is not None else case.sgs_variants[0]
    if sgs not in case.sgs_variants:
        raise SystemExit(f"--sgs {sgs!r} not in {list(case.sgs_variants)}")
    Q0 = args.q0 if args.q0 is not None else case.surface_theta_flux_K_m_s

    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    n_y, n_x = most_square_factors(n)
    ny_g, nx_g = global_grid_shape(bargs.mode, bargs.ny, bargs.nx, n_y, n_x)
    if ny_g % n:
        raise SystemExit(f"global ny {ny_g} not divisible by {n} ranks")
    ny_l = ny_g // n
    grid = scaled_grid(case.grid, nx_g, ny_g, bargs.nz)
    cfg = les.cbl_les_config(grid, args, sgs)
    layout = (sl.SpectralLESLayout(rank=rank, n_ranks=n, ny_global=ny_g,
                                   nx=nx_g, comm=comm) if n > 1 else None)
    g = sl.make_grid(cfg, dtype=dtype, layout=layout)
    st = local_cbl_state(sl, g, args, rank * ny_l, ny_l, u_geo, dtype)
    step = les._make_emit_step(g, case.regime, (u_geo, 0.0), f_cor, Q0,
                               grid.Lz_m, dtype)
    dt0 = (float(args.dt) if args.dt is not None else
           select_dt(g.dx, max_wind_safe=args.max_wind, cfl_safe=args.cfl,
                     dt_cap=args.dt_max))
    dt = jnp.asarray(dt0, dtype)

    theta_ic = st.theta

    def sync(tree):
        jax.block_until_ready(tree)

    first = [True]

    def one_step():
        nonlocal st
        st, _ = step(st, dt=dt, first=first[0])
        first[0] = False
        return st

    t_compile = _timed(one_step, 1, sync)
    _timed(one_step, 1, sync)  # first=False variant compiles here
    _timed(one_step, max(bargs.warmup_steps - 2, 0), sync)
    windows = []
    for _ in range(bargs.windows):
        comm.Barrier()
        windows.append(comm.allreduce(_timed(one_step, bargs.time_steps, sync),
                                      op=MPI.MAX))
    finite = all(bool(jnp.all(jnp.isfinite(a))) for a in (st.u, st.v, st.w,
                                                          st.theta))
    finite = comm.allreduce(int(finite), op=MPI.MIN) == 1
    max_dtheta = comm.allreduce(float(jnp.max(jnp.abs(st.theta - theta_ic))),
                                op=MPI.MAX)
    if rank == 0:
        per = [w / bargs.time_steps for w in windows]
        rec = {
            "model": "spectral_les", "case": case.name, "sgs": sgs,
            "mode": bargs.mode, "precision": bargs.precision, "n_ranks": n,
            "ny_global": ny_g, "nx_global": nx_g, "nz": bargs.nz,
            "ny_local": ny_l, "dx": g.dx, "dy": g.dy, "dz": g.dz, "dt": dt0,
            "time_steps": bargs.time_steps, "windows": bargs.windows,
            "compile_plus_first_step_s": t_compile,
            "ms_per_step_windows": [1e3 * s for s in per],
            "ms_per_step_min": 1e3 * min(per),
            "ms_per_step_median": 1e3 * statistics.median(per),
            "finite": finite, "max_abs_dtheta": max_dtheta,
            "les_config": {k: (v if isinstance(v, (int, float, str, bool,
                                                   type(None))) else str(v))
                           for k, v in cfg._asdict().items()},
            "provenance": provenance(),
        }
        line = json.dumps(rec)
        print(line, flush=True)
        if bargs.output:
            out = Path(bargs.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a") as f:
                f.write(line + "\n")
    return 0 if finite else 3


if __name__ == "__main__":
    sys.exit(main())
