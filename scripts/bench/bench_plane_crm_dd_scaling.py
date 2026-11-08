"""MPI strong + weak scaling benchmark for the plane CRM ``step_halo`` path.

The dycore configuration, height coordinate, tracer count and initial state
come from the production RCEMIP plane driver (``scripts/run/run_rcemip_plane.py``)
through its own builders: every argument this bench does not own is handed to
that driver's ``parse_args`` after ``RCEMIP1_DECK_ARGV`` (the production deck
``run_rcemip1_crm.sh`` values, nlev 64), so any driver flag (``--advection``,
``--sgs-vertical``, ...) passed here overrides the deck value.

* ``--mode strong``: ``--nx/--ny`` are the GLOBAL grid.
  Efficiency(N) = t(1) / (N * t(N)).
* ``--mode weak``: ``--nx/--ny`` are the PER-RANK tile; the global grid is
  ``(ny * n_ranks_y, nx * n_ranks_x)``. Efficiency(N) = t(1) / t(N).

The rank mesh is the most square factorisation of N (``most_square_factors``).
Each rank builds its initial state directly on its own tile (no global array),
equal to the slice of the driver's global initial state (unit-tested).

Timed region = dynamics only (``step_halo``, which includes the MPI mass fixer).
``--time-physics`` additionally times the driver's column physics on the rank's
tile, separately: the per-step branch (surface flux + microphysics) and the
radiation call, plus the amortised per-step cost
``k * non_rad + rad / radiation_interval`` with ``k`` the number of
slow-tendency calls per outer step (counted by tracing the step).

Each run appends one JSON line (``--output``) with the timings, the rank
layout, the resolved dycore config, git SHA, argv and the thread/affinity
environment.

    JAX_PLATFORMS=cpu mpiexec -n 4 python scripts/bench/bench_plane_crm_dd_scaling.py \\
        --mode strong --precision float32 --nx 512 --ny 512 --nlev 64 \\
        --output results/crm_les_scaling/crm.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")


# Production deck = scripts/run/run_rcemip1_crm.sh (production phase) where it
# differs from the driver defaults, with nlev 64 (user decision 2026-11-04).
# Prepended to the driver argv, so an explicit flag on the command line wins.
RCEMIP1_DECK_ARGV = (
    "--dx", "4000", "--dt", "6", "--hyperdiff", "1.0e7",
    "--n-acoustic-substeps", "6", "--off-centering", "0.1",
    "--smag-cs", "0.19", "--semi-implicit", "--microphysics", "morrison",
    "--nlev", "64", "--T-sfc", "300.0", "--insolation", "rcemip",
    "--radiation", "rrtmgp", "--clouds", "--radiation-interval", "150",
)


def bench_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
    p.add_argument("--precision", choices=["float32", "float64"],
                   default="float64")
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--time-steps", type=int, default=10,
                   help="Outer steps per timed window.")
    p.add_argument("--windows", type=int, default=3,
                   help="Number of timed windows.")
    p.add_argument("--time-physics", action="store_true")
    p.add_argument("--physics-calls", type=int, default=5,
                   help="Calls per physics timing (after one warm-up call).")
    p.add_argument("--output", type=str, default=None,
                   help="Append one JSON line (rank 0).")
    return p


def most_square_factors(n: int) -> tuple[int, int]:
    """``(n_y, n_x)`` with ``n_y * n_x == n``, ``n_y <= n_x``, n_y maximal."""
    if n < 1:
        raise ValueError(f"n_ranks must be >= 1, got {n}")
    n_y = int(n ** 0.5)
    while n % n_y:
        n_y -= 1
    return n_y, n // n_y


def global_grid_shape(mode: str, ny: int, nx: int,
                      n_ranks_y: int, n_ranks_x: int) -> tuple[int, int]:
    if mode == "strong":
        return ny, nx
    if mode == "weak":
        return ny * n_ranks_y, nx * n_ranks_x
    raise ValueError(f"unknown mode {mode!r}")


def provenance() -> dict:
    repo = Path(__file__).resolve().parents[2]

    def _git(*a):
        try:
            return subprocess.run(["git", "-C", str(repo), *a], check=True,
                                  capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    env_keys = ("XLA_FLAGS", "JAX_PLATFORMS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "PALS_CPU_BIND", "MPICH_OFI_NIC_POLICY", "PBS_JOBID")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "argv": sys.argv,
        "env": {k: os.environ.get(k) for k in env_keys},
        "host0": socket.gethostname(),
    }


def build_local_ic(rcp, rargs, grid_local, hc, layout, q_sfc, dtype):
    """Driver IC restricted to this rank's tile.

    Builds the driver's horizontally uniform state on the tile
    (``theta_noise_amp=0``) and adds the driver's ``smooth_k1`` seed evaluated
    at GLOBAL indices, so the tile equals the slice of the driver's global IC.
    """
    import jax.numpy as jnp

    if rargs.seed_kind != "smooth_k1":
        raise SystemExit(
            f"--seed-kind {rargs.seed_kind}: only smooth_k1 has a tile-local "
            "form equal to the global IC; band_noise needs the global field.")
    n_tracers = rcp.microphysics_n_tracers(rargs.microphysics)
    state = rcp._build_rcemip_initial_state(
        grid_local, hc, dtype=dtype, theta_noise_amp=0.0,
        n_tracers=n_tracers, seed_kind="smooth_k1", q_sfc=q_sfc,
    )
    if rargs.theta_noise_amp == 0.0:
        return state
    ny_l, nx_l, nlev = state.theta_prime.data.shape
    iy = jnp.arange(layout.iy_start, layout.iy_start + ny_l, dtype=dtype)
    ix = jnp.arange(layout.ix_start, layout.ix_start + nx_l, dtype=dtype)
    pattern = (jnp.cos(2 * jnp.pi * iy / layout.ny_global)[:, None]
               * jnp.cos(2 * jnp.pi * ix / layout.nx_global)[None, :])
    n_seed = min(4, nlev)  # the driver builder's n_seed_lev default
    theta_p = jnp.zeros_like(state.theta_prime.data)
    theta_p = theta_p.at[..., -n_seed:].set(
        rargs.theta_noise_amp * pattern[:, :, None]
        * jnp.ones((1, 1, n_seed), dtype=dtype))
    rho_0 = jnp.asarray(hc.rho_ref, dtype=dtype)
    theta_0 = jnp.asarray(hc.theta_ref, dtype=dtype)
    return state._replace(
        theta_prime=state.theta_prime.replace(data=theta_p),
        rho_prime=state.rho_prime.replace(data=-rho_0 * theta_p / theta_0),
    )


def _timed(fn, n, sync):
    t0 = time.perf_counter()
    out = None
    for _ in range(n):
        out = fn()
    sync(out)
    return time.perf_counter() - t0


def main(argv=None):
    bargs, runner_argv = bench_parser().parse_known_args(argv)
    # The driver toggles x64 at IMPORT time from this env var.
    if bargs.precision == "float32":
        os.environ["LEGOESM_RCEMIP_PLANE_FP32"] = "1"
    else:
        os.environ.pop("LEGOESM_RCEMIP_PLANE_FP32", None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))
    import jax
    import jax.numpy as jnp
    from mpi4py import MPI

    import run_rcemip_plane as rcp
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        PlaneCompressibleEulerModel, horizontal_advection_halo_width,
        make_flat_plane_terrain_metric,
    )
    from legoesm.grids.plane import create_plane_grid
    from legoesm.parallel.plane_mpi import (
        make_plane_pencil_grid, make_plane_pencil_layout,
    )

    jax.config.update("jax_enable_x64", bargs.precision == "float64")
    dtype = jnp.float32 if bargs.precision == "float32" else jnp.float64
    rargs = rcp.parse_args([*RCEMIP1_DECK_ARGV, *runner_argv,
                            "--precision", bargs.precision])
    if rargs.hyperdiff is None:  # same resolution as the driver's main()
        rargs.hyperdiff = rcp.dx_aware_hyperdiff(rargs.dx)
    if rargs.sam_grd is not None or rargs.sounding is not None:
        raise SystemExit("--sam-grd / --sounding ICs are global; not supported.")
    if rargs.restart or rargs.vortex_seed_vmax > 0.0 or rargs.land:
        raise SystemExit("--restart / --vortex-seed-vmax / --land are not "
                         "reproduced by this bench; refusing to time a "
                         "different configuration.")
    if bargs.time_physics and rargs.no_physics:
        raise SystemExit("--time-physics with --no-physics: nothing to time.")

    comm = MPI.COMM_WORLD
    rank, n_ranks = comm.Get_rank(), comm.Get_size()
    n_ranks_y, n_ranks_x = most_square_factors(n_ranks)
    ny_g, nx_g = global_grid_shape(bargs.mode, rargs.ny, rargs.nx,
                                   n_ranks_y, n_ranks_x)
    cfg = rcp.build_dycore_config(rargs)
    halo = horizontal_advection_halo_width(cfg)
    layout = make_plane_pencil_layout(
        rank=rank, n_ranks=n_ranks, n_ranks_y=n_ranks_y, n_ranks_x=n_ranks_x,
        ny_global=ny_g, nx_global=nx_g, halo=halo,
    )
    if n_ranks > 1:
        grid = make_plane_pencil_grid(
            layout, dx=rargs.dx, dy=rargs.dx, nlev=rargs.nlev, dtype=dtype,
            coriolis_mode=rargs.coriolis_mode, f0=rargs.f0, beta=rargs.beta)
    else:
        grid = create_plane_grid(
            nx=nx_g, ny=ny_g, nlev=rargs.nlev, dx=rargs.dx, dy=rargs.dx,
            dtype=dtype, coriolis_mode=rargs.coriolis_mode, f0=rargs.f0,
            beta=rargs.beta)
    hc, _snd, q_sfc = rcp.build_rcemip_height_coord(rargs, rcp.P_SFC_RCEMIP,
                                                    dtype)
    tm = make_flat_plane_terrain_metric(grid, hc)
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    state = build_local_ic(rcp, rargs, grid, hc, layout, q_sfc, dtype)
    owned_mask = jnp.ones((layout.ny_local, layout.nx_local), dtype=dtype)
    theta_ic = state.theta_prime.data

    def sync(tree):
        jax.block_until_ready(tree)

    def one_step():
        nonlocal state
        state = model.step_halo(state, dt=rargs.dt, layout=layout,
                                owned_mask=owned_mask)
        return state

    t_compile = _timed(one_step, 1, sync)
    _timed(one_step, max(bargs.warmup_steps - 1, 0), sync)
    finite_ok = all(bool(jnp.all(jnp.isfinite(f.data))) for f in
                    (state.u, state.v, state.w, state.theta_prime,
                     state.rho_prime, state.tracers))
    window_s = []
    for _ in range(bargs.windows):
        comm.Barrier()
        dt_local = _timed(one_step, bargs.time_steps, sync)
        window_s.append(comm.allreduce(dt_local, op=MPI.MAX))
    comm.Barrier()
    finite_ok = finite_ok and all(
        bool(jnp.all(jnp.isfinite(f.data))) for f in
        (state.u, state.v, state.w, state.theta_prime, state.rho_prime,
         state.tracers))
    finite_all = comm.allreduce(int(finite_ok), op=MPI.MIN) == 1
    max_dtheta = comm.allreduce(
        float(jnp.max(jnp.abs(state.theta_prime.data - theta_ic))), op=MPI.MAX)

    physics = None
    if bargs.time_physics:
        rad_cfg = rcp._build_radiation_config(
            rargs.radiation, update_interval_steps=rargs.radiation_interval,
            clouds=rargs.clouds, insolation=rargs.insolation,
            t_sfc=rargs.T_sfc)
        mp_cfg = rcp._build_microphysics_config(
            rargs.microphysics, rargs.homogeneous_ice_nucleation,
            hard_saturation_adjustment=rargs.hard_saturation_adjustment,
            hard_sat_adjust_threshold=rargs.hard_sat_adjust_threshold,
            hard_sat_max_heating_K=rargs.hard_sat_max_heating_k,
            saturation_sharpness=rargs.saturation_sharpness)
        non_rad_fn, rad_fn = rcp.split_rad_from_other_physics(
            grid, hc, tm, radiation_config=rad_cfg,
            microphysics_config=mp_cfg, dt=rargs.dt, T_sfc=rargs.T_sfc,
            p_sfc=rcp.P_SFC_RCEMIP, surface_flux=not rargs.no_surface_flux)
        physics = {}
        for name, fn in (("non_rad", non_rad_fn), ("rad", rad_fn)):
            if fn is None:
                physics[f"{name}_ms"] = 0.0
                continue
            jf = jax.jit(lambda s, f=fn: f(s, grid, hc, tm))
            sync(jf(state))
            comm.Barrier()
            t = _timed(lambda: jf(state), bargs.physics_calls, sync)
            physics[f"{name}_ms"] = 1e3 * comm.allreduce(
                t / bargs.physics_calls, op=MPI.MAX)
        # The per-step branch runs in every slow-tendency call of the outer
        # step; count those calls by tracing the model's own step.
        calls = [0]

        def _counting(s, g_, h_, t_):
            calls[0] += 1
            return non_rad_fn(s, g_, h_, t_)
        jax.eval_shape(lambda s: model.step(s, dt=rargs.dt,
                                            physics_fn=_counting), state)
        interval = rargs.radiation_interval if rad_fn is not None else 1
        physics["non_rad_calls_per_step"] = calls[0]
        physics["radiation_interval"] = interval
        if interval > 1:  # gated: cached radiation refreshed once/interval
            physics["amortised_ms_per_step"] = (
                calls[0] * physics["non_rad_ms"] + physics["rad_ms"] / interval)
        else:  # the driver folds radiation into the per-stage physics_fn
            physics["amortised_ms_per_step"] = calls[0] * (
                physics["non_rad_ms"] + physics["rad_ms"])

    if rank == 0:
        per_step = [w / bargs.time_steps for w in window_s]
        rec = {
            "model": "crm_plane", "mode": bargs.mode,
            "precision": bargs.precision, "n_ranks": n_ranks,
            "n_ranks_y": n_ranks_y, "n_ranks_x": n_ranks_x,
            "ny_global": ny_g, "nx_global": nx_g,
            "ny_local": layout.ny_local, "nx_local": layout.nx_local,
            "nlev": rargs.nlev, "halo": halo, "dt": rargs.dt,
            "time_steps": bargs.time_steps, "windows": bargs.windows,
            "compile_plus_first_step_s": t_compile,
            "ms_per_step_windows": [1e3 * s for s in per_step],
            "ms_per_step_min": 1e3 * min(per_step),
            "ms_per_step_median": 1e3 * statistics.median(per_step),
            "finite": finite_all, "max_abs_dtheta_prime": max_dtheta,
            "physics": physics,
            "dycore_config": {k: (v if isinstance(v, (int, float, str, bool,
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
    return 0 if finite_all else 3


if __name__ == "__main__":
    sys.exit(main())
