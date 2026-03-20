#!/usr/bin/env python
"""AMIP gray-atmosphere scaling benchmarks for DKRZ Levante.

Benchmarks legoESM's atmospheric dynamical core with Held-Suarez gray-radiation
forcing on two grid types:

  1. **Spectral** (Gaussian grid, T21-T170): spectral primitive equations
  2. **Icosahedral** (MPAS Voronoi, L3-L7): MPAS hydrostatic primitive equations

Supports strong and weak scaling on:
  - GPU (single node, 1-4 NVIDIA A100-80GB)
  - CPU (single node, 1-128 virtual XLA devices on AMD EPYC 7763)
  - MPI (multi-node, cubed-sphere face/sub-face decomposition)

Publication-quality outputs: CSV, LaTeX tables, Markdown report.

Subcommands
-----------
worker   : Run a single benchmark point (emits JSON to stdout).
run      : Run a scaling suite (spawns worker subprocesses, collects results).
generate : Create SLURM batch scripts for Levante CPU and GPU partitions.
collate  : Aggregate results into paper artifacts (CSV, LaTeX, Markdown).

Examples
--------
# Local test (1 device, spectral T42):
JAX_ENABLE_X64=1 python scripts/run_levante_amip_scaling.py run \\
    --grids spectral --resolutions T42 --devices 1 --iterations 20

# Generate Levante SLURM jobs (dry run):
python scripts/run_levante_amip_scaling.py generate --account bb1234

# Generate and submit:
python scripts/run_levante_amip_scaling.py generate --account bb1234 --submit

# Collate results from a completed run:
python scripts/run_levante_amip_scaling.py collate \\
    --input results/levante_amip_scaling/20260317T120000Z
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any

# ======================================================================
# Levante hardware facts (for paper metadata)
# ======================================================================

LEVANTE_FACTS = {
    "system": "DKRZ Levante, Atos BullSequana XH2000",
    "cpu_partition": {
        "nodes": 2982,
        "cpu": "2x AMD EPYC 7763 (128 cores/node)",
        "memory": "256 GB / 512 GB / 1 TB DDR4",
        "peak_pflops": 14.0,
    },
    "gpu_partition": {
        "nodes": 60,
        "cpu": "2x AMD EPYC 7763",
        "gpu": "4x NVIDIA A100 80GB SXM4",
        "memory_gpu": "320 GB HBM2e per node",
        "peak_pflops": 2.8,
    },
    "network": "NVIDIA Mellanox InfiniBand HDR 100G/200G",
    "storage": "130 PB Lustre",
}

# ======================================================================
# Resolution configurations
# ======================================================================

SPECTRAL_RESOLUTIONS: dict[str, dict[str, Any]] = {
    "T21":  {"n_max": 21,  "nlev": 20, "dt": 900.0, "n_lat": 32,  "n_lon": 64,   "deg": 5.6},
    "T42":  {"n_max": 42,  "nlev": 40, "dt": 450.0, "n_lat": 64,  "n_lon": 128,  "deg": 2.8},
    "T60":  {"n_max": 60,  "nlev": 40, "dt": 300.0, "n_lat": 90,  "n_lon": 180,  "deg": 2.0},
    "T85":  {"n_max": 85,  "nlev": 40, "dt": 200.0, "n_lat": 128, "n_lon": 256,  "deg": 1.4},
    "T106": {"n_max": 106, "nlev": 40, "dt": 150.0, "n_lat": 160, "n_lon": 320,  "deg": 1.0},
    "T170": {"n_max": 170, "nlev": 40, "dt": 90.0,  "n_lat": 256, "n_lon": 512,  "deg": 0.7},
}

ICOSAHEDRAL_RESOLUTIONS: dict[str, dict[str, Any]] = {
    "L3": {"subdiv": 3, "nlev": 20, "dt": 900.0, "nCells": 642,    "deg": 4.3},
    "L4": {"subdiv": 4, "nlev": 40, "dt": 450.0, "nCells": 2562,   "deg": 2.2},
    "L5": {"subdiv": 5, "nlev": 40, "dt": 200.0, "nCells": 10242,  "deg": 1.1},
    "L6": {"subdiv": 6, "nlev": 40, "dt": 120.0, "nCells": 40962,  "deg": 1.0},
    "L7": {"subdiv": 7, "nlev": 40, "dt": 60.0,  "nCells": 163842, "deg": 0.5},
}

# Standard weak-scaling ladders (resolution per device count)
SPECTRAL_WEAK_LADDER = [
    (1, "T42"), (2, "T60"), (4, "T85"), (8, "T106"), (16, "T170"),
]
ICOSAHEDRAL_WEAK_LADDER = [
    (1, "L4"), (4, "L5"), (16, "L6"), (64, "L7"),
]

# Default strong-scaling resolutions (including 1-degree)
SPECTRAL_STRONG_RESOLUTIONS = ["T42", "T85", "T106"]
ICOSAHEDRAL_STRONG_RESOLUTIONS = ["L4", "L5", "L6"]


def _dof(grid: str, res: str) -> int:
    """Total degrees of freedom (horizontal x vertical)."""
    if grid == "spectral":
        cfg = SPECTRAL_RESOLUTIONS[res]
        return cfg["n_lat"] * cfg["n_lon"] * cfg["nlev"]
    cfg = ICOSAHEDRAL_RESOLUTIONS[res]
    return cfg["nCells"] * cfg["nlev"]


def _n_cols(grid: str, res: str) -> int:
    """Number of horizontal columns."""
    if grid == "spectral":
        cfg = SPECTRAL_RESOLUTIONS[res]
        return cfg["n_lat"] * cfg["n_lon"]
    return ICOSAHEDRAL_RESOLUTIONS[res]["nCells"]


def _compute_sypd(ms_per_step: float, dt_s: float) -> float:
    """Simulated years per day of wall-clock time."""
    if ms_per_step <= 0:
        return 0.0
    return dt_s * 1000.0 / (ms_per_step * 365.25)


# ======================================================================
# Worker: benchmark functions (run inside subprocess)
# ======================================================================

def _block_ready(state: Any) -> None:
    """Block until all arrays in a pytree are materialized."""
    import jax
    import jax.numpy as jnp
    leaves = jax.tree.leaves(state)
    for leaf in leaves:
        if isinstance(leaf, (jax.Array, jnp.ndarray)):
            leaf.block_until_ready()
            return  # first array is enough for sync


def _benchmark_spectral(
    n_max: int, nlev: int, dt: float, iterations: int, warmup: int,
) -> dict[str, Any]:
    """Spectral PE + Held-Suarez gray forcing benchmark."""
    import jax
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.spectral_pe import (
        SpectralPEConfig,
        SpectralPrimitiveEquationModel,
        isothermal_rest_state_spectral,
    )
    from legoesm.atmosphere.physics.held_suarez import held_suarez_forcing_spectral

    grid = create_gaussian_grid(n_max)
    sigma = create_sigma_coordinate(nlev)
    state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0, p_s_init=1e5)

    a = float(grid.radius)
    eig_max = n_max * (n_max + 1) / (a * a)
    config = SpectralPEConfig(
        hyperdiff_coeff=1.0 / (0.5 * 3600.0 * eig_max ** 2),
        hyperdiff_order=2,
        semi_implicit=True,
        time_integrator="leapfrog_si",
        spectral_filter_order=8,
        spectral_filter_strength=0.01,
        sponge_sigma=0.1,
        sponge_tau=3600.0,
    )
    model = SpectralPrimitiveEquationModel(grid, sigma, config)

    @jax.jit
    def step_fn(s):
        return model.step_with_physics(s, dt, held_suarez_forcing_spectral)

    # --- compile ---
    t_c0 = time.perf_counter()
    state = step_fn(state)
    _block_ready(state)
    compile_s = time.perf_counter() - t_c0

    # --- warmup ---
    for _ in range(warmup):
        state = step_fn(state)
    _block_ready(state)

    # --- steady-state timing ---
    t0 = time.perf_counter()
    for _ in range(iterations):
        state = step_fn(state)
    _block_ready(state)
    elapsed_s = time.perf_counter() - t0

    ncols = int(grid.n_lat * grid.n_lon)
    dof = ncols * nlev
    ms_per_step = 1000.0 * elapsed_s / max(1, iterations)
    throughput = (dof * iterations / max(elapsed_s, 1e-12)) / 1e6
    sypd = _compute_sypd(ms_per_step, dt)

    return {
        "status": "pass",
        "grid": "spectral",
        "n_max": n_max,
        "nlev": nlev,
        "dt_s": dt,
        "n_cols": ncols,
        "dof": dof,
        "iterations": iterations,
        "warmup": warmup,
        "compile_s": float(compile_s),
        "elapsed_s": float(elapsed_s),
        "ms_per_step": float(ms_per_step),
        "throughput_mcells_s": float(throughput),
        "sypd": float(sypd),
        "backend": jax.default_backend(),
        "n_devices": int(jax.local_device_count()),
    }


def _benchmark_icosahedral(
    subdiv: int, nlev: int, dt: float, iterations: int, warmup: int,
) -> dict[str, Any]:
    """MPAS PE + Held-Suarez forcing benchmark."""
    import jax
    import numpy as np
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
        MPASPrimitiveEquationModel,
        MPASPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.physics.held_suarez_mpas import (
        held_suarez_forcing_mpas,
        held_suarez_init_mpas,
    )

    mesh = create_voronoi_mesh(subdiv)
    nCells = int(mesh.nCells)
    sigma = create_sigma_coordinate(nlev)

    dx_mean = np.sqrt(4 * np.pi * mesh.radius ** 2 / nCells)
    nu_del4 = dx_mean ** 4 / (48.0 * 3600.0)

    config = MPASPrimitiveEquationConfig(nu_del4=nu_del4, fix_mass=True)
    model = MPASPrimitiveEquationModel(mesh, sigma, config)
    state = held_suarez_init_mpas(mesh, sigma)

    @jax.jit
    def step_fn(s):
        return model.step(s, dt, held_suarez_forcing_mpas)

    # --- compile ---
    t_c0 = time.perf_counter()
    state = step_fn(state)
    _block_ready(state)
    compile_s = time.perf_counter() - t_c0

    # --- warmup ---
    for _ in range(warmup):
        state = step_fn(state)
    _block_ready(state)

    # --- steady-state timing ---
    t0 = time.perf_counter()
    for _ in range(iterations):
        state = step_fn(state)
    _block_ready(state)
    elapsed_s = time.perf_counter() - t0

    dof = nCells * nlev
    ms_per_step = 1000.0 * elapsed_s / max(1, iterations)
    throughput = (dof * iterations / max(elapsed_s, 1e-12)) / 1e6
    sypd = _compute_sypd(ms_per_step, dt)

    return {
        "status": "pass",
        "grid": "icosahedral",
        "subdiv": subdiv,
        "nlev": nlev,
        "dt_s": dt,
        "n_cols": nCells,
        "dof": dof,
        "iterations": iterations,
        "warmup": warmup,
        "compile_s": float(compile_s),
        "elapsed_s": float(elapsed_s),
        "ms_per_step": float(ms_per_step),
        "throughput_mcells_s": float(throughput),
        "sypd": float(sypd),
        "backend": jax.default_backend(),
        "n_devices": int(jax.local_device_count()),
    }


def _benchmark_mpi_cubedsphere(
    grid_size: int, nlev: int, dt: float, iterations: int, warmup: int,
) -> dict[str, Any]:
    """Cubed-sphere PE + Held-Suarez via MPI face decomposition."""
    import jax
    from mpi4py import MPI
    from legoesm.parallel.comm import build_comm_topology
    from legoesm.parallel.distributed import partition_state
    from legoesm.parallel.mesh import create_device_mesh, replicate_pytree, shard_pytree
    from legoesm.grids.halo import set_halo_backend
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel as FVPrimitiveEquationModel,
        CDGridPrimitiveEquationConfig as FVPrimitiveEquationConfig,
    )
    from legoesm.core.operators_fv_cubed import default_div_damp_coeffs
    from legoesm.atmosphere.physics.held_suarez import held_suarez_forcing, held_suarez_init

    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    n_ranks = int(comm.Get_size())

    topology = build_comm_topology(rank, n_ranks)
    set_halo_backend("mpi", topology)

    config_mesh = create_device_mesh(n_devices=1, devices=jax.local_devices()[:1])

    grid = create_cubed_sphere(grid_size)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init(grid, sigma)

    state = partition_state(state, topology)
    grid = replicate_pytree(grid, config_mesh)
    state = shard_pytree(state, config_mesh)

    nu2, nu4 = default_div_damp_coeffs(grid, dt=dt)
    ref_coeff = 5e16
    hyperdiff = ref_coeff * (48 / grid_size) ** 4

    fv_config = FVPrimitiveEquationConfig(
        hyperdiff_coeff=hyperdiff,
        hyperdiff_ps_coeff=hyperdiff,
        div_damp_2=nu2,
        div_damp_4=nu4,
        use_conservation_fixer=True,
        fix_mass=True,
        time_integrator="ssp45",
        use_limiter=True,
    )
    model = FVPrimitiveEquationModel(grid, sigma, fv_config)

    @jax.jit
    def step_fn(s):
        return model.step_with_physics(s, dt, held_suarez_forcing)

    # Compile
    comm.Barrier()
    t_c0 = time.perf_counter()
    state = step_fn(state)
    _block_ready(state)
    local_compile = time.perf_counter() - t_c0
    compile_s = float(comm.allreduce(local_compile, op=MPI.MAX))

    # Warmup
    for _ in range(warmup):
        state = step_fn(state)
    _block_ready(state)

    # Steady-state timing
    comm.Barrier()
    t0 = time.perf_counter()
    for _ in range(iterations):
        state = step_fn(state)
    _block_ready(state)
    local_elapsed = time.perf_counter() - t0
    elapsed_s = float(comm.allreduce(local_elapsed, op=MPI.MAX))

    ncols = 6 * grid_size * grid_size
    dof = ncols * nlev
    ms_per_step = 1000.0 * elapsed_s / max(1, iterations)
    throughput = (dof * iterations / max(elapsed_s, 1e-12)) / 1e6
    sypd = _compute_sypd(ms_per_step, dt)

    return {
        "status": "pass",
        "grid": "cubed_sphere",
        "grid_size": grid_size,
        "nlev": nlev,
        "dt_s": dt,
        "n_cols": ncols,
        "dof": dof,
        "iterations": iterations,
        "warmup": warmup,
        "compile_s": compile_s,
        "elapsed_s": elapsed_s,
        "ms_per_step": ms_per_step,
        "throughput_mcells_s": throughput,
        "throughput_per_rank": throughput / max(1, n_ranks),
        "sypd": sypd,
        "backend": jax.default_backend(),
        "n_ranks": n_ranks,
        "rank": rank,
        "n_devices": int(jax.local_device_count()),
    }


# ======================================================================
# Worker entry point (--worker subprocess)
# ======================================================================

def _run_worker(args: argparse.Namespace) -> int:
    """Run a single benchmark point and print JSON to stdout."""
    try:
        grid = args.grid
        res = args.resolution
        iterations = args.iterations
        warmup = args.warmup

        if grid == "spectral":
            cfg = SPECTRAL_RESOLUTIONS[res]
            result = _benchmark_spectral(
                n_max=cfg["n_max"], nlev=cfg["nlev"], dt=cfg["dt"],
                iterations=iterations, warmup=warmup,
            )
        elif grid == "icosahedral":
            cfg = ICOSAHEDRAL_RESOLUTIONS[res]
            result = _benchmark_icosahedral(
                subdiv=cfg["subdiv"], nlev=cfg["nlev"], dt=cfg["dt"],
                iterations=iterations, warmup=warmup,
            )
        elif grid == "cubed_sphere_mpi":
            result = _benchmark_mpi_cubedsphere(
                grid_size=args.cs_grid_size, nlev=args.cs_nlev,
                dt=args.cs_dt, iterations=iterations, warmup=warmup,
            )
        else:
            result = {"status": "fail", "error": f"Unknown grid: {grid}"}

        result["resolution"] = res if grid != "cubed_sphere_mpi" else f"C{args.cs_grid_size}"
        print(json.dumps(result), flush=True)
        return 0 if result.get("status") == "pass" else 1

    except Exception as exc:
        err = {
            "status": "fail",
            "grid": getattr(args, "grid", "unknown"),
            "resolution": getattr(args, "resolution", "unknown"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(err), flush=True)
        return 1


# ======================================================================
# Subprocess spawner
# ======================================================================

def _extract_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _spawn_worker(
    *,
    python_exe: str,
    script: str,
    grid: str,
    resolution: str,
    n_devices: int,
    iterations: int,
    warmup: int,
    platform: str,
    x64: bool,
    timeout_s: float,
    extra_env: dict[str, str] | None = None,
    mpi_launcher: str | None = None,
    mpi_np: int = 0,
    cs_grid_size: int = 0,
    cs_nlev: int = 40,
    cs_dt: float = 600.0,
) -> dict[str, Any]:
    """Launch a worker subprocess and return its result."""
    cmd: list[str] = []

    if mpi_launcher and mpi_np > 0:
        cmd.extend([mpi_launcher, "-np", str(mpi_np)])

    cmd.extend([
        python_exe, script, "worker",
        "--grid", grid,
        "--resolution", resolution,
        "--iterations", str(iterations),
        "--warmup", str(warmup),
    ])
    if grid == "cubed_sphere_mpi":
        cmd.extend([
            "--cs-grid-size", str(cs_grid_size),
            "--cs-nlev", str(cs_nlev),
            "--cs-dt", str(cs_dt),
        ])

    env = dict(os.environ)
    env["JAX_PLATFORMS"] = platform
    if platform == "cpu":
        xla = env.get("XLA_FLAGS", "")
        flag = f"--xla_force_host_platform_device_count={n_devices}"
        if flag not in xla:
            env["XLA_FLAGS"] = f"{xla} {flag}".strip()
    if x64:
        env["JAX_ENABLE_X64"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    if extra_env:
        env.update(extra_env)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, env=env,
            timeout=timeout_s,
        )
        wall_s = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "grid": grid,
            "resolution": resolution,
            "n_devices": n_devices,
            "command": shlex.join(cmd),
        }

    payload = _extract_json(proc.stdout or "")
    if payload is None:
        payload = {
            "status": "fail",
            "error": "No JSON from worker",
            "stderr_tail": (proc.stderr or "")[-2000:],
            "stdout_tail": (proc.stdout or "")[-2000:],
        }

    payload["n_devices_requested"] = n_devices
    payload["platform"] = platform
    payload["subprocess_wall_s"] = wall_s
    payload["command"] = shlex.join(cmd)
    payload["returncode"] = proc.returncode
    return payload


# ======================================================================
# Scaling suite (run mode)
# ======================================================================

def _run_suite(args: argparse.Namespace) -> int:
    """Run the complete scaling benchmark suite."""
    script_path = str(Path(__file__).resolve())
    grids = [g.strip() for g in args.grids.split(",") if g.strip()]
    resolutions = [r.strip() for r in args.resolutions.split(",") if r.strip()]
    devices = sorted(set(int(d) for d in args.devices.split(",") if d.strip()))
    platform = args.platform
    scaling = args.scaling
    repeats = args.repeats

    output = Path(args.output or f"results/levante_amip_scaling/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    output.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    point_idx = 0

    for repeat in range(1, repeats + 1):
        for grid in grids:
            if grid == "spectral":
                res_table = SPECTRAL_RESOLUTIONS
            elif grid == "icosahedral":
                res_table = ICOSAHEDRAL_RESOLUTIONS
            else:
                print(f"[WARN] Unknown grid {grid}, skipping", flush=True)
                continue

            # Filter requested resolutions
            valid_res = [r for r in resolutions if r in res_table]
            if not valid_res:
                if scaling == "weak":
                    ladder = SPECTRAL_WEAK_LADDER if grid == "spectral" else ICOSAHEDRAL_WEAK_LADDER
                    for n_dev, res in ladder:
                        if n_dev in devices:
                            valid_res.append(res)
                else:
                    valid_res = list(res_table.keys())

            for res in valid_res:
                for n_dev in devices:
                    if scaling == "weak":
                        # For weak scaling, pair resolution to device count
                        ladder = SPECTRAL_WEAK_LADDER if grid == "spectral" else ICOSAHEDRAL_WEAK_LADDER
                        match = [(nd, r) for nd, r in ladder if r == res and nd == n_dev]
                        if not match:
                            continue

                    point_idx += 1
                    cfg = res_table[res]
                    print(
                        f"[{point_idx:3d}] repeat={repeat} grid={grid} res={res} "
                        f"n_dev={n_dev} platform={platform} scaling={scaling}",
                        flush=True,
                    )

                    result = _spawn_worker(
                        python_exe=args.python,
                        script=script_path,
                        grid=grid,
                        resolution=res,
                        n_devices=n_dev,
                        iterations=args.iterations,
                        warmup=args.warmup,
                        platform=platform,
                        x64=args.x64,
                        timeout_s=args.timeout,
                    )
                    result["repeat"] = repeat
                    result["scaling"] = scaling
                    result["deg"] = cfg.get("deg", 0)
                    all_results.append(result)

                    status = result.get("status", "fail")
                    ms = result.get("ms_per_step", 0)
                    sypd = result.get("sypd", 0)
                    print(
                        f"       -> {status}  {ms:.1f} ms/step  {sypd:.2f} SYPD",
                        flush=True,
                    )

    # --- MPI cubed-sphere benchmarks ---
    if args.mpi_ranks and not args.no_mpi:
        mpi_ranks = sorted(set(int(r) for r in args.mpi_ranks.split(",") if r.strip()))
        cs_sizes = [int(s) for s in args.mpi_cs_sizes.split(",") if s.strip()]

        for repeat in range(1, repeats + 1):
            for cs_n in cs_sizes:
                for n_ranks in mpi_ranks:
                    # Validate cubed-sphere parallel count
                    if n_ranks > 6 and n_ranks % 6 != 0:
                        continue
                    if n_ranks <= 6 and 6 % n_ranks != 0:
                        continue

                    point_idx += 1
                    cs_dt = max(120.0, 600.0 * (48 / cs_n) ** 1.5)
                    print(
                        f"[{point_idx:3d}] repeat={repeat} grid=cubed_sphere_mpi "
                        f"C{cs_n}/L40 n_ranks={n_ranks}",
                        flush=True,
                    )

                    result = _spawn_worker(
                        python_exe=args.python,
                        script=script_path,
                        grid="cubed_sphere_mpi",
                        resolution=f"C{cs_n}",
                        n_devices=1,
                        iterations=args.mpi_iterations,
                        warmup=args.mpi_warmup,
                        platform="cpu",
                        x64=args.x64,
                        timeout_s=args.timeout,
                        mpi_launcher=args.mpi_launcher,
                        mpi_np=n_ranks,
                        cs_grid_size=cs_n,
                        cs_nlev=40,
                        cs_dt=cs_dt,
                    )
                    result["repeat"] = repeat
                    result["scaling"] = "strong_mpi" if scaling == "strong" else "weak_mpi"
                    all_results.append(result)

                    status = result.get("status", "fail")
                    ms = result.get("ms_per_step", 0)
                    print(f"       -> {status}  {ms:.1f} ms/step", flush=True)

    # --- Save raw results ---
    results_file = output / "results.json"
    results_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {results_file}", flush=True)

    # --- Generate paper artifacts ---
    _write_raw_csv(all_results, output / "paper_raw.csv")
    summary = _summarize_results(all_results)
    _write_summary_csv(summary, output / "paper_summary.csv")
    _write_latex_tables(summary, output / "paper_tables.tex")
    _write_markdown_report(all_results, summary, output / "paper_report.md", args)
    _save_metadata(output / "metadata")

    print(f"\nPaper artifacts:")
    print(f"  Raw CSV:     {output / 'paper_raw.csv'}")
    print(f"  Summary CSV: {output / 'paper_summary.csv'}")
    print(f"  LaTeX:       {output / 'paper_tables.tex'}")
    print(f"  Report:      {output / 'paper_report.md'}")
    return 0


# ======================================================================
# Results aggregation and paper artifacts
# ======================================================================

def _write_raw_csv(results: list[dict], path: Path) -> None:
    cols = [
        "repeat", "grid", "resolution", "scaling", "platform",
        "n_devices_requested", "n_devices", "dof", "n_cols", "deg",
        "dt_s", "compile_s", "ms_per_step", "throughput_mcells_s", "sypd",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)


def _q(vals: list[float], quantile: float) -> float:
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    s = sorted(vals)
    pos = (len(s) - 1) * quantile
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _summarize_results(results: list[dict]) -> list[dict]:
    """Group by (grid, resolution, scaling, platform, n_devices) and compute statistics."""
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        if r.get("status") != "pass":
            continue
        key = (
            r.get("grid", ""),
            r.get("resolution", ""),
            r.get("scaling", ""),
            r.get("platform", ""),
            int(r.get("n_devices_requested", r.get("n_devices", 1))),
        )
        groups.setdefault(key, []).append(r)

    summary: list[dict] = []
    for key, group in sorted(groups.items()):
        grid, res, scaling, platform, n_dev = key
        ms_vals = [g["ms_per_step"] for g in group if "ms_per_step" in g]
        sypd_vals = [g["sypd"] for g in group if "sypd" in g]
        tput_vals = [g["throughput_mcells_s"] for g in group if "throughput_mcells_s" in g]
        comp_vals = [g["compile_s"] for g in group if "compile_s" in g]
        dof = group[0].get("dof", 0)
        ncols = group[0].get("n_cols", 0)
        deg = group[0].get("deg", 0)
        dt_s = group[0].get("dt_s", 0)

        summary.append({
            "grid": grid,
            "resolution": res,
            "scaling": scaling,
            "platform": platform,
            "n_devices": n_dev,
            "dof": dof,
            "n_cols": ncols,
            "deg": deg,
            "dt_s": dt_s,
            "n_samples": len(group),
            "ms_per_step_median": statistics.median(ms_vals) if ms_vals else None,
            "ms_per_step_p10": _q(ms_vals, 0.1) if ms_vals else None,
            "ms_per_step_p90": _q(ms_vals, 0.9) if ms_vals else None,
            "sypd_median": statistics.median(sypd_vals) if sypd_vals else None,
            "sypd_p10": _q(sypd_vals, 0.1) if sypd_vals else None,
            "sypd_p90": _q(sypd_vals, 0.9) if sypd_vals else None,
            "throughput_median": statistics.median(tput_vals) if tput_vals else None,
            "compile_s_median": statistics.median(comp_vals) if comp_vals else None,
        })
    return summary


def _write_summary_csv(summary: list[dict], path: Path) -> None:
    if not summary:
        return
    cols = list(summary[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in summary:
            w.writerow(r)


def _fmt(val: Any, fmt: str = ".1f") -> str:
    if val is None:
        return "--"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    if math.isnan(v):
        return "--"
    return f"{v:{fmt}}"


def _compute_efficiency(summary: list[dict], grid: str, res: str, scaling: str, platform: str) -> dict[int, float]:
    """Compute parallel efficiency relative to smallest device count."""
    rows = [
        s for s in summary
        if s["grid"] == grid and s["resolution"] == res
        and s["scaling"] == scaling and s["platform"] == platform
        and s["ms_per_step_median"] is not None
    ]
    if not rows:
        return {}
    rows.sort(key=lambda r: r["n_devices"])
    base_ms = rows[0]["ms_per_step_median"]
    base_dev = rows[0]["n_devices"]
    eff: dict[int, float] = {}
    for r in rows:
        speedup = base_ms / max(r["ms_per_step_median"], 1e-12)
        scale = r["n_devices"] / max(base_dev, 1)
        eff[r["n_devices"]] = 100.0 * speedup / max(scale, 1e-12)
    return eff


def _write_latex_tables(summary: list[dict], path: Path) -> None:
    """Generate LaTeX tables for direct inclusion in a paper."""
    lines: list[str] = []

    lines.append("% Auto-generated by run_levante_amip_scaling.py")
    lines.append(f"% {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    # --- Table 1: Strong scaling (device parallelism) ---
    lines.append("% === Table 1: Strong scaling ===")
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Strong scaling: wall-clock time per time step (ms) and")
    lines.append("  simulated years per day (SYPD) for the Held--Suarez gray-atmosphere")
    lines.append("  benchmark. Parallel efficiency $\\eta$ is relative to 1 device.}")
    lines.append("\\label{tab:strong_scaling}")
    lines.append("\\begin{tabular}{llr rr rr rr r}")
    lines.append("\\toprule")
    lines.append("Grid & Resolution & DOF &")
    lines.append("  \\multicolumn{2}{c}{1 dev} & \\multicolumn{2}{c}{2 dev} &")
    lines.append("  \\multicolumn{2}{c}{4 dev} & $\\eta_4$ \\\\")
    lines.append("     &            &     & ms & SYPD & ms & SYPD & ms & SYPD & (\\%) \\\\")
    lines.append("\\midrule")

    strong_rows = [s for s in summary if s["scaling"] == "strong"]
    seen = set()
    for s in strong_rows:
        key = (s["grid"], s["resolution"])
        if key in seen:
            continue
        seen.add(key)

        grid_label = "Spectral" if s["grid"] == "spectral" else "Icosahedral"
        eff = _compute_efficiency(summary, s["grid"], s["resolution"], "strong", s["platform"])
        peer = {
            r["n_devices"]: r
            for r in strong_rows
            if r["grid"] == s["grid"] and r["resolution"] == s["resolution"]
        }
        dof_k = s["dof"] / 1000 if s["dof"] < 1e6 else s["dof"] / 1e6
        dof_unit = "k" if s["dof"] < 1e6 else "M"

        cols: list[str] = [f"{grid_label}", s["resolution"], f"{dof_k:.0f}{dof_unit}"]
        for nd in [1, 2, 4]:
            if nd in peer:
                cols.append(_fmt(peer[nd]["ms_per_step_median"]))
                cols.append(_fmt(peer[nd]["sypd_median"], ".2f"))
            else:
                cols.extend(["--", "--"])
        cols.append(_fmt(eff.get(4), ".0f"))
        lines.append(" & ".join(cols) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # --- Table 2: Weak scaling ---
    lines.append("% === Table 2: Weak scaling ===")
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Weak scaling: wall-clock time per step (ms) with problem")
    lines.append("  size proportional to device count. Efficiency $\\eta$ measures")
    lines.append("  constant throughput per device.}")
    lines.append("\\label{tab:weak_scaling}")
    lines.append("\\begin{tabular}{ll rrrr}")
    lines.append("\\toprule")
    lines.append("Grid & Devices & Resolution & DOF & ms/step & SYPD \\\\")
    lines.append("\\midrule")

    weak_rows = [s for s in summary if s["scaling"] == "weak"]
    for s in sorted(weak_rows, key=lambda x: (x["grid"], x["n_devices"])):
        grid_label = "Spectral" if s["grid"] == "spectral" else "Icosahedral"
        dof_k = s["dof"] / 1000 if s["dof"] < 1e6 else s["dof"] / 1e6
        dof_unit = "k" if s["dof"] < 1e6 else "M"
        lines.append(
            f"{grid_label} & {s['n_devices']} & {s['resolution']} & "
            f"{dof_k:.0f}{dof_unit} & "
            f"{_fmt(s['ms_per_step_median'])} & "
            f"{_fmt(s['sypd_median'], '.2f')} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # --- Table 3: SYPD at 1-degree ---
    lines.append("% === Table 3: Performance at 1-degree resolution ===")
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Performance at $\\sim$1\\textdegree{} resolution:")
    lines.append("  simulated years per day (SYPD) for both dynamical cores")
    lines.append("  with Held--Suarez gray-atmosphere forcing.}")
    lines.append("\\label{tab:sypd_1deg}")
    lines.append("\\begin{tabular}{llr rrr}")
    lines.append("\\toprule")
    lines.append("Grid & Resolution & $\\Delta x$ &")
    lines.append("  1 device & 4 devices & $\\eta_4$ (\\%) \\\\")
    lines.append("\\midrule")

    one_deg = [
        s for s in summary
        if s["resolution"] in ("T106", "L6") and s["scaling"] == "strong"
    ]
    if one_deg:
        for grid_key, res_key, label in [("spectral", "T106", "Spectral"), ("icosahedral", "L6", "Icosahedral")]:
            rows = {r["n_devices"]: r for r in one_deg if r["grid"] == grid_key}
            if not rows:
                continue
            first_plat = next((r["platform"] for r in one_deg if r["grid"] == grid_key), "cpu")
            eff = _compute_efficiency(summary, grid_key, res_key, "strong", first_plat)
            s1 = rows.get(1, {})
            s4 = rows.get(4, {})
            deg = s1.get("deg", s4.get("deg", ""))
            lines.append(
                f"{label} & {res_key}/L40 & {_fmt(deg)}" + "\\textdegree{}"
                f" & {_fmt(s1.get('sypd_median'), '.2f')}"
                f" & {_fmt(s4.get('sypd_median'), '.2f')}"
                f" & {_fmt(eff.get(4), '.0f')} \\\\"
            )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # --- Table 4: MPI scaling ---
    mpi_rows = [s for s in summary if "mpi" in s.get("scaling", "")]
    if mpi_rows:
        lines.append("% === Table 4: MPI scaling (cubed-sphere) ===")
        lines.append("\\begin{table}[ht]")
        lines.append("\\centering")
        lines.append("\\caption{MPI strong scaling for the cubed-sphere finite-volume")
        lines.append("  dynamical core with Held--Suarez forcing on Levante CPU partition.}")
        lines.append("\\label{tab:mpi_scaling}")
        lines.append("\\begin{tabular}{lrrrrr}")
        lines.append("\\toprule")
        lines.append("Resolution & Ranks & ms/step & Speedup & $\\eta$ (\\%) & SYPD \\\\")
        lines.append("\\midrule")

        for s in sorted(mpi_rows, key=lambda x: (x["resolution"], x["n_devices"])):
            eff = _compute_efficiency(
                summary, s["grid"], s["resolution"], s["scaling"], s["platform"]
            )
            # Compute speedup
            base_rows = [r for r in mpi_rows if r["grid"] == s["grid"]
                         and r["resolution"] == s["resolution"]]
            base_rows.sort(key=lambda x: x["n_devices"])
            base_ms = base_rows[0]["ms_per_step_median"] if base_rows else None
            speedup = (base_ms / s["ms_per_step_median"]) if base_ms and s["ms_per_step_median"] else None

            lines.append(
                f"{s['resolution']} & {s['n_devices']} & "
                f"{_fmt(s['ms_per_step_median'])} & "
                f"{_fmt(speedup, '.1f')}$\\times$ & "
                f"{_fmt(eff.get(s['n_devices']), '.0f')} & "
                f"{_fmt(s['sypd_median'], '.2f')} \\\\"
            )

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown_report(
    results: list[dict], summary: list[dict], path: Path, args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append("# legoESM AMIP Gray-Atmosphere Scaling Benchmark")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Platform: {getattr(args, 'platform', 'N/A')}")
    lines.append(f"- Grids: {getattr(args, 'grids', 'N/A')}")
    lines.append(f"- Scaling: {getattr(args, 'scaling', 'N/A')}")
    lines.append(f"- Repeats: {getattr(args, 'repeats', 'N/A')}")
    lines.append(f"- Iterations: {getattr(args, 'iterations', 'N/A')}")
    lines.append("")

    n_pass = sum(1 for r in results if r.get("status") == "pass")
    n_fail = sum(1 for r in results if r.get("status") == "fail")
    n_total = len(results)
    lines.append(f"## Summary: {n_pass}/{n_total} passed, {n_fail} failed")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| Grid | Resolution | Devices | ms/step | SYPD | Throughput (Mcells/s) | Compile (s) |")
    lines.append("|------|-----------|---------|---------|------|----------------------|-------------|")

    for s in summary:
        lines.append(
            f"| {s['grid']} | {s['resolution']} | {s['n_devices']} | "
            f"{_fmt(s['ms_per_step_median'])} | {_fmt(s['sypd_median'], '.2f')} | "
            f"{_fmt(s['throughput_median'], '.1f')} | "
            f"{_fmt(s['compile_s_median'], '.1f')} |"
        )

    # Efficiency section
    lines.append("")
    lines.append("## Parallel Efficiency")
    lines.append("")

    strong_configs = set(
        (s["grid"], s["resolution"])
        for s in summary if s["scaling"] == "strong"
    )
    for grid, res in sorted(strong_configs):
        plat = next(
            (s["platform"] for s in summary if s["grid"] == grid and s["resolution"] == res),
            "cpu",
        )
        eff = _compute_efficiency(summary, grid, res, "strong", plat)
        if eff:
            eff_str = ", ".join(f"{nd}: {e:.0f}%" for nd, e in sorted(eff.items()))
            lines.append(f"- **{grid} {res}**: {eff_str}")

    # SYPD at 1-degree
    lines.append("")
    lines.append("## SYPD at ~1-degree Resolution")
    lines.append("")
    for grid_key, res_key in [("spectral", "T106"), ("icosahedral", "L6")]:
        rows = [s for s in summary if s["grid"] == grid_key and s["resolution"] == res_key]
        if rows:
            for r in sorted(rows, key=lambda x: x["n_devices"]):
                lines.append(
                    f"- **{grid_key} {res_key}** ({r['n_devices']} dev): "
                    f"{_fmt(r['sypd_median'], '.2f')} SYPD, "
                    f"{_fmt(r['ms_per_step_median'])} ms/step"
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_metadata(meta_dir: Path) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)

    env_capture = {
        k: v for k, v in sorted(os.environ.items())
        if k.startswith(("SLURM_", "CUDA", "NCCL", "OMPI", "PMI", "JAX_", "XLA_"))
    }
    (meta_dir / "env.json").write_text(json.dumps(env_capture, indent=2), encoding="utf-8")

    for fname, cmd in [
        ("git_head.txt", ["git", "rev-parse", "HEAD"]),
        ("python_version.txt", [sys.executable, "--version"]),
        ("nvidia_smi.txt", ["nvidia-smi"]),
        ("lscpu.txt", ["lscpu"]),
    ]:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            (meta_dir / fname).write_text(proc.stdout + proc.stderr, encoding="utf-8")
        except Exception:
            pass

    (meta_dir / "levante_facts.json").write_text(
        json.dumps(LEVANTE_FACTS, indent=2), encoding="utf-8",
    )


# ======================================================================
# SLURM job generation (generate mode)
# ======================================================================

def _render_sbatch(
    *,
    job_name: str,
    partition: str,
    nodes: int,
    ntasks: int,
    gpus_per_node: int,
    time_limit: str,
    account: str,
    run_cmd: str,
    repo_root: Path,
    script_path: Path,
    mode: str,
    modules: str,
    extra_env: str,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks-per-node={ntasks}",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={script_path.with_suffix('.out')}",
        f"#SBATCH --error={script_path.with_suffix('.err')}",
        "#SBATCH --exclusive",
    ]
    if account:
        lines.append(f"#SBATCH --account={account}")
    if mode == "gpu" and gpus_per_node > 0:
        lines.append(f"#SBATCH --gpus-per-node={gpus_per_node}")

    lines.extend([
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo_root))}",
        "export PYTHONUNBUFFERED=1",
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
    ])
    if mode == "gpu":
        lines.append('export JAX_PLATFORMS="${JAX_PLATFORMS:-gpu,cpu}"')
    else:
        lines.append('export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"')

    if modules:
        for m in modules.split(","):
            m = m.strip()
            if m:
                lines.append(f"module load {shlex.quote(m)}")

    if extra_env:
        for token in extra_env.split(","):
            token = token.strip()
            if "=" in token:
                lines.append(f"export {token}")

    lines.extend(["", f"echo '[amip-scaling] {job_name}'", run_cmd])
    return "\n".join(lines) + "\n"


def _generate_slurm(args: argparse.Namespace) -> int:
    """Generate SLURM batch scripts for Levante."""
    repo_root = Path(__file__).resolve().parents[1]
    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (repo_root / args.output / tag).resolve()
    jobs_dir = out / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out / "runs"
    results_dir.mkdir(parents=True, exist_ok=True)

    script_rel = "scripts/run_levante_amip_scaling.py"
    python_exe = args.python
    idx = 0
    manifest: list[dict[str, Any]] = []

    # --- GPU jobs: strong scaling ---
    for grid, strong_res, platform_str in [
        ("spectral", SPECTRAL_STRONG_RESOLUTIONS, "gpu"),
        ("icosahedral", ICOSAHEDRAL_STRONG_RESOLUTIONS, "gpu"),
    ]:
        for res in strong_res:
            idx += 1
            job_name = f"amip_{grid[:4]}_{res}_strong_gpu"
            run_out = results_dir / f"{idx:02d}_{job_name}"
            cmd = (
                f"{python_exe} {script_rel} run"
                f" --grids {grid} --resolutions {res}"
                f" --devices 1,2,4 --platform gpu --scaling strong"
                f" --repeats {args.repeats} --iterations {args.iterations}"
                f" --warmup {args.warmup} --x64"
                f" --output {run_out}"
            )
            script_path = jobs_dir / f"{idx:02d}_{job_name}.sbatch"
            text = _render_sbatch(
                job_name=job_name, partition=args.gpu_partition, nodes=1,
                ntasks=4, gpus_per_node=4, time_limit=args.time_limit,
                account=args.account, run_cmd=cmd, repo_root=repo_root,
                script_path=script_path, mode="gpu", modules=args.modules,
                extra_env=args.extra_env,
            )
            script_path.write_text(text, encoding="utf-8")
            manifest.append({"index": idx, "job": job_name, "script": str(script_path)})

    # --- GPU jobs: weak scaling ---
    for grid, ladder in [("spectral", SPECTRAL_WEAK_LADDER), ("icosahedral", ICOSAHEDRAL_WEAK_LADDER)]:
        idx += 1
        gpu_ladder_entries = [(nd, r) for nd, r in ladder if nd <= 4]
        devs = ",".join(str(nd) for nd, _ in gpu_ladder_entries)
        ress = ",".join(r for _, r in gpu_ladder_entries)
        job_name = f"amip_{grid[:4]}_weak_gpu"
        run_out = results_dir / f"{idx:02d}_{job_name}"
        cmd = (
            f"{python_exe} {script_rel} run"
            f" --grids {grid} --resolutions {ress}"
            f" --devices {devs} --platform gpu --scaling weak"
            f" --repeats {args.repeats} --iterations {args.iterations}"
            f" --warmup {args.warmup} --x64"
            f" --output {run_out}"
        )
        script_path = jobs_dir / f"{idx:02d}_{job_name}.sbatch"
        text = _render_sbatch(
            job_name=job_name, partition=args.gpu_partition, nodes=1,
            ntasks=4, gpus_per_node=4, time_limit=args.time_limit,
            account=args.account, run_cmd=cmd, repo_root=repo_root,
            script_path=script_path, mode="gpu", modules=args.modules,
            extra_env=args.extra_env,
        )
        script_path.write_text(text, encoding="utf-8")
        manifest.append({"index": idx, "job": job_name, "script": str(script_path)})

    # --- CPU jobs: strong scaling ---
    cpu_devices = "1,2,4,8,16,32,64,128"
    for grid, strong_res in [
        ("spectral", SPECTRAL_STRONG_RESOLUTIONS),
        ("icosahedral", ICOSAHEDRAL_STRONG_RESOLUTIONS),
    ]:
        for res in strong_res:
            idx += 1
            job_name = f"amip_{grid[:4]}_{res}_strong_cpu"
            run_out = results_dir / f"{idx:02d}_{job_name}"
            cmd = (
                f"{python_exe} {script_rel} run"
                f" --grids {grid} --resolutions {res}"
                f" --devices {cpu_devices} --platform cpu --scaling strong"
                f" --repeats {args.repeats} --iterations {args.iterations}"
                f" --warmup {args.warmup} --x64"
                f" --output {run_out}"
            )
            script_path = jobs_dir / f"{idx:02d}_{job_name}.sbatch"
            text = _render_sbatch(
                job_name=job_name, partition=args.cpu_partition, nodes=1,
                ntasks=128, gpus_per_node=0, time_limit=args.time_limit,
                account=args.account, run_cmd=cmd, repo_root=repo_root,
                script_path=script_path, mode="cpu", modules=args.modules,
                extra_env=args.extra_env,
            )
            script_path.write_text(text, encoding="utf-8")
            manifest.append({"index": idx, "job": job_name, "script": str(script_path)})

    # --- CPU jobs: weak scaling ---
    for grid, ladder in [("spectral", SPECTRAL_WEAK_LADDER), ("icosahedral", ICOSAHEDRAL_WEAK_LADDER)]:
        idx += 1
        cpu_ladder_entries = [(nd, r) for nd, r in ladder if nd <= 128]
        devs = ",".join(str(nd) for nd, _ in cpu_ladder_entries)
        ress = ",".join(r for _, r in cpu_ladder_entries)
        job_name = f"amip_{grid[:4]}_weak_cpu"
        run_out = results_dir / f"{idx:02d}_{job_name}"
        cmd = (
            f"{python_exe} {script_rel} run"
            f" --grids {grid} --resolutions {ress}"
            f" --devices {devs} --platform cpu --scaling weak"
            f" --repeats {args.repeats} --iterations {args.iterations}"
            f" --warmup {args.warmup} --x64"
            f" --output {run_out}"
        )
        script_path = jobs_dir / f"{idx:02d}_{job_name}.sbatch"
        text = _render_sbatch(
            job_name=job_name, partition=args.cpu_partition, nodes=1,
            ntasks=128, gpus_per_node=0, time_limit=args.time_limit,
            account=args.account, run_cmd=cmd, repo_root=repo_root,
            script_path=script_path, mode="cpu", modules=args.modules,
            extra_env=args.extra_env,
        )
        script_path.write_text(text, encoding="utf-8")
        manifest.append({"index": idx, "job": job_name, "script": str(script_path)})

    # --- MPI jobs: cubed-sphere strong scaling ---
    for cs_n, mpi_ranks in [(48, "1,2,3,6"), (96, "1,2,3,6,24")]:
        idx += 1
        job_name = f"amip_cs_C{cs_n}_strong_mpi"
        run_out = results_dir / f"{idx:02d}_{job_name}"
        max_ranks = max(int(r) for r in mpi_ranks.split(","))
        n_nodes = max(1, (max_ranks + 127) // 128)
        cmd = (
            f"{python_exe} {script_rel} run"
            f" --grids spectral --resolutions T42 --devices 1"  # placeholder grid
            f" --platform cpu --scaling strong"
            f" --repeats {args.repeats} --iterations {args.mpi_iterations}"
            f" --warmup {args.mpi_warmup} --x64"
            f" --mpi-ranks {mpi_ranks}"
            f" --mpi-cs-sizes {cs_n}"
            f" --mpi-launcher {args.mpi_launcher}"
            f" --output {run_out}"
        )
        script_path = jobs_dir / f"{idx:02d}_{job_name}.sbatch"
        text = _render_sbatch(
            job_name=job_name, partition=args.cpu_partition, nodes=n_nodes,
            ntasks=max_ranks, gpus_per_node=0, time_limit=args.time_limit,
            account=args.account, run_cmd=cmd, repo_root=repo_root,
            script_path=script_path, mode="cpu", modules=args.modules,
            extra_env=args.extra_env,
        )
        script_path.write_text(text, encoding="utf-8")
        manifest.append({"index": idx, "job": job_name, "script": str(script_path)})

    # --- Save manifest ---
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "output_root": str(out),
        "levante_facts": LEVANTE_FACTS,
        "jobs": manifest,
    }, indent=2), encoding="utf-8")

    # --- Submit if requested ---
    if args.submit:
        for rec in manifest:
            proc = subprocess.run(
                ["sbatch", rec["script"]], capture_output=True, text=True, check=False,
            )
            rec["submit_status"] = "pass" if proc.returncode == 0 else "fail"
            rec["submit_stdout"] = proc.stdout.strip()
            if proc.returncode != 0:
                rec["submit_stderr"] = proc.stderr.strip()
        manifest_path.write_text(json.dumps({
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "tag": tag, "output_root": str(out),
            "levante_facts": LEVANTE_FACTS, "jobs": manifest,
        }, indent=2), encoding="utf-8")

    print("=" * 72)
    print("AMIP Gray-Atmosphere Scaling Job Generator")
    print("=" * 72)
    print(f"Output: {out}")
    print(f"Jobs generated: {len(manifest)}")
    print(f"Submitted: {args.submit}")
    for rec in manifest:
        sub = rec.get("submit_stdout", "not submitted")
        print(f"  [{rec['index']:2d}] {rec['job']:40s} {sub}")
    print("=" * 72)
    return 0


# ======================================================================
# Collation (collate mode)
# ======================================================================

def _collate(args: argparse.Namespace) -> int:
    """Aggregate results from multiple run directories."""
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    all_results: list[dict] = []

    # Scan for results.json files
    for results_file in sorted(input_dir.rglob("results.json")):
        try:
            data = json.loads(results_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                all_results.extend(data)
            elif isinstance(data, dict) and "results" in data:
                all_results.extend(data["results"])
        except Exception as exc:
            print(f"[WARN] Failed to parse {results_file}: {exc}", file=sys.stderr)

    if not all_results:
        print("No results found.", file=sys.stderr)
        return 1

    output = Path(args.collate_output or input_dir)
    output.mkdir(parents=True, exist_ok=True)

    _write_raw_csv(all_results, output / "paper_raw.csv")
    summary = _summarize_results(all_results)
    _write_summary_csv(summary, output / "paper_summary.csv")
    _write_latex_tables(summary, output / "paper_tables.tex")

    # Minimal markdown for collation (no args context)
    ns = argparse.Namespace(
        platform="mixed", grids="mixed", scaling="mixed",
        repeats="N/A", iterations="N/A",
    )
    _write_markdown_report(all_results, summary, output / "paper_report.md", ns)

    n_pass = sum(1 for r in all_results if r.get("status") == "pass")
    print(f"Collated {len(all_results)} results ({n_pass} pass)")
    print(f"  CSV:   {output / 'paper_raw.csv'}")
    print(f"  LaTeX: {output / 'paper_tables.tex'}")
    print(f"  MD:    {output / 'paper_report.md'}")
    return 0


# ======================================================================
# CLI
# ======================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AMIP gray-atmosphere scaling benchmarks for Levante.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- worker ---
    pw = sub.add_parser("worker", help="Run a single benchmark point (subprocess)")
    pw.add_argument("--grid", required=True, choices=["spectral", "icosahedral", "cubed_sphere_mpi"])
    pw.add_argument("--resolution", required=True)
    pw.add_argument("--iterations", type=int, default=50)
    pw.add_argument("--warmup", type=int, default=5)
    pw.add_argument("--cs-grid-size", type=int, default=48)
    pw.add_argument("--cs-nlev", type=int, default=40)
    pw.add_argument("--cs-dt", type=float, default=600.0)

    # --- run ---
    pr = sub.add_parser("run", help="Run scaling suite (spawns worker subprocesses)")
    pr.add_argument("--grids", default="spectral,icosahedral",
                     help="Comma-separated grid types (default: spectral,icosahedral)")
    pr.add_argument("--resolutions", default="T42,T85,T106,L4,L5,L6",
                     help="Comma-separated resolution keys")
    pr.add_argument("--devices", default="1,2,4",
                     help="Comma-separated device counts (default: 1,2,4)")
    pr.add_argument("--platform", default="cpu", choices=["cpu", "gpu"],
                     help="JAX platform (default: cpu)")
    pr.add_argument("--scaling", default="strong", choices=["strong", "weak"],
                     help="Scaling type (default: strong)")
    pr.add_argument("--repeats", type=int, default=3)
    pr.add_argument("--iterations", type=int, default=50)
    pr.add_argument("--warmup", type=int, default=5)
    pr.add_argument("--timeout", type=float, default=3600.0)
    pr.add_argument("--python", default=sys.executable)
    pr.add_argument("--x64", action="store_true", default=True)
    pr.add_argument("--output", default="")
    # MPI options
    pr.add_argument("--mpi-ranks", default="",
                     help="Comma-separated MPI rank counts for cubed-sphere (empty = skip MPI)")
    pr.add_argument("--mpi-cs-sizes", default="48,96",
                     help="Cubed-sphere grid sizes for MPI benchmarks")
    pr.add_argument("--mpi-launcher", default="mpirun")
    pr.add_argument("--mpi-iterations", type=int, default=30)
    pr.add_argument("--mpi-warmup", type=int, default=3)
    pr.add_argument("--no-mpi", action="store_true")

    # --- generate ---
    pg = sub.add_parser("generate", help="Generate SLURM batch scripts for Levante")
    pg.add_argument("--account", required=True, help="SLURM account")
    pg.add_argument("--output", default="results/levante_amip_scaling")
    pg.add_argument("--tag", default="")
    pg.add_argument("--submit", action="store_true")
    pg.add_argument("--python", default=".venv/bin/python")
    pg.add_argument("--cpu-partition", default="compute")
    pg.add_argument("--gpu-partition", default="gpu")
    pg.add_argument("--time-limit", default="02:00:00")
    pg.add_argument("--modules", default="", help="Comma-separated modules to load")
    pg.add_argument("--extra-env", default="", help="Extra KEY=VALUE env vars (comma-separated)")
    pg.add_argument("--repeats", type=int, default=3)
    pg.add_argument("--iterations", type=int, default=50)
    pg.add_argument("--warmup", type=int, default=5)
    pg.add_argument("--mpi-iterations", type=int, default=30)
    pg.add_argument("--mpi-warmup", type=int, default=3)
    pg.add_argument("--mpi-launcher", default="mpirun")

    # --- collate ---
    pc = sub.add_parser("collate", help="Aggregate results into paper artifacts")
    pc.add_argument("--input", required=True, help="Directory with results.json files")
    pc.add_argument("--collate-output", default="", help="Output directory (default: same as input)")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "worker":
        return _run_worker(args)
    elif args.command == "run":
        return _run_suite(args)
    elif args.command == "generate":
        return _generate_slurm(args)
    elif args.command == "collate":
        return _collate(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
