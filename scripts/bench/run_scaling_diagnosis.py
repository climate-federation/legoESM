#!/usr/bin/env python
"""Comprehensive scaling diagnostics for Levante (DKRZ).

Runs systematic profiling of legoESM at various GPU/MPI configurations,
producing structured JSON reports that can be analyzed offline by Claude Code.

Unlike ``run_levante_gpu_scaling.py`` which measures aggregate SYPD/throughput,
this script focuses on **bottleneck identification**: per-phase timing breakdown,
halo exchange bandwidth, reduction latency, memory utilization, compute/comm
overlap potential, and roofline model data.

Modes
-----
  full       -- Run all diagnostics (default)
  quick      -- Abbreviated run for sanity checks (~2 min)
  halo-only  -- Isolate halo exchange profiling
  reduction  -- Isolate MPI reduction profiling
  roofline   -- Collect roofline model data
  profile    -- Capture XLA profile traces (for TensorBoard)

Usage
-----
Single-node multi-GPU::

    python scripts/run_scaling_diagnosis.py --grid cubed-sphere --n 48 --nlev 40

MPI multi-node::

    mpirun -np 6 python scripts/run_scaling_diagnosis.py \
        --grid cubed-sphere --n 48 --nlev 40

Quick sanity check::

    python scripts/run_scaling_diagnosis.py --mode quick --n 24

All diagnostics with XLA profile::

    mpirun -np 6 python scripts/run_scaling_diagnosis.py \
        --mode full --grid cubed-sphere --n 96 --nlev 40 --xla-profile

Output
------
Creates ``results/scaling_diagnosis/<timestamp>/`` containing:

  - ``diag_rank<N>.json``: Per-rank diagnostic report
  - ``summary.json``: Aggregated cross-rank summary (rank 0 only)
  - ``halo_profile.json``: Detailed halo exchange profiling
  - ``reduction_profile.json``: MPI reduction profiling
  - ``roofline.json``: Roofline model data
  - ``xla_profile_rank<N>/``: XLA/TensorBoard traces (if --xla-profile)
  - ``env.json``: Full environment snapshot

All files are self-contained and designed for offline analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# JAX configuration — must happen before jax import
# ---------------------------------------------------------------------------

def _configure_env(precision: str = "float64"):
    """Set JAX env vars before import.

    Iter 24: stop forcing ``JAX_PLATFORMS=gpu,cpu`` (rejected by JAX
    0.10+, see iter 21 fix in ``run_levante_gpu_scaling.py``).  Wrappers
    that need a specific backend should ``export JAX_PLATFORMS=cuda,cpu``
    (NVIDIA) or ``rocm,cpu`` (AMD) before invoking the script.

    Also ensure the project root is on ``sys.path`` so that
    ``tests.test_cases`` (canonical IC location) imports cleanly when
    the script is run as a standalone executable rather than via
    ``pytest`` from the repo root.
    """
    if precision == "float64":
        os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

    # Make the repository root importable so ``tests.test_cases`` works
    # when the script is launched as ``python scripts/...`` (no
    # ``PYTHONPATH=$PWD``).
    import sys
    from pathlib import Path
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    # GPU affinity for MPI
    local_rank = (
        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or os.environ.get("MV2_COMM_WORLD_LOCAL_RANK")
        or os.environ.get("SLURM_LOCALID")
    )
    if local_rank is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = local_rank

    # Enable profiling-friendly XLA flags when running on GPU (or
    # auto-detect / unset).  Skip when ``JAX_PLATFORMS`` is explicitly
    # ``cpu`` or ``tpu`` to avoid pinging GPU-specific flags that the
    # chosen backend rejects.
    platforms = os.environ.get("JAX_PLATFORMS", "")
    is_gpu_run = (
        not platforms
        or any(tok in platforms for tok in ("gpu", "cuda", "rocm"))
    )
    if is_gpu_run:
        xla_flags = os.environ.get("XLA_FLAGS", "")
        for flag in [
            "--xla_gpu_enable_latency_hiding_scheduler=true",
        ]:
            if flag not in xla_flags:
                xla_flags = f"{xla_flags} {flag}" if xla_flags else flag
        os.environ["XLA_FLAGS"] = xla_flags


def _init_distributed(grid_type: str, global_n: int | None = None):
    """Detect and initialize distributed JAX/MPI.

    Returns (rank, world_size).
    """
    if "OMPI_COMM_WORLD_SIZE" in os.environ or "PMI_SIZE" in os.environ:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            n_procs = comm.Get_size()

            if grid_type == "cubed-sphere":
                from legoesm.parallel.distributed import initialize_distributed
                initialize_distributed(global_n=global_n)
            elif n_procs > 1:
                import socket
                import jax
                hostnames = comm.allgather(socket.gethostname())
                if len(set(hostnames)) > 1:
                    from legoesm.parallel.early_init import (
                        resolve_coordinator_port,
                    )
                    _port = resolve_coordinator_port()
                    jax.distributed.initialize(
                        coordinator_address=f"{hostnames[0]}:{_port}",
                        num_processes=n_procs,
                        process_id=rank,
                    )
            return rank, n_procs
        except ImportError:
            pass

    slurm_ntasks = os.environ.get("SLURM_NTASKS")
    if slurm_ntasks and int(slurm_ntasks) > 1:
        import jax
        jax.distributed.initialize()
        return jax.process_index(), jax.process_count()

    return 0, 1


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------

def _env_snapshot() -> dict:
    """Capture full environment for reproducibility."""
    import jax
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": os.environ.get("HOSTNAME", os.uname().nodename),
        "python": sys.version,
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "n_local_devices": len(jax.local_devices()),
        "devices": [str(d) for d in jax.local_devices()],
        "env_vars": {
            k: v for k, v in sorted(os.environ.items())
            if any(k.startswith(p) for p in
                   ["JAX", "XLA", "CUDA", "SLURM", "OMPI", "PMI", "NCCL",
                    "MPI", "LEGOESM"])
        },
    }
    # nvidia-smi
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "-q", "--display=MEMORY,UTILIZATION,CLOCKS"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            env["nvidia_smi"] = r.stdout[:5000]
    except Exception:
        pass

    # NCCL / UCX info
    try:
        import subprocess
        r = subprocess.run(["nccl-info"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            env["nccl_info"] = r.stdout[:2000]
    except Exception:
        pass

    return env


# ---------------------------------------------------------------------------
# Test case setup (reuses patterns from run_levante_gpu_scaling.py)
# ---------------------------------------------------------------------------

def _setup_cubed_sphere(n: int, nlev: int, dt: float, precision: str,
                        physics_level: str, rank: int, world_size: int):
    """Set up cubed-sphere benchmark case. Returns (step_fn, state, config_info).

    Iter 23 cleanup: ported from the legacy
    ``create_cubed_sphere_grid`` / ``SigmaCoordinate`` /
    ``PrimitiveEquationModel`` / ``jablonowski_williamson_ic`` API
    surface to the current modules.  The diagnosis script had bit-
    rotted; codex iter 23 quick-smoke confirmed import-time crashes
    on a clean install before reaching any benchmark code.
    """
    import jax
    import jax.numpy as jnp
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.grids.vertical import create_sigma_coordinate

    dtype = jnp.float64 if precision == "float64" else jnp.float32

    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sigma = create_sigma_coordinate(nlev)

    # Jablonowski-Williamson baroclinic wave IC (canonical helper now
    # lives under tests/test_cases — same routine the GPU/CPU scaling
    # drivers use).
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
        CDGridPrimitiveEquationConfig,
        hydrostatic_to_fv3,
    )
    state_cc = baroclinic_wave_init(grid, sigma, perturbed=True)

    config = CDGridPrimitiveEquationConfig()
    model = CDGridPrimitiveEquationModel(grid, sigma, config)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    # Physics: Held-Suarez forcing not wired here; the diagnosis script
    # focuses on parallel-runtime bottlenecks (halo bandwidth, reduction
    # latency, etc.) rather than physics throughput.  ``physics_level``
    # is accepted for API compatibility but ignored.
    physics_fn = None

    # Distributed step function.  Iter 23 cleanup: the legacy
    # ``create_device_config`` factory is gone; use ``detect_devices`` +
    # ``configure_jax_for_device`` (the canonical entry point in
    # ``parallel.device_config``).  When more than one device is
    # visible *and* we are not under MPI, route through the sharded
    # step; under MPI use the bare model step (each rank handles its
    # own JAX devices).
    from legoesm.parallel.device_config import detect_devices
    dev_config = detect_devices()
    n_local_devices = dev_config.devices_per_host

    if world_size > 1:
        step_fn = model.step
    elif n_local_devices > 1:
        from legoesm.parallel.sharded_dynamics import make_sharded_step
        from legoesm.parallel.mesh import create_device_mesh, shard_pytree
        mesh_cfg = create_device_mesh(n_devices=n_local_devices)
        step_fn = make_sharded_step(model, mesh_cfg, n=n, nlev=nlev)
        # CompiledShardedStep now constrains inputs via in_shardings:
        # commit the initial state to the face sharding up front.
        state = shard_pytree(state, mesh_cfg)
    else:
        step_fn = model.step

    # Wrap physics (currently unused; left for forward compatibility).
    if physics_fn is not None:
        _step = step_fn
        _phys = physics_fn
        step_fn = lambda s, dt_val: _step(s, dt_val, physics_fn=_phys)

    config_info = {
        "grid_type": "cubed-sphere",
        "n": n,
        "nlev": nlev,
        "dt": dt,
        "precision": precision,
        "physics": physics_level,
        "total_cells": 6 * n * n,
        "cells_per_rank": 6 * n * n // max(world_size, 1),
    }

    return step_fn, state, config_info, grid, model


def _setup_halo_fn(grid, state, rank, world_size):
    """Create a standalone halo exchange function for overlap estimation."""
    import jax
    from legoesm.grids.halo import pad_halo_4d

    # Extract a representative 4D field from state for halo profiling
    if hasattr(state, 'u'):
        field = state.u
    elif isinstance(state, dict) and 'u' in state:
        field = state['u']
    else:
        leaves = jax.tree.leaves(state)
        field = leaves[0] if leaves else None

    if field is None:
        return lambda s: s

    def halo_fn(s):
        leaves = jax.tree.leaves(s)
        # Just pad the first 4D field as a representative operation
        for leaf in leaves:
            if leaf.ndim == 4:
                _ = pad_halo_4d(leaf, halo=1)
                break
        return s

    return halo_fn


# ---------------------------------------------------------------------------
# Diagnostic routines
# ---------------------------------------------------------------------------

def run_step_profiling(
    step_fn, state, dt: float,
    n_warmup: int, n_steps: int,
    harness,
    rank: int, world_size: int,
) -> dict:
    """Profile per-step timing with device sync."""
    import jax
    from legoesm.parallel.scaling_diagnostics import make_instrumented_step

    instr_step = make_instrumented_step(step_fn, harness)

    # JIT compile
    t0 = time.perf_counter()
    state = step_fn(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    compile_time = time.perf_counter() - t0
    harness.record_compilation("step_fn", compile_time)

    # Memory after compilation
    harness.memory_snapshot("post_compile")

    # Warmup
    for _ in range(n_warmup):
        state = step_fn(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    harness.memory_snapshot("post_warmup")

    # MPI barrier before timing
    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_size() > 1:
            MPI.COMM_WORLD.Barrier()
    except ImportError:
        pass

    # Profiled steps (with device sync per step for accurate timing)
    for i in range(n_steps):
        state = instr_step(state, dt)

    # MPI barrier after timing
    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_size() > 1:
            MPI.COMM_WORLD.Barrier()
    except ImportError:
        pass

    harness.memory_snapshot("post_timing")

    return {"compile_time_s": round(compile_time, 3), "n_steps": n_steps}


def run_scan_timing(
    step_fn, state, dt: float,
    n_warmup: int, n_timing: int,
    rank: int, world_size: int,
) -> dict:
    """Run scan-based timing (same as scaling benchmark) for reference.

    This gives the 'production' throughput without per-step sync overhead.
    """
    import jax

    # JIT compile
    t0 = time.perf_counter()
    state = step_fn(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    compile_time = time.perf_counter() - t0

    # Warmup
    for _ in range(n_warmup):
        state = step_fn(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))

    # Build scan runner
    input_dtypes = jax.tree.map(
        lambda x: x.dtype if hasattr(x, "dtype") else None, state)

    @jax.jit
    def _scan_run(st, dt_val):
        def _body(carry, _):
            new = step_fn(carry, dt_val)
            new = jax.tree.map(
                lambda x, d: x.astype(d)
                if d is not None and hasattr(x, "astype") else x,
                new, input_dtypes,
            )
            return new, None
        return jax.lax.scan(_body, st, None, length=n_timing)[0]

    # Pre-compile scan
    state = _scan_run(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))

    # Sync
    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_size() > 1:
            MPI.COMM_WORLD.Barrier()
    except ImportError:
        pass

    # Timed run
    t0 = time.perf_counter()
    state = _scan_run(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))

    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_size() > 1:
            MPI.COMM_WORLD.Barrier()
    except ImportError:
        pass
    t1 = time.perf_counter()

    total_time = t1 - t0
    time_per_step_ms = total_time / n_timing * 1000
    sypd = (dt / (total_time / n_timing)) / (365.25 * 86400) * 86400

    return {
        "compile_time_s": round(compile_time, 3),
        "scan_total_s": round(total_time, 4),
        "time_per_step_ms": round(time_per_step_ms, 3),
        "sypd": round(sypd, 4),
        "n_timing": n_timing,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Comprehensive scaling diagnostics for legoESM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["full", "quick", "halo-only",
                                       "reduction", "roofline", "profile"],
                   default="full")
    p.add_argument("--grid", choices=["cubed-sphere"], default="cubed-sphere",
                   help="Grid type (cubed-sphere supported for now)")
    p.add_argument("--n", type=int, default=48, help="Per-face resolution")
    p.add_argument("--nlev", type=int, default=40, help="Vertical levels")
    p.add_argument("--dt", type=float, default=600.0, help="Timestep (s)")
    p.add_argument("--precision", choices=["float32", "float64"],
                   default="float64")
    p.add_argument("--physics", choices=["none", "held_suarez"],
                   default="none")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-steps", type=int, default=50,
                   help="Steps for per-step profiling")
    p.add_argument("--n-timing", type=int, default=100,
                   help="Steps for scan-based throughput measurement")
    p.add_argument("--xla-profile", action="store_true",
                   help="Capture XLA/TensorBoard profile traces")
    p.add_argument("--output-dir", default="results/scaling_diagnosis")
    p.add_argument("--no-timestamp", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # ---------------------------------------------------------------
    # Environment setup
    # ---------------------------------------------------------------
    _configure_env(args.precision)

    import jax
    import jax.numpy as jnp

    rank, world_size = _init_distributed(args.grid, global_n=args.n)
    is_rank0 = (rank == 0)

    from legoesm.parallel.scaling_diagnostics import (
        DiagnosticHarness,
        collect_roofline_data,
        profile_halo_exchange,
        profile_reductions,
        capture_xla_profile,
        estimate_overlap_potential,
    )

    # Output directory
    output_dir = Path(args.output_dir)
    if not args.no_timestamp:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = output_dir / ts
    if is_rank0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Sync dir creation across ranks
    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_size() > 1:
            MPI.COMM_WORLD.Barrier()
    except ImportError:
        pass

    if is_rank0:
        print("=" * 72)
        print("  legoESM Scaling Diagnostics")
        print("=" * 72)
        print(f"  Mode:        {args.mode}")
        print(f"  Grid:        {args.grid} N={args.n} L={args.nlev}")
        print(f"  Precision:   {args.precision}")
        print(f"  Physics:     {args.physics}")
        print(f"  Ranks:       {world_size}")
        print(f"  Backend:     {jax.default_backend()}")
        print(f"  Devices:     {jax.local_devices()}")
        print(f"  Output:      {output_dir}")
        print("=" * 72)

    # ---------------------------------------------------------------
    # Save environment snapshot
    # ---------------------------------------------------------------
    if is_rank0:
        env = _env_snapshot()
        with open(output_dir / "env.json", "w") as f:
            json.dump(env, f, indent=2)

    # ---------------------------------------------------------------
    # Quick mode overrides
    # ---------------------------------------------------------------
    if args.mode == "quick":
        args.n_warmup = 2
        args.n_steps = 10
        args.n_timing = 20

    # ---------------------------------------------------------------
    # Set up test case
    # ---------------------------------------------------------------
    if is_rank0:
        print("\n[1/6] Setting up test case...", flush=True)

    step_fn, state, config_info, grid, model = _setup_cubed_sphere(
        args.n, args.nlev, args.dt, args.precision,
        args.physics, rank, world_size,
    )

    if is_rank0:
        print(f"  Total cells: {config_info['total_cells']:,}")
        print(f"  Cells/rank:  {config_info['cells_per_rank']:,}")

    # ---------------------------------------------------------------
    # Create harness
    # ---------------------------------------------------------------
    harness = DiagnosticHarness(
        rank=rank, world_size=world_size,
        grid_type=args.grid, config=config_info,
    )
    harness.memory_snapshot("initial")

    # ---------------------------------------------------------------
    # [2] Per-step profiling
    # ---------------------------------------------------------------
    if args.mode in ("full", "quick"):
        if is_rank0:
            print(f"\n[2/6] Per-step profiling ({args.n_steps} steps)...",
                  flush=True)
        step_info = run_step_profiling(
            step_fn, state, args.dt,
            args.n_warmup, args.n_steps,
            harness, rank, world_size,
        )
        if is_rank0:
            print(f"  Compile: {step_info['compile_time_s']:.2f}s")

    # ---------------------------------------------------------------
    # [3] Scan-based throughput (reference)
    # ---------------------------------------------------------------
    scan_info = None
    if args.mode in ("full", "quick"):
        if is_rank0:
            print(f"\n[3/6] Scan throughput ({args.n_timing} steps)...",
                  flush=True)
        # Re-init state for clean scan run
        _, state2, _, _, _ = _setup_cubed_sphere(
            args.n, args.nlev, args.dt, args.precision,
            args.physics, rank, world_size,
        )
        scan_info = run_scan_timing(
            step_fn, state2, args.dt,
            args.n_warmup, args.n_timing,
            rank, world_size,
        )
        if is_rank0:
            print(f"  {scan_info['time_per_step_ms']:.2f} ms/step, "
                  f"SYPD={scan_info['sypd']:.3f}")

    # ---------------------------------------------------------------
    # [4] Halo exchange profiling
    # ---------------------------------------------------------------
    halo_info = None
    if args.mode in ("full", "quick", "halo-only"):
        if is_rank0:
            print("\n[4/6] Halo exchange profiling...", flush=True)
        halo_info = profile_halo_exchange(
            args.n, args.nlev, halo=1,
            n_warmup=5, n_iters=50 if args.mode != "quick" else 10,
            rank=rank, world_size=world_size,
        )
        if is_rank0:
            for k, v in halo_info.items():
                if isinstance(v, dict) and "mean_us" in v:
                    print(f"  {k}: {v['mean_us']:.1f} us, "
                          f"BW={v.get('bandwidth_gb_s', '?')} GB/s")

    # ---------------------------------------------------------------
    # [5] Reduction profiling
    # ---------------------------------------------------------------
    reduction_info = None
    if args.mode in ("full", "reduction"):
        if is_rank0:
            print("\n[5/6] MPI reduction profiling...", flush=True)
        reduction_info = profile_reductions(
            args.n, args.nlev, n_iters=100,
            rank=rank, world_size=world_size,
        )
        if is_rank0:
            for k, v in reduction_info.items():
                print(f"  {k}: local={v['local_sum_mean_us']} us, "
                      f"mpi={v.get('mpi_sum_mean_us', 'N/A')} us")

    # ---------------------------------------------------------------
    # [6] Roofline / overlap estimation
    # ---------------------------------------------------------------
    roofline_info = None
    overlap_info = None
    if args.mode in ("full", "roofline"):
        if is_rank0:
            print("\n[6/6] Roofline & overlap estimation...", flush=True)

        # Re-init for clean measurement
        _, state3, _, grid3, _ = _setup_cubed_sphere(
            args.n, args.nlev, args.dt, args.precision,
            args.physics, rank, world_size,
        )
        roofline_info = collect_roofline_data(
            step_fn, state3, args.dt,
            n_steps=20 if args.mode != "quick" else 5,
            rank=rank,
        )
        if is_rank0:
            print(f"  State: {roofline_info['state_mb']:.1f} MB, "
                  f"BW est: {roofline_info['bandwidth_estimate_gb_s']:.1f} GB/s")

        # Overlap estimation
        halo_fn = _setup_halo_fn(grid3, state3, rank, world_size)
        overlap_info = estimate_overlap_potential(
            step_fn, halo_fn, state3, args.dt,
            n_iters=15 if args.mode != "quick" else 5,
            rank=rank,
        )
        if is_rank0:
            print(f"  Halo fraction: {overlap_info['halo_fraction_pct']:.1f}%")

    # ---------------------------------------------------------------
    # XLA profile (optional)
    # ---------------------------------------------------------------
    xla_info = None
    if args.xla_profile:
        if is_rank0:
            print("\n[+] Capturing XLA profile...", flush=True)
        _, state_xla, _, _, _ = _setup_cubed_sphere(
            args.n, args.nlev, args.dt, args.precision,
            args.physics, rank, world_size,
        )
        xla_info = capture_xla_profile(
            step_fn, state_xla, args.dt,
            str(output_dir), n_steps=10, rank=rank,
        )
        if is_rank0:
            print(f"  Profile: {xla_info.get('status', 'unknown')}")

    # ---------------------------------------------------------------
    # Build and save reports
    # ---------------------------------------------------------------
    extra = {}
    if scan_info:
        extra["scan_throughput"] = scan_info
    if halo_info:
        extra["halo_profile"] = halo_info
    if reduction_info:
        extra["reduction_profile"] = reduction_info
    if roofline_info:
        extra["roofline"] = roofline_info
    if overlap_info:
        extra["overlap_estimate"] = overlap_info
    if xla_info:
        extra["xla_profile"] = xla_info

    harness.dump(output_dir / f"diag_rank{rank}.json", extra=extra)

    # Also save individual profile files for convenience
    if is_rank0:
        if halo_info:
            with open(output_dir / "halo_profile.json", "w") as f:
                json.dump(halo_info, f, indent=2)
        if reduction_info:
            with open(output_dir / "reduction_profile.json", "w") as f:
                json.dump(reduction_info, f, indent=2)
        if roofline_info:
            with open(output_dir / "roofline.json", "w") as f:
                json.dump(roofline_info, f, indent=2)

    # ---------------------------------------------------------------
    # Cross-rank summary (rank 0 gathers)
    # ---------------------------------------------------------------
    if is_rank0 and world_size > 1:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            # Gather scan timing from all ranks
            local_data = {
                "rank": rank,
                "phase_timing": harness.timer.summary(),
            }
            all_data = comm.gather(local_data, root=0)
            if all_data:
                summary = {
                    "world_size": world_size,
                    "config": config_info,
                    "per_rank_timing": all_data,
                    "scan_throughput": scan_info,
                }
                with open(output_dir / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2, default=str)
        except Exception as e:
            print(f"  Warning: cross-rank summary failed: {e}")
    elif is_rank0 and world_size == 1:
        summary = {
            "world_size": 1,
            "config": config_info,
            "scan_throughput": scan_info,
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    if is_rank0:
        print(f"\n{'='*72}")
        print(f"  Diagnostics complete. Results in: {output_dir}/")
        print(f"  Files:")
        for p in sorted(output_dir.glob("*.json")):
            print(f"    {p.name} ({p.stat().st_size / 1024:.1f} KB)")
        print(f"{'='*72}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
