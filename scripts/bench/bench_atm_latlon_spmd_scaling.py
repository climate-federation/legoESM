"""Strong/weak scaling bench for the lat-band SPMD lat-lon C-grid hydrostatic
atm step (make_sharded_atm_latlon_step / run_atm_latlon_spmd, the A1 work).

Times the SHARDED step across an N-device ("lat",) mesh and reports per-step
wall time + speedup vs 1 device. Per-step granularity exposes whether the
shard_map is being RE-TRACED every call (the make_sharded_atm_latlon_step
sharded_step rebuilds shard_map per call): if steps 1.. are as slow as step 0,
the cost is host tracing, not device compute, and the scaling number is
meaningless until the shard_map is built once.

  strong: fixed (n_lat, n_lon, nlev), vary n_devices -> speedup = t(1)/t(n).
  weak:   n_lat = nlat_per_dev * n_devices (fixed per-device rows) -> ideal flat.

Device count is fixed at process start, so each n_devices runs as a SEPARATE
process (one sbatch step per count); this script benches ONE n_devices and
appends a JSON line. JAX_PLATFORMS=cpu with --xla_force_host_platform_device_count
gives virtual CPU devices (communication-overhead characterization, NOT a real
speedup); a real number needs one GPU per band.

Multi-controller (route-B, ``--multicontroller``): the lat-lon analogue of the
cubed-sphere ``run_cpu_mpi_scaling --cs-spmd`` A1 path. Every process calls
``jax.distributed.initialize`` BEFORE any other JAX use, the ("lat",) mesh is
built over the GLOBAL ``jax.devices()`` (all processes), and the existing
``make_sharded_atm_latlon_step`` + band-ppermute halo runs unchanged — the
ppermute/psum collectives cross processes via the distributed runtime (NCCL on
GPU / gloo on CPU). NO mpi4jax is armed in this mode (mixing the mpi4jax halo
machinery with jax.distributed collectives in one program is the documented
mixed-stack deadlock hazard — see run_cpu_mpi_scaling._build_cubed_sphere_spmd).
This is the halo path that keeps intra-node GPU traffic on NCCL and bypasses
the host-staged / CXI-inject-broken cross-node GPU-direct MPI route (see
docs/performance/multinode_gpu_direct_cxi.md).

Launch (cluster, one process per GPU):
  srun -n 8 python bench_atm_latlon_spmd_scaling.py --multicontroller \
      --n-devices 8 ...            # SLURM: coordinator auto-detected
  mpiexec -n 8 python ... --multicontroller --coordinator host0:9876
"""
from __future__ import annotations

import argparse
import json
import os
import time

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Bench dir for the shared metadata module (sibling-script import pattern —
# needed when this file is loaded by path from tests, not run as a script).
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Shared self-describing scaling metadata (anti-fake-scaling audit): merged
# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
# imports JAX lazily, so this is safe before jax.distributed.initialize.
from metadata import annotate_incomplete, scaling_metadata, tidy_throughput_fields  # noqa: E402


def _build(n_lat, n_lon, nlev):
    from legoesm import constants
    from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel)
    from legoesm.atmosphere.held_suarez import held_suarez_init_latlon
    from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
        hydrostatic_to_cgrid)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon, radius=constants.R_earth,
                              omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=nlev)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True, use_polar_filter=False, use_ppm_transport=True,
        time_integrator="ssp_rk3")
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    hs0 = held_suarez_init_latlon(grid, sigma)
    c0 = hydrostatic_to_cgrid(hs0, grid)
    return model, c0


def _block(state):
    jax.block_until_ready(jax.tree.leaves(state))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-lat", type=int, default=128)
    p.add_argument("--n-lon", type=int, default=256)
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--n-devices", type=int, required=True)
    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
    p.add_argument("--nlat-per-dev", type=int, default=32,
                   help="weak mode: lat rows per device")
    p.add_argument("--steps", type=int, default=12,
                   help="Steps per fused lax.scan timing block.")
    p.add_argument("--warmup", type=int, default=2,
                   help="(retained for CLI compat; fused-block timing "
                        "separates compile/probe/blocks explicitly).")
    p.add_argument("--blocks", type=int, default=2,
                   help="Timed fused blocks (per-block times expose drift).")
    p.add_argument("--probe-steps", type=int, default=3,
                   help="Individually-synced steps for the SEPARATE "
                        "dispatch-latency probe (step_latency_ms).")
    p.add_argument("--physics", choices=["none", "held_suarez"], default="none")
    p.add_argument("--dt", type=float, default=60.0)
    p.add_argument("--out", type=str, default="results/a1/spmd_scaling.jsonl")
    p.add_argument("--multicontroller", action="store_true",
                   help="Route-B multi-controller: jax.distributed.initialize "
                        "per process, ('lat',) mesh over the GLOBAL device set "
                        "(one process per GPU / per CPU-device group). NO "
                        "mpi4jax. --n-devices must equal the global device "
                        "count.")
    p.add_argument("--coordinator", type=str, default=None,
                   help="host:port for jax.distributed when auto-detection "
                        "(SLURM) is unavailable; process count/id then come "
                        "from OMPI_COMM_WORLD_SIZE/RANK.")
    args = p.parse_args()

    if args.multicontroller:
        # MUST run before any other JAX use (backend init).  The SHARED
        # helper owns the launcher-env contract (SLURM/OMPI auto-detect,
        # PALS mpi4py bootstrap, explicit-coordinator path) AND the
        # hardening: post-init silent-fallback guard + NCCL net-plugin
        # warning — an inline init here would bypass both (codex).
        from legoesm.parallel.early_init import (
            init_multicontroller_distributed,
        )
        init_multicontroller_distributed(args.coordinator)

    from legoesm.atmosphere.dynamics.sharded_atm_latlon_step import (
        make_sharded_atm_latlon_step, shard_state_atm_latlon)
    physics_fn = None
    if args.physics == "held_suarez":
        from legoesm.atmosphere.held_suarez import held_suarez_forcing_latlon
        physics_fn = held_suarez_forcing_latlon

    nd = args.n_devices
    avail = len(jax.devices())
    if avail < nd:
        raise SystemExit(f"need {nd} devices, have {avail} "
                         f"(set --xla_force_host_platform_device_count)")
    if args.multicontroller and nd != avail:
        # A mesh over a strict subset would leave some processes' devices out
        # of the program (non-addressable participation hazard). Route-B uses
        # ALL global devices: one band per device across every process.
        raise SystemExit(
            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
            f"device count ({avail} across {jax.process_count()} processes).")
    n_lat = args.n_lat if args.mode == "strong" else args.nlat_per_dev * nd
    if n_lat % nd != 0:
        raise SystemExit(f"n_lat {n_lat} not divisible by n_devices {nd}")

    model, c0 = _build(n_lat, args.n_lon, args.nlev)

    if nd == 1:
        mesh = None
        step = make_sharded_atm_latlon_step(model, None, physics_fn=physics_fn)
        c = c0
    else:
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
                                 axis_names=("lat",))
        step = make_sharded_atm_latlon_step(model, mesh, physics_fn=physics_fn)
        c = shard_state_atm_latlon(c0, mesh)

    # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
    # blocks with sync only AROUND the block — the previous per-step
    # host-synced loop measured dispatch+sync latency, not fused device
    # throughput.  Dispatch latency stays measured SEPARATELY
    # (``step_latency_ms``); multi-controller runs record the slowest-process
    # block time + imbalance ratio.
    from metadata import timed_scan_blocks
    c, timing = timed_scan_blocks(
        lambda st: step(st, args.dt), c,
        block_steps=args.steps, n_blocks=args.blocks,
        probe_steps=args.probe_steps,
        sync_label="atm_latlon_spmd_bench")

    # Headline = fused per-step time from the SLOWEST process; key name kept
    # for the aggregators.
    med = float(timing["fused_step_ms"])
    rec = dict(
        mode=args.mode, n_devices=nd, n_lat=n_lat, n_lon=args.n_lon,
        nlev=args.nlev, physics=args.physics, steps=args.steps,
        platform=jax.default_backend(),
        n_processes=jax.process_count(),
        multicontroller=bool(args.multicontroller),
        steady_median_ms=round(med, 4),
        cells=n_lat * args.n_lon * args.nlev,
        **timing,
    )
    # Flat aggregator-compatible identity + metric fields: without a
    # top-level ``sypd``/``grid_type`` this lane's rows are invisible to
    # aggregate_bcw_scaling.py → empty SYPD panels in the CPU-vs-GPU plots.
    rec.update(
        grid_type="latlon",
        resolution=n_lat,
        n_levels=args.nlev,
        precision="float64" if jax.config.jax_enable_x64 else "float32",
        physics_level=args.physics,
        backend=jax.default_backend(),
        **tidy_throughput_fields(
            dt_seconds=args.dt, time_per_step_ms=med,
            total_cells=n_lat * args.n_lon * args.nlev),
    )
    from legoesm.parallel.early_init import nccl_transport_report
    _nccl_report = nccl_transport_report()
    rec["metadata"] = annotate_incomplete(scaling_metadata(
        grid="latlon",
        component="atmosphere",
        resolution=f"{n_lat}x{args.n_lon}",
        n_levels=args.nlev,
        precision="float64" if jax.config.jax_enable_x64 else "float32",
        n_gpus=(nd if jax.default_backend() in ("gpu", "cuda", "rocm")
                else 0),
        decomposition="band" if nd > 1 else "none",
        # cells_per_rank is per PROCESS (n_ranks semantics); the per-device
        # share lives in extra.cells_per_device — a single-process 4-device
        # SPMD run has 1 rank owning ALL cells (codex finding 3).
        cells_per_rank=(n_lat * args.n_lon * args.nlev)
        // max(jax.process_count(), 1),
        scaling_kind=args.mode,
        extra={
            "physics": args.physics,
            "steps": args.steps,
            "warmup": args.warmup,
            "multicontroller": bool(args.multicontroller),
            # Route-B transport facts (socket-fallback flag): a
            # multi-node row without an NCCL net plugin is
            # falsifiable from the record alone.
            "nccl": (_nccl_report if args.multicontroller
                     else None),
            "cells_per_device": (n_lat // nd) * args.n_lon * args.nlev,
        },
    ))
    # Multi-controller: every process times the same program; process 0 owns
    # the JSONL + stdout (others would duplicate/corrupt the append).
    if jax.process_index() == 0:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))
        print(f"[nd={nd} {args.mode} {n_lat}x{args.n_lon}x{args.nlev}] "
              f"compile={rec['compile_ms']}ms fused={med:.3f}ms/step "
              f"latency={rec['step_latency_ms']}ms/step "
              f"imbalance={rec['rank_imbalance']} blocks={rec['block_ms']}")
        if rec["metadata"]["virtual_cpu_devices"]:
            print("[virtual-cpu] forced host-platform CPU devices: this row "
                  "is a communication-overhead / correctness proxy, NOT "
                  "hardware scaling — do not report it as a speedup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
