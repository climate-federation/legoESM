"""Strong scaling bench for the device-sharded icosahedral/MPAS (TRiSK)
hydrostatic atm step (``make_voronoi_sharded_step`` — cell-partition reorder +
ppermute halo).

The Voronoi twin of ``bench_atm_latlon_spmd_scaling.py`` (mirrored
flag-for-flag where the grids allow): the global mesh is REORDERED with
``reorder_voronoi_for_sharding`` (METIS/RCB/Hilbert-SFC cell partition, ghost-
padded to an even device split) so each device's contiguous ``P("device")``
shard is a spatially compact cell cluster, then the SSP-RK3 step exchanges
only the partition-boundary halo per stage via ``jax.lax.ppermute``.

  strong: fixed subdivision level, vary n_devices -> speedup = t(1)/t(n).
  (Weak scaling rides the subdivision ladder: one level = 4x the cells, so
  L at 4*n_dev matches L-1 at n_dev per-device load — there is no per-device
  row knob like the lat-lon benches' --nlat-per-dev.)

Device count is fixed at process start, so each n_devices runs as a SEPARATE
process; this script benches ONE n_devices and appends a JSON line.
JAX_PLATFORMS=cpu with --xla_force_host_platform_device_count gives virtual
CPU devices (communication-overhead characterization, NOT a real speedup).

Multi-controller (route-B, ``--multicontroller``): identical contract to the
lat-lon benches — every process calls ``jax.distributed.initialize`` BEFORE
any other JAX use, the ("device",) mesh is built over the GLOBAL
``jax.devices()``, and the existing ``make_voronoi_sharded_step`` ppermute
halo + the mass-fix psum run unchanged across processes (NCCL on GPU / gloo
on CPU). NO mpi4jax is armed in this mode (the documented mixed-stack
deadlock hazard). Every process computes the SAME reorder host-side; under
``--multicontroller`` the partition checksum is asserted equal across
processes (a rank-divergent partition — e.g. one rank resolving
``--partition-method auto`` to METIS and another to RCB — would silently
corrupt the halo schedule).

Launch (cluster, one process per GPU):
  srun -n 6 python bench_mpas_spmd_scaling.py --multicontroller \
      --n-devices 6 ...            # SLURM: coordinator auto-detected
  mpiexec -n 6 python ... --multicontroller --coordinator host0:9876
CPU smoke (single process, virtual devices):
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  JAX_ENABLE_X64=1 python scripts/bench/bench_mpas_spmd_scaling.py \
      --subdivision 3 --nlev 4 --n-devices 2 --steps 4 --parity-gate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import jax
import numpy as np

# Repo root on the path for tests.test_cases.baroclinic_wave (the same
# baroclinic-wave IC the icosahedral lanes of run_levante_gpu_scaling use).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Bench dir for the shared metadata module (sibling-script import pattern).
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Shared self-describing scaling metadata (anti-fake-scaling audit): merged
# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
# imports JAX lazily, so this is safe before jax.distributed.initialize.
from metadata import annotate_incomplete, scaling_metadata, tidy_throughput_fields  # noqa: E402

# SPMD full-step parity tolerances — the FLOATING-POINT RE-ASSOCIATION floor
# of the sharded step (ppermute halo + mass-fix psum reduction-order change),
# NOT a bug margin; a real halo/partition regression shows up orders of
# magnitude above these.  Values extend the 1-step envelope of
# tests/parallel/test_voronoi_sharded_equivalence.py (u/T atol 1e-6, p_s
# atol 1e-1) to the smoke window; the floor grows with steps, hence the cap.
MPAS_PARITY_TOLS = {  # precision -> field -> (rtol, atol)
    "float64": {"u": (1.0e-5, 1.0e-5), "T": (1.0e-6, 1.0e-5),
                "p_s": (1.0e-5, 1.0)},
    "float32": {"u": (1.0e-3, 1.0e-3), "T": (1.0e-4, 1.0e-3),
                "p_s": (1.0e-3, 50.0)},
}
MPAS_PARITY_MAX_STEPS = 8

# Conservation gate default: with fix_mass=True the step restores the global
# dry mass to the pre-step value each step, so the drift over a smoke window
# is the allreduce rounding floor, not scheme drift.
MASS_RTOL_DEFAULTS = {"float64": 1.0e-11, "float32": 1.0e-5}


def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method):
    """Reordered+padded global mesh, MPAS PE model, baroclinic-wave IC.

    ``reorder_target`` sets the PARTITION (and ghost padding) so every run
    of a strong-scaling ladder times the IDENTICAL mesh; ``run_nd`` is the
    device count of THIS run's mesh/model (the two differ for the
    single-device reference leg of a ladder, via ``--reorder-for``).
    """
    from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
        MPASPrimitiveEquationModel,
    )
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.mesh import create_voronoi_device_mesh
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas

    mesh = create_voronoi_mesh(subdivision_level=subdivision)
    mesh = reorder_voronoi_for_sharding(mesh, reorder_target, method=method)
    if run_nd > 1 and (mesh.nCells % run_nd or mesh.nEdges % run_nd):
        # Padding only guarantees divisibility for reorder_target.
        raise SystemExit(
            f"padded mesh (nCells={mesh.nCells}, nEdges={mesh.nEdges}) not "
            f"divisible by --n-devices {run_nd}; use a ladder where every "
            f"count divides --reorder-for ({reorder_target}).")
    sigma = create_sigma_coordinate(nlev)
    # Same recipe as the icosahedral lane of run_levante_gpu_scaling /
    # tests/parallel/test_voronoi_sharded_equivalence.py: del4 hyperdiffusion,
    # energy-conserving PV flux, SSP-RK3, global mass fixer.
    cfg = MPASPrimitiveEquationConfig(
        nu_del4=1e16, nu_del4_ps=1e16, fix_mass=True,
        pv_scheme="energy", time_integrator="ssp_rk3",
    )
    dev_config = create_voronoi_device_mesh(
        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
        n_devices=run_nd,
    )
    if dev_config.n_devices > 1:
        from legoesm.parallel.mesh import replicate_pytree
        mesh_model = replicate_pytree(mesh, dev_config)
    else:
        mesh_model = mesh
    model = MPASPrimitiveEquationModel(mesh_model, sigma, cfg)
    state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
    return mesh, model, state, dev_config


def _block(state):
    jax.block_until_ready([leaf for leaf in jax.tree.leaves(state)
                           if leaf is not None])


def _global_dry_mass(state, mesh):
    """sum(p_s * areaCell) on host arrays — the quantity fix_mass pins."""
    ps = np.asarray(state.p_s.data)
    area = np.asarray(mesh.areaCell)
    return float(np.sum(ps * area))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subdivision", type=int, default=5,
                   help="icosahedral subdivision level L "
                        "(nCells = 10*4^L + 2 before ghost padding)")
    p.add_argument("--nlev", type=int, default=8)
    p.add_argument("--n-devices", type=int, required=True)
    p.add_argument("--reorder-for", type=int, default=None,
                   help="partition/reorder the mesh for THIS device count "
                        "(default: --n-devices). Pin it to the ladder's "
                        "max so single-device reference runs time the "
                        "identical reordered mesh.")
    p.add_argument("--partition-method",
                   choices=["auto", "geometric", "metis", "sfc"],
                   default="auto")
    p.add_argument("--physics", choices=["none", "held_suarez"],
                   default="none")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--dt", type=float, default=None,
                   help="timestep [s]; default auto: 600 * 4**(4-L) "
                        "(CFL: dx halves per level), min 30 s.")
    p.add_argument("--out", type=str,
                   default="results/a1/mpas_spmd_scaling.jsonl")
    p.add_argument(
        "--parity-gate", action="store_true",
        help="Correctness gate: compare the gathered sharded trajectory "
             "against the single-device model.step trajectory on the SAME "
             "reordered mesh (smoke windows only; the re-association floor "
             "grows with steps).")
    p.add_argument(
        "--check-conservation", action="store_true",
        help="Gate global dry-mass drift sum(p_s*areaCell) over the run "
             "(pre-shard state vs gathered final state; exits nonzero on "
             "breach).")
    p.add_argument("--mass-rtol", type=float, default=None,
                   help="Conservation tolerance (default: 1e-11 f64 / "
                        "1e-5 f32 — fix_mass pins the mass each step).")
    p.add_argument("--multicontroller", action="store_true",
                   help="Route-B multi-controller: jax.distributed.initialize "
                        "per process, ('device',) mesh over the GLOBAL device "
                        "set (one process per GPU / per CPU-device group). NO "
                        "mpi4jax. --n-devices must equal the global device "
                        "count.")
    p.add_argument("--coordinator", type=str, default=None,
                   help="host:port for jax.distributed when auto-detection "
                        "(SLURM) is unavailable; process count/id then come "
                        "from OMPI_COMM_WORLD_SIZE/RANK.")
    args = p.parse_args()

    # Validate the timing window BEFORE any model/device work (codex, ocean
    # twin): an empty steady slice would only fail after the expensive run.
    if args.steps < 1:
        raise SystemExit(f"--steps must be >= 1, got {args.steps}")
    if not (0 <= args.warmup < args.steps):
        raise SystemExit(
            f"--warmup must satisfy 0 <= warmup < steps "
            f"(got warmup={args.warmup}, steps={args.steps})")
    if args.parity_gate and args.steps > MPAS_PARITY_MAX_STEPS:
        raise SystemExit(
            f"--parity-gate is a smoke gate (re-association floor grows "
            f"with steps); --steps {args.steps} > {MPAS_PARITY_MAX_STEPS} "
            f"cap.")

    if args.multicontroller:
        # MUST run before any other JAX use (backend init). SLURM auto-detects;
        # mpiexec needs the explicit coordinator + launcher env vars (OpenMPI
        # OMPI_*, or Cray PALS PMI_* on Derecho).
        if args.coordinator is not None:
            n_procs = int(os.environ.get(
                "OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "0")))
            proc_id = int(os.environ.get(
                "OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "-1")))
            if n_procs < 1 or proc_id < 0:
                raise SystemExit(
                    "--coordinator given but no launcher rank env found "
                    "(OMPI_COMM_WORLD_SIZE/RANK or PMI_SIZE/PMI_RANK).")
            jax.distributed.initialize(
                coordinator_address=args.coordinator,
                num_processes=n_procs, process_id=proc_id)
        else:
            # Environment-routed: SLURM/OMPI -> bare auto-detect; PALS/PMI
            # (Derecho mpiexec) -> mpi4py bootstrap. Real init failures
            # re-raise loudly.
            from legoesm.parallel.early_init import (
                init_jax_distributed_with_fallback,
            )
            init_jax_distributed_with_fallback()

    from legoesm.parallel.mesh import shard_pytree
    from legoesm.parallel.sharded_dynamics import (
        gather_voronoi_state_spmd,
        make_voronoi_sharded_step,
    )

    nd = args.n_devices
    avail = len(jax.devices())
    if avail < nd:
        raise SystemExit(f"need {nd} devices, have {avail} "
                         f"(set --xla_force_host_platform_device_count)")
    if args.multicontroller and nd != avail:
        # A mesh over a strict subset would leave some processes' devices out
        # of the program (non-addressable participation hazard). Route-B uses
        # ALL global devices: one shard per device across every process.
        raise SystemExit(
            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
            f"device count ({avail} across {jax.process_count()} processes).")

    dt = args.dt
    if dt is None:
        dt = max(600.0 * 4.0 ** (4 - args.subdivision), 30.0)

    reorder_for = args.reorder_for if args.reorder_for is not None else nd
    if reorder_for < nd:
        raise SystemExit(
            f"--reorder-for ({reorder_for}) must be >= --n-devices ({nd}): "
            f"the ghost padding only guarantees divisibility for the "
            f"partition target.")
    mesh, model, s0, dev_config = build_model_and_state(
        args.subdivision, args.nlev, reorder_for, nd, args.partition_method)

    if args.multicontroller:
        # Every process computed the reorder independently — assert the
        # partitions agree before any collective uses the halo schedule.
        # The checksum covers the entity ORDER (coordinates) and the
        # connectivity the ppermute schedule + TRiSK stencils read; a
        # rank-divergent partition (e.g. one rank resolving
        # --partition-method auto to METIS, another to RCB) cannot slip
        # through on cell positions alone.
        from jax.experimental import multihost_utils
        crc = 0
        for arr, dtype in (
            (mesh.latCell, np.float64), (mesh.latEdge, np.float64),
            (mesh.cellsOnEdge, np.int64), (mesh.edgesOnCell, np.int64),
            (mesh.cellsOnCell, np.int64), (mesh.areaCell, np.float64),
        ):
            crc = zlib.crc32(np.ascontiguousarray(
                np.asarray(arr, dtype=dtype)).tobytes(), crc)
        crc = zlib.crc32(
            np.asarray([mesh.nCells, mesh.nEdges, mesh.nVertices],
                       dtype=np.int64).tobytes(), crc)
        multihost_utils.assert_equal(
            np.uint32(crc),
            fail_message="partition/reorder checksum differs across "
                         "processes (rank-divergent --partition-method "
                         "resolution?)")

    physics_fn = None
    if args.physics == "held_suarez":
        from legoesm.atmosphere.held_suarez import held_suarez_forcing_mpas
        physics_fn = held_suarez_forcing_mpas

    # Parity reference: the plain single-device trajectory on the SAME
    # reordered mesh, computed BEFORE any sharding (deterministic identical
    # build on every process).  model.step's signature is call-compatible.
    serial_final = None
    if args.parity_gate:
        from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
            MPASPrimitiveEquationModel,
        )
        ref_model = (model if dev_config.n_devices <= 1
                     else MPASPrimitiveEquationModel(
                         mesh, model.sigma_coord, model.config))
        _s = s0
        for _ in range(args.steps):
            _s = ref_model.step(_s, dt, physics_fn=physics_fn)
        _block(_s)
        serial_final = _s

    mass_before = None
    if args.check_conservation:
        mass_before = _global_dry_mass(s0, mesh)

    step = make_voronoi_sharded_step(model, dev_config)
    if dev_config.n_devices > 1:
        s = shard_pytree(s0, dev_config)
    else:
        s = s0

    # Multi-controller: align every process around the timed loop.
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("mpas_spmd_bench_start")

    # Per-step timing: step 0 includes compile; record each step so re-trace
    # (every step slow) is visible vs steady-state (steps 1.. fast).
    per_step_ms = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        if physics_fn is not None:
            s = step(s, dt, physics_fn=physics_fn)
        else:
            s = step(s, dt)
        _block(s)
        per_step_ms.append((time.perf_counter() - t0) * 1e3)

    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("mpas_spmd_bench_end")

    # --- Correctness gates (before any timing is reported) -----------------
    if args.parity_gate or args.check_conservation:
        final_global = (gather_voronoi_state_spmd(s, dev_config)
                        if dev_config.n_devices > 1 else s)
        prec = "float64" if jax.config.jax_enable_x64 else "float32"
        rank0 = jax.process_index() == 0
        if args.check_conservation:
            mass_after = _global_dry_mass(final_global, mesh)
            tol = (args.mass_rtol if args.mass_rtol is not None
                   else MASS_RTOL_DEFAULTS[prec])
            rel = abs(mass_after - mass_before) / abs(mass_before)
            if rank0:
                print(f"    conservation dry-mass: rel drift={rel:.3e} "
                      f"(tol {tol:.1e}) over {args.steps} steps", flush=True)
            if rel > tol:
                if rank0:
                    print("ERROR: conservation gate BREACHED.", flush=True)
                return 4
        if args.parity_gate:
            tols = MPAS_PARITY_TOLS[prec]
            ok = True
            for name, (rtol, atol) in tols.items():
                want = np.asarray(getattr(serial_final, name).data)
                got = np.asarray(getattr(final_global, name).data)
                field_ok = bool(np.allclose(got, want, rtol=rtol, atol=atol))
                ok &= field_ok
                if rank0:
                    mx = (float(np.max(np.abs(got - want)))
                          if want.size else 0.0)
                    print(f"    parity {name:>4s}: max|diff|={mx:.3e} "
                          f"{'OK' if field_ok else 'MISMATCH'}", flush=True)
            if not ok:
                if rank0:
                    print("ERROR: SPMD parity gate MISMATCH vs the "
                          "single-device reference.", flush=True)
                return 5

    steady = per_step_ms[args.warmup:]
    med = float(np.median(steady))
    rec = dict(
        component="mpas_atm",
        subdivision=args.subdivision, n_devices=nd,
        n_cells=int(mesh.nCells), n_edges=int(mesh.nEdges), nlev=args.nlev,
        partition_method=args.partition_method, physics=args.physics,
        steps=args.steps, dt=dt,
        platform=jax.default_backend(),
        n_processes=jax.process_count(),
        multicontroller=bool(args.multicontroller),
        compile_ms=round(per_step_ms[0], 1),
        steady_median_ms=round(med, 2),
        steady_min_ms=round(float(np.min(steady)), 2),
        per_step_ms=[round(x, 1) for x in per_step_ms],
        cells=int(mesh.nCells) * args.nlev,
    )
    # Flat aggregator-compatible identity + metric fields (see the latlon
    # twin): resolution = subdivision level, matching run_cpu_mpi_scaling's
    # icosahedral convention so both lanes land on the same plot curves.
    rec.update(
        grid_type="icosahedral",
        resolution=args.subdivision,
        n_levels=args.nlev,
        mode="strong",  # this bench fixes the mesh and sweeps devices
        precision="float64" if jax.config.jax_enable_x64 else "float32",
        physics_level=args.physics,
        backend=jax.default_backend(),
        **tidy_throughput_fields(
            dt_seconds=dt, time_per_step_ms=med,
            total_cells=int(mesh.nCells) * args.nlev),
    )
    rec["metadata"] = annotate_incomplete(scaling_metadata(
        grid="icosahedral",
        component="atmosphere",
        resolution=f"L{args.subdivision}",
        n_levels=args.nlev,
        precision="float64" if jax.config.jax_enable_x64 else "float32",
        n_gpus=(nd if jax.default_backend() in ("gpu", "cuda", "rocm")
                else 0),
        decomposition="cell_partition" if nd > 1 else "none",
        # cells_per_rank is per PROCESS (n_ranks semantics); the per-device
        # share lives in extra.cells_per_device — a single-process 4-device
        # SPMD run has 1 rank owning ALL cells (codex finding 3).
        cells_per_rank=int(mesh.nCells) * args.nlev
        // max(jax.process_count(), 1),
        scaling_kind="strong",  # this bench fixes the mesh and sweeps devices
        extra={
            "partition_method": args.partition_method,
            "physics": args.physics,
            "steps": args.steps,
            "multicontroller": bool(args.multicontroller),
            "cells_per_device": int(mesh.nCells) // nd * args.nlev,
        },
    ))
    # Multi-controller: every process times the same program; process 0 owns
    # the JSONL + stdout (others would duplicate/corrupt the append).
    if jax.process_index() == 0:
        _outdir = os.path.dirname(args.out)
        if _outdir:  # a bare basename --out needs no mkdir
            os.makedirs(_outdir, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))
        print(f"[mpas nd={nd} L{args.subdivision} nCells={mesh.nCells} "
              f"nlev={args.nlev}] compile={rec['compile_ms']}ms "
              f"steady_median={med:.2f}ms/step "
              f"(per-step: {rec['per_step_ms']})")
        if rec["metadata"]["virtual_cpu_devices"]:
            print("[virtual-cpu] forced host-platform CPU devices: this row "
                  "is a communication-overhead / correctness proxy, NOT "
                  "hardware scaling — do not report it as a speedup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
