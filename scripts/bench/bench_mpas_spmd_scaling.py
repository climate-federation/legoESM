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

nlev caveat (#1113): the multi-node ceiling at THIN nlev is the ppermute ROUND
count (``hlo_collective_permutes``, recorded per row) x the ~0.11 ms launch
floor, NOT bandwidth — a fixed per-step overhead. At the ``--nlev 8`` default it
dominates (~1.34 Gcells/s wall from N=4), so the default UNDERSTATES production
scalability: at ``--nlev 26`` the per-cell compute grows ~3.25x, the flat wall
dissolves (~2.07+ Gcells/s, ~1.9x higher at 16 GPUs), and a size-dependent term
enters. Report the production curve at production thickness; nlev=8 is the
overhead-mechanism receipt, not the campaign number.

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
from metadata import (  # noqa: E402
    annotate_incomplete, hlo_collective_census, scaling_metadata,
    state_all_finite, tidy_throughput_fields)
from run_levante_gpu_scaling import hyperdiff_coeff  # noqa: E402

# Cap on the del4 stability number nu*dt/dx^4 (dx = mean cell spacing),
# user-approved 2026-09-27.  Measured on the lloyd-0 mesh over 500 steps:
# finite at 0.0006-0.003, non-finite from 0.008 up; 6e-4 keeps a margin.
DEL4_S_MAX = 6e-4


def del4_coeff(subdivision: int, dt: float) -> float:
    """Levante resolution rule for the del4 coefficient, capped at DEL4_S_MAX."""
    from legoesm import constants
    dx = constants.R_earth * np.sqrt(4.0 * np.pi / (10 * 4 ** subdivision + 2))
    return float(min(hyperdiff_coeff(subdivision, "icosahedral"),
                     DEL4_S_MAX * dx ** 4 / dt))

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


def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method, *,
                          dt, moist=False, lloyd_iterations=50, fix_mass=True):
    """Reordered+padded global mesh, MPAS PE model, baroclinic-wave IC.

    ``reorder_target`` sets the PARTITION (and ghost padding) so every run
    of a strong-scaling ladder times the IDENTICAL mesh; ``run_nd`` is the
    device count of THIS run's mesh/model (the two differ for the
    single-device reference leg of a ladder, via ``--reorder-for``).
    ``moist=True`` attaches the q_v/q_c/q_r tracers (moist baroclinic
    wave) so the sharded step's packed tracer halo exchange + RK tracer
    advection sit on the timed/gated path.
    """
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
        MPASPrimitiveEquationModel,
    )
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.mesh import create_voronoi_device_mesh
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding

    mesh = create_voronoi_mesh(subdivision_level=subdivision,
                               lloyd_iterations=lloyd_iterations)
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
    # energy-conserving PV flux, SSP-RK3, global mass fixer.  The del4
    # coefficient follows that lane's resolution rule (5e16 m^4/s at level 5,
    # scaled with dx^4), capped at the stability limit measured for THIS dt.
    # It was a fixed 1e16 at every level, which broke the limit from level 7
    # up (levels 7 and 8 non-finite within 500 steps, 2026-09-27); the plain
    # rule broke it at levels 3-4 with this bench's automatic dt.
    nu4 = del4_coeff(subdivision, dt)
    cfg = MPASPrimitiveEquationConfig(
        nu_del4=nu4, nu_del4_ps=nu4, fix_mass=fix_mass,
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
    # #1100 MPAS twin: the timed path never global-builds the state.
    # build_sharded_baroclinic_wave_state_mpas creates every leaf via
    # jax.make_array_from_callback (only THIS process's shard rows are
    # ever materialised; value-identical (few-ULP contract, measured
    # exact on the pinned CPU stack) to global-build + shard_pytree —
    # tests/parallel/test_mpas_partitionlocal_build.py).  The GLOBAL
    # state is built lazily in main() only for the parity/conservation
    # gates (small smoke scales).  The mesh itself is still global per
    # process — its SFC-partition-local construction is the open
    # remainder of #1100.
    from tests.test_cases.baroclinic_wave import (
        build_sharded_baroclinic_wave_state_mpas,
    )
    state_sharded = build_sharded_baroclinic_wave_state_mpas(
        mesh, sigma, dev_config, perturbed=True, moist=moist)
    return mesh, model, state_sharded, dev_config


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
    p.add_argument("--lloyd", type=int, default=50,
                   help="Lloyd relaxation iterations for the mesh. 50 = "
                        "production SCVT; 0 = labelled synthetic scaling "
                        "mesh (scaling receipts only, never physics — "
                        "must match the prewarmed cache key at subdiv>=9).")
    p.add_argument("--n-devices", type=int, required=True)
    p.add_argument("--reorder-for", type=int, default=None,
                   help="partition/reorder the mesh for THIS device count "
                        "(default: --n-devices). Pin it to the ladder's "
                        "max so single-device reference runs time the "
                        "identical reordered mesh.")
    p.add_argument("--partition-method",
                   choices=["auto", "geometric", "metis", "sfc"],
                   default="auto")
    p.add_argument("--physics", choices=["none", "held_suarez", "kessler"],
                   default="none",
                   help="Operator-split physics on the timed path. "
                        "'kessler' also attaches the q_v/q_c/q_r moist-"
                        "baroclinic-wave tracers (packed tracer halo "
                        "exchange + RK tracer advection on the gated "
                        "path) and extends the parity gate to the "
                        "tracer fields.")
    p.add_argument("--halo-strategy",
                   choices=["auto", "ppermute", "allgather"],
                   default="auto",
                   help="Halo strategy for make_voronoi_sharded_step. "
                        "'auto' picks allgather below the per-device "
                        "cell threshold — force 'ppermute' to exercise "
                        "the neighbor-round schedule on small gate "
                        "meshes (the multicontroller selfspawn tests "
                        "do).  Recorded in the JSONL row.")
    p.add_argument("--wide-halo", action="store_true",
                   help="Set LEGOESM_MPAS_WIDE_HALO=1 before building "
                        "the step: ONE halo fill per step at depth "
                        "evals x SPMD_HALO_DEPTH (communication-"
                        "avoiding), whole RK body inside shard_map. "
                        "Effective mode + depth recorded in the JSONL "
                        "row.")
    p.add_argument("--profile-dir", type=str, default=None,
                   help="Write a jax.profiler trace of the timed loop from "
                        "ranks 0-3 (one node under block:block) into "
                        "<dir>/rank<k>/. The chrome-format trace.json.gz "
                        "gives per-thunk device spans — the gap/duration "
                        "attribution nsys kept silently dropping collectives "
                        "from (campaign 2026-08-07). Ranks 0-3 share a node "
                        "clock, so cross-rank collective start-spread is "
                        "measurable; collective END coincidence is the "
                        "built-in calibration check.")
    p.add_argument("--timed-scan", action="store_true",
                   help="Time the steady window as ONE jit(lax.scan) of "
                        "(steps - warmup) steps with a single device sync, "
                        "instead of the per-step Python loop with a "
                        "block_until_ready every step. Discriminates "
                        "host-dispatch/per-step-sync share: the per-step "
                        "loop both pays a host round-trip per step and "
                        "forbids cross-step pipelining. Warmup steps still "
                        "run the Python loop (compile + steady check).")
    p.add_argument("--no-fix-mass", action="store_true",
                   help="Disable the global mass fixer. TIMING ONLY on a "
                        "scaling arm: it removes the ONE global allreduce "
                        "the step performs, so the arm prices that "
                        "reduction. Mass is then not pinned, and the "
                        "conservation gate must not be used with it.")
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
    # Precision: the state dtype comes from the precision POLICY (default
    # fp32), NOT JAX_ENABLE_X64 — set it to match the x64 flag or the
    # "float64" arm silently runs fp32 state.
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64() if jax.config.jax_enable_x64
               else PrecisionPolicy.fp32())
    mesh, model, s0, dev_config = build_model_and_state(
        args.subdivision, args.nlev, reorder_for, nd, args.partition_method,
        dt=dt, moist=(args.physics == "kessler"), lloyd_iterations=args.lloyd,
        fix_mass=not args.no_fix_mass)

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
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_mpas
        physics_fn = held_suarez_forcing_mpas
    elif args.physics == "kessler":
        # Warm-rain microphysics over the moist BCW tracers.  Kessler's
        # saturation adjustment is a rate over the dt bound HERE, so it
        # must match the stepping dt (make_kessler_forcing_mpas contract).
        from legoesm.atmosphere.forcing.idealized.kessler_forcing import (
            make_kessler_forcing_mpas,
        )
        physics_fn = make_kessler_forcing_mpas(dt)

    # #1100: s0 from build_model_and_state is ALREADY partition-local-sharded
    # when n_devices > 1 (no per-process global build on the timed path).
    # The parity/conservation gates are the ONLY consumers of a global
    # initial state — build it lazily here, at their smoke scales only
    # (deterministic identical build on every process; bit-identical to the
    # sharded s0 per tests/parallel/test_mpas_partitionlocal_build.py).
    s0_global = None
    if args.parity_gate or args.check_conservation:
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
        s0_global = (s0 if dev_config.n_devices <= 1
                     else baroclinic_wave_init_mpas(
                         mesh, model.sigma_coord, perturbed=True,
                         moist=(args.physics == "kessler")))

    # Parity reference: the plain single-device trajectory on the SAME
    # reordered mesh.  model.step's signature is call-compatible.
    serial_final = None
    if args.parity_gate:
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationModel,
        )
        ref_model = (model if dev_config.n_devices <= 1
                     else MPASPrimitiveEquationModel(
                         mesh, model.sigma_coord, model.config))
        _s = s0_global
        for _ in range(args.steps):
            _s = ref_model.step(_s, dt, physics_fn=physics_fn)
        _block(_s)
        serial_final = _s

    mass_before = None
    if args.check_conservation:
        mass_before = _global_dry_mass(s0_global, mesh)

    if args.wide_halo:
        os.environ["LEGOESM_MPAS_WIDE_HALO"] = "1"
    step = make_voronoi_sharded_step(
        model, dev_config, halo_strategy=args.halo_strategy)
    # Already in the sharded layout (partition-local build) for nd > 1;
    # single-device s0 is the plain global state.
    s = s0

    # Multi-controller: align every process around the timed loop.
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("mpas_spmd_bench_start")

    # Per-step timing: step 0 includes compile; record each step so re-trace
    # (every step slow) is visible vs steady-state (steps 1.. fast).
    per_step_ms = []
    scan_median_ms = None
    _profiling = (args.profile_dir is not None
                  and jax.process_index() < 4)
    _prof_on = False
    if _profiling:
        import pathlib
        _pdir = pathlib.Path(args.profile_dir) / f"rank{jax.process_index()}"
        _pdir.mkdir(parents=True, exist_ok=True)
    for _i in range(args.warmup + 1 if args.timed_scan else args.steps):
        # Trace ONLY steps [warmup, warmup+4): tracing from step 0 fills
        # the profiler's 1M-event cap with compile-phase HOST events and
        # the device tracks arrive EMPTY (job 26854167: every X event on
        # pid /host:CPU, zero on /device:GPU:*).
        if _profiling and _i == args.warmup:
            jax.profiler.start_trace(str(_pdir))
            _prof_on = True
        if _profiling and _prof_on and _i == min(
                args.warmup + 4, args.steps - 1):
            jax.profiler.stop_trace()
            _prof_on = False
        t0 = time.perf_counter()
        if physics_fn is not None:
            s = step(s, dt, physics_fn=physics_fn)
        else:
            s = step(s, dt)
        _block(s)
        per_step_ms.append((time.perf_counter() - t0) * 1e3)

    if args.timed_scan:
        # ASYNC WINDOW: dispatch (steps - warmup) steps with NO per-step
        # block_until_ready, ONE sync at the end.  Removes the per-step
        # host round-trip and lets XLA pipeline across steps — the same
        # discriminator an outer jit(lax.scan) would give, WITHOUT a new
        # outer jit: wrapping the step in one closes over its sharded
        # closure constants (stacked meshes / halo schedules), which
        # multicontroller forbids (this killed the first scan_b arm,
        # job 26851745).
        n_scan = args.steps - args.warmup
        t0 = time.perf_counter()
        for _ in range(n_scan):
            if physics_fn is not None:
                s = step(s, dt, physics_fn=physics_fn)
            else:
                s = step(s, dt)
        _block(s)
        scan_median_ms = (time.perf_counter() - t0) * 1e3 / n_scan
        # Fill per_step_ms so the steady slice below stays meaningful.
        per_step_ms += [scan_median_ms] * n_scan

    if _profiling and _prof_on:
        jax.profiler.stop_trace()

    # Per-rank timing spread (skew attribution). Every process measured
    # the SAME steps with its own wall clock; the cross-rank spread of
    # the steady medians is the cheapest honest skew signal available on
    # this stack (nsys records no halo collectives, and wall/max
    # bucketing was shown to smear arrival variance into whichever term
    # an arm was measuring). NOTE the floor: each per-step time already
    # includes a device sync (_block), so what this sees is the spread
    # of ARRIVALS at the end-of-step sync, not per-collective skew.
    per_rank_median_ms = None
    per_rank_spread_ms = None
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        _steady = per_step_ms[args.warmup:] or per_step_ms
        _my_med = float(np.median(np.asarray(_steady)))
        _all = multihost_utils.process_allgather(
            np.asarray([_my_med], dtype=np.float32))
        per_rank_median_ms = [round(float(x), 4)
                              for x in np.asarray(_all).ravel()]
        per_rank_spread_ms = round(
            float(np.max(per_rank_median_ms)
                  - np.min(per_rank_median_ms)), 4)
        multihost_utils.sync_global_devices("mpas_spmd_bench_end")

    # HLO collective-permute census (#1113 ask 2): a STATIC compile property of
    # the sharded step — the ppermute ROUND count that decomposes multi-node
    # overhead (overhead ~= CPs/step * ~0.11 ms launch floor). The cube benches
    # record this; the MPAS row did not, forcing an out-of-band census. Counted
    # AFTER the timed loop so the census compile can't perturb per_step_ms[0]'s
    # compile timing (the executable is already cached — this re-lower/compile
    # is a cache hit; the count is data-independent, static in the partition).
    # Best-effort (None if compilation is unsupported); the serial n=1 leg has
    # no ppermute halo -> 0.
    if physics_fn is not None:
        _census_fn = lambda st: step(st, dt, physics_fn=physics_fn)  # noqa: E731
    else:
        _census_fn = lambda st: step(st, dt)  # noqa: E731
    # ONE compile → full per-family census; the CP scalar (the #1113 round-count
    # wall) is the collective_permute member, so no second compile for it.
    # The interior/rim overlap step threads its rim plan as a jit argument, so
    # re-lowering the PUBLIC step inside the census closes over those sharded
    # arrays and raises under multi-controller. The census is a diagnostic
    # (round count, unchanged by the overlap), so skip it there rather than
    # record a spurious _error on every overlap arm.
    if os.environ.get("LEGOESM_MPAS_HALO_OVERLAP", "") == "1":
        hlo_census = {"_skipped": "halo_overlap"}
    else:
        hlo_census = hlo_collective_census(_census_fn, s)
    # .get: census can return {"_error": ...} (never-silent contract) — a
    # failed census must not KeyError the bench after the timed loop
    # (it killed every multicontroller arm of job 26820846).
    hlo_cp = (hlo_census.get("collective_permute")
              if hlo_census else None)

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
            checks = [
                (name, getattr(serial_final, name).data,
                 getattr(final_global, name).data, rtol, atol)
                for name, (rtol, atol) in tols.items()
            ]
            if serial_final.tracers is not None:
                # Moist run: the tracer fields ride the packed exchange +
                # RK advection — gate them too (q re-association floor is
                # far below the q_v scale; reuse the T tolerances).
                q_rtol, q_atol = tols["T"]
                if set(final_global.tracers or {}) != set(
                        serial_final.tracers):
                    if rank0:
                        print("ERROR: sharded run dropped tracer fields.",
                              flush=True)
                    return 5
                checks += [
                    (k, serial_final.tracers[k].data,
                     final_global.tracers[k].data, q_rtol, q_atol * 1e-3)
                    for k in sorted(serial_final.tracers)
                ]
            for name, want, got, rtol, atol in checks:
                want = np.asarray(want)
                got = np.asarray(got)
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
    finite = state_all_finite(s)
    rec = dict(
        component="mpas_atm",
        subdivision=args.subdivision, n_devices=nd,
        n_cells=int(mesh.nCells), n_edges=int(mesh.nEdges), nlev=args.nlev,
        partition_method=args.partition_method, physics=args.physics,
        nu_del4=float(model.config.nu_del4),
        # lloyd=0 is the LABELLED synthetic scaling mesh — anti-masquerade:
        # a row without this field could pass as a production-SCVT receipt.
        lloyd_iterations=args.lloyd,
        # Requested vs EFFECTIVE (post-"auto") strategy — a JSONL row
        # saying "auto" would not reveal whether ppermute or allgather
        # was actually measured (codex M3c-2 MINOR).
        halo_strategy_requested=args.halo_strategy,
        halo_strategy_effective=getattr(
            step, "_halo_strategy_effective", "serial"),
        wide_halo=bool(getattr(step, "_wide_halo_effective", False)),
        halo_depth=int(getattr(step, "_halo_depth_effective", 3)),
        timed_scan=bool(args.timed_scan),
        scan_median_ms=(round(scan_median_ms, 3)
                        if scan_median_ms is not None else None),
        steps=args.steps, dt=dt,
        platform=jax.default_backend(),
        n_processes=jax.process_count(),
        multicontroller=bool(args.multicontroller),
        compile_ms=round(per_step_ms[0], 1),
        steady_median_ms=round(med, 2), finite_ok=finite, valid=finite,
        steady_min_ms=round(float(np.min(steady)), 2),
        per_step_ms=[round(x, 1) for x in per_step_ms],
        cells=int(mesh.nCells) * args.nlev,
        # ppermute round count/step (static compile property; #1113) — the
        # multi-node ceiling is this count x the ~0.11 ms launch floor, so it
        # belongs on every row like the cube benches.
        hlo_collective_permutes=hlo_cp,
        # full per-family census (permute + all-reduce + all-gather + ...) on
        # the SAME compile: exposes any reduction the ico step introduces.
        hlo_collectives=hlo_census,
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
            dt_seconds=dt, time_per_step_ms=med if finite else None,
            total_cells=int(mesh.nCells) * args.nlev),
    )
    if not finite:
        # A timing of a non-finite state is not a measurement.
        for _k in ("steady_median_ms", "steady_min_ms", "scan_median_ms"):
            rec[_k] = None
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
            # Which arm actually ran. Without this the receipts of a
            # measurement arm and of the baseline are distinguishable
            # only by their FILENAME, and a knob that failed to take
            # effect is indistinguishable from one that did.
            "fix_mass": not args.no_fix_mass,
            "per_rank_median_ms": per_rank_median_ms,
            "per_rank_spread_ms": per_rank_spread_ms,
            # The NCCL transport the arm ran with: the channel count moves
            # the s9 ATMOSPHERE step 17% at 128 GPUs, so rows at different settings are
            # different measurements (plot_nature_scaling.py refuses mixes).
            "nccl_env": {
                k: os.environ.get(k, "")
                for k in ("NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS",
                          "NCCL_P2P_NET_CHUNKSIZE")
            },
            "halo_knobs": {
                k: os.environ.get(k, "")
                for k in ("LEGOESM_MPAS_WIDE_HALO",
                          "LEGOESM_MPAS_WIDE_HALO_STRIDE",
                          "LEGOESM_MPAS_RAGGED_HALO",
                          "LEGOESM_MPAS_HALO_BALLAST",
                          "LEGOESM_MPAS_HALO_NOCOMM",
                          "LEGOESM_MPAS_HALO_NOSTAGE")
            },
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
    return 0 if finite else 3


if __name__ == "__main__":
    raise SystemExit(main())
