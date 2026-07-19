"""Strong/weak scaling bench for the lat-band SPMD lat-lon C-grid hydrostatic
atm step (make_sharded_atm_latlon_step / run_atm_latlon_spmd, the A1 work).

Times the SHARDED step across an N-device ("lat",) mesh and reports per-step
wall time + speedup vs 1 device.  The DEFAULT lane follows the M1 measurement
contract (``metadata.timed_scan_blocks``): fused ``lax.scan`` blocks of
``--steps`` steps with device sync only AROUND each block (``fused_step_ms``,
slowest process across controllers) plus a SEPARATE individually-synced
dispatch-latency probe (``step_latency_ms``) — never mixed.  The
jit(shard_map) step is built once and cached by make_sharded_atm_latlon_step;
re-tracing would show up as every block paying the scan-compile cost again.

  strong: fixed (n_lat, n_lon, nlev), vary n_devices -> speedup = t(1)/t(n).
  weak:   n_lat = nlat_per_dev * n_devices (fixed per-device rows) -> ideal flat.

``--segment-steps N`` (M2b): times the COMPILED-SEGMENT lane instead — ONE
jitted lax.scan of N sharded steps per block (make_sharded_atm_latlon_segment,
band-SHARDED geometry, in-graph finite scalar), so --steps counts BLOCKS of N
steps and the per-step numbers derive from whole-block wall times.  Unlike the
default lane (which scans the bench-local step fn), the segment is the
PRODUCTION artifact; the record carries ``segment_mode=true`` +
``per_block_ms`` and the per-device geometry bytes (replicated vs
band-sharded) computed from the real band-grid shapes.

Receipt honesty: a segment whose in-graph finite scalar reports a non-finite
state STOPS the timed loop and stamps the record ``finite_ok=false`` +
``valid=false`` (with ``completed_blocks`` saying how far it got, a
``diverged: ...`` entry under ``metadata._incomplete``, and nulled
``sypd``/``mcells_per_s``) — a diverging trajectory is never serialized as
valid scaling data.  The default fused lane has no in-graph finite check, so
its rows carry ``finite_ok=null``.

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
from metadata import (  # noqa: E402
    annotate_incomplete,
    calibrated_bound,
    comm_accounting,
    scaling_metadata,
    tidy_throughput_fields,
)


def _build_model(n_lat, n_lon, nlev):
    from legoesm import constants
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon, radius=constants.R_earth,
                              omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=nlev)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True, use_polar_filter=False, use_ppm_transport=True,
        time_integrator="ssp_rk3")
    return CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)


def _build(n_lat, n_lon, nlev):
    # nd=1 lane + tests: global (unsharded) IC build, unchanged protocol.
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_latlon
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        hydrostatic_to_cgrid)

    model = _build_model(n_lat, n_lon, nlev)
    hs0 = held_suarez_init_latlon(model.grid, model.sigma_coord)
    c0 = hydrostatic_to_cgrid(hs0, model.grid)
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
                   help="Default lane: steps per fused lax.scan timing "
                        "block. Segment mode: number of timed BLOCKS of "
                        "--segment-steps steps each.")
    p.add_argument("--warmup", type=int, default=2,
                   help="Segment mode: timed blocks dropped from steady "
                        "stats (block 0 includes the scan compile). Default "
                        "lane: retained for CLI compat (fused-block timing "
                        "separates compile/probe/blocks explicitly).")
    p.add_argument("--blocks", type=int, default=2,
                   help="Default lane: timed fused blocks (per-block times "
                        "expose drift).")
    p.add_argument("--probe-steps", type=int, default=3,
                   help="Default lane: individually-synced steps for the "
                        "SEPARATE dispatch-latency probe (step_latency_ms).")
    p.add_argument("--segment-steps", type=int, default=0,
                   help="M2b: >0 compiles ONE lax.scan segment of this many "
                        "steps (built once, reused; band-sharded geometry) "
                        "and times BLOCKS of segment calls instead of "
                        "per-step host dispatch. 0 = default fused lane.")
    p.add_argument("--physics", choices=["none", "held_suarez"], default="none")
    p.add_argument("--dt", type=float, default=60.0)
    p.add_argument("--single-dev-fused-ms", type=float, default=None,
                   help="fused_step_ms of the nd=1 row at the SAME per-device "
                        "size (compute ingredient of the calibrated T_bound, "
                        "audit item 8). Omitted at nd>1 -> bound emitted null "
                        "+ flagged incomplete; nd=1 uses its own measurement.")
    p.add_argument("--comm-latency-us", type=float, default=None,
                   help="MEASURED per-message latency [us] of THIS machine's "
                        "fabric. Default: MACHINE-CALIBRATED-REQUIRED "
                        "placeholder in metadata.py -> bound_calibrated=false.")
    p.add_argument("--comm-bandwidth-gbs", type=float, default=None,
                   help="MEASURED link bandwidth [GB/s] of THIS machine's "
                        "fabric. Default: MACHINE-CALIBRATED-REQUIRED "
                        "placeholder in metadata.py -> bound_calibrated=false.")
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

    # Validate the schedule BEFORE any model/device work: a zero/negative
    # --steps would otherwise surface only as timed_scan_blocks' None
    # headline (default lane) or an empty timed loop (segment mode) after
    # the expensive build (the ocean twin's guard).
    if args.steps < 1:
        raise SystemExit(f"--steps must be >= 1, got {args.steps}")

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

    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
        atm_latlon_geometry_bytes,
        build_sharded_held_suarez_state_atm_latlon,
        make_sharded_atm_latlon_segment,
        make_sharded_atm_latlon_step)
    seg_n = int(args.segment_steps)
    if seg_n < 0:
        raise SystemExit(f"--segment-steps must be >= 0, got {seg_n}")
    physics_fn = None
    if args.physics == "held_suarez":
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_latlon
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

    if nd == 1:
        model, c0 = _build(n_lat, args.n_lon, args.nlev)
        mesh = None
        c = c0
    else:
        # #1100: band-local IC construction. The nd>1 lanes never materialise
        # the global (n_lat, n_lon, nlev) state per process — each leaf is
        # created via make_array_from_callback for the rows this process's
        # devices own (no global build, no device_put replication, no
        # assert_equal all-gather). This is what lets full-node-packed CPU
        # rungs (128 procs/node) survive at large n_lat.
        model = _build_model(n_lat, args.n_lon, args.nlev)
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
                                 axis_names=("lat",))
        c = build_sharded_held_suarez_state_atm_latlon(
            model.grid, model.sigma_coord, mesh)
    if seg_n > 0:
        seg_fn = make_sharded_atm_latlon_segment(
            model, mesh, seg_n, physics_fn=physics_fn)
    else:
        step = make_sharded_atm_latlon_step(model, mesh,
                                            physics_fn=physics_fn)

    per_block_ms = None
    completed_blocks = None
    finite_ok = None   # default fused lane: no in-graph finite check -> null
    timing = None   # metadata.timed_scan_blocks metrics (default lane only)
    if seg_n > 0:
        # Multi-controller: align every process before the timed loop so
        # block wall times aren't skewed by startup jitter (and once after,
        # so no process exits while peers still hold collectives in flight).
        # (The default lane's fences live inside timed_scan_blocks.)
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices("atm_latlon_spmd_bench_start")
        # Segment mode: each timed BLOCK is one compiled lax.scan of seg_n
        # steps; the host sync per block is the production pattern — read the
        # in-graph finite SCALAR, then block on the state for honest timing.
        per_block_ms = []
        finite_ok = True
        for i in range(args.steps):
            t0 = time.perf_counter()
            c, ok = seg_fn(c, args.dt)
            ok_b = bool(ok)
            _block(c)
            per_block_ms.append((time.perf_counter() - t0) * 1e3)
            if not ok_b:
                # A non-finite state poisons every later block: stop timing
                # and mark the whole record invalid — a warning alone let a
                # diverging trajectory serialize as valid scaling data, and
                # an early false was even forgotten by later true blocks
                # (codex batch4).
                finite_ok = False
                print(f"[warn] segment finite scalar FALSE after block {i} "
                      f"(step {(i + 1) * seg_n}) — stopping the timed loop; "
                      "the record is marked INVALID (finite_ok=false, "
                      "valid=false) and its throughput fields are nulled")
                break
        completed_blocks = len(per_block_ms)
        per_step_ms = [b / seg_n for b in per_block_ms]
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices("atm_latlon_spmd_bench_end")
        steady = per_step_ms[args.warmup:]
        if not steady:
            # Divergence stopped the run inside the warmup window — fall back
            # to every completed unit (the record is already marked invalid;
            # this only keeps the diagnostic median well-defined).
            steady = per_step_ms
        med = float(np.median(steady))
    else:
        # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
        # blocks with sync only AROUND the block — the previous per-step
        # host-synced loop measured dispatch+sync latency, not fused device
        # throughput.  Dispatch latency stays measured SEPARATELY
        # (``step_latency_ms``); multi-controller runs record the
        # slowest-process block time + imbalance ratio.
        from metadata import timed_scan_blocks
        c, timing = timed_scan_blocks(
            lambda st: step(st, args.dt), c,
            block_steps=args.steps, n_blocks=args.blocks,
            probe_steps=args.probe_steps,
            sync_label="atm_latlon_spmd_bench")
        # Headline = fused per-step time from the SLOWEST process; key name
        # kept for the aggregators.
        med = float(timing["fused_step_ms"])

    # valid=false ONLY on an observed non-finite state; the default fused
    # lane (finite_ok=None: unchecked) stays valid.
    valid = finite_ok is not False
    # A diverging segment run's med is not a measurement: feed the bound
    # honest nulls (its flat throughput twins are nulled after assembly).
    _measured_med = med if valid else None
    # Honest per-device geometry residency (from the real band-grid shapes):
    # the default lane replicates all-band stacks; the segment lane shards.
    geom_bytes = (atm_latlon_geometry_bytes(model.grid, nd) if nd > 1
                  else None)

    # Communication accounting (audit item 4) + calibrated T_bound (item 8).
    # nd=1: zero inter-device traffic is a FACT (recorded as 0), so the
    # bound is complete and trivially equals the measured compute.  nd>1:
    # there is no analytic halo-message census for the atm latlon step yet
    # (the ocean twin derives one from its barotropic solver) — the comm
    # ingredients are recorded null with this reason and the bound is
    # emitted incomplete rather than fabricated.
    if nd <= 1:
        _msgs, _bytes_msg, _nred = 0, 0, 0
        _bytes_lower = False   # zero traffic is exact, not an undercount
        _comm_note = "single device: no inter-device halo/reduction traffic"
    else:
        _msgs, _bytes_msg, _nred = None, None, None
        _bytes_lower = None
        _comm_note = ("no analytic halo-message census for the atm latlon "
                      "step yet (audit item 4 follow-up) — comm fields null, "
                      "not fabricated")
    comm_rec = comm_accounting(
        halo_messages_per_step=_msgs,
        bytes_per_message=_bytes_msg,
        full_state_gathers_per_step=0,   # fused scan/segment: no per-step gather
        scope_note=_comm_note,
        bytes_are_lower_bound=_bytes_lower,
    )
    bound_rec = calibrated_bound(
        measured_fused_step_ms=_measured_med,
        # Invalid rows feed the bound NOTHING: even the CLI-provided nd=1
        # baseline is withheld so bound_ingredients.compute_ms cannot dress
        # a diverging row up as a modelled one (codex).
        single_device_fused_step_ms=(
            _measured_med if nd == 1
            else (args.single_dev_fused_ms if valid else None)),
        halo_messages_per_step=comm_rec["halo_messages_per_step"],
        halo_bytes_per_step=comm_rec["halo_bytes_per_step"],
        n_reductions_per_step=_nred,
        # rank imbalance is measured by timed_scan_blocks (default lane);
        # the segment lane records no cross-process block gather -> null.
        rank_imbalance=(float(timing["rank_imbalance"])
                        if timing is not None else None),
        latency_us=args.comm_latency_us,
        bandwidth_GBs=args.comm_bandwidth_gbs,
    )

    rec = dict(
        mode=args.mode, n_devices=nd, n_lat=n_lat, n_lon=args.n_lon,
        nlev=args.nlev, physics=args.physics, steps=args.steps,
        platform=jax.default_backend(),
        n_processes=jax.process_count(),
        multicontroller=bool(args.multicontroller),
        segment_mode=(seg_n > 0),
        segment_steps=(seg_n if seg_n > 0 else None),
        # Measurement validity (codex batch4): finite_ok is the ACCUMULATED
        # in-graph finite verdict (null in the unchecked default fused lane);
        # valid=false marks the row as NOT scaling data; completed_blocks
        # says where a diverging segment run stopped.
        finite_ok=finite_ok,
        valid=valid,
        completed_blocks=completed_blocks,
        steady_median_ms=round(med, 4),
        cells=n_lat * args.n_lon * args.nlev,
    )
    if seg_n > 0:
        # Segment lane: unit 0 = the first BLOCK (includes the scan
        # compile); per-step numbers derive from whole blocks.
        rec.update(
            compile_ms=round(per_block_ms[0], 1),
            steady_min_ms=round(float(np.min(steady)), 2),
            per_step_ms=[round(x, 2) for x in per_step_ms],
            per_block_ms=[round(x, 2) for x in per_block_ms],
        )
    else:
        # Default lane: the timed_scan_blocks metrics (fused_step_ms,
        # step_latency_ms, block_ms, parallel_block_ms, rank_imbalance, ...
        # — the M1 measurement contract).
        rec.update(**timing)
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
    # Increment-2 accounting fields (audit items 4/8), flat for aggregators.
    rec.update(**comm_rec, **bound_rec)
    if not valid:
        # Throughput derived from a diverging trajectory is not a
        # measurement: null it explicitly so aggregators cannot ingest it
        # even if they miss the valid flag (codex batch4) — including the
        # M1 fused/per-step-time keys plotters fall back to
        # (time_per_step_ms, steady_min_ms) and the med-derived bound
        # diagnostics (already null via the _measured_med feed above).
        for _k in ("sypd", "mcells_per_s", "time_per_step_ms",
                   "steady_min_ms", "fused_step_ms", "step_latency_ms",
                   "t_bound_ms", "measured_over_bound"):
            if _k in rec:
                rec[_k] = None
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
            # M2b compiled-segment lane facts: a segment row is falsifiable
            # from the record alone (block timings + geometry residency +
            # the finite/validity verdict — codex batch4).
            "segment_mode": seg_n > 0,
            "segment_steps": (seg_n if seg_n > 0 else None),
            "geometry_bytes_per_device": geom_bytes,
            "finite_ok": finite_ok,
            "valid": valid,
            "completed_blocks": completed_blocks,
            # Route-B transport facts (socket-fallback flag): a
            # multi-node row without an NCCL net plugin is
            # falsifiable from the record alone.
            "nccl": (_nccl_report if args.multicontroller
                     else None),
            "cells_per_device": (n_lat // nd) * args.n_lon * args.nlev,
        },
    ))
    if not valid:
        # Divergence reason on the aggregator-facing incomplete list (the
        # same channel annotate_incomplete uses for missing metadata).
        rec["metadata"].setdefault("_incomplete", []).append(
            f"diverged: segment finite scalar false after block "
            f"{completed_blocks - 1} of {args.steps} — timings describe a "
            "non-finite trajectory, not valid scaling data")
    # Multi-controller: every process times the same program; process 0 owns
    # the JSONL + stdout (others would duplicate/corrupt the append).
    if jax.process_index() == 0:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))
        seg_note = (f" segment[{seg_n}-step blocks]" if seg_n > 0 else "")
        invalid_note = ("" if valid
                        else " INVALID[diverged: finite_ok=false]")
        head = (f"[nd={nd} {args.mode} {n_lat}x{args.n_lon}x{args.nlev}"
                f"{seg_note}]{invalid_note} ")
        if seg_n > 0:
            print(head +
                  f"compile={rec['compile_ms']}ms "
                  f"steady_median={med:.2f}ms/step "
                  f"(per-step: {rec['per_step_ms']})")
        else:
            print(head +
                  f"compile={rec['compile_ms']}ms fused={med:.3f}ms/step "
                  f"latency={rec['step_latency_ms']}ms/step "
                  f"imbalance={rec['rank_imbalance']} "
                  f"blocks={rec['block_ms']}")
        if rec["metadata"]["virtual_cpu_devices"]:
            print("[virtual-cpu] forced host-platform CPU devices: this row "
                  "is a communication-overhead / correctness proxy, NOT "
                  "hardware scaling — do not report it as a speedup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
