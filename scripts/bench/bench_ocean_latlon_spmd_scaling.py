"""Strong/weak scaling bench for the lat-band SPMD lat-lon C-grid OCEAN step.

The ocean FULL-STEP twin of ``bench_atm_latlon_spmd_scaling.py`` (which this
mirrors flag-for-flag), closing the "no automated ocean full-step strong/weak
harness" gap: ``bench_ocean_mpi_scaling.py`` is the route-A (mpi4jax) phase-
split bench and ``bench_ocean_latlon_spmd_pcg.py`` times the barotropic PCG
KERNEL only — neither times the composed production step
(``make_sharded_ocean_step``: baroclinic + split-explicit barotropic +
implicit vmix + tracers) under the lat-band SPMD backend.

  strong: fixed (n_lat, n_lon, nlev), vary n_devices -> speedup = t(1)/t(n).
  weak:   n_lat = nlat_per_dev * n_devices (fixed per-device rows) -> ideal flat.

Device count is fixed at process start, so each n_devices runs as a SEPARATE
process (one sbatch step per count); this script benches ONE n_devices and
appends a JSON line. JAX_PLATFORMS=cpu with --xla_force_host_platform_device_count
gives virtual CPU devices (communication-overhead characterization, NOT a real
speedup); a real number needs one GPU per band. Run with JAX_ENABLE_X64=1 (the
ocean step's validated precision lane).

Multi-controller (route-B, ``--multicontroller``): identical contract to the
atm bench — every process calls ``jax.distributed.initialize`` BEFORE any
other JAX use, the ("lat",) mesh is built over the GLOBAL ``jax.devices()``,
and the existing ``make_sharded_ocean_step`` band-ppermute halo + psum
reductions (incl. the barotropic ``_global_sum_pair``) run unchanged across
processes (NCCL on GPU / gloo on CPU). NO mpi4jax is armed in this mode (the
documented mixed-stack deadlock hazard).

Launch (cluster, one process per GPU):
  srun -n 8 python bench_ocean_latlon_spmd_scaling.py --multicontroller \
      --n-devices 8 ...            # SLURM: coordinator auto-detected
  mpiexec -n 8 python ... --multicontroller --coordinator host0:9876
CPU smoke (single process, virtual devices):
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  JAX_ENABLE_X64=1 python scripts/bench/bench_ocean_latlon_spmd_scaling.py \
      --n-lat 48 --n-lon 96 --nlev 10 --n-devices 4 --steps 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np

# Sibling-script import (ocean_invariants / conservation helpers reuse —
# same pattern as bench_ocean_mpi_scaling's own cross-script imports).
sys.path.insert(0, str(Path(__file__).parent))

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
    wet_cell_metrics,
)

# SPMD full-step parity tolerances — the FLOATING-POINT RE-ASSOCIATION floor
# of the sharded split-explicit barotropic (ppermute/psum reduction-order
# change over the ~30-substep loop, pole-amplified), NOT a bug margin; a real
# missing-halo regression shows up at O(1e-3+) at the band cuts.  Values
# mirror the equivalence gate (tests/parallel/test_latlon_ocean_spmd_step.py,
# 3 steps: atol 2e-4); the floor grows with steps, hence the smoke cap.
SPMD_PARITY_TOLS = {  # precision -> (rtol, atol)
    "float64": (1.0e-3, 2.0e-4),
    "float32": (1.0e-2, 2.0e-3),
}
SPMD_PARITY_MAX_STEPS = 8


def build_model_and_state(n_lat, n_lon, nlev, seed=0, *,
                          wide_halo=False, wide_halo_chunk=0,
                          tripole=False, baro_solver="implicit_cn"):
    """Ocean model + gently perturbed rest state (flat 4000 m bottom).

    The perturbation (small u/v/eta/T noise on the rest stratification)
    exercises every term of the composed step — advection, Coriolis, PGF, the
    split-explicit barotropic and implicit vmix — instead of the trivial rest
    fixed point, mirroring the SPMD equivalence gate's IC recipe.
    """
    import jax.numpy as jnp
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    if tripole:
        # Synthetic tripole (ORCA fold): the sharded step's fold support
        # is gated by tests/parallel/test_latlon_ocean_spmd_tripole.py;
        # the wide-halo lever refuses folds at model construction.
        from legoesm.grids.tripole import create_synthetic_tripole

        grid = create_synthetic_tripole(n_lat=n_lat, n_lon=n_lon)
    else:
        grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    # Wide-halo lever (A/B): one fused wide lat-halo exchange per chunk of
    # barotropic substeps instead of ~4 ppermute pads per substep.  The wide
    # path's per-substep clamp is local by contract, so pin local clamping
    # in BOTH arms for a controlled comparison.
    # Production-matching solver (scaling audit, bottleneck 4): OMIP runs
    # implicit_cn (run_omip.py full preset); the config-dataclass default
    # is explicit_substep, so it MUST be set explicitly here or the bench
    # measures a non-production step.
    flat = {"barotropic_solver": baro_solver}
    if wide_halo:
        if baro_solver != "explicit_substep":
            raise SystemExit(
                "--wide-halo is a split-explicit barotropic lever; it "
                "requires --baro-solver explicit_substep (implicit_cn has "
                "no substep halo to widen).")
        flat.update(barotropic_wide_halo=True,
                    barotropic_wide_halo_chunk=int(wide_halo_chunk),
                    barotropic_local_subcycle_clamp=True)
    model = LatLonCGridOceanModel(grid, z_coord,
                                  LatLonCGridOceanConfig.from_flat(**flat))
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    rng = np.random.default_rng(seed)
    # Project the velocity noise through the face masks: v_mask zeroes the pole
    # WALL rows, so the nd=1 and nd>1 runs time the SAME initial state (the
    # nd>1 shard drops v[n_lat] and reconstructs it as the pole-wall zero — a
    # random value there would make strong-scaling ICs differ across device
    # counts; codex).
    u_mask = np.asarray(state.u_mask.data)[..., None]
    v_mask = np.asarray(state.v_mask.data)[..., None]
    u = 0.02 * rng.standard_normal((n_lat, n_lon + 1, nlev)) * u_mask
    v = 0.02 * rng.standard_normal((n_lat + 1, n_lon, nlev)) * v_mask
    eta = 0.005 * rng.standard_normal((n_lat, n_lon))
    temp = (5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
            + 0.05 * rng.standard_normal((n_lat, n_lon, nlev)))
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(eta)),
        T=state.T.replace(data=jnp.asarray(temp)))
    return model, state


def _block(state):
    jax.block_until_ready([leaf for leaf in jax.tree.leaves(state)
                           if leaf is not None])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-lat", type=int, default=96)
    p.add_argument("--n-lon", type=int, default=192)
    p.add_argument("--nlev", type=int, default=20)
    p.add_argument("--n-devices", type=int, required=True)
    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
    p.add_argument("--nlat-per-dev", type=int, default=24,
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
    p.add_argument("--baro-solver",
                   choices=["implicit_cn", "explicit_substep"],
                   default="implicit_cn",
                   help="Barotropic solver. Default implicit_cn MATCHES "
                        "production OMIP (run_omip.py full preset); the "
                        "previous silent explicit_substep default made the "
                        "bench measure a non-production configuration "
                        "(scaling audit, bottleneck 4).")
    p.add_argument("--dt", type=float, default=600.0)
    p.add_argument("--single-dev-fused-ms", type=float, default=None,
                   help="fused_step_ms of the nd=1 row at the SAME per-device "
                        "size (the compute ingredient of the calibrated "
                        "T_bound, audit item 8). Omitted at nd>1 -> the bound "
                        "is emitted null + flagged incomplete (never "
                        "fabricated); nd=1 rows use their own measurement.")
    p.add_argument("--comm-latency-us", type=float, default=None,
                   help="MEASURED per-message latency [us] of THIS machine's "
                        "fabric (ping-pong microbenchmark). Default: the "
                        "MACHINE-CALIBRATED-REQUIRED placeholder in "
                        "metadata.py -> the record carries "
                        "bound_calibrated=false.")
    p.add_argument("--comm-bandwidth-gbs", type=float, default=None,
                   help="MEASURED link bandwidth [GB/s] of THIS machine's "
                        "fabric. Default: the MACHINE-CALIBRATED-REQUIRED "
                        "placeholder in metadata.py -> "
                        "bound_calibrated=false.")
    p.add_argument("--out", type=str,
                   default="results/a1/ocean_spmd_scaling.jsonl")
    p.add_argument(
        "--parity-gate", action="store_true",
        help="Correctness gate: compare the gathered sharded trajectory "
             "against the single-device trajectory at the sharded "
             "split-explicit re-association-floor tolerances (smoke windows "
             "only; the floor grows with steps).")
    p.add_argument(
        "--check-conservation", action="store_true",
        help="Gate global area/eta/heat/salt drift over the run "
             "(pre-shard global state vs gathered final state; exits "
             "nonzero on breach).")
    p.add_argument(
        "--cons-rtol", type=float, default=None,
        help="Conservation tolerance (default: 1e-9 f64 / 1e-4 f32; the "
             "raw scheme drifts ~1e-8/step — calibrate to the window).")
    p.add_argument("--tripole", action="store_true",
                   help="Synthetic tripole (ORCA-fold) lane: the sharded "
                        "step folds the north band data-dependently "
                        "(SPMD equivalence gated at 4 devices). Rows are "
                        "tagged grid=tripole. Incompatible with "
                        "--wide-halo (fold refused at construction).")
    p.add_argument("--fused-halo", action="store_true",
                   help="Opt-in SPMD halo message aggregation "
                        "(LEGOESM_LATLON_SPMD_FUSED_HALO=1): one ppermute "
                        "pair per direction per dtype group at every "
                        "pad_multi site instead of one per field — "
                        "measured 25% fewer static collective-permutes on "
                        "this step, bit-identical results. A/B against "
                        "the default run.")
    p.add_argument("--wide-halo", action="store_true",
                   help="Opt-in wide-halo split-explicit barotropic: one "
                        "fused wide lat-halo exchange per chunk of substeps "
                        "instead of ~4 ppermute pads per substep (implies "
                        "local per-substep clamping in this arm; A/B against "
                        "the default run).")
    p.add_argument("--wide-halo-chunk", type=int, default=0,
                   help="Substeps per wide exchange (0 = auto from the band "
                        "height).")
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

    # Validate the timing window BEFORE any model/device work: an empty steady
    # slice would make np.median NaN / np.min raise only AFTER the (expensive)
    # benchmark already ran (codex).
    if args.steps < 1:
        raise SystemExit(f"--steps must be >= 1, got {args.steps}")
    if not (0 <= args.warmup < args.steps):
        raise SystemExit(
            f"--warmup must satisfy 0 <= warmup < steps "
            f"(got warmup={args.warmup}, steps={args.steps})")

    # Align the legoESM precision POLICY with the jax x64 flag: the ocean
    # state dtype comes from get_policy().storage (default fp32), so an
    # x64-flag-only run would build f32 states and gate them against
    # f64-labeled tolerances (the tripole lane caught this; the same fix
    # as bench_ocean_mpi_scaling._ensure_precision).
    from legoesm.core.precision import PrecisionPolicy, set_policy

    # Single source of truth = the LIVE jax x64 flag (an in-process caller
    # may have enabled it without the env var; keying on the env would
    # build fp32 states while every gate/metadata site keys on
    # jax.config — the exact mislabel this block exists to kill; codex).
    if jax.config.jax_enable_x64:
        set_policy(PrecisionPolicy.fp64())
    else:
        set_policy(PrecisionPolicy.fp32())

    if args.multicontroller:
        # MUST run before any other JAX use (backend init). Shared helper:
        # explicit --coordinator -> OMPI/PALS launcher-env init; else
        # SLURM/OMPI auto-detect or PALS mpi4py bootstrap.
        from legoesm.parallel.early_init import init_multicontroller_distributed
        init_multicontroller_distributed(args.coordinator)

    from legoesm.ocean.dynamics.sharded_ocean_step import (
        make_sharded_ocean_step,
        shard_state_latlon,
    )

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

    if args.parity_gate and args.steps > SPMD_PARITY_MAX_STEPS:
        raise SystemExit(
            f"--parity-gate is a smoke gate (re-association floor grows "
            f"with steps); --steps {args.steps} > {SPMD_PARITY_MAX_STEPS} "
            f"cap.")

    if args.fused_halo:
        # Trace-time switch — set BEFORE the sharded step is built/jitted.
        os.environ["LEGOESM_LATLON_SPMD_FUSED_HALO"] = "1"

    model, s0 = build_model_and_state(
        n_lat, args.n_lon, args.nlev,
        wide_halo=args.wide_halo, wide_halo_chunk=args.wide_halo_chunk,
        tripole=args.tripole, baro_solver=args.baro_solver)
    # Prime the build-once vertex-mask cache from the CONCRETE state so the
    # wrapper can build the per-band vertex masks host-side.
    model._ensure_vertex_mask(s0)

    # Wet-cell weak metric (audit item 9): ACTIVE cell-levels from the state
    # land mask (2-D column mask, z-star: a wet column is wet at all nlev
    # levels — the bench_ocean_mpi_scaling wet-count convention).  Computed
    # host-side from the pre-shard global state; bands are contiguous
    # equal-row blocks, so the per-device split is an exact reshape.
    _mask_np = np.asarray(s0.land_mask.data)
    _wet_cols_per_dev = _mask_np.reshape(
        nd, n_lat // nd, args.n_lon).sum(axis=(1, 2))
    wet_rec = wet_cell_metrics(
        wet_columns=float(_mask_np.sum()),
        nlev=args.nlev,
        total_cells=n_lat * args.n_lon * args.nlev,
        n_devices=nd,
        wet_columns_per_device=[float(w) for w in _wet_cols_per_dev],
    )

    # Parity reference: the plain single-device trajectory, computed BEFORE
    # any sharding (deterministic identical build on every process).
    serial_final = None
    if args.parity_gate:
        _s = s0
        for _ in range(args.steps):
            _s = model.step(_s, args.dt)
        _block(_s)
        serial_final = _s

    inv_before = None
    if args.check_conservation:
        from bench_ocean_mpi_scaling import ocean_invariants
        inv_before = ocean_invariants(model, s0, n_ranks=1)

    if nd == 1:
        mesh = None
        step = make_sharded_ocean_step(model, None)
        s = s0
    else:
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
                                 axis_names=("lat",))
        step = make_sharded_ocean_step(model, mesh)
        s = shard_state_latlon(s0, mesh)

    # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
    # blocks with sync only AROUND the block — the previous per-step
    # host-synced loop measured dispatch+sync latency, not fused device
    # throughput.  Per-step dispatch latency is still measured, SEPARATELY,
    # by an individually-synced probe (``step_latency_ms``); multi-controller
    # runs record the SLOWEST-process block time + imbalance ratio (a
    # straggler band is invisible to a single process's clock).
    from metadata import timed_scan_blocks
    # Under --parity-gate the serial reference above ran EXACTLY args.steps
    # steps, so the SPMD arm must execute the same count: 1 compile step +
    # one (args.steps - 1)-long block, no probe.  Timing from a parity smoke
    # run is not reported as a scaling number anyway (steps are capped).
    if args.parity_gate:
        _blk, _nblk, _probe = max(0, args.steps - 1), 1, 0
    else:
        _blk, _nblk, _probe = args.steps, args.blocks, args.probe_steps
    s, timing = timed_scan_blocks(
        lambda st: step(st, args.dt), s,
        block_steps=_blk, n_blocks=_nblk, probe_steps=_probe,
        sync_label="ocean_latlon_spmd_bench")

    # Post-run solver-residual probe eligibility (audit item 6): needs the
    # gathered global final state on ONE process; multicontroller runs skip
    # it (recorded as residual_measured=False, never faked).
    _probe_residual = (args.baro_solver == "implicit_cn"
                       and jax.process_count() == 1)

    final_global = None
    if args.parity_gate or args.check_conservation or _probe_residual:
        from legoesm.ocean.dynamics.sharded_ocean_step import (
            gather_state_latlon,
        )
        final_global = (gather_state_latlon(s, mesh) if mesh is not None
                        else s)

    # --- Correctness gates (before any timing is reported) -----------------
    if args.parity_gate or args.check_conservation:
        prec = "float64" if jax.config.jax_enable_x64 else "float32"
        rank0 = jax.process_index() == 0
        if args.check_conservation:
            from bench_ocean_mpi_scaling import (
                CONS_RTOL_DEFAULTS,
                conservation_breaches,
                ocean_invariants,
                print_conservation,
            )
            inv_after = ocean_invariants(model, final_global, n_ranks=1)
            tol = (args.cons_rtol if args.cons_rtol is not None
                   else CONS_RTOL_DEFAULTS[prec])
            if rank0:
                print_conservation(f"{args.steps} steps", inv_before,
                                   inv_after, tol)
            if conservation_breaches(inv_before, inv_after, tol):
                if rank0:
                    print("ERROR: conservation gate BREACHED.", flush=True)
                return 4
        if args.parity_gate:
            rtol, atol = SPMD_PARITY_TOLS[prec]
            ok = True
            for name in ("T", "S", "eta", "u", "v"):
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

    # --- Solver iterations + post-run residual (audit item 6) --------------
    # implicit_cn dispatch (barotropic_implicit_latlon_cgrid): the SPMD /
    # distributed / force_pcg path runs a FIXED-iteration PCG
    # (barotropic_implicit_pcg_fixed_iters, static fori_loop); the plain
    # single-device path runs the stock adaptive jax.scipy CG whose
    # iteration count is not exposed (tol/maxiter recorded instead).
    _baro_cfg = model.config.barotropic
    _pcg_fixed_path = (args.baro_solver == "implicit_cn"
                       and (nd > 1
                            or bool(_baro_cfg.barotropic_implicit_force_pcg)))
    if _pcg_fixed_path:
        solver_iters = int(_baro_cfg.barotropic_implicit_pcg_fixed_iters)
        solver_iters_mode = (
            f"fixed_pcg[{_baro_cfg.barotropic_implicit_pcg_variant},"
            f"{_baro_cfg.barotropic_implicit_preconditioner}]")
    elif args.baro_solver == "implicit_cn":
        solver_iters = None
        solver_iters_mode = (
            "adaptive_stock_cg(maxiter="
            f"{int(_baro_cfg.barotropic_implicit_pcg_maxiter)},"
            f"tol={_baro_cfg.barotropic_implicit_pcg_tol:g}) — iteration "
            "count not exposed by jax.scipy CG")
    else:
        solver_iters = None
        solver_iters_mode = "explicit_substep (no iterative solve)"

    solver_residual = None
    residual_measured = False
    residual_reason = None
    if args.baro_solver != "implicit_cn":
        residual_reason = ("explicit_substep barotropic has no iterative "
                           "solve — no solver residual exists to measure")
    elif not _probe_residual:
        residual_reason = ("multicontroller run: the residual probe needs "
                           "the gathered global state on one process; "
                           "skipped (not faked)")
    else:
        # ONE extra implicit free-surface solve on the gathered FINAL state,
        # OUTSIDE the timing loop, using the solver's return_residual
        # diagnostic mode — the GLOBAL relative Helmholtz residual of the
        # returned eta.  At nd>1 the timed run took the fixed-iteration PCG
        # path (SPMD), so the probe forces the SAME fixed-M PCG body
        # (force_pcg; its global dots reduce locally single-process) — the
        # residual measured is that of the solver configuration actually
        # benchmarked.  Probe state is discarded; F_slow terms are zero
        # (standalone solve at the final state, not a step replay).
        from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
            barotropic_implicit_latlon_cgrid,
        )
        _probe_cfg = model.config
        if nd > 1:
            _probe_cfg = _probe_cfg._replace(
                barotropic=_probe_cfg.barotropic._replace(
                    barotropic_implicit_force_pcg=True))
        _probe_out = barotropic_implicit_latlon_cgrid(
            final_global, args.dt, model.grid, model.z_coord, _probe_cfg,
            return_residual=True)
        solver_residual = float(jax.block_until_ready(_probe_out[2]))
        residual_measured = True

    # --- Communication accounting (audit item 4) + calibrated bound (8) ----
    # Analytic INTER-DEVICE census, barotropic-solver scope ONLY (the
    # baroclinic 3-D pads are not counted -> bytes/comm are a LOWER bound;
    # T_bound below therefore stays a valid lower bound on the step time).
    # implicit_cn PCG: each Helmholtz apply pads eta N+S (gradient stencil)
    # + the v-face flux row (divergence) ~= 2 exchanges/apply, applied
    # iters + 1 times (incl. the initial residual); reductions = the dot
    # batches (2/iter standard, 1/iter single_reduce) + the initial batch
    # + the mass-projection psum + the eta-floor clamp psum.
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        estimate_barotropic_halo_messages,
    )
    _halo_est = estimate_barotropic_halo_messages(
        model.config, model.config.barotropic.n_barotropic_substeps)
    _dtype_bytes = 8 if jax.config.jax_enable_x64 else 4
    _n_reductions = None
    if nd <= 1:
        _msgs, _n_reductions = 0, 0
        _comm_note = ("single device: no inter-device halo/reduction "
                      "traffic")
    elif _pcg_fixed_path:
        _msgs = 2 * (solver_iters + 1)
        _per_iter = (1 if _baro_cfg.barotropic_implicit_pcg_variant
                     == "single_reduce" else 2)
        _n_reductions = _per_iter * solver_iters + 3
        _comm_note = ("analytic, barotropic implicit-CN PCG scope only; "
                      "2-D eta row slabs (nlev=1), one row per direction "
                      "(rows_per_message=2); baroclinic 3-D pads NOT "
                      "counted — bytes are a lower bound")
    elif args.wide_halo:
        _msgs = None
        _comm_note = ("wide-halo arm: per-chunk exchange count depends on "
                      "the auto chunk size; census in "
                      "extra.barotropic_halo_messages — bytes not derived "
                      "(not fabricated)")
    else:
        # explicit_substep standard path: the analytic per-substep pad
        # census (also recorded verbatim in extra.barotropic_halo_messages).
        _msgs = int(_halo_est["standard_messages"])
        _comm_note = ("analytic, explicit-substep barotropic scope only "
                      "(2-D slabs, one row per direction); reduction census "
                      "not derived for this path; baroclinic 3-D pads NOT "
                      "counted")
    comm_rec = comm_accounting(
        halo_messages_per_step=_msgs,
        n_lon=args.n_lon, nlev=1, dtype_bytes=_dtype_bytes,
        rows_per_message=2,   # a pad exchange moves one row N + one row S
        full_state_gathers_per_step=0,   # fused scan: no per-step gather
        scope_note=_comm_note,
    )

    # Headline number = the fused-scan per-step time from the SLOWEST process
    # (measurement contract).  ``steady_median_ms`` keeps its aggregator-facing
    # name but now carries the fused number; the individually-synced dispatch
    # latency is reported separately as ``step_latency_ms``.
    med = float(timing["fused_step_ms"])

    # Calibrated T_bound (audit item 8): nd=1 rows ARE their own compute
    # ingredient; nd>1 rows need the nd=1 fused number passed in (else the
    # bound is emitted null + flagged).  launch_host_ms=0: per-step launch
    # cost inside a fused lax.scan block is amortized to ~0.
    bound_rec = calibrated_bound(
        measured_fused_step_ms=med,
        single_device_fused_step_ms=(med if nd == 1
                                     else args.single_dev_fused_ms),
        halo_messages_per_step=comm_rec["halo_messages_per_step"],
        halo_bytes_per_step=comm_rec["halo_bytes_per_step"],
        n_reductions_per_step=_n_reductions,
        rank_imbalance=float(timing["rank_imbalance"]),
        latency_us=args.comm_latency_us,
        bandwidth_GBs=args.comm_bandwidth_gbs,
    )

    rec = dict(
        component="ocean",
        mode=args.mode, n_devices=nd, n_lat=n_lat, n_lon=args.n_lon,
        nlev=args.nlev, steps=args.steps,
        platform=jax.default_backend(),
        n_processes=jax.process_count(),
        multicontroller=bool(args.multicontroller),
        steady_median_ms=round(med, 4),
        cells=n_lat * args.n_lon * args.nlev,
        **timing,
    )
    # Flat aggregator-compatible identity + metric fields (see the atm
    # latlon twin): rows become visible to aggregate_bcw_scaling.py /
    # the CPU-vs-GPU plots, keyed as component="ocean" (already in rec).
    rec.update(
        grid_type="tripole" if args.tripole else "latlon",
        resolution=n_lat,
        n_levels=args.nlev,
        precision="float64" if jax.config.jax_enable_x64 else "float32",
        physics_level="none",
        backend=jax.default_backend(),
        **tidy_throughput_fields(
            dt_seconds=args.dt, time_per_step_ms=med,
            total_cells=n_lat * args.n_lon * args.nlev),
    )
    # Increment-2 accounting (audit items 4/6/8/9): flat fields so
    # aggregators read them without descending into metadata/extra.
    rec.update(
        solver_iters=solver_iters,
        solver_iters_mode=solver_iters_mode,
        solver_residual=solver_residual,
        residual_measured=residual_measured,
        residual_reason=residual_reason,
        **wet_rec,
        **comm_rec,
        **bound_rec,
    )
    from legoesm.parallel.early_init import nccl_transport_report
    _nccl_report = nccl_transport_report()
    rec["metadata"] = annotate_incomplete(scaling_metadata(
        grid="tripole" if args.tripole else "latlon",
        component="ocean",
        resolution=f"{n_lat}x{args.n_lon}",
        n_levels=args.nlev,
        precision="float64" if jax.config.jax_enable_x64 else "float32",
        n_gpus=(nd if jax.default_backend() in ("gpu", "cuda", "rocm")
                else 0),
        decomposition="band" if nd > 1 else "none",
        # +wide_halo marks the A/B arm so aggregation never conflates it
        # with the per-substep-pad baseline.
        solver_variant=(model.config.barotropic.barotropic_solver
                        + ("+wide_halo" if args.wide_halo else "")),
        # MEASURED post-run relative Helmholtz residual (audit item 6);
        # None when not measurable — see rec.residual_reason.
        solver_residual=solver_residual,
        # cells_per_rank is per PROCESS (n_ranks semantics); the per-device
        # share lives in extra.cells_per_device — a single-process 4-device
        # SPMD run has 1 rank owning ALL cells (codex finding 3).
        cells_per_rank=(n_lat * args.n_lon * args.nlev)
        // max(jax.process_count(), 1),
        scaling_kind=args.mode,
        extra={
            "steps": args.steps,
            "warmup": args.warmup,
            "multicontroller": bool(args.multicontroller),
            "fused_halo": os.environ.get(
                "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0",
            # Route-B transport facts (socket-fallback flag): a
            # multi-node row without an NCCL net plugin is
            # falsifiable from the record alone.
            "nccl": (_nccl_report if args.multicontroller
                     else None),
            "parity_gate": bool(args.parity_gate),
            "check_conservation": bool(args.check_conservation),
            "cells_per_device": (n_lat // nd) * args.n_lon * args.nlev,
            # Analytic barotropic lat-halo message census (the wide-halo
            # audit item's halo-count metric): standard per-substep pads
            # vs the wide path's fused per-chunk exchanges.
            "barotropic_halo_messages": _halo_est,
            # Solver-iteration facts next to the residual (audit item 6).
            "solver_iters": solver_iters,
            "solver_iters_mode": solver_iters_mode,
            "residual_measured": residual_measured,
        },
    ))
    # Multi-controller: every process times the same program; process 0 owns
    # the JSONL + stdout (others would duplicate/corrupt the append).
    if jax.process_index() == 0:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))
        print(f"[ocean nd={nd} {args.mode} {n_lat}x{args.n_lon}x{args.nlev}] "
              f"compile={rec['compile_ms']}ms fused={med:.3f}ms/step "
              f"latency={rec['step_latency_ms']}ms/step "
              f"imbalance={rec['rank_imbalance']} blocks={rec['block_ms']}")
        _res_txt = (f"{solver_residual:.3e}" if solver_residual is not None
                    else f"n/a ({residual_reason})")
        print(f"[solver] {args.baro_solver} iters={solver_iters} "
              f"({solver_iters_mode}) post-run rel_residual={_res_txt}")
        print(f"[bound] t_bound_ms={rec['t_bound_ms']} "
              f"measured_over_bound={rec['measured_over_bound']} "
              f"calibrated={rec['bound_calibrated']}"
              + (f" incomplete={rec['bound_incomplete_reason']}"
                 if rec["bound_incomplete_reason"] else ""))
        if rec["wet_equals_total"]:
            print("[wet-metric] NOTE: wet_cell_levels == total cells — this "
                  "run's state is ALL-WET, so the wet-cell weak metric is "
                  "NON-INFORMATIVE here; it becomes meaningful with a real "
                  "land mask / bathymetry. (The default latlon rest state "
                  "carries polar land-cap rows, so wet < total there.)")
        if rec["metadata"]["virtual_cpu_devices"]:
            print("[virtual-cpu] forced host-platform CPU devices: this row "
                  "is a communication-overhead / correctness proxy, NOT "
                  "hardware scaling — do not report it as a speedup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
