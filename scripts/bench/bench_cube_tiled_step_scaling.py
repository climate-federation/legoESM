"""Strong-scaling bench for the SUB-FACE TILED cube hydrostatic step (>6 GPUs).

Scaling-audit item 5: the >6-device cubed-sphere lever.  Face-only sharding
tops out at 6 devices; this lane times the PRODUCTION-ENVELOPE tiled step
(``make_tiled_cc_step``: the 24-proc bit-parity-proven tiled D-grid SSP-RK3
core wrapped in the serial step's own cc<->corner conversions) on a
``(6, kt, kt)`` mesh — ``n_devices = 6*kt^2`` (24, 54, 96, ...).

SCOPE (honest): the tiled step covers the DYNAMICS-ONLY base cut — the
adapter REFUSES configs with hyperdiffusion / divergence damping / del-6 /
Smagorinsky / implicit sponge / duogrid / the inner mass fixer (fail-closed
envelope, ``tiled_step_adapter._refuse``).  The full production driver keeps
its loud "tiled dycore unwired (P4)" warning; this lane is where >6-GPU
production stepping is measured TODAY.

Anti-fake-scaling guards:
  * compiled-HLO census: the step must contain collective-permutes and NO
    full-cube all-gather (``find_fullcube_allgathers`` — an all-gather means
    replicated, not tiled, execution): the row is REFUSED otherwise;
  * shared metadata v2 rows (virtual-CPU devices flagged; transport
    auto-resolves) — a CPU smoke row can never masquerade as GPU scaling;
  * --parity-gate: ONE tiled step vs one serial untiled step at the adapter
    gate's f32-honest tolerances (the adapter is single-shot — tile-replicated
    in, tile-sharded out; single-process only; smoke windows).

Launch:
  CPU smoke (24 virtual devices; NOTE the tiled-stage compile at 24
  virtual devices is CLUSTER-scale — the pre-existing adapter parity gate
  itself runs ~10 min on a laptop CPU, so budget accordingly or run the
  smoke on a cluster CPU node):
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
    python scripts/bench/bench_cube_tiled_step_scaling.py --kt 2 \
        --resolution 48 --nlev 30 --steps 6 --parity-gate
  Derecho (PBS/PALS, 24 A100 = 6 nodes x 4):
    #PBS -l select=6:ncpus=64:mpiprocs=4:ngpus=4
    mpiexec -n 24 python scripts/bench/bench_cube_tiled_step_scaling.py \
        --kt 2 --resolution 192 --nlev 60 --steps 12 --multicontroller
  Levante (SLURM, 24 GPUs = 6 nodes x 4):
    srun -N6 --ntasks-per-node=4 --gpus-per-task=1 \
        python scripts/bench/bench_cube_tiled_step_scaling.py \
        --kt 2 --resolution 192 --nlev 60 --steps 12 --multicontroller
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metadata import annotate_incomplete, scaling_metadata, tidy_throughput_fields  # noqa: E402

#: Parity tolerances vs the serial untiled step — the adapter gate's
#: f32-honest bounds (exact f32 ulps of the field scales; a real stage
#: regression is 2e-5-abs class).  These bound the SINGLE tiled-vs-serial
#: step the parity gate checks (measured ~4e-6 u, ~3e-5 T, ~8e-3 p_s), and
#: stay the adapter gate's own tolerances so a silent loosening here can't let
#: a stage regression pass the lane.
TILED_PARITY_ATOL = {"u": 2e-5, "v": 2e-5, "T": 1e-4, "p_s": 0.06}


def _count_collective_permutes(hlo_text: str) -> int:
    return sum(1 for line in hlo_text.splitlines()
               if "collective-permute" in line and "done" not in line)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kt", type=int, required=True,
                   help="Tiles per face edge; n_devices = 6*kt^2 "
                        "(kt=2 -> 24, kt=3 -> 54).")
    p.add_argument("--resolution", type=int, default=48,
                   help="Cells per cube edge N (must divide by kt).")
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--steps", type=int, default=6,
                   help="Timing samples — repeated single-shot steps on the "
                        "pristine input (the tiled adapter is one-shot, not a "
                        "feedback loop; see the timed-loop note).")
    p.add_argument("--warmup", type=int, default=2,
                   help="Leading samples discarded before the steady median.")
    p.add_argument("--dt", type=float, default=60.0)
    p.add_argument("--parity-gate", action="store_true",
                   help="Gate vs the serial untiled step (single-process "
                        "smoke windows only).")
    p.add_argument("--closed-loop", action="store_true",
                   help="Persistent BLOCKED-layout lane (the np>6 production "
                        "assembly): state stays tile-sharded across steps "
                        "(make_tiled_cc_loop), s = step(s) feedback with the "
                        "in-stage telescoping mass fixer (production "
                        "conservation config).  Default off = the single-shot "
                        "adapter lane (dynamics-only, pristine-input samples).")
    p.add_argument("--multicontroller", action="store_true",
                   help="Route-B: one process per GPU; shared hardened "
                        "init (jax.distributed) BEFORE any JAX use.")
    p.add_argument("--coordinator", type=str, default=None)
    p.add_argument("--out", type=str,
                   default="results/a1/cube_tiled_scaling.jsonl")
    args = p.parse_args()

    if args.kt < 2:
        raise SystemExit("--kt must be >= 2 (kt=1 is the face-only lane: "
                         "use bench_cube_shardmap_halo / run_levante).")
    if args.steps < 1 or not (0 <= args.warmup < args.steps):
        raise SystemExit("need steps >= 1 and 0 <= warmup < steps")
    if args.resolution % args.kt != 0:
        raise SystemExit(
            f"--resolution {args.resolution} must divide by --kt {args.kt} "
            f"(tile-local edge = N/kt).")

    if args.multicontroller:
        from legoesm.parallel.early_init import (
            init_multicontroller_distributed,
        )
        init_multicontroller_distributed(args.coordinator)

    import jax

    n_devices = 6 * args.kt * args.kt
    avail = len(jax.devices())
    if avail < n_devices:
        raise SystemExit(
            f"need {n_devices} devices (6*kt^2), have {avail} "
            f"(set --xla_force_host_platform_device_count for CPU smoke).")
    if args.multicontroller and n_devices != avail:
        raise SystemExit(
            f"--multicontroller: 6*kt^2 ({n_devices}) must equal the "
            f"GLOBAL device count ({avail}).")
    if args.parity_gate and jax.process_count() > 1:
        raise SystemExit(
            "--parity-gate is single-process only (the serial reference "
            "and gathered comparison are process-local).")

    from jax.sharding import Mesh
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationConfig,
        CDGridPrimitiveEquationModel,
    )
    from legoesm.atmosphere.dynamics.gcm.tiled_step_adapter import (
        make_tiled_cc_loop,
        make_tiled_cc_step,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.halo import set_halo_backend
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.parallel.cubesphere_exchange import find_fullcube_allgathers

    grid = create_cubed_sphere(args.resolution)
    coord = create_sigma_coordinate(args.nlev)
    if args.closed_loop:
        # PRODUCTION conservation config: the blocked loop applies the
        # serial telescoping post-step fix_ps_mass IN-STAGE (make_tiled_cc_
        # loop validates the envelope; anchor stays off so serial + tiled
        # both telescope the per-step pre-step mass).
        cfg = CDGridPrimitiveEquationConfig(
            use_conservation_fixer=True, fix_mass=True,
            anchor_mass_to_initial=False, zero_mean_ps_tendency=True)
    else:
        # Base-cut envelope config (the single-shot adapter refuses
        # anything outside it): dynamics only, fixers externalized.
        cfg = CDGridPrimitiveEquationConfig(
            use_conservation_fixer=False, fix_mass=False)
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    state0 = held_suarez_init(grid, coord)

    # ONE serial reference step BEFORE any mesh/backend arming (parity gate).
    # The tiled adapter is single-shot: it consumes tile-REPLICATED state (so
    # the pre-shard_map cc->corner interp sees the full face) and emits
    # tile-SHARDED state, so its output cannot be fed back to the SAME AOT
    # executable — re-replicating it to close a feedback loop is precisely the
    # full-cube all-gather the census refuses.  The lane therefore times
    # repeated single steps on the pristine input and checks one-step parity
    # (codex).
    serial_ref = None
    if args.parity_gate:
        ref = model.step(state0, args.dt)
        jax.block_until_ready(jax.tree.leaves(ref))
        serial_ref = {nm: np.asarray(getattr(ref, nm).data)
                      for nm in ("u", "v", "T", "p_s")}

    # RAW (6, kt, kt) mesh + local halo backend — the adapter-gate setup
    # (the tiled stage manages its own ppermutes through the mesh).
    set_halo_backend("local")
    dev = np.array(jax.devices()[:n_devices]).reshape(6, args.kt, args.kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    if args.closed_loop:
        # Persistent blocked layout: enter ONCE, feed the step its own
        # output (input layout == output layout — the production loop).
        enter, loop_step, loop_exit = make_tiled_cc_loop(
            model, mesh, kt=args.kt, dt=args.dt)
        blk0 = enter(state0)
        jax.block_until_ready(jax.tree.leaves(blk0))
        tiled_step = jax.jit(loop_step)
        _lower_arg = blk0
    else:
        tiled_step = jax.jit(make_tiled_cc_step(model, mesh, kt=args.kt,
                                                dt=args.dt))
        _lower_arg = state0

    # --- Anti-fake HLO census on the compiled step --------------------------
    # The SAME jitted callable is audited AND timed (auditing a separate
    # jit while timing the bare adapter would census a different
    # executable; codex) — and the audit compile is reused by the timed
    # loop (per_step_ms[0] is then dispatch, not compile; recorded).
    lowered = tiled_step.lower(_lower_arg)
    _t_compile0 = time.perf_counter()
    compiled = lowered.compile()
    compile_ms = (time.perf_counter() - _t_compile0) * 1e3
    hlo = compiled.as_text()
    n_ppermute = _count_collective_permutes(hlo)
    allgathers = find_fullcube_allgathers(hlo, n=args.resolution)
    if n_ppermute == 0:
        raise SystemExit(
            "compiled tiled step contains NO collective-permutes — the "
            "halos did not tile (replicated execution); refusing to "
            "record a fake scaling row.")
    if allgathers:
        raise SystemExit(
            f"compiled tiled step contains full-cube all-gathers "
            f"({allgathers[:3]}...) — replicated, not tiled, execution; "
            f"refusing to record a fake scaling row.")

    # --- Parity gate FIRST: the correctness receipt BEFORE any timing -------
    # One tiled step on the pristine tile-replicated input vs one serial step
    # (--parity-gate runs only in the small Stage-A config); no timing row is
    # ever produced for an unverified execution (codex).
    if args.parity_gate:
        s = compiled(_lower_arg)
        jax.block_until_ready(jax.tree.leaves(s))
        # Closed loop: reassemble the blocked state to cc (the exit path)
        # so BOTH lanes gate the same cc fields vs the same serial step.
        s_cc = loop_exit(s, state0) if args.closed_loop else s
        ok = True
        for nm in ("u", "v", "T", "p_s"):
            got = np.asarray(getattr(s_cc, nm).data)
            mx = float(np.max(np.abs(got - serial_ref[nm])))
            field_ok = mx < TILED_PARITY_ATOL[nm]
            ok &= field_ok
            print(f"    parity {nm:>4s}: max|diff|={mx:.3e} "
                  f"{'OK' if field_ok else 'MISMATCH'}", flush=True)
        if not ok:
            print("ERROR: tiled-vs-serial parity gate MISMATCH.",
                  flush=True)
            return 5

    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices("cube_tiled_bench_start")

    if args.closed_loop:
        # #921: the closed-loop step fuses the halo collective-permutes with
        # the in-stage mass-fixer psum in ONE executable; on multi-process GPU
        # the NCCL comm-init of those two clique kinds can be ordered
        # differently per rank and DEADLOCK.  Prime every clique in a fixed,
        # rank-independent order FIRST (no-op single-process / CPU-virtual).
        from legoesm.parallel.tiled_production_cdgrid import (
            warmup_tiled_cube_comms,
        )
        warmup_tiled_cube_comms(mesh, args.kt)

    # Time the AUDITED AOT executable itself — jit's dispatch cache does NOT
    # reuse lower().compile()'s output, so calling the jit wrapper would
    # recompile a second (unaudited) executable (codex).
    #
    #   closed loop: s = compiled(s) FEEDBACK — valid because the blocked
    #     step's input layout == output layout (the production contract);
    #     the timed trajectory is a real multi-step integration.
    #   single-shot: each sample re-runs the SAME pristine tile-replicated
    #     input (tile-sharded out != tile-replicated in), measuring
    #     per-step latency WITHOUT the fake output re-replication a
    #     feedback loop would require.
    per_step_ms = []
    s = _lower_arg
    for _ in range(args.steps):
        t0 = time.perf_counter()
        s = compiled(s) if args.closed_loop else compiled(_lower_arg)
        jax.block_until_ready(jax.tree.leaves(s))
        per_step_ms.append((time.perf_counter() - t0) * 1e3)

    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices("cube_tiled_bench_end")

    if args.closed_loop:
        # A feedback trajectory can blow up where pristine-input samples
        # cannot — never record a timing row for a non-finite integration.
        _finite = all(bool(np.all(np.isfinite(np.asarray(x))))
                      for x in jax.tree.leaves(s))
        if not _finite:
            print("ERROR: closed-loop state went non-finite during the "
                  "timed window — refusing to record the row.", flush=True)
            return 6

    steady = per_step_ms[args.warmup:]
    med = float(np.median(steady))
    total_cells = 6 * args.resolution ** 2 * args.nlev
    rec = dict(
        component="atmosphere",
        grid="cubed-sphere",
        mode="strong",
        closed_loop=bool(args.closed_loop),
        kt=args.kt, n_devices=n_devices,
        resolution=args.resolution, nlev=args.nlev,
        steps=args.steps, dt=args.dt,
        platform=jax.default_backend(),
        n_processes=int(jax.process_count()),
        multicontroller=bool(args.multicontroller),
        # compile_ms is the real lowered.compile() wall-clock. per_step_ms[0]
        # is the first TIMED step (first_timed_step_ms); the HLO census — and,
        # under --parity-gate, the untimed parity loop — already dispatched the
        # SAME executable, so it is warm, not a cold compile (codex).
        compile_prewarmed_by_hlo_census=True,
        compile_ms=round(compile_ms, 1),
        first_timed_step_ms=round(per_step_ms[0], 1),
        steady_median_ms=round(med, 2),
        steady_min_ms=round(float(np.min(steady)), 2),
        per_step_ms=[round(x, 1) for x in per_step_ms],
        cells=total_cells,
        hlo_collective_permutes=n_ppermute,
    )
    # Flat aggregator-compatible identity + metric fields (see the latlon
    # twin).  grid_type (not just rec["grid"]) is the aggregator's key.
    rec.update(
        grid_type="cubed-sphere",
        n_levels=args.nlev,
        precision="float64" if jax.config.jax_enable_x64 else "float32",
        physics_level="none",
        backend=jax.default_backend(),
        **tidy_throughput_fields(
            dt_seconds=args.dt, time_per_step_ms=med,
            total_cells=total_cells),
    )
    from legoesm.parallel.early_init import nccl_transport_report
    _nccl_report = nccl_transport_report()
    rec["metadata"] = annotate_incomplete(scaling_metadata(
        grid="cubed-sphere",
        component="atmosphere",
        resolution=f"C{args.resolution}",
        n_levels=args.nlev,
        precision=("float64" if jax.config.jax_enable_x64 else "float32"),
        n_gpus=(n_devices if jax.default_backend() in ("gpu", "cuda",
                                                       "rocm") else 0),
        decomposition="subface_tile",
        solver_variant=("tiled_blocked_loop+fix_mass" if args.closed_loop
                        else "tiled_cc_base_cut"),
        cells_per_rank=total_cells // max(int(jax.process_count()), 1),
        scaling_kind="strong",
        extra={
            # Route-B transport facts (the PBS wrapper's contract): a
            # multi-node row without an NCCL net plugin is falsifiable.
            "nccl": (_nccl_report if args.multicontroller else None),
            "kt": args.kt,
            "steps": args.steps,
            "warmup": args.warmup,
            "multicontroller": bool(args.multicontroller),
            "cells_per_device": total_cells // n_devices,
            "hlo_collective_permutes": n_ppermute,
            "closed_loop": bool(args.closed_loop),
            "envelope": (
                "blocked closed loop + in-stage telescoping mass fixer "
                "(production conservation config; adapter-refused knobs "
                "documented in tiled_step_adapter)" if args.closed_loop
                else "dynamics-only base cut (adapter-refused knobs "
                     "documented in tiled_step_adapter)"),
        },
    ))
    if jax.process_index() == 0:
        outdir = os.path.dirname(args.out)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))
        print(f"[cube-tiled kt={args.kt} nd={n_devices} "
              f"C{args.resolution}x{args.nlev}] "
              f"compile={rec['compile_ms']}ms "
              f"steady_median={med:.2f}ms/step "
              f"ppermutes={n_ppermute}")
        if rec["metadata"]["virtual_cpu_devices"]:
            print("[virtual-cpu] forced host-platform CPU devices: this row "
                  "is a communication-overhead / correctness proxy, NOT "
                  "hardware scaling — do not report it as a speedup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
