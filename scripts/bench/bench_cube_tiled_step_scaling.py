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
  * --parity-gate: N steps vs the serial untiled step at the adapter gate's
    f32-honest tolerances (single-process only; smoke windows).

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

from metadata import annotate_incomplete, scaling_metadata  # noqa: E402

#: Parity tolerances vs the serial untiled step — the adapter gate's
#: f32-honest bounds (exact f32 ulps of the field scales over a smoke
#: window; a real stage regression is 2e-5-abs class at 3 steps).
TILED_PARITY_ATOL = {"u": 2e-5, "v": 2e-5, "T": 1e-4, "p_s": 0.06}
PARITY_MAX_STEPS = 6


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
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--dt", type=float, default=60.0)
    p.add_argument("--parity-gate", action="store_true",
                   help="Gate vs the serial untiled step (single-process "
                        "smoke windows only).")
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
    if args.parity_gate and args.steps > PARITY_MAX_STEPS:
        raise SystemExit(
            f"--parity-gate is a smoke gate; --steps {args.steps} > "
            f"{PARITY_MAX_STEPS} cap.")

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

    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationConfig,
        CDGridPrimitiveEquationModel,
    )
    from legoesm.atmosphere.dynamics.tiled_step_adapter import (
        make_tiled_cc_step,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.halo import set_halo_backend
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.parallel.cubesphere_exchange import find_fullcube_allgathers

    # Base-cut envelope config (the adapter validates + refuses anything
    # outside it): dynamics only, fixers externalized.
    grid = create_cubed_sphere(args.resolution)
    coord = create_sigma_coordinate(args.nlev)
    cfg = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=False, fix_mass=False)
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    state0 = held_suarez_init(grid, coord)

    # Serial reference BEFORE any mesh/backend arming (parity gate).
    serial_ref = None
    if args.parity_gate:
        ref = state0
        for _ in range(args.steps):
            ref = model.step(ref, args.dt)
        jax.block_until_ready(jax.tree.leaves(ref))
        serial_ref = {nm: np.asarray(getattr(ref, nm).data)
                      for nm in ("u", "v", "T", "p_s")}

    # RAW (6, kt, kt) mesh + local halo backend — the adapter-gate setup
    # (the tiled stage manages its own ppermutes through the mesh).
    set_halo_backend("local")
    dev = np.array(jax.devices()[:n_devices]).reshape(6, args.kt, args.kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    tiled_step = jax.jit(make_tiled_cc_step(model, mesh, kt=args.kt,
                                            dt=args.dt))

    # --- Anti-fake HLO census on the compiled step --------------------------
    # The SAME jitted callable is audited AND timed (auditing a separate
    # jit while timing the bare adapter would census a different
    # executable; codex) — and the audit compile is reused by the timed
    # loop (per_step_ms[0] is then dispatch, not compile; recorded).
    lowered = tiled_step.lower(state0)
    compiled = lowered.compile()
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

    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices("cube_tiled_bench_start")

    # Time the AUDITED AOT executable itself — jit's dispatch cache does
    # NOT reuse lower().compile()'s output, so calling the jit wrapper
    # would recompile a second (unaudited) executable (codex).
    per_step_ms = []
    s = state0
    for _ in range(args.steps):
        t0 = time.perf_counter()
        s = compiled(s)
        jax.block_until_ready(jax.tree.leaves(s))
        per_step_ms.append((time.perf_counter() - t0) * 1e3)

    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices("cube_tiled_bench_end")

    if args.parity_gate:
        ok = True
        for nm in ("u", "v", "T", "p_s"):
            got = np.asarray(getattr(s, nm).data)
            mx = float(np.max(np.abs(got - serial_ref[nm])))
            field_ok = mx < TILED_PARITY_ATOL[nm]
            ok &= field_ok
            print(f"    parity {nm:>4s}: max|diff|={mx:.3e} "
                  f"{'OK' if field_ok else 'MISMATCH'}", flush=True)
        if not ok:
            print("ERROR: tiled-vs-serial parity gate MISMATCH.",
                  flush=True)
            return 5

    steady = per_step_ms[args.warmup:]
    med = float(np.median(steady))
    total_cells = 6 * args.resolution ** 2 * args.nlev
    rec = dict(
        component="atmosphere",
        grid="cubed-sphere",
        mode="strong",
        kt=args.kt, n_devices=n_devices,
        resolution=args.resolution, nlev=args.nlev,
        steps=args.steps, dt=args.dt,
        platform=jax.default_backend(),
        n_processes=int(jax.process_count()),
        multicontroller=bool(args.multicontroller),
        # The HLO census pre-compiled the SAME executable, so step 0 is
        # dispatch, not compile — recorded honestly.
        compile_prewarmed_by_hlo_census=True,
        compile_ms=round(per_step_ms[0], 1),
        steady_median_ms=round(med, 2),
        steady_min_ms=round(float(np.min(steady)), 2),
        per_step_ms=[round(x, 1) for x in per_step_ms],
        cells=total_cells,
        hlo_collective_permutes=n_ppermute,
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
        solver_variant="tiled_cc_base_cut",
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
            "envelope": "dynamics-only base cut (adapter-refused knobs "
                        "documented in tiled_step_adapter)",
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
