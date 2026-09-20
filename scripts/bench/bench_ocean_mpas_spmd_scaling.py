"""Strong/weak scaling bench for the SPMD (shard_map) MPAS/Voronoi OCEAN step.

The multi-device ocean-MPAS lane is ``legoesm.parallel.voronoi_spmd_ocean``
(``run_omip.py --enable-mpas-spmd``); this bench times exactly that step on
the same reordered+padded global mesh, NCCL on GPU / gloo on CPU via
``--multicontroller`` (identical contract to ``bench_mpas_spmd_scaling``).
The mpi4jax sibling ``bench_ocean_mpas_scaling`` cannot run multi-GPU on
Levante (no CUDA-aware OpenMPI), which is why this lane exists.

Problem: ``bench_ocean_mpas_scaling.build_global_problem`` (implicit-CN
barotropic solve = the OMIP production path, conservation fixers ON).
Timing: ``metadata.timed_scan_blocks`` (fused lax.scan blocks, slowest
process per block) -> ``steady_median_ms`` is the fused per-step time.

  srun -n 8 python bench_ocean_mpas_spmd_scaling.py --multicontroller \
      --n-devices 8 --subdivision 8 --nlev 40 --out rows.jsonl
  weak: --mode weak --cells-per-device 40962  (level quantized, 4x per level)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_ocean_mpas_scaling import (  # noqa: E402
    _ncells,
    build_problem_config,
    perturbed_rest_state,
)
from metadata import (  # noqa: E402
    annotate_incomplete,
    scaling_metadata,
    tidy_throughput_fields,
    timed_scan_blocks,
)


def weak_level_for(cells_per_device: int, n_devices: int, levels=range(2, 11)) -> int:
    """Icosahedral level whose cells/device is nearest the target (4x per
    level, so the campaign's targets are hit exactly at s5..s10); refuses a
    target more than 2x off every level so a capped ladder cannot pass as
    weak scaling."""
    best = min(levels, key=lambda lv: abs(_ncells(lv) / n_devices - cells_per_device))
    ratio = _ncells(best) / n_devices / cells_per_device
    if not 0.5 <= ratio <= 2.0:
        raise SystemExit(f"weak target {cells_per_device} cells/device at {n_devices} "
                         f"devices: nearest level {best} gives ratio {ratio:.2f}")
    return best


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-devices", type=int, required=True)
    p.add_argument("--subdivision", type=int, default=6)
    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
    p.add_argument("--cells-per-device", type=int, default=40962,
                   help="weak mode: target cells/device (level quantized)")
    p.add_argument("--nlev", type=int, default=40)
    p.add_argument("--dt", type=float, default=600.0)
    p.add_argument("--lloyd", type=int, default=0,
                   help="Lloyd iterations of the cached mesh (0 = the "
                        "campaign scaling meshes)")
    p.add_argument("--partition-method", default="auto",
                   choices=["auto", "geometric", "metis", "sfc"])
    p.add_argument("--block-steps", type=int, default=20)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--probe-steps", type=int, default=3)
    p.add_argument("--multicontroller", action="store_true")
    p.add_argument("--pcg-precond", default="jacobi",
                   choices=["jacobi", "poly"],
                   help="distributed PCG preconditioner; 'poly' is the "
                        "communication-free local Neumann polynomial")
    p.add_argument("--pcg-poly-sweeps", type=int, default=4,
                   help="sweeps K of the local polynomial preconditioner")
    # Barotropic-solve comm knobs. The distributed implicit_cn solve costs
    # 1+2M batched allreduces per step at pcg_variant="standard" and 1+M at
    # "single_reduce" (Chronopoulos-Gear, parity-gated), plus one cell-halo
    # exchange per PCG iteration -- i.e. the solve's communication is set by
    # these two numbers alone. They are exposed so a ladder arm can measure
    # how much of the plateau the barotropic solve owns, instead of inferring
    # it from a reduction count.
    p.add_argument("--pcg-variant", choices=["standard", "single_reduce"],
                   default="standard")
    p.add_argument("--pcg-fixed-iters", type=int, default=None,
                   help="distributed PCG iteration count (config default 60); "
                        "a PROBE knob -- lowering it changes the solve")
    p.add_argument("--eta-clamp-iters", type=int, default=3)
    p.add_argument("--out", type=str, default="ocean_mpas_spmd_scaling.jsonl")
    args = p.parse_args()

    import jax

    if args.multicontroller:
        from legoesm.parallel.early_init import init_jax_distributed_with_fallback
        init_jax_distributed_with_fallback()

    nd = args.n_devices
    avail = len(jax.devices())
    if avail < nd or (args.multicontroller and nd != avail):
        raise SystemExit(f"--n-devices {nd} vs {avail} visible devices "
                         f"({jax.process_count()} processes)")

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64() if jax.config.jax_enable_x64
               else PrecisionPolicy.fp32())
    precision = "float64" if jax.config.jax_enable_x64 else "float32"

    subdivision = args.subdivision
    if args.mode == "weak":
        subdivision = weak_level_for(args.cells_per_device, nd)

    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
    from legoesm.parallel.voronoi_spmd_ocean import (
        build_mpas_ocean_spmd_layout,
        disarm_mpas_ocean_spmd,
        make_sharded_mpas_ocean_step,
        n_real_cells,
        shard_state_mpas_ocean_spmd,
    )

    t0 = time.perf_counter()
    # Same config/perturbation as the mpi4jax sibling (implicit-CN production
    # barotropic path) on the ONE requested mesh, reordered+padded for nd
    # devices exactly as run_omip._create_setup does under --enable-mpas-spmd.
    z_coord, config = build_problem_config(
        args.nlev, barotropic_solver="implicit_cn",
        pcg_variant=args.pcg_variant,
        eta_floor_clamp_iters=args.eta_clamp_iters)
    if args.pcg_fixed_iters is not None:
        config = config._replace(
            barotropic_implicit_pcg_fixed_iters=int(args.pcg_fixed_iters))
    config = config._replace(
        barotropic_implicit_pcg_precond=str(args.pcg_precond),
        barotropic_implicit_pcg_poly_sweeps=int(args.pcg_poly_sweeps))
    mesh = create_voronoi_mesh(subdivision_level=subdivision,
                               lloyd_iterations=args.lloyd)
    n_cells_orig = int(mesh.nCells)
    if nd > 1:
        mesh = reorder_voronoi_for_sharding(mesh, nd, method=args.partition_method)
    n_real = n_real_cells(mesh)
    state = perturbed_rest_state(mesh, z_coord, n_cells_real=n_real)
    model = MPASOceanModel(mesh, z_coord, config)
    setup_s = time.perf_counter() - t0

    if nd > 1:
        layout = build_mpas_ocean_spmd_layout(
            mesh, nd, n_cells_real=n_real,
            tracer_advection=str(config.tracer_advection), nlev=args.nlev)
        spmd_step = make_sharded_mpas_ocean_step(model, layout)
        state = shard_state_mpas_ocean_spmd(state, layout)
        rounds = len(layout.ppermute_perms)
        cells_per = int(layout.cells_per)
        # The sharded geometry stacks cross the timing helper's jit as an
        # ARGUMENT: closed over, they are outer-trace constants jax cannot
        # fetch for non-addressable arrays under multicontroller.
        aux = spmd_step.aux

        def advance(st, aux):
            return spmd_step(st, args.dt, aux=aux)
    else:
        rounds, cells_per, aux = 0, n_cells_orig, None

        def advance(st, aux=None):   # timed_scan_blocks calls 1-arg when aux is None
            return model.step(st, args.dt)

    @jax.jit
    def _all_finite(st):
        return jnp.all(jnp.array([jnp.isfinite(l).all()
                                  for l in jax.tree.leaves(st)]))

    try:
        t0 = time.perf_counter()
        jax.block_until_ready(jax.tree.leaves(advance(state, aux)))
        compile_ms = (time.perf_counter() - t0) * 1e3
        state, t = timed_scan_blocks(
            advance, state, block_steps=args.block_steps, n_blocks=args.blocks,
            probe_steps=args.probe_steps, sync_label="ocean_mpas_spmd_bench",
            aux=aux)
        # jitted global reduction -> replicated scalar (fully addressable) over
        # EVERY prognostic leaf, not a host fetch of one sharded field.
        finite = bool(_all_finite(state))
    finally:
        if nd > 1:
            disarm_mpas_ocean_spmd()

    med = float(t["fused_step_ms"])
    total_cells = n_cells_orig * args.nlev
    rec = dict(
        component="mpas_ocean", grid_type="mpas", subdivision=subdivision,
        resolution=subdivision, n_devices=nd, n_cells=n_cells_orig,
        n_cells_padded=int(mesh.nCells), nlev=args.nlev, n_levels=args.nlev,
        mode=args.mode, precision=precision, dt=args.dt,
        lloyd_iterations=args.lloyd, partition_method=args.partition_method,
        barotropic_solver="implicit_cn",
        platform=jax.default_backend(), backend=jax.default_backend(),
        n_processes=jax.process_count(),
        multicontroller=bool(args.multicontroller),
        compile_ms=round(compile_ms, 1), setup_s=round(setup_s, 1),
        steady_median_ms=round(med, 4), finite_ok=finite, valid=finite,
        cells=total_cells, ppermute_rounds=rounds,
        cells_per_device=cells_per,                 # padded shard size
        cells_per_device_real=n_cells_orig // nd,
        **{k: t[k] for k in ("block_ms", "parallel_block_ms",
                             "step_latency_ms", "rank_imbalance") if k in t},
        **tidy_throughput_fields(dt_seconds=args.dt, time_per_step_ms=med,
                                 total_cells=total_cells),
    )
    rec["metadata"] = annotate_incomplete(scaling_metadata(
        grid="mpas", component="ocean", resolution=f"L{subdivision}",
        n_levels=args.nlev, precision=precision,
        n_gpus=nd if jax.default_backend() in ("gpu", "cuda", "rocm") else 0,
        decomposition="cell_partition" if nd > 1 else "none",
        solver_variant="implicit_cn",
        cells_per_rank=total_cells // max(jax.process_count(), 1),
        scaling_kind=args.mode,
        extra={"cells_per_device": cells_per, "ppermute_rounds": rounds,
               "block_steps": args.block_steps, "blocks": args.blocks,
               "target_cells_per_device": (args.cells_per_device
                                           if args.mode == "weak" else None),
               # The barotropic solve's whole communication bill, so a row
               # can never be compared against one that solved differently:
               # allreduces/step = 1+2M (standard) or 1+M (single_reduce),
               # plus M cell-halo exchanges. Single-device rows run stock CG
               # to a tolerance instead, so they do NOT do fixed_iters work.
               "pcg_variant": args.pcg_variant,
               "pcg_fixed_iters": int(config.barotropic_implicit_pcg_fixed_iters),
               "pcg_precond": str(config.barotropic_implicit_pcg_precond),
               "pcg_poly_sweeps": int(config.barotropic_implicit_pcg_poly_sweeps),
               "pcg_solver_path": ("fixed_iter_pcg" if nd > 1 else "stock_cg_to_tol"),
               "eta_floor_clamp_iters": args.eta_clamp_iters,
               "barotropic_allreduces_per_step": (
                   1 + (1 if args.pcg_variant == "single_reduce" else 2)
                   * int(config.barotropic_implicit_pcg_fixed_iters)
                   if nd > 1 else None)},
    ))
    if jax.process_index() == 0:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"ocean-mpas SPMD L{subdivision} nlev={args.nlev} nd={nd} "
              f"{precision} {jax.default_backend()}: {med:.3f} ms/step "
              f"(rounds={rounds}, finite={finite})", flush=True)
    return 0 if finite else 3


if __name__ == "__main__":
    raise SystemExit(main())
