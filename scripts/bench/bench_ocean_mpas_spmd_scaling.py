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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_ocean_mpas_scaling import (  # noqa: E402
    _ncells,
    build_global_problem,
    weak_level_for,
)
from metadata import (  # noqa: E402
    annotate_incomplete,
    scaling_metadata,
    tidy_throughput_fields,
    timed_scan_blocks,
)


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
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
    from legoesm.parallel.voronoi_spmd_ocean import (
        build_mpas_ocean_spmd_layout,
        disarm_mpas_ocean_spmd,
        make_sharded_mpas_ocean_step,
        mask_padded_cells,
        n_real_cells,
        shard_state_mpas_ocean_spmd,
    )

    t0 = time.perf_counter()
    # Same config/perturbation as the mpi4jax sibling (implicit-CN production
    # barotropic path); the mesh is rebuilt reordered+padded for nd devices,
    # exactly as run_omip._create_setup does under --enable-mpas-spmd.
    _, z_coord, config, _ = build_global_problem(
        subdivision, args.nlev, barotropic_solver="implicit_cn")
    mesh = create_voronoi_mesh(subdivision_level=subdivision,
                               lloyd_iterations=args.lloyd)
    n_cells_orig = int(mesh.nCells)
    if nd > 1:
        mesh = reorder_voronoi_for_sharding(mesh, nd, method=args.partition_method)
    state = rest_state_mpas_ocean(
        mesh, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0, land_lat_threshold=85.0)
    n_real = n_real_cells(mesh)
    state = mask_padded_cells(state, n_real)
    import jax.numpy as jnp
    mask = state.land_mask.data
    state = state._replace(
        T=state.T.replace(data=state.T.data + 0.5 * jnp.sin(
            4 * mesh.latCell)[:, None] * mask[:, None]),
        eta=state.eta.replace(data=state.eta.data + 0.01 * jnp.sin(
            3 * mesh.lonCell) * mask))
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

        def advance(st):
            return spmd_step(st, args.dt)
    else:
        rounds, cells_per = 0, n_cells_orig

        def advance(st):
            return model.step(st, args.dt)

    try:
        t0 = time.perf_counter()
        jax.block_until_ready(jax.tree.leaves(advance(state)))
        compile_ms = (time.perf_counter() - t0) * 1e3
        state, t = timed_scan_blocks(
            advance, state, block_steps=args.block_steps, n_blocks=args.blocks,
            probe_steps=args.probe_steps, sync_label="ocean_mpas_spmd_bench")
        finite = bool(np.isfinite(np.asarray(state.eta.data)).all())
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
        cells_per_device=cells_per,
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
                                           if args.mode == "weak" else None)},
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
