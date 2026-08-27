"""FV3 duo multi-process SPMD scaling: sharded vs single-device wall time.

The one regime the 2-GPU verdict left open: multi-NODE aggregate
bandwidth (3 nodes x 2 GPU = 6 devices, 1 face/device).  The 2xPCIe
single-node measurement was comm-bound and SLOWER than single-device;
this asks whether spreading the six faces over three nodes' inter-node
links crosses over.

Launched with ``srun`` (one python per task, 6 tasks = 6 global
devices).  Every process:

1. ``jax.distributed.initialize()`` (global world);
2. builds the deterministic IC + the global ``('face',)`` mesh;
3. times TWO arms after a warmup step that pays compile:
   - SINGLE-DEVICE: this process's local (unsharded) model, one card;
   - SHARDED: the 6-device face-sharded + ring + face-batched step.
4. reports per-step wall (median over --n-timed) for both, and the
   speedup t_single/t_shard (>1 = the sharded run is FASTER = crossover).

DESCRIPTIVE timing (RULE: pin the card -- ``--constraint=a100`` in the
sbatch, the 2-GPU trap was an accidental A100 landing).  Correctness is
NOT re-checked here (spmd_multiprocess_parity + the restart round-trip
own that); this is wall time only.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time


def _log(msg):
    print(f"[pid={os.getpid()} PROCID={os.environ.get('SLURM_PROCID')}] {msg}",
          flush=True, file=sys.stderr)


def _time_steps(model, bundle, dt, n_warmup, n_timed):
    import jax
    b = bundle
    for _ in range(n_warmup):                 # pays compile + warms caches
        b = model.step(b, dt)
    jax.block_until_ready(b["state"]["pt"])
    per_step = []
    for _ in range(n_timed):
        t0 = time.time()
        b = model.step(b, dt)
        jax.block_until_ready(b["state"]["pt"])
        per_step.append(time.time() - t0)
    return b, per_step


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=int, default=192)
    ap.add_argument("--km", type=int, default=10, choices=(5, 10))
    ap.add_argument("--dt", type=float, default=120.0)
    ap.add_argument("--n-warmup", type=int, default=2)
    ap.add_argument("--n-timed", type=int, default=20)
    ap.add_argument("--skip-single", action="store_true",
                    help="skip the single-device arm (e.g. it OOMs at "
                         "this card size); report sharded wall only.")
    args = ap.parse_args(argv)

    _log("importing jax")
    import jax
    _log("jax.distributed.initialize()")
    jax.distributed.initialize(initialization_timeout=180)
    jax.config.update("jax_enable_x64", True)
    rank = jax.process_index()
    nproc = jax.process_count()
    _log(f"init OK: process {rank}/{nproc}, local {jax.local_devices()}")

    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.grids.factory import create_fv3_duo_grid

    devs = jax.devices()
    if 6 % len(devs) != 0:
        raise SystemExit(f"{len(devs)} devices does not divide 6 faces")
    grid = create_fv3_duo_grid(args.resolution)
    mesh = Mesh(np.array(devs), ("face",))
    shard = NamedSharding(mesh, P("face"))
    cfg = FV3DuoConfig(km=args.km)

    model_sh = FV3DuoDynamicsModel(grid, cfg, step_out_shardings=shard,
                                   step_spmd_mesh=mesh,
                                   step_face_batched=True)
    ic = model_sh.dcmip16_initial_state(do_pert=True)

    def _put(x):
        h = np.asarray(x)
        return jax.make_array_from_callback(h.shape, shard, lambda idx: h[idx])

    def _global(bundle):
        out = {"state": {k: _put(v) for k, v in bundle["state"].items()},
               "press": {k: _put(v) for k, v in bundle["press"].items()},
               "q": [_put(q) for q in bundle["q"]],
               "omga": _put(bundle["omga"])}
        out["nh"] = (None if bundle.get("nh") is None else
                     {k: _put(v) for k, v in bundle["nh"].items()})
        return out

    b_sh = _global(ic)
    _, sh_steps = _time_steps(model_sh, b_sh, args.dt,
                              args.n_warmup, args.n_timed)
    sh_med = statistics.median(sh_steps)

    single_med = None
    if not args.skip_single:
        # single-device: this process's OWN local card, unsharded model.
        try:
            model_1 = FV3DuoDynamicsModel(grid, cfg)
            _, s1 = _time_steps(model_1, ic, args.dt,
                                args.n_warmup, args.n_timed)
            single_med = statistics.median(s1)
        except Exception as exc:               # OOM at big card sizes
            _log(f"single-device arm failed ({type(exc).__name__}: {exc}); "
                 f"reporting sharded only")

    if rank == 0:
        card = str(jax.local_devices()[0])
        line = (f"[SCALING] C{args.resolution} km={args.km} "
                f"{len(devs)}dev({nproc}proc) {card}: "
                f"sharded {sh_med * 1e3:.1f} ms/step")
        if single_med is not None:
            line += (f" | single-device {single_med * 1e3:.1f} ms/step"
                     f" | speedup {single_med / sh_med:.2f}x"
                     f" ({'FASTER' if single_med > sh_med else 'SLOWER'})")
        else:
            line += " | single-device SKIPPED/OOM"
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
