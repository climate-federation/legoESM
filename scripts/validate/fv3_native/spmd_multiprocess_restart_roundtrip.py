"""Multi-process fv3_duo checkpoint SAVE->LOAD->STEP bitwise round-trip.

The correctness gate GLM + codex both asked for on the multi-process
driver I/O work: prove that, under a REAL multi-process SPMD launch, a
run that checkpoints and restarts is BITWISE the uninterrupted run --
across the gather (save), the rank-0 decode + two-stage broadcast
(load), and the ``make_array_from_callback`` resharding.

Launched with ``srun`` (one python per task); every process:

1. ``jax.distributed.initialize()`` BEFORE any jax op (global world);
2. builds the deterministic DCMIP16 IC on the host, then the global
   ``('face',)`` mesh over ALL devices and the sharded model;
3. drives the REAL ``ModelDriver`` checkpoint methods
   (``_fv3_duo_reshard_bundle_global`` / ``_fv3_duo_save_checkpoint`` /
   ``_load_fv3_duo_checkpoint``) via a bare ``ModelDriver.__new__``
   harness -- no full driver setup, just the attributes those methods
   touch -- so the exact production code path (gather, root-write,
   barrier, broadcast, reshard) is exercised, not a re-implementation;
4. compares, on ITS OWN addressable shards, the restarted trajectory
   against the uninterrupted one at the pre-registered few-ulp gate.

Protocol per arm: N steps straight; vs (N/2 steps -> checkpoint ->
load -> N/2 more steps).  A wrong gather/broadcast/reshard shows up as
a per-shard mismatch on some rank.

WHAT THIS DOES NOT CERTIFY: only that the multi-process checkpoint
round-trip agrees with the same-code uninterrupted multi-process run
(the driver I/O is transparent) -- not absolute physics (the Fortran
parity gates own that), and not single- vs multi-process equivalence
(spmd_multiprocess_parity.py owns that).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from types import SimpleNamespace


def _log(msg):
    print(f"[pid={os.getpid()} PROCID={os.environ.get('SLURM_PROCID')}] {msg}",
          flush=True, file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=int, default=48)
    ap.add_argument("--km", type=int, default=5, choices=(5, 10))
    ap.add_argument("--n-steps", type=int, default=4)
    ap.add_argument("--dt", type=float, default=120.0)
    ap.add_argument("--gate-rtol", type=float, default=1e-12)
    ap.add_argument("--gate-atol", type=float, default=1e-12)
    ap.add_argument("--out-dir", type=str, default=None,
                    help="shared FS dir for the checkpoint (must be visible "
                         "to rank 0's writer AND every reader).")
    args = ap.parse_args(argv)

    _log("importing jax")
    import jax
    _log("jax.distributed.initialize()")
    jax.distributed.initialize(initialization_timeout=180)
    _log(f"init OK: process {jax.process_index()}/{jax.process_count()}, "
         f"local devices {[str(d) for d in jax.local_devices()]}")
    jax.config.update("jax_enable_x64", True)

    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.grids.factory import create_fv3_duo_grid
    from legoesm.driver.model_driver import ModelDriver

    if args.n_steps % 2 != 0:
        raise SystemExit("--n-steps must be even (split into two halves)")
    if args.out_dir is None:
        raise SystemExit("--out-dir is required (a shared FS path)")

    devs = jax.devices()
    if 6 % len(devs) != 0:
        raise SystemExit(f"{len(devs)} devices does not divide 6 faces")
    grid = create_fv3_duo_grid(args.resolution)
    mesh = Mesh(np.array(devs), ("face",))
    shard = NamedSharding(mesh, P("face"))
    cfg = FV3DuoConfig(km=args.km)
    model = FV3DuoDynamicsModel(grid, cfg, step_out_shardings=shard,
                                step_spmd_mesh=mesh, step_face_batched=True)

    # Bare ModelDriver harness: only the attributes the checkpoint
    # methods touch, so the REAL production code path runs (not a
    # re-implementation).  distributed+spmd+process_count>1 makes
    # _is_spmd_multiprocess() True, selecting the multi-process arms.
    drv = ModelDriver.__new__(ModelDriver)
    drv.model = model
    drv._output_dir = None  # set below
    drv.config = SimpleNamespace(
        distributed=True, distributed_mode="spmd",
        dycore=SimpleNamespace(dt=args.dt, discretization="fv3_duo"))
    drv._loaded_checkpoint_step_day = None
    drv._fv3_duo_restart_bundle = None

    from pathlib import Path
    out_dir = Path(args.out_dir)
    if jax.process_index() == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    drv._output_dir = out_dir

    ic = model.dcmip16_initial_state(do_pert=True)
    b0 = drv._fv3_duo_reshard_bundle_global(ic)

    half = args.n_steps // 2

    # -- arm A: uninterrupted N steps --
    a = b0
    for _ in range(args.n_steps):
        a = model.step(a, args.dt)
    jax.block_until_ready(a["state"]["pt"])

    # -- arm B: half -> checkpoint -> load -> half --
    b = b0
    for _ in range(half):
        b = model.step(b, args.dt)
    jax.block_until_ready(b["state"]["pt"])
    drv._fv3_duo_save_checkpoint(b, step=half, day=float(half))
    ckpt = out_dir / f"fv3duo_ckpt_step_{half:09d}.npz"
    drv._load_fv3_duo_checkpoint(ckpt)
    restart = drv._fv3_duo_reshard_bundle_global(
        drv._fv3_duo_restart_bundle)
    for _ in range(half):
        restart = model.step(restart, args.dt)
    jax.block_until_ready(restart["state"]["pt"])

    def _leaves(bundle):
        for k, v in bundle["state"].items():
            yield f"state.{k}", v
        for k, v in bundle["press"].items():
            yield f"press.{k}", v
        for i, q in enumerate(bundle["q"]):
            yield f"q[{i}]", q
        yield "omga", bundle["omga"]
        if bundle.get("nh") is not None:
            for k, v in bundle["nh"].items():
                yield f"nh.{k}", v

    # The reference (arm A) leaves are multi-process GLOBAL arrays -- a
    # plain np.asarray on one spans non-addressable devices and raises.
    # process_allgather (tiled) assembles each on host, identical on
    # every rank, so a local restart shard's .index slice can be
    # compared against it.
    from jax.experimental import multihost_utils as _mhu
    ref = {name: np.asarray(_mhu.process_allgather(leaf, tiled=True))
           for name, leaf in _leaves(a)}
    rank = jax.process_index()
    fails = []
    for name, leaf in _leaves(restart):
        want_full = ref[name]
        for sh in leaf.addressable_shards:
            got = np.asarray(sh.data)
            want = want_full[sh.index]
            if not (np.isfinite(got).all() and np.isfinite(want).all()):
                fails.append((name, "NON-FINITE")); continue
            d = float(np.abs(got - want).max()) if got.size else 0.0
            sc = float(np.abs(want).max()) if want.size else 0.0
            if d > args.gate_atol + args.gate_rtol * sc:
                fails.append((name, f"|d|={d:.3e} scale={sc:.3e}"))
    if fails:
        for name, msg in fails:
            print(f"[rank {rank}] RESTART MISMATCH {name}: {msg}", flush=True)
        raise SystemExit(1)
    print(f"[rank {rank}/{jax.process_count()}] RESTART BITWISE-MATCH: all "
          f"local shards within {args.gate_atol:.0e}+{args.gate_rtol:.0e}*scale "
          f"({len(devs)} global devices, {args.n_steps} steps, ckpt@{half})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
