"""Multi-process face-shard parity for the FV3 duo step (SPMD multi-node rung).

Launched with ``srun`` (one python per task); every process:

1. ``jax.distributed.initialize()`` (SLURM auto-detect) -- BEFORE any jax
   op, so ``jax.devices()`` is the GLOBAL world;
2. builds the deterministic DCMIP16 IC on the host (numpy; identical on
   every process by construction);
3. builds the global ``('face',)`` mesh over ALL devices --
   ``build_ring_comm``'s world-span assert refuses a local submesh, the
   multi-node land mine GLM pre-named (halos silently never crossing
   processes);
4. assembles the face-sharded bundle with ``make_array_from_callback``
   (each process materialises ONLY its addressable shards);
5. steps the ring + face-batched jitted lane;
6. computes the SINGLE-device reference locally (the certified loop
   trace on this process's device 0 -- replicated cheap compute, which
   makes every process an independent referee), and compares ITS OWN
   addressable shard of every output leaf against the reference slice.

PASS = every process prints its per-field verdict and asserts within the
pre-registered few-ulp gate (per-field ``atol + rtol*scale``).  A wrong
neighbor graph across processes (device-order bugs, local submesh) is
exactly what the per-process slice comparison catches: process p's faces
would carry another face's halo values.

WHAT THIS DOES NOT CERTIFY (GLM): only that multi-process sharded
execution agrees with an independent single-device recomputation of the
SAME code -- not absolute correctness against the external oracle (the
Fortran parity gates own that), not defects shared by both lanes, and
not shard-decomposition fidelity beyond this probe's own index
arithmetic (``addressable_shards[i].index`` slicing the host reference).
"""
from __future__ import annotations

import argparse

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=int, default=48)
    ap.add_argument("--km", type=int, default=5, choices=(5, 10))
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--dt", type=float, default=120.0)
    ap.add_argument("--gate-rtol", type=float, default=1e-12)
    ap.add_argument("--gate-atol", type=float, default=1e-12)
    args = ap.parse_args(argv)

    import os
    import sys

    def _log(msg):
        print(f"[pre-init pid={os.getpid()} "
              f"SLURM_PROCID={os.environ.get('SLURM_PROCID')}] {msg}",
              flush=True, file=sys.stderr)

    _log("importing jax")
    import jax

    _log("calling jax.distributed.initialize()")
    jax.distributed.initialize(
        initialization_timeout=180)       # SLURM auto-detect; fail FAST
    _log(f"initialize OK: process {jax.process_index()} of "
         f"{jax.process_count()}, local devices "
         f"{[str(d) for d in jax.local_devices()]}")
    jax.config.update("jax_enable_x64", True)
    rank = jax.process_index()
    nproc = jax.process_count()
    devs = jax.devices()                  # GLOBAL world
    if 6 % len(devs) != 0:
        raise SystemExit(
            f"[rank {rank}] {len(devs)} global devices does not divide "
            f"6 faces (need 1/2/3/6)")

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
    FV3DuoConfig,
    FV3DuoDynamicsModel,
    ORACLE_DAMPING,
)
    from legoesm.grids.factory import create_fv3_duo_grid

    grid = create_fv3_duo_grid(args.resolution)
    mesh = Mesh(np.array(devs), ("face",))
    shard = NamedSharding(mesh, P("face"))

    # sharded lane: ring exchanges + face-batched phases + pinned outputs
    model_sh = FV3DuoDynamicsModel(grid, FV3DuoConfig(**ORACLE_DAMPING, km=args.km),
                                   step_out_shardings=shard,
                                   step_spmd_mesh=mesh,
                                   step_face_batched=True)
    # reference lane: the certified loop trace, local single device
    # (pinned: the model default is the batched arm since 2026-09-30)
    model_ref = FV3DuoDynamicsModel(grid, FV3DuoConfig(**ORACLE_DAMPING, km=args.km),
                                    step_face_batched=False)

    ic = model_ref.dcmip16_initial_state(do_pert=True)

    def _host(x):
        return np.asarray(x)

    def _global_arrays(bundle):
        """Host bundle -> face-sharded global jax arrays, each process
        materialising only its addressable shards."""
        def put(x):
            h = _host(x)
            return jax.make_array_from_callback(
                h.shape, shard, lambda idx: h[idx])
        out = {"state": {k: put(v) for k, v in bundle["state"].items()},
               "press": {k: put(v) for k, v in bundle["press"].items()},
               "q": [put(q) for q in bundle["q"]],
               "omga": put(bundle["omga"])}
        out["nh"] = (None if bundle.get("nh") is None else
                     {k: put(v) for k, v in bundle["nh"].items()})
        return out

    b = _global_arrays(ic)
    for _ in range(args.n_steps):
        b = model_sh.step(b, args.dt)
    jax.block_until_ready(b["state"]["pt"])

    r = ic
    for _ in range(args.n_steps):
        r = model_ref.step(r, args.dt)
    jax.block_until_ready(r["state"]["pt"])

    def _leaves(bundle, prefix=""):
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

    ref = dict(_leaves(r))
    fails = []
    for name, leaf in _leaves(b):
        want_full = _host(ref[name])
        for sh_piece in leaf.addressable_shards:
            got = np.asarray(sh_piece.data)
            want = want_full[sh_piece.index]
            if not (np.isfinite(got).all() and np.isfinite(want).all()):
                fails.append((name, "NON-FINITE"))
                continue
            d = float(np.abs(got - want).max()) if got.size else 0.0
            sc = float(np.abs(want).max()) if want.size else 0.0
            if d > args.gate_atol + args.gate_rtol * sc:
                fails.append((name, f"|d|={d:.3e} scale={sc:.3e}"))
    if fails:
        for name, msg in fails:
            print(f"[rank {rank}] GATE FAILED {name}: {msg}", flush=True)
        raise SystemExit(1)
    print(f"[rank {rank}/{nproc}] GATE PASSED: all local shards within "
          f"{args.gate_atol:.0e} + {args.gate_rtol:.0e}*scale "
          f"({len(devs)} global devices)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
