"""Minimal multi-controller NCCL canary probe (pure JAX, no legoesm imports).

Answers ONE infra question on Derecho: does route-B ``jax.distributed``
(multi-controller, one process per GPU, collectives over NCCL) initialize and
communicate ACROSS NODES on the Slingshot-11 fabric — where the route-A
cross-node GPU-direct MPI path aborts in the CXI inject (see
docs/performance/multinode_gpu_direct_cxi.md)?

Stages (printed per rank, fail-fast, non-zero exit on any mismatch):
  1. jax.distributed.initialize (explicit coordinator; PALS/OMPI env bridge)
  2. global device visibility (process_count, global/local device counts)
  3. psum over the global ("lat",) mesh == analytic world sum (allreduce path)
  4. ppermute ring shift correct                            (halo path)
  5. ppermute latency loop -> min/median ms — the number that decides whether
     route-B is the multi-node halo transport (compare the ~0.27 ms mpi4jax
     sendrecv floor and the ~0.11 ms ppermute floor measured intra-node in
     docs/performance/scaling/scaling_theoretical_limit_report_2026-06-15.md)

Single-process (size 1) runs every stage on the local device set with no
distributed init — a trivially-green local smoke, so the probe is testable
off-cluster. NO mpi4jax anywhere (mixing mpi4jax with jax.distributed
collectives in one program is the documented mixed-stack deadlock hazard).

Env contract (mc_nccl_canary.sh's per-rank shim bridges PALS -> these):
  OMPI_COMM_WORLD_SIZE / OMPI_COMM_WORLD_RANK  (or --size/--rank)
  LEGOESM_JAX_COORDINATOR = host:port          (or --coordinator)
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--coordinator",
                   default=os.environ.get("LEGOESM_JAX_COORDINATOR"))
    p.add_argument("--size", type=int, default=int(os.environ.get(
        "OMPI_COMM_WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))))
    p.add_argument("--rank", type=int, default=int(os.environ.get(
        "OMPI_COMM_WORLD_RANK", os.environ.get("SLURM_PROCID", "0"))))
    p.add_argument("--iters", type=int, default=50,
                   help="stage-5 ppermute latency-loop iterations")
    args = p.parse_args()
    if args.iters < 1:
        print(f"mc_nccl_probe: --iters must be >= 1, got {args.iters}",
              file=sys.stderr)
        return 2

    import jax

    if args.size > 1:
        if not args.coordinator:
            print("mc_nccl_probe: --coordinator (or LEGOESM_JAX_COORDINATOR) "
                  "required for size > 1", file=sys.stderr)
            return 2
        # BEFORE any backend touch — no jax.process_count()/jax.devices()
        # query here (that would instantiate the local client pre-federation).
        jax.distributed.initialize(
            coordinator_address=args.coordinator,
            num_processes=args.size, process_id=args.rank)

    r = args.rank
    print(f"[rank {r}] stage1 init OK: processes={jax.process_count()} "
          f"backend={jax.default_backend()}", flush=True)
    if args.size > 1 and jax.process_count() != args.size:
        print(f"[rank {r}] FAIL stage1: federated {jax.process_count()} "
              f"processes, launcher started {args.size}", file=sys.stderr)
        return 1

    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    gdev = jax.devices()
    n = len(gdev)
    print(f"[rank {r}] stage2 devices: global={n} "
          f"local={[d.id for d in jax.local_devices()]}", flush=True)

    mesh = Mesh(np.array(gdev), ("lat",))
    x = jnp.arange(n * 8, dtype=jnp.float32).reshape(n, 8)
    xs = jax.device_put(x, NamedSharding(mesh, P("lat", None)))
    perm = [(i, (i + 1) % n) for i in range(n)]

    from jax.experimental.shard_map import shard_map

    def body(xl):
        total = jax.lax.psum(jnp.sum(xl), "lat")          # allreduce path
        y = jax.lax.ppermute(xl, "lat", perm)             # halo path
        return y, total * jnp.ones((1,), jnp.float32)

    f = jax.jit(shard_map(body, mesh=mesh, in_specs=P("lat", None),
                          out_specs=(P("lat", None), P("lat"))))
    y, tot = f(xs)
    rep = NamedSharding(mesh, P())
    y_rep = jax.jit(lambda a: a, out_shardings=rep)(y)
    tot_rep = jax.jit(lambda a: a, out_shardings=rep)(tot)

    expect_sum = float(x.sum())
    got_sum = float(np.asarray(jax.device_get(tot_rep))[0])
    ok_sum = abs(got_sum - expect_sum) < 1e-3
    print(f"[rank {r}] stage3 psum: got={got_sum} expect={expect_sum} "
          f"OK={ok_sum}", flush=True)

    expect_ring = np.roll(np.asarray(x), 1, axis=0)
    got_ring = np.asarray(jax.device_get(y_rep))
    ok_ring = bool(np.array_equal(got_ring, expect_ring))
    print(f"[rank {r}] stage4 ppermute ring OK={ok_ring}", flush=True)
    if not (ok_sum and ok_ring):
        return 1

    # Stage 5: ppermute latency (small message — the halo latency regime).
    small = jax.device_put(
        jnp.ones((n, 64), jnp.float32), NamedSharding(mesh, P("lat", None)))
    g = jax.jit(shard_map(lambda a: jax.lax.ppermute(a, "lat", perm),
                          mesh=mesh, in_specs=P("lat", None),
                          out_specs=P("lat", None)))
    g(small).block_until_ready()                          # compile outside loop
    times_ms = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        g(small).block_until_ready()
        times_ms.append((time.perf_counter() - t0) * 1e3)
    print(f"[rank {r}] stage5 ppermute latency ms: "
          f"min={min(times_ms):.3f} median={sorted(times_ms)[len(times_ms)//2]:.3f} "
          f"(iters={args.iters}, {n} devices, "
          f"{jax.process_count()} processes)", flush=True)
    print(f"[rank {r}] CANARY PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
