"""Measure this machine's ppermute latency + bandwidth for roofline lines.

The ocean/atm SPMD benches compute an analytic `t_bound` (a comm roofline)
from a per-message latency and a link bandwidth, but they ship with
PLACEHOLDER values and therefore report ``bound_calibrated=false`` — so
every "theoretical limit" line drawn from them is generic rather than
machine-specific.

This microbenchmark supplies the two missing constants by timing the SAME
collective the sharded steps use (``jax.lax.ppermute`` on a ring, inside a
``shard_map``), swept over message size. Fitting the classic

    t(bytes) = latency + bytes / bandwidth

to the measured curve yields the intercept (per-message latency) and slope
(achieved link bandwidth). Feed the results back with
``--comm-latency-us`` / ``--comm-bandwidth-gbs``.

Single-process multi-device (NVLink within a node) or multicontroller
(``--multicontroller``, NCCL over the fabric) — the two give different
constants, which is the point: quote the one matching the lane you are
drawing a bound for.

Usage
-----
    # intra-node NVLink, 4 local GPUs
    python scripts/bench/bench_ppermute_microbench.py --n-devices 4

    # inter-node over IB (one process per GPU)
    srun --ntasks=8 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
        python scripts/bench/bench_ppermute_microbench.py \
        --multicontroller --n-devices 8
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from legoesm.parallel.shard_map_compat import shard_map

AXIS = "dev"


def _ring(n):
    return [(i, (i + 1) % n) for i in range(n)]


def time_one(mesh, n_dev, n_elem, dtype, n_warmup, n_iters):
    """Median wall time of one ring ppermute of n_elem elements per device."""
    perm = _ring(n_dev)

    @jax.jit
    def run(x):
        def body(xl):
            return jax.lax.ppermute(xl, axis_name=AXIS, perm=perm)

        # check_vma is the current spelling of the old check_rep (matches
        # sharded_dynamics.py); fall back for older JAX.
        try:
            sm = shard_map(body, mesh=mesh, in_specs=P(AXIS),
                           out_specs=P(AXIS), check_vma=False)
        except TypeError:  # pragma: no cover - JAX < 0.9 spelling
            sm = shard_map(body, mesh=mesh, in_specs=P(AXIS),
                           out_specs=P(AXIS), check_rep=False)
        return sm(x)

    x = jnp.zeros((n_dev * n_elem,), dtype=dtype)
    for _ in range(n_warmup):
        x2 = run(x)
    jax.block_until_ready(x2)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        out = run(x)
        jax.block_until_ready(out)
        times.append((time.perf_counter_ns() - t0) / 1e3)   # us
    return statistics.median(times)


def fit_latency_bandwidth(sizes_bytes, times_us):
    """Least-squares fit of t = a + b*bytes -> (latency_us, bandwidth_GB/s)."""
    x = np.asarray(sizes_bytes, dtype=np.float64)
    y = np.asarray(times_us, dtype=np.float64)
    b, a = np.polyfit(x, y, 1)          # y = b*x + a
    lat_us = float(a)
    # b is us per byte -> bytes per us = 1/b -> GB/s = 1/b * 1e6 / 1e9
    bw_gbs = float(1.0 / b * 1e-3) if b > 0 else float("nan")
    return lat_us, bw_gbs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-devices", type=int, default=0,
                   help="Devices in the ring (0 = all visible).")
    p.add_argument("--multicontroller", action="store_true",
                   help="Federate one process per GPU via jax.distributed "
                        "(inter-node NCCL constants).")
    p.add_argument("--coordinator", default=None)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-iters", type=int, default=50)
    p.add_argument("--out", default=None, help="Append one JSON line here.")
    args = p.parse_args()

    if args.multicontroller:
        from legoesm.parallel.early_init import init_multicontroller_distributed

        init_multicontroller_distributed(args.coordinator)

    devices = jax.devices()
    n_dev = args.n_devices or len(devices)
    if n_dev < 2:
        raise SystemExit(
            f"ppermute needs >=2 devices; got {n_dev}. A latency/bandwidth "
            f"fit from a single device would be meaningless.")
    if n_dev > len(devices):
        raise SystemExit(f"--n-devices {n_dev} > {len(devices)} visible")
    mesh = Mesh(np.array(devices[:n_dev]), (AXIS,))
    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    itemsize = jnp.dtype(dtype).itemsize

    # Sweep from a latency-dominated message to a bandwidth-dominated one.
    elems = [1 << k for k in range(6, 23)]      # 64 .. 4M elements/device
    rows = []
    for n_elem in elems:
        t_us = time_one(mesh, n_dev, n_elem, dtype,
                        args.n_warmup, args.n_iters)
        rows.append((n_elem * itemsize, t_us))
        if jax.process_index() == 0:
            print(f"  {n_elem * itemsize / 1024:10.1f} KiB  {t_us:9.2f} us",
                  flush=True)

    # Latency from the SMALL-message end (where bytes/bandwidth is
    # negligible); bandwidth from the large end. A single global fit is
    # dominated by the large messages and biases the intercept.
    small = [(b, t) for b, t in rows if b <= 64 * 1024]
    large = [(b, t) for b, t in rows if b >= 256 * 1024]
    lat_us = float(np.median([t for _, t in small])) if small else float("nan")
    if len(large) >= 2:
        _, bw_gbs = fit_latency_bandwidth([b for b, _ in large],
                                          [t for _, t in large])
    else:
        bw_gbs = float("nan")

    rec = {
        "component": "comm_microbench",
        "collective": "ppermute_ring",
        "n_devices": n_dev,
        "n_processes": jax.process_count(),
        "multicontroller": bool(args.multicontroller),
        "dtype": args.dtype,
        "backend": jax.default_backend(),
        "latency_us": round(lat_us, 3),
        "bandwidth_gbs": round(bw_gbs, 2),
        "sweep": [{"bytes": b, "median_us": round(t, 3)} for b, t in rows],
        "note": ("latency = median of messages <=64 KiB (latency-dominated); "
                 "bandwidth = slope fit over messages >=256 KiB. Feed to the "
                 "SPMD benches via --comm-latency-us / --comm-bandwidth-gbs "
                 "so t_bound is calibrated for THIS lane."),
    }
    if jax.process_index() == 0:
        print(json.dumps(rec))
        print(f"\n==> --comm-latency-us {rec['latency_us']} "
              f"--comm-bandwidth-gbs {rec['bandwidth_gbs']}")
        if args.out:
            import os
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
