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
import os
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from legoesm.parallel.shard_map_compat import shard_map

AXIS = "dev"


def _ring(n, stride=1):
    """Ring where each device sends to the one ``stride`` places along.

    Every stride is a bijection, so the pattern is always a legal permutation;
    the identity is rejected anyway because a device sending to itself times
    nothing.  The point of the knob is to vary WHERE the partner sits without
    moving a single process: with four GPUs per node, stride 1 keeps three of
    every four links inside a node on NVLink, and any stride of 4 or more puts
    every link on the network, at a distance of ``stride`` ranks.

    Two cautions for anyone reading a stride sweep.  The permutation decomposes
    into ``gcd(n, stride)`` cycles of length ``n / gcd(n, stride)``, and that
    structure -- not just the distance -- can change what the collective library
    does; comparing strides that share a gcd with ``n`` compares two things at
    once.  Strides coprime with ``n`` all give a single cycle and are the safe
    family to sweep.  And stride 1 differs from the rest in KIND, not only in
    distance, because most of its links never leave the node.
    """
    if stride % n == 0:
        raise ValueError(
            f"--ring-stride {stride} is a multiple of the device count {n}: "
            "every device would send to itself, which measures nothing.")
    return [(i, (i + stride) % n) for i in range(n)]


def _build(mesh, n_dev, n_reps, stride=1):
    """jit'd program doing n_reps back-to-back ring ppermutes on device."""
    perm = _ring(n_dev, stride)

    @jax.jit
    def run(x):
        def body(xl):
            def one(_, v):
                return jax.lax.ppermute(v, axis_name=AXIS, perm=perm)

            return jax.lax.fori_loop(0, n_reps, one, xl)

        # check_vma is the current spelling of the old check_rep (matches
        # sharded_dynamics.py); fall back for older JAX.
        try:
            sm = shard_map(body, mesh=mesh, in_specs=P(AXIS),
                           out_specs=P(AXIS), check_vma=False)
        except TypeError:  # pragma: no cover - JAX < 0.9 spelling
            sm = shard_map(body, mesh=mesh, in_specs=P(AXIS),
                           out_specs=P(AXIS), check_rep=False)
        return sm(x)

    return run


def verify_ring(mesh, n_dev, stride):
    """Check the pattern actually delivers, before anything is timed.

    Timing an exchange of zeros cannot distinguish a working permutation from
    one that silently moved nothing, and "the partner did not change" is one of
    the readings a stride sweep has to be able to rule out. So each device
    sends its own index and must receive its source's index.

    Returns the largest disagreement over all devices; zero means every device
    got exactly what the permutation promised.
    """
    perm = _ring(n_dev, stride)

    @jax.jit
    def run(x):
        def body(xl):
            me = jax.lax.axis_index(AXIS)
            sent = jnp.full_like(xl, me)
            got = jax.lax.ppermute(sent, axis_name=AXIS, perm=perm)
            want = jnp.mod(me - stride, n_dev)
            return jax.lax.pmax(jnp.max(jnp.abs(got - want)), AXIS)[None]

        try:
            sm = shard_map(body, mesh=mesh, in_specs=P(AXIS), out_specs=P(AXIS),
                           check_vma=False)
        except TypeError:  # pragma: no cover - JAX < 0.9 spelling
            sm = shard_map(body, mesh=mesh, in_specs=P(AXIS), out_specs=P(AXIS),
                           check_rep=False)
        return sm(x)

    out = run(jnp.zeros((n_dev,), dtype=jnp.int32))
    return int(max(abs(int(v)) for sh in out.addressable_shards
                   for v in np.asarray(sh.data).ravel()))


def _median_us(run, x, n_warmup, n_iters):
    for _ in range(n_warmup):
        out = run(x)
    jax.block_until_ready(out)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        out = run(x)
        jax.block_until_ready(out)
        times.append((time.perf_counter_ns() - t0) / 1e3)   # us
    return statistics.median(times)


def time_one(mesh, n_dev, n_elem, dtype, n_warmup, n_iters, n_reps=64,
             stride=1):
    """Per-ppermute time with HOST DISPATCH SUBTRACTED.

    A single jit call per exchange measures dispatch + launch + wire, and on
    this stack dispatch DOMINATES (287-518 us intercepts on A100 NVLink/IB —
    two orders above the wire latency those fabrics actually have). Timing
    1 rep and ``n_reps`` reps of the same program and taking the difference
    cancels the constant per-call overhead:

        t_per_exchange = (t[n_reps] - t[1]) / (n_reps - 1)

    Returns (per_exchange_us, single_call_us) so the contaminated number
    stays visible alongside the corrected one.
    """
    x = jnp.zeros((n_dev * n_elem,), dtype=dtype)
    t1 = _median_us(_build(mesh, n_dev, 1, stride), x, n_warmup, n_iters)
    tn = _median_us(_build(mesh, n_dev, n_reps, stride), x, n_warmup, n_iters)
    per = (tn - t1) / (n_reps - 1)
    return per, t1


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
    p.add_argument("--n-reps", type=int, default=64,
                   help="Back-to-back ppermutes inside ONE jit call; the "
                        "1-rep vs n-rep difference cancels host dispatch, "
                        "which otherwise dominates the intercept.")
    p.add_argument("--ring-stride", type=int, default=1,
                   help="Each device sends to the one this many places along "
                        "the ring. With four GPUs per node, 1 keeps three of "
                        "every four links on NVLink, 4 puts every link on the "
                        "network, and half the device count puts every link "
                        "across the whole allocation. Changes WHERE the "
                        "partner is without moving any process.")
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

    # The pattern has to be shown to deliver before any of its timings mean
    # anything: a silently degraded permutation still times cleanly.
    mismatch = verify_ring(mesh, n_dev, args.ring_stride)
    if mismatch != 0:
        raise SystemExit(
            f"ring stride {args.ring_stride} on {n_dev} devices did not "
            f"deliver: worst device received an index {mismatch} away from "
            f"its source. Timings from this pattern would be meaningless.")
    if jax.process_index() == 0:
        print(f"ring stride {args.ring_stride} verified on {n_dev} devices",
              flush=True)

    # Sweep from a latency-dominated message to a bandwidth-dominated one.
    elems = [1 << k for k in range(6, 23)]      # 64 .. 4M elements/device
    rows = []
    dispatch_us = []
    for n_elem in elems:
        t_us, t_single = time_one(mesh, n_dev, n_elem, dtype,
                                  args.n_warmup, args.n_iters, args.n_reps,
                                  args.ring_stride)
        rows.append((n_elem * itemsize, t_us))
        dispatch_us.append(t_single)
        if jax.process_index() == 0:
            print(f"  {n_elem * itemsize / 1024:10.1f} KiB  {t_us:9.2f} us "
                  f"(single-call {t_single:8.1f} us incl. dispatch)",
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
        "n_reps": args.n_reps,
        "n_iters": args.n_iters,
        "n_warmup": args.n_warmup,
        "ring_stride": args.ring_stride,
        "ring_verified": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "dispatch_us_median": round(float(np.median(dispatch_us)), 2),
        "dispatch_subtracted": True,
        "sweep": [{"bytes": b, "median_us": round(t, 3)} for b, t in rows],
        "note": ("Per-exchange times have HOST DISPATCH SUBTRACTED via the "
                 "1-rep vs n-rep difference; dispatch_us_median is the "
                 "single-call overhead that was removed (it dominated the "
                 "raw intercept). latency = median of messages <=64 KiB; "
                 "bandwidth = slope fit over messages >=256 KiB. Feed to "
                 "the SPMD benches via --comm-latency-us / "
                 "--comm-bandwidth-gbs so t_bound is calibrated for THIS "
                 "lane."),
    }
    if jax.process_index() == 0:
        print(json.dumps(rec))
        print(f"\n==> --comm-latency-us {rec['latency_us']} "
              f"--comm-bandwidth-gbs {rec['bandwidth_gbs']}")
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
