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


def _build(mesh, n_dev, n_reps, stride=1, collective="ppermute", unroll=False):
    """jit'd program doing n_reps back-to-back collectives on device.

    ``ppermute`` is the neighbour exchange the halo fill uses; ``allreduce``
    is the global reduction the solvers use.  Both are timed here because
    separating them is the whole question: a ring exchange talks to a fixed
    number of partners at any deployment size, while a reduction is a global
    operation whose cost is expected to grow with the rank count.  Measuring
    only one of them cannot say which is responsible for a lane's growth.

    The reduction is divided by the rank count each repetition.  Chaining raw
    sums would multiply the value by ``n_dev`` per repetition and overflow
    long before the loop ends, which would time an exception path rather than
    a collective; the division holds the magnitude steady while keeping each
    repetition dependent on the previous one, so none of them can be folded
    away.
    """
    perm = _ring(n_dev, stride)

    @jax.jit
    def run(x):
        def body(xl):
            if collective == "allreduce":
                def one(_, v):
                    return jax.lax.psum(v, AXIS) / n_dev
            else:
                def one(_, v):
                    return jax.lax.ppermute(v, axis_name=AXIS, perm=perm)

            if unroll:
                # Same program shape as the MPI arm: a Python-unrolled chain
                # with a data dependency between repetitions. Looped-vs-
                # unrolled on ONE transport isolates the loop-shape overhead
                # that would otherwise be confounded with the transport gap.
                v = xl
                for i in range(n_reps):
                    v = one(i, v)
                return v
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


def verify_allreduce(mesh, n_dev):
    """Check the reduction actually reduces, before anything is timed.

    Every device contributes its own index, so a correct sum is the same
    known constant on all of them.  A reduction that silently covered only
    the local shard would return the device's own index instead, and time
    beautifully.

    Returns the largest disagreement with that constant over all devices.
    """
    want = float(n_dev * (n_dev - 1) // 2)

    @jax.jit
    def run(x):
        def body(xl):
            me = jax.lax.axis_index(AXIS).astype(x.dtype)
            got = jax.lax.psum(jnp.full_like(xl, me), AXIS)
            return jnp.abs(got - want)

        try:
            sm = shard_map(body, mesh=mesh, in_specs=P(AXIS), out_specs=P(AXIS),
                           check_vma=False)
        except TypeError:  # pragma: no cover - JAX < 0.9 spelling
            sm = shard_map(body, mesh=mesh, in_specs=P(AXIS), out_specs=P(AXIS),
                           check_rep=False)
        return sm(x)

    out = run(jnp.zeros((n_dev,), dtype=jnp.float32))
    return float(max(abs(float(v)) for sh in out.addressable_shards
                     for v in np.asarray(sh.data).ravel()))


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


# What the OPTIMIZED HLO must contain for each arm. The two gloo arms are XLA
# collectives; the two MPI arms lower to mpi4jax FFI custom calls whose target
# names are fixed in mpi4jax._src.collective_ops (0.9.1: "mpi_sendrecv_ffi",
# "mpi_allreduce_ffi"). Counting these is what makes the "program actually
# contains the primitive" gate non-vacuous for the MPI arms too.
_HLO_TOKEN = {
    "ppermute": "collective-permute",
    "allreduce": "all-reduce",
    "mpi_sendrecv": "mpi_sendrecv_ffi",
    "mpi_allreduce": "mpi_allreduce_ffi",
}
def transport_of(collective):
    """The transport a receipt is attributed to, READ from the runtime.

    The XLA arms are gloo only on CPU with the default collectives
    implementation; JAX_CPU_COLLECTIVES_IMPLEMENTATION can select MPI for the
    same program, and on a GPU backend they are NCCL. A hardcoded label would
    attribute a run to the wrong lane, which is a wrong number, not a missing
    one.
    """
    if collective.startswith("mpi_"):
        return "mpi4jax"
    # Called only AFTER any jax.distributed.initialize(): touching the backend
    # earlier breaks multi-process startup (the previous version did this at
    # import and the gloo arms could not start).
    backend = jax.default_backend()
    if backend == "cpu":
        # JAX builds the gloo/MPI collectives ONLY when a distributed client
        # exists (xla_bridge.make_cpu_client); a single-process run with
        # virtual devices exchanges through local memory whatever the option
        # says, so its label must not name a transport it never used.
        if jax.process_count() == 1:
            return "xla-cpu/local"
        # The RESOLVED option, not the environment variable that may or may
        # not have been applied to it.
        return f"xla-cpu/{str(jax.config.jax_cpu_collectives_implementation).lower()}"
    return f"xla-{backend}"


COLLECTIVE_CHOICES = ("ppermute", "allreduce", "mpi_sendrecv", "mpi_allreduce")


def sweep_elems(max_kib, itemsize, lo_pow=6, hi_pow=23):
    """Element counts per device from 64 up to 4M, capped at ``max_kib``.

    The cap exists because an arm on a slow transport at the top of the sweep
    (8 MiB messages at ~0.1 GB/s are ~75 ms EACH, times n_reps times n_iters)
    can run past a job's per-arm timeout and lose its receipt entirely -- both
    8-rank gloo arms did exactly that on 2026-09-24. A capped sweep still
    brackets the production message (~850 KB) and still fits both ends of the
    latency/bandwidth model; the cap is recorded in the receipt.
    """
    elems = [1 << k for k in range(lo_pow, hi_pow)]
    if max_kib is not None:
        elems = [n for n in elems if n * itemsize <= max_kib * 1024]
    # The fit reads latency from messages <= 64 KiB and bandwidth from a
    # slope over messages >= 256 KiB, so it needs at least one of the former
    # and TWO of the latter. A cap that leaves fewer writes a receipt whose
    # bandwidth is NaN, which is a number that is missing rather than wrong
    # -- but it still costs the arm, so refuse before running it.
    n_small = sum(1 for n in elems if n * itemsize <= 64 * 1024)
    n_large = sum(1 for n in elems if n * itemsize >= 256 * 1024)
    if n_small < 1 or n_large < 2:
        raise SystemExit(
            f"--max-kib {max_kib} leaves {n_small} size(s) <= 64 KiB and "
            f"{n_large} size(s) >= 256 KiB; the latency/bandwidth fit needs "
            f"at least 1 and 2. The smallest usable cap is 512.")
    return elems


def mpi_expected_source(rank, n_dev, stride):
    """Under ``_ring(n_dev, stride)`` device ``rank`` receives from this rank."""
    return (rank - stride) % n_dev


# ---------------------------------------------------------------------------
# MPI arms: the SAME ring and the SAME timing method on mpi4jax instead of the
# jax.distributed (gloo) CPU collectives, so the two transports are compared
# on one instrument. Each rank is a single-process JAX program here -- there is
# no device mesh and no shard_map, which is exactly how the one CPU lane that
# scales (the Voronoi ocean, 62x over 512 ranks) runs. Every message goes
# through the repo's own AD-safe wrapper, get_sendrecv_vjp, the same choke
# point the production halo uses; nothing here re-derives an MPI call.
# ---------------------------------------------------------------------------

def _mpi_modules():
    from legoesm.parallel.reductions import require_mpi_stack
    return require_mpi_stack()


def _MPI():
    return _mpi_modules()[1]


def _build_mpi(comm, n_dev, n_reps, stride=1, collective="mpi_sendrecv"):
    """jit'd program doing n_reps back-to-back MPI ops on this rank.

    The repetitions are a Python-unrolled chain rather than a fori_loop: that
    is the shape the production Voronoi exchange compiles, and mpi4jax orders
    its operations through JAX's effect system, so each call depends on the
    previous one and none can be folded away. The reduction is divided by
    the rank count each repetition for the same overflow reason as the gloo
    arm.
    """
    rank = comm.Get_rank()
    dest = dict(_ring(n_dev, stride))[rank]      # _ring refuses the identity
    source = mpi_expected_source(rank, n_dev, stride)
    mpi4jax, MPI = _mpi_modules()
    from legoesm.parallel.halo_exchange import get_sendrecv_vjp
    from legoesm.parallel.reductions import mpi4jax_array_result
    sendrecv = get_sendrecv_vjp(mpi4jax)

    @jax.jit
    def run(x):
        v = x
        for _ in range(n_reps):
            if collective == "mpi_allreduce":
                v = mpi4jax_array_result(
                    mpi4jax.allreduce(v, op=MPI.SUM, comm=comm)) / n_dev
            else:
                v = sendrecv(v, jnp.zeros_like(v), source, dest, 0, 0, comm)
        return v

    return run


def verify_mpi(comm, n_dev, stride, collective):
    """Known-answer check for the MPI arms; returns the worst disagreement
    over ALL ranks (reduced with a max so every rank agrees on the verdict)."""
    rank = comm.Get_rank()
    dest = dict(_ring(n_dev, stride))[rank]      # _ring refuses the identity
    mpi4jax, MPI = _mpi_modules()
    from legoesm.parallel.halo_exchange import get_sendrecv_vjp
    from legoesm.parallel.reductions import mpi4jax_array_result
    me = jnp.full((8,), rank, dtype=jnp.int32)
    if collective == "mpi_allreduce":
        got = mpi4jax_array_result(mpi4jax.allreduce(me, op=MPI.SUM, comm=comm))
        want = n_dev * (n_dev - 1) // 2
    else:
        sendrecv = get_sendrecv_vjp(mpi4jax)
        got = sendrecv(me, jnp.zeros_like(me),
                       mpi_expected_source(rank, n_dev, stride),
                       dest, 0, 0, comm)
        want = mpi_expected_source(rank, n_dev, stride)
    local = int(np.max(np.abs(np.asarray(got) - want)))
    return int(comm.allreduce(local, op=MPI.MAX))


def count_collectives(run, x, collective):
    """How many of the expected collective the COMPILED program contains.

    The timed program is a fori_loop, so the optimized HLO holds one
    collective inside a while body rather than one per repetition -- this
    counts what is there, and the only reading that matters is ZERO. A loop
    whose collective was folded away, or lowered to something with a
    different name than the one being attributed, would otherwise time
    cleanly and be reported as a communication cost.
    """
    try:
        text = jax.jit(run).lower(x).compile().as_text()
    except Exception as exc:                    # pragma: no cover
        print(f"  (HLO unavailable: {exc})", flush=True)
        return -1
    return text.count(_HLO_TOKEN[collective])


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
             stride=1, collective="ppermute", unroll=False):
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
    if collective.startswith("mpi_"):
        # mesh is the MPI communicator here; every rank holds n_elem elements.
        x = jnp.zeros((n_elem,), dtype=dtype)
        run1 = _build_mpi(mesh, n_dev, 1, stride, collective)
        runn = _build_mpi(mesh, n_dev, n_reps, stride, collective)
        if collective == "mpi_sendrecv":
            # The chain's FINAL VALUE is checkable for a ring: after n_reps
            # steps of stride s, rank r holds what rank (r - n_reps*s) mod n
            # started with. A chain that dropped, aliased or reordered a
            # repetition cannot land on that value; the HLO count alone only
            # shows 64 call sites exist. Checked once per size, on the timed
            # program itself, before it is timed.
            rank = mesh.Get_rank()
            seed = jnp.full((n_elem,), float(rank), dtype=dtype)
            got = np.asarray(runn(seed))
            want = float((rank - n_reps * stride) % n_dev)
            bad = int(mesh.allreduce(int(np.any(got != want)), op=_MPI().MAX))
            if bad:
                raise SystemExit(
                    f"mpi_sendrecv chain of {n_reps} at {n_elem} elements "
                    f"did not deliver the ring's final value on every rank; "
                    f"a repetition was dropped or reordered. Timings from it "
                    f"would be meaningless.")
        t1 = _median_us(run1, x, n_warmup, n_iters)
        tn = _median_us(runn, x, n_warmup, n_iters)
    else:
        x = jnp.zeros((n_dev * n_elem,), dtype=dtype)
        t1 = _median_us(_build(mesh, n_dev, 1, stride, collective, unroll),
                        x, n_warmup, n_iters)
        tn = _median_us(_build(mesh, n_dev, n_reps, stride, collective, unroll),
                        x, n_warmup, n_iters)
    # n_reps repetitions cannot be faster than one. If they are, the loop was
    # folded, the repetitions overlapped, or the timer is measuring noise --
    # and the subtraction below would hand back a NEGATIVE communication cost
    # that reads as a fast lane.
    if tn <= t1:
        raise SystemExit(
            f"{collective} at {n_elem} elements: {n_reps} repetitions "
            f"({tn:.1f} us) were not slower than one ({t1:.1f} us). The "
            f"timed loop is not doing {n_reps} collectives.")
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
    p.add_argument("--collective", choices=list(COLLECTIVE_CHOICES),
                   default="ppermute",
                   help="which collective to time. ppermute is the halo "
                        "fill's neighbour exchange, whose partner count does "
                        "not depend on the deployment size; allreduce is the "
                        "solvers' global reduction, whose cost is expected to "
                        "grow with it. Attributing a lane's growth to one "
                        "requires both. The mpi_* arms are the SAME ring and "
                        "the SAME reduction on mpi4jax instead of the "
                        "jax.distributed (gloo) CPU collectives: launch them "
                        "under srun/mpirun with one rank per process and NO "
                        "--multicontroller; each rank is a single-process JAX "
                        "program, exactly how the one CPU lane that scales "
                        "runs.")
    p.add_argument("--unroll", action="store_true",
                   help="XLA arms only: compile the repetitions as a Python-"
                        "unrolled chain (the MPI arm's program shape) instead "
                        "of a fori_loop. Looped vs unrolled on ONE transport "
                        "isolates loop-shape overhead from the transport gap.")
    p.add_argument("--max-kib", type=int, default=None,
                   help="cap the sweep's largest message (KiB). A slow "
                        "transport at 8 MiB x n_reps x n_iters can outrun a "
                        "job's per-arm timeout and lose the receipt; the cap "
                        "keeps the fit's large end while bounding the arm.")
    p.add_argument("--out", default=None, help="Append one JSON line here.")
    args = p.parse_args()
    if args.n_reps < 8:
        # The per-op cost is (t[n_reps] - t[1]) / (n_reps - 1). With a fast
        # transport and a tiny message the n_reps ops can cost less than the
        # timer's own noise, and the fold guard then reports a loop that "is
        # not doing n_reps collectives" when it is doing exactly that. Eight
        # is the smallest count at which the on-node MPI arm resolves above
        # the ~300 us dispatch it is subtracted from.
        raise SystemExit(
            f"--n-reps {args.n_reps} is too few to resolve a per-op cost "
            f"above the timer noise; use >= 8 (default 64).")

    if args.multicontroller:
        from legoesm.parallel.early_init import init_multicontroller_distributed

        init_multicontroller_distributed(args.coordinator)

    is_mpi = args.collective.startswith("mpi_")
    if is_mpi and args.unroll:
        raise SystemExit("--unroll applies to the XLA arms; the MPI arm is "
                         "always unrolled.")
    if is_mpi:
        if args.multicontroller:
            raise SystemExit(
                "the mpi_* arms run one single-process JAX program per MPI "
                "rank; --multicontroller would ALSO start jax.distributed "
                "and the two would be timing different things.")
        _, MPI = _mpi_modules()
        comm = MPI.COMM_WORLD
        n_dev = comm.Get_size()
        if args.n_devices and args.n_devices != n_dev:
            raise SystemExit(
                f"--n-devices {args.n_devices} but the MPI world has "
                f"{n_dev} ranks; the rank count is what the launcher gave.")
        if n_dev < 2:
            raise SystemExit(
                f"the MPI arms need >=2 ranks; got {n_dev}. Launch under "
                f"srun/mpirun.")
        mesh = comm            # the MPI arms carry the communicator here
        is_root = comm.Get_rank() == 0
    else:
        devices = jax.devices()
        n_dev = args.n_devices or len(devices)
        if n_dev < 2:
            raise SystemExit(
                f"ppermute needs >=2 devices; got {n_dev}. A "
                f"latency/bandwidth fit from a single device would be "
                f"meaningless.")
        if n_dev > len(devices):
            raise SystemExit(f"--n-devices {n_dev} > {len(devices)} visible")
        mesh = Mesh(np.array(devices[:n_dev]), (AXIS,))
        # Under jax.distributed process_index() is the rank; without it
        # every process reports 0, which is why the MPI arms gate on the
        # communicator's rank instead.
        is_root = jax.process_index() == 0
    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    if args.dtype == "float64" and not jax.config.read("jax_enable_x64"):
        raise SystemExit(
            "--dtype float64 without JAX x64 enabled: the arrays would be "
            "float32 on the wire while the receipt records 8-byte elements, "
            "which corrupts every byte count. Set JAX_ENABLE_X64=1.")
    itemsize = jnp.dtype(dtype).itemsize

    # The pattern has to be shown to deliver before any of its timings mean
    # anything: a silently degraded permutation still times cleanly.
    if is_mpi:
        mismatch = verify_mpi(mesh, n_dev, args.ring_stride, args.collective)
        if mismatch != 0:
            raise SystemExit(
                f"{args.collective} on {n_dev} ranks did not deliver: worst "
                f"rank is {mismatch} away from the known answer. Timings "
                f"from it would be meaningless.")
        if is_root:
            print(f"{args.collective} verified on {n_dev} ranks (stride "
                  f"{args.ring_stride})", flush=True)
    elif args.collective == "allreduce":
        mismatch = verify_allreduce(mesh, n_dev)
        if mismatch != 0:
            raise SystemExit(
                f"the reduction on {n_dev} devices did not reduce: worst "
                f"device is {mismatch} away from the known sum. Timings from "
                f"it would be meaningless.")
        if is_root:
            print(f"allreduce verified on {n_dev} devices", flush=True)
    else:
        mismatch = verify_ring(mesh, n_dev, args.ring_stride)
        if mismatch != 0:
            raise SystemExit(
                f"ring stride {args.ring_stride} on {n_dev} devices did not "
                f"deliver: worst device received an index {mismatch} away "
                f"from its source. Timings from this pattern would be "
                f"meaningless.")
        if is_root:
            print(f"ring stride {args.ring_stride} verified on {n_dev} "
                  f"devices", flush=True)

    # The compiled program must actually contain the collective being
    # attributed. Checked once, on the timed program at a mid sweep size.
    if is_mpi:
        probe_x = jnp.zeros((4096,), dtype=dtype)
        n_hlo = count_collectives(
            _build_mpi(mesh, n_dev, args.n_reps, args.ring_stride,
                       args.collective),
            probe_x, args.collective)
    else:
        probe_x = jnp.zeros((n_dev * 4096,), dtype=dtype)
        n_hlo = count_collectives(
            _build(mesh, n_dev, args.n_reps, args.ring_stride,
                   args.collective, args.unroll),
            probe_x, args.collective)
    if n_hlo < 0:
        raise SystemExit(
            "the optimized HLO could not be read, so the program cannot be "
            "shown to contain the collective being attributed.")
    # The XLA arm is a fori_loop (one collective in a while body) unless
    # --unroll; the MPI arm and the unrolled XLA arm carry one per repetition.
    # Anything else means the chain was folded, split, or lowered under
    # another name.
    expect = args.n_reps if (is_mpi or args.unroll) else 1
    if n_hlo != expect:
        raise SystemExit(
            f"the compiled program contains {n_hlo} "
            f"{_HLO_TOKEN[args.collective]} instruction(s); {expect} "
            f"expected for this arm. Whatever this would time, it is not "
            f"{args.n_reps} x {args.collective}.")
    if is_root:
        print(f"optimized HLO contains {n_hlo} "
              f"{_HLO_TOKEN[args.collective]} instruction(s)", flush=True)

    # Sweep from a latency-dominated message to a bandwidth-dominated one.
    elems = sweep_elems(args.max_kib, itemsize)  # 64 .. 4M elements/device
    rows = []
    dispatch_us = []
    for n_elem in elems:
        t_us, t_single = time_one(mesh, n_dev, n_elem, dtype,
                                  args.n_warmup, args.n_iters, args.n_reps,
                                  args.ring_stride, args.collective,
                                  args.unroll)
        rows.append((n_elem * itemsize, t_us))
        dispatch_us.append(t_single)
        if is_root:
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
        "collective": {"allreduce": "allreduce", "ppermute": "ppermute_ring",
                       "mpi_allreduce": "mpi_allreduce",
                       "mpi_sendrecv": "mpi_sendrecv_ring"}[args.collective],
        "transport": transport_of(args.collective),
        "gloo_iface": os.environ.get("LEGOESM_GLOO_IFACE_PINNED"),
        "program_shape": ("unrolled" if (is_mpi or args.unroll) else "loop"),
        "n_devices": n_dev,
        "n_processes": n_dev if is_mpi else jax.process_count(),
        "max_kib": args.max_kib,
        "multicontroller": bool(args.multicontroller),
        "dtype": args.dtype,
        "backend": jax.default_backend(),
        "latency_us": round(lat_us, 3),
        "bandwidth_gbs": round(bw_gbs, 2),
        "n_reps": args.n_reps,
        "n_iters": args.n_iters,
        "n_warmup": args.n_warmup,
        "ring_stride": args.ring_stride,
        "verified": args.collective,
        "hlo_collective_count": n_hlo,
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
    if is_mpi:
        # Every rank must have finished its last timed collective before the
        # receipt exists, so a peer failing late cannot leave a receipt behind.
        mesh.Barrier()
    if is_root:
        print(json.dumps(rec))
        print(f"\n==> --comm-latency-us {rec['latency_us']} "
              f"--comm-bandwidth-gbs {rec['bandwidth_gbs']}")
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    return 0


def _main_abort_on_failure():
    """Under MPI a failure on one rank must take the job down, not strand the
    other ranks in a collective until the walltime expires (which also loses
    the ladder its remaining arms). SystemExit(0) is a clean exit; anything
    else aborts the communicator. Outside MPI this is a plain main()."""
    import sys as _sys
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if code and "--collective" in _sys.argv and any(
                a.startswith("mpi_") for a in _sys.argv):
            print(f"[abort] rank-local failure, aborting the MPI job: {e}",
                  file=_sys.stderr, flush=True)
            _MPI().COMM_WORLD.Abort(code)
        raise
    except BaseException:
        if "--collective" in _sys.argv and any(
                a.startswith("mpi_") for a in _sys.argv):
            import traceback
            traceback.print_exc()
            _MPI().COMM_WORLD.Abort(1)
        raise
    return code


if __name__ == "__main__":
    raise SystemExit(_main_abort_on_failure())
