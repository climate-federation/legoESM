#!/usr/bin/env python
"""MPI transport probe: is the slow halo/allreduce in mpi4jax, MPI or placement?

Times the same exchange through four transports over a payload sweep, under
``mpiexec`` with an even rank count (rank r pairs with r^1):

* ``mpi4py``      -- ``Sendrecv`` on numpy buffers (no JAX at all)
* ``jit_raw``     -- ``mpi4jax.sendrecv`` inside ``jax.jit``
* ``jit_vjp``     -- the model's AD-safe wrapper (``get_sendrecv_vjp``) in jit
* ``eager_raw``   -- ``mpi4jax.sendrecv`` called eagerly

plus a one-scalar allreduce and an alltoall through ``mpi4py`` and
``mpi4jax`` (jit), and K chained ops inside ONE jit (``*_chain`` rows, at
each ``--chain-bytes`` payload): the slope over K is the per-op cost a fused
model step pays, the intercept is JAX's per-dispatch cost of a jit carrying
ordered effects.  An ordered no-op host callback gives that dispatch cost
without MPI.  Eager rows are the cost of an mpi4jax op called outside jit.

Every row's result is checked against the expected values (``ok``), so a
mis-paired or no-op exchange cannot produce a timing.  Each time is the MAX
over ranks of the per-call mean (opening barrier, block on the result every
call).  ``GBps`` for sendrecv counts the bytes SENT by one rank;
``GBps_offrank_per_rank`` for alltoall counts the bytes one rank sends to
other ranks.  Per-rank placement (CPU affinity, thread count after JAX init)
and library versions are recorded.  Writes one JSON line to ``--output``
(rank 0).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time

import numpy as np


def _threads() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("Threads:"):
                return int(line.split()[1])
    return -1


def _time(fn, n_warm, n_time, comm, op_max):
    for _ in range(n_warm):
        fn()
    comm.Barrier()
    t0 = time.perf_counter()
    for _ in range(n_time):
        fn()
    dt = (time.perf_counter() - t0) / n_time
    return comm.allreduce(dt, op=op_max) * 1e6  # us per call


def _ints(v):
    return [int(x) for x in v.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log2-bytes", type=_ints, default=[3, 10, 14, 17, 20, 24],
                    help="log2(payload bytes) per message for the sweep.")
    ap.add_argument("--chain", type=_ints, default=[1, 4, 16, 64],
                    help="K values for K chained ops inside one jit.")
    ap.add_argument("--chain-bytes", type=_ints, default=[8, 1 << 20],
                    help="Payload bytes per message for the chain rows.")
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-timing", type=int, default=200)
    ap.add_argument("--max-rank-bytes", type=int, default=1 << 26,
                    help="Skip alltoall payloads whose per-rank send buffer "
                         "(ranks * bytes per peer) exceeds this.")
    ap.add_argument("--tag", default="")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    from mpi4py import MPI
    import jax
    import jax.numpy as jnp
    import mpi4jax
    from jax.experimental import io_callback
    from legoesm.parallel.halo_exchange import get_sendrecv_vjp

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size % 2:
        raise SystemExit("probe_mpi_transport needs an even rank count")
    peer = rank ^ 1
    jax.block_until_ready(jnp.zeros(1) + 1)  # initialise JAX thread pools
    placement = comm.gather({
        "rank": rank, "host": socket.gethostname(),
        "affinity": sorted(os.sched_getaffinity(0)), "threads": _threads(),
    })
    vjp_sendrecv = get_sendrecv_vjp(mpi4jax)
    raw_sendrecv = mpi4jax.sendrecv

    def timed(fn, n_time=args.n_timing):
        return _time(fn, args.n_warmup, n_time, comm, MPI.MAX)

    def all_ok(flag):
        return bool(comm.allreduce(int(bool(flag)), op=MPI.MIN))

    rows = []
    for lb in args.log2_bytes:
        n = max(1, (1 << lb) // 4)
        nb = n * 4
        a = np.full(n, rank, np.float32)
        b = np.empty_like(a)
        ja = jnp.asarray(a)
        jt = jnp.zeros_like(ja)
        jraw = jax.jit(lambda x, t: raw_sendrecv(x, t, peer, peer, comm=comm))
        jvjp = jax.jit(lambda x, t: vjp_sendrecv(x, t, peer, peer, 0, 0, comm))
        out = {}

        def p4py():
            comm.Sendrecv(a, dest=peer, recvbuf=b, source=peer)
            out["mpi4py"] = b

        def run(name, f):
            def g():
                out[name] = jax.block_until_ready(f())
            return g

        for name, fn in (
                ("mpi4py", p4py),
                ("jit_raw", run("jit_raw", lambda: jraw(ja, jt))),
                ("jit_vjp", run("jit_vjp", lambda: jvjp(ja, jt))),
                ("eager_raw", run("eager_raw", lambda: raw_sendrecv(
                    ja, jt, peer, peer, comm=comm)))):
            us = timed(fn)
            ok = all_ok(np.all(np.asarray(out[name]) == peer))
            rows.append({"op": "sendrecv", "transport": name, "bytes": nb,
                         "us": us, "GBps": nb / (us * 1e-6) / 1e9, "ok": ok})

        # alltoall: every rank sends nb bytes to every rank.
        if size * nb > args.max_rank_bytes:
            rows.append({"op": "alltoall", "bytes_per_peer": nb,
                         "skipped": "send buffer > --max-rank-bytes"})
            continue
        aa = np.full((size, n), rank, np.float32)
        ab = np.empty_like(aa)
        jaa = jnp.asarray(aa)
        ja2a = jax.jit(lambda x: mpi4jax.alltoall(x, comm=comm))
        want = np.arange(size, dtype=np.float32)[:, None]

        def p4py_a2a():
            comm.Alltoall(aa, ab)
            out["a2a_mpi4py"] = ab

        for name, fn in (("mpi4py", p4py_a2a),
                         ("jit_raw", run("a2a_jit_raw", lambda: ja2a(jaa)))):
            us = timed(fn, max(args.n_timing // 4, 10))
            ok = all_ok(np.all(np.asarray(out["a2a_" + name]) == want))
            rows.append({"op": "alltoall", "transport": name,
                         "bytes_per_peer": nb, "us": us,
                         "GBps_offrank_per_rank":
                             nb * (size - 1) / (us * 1e-6) / 1e9, "ok": ok})

    # Chains: K data-dependent ops in ONE jit.  Each op's input is the
    # previous op's output, so XLA cannot drop or reorder them.  Expected
    # results after K steps are closed-form, so a no-op chain fails ``ok``.
    for cb in args.chain_bytes:
        n = max(1, cb // 4)
        ops = {
            # x -> peer's x + 1; after K steps rank's value alternates.
            "sendrecv": (jnp.full(n, float(rank), jnp.float32),
                         lambda x: raw_sendrecv(x, x, peer, peer, comm=comm)),
            # x -> mean over ranks + 1 (stays finite for any K).
            "allreduce": (jnp.full(n, float(rank), jnp.float32),
                          lambda x: mpi4jax.allreduce(
                              x, op=MPI.SUM, comm=comm) * (1.0 / size)),
        }
        if size * cb <= args.max_rank_bytes:
            ops["alltoall"] = (jnp.full((size, n), float(rank), jnp.float32),
                               lambda x: mpi4jax.alltoall(x, comm=comm))
        mean_rank = (size - 1) / 2.0
        for op, (x0, f) in ops.items():
            for k in args.chain:
                def chain(x, k=k, f=f):
                    for _ in range(k):
                        x = f(x) + 1
                    return x
                jc = jax.jit(chain)
                res = {}

                def go(jc=jc, res=res, x0=x0):
                    res["y"] = jax.block_until_ready(jc(x0))

                us = timed(go)
                y = np.asarray(res["y"])
                if op == "sendrecv":
                    expect = (peer if k % 2 else rank) + k
                elif op == "allreduce":
                    expect = mean_rank + k
                else:  # alltoall: row i ends as i's (k-1)-step value, + k
                    expect = None
                if expect is None:
                    ok = np.allclose(y, (np.arange(size)[:, None] if k % 2
                                         else rank) + k)
                else:
                    ok = np.allclose(y, expect)
                rows.append({"op": op + "_chain", "transport": "jit_raw",
                             "k": k, "bytes": n * 4, "us": us,
                             "ok": all_ok(ok)})

    x8 = jnp.ones(2, jnp.float32)
    jcb = jax.jit(lambda x: io_callback(lambda v: v, x, x, ordered=True))
    rows.append({"op": "ordered_callback", "transport": "jax",
                 "us": timed(lambda: jax.block_until_ready(jcb(x8)))})
    s = np.ones(1, np.float32)
    r = np.empty_like(s)
    js = jnp.ones(1, jnp.float32)
    jar = jax.jit(lambda x: mpi4jax.allreduce(x, op=MPI.SUM, comm=comm))
    us = timed(lambda: comm.Allreduce(s, r, op=MPI.SUM))
    rows.append({"op": "allreduce1", "transport": "mpi4py", "us": us,
                 "ok": all_ok(r[0] == size)})
    res = {}
    us = timed(lambda: res.__setitem__("y", jax.block_until_ready(jar(js))))
    rows.append({"op": "allreduce1", "transport": "jit_raw", "us": us,
                 "ok": all_ok(np.asarray(res["y"])[0] == size)})
    jnoop = jax.jit(lambda x: x + 1)
    rows.append({"op": "jit_noop", "transport": "jax",
                 "us": timed(lambda: jax.block_until_ready(jnoop(js)))})

    every_ok = all(r.get("ok", True) for r in rows)  # ok flags are rank-agreed
    if rank == 0:
        import jaxlib
        rec = {
            "tag": args.tag, "n_ranks": size, "rows": rows,
            "all_ok": every_ok,
            "placement": placement,
            "versions": {"jax": jax.__version__, "jaxlib": jaxlib.__version__,
                         "mpi4jax": mpi4jax.__version__,
                         "mpi_lib": MPI.Get_library_version().splitlines()[0]},
            "env": {k: v for k, v in os.environ.items()
                    if k.startswith(("MPICH", "FI_", "XLA", "OMP", "PALS",
                                     "MPI4JAX"))},
        }
        line = json.dumps(rec)
        print(line, flush=True)
        if args.output:
            with open(args.output, "a") as f:
                f.write(line + "\n")
    return 0 if every_ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
