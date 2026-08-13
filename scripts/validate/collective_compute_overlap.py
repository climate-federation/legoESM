"""Can XLA:GPU hide a shard_map ``ppermute`` behind independent compute?

This is the a-priori gate for the interior/rim split (the thing
MPAS-Fortran, ICON and FV3 all do by hand: run the cells that need no
halo while the halo is in flight).  legoESM cannot do it today — the
compiled MPAS step puts 1 instruction between every
``collective-permute-start`` and its ``-done`` — but that is because
nothing in the step is independent of the exchange, not necessarily
because XLA refuses.  Building the split is multi-week work, so test the
compiler's capability first, on a model-free microbenchmark.

Method
------
One ``shard_map`` region issues a ``ppermute`` of a fixed payload and,
in parallel, runs ``n_flop`` rounds of a dependency-chained FMA on data
the exchange never touches.  Sweep ``n_flop``.

    OVERLAP WORKS : total time stays ~flat while the dummy is cheaper
                    than the collective, then rises with slope 1.
                    Reading: t(n) ~ max(t_comm, t_flop(n)).
    NO OVERLAP    : total time rises with slope 1 from n=0, i.e.
                    t(n) ~ t_comm + t_flop(n) — the collective and the
                    compute are serialised and the split buys nothing
                    without also changing how collectives are issued.

The control is the same sweep with the ``ppermute`` removed, which
gives t_flop(n) on its own; the comparison is (with-comm) minus
(no-comm), which must fall toward 0 if the collective is being hidden.

Run (2+ GPUs, one process):
    srun -p gpu-devel --gpus-per-node=2 python <this file>
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from jax import shard_map


def _build(n_dev: int, payload: int, work: int, n_flop: int, comm: bool):
    mesh = Mesh(np.array(jax.devices()[:n_dev]), ("device",))
    perm = [(i, (i + 1) % n_dev) for i in range(n_dev)]

    def body(x, y):
        # x: exchanged payload. y: independent work, never touched by
        # the collective — this is the stand-in for the interior cells.
        if comm:
            x = jax.lax.ppermute(x, "device", perm=perm)
        acc = y
        for _ in range(n_flop):
            acc = acc * 1.0000001 + 1e-7
        return x.sum() + acc.sum()

    f = jax.jit(shard_map(
        body, mesh=mesh, in_specs=(P("device"), P("device")),
        out_specs=P(), check_vma=False))
    x = jnp.ones((n_dev, payload), jnp.float32)
    y = jnp.ones((n_dev, work), jnp.float32)
    return f, x, y


def _time(f, x, y, reps=60):
    f(x, y).block_until_ready()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(x, y).block_until_ready()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-devices", type=int, default=2)
    ap.add_argument("--payload", type=int, default=2_000_000,
                    help="floats exchanged per device (8 MB at 2e6)")
    ap.add_argument("--work", type=int, default=2_000_000,
                    help="floats in the independent FMA chain")
    ap.add_argument("--flops", type=int, nargs="+",
                    default=[0, 1, 2, 4, 8, 16, 32, 64])
    args = ap.parse_args()

    n = min(args.n_devices, jax.device_count())
    if n < 2:
        print("need >= 2 devices")
        return 2
    print(f"devices={n} payload={args.payload} floats "
          f"({args.payload * 4 / 1e6:.1f} MB/device) work={args.work}")

    print(f"{'n_flop':>7} {'comm+work':>10} {'work only':>10} "
          f"{'delta':>8} {'verdict':>9}")
    base_delta = None
    for nf in args.flops:
        f1, x, y = _build(n, args.payload, args.work, nf, comm=True)
        f0, _, _ = _build(n, args.payload, args.work, nf, comm=False)
        t1, t0 = _time(f1, x, y), _time(f0, x, y)
        d = t1 - t0
        if base_delta is None:
            base_delta = d
        # delta = time the collective ADDS on top of the same compute.
        # Constant delta  -> serialised.  Delta shrinking toward 0 as the
        # compute grows -> the collective is being hidden behind it.
        frac = d / base_delta if base_delta > 1e-9 else float("nan")
        print(f"{nf:7d} {t1:10.3f} {t0:10.3f} {d:8.3f} {frac:9.2f}")
    print("\ndelta column is the collective's UNHIDDEN cost (ms). If the "
          "last rows' delta is a small fraction of the n_flop=0 row, XLA "
          "hides collectives behind independent compute and the "
          "interior/rim split is worth building; if it stays ~1.00, it "
          "does not and the split buys nothing on its own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
