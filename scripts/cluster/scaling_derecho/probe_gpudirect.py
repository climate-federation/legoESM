#!/usr/bin/env python
"""Fast GPU-direct fabric probe for Derecho (no dycore, ~seconds).

Reproduces the EXACT MPI operation that aborts the multi-node GPU scaling job:
a 4-byte (``scount=1 MPI_FLOAT``) device-pointer ``sendrecv`` across a node
boundary. On Slingshot/CXI this goes through the OFI *inject* path, which needs
a GPU-memory registration route (GDRCopy) to accept a device pointer — without
it the send aborts with ``cxil_map: write error`` / ``injectdata Invalid
argument``. See docs/performance/multinode_gpu_direct_cxi.md.

Use it to regression-check the fabric and to iterate on env fixes (load GDRCopy,
flip an inject knob) in seconds instead of re-queuing a 16-GPU model job.

Run (inside a >=2-node GPU allocation) via the wrapper:
    bash scripts/cluster/scaling_derecho/probe_gpudirect.sh
    LOAD_GDRCOPY=1 bash scripts/cluster/scaling_derecho/probe_gpudirect.sh

or directly:
    mpiexec --ppn 4 -n 8 bash -c 'export CUDA_VISIBLE_DEVICES=${PALS_LOCAL_RANKID:-0}; exec "$@"' _ \
        python scripts/cluster/scaling_derecho/probe_gpudirect.py

PASS = every rank prints ``backend=gpu`` and the correct neighbour value, and
rank 0 prints ``PASS``. Two traps that produce a meaningless "pass":
  * a CPU backend tests nothing — require ``backend=gpu``;
  * ``MPI4JAX_USE_CUDA_MPI=1`` must be set, else the halo host-stages and the
    probe never exercises the CXI inject path (the misread behind the earlier
    false "RESOLVED").
"""
from __future__ import annotations

import argparse
import os


def expected_recv_value(rank: int, nproc: int) -> float:
    """Value ``rank`` must receive in the ring ``sendrecv``.

    Each rank sends payload ``rank + 1`` to ``(rank + 1) % nproc`` and receives
    from ``(rank - 1) % nproc``; so the received value is ``sender + 1`` where
    ``sender = (rank - 1) % nproc``. Pure (no MPI) so it is unit-testable.
    """
    if nproc <= 0:
        raise ValueError(f"nproc must be positive, got {nproc}")
    sender = (rank - 1) % nproc
    return float(sender + 1)


def n_elements(nbytes: int) -> int:
    """float32 element count for a message of ``nbytes`` (>=1, 4-byte floats)."""
    if nbytes < 4:
        raise ValueError(f"nbytes must be >= 4, got {nbytes}")
    return max(1, nbytes // 4)


def parse_args(argv=None) -> argparse.Namespace:
    """Escalation knobs that march the trivial ring toward the model halo.

    The basic 4-byte ring passes cross-node on cray-mpich 9.0.0; the model
    aborts. These flags close the gap one axis at a time so the trigger can be
    isolated:
      --jit    run the sendrecv inside jax.jit (model does mpi4jax under XLA)
      --iters  repeat the (jitted) exchange N times (repeated invocation)
      --bytes  realistic message size instead of 4 bytes
    """
    p = argparse.ArgumentParser(description="GPU-direct fabric probe")
    p.add_argument("--jit", action="store_true",
                   help="wrap the exchange in jax.jit (compiled mpi4jax path)")
    p.add_argument("--iters", type=int, default=1,
                   help="repeat the exchange this many times")
    p.add_argument("--bytes", type=int, default=4, dest="nbytes",
                   help="message size in bytes (>=4, rounded to float32)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    # Heavy imports live INSIDE main so the module stays import-safe (and
    # unit-testable) on a host without jax/mpi4py/mpi4jax.
    import jax
    import jax.numpy as jnp
    import mpi4jax
    from mpi4py import MPI

    args = parse_args(argv)
    comm = MPI.COMM_WORLD
    r, n = comm.Get_rank(), comm.Get_size()
    backend = jax.default_backend()
    cuda_mpi = os.environ.get("MPI4JAX_USE_CUDA_MPI", "<unset>")
    src, dst = (r - 1) % n, (r + 1) % n

    # nelem float32 = nbytes, resident on the pinned A100 (JAX default device).
    # The model's FAILING message is itself tiny (scount=1 MPI_FLOAT); --bytes is
    # for stressing the registration/MR-cache path, not because size is the known
    # trigger. The payload is constant across iters, so `want` is invariant.
    nelem = n_elements(args.nbytes)
    x = jnp.ones(nelem, dtype=jnp.float32) * (r + 1)

    def _exchange(buf):
        return mpi4jax.sendrecv(buf, buf, source=src, dest=dst, comm=comm)
    if args.jit:
        _exchange = jax.jit(_exchange)

    y = x
    for _ in range(max(1, args.iters)):
        y = _exchange(x)          # send the SAME payload each iter -> want invariant
        y.block_until_ready()     # force the (lazy) collective to actually run

    want = expected_recv_value(r, n)
    ok = bool((y == want).all()) and (backend == "gpu")
    print(
        f"rank {r}/{n} host={MPI.Get_processor_name()} backend={backend} "
        f"cuda_mpi={cuda_mpi} jit={args.jit} iters={args.iters} bytes={args.nbytes} "
        f"recv={float(y[0])} want={want} {'OK' if ok else 'BAD'}",
        flush=True,
    )

    n_ok = comm.allreduce(1 if ok else 0, op=MPI.SUM)
    if r == 0:
        print(
            f"\n=== probe summary: {n_ok}/{n} ranks OK "
            f"(MPI4JAX_USE_CUDA_MPI={cuda_mpi}, "
            f"cray-mpich={os.environ.get('CRAY_MPICH_VERSION', '?')}) ===",
            flush=True,
        )
        if cuda_mpi != "1":
            print(
                "WARNING: MPI4JAX_USE_CUDA_MPI != 1 -> NOT a GPU-direct test; a "
                "host-staged 'pass' proves nothing about the CXI inject path.",
                flush=True,
            )
        print("PASS" if n_ok == n else "FAIL", flush=True)
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
