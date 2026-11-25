"""Stagger a memory-heavy first call across the ranks of a node.

The first call of a large jitted function (XLA compile + first run) needs far
more host memory than its steady state, and glibc keeps most of the freed
compile heap instead of returning it to the OS. When every rank on a node does
that at once, the node runs out of memory although each rank's steady state
fits. :func:`staggered_first_call` admits ``ranks_per_round`` ranks of each
node at a time and has each rank return its freed heap
(:func:`release_freed_heap`) before the next round starts.
"""

from __future__ import annotations

import ctypes
import warnings

import jax

_warned_no_trim = False


def release_freed_heap() -> bool:
    """Return freed heap pages to the OS with glibc ``malloc_trim(0)``.

    Returns ``False`` (with a one-time warning) where the C library has no
    ``malloc_trim`` (macOS, musl); the caller's memory is then unchanged.
    """
    global _warned_no_trim
    trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
    if trim is None:
        if not _warned_no_trim:
            warnings.warn("malloc_trim is not available in this C library; "
                          "freed heap is not returned to the OS.",
                          RuntimeWarning, stacklevel=2)
            _warned_no_trim = True
        return False
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    trim(0)
    return True


def staggered_first_call(fn, *args, comm=None, ranks_per_round=None):
    """Call ``fn(*args)`` once, ``ranks_per_round`` ranks per node at a time.

    ``fn`` must be rank-local (no communication inside): ranks of later rounds
    are waiting in a node barrier while it runs. Each rank blocks on the result
    and calls :func:`release_freed_heap` before entering the barrier. With
    ``comm=None`` (serial driver) it is ``fn(*args)`` followed by the trim.
    """
    if comm is None:
        out = jax.block_until_ready(fn(*args))
        release_freed_heap()
        return out
    if ranks_per_round is None or ranks_per_round < 1:
        raise ValueError(
            f"ranks_per_round must be >= 1 with a communicator, got "
            f"{ranks_per_round!r}.")
    from mpi4py import MPI
    node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    my_round = node.Get_rank() // ranks_per_round
    n_rounds = -(-node.Get_size() // ranks_per_round)
    out, err = None, None
    # A failing rank still attends every barrier so its peers are not left
    # waiting; the error is raised once the rounds are over.
    for r in range(n_rounds):
        if r == my_round:
            try:
                out = jax.block_until_ready(fn(*args))
                release_freed_heap()
            except BaseException as e:  # noqa: BLE001 - re-raised below
                err = e
        node.Barrier()
    node.Free()
    if err is not None:
        raise err
    return out
