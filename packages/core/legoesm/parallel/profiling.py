"""Lightweight MPI communication profiling.

Enabled by setting the environment variable ``LEGOESM_PROFILE_MPI=1``.
When disabled (the default), all instrumentation is a no-op with zero
overhead — the ``mpi_timer`` context manager returns immediately.

Usage
-----
::

    from legoesm.parallel.profiling import mpi_timer, print_mpi_profile

    with mpi_timer("pad_halo_mpi_4d"):
        ...  # MPI communication

    # At the end of the run:
    print_mpi_profile(rank=0)
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_ENABLED = os.environ.get("LEGOESM_PROFILE_MPI", "").strip().lower() in {
    "1", "true", "yes", "on",
}

# Accumulates (call_count, total_ns) per operation name.
_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@contextmanager
def mpi_timer(name: str):
    """Context manager that times a block and records it under *name*.

    No-op when profiling is disabled.
    """
    if not _ENABLED:
        yield
        return
    t0 = time.perf_counter_ns()
    yield
    elapsed = time.perf_counter_ns() - t0
    entry = _stats[name]
    entry[0] += 1
    entry[1] += elapsed


def get_stats() -> dict[str, dict[str, int | float]]:
    """Return a copy of the profiling stats.

    Returns
    -------
    dict mapping operation name to:
        - ``calls``: number of invocations
        - ``total_ms``: cumulative time in milliseconds
        - ``avg_us``: average time per call in microseconds
    """
    result = {}
    for name, (count, total_ns) in sorted(_stats.items()):
        total_ms = total_ns / 1_000_000
        avg_us = (total_ns / count / 1_000) if count > 0 else 0.0
        result[name] = {
            "calls": count,
            "total_ms": round(total_ms, 2),
            "avg_us": round(avg_us, 1),
        }
    return result


def reset_stats() -> None:
    """Clear all accumulated profiling data."""
    _stats.clear()


def print_mpi_profile(rank: int = 0) -> None:
    """Print a summary table of MPI communication costs.

    Only prints on the specified rank (default 0).  Call from all
    ranks — non-matching ranks return silently.

    Parameters
    ----------
    rank : int
        Only this MPI rank will print.  Pass -1 to print on all ranks.
    """
    if not _ENABLED:
        return

    try:
        from mpi4py import MPI
        my_rank = MPI.COMM_WORLD.Get_rank()
    except (ImportError, Exception):
        my_rank = 0

    if rank >= 0 and my_rank != rank:
        return

    stats = get_stats()
    if not stats:
        print(f"[Rank {my_rank}] No MPI profiling data collected.")
        return

    # Header
    print(f"\n{'='*72}")
    print(f"  MPI Communication Profile (Rank {my_rank})")
    print(f"{'='*72}")
    print(f"  {'Operation':<35s} {'Calls':>8s} {'Total(ms)':>10s} {'Avg(us)':>10s}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*10}")

    total_ms = 0.0
    for name, info in stats.items():
        print(
            f"  {name:<35s} {info['calls']:>8d} "
            f"{info['total_ms']:>10.2f} {info['avg_us']:>10.1f}"
        )
        total_ms += info["total_ms"]

    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*10}")
    print(f"  {'TOTAL':<35s} {'':>8s} {total_ms:>10.2f}")
    print(f"{'='*72}\n")
