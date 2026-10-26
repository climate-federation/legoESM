#!/usr/bin/env python
"""Is the per-rank core share actually reaching XLA's thread pool?

A sweep over ``--cpus-per-task`` is only a sweep over THREADS if the pool is
sized from the affinity mask.  If the pool is sized from the node's core
count instead, every arm runs the same number of threads and the sweep
varies pinning alone -- which still changes timings, so the result would
look like information and be an artifact.

Run under the same srun flags as the arm it validates.  Prints one JSON line
per rank: the affinity size, what Python thinks the machine has, the thread
environment, and a census of this process's OS threads taken AFTER real work
has forced the pool to spin up.  If the worker count tracks the share, the
knob is live.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter


def thread_names() -> Counter:
    names = Counter()
    for p in glob.glob("/proc/self/task/*/comm"):
        try:
            names[open(p).read().strip().rstrip("0123456789-") or "?"] += 1
        except OSError:
            pass
    return names


def main() -> int:
    import jax
    import jax.numpy as jnp

    # Enough work that a multi-threaded pool must actually start its workers;
    # a trivial op can complete on the calling thread and census as 1.
    x = jnp.ones((2048, 2048), jnp.float32)
    for _ in range(3):
        x = (x @ x.T) / 2048.0
    x.block_until_ready()

    names = thread_names()
    rec = {
        "rank": int(os.environ.get("SLURM_PROCID", "0")),
        "affinity": len(os.sched_getaffinity(0)),
        "cpu_count": os.cpu_count(),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "mkl": os.environ.get("MKL_NUM_THREADS"),
        "n_os_threads": sum(names.values()),
        "threads": dict(names.most_common(8)),
        "backend": jax.default_backend(),
    }
    print(json.dumps(rec), flush=True)
    if len(sys.argv) > 1:
        with open(f"{sys.argv[1]}.{rec['rank']}", "w") as f:
            json.dump(rec, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
