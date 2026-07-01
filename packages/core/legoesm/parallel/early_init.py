"""Early JAX distributed initialization.

Must be called before any legoESM or jax.numpy import that triggers XLA
backend discovery.  Importing this module is cheap: it only touches stdlib
and optionally mpi4py + jax (base package), never jax.numpy or legoESM.

Usage (at the very top of an entry-point script, before all other imports)::

    from legoesm.parallel.early_init import maybe_init_jax_distributed
    maybe_init_jax_distributed()

    # Safe to import legoESM / jax.numpy from here onward.
"""

from __future__ import annotations

import os
import socket

# Idempotency flag — must NOT be a jax.process_count() probe: that call
# initialises the XLA backend, after which jax.distributed.initialize() raises
# "must be called before any JAX calls that might initialise the XLA backend"
# (issue #693: every multi-node run died here).
_INITIALIZED = False


def maybe_init_jax_distributed(coordinator_port: int = 1234) -> bool:
    """Initialize ``jax.distributed`` if running under multi-node MPI.

    Detects the MPI world size from the environment (SLURM_NTASKS /
    PMI_SIZE / OMPI_COMM_WORLD_SIZE).  When >1 rank and the ranks span
    more than one hostname, calls ``jax.distributed.initialize()`` with
    rank-0's hostname as coordinator.  On single-node MPI or serial runs,
    does nothing.

    Returns True if ``jax.distributed.initialize()`` was called, False
    otherwise.  Safe to call multiple times — idempotency is tracked via a
    module-level flag (NOT a ``jax.process_count()`` probe, which would
    initialise the XLA backend and then make ``initialize()`` raise; #693).
    """
    global _INITIALIZED
    if _INITIALIZED:
        return False

    ntasks = int(
        os.environ.get(
            "SLURM_NTASKS",
            os.environ.get("PMI_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")),
        )
    )
    if ntasks <= 1:
        return False

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    hosts = comm.allgather(socket.gethostname())
    if len(set(hosts)) <= 1:
        # Single-node MPI: JAX distributed not needed.
        return False

    import jax
    coordinator = f"{hosts[0]}:{coordinator_port}"
    # local_device_ids=[0]: the repo's multi-node launch standard pins 1 GPU per
    # task via SLURM `--gpu-bind=single:1`, so every process sees exactly one GPU
    # as local index 0. Without this, jax auto-assigns device index by process
    # rank (0,1,2,...) and the >1-task-per-node ranks fail with "no supported
    # devices found for platform CUDA" (issue #693, confirmed C48 6-GPU/2-node).
    # ponytail: assumes 1 GPU/task; a launch that exposes all GPUs to each task
    # would instead need local_device_ids=[SLURM_LOCALID].
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=size,
        process_id=rank,
        local_device_ids=[0],
    )
    _INITIALIZED = True
    return True
