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


def maybe_init_jax_distributed(coordinator_port: int = 1234) -> bool:
    """Initialize ``jax.distributed`` if running under multi-node MPI.

    Detects the MPI world size from the environment (SLURM_NTASKS /
    PMI_SIZE / OMPI_COMM_WORLD_SIZE).  When >1 rank and the ranks span
    more than one hostname, calls ``jax.distributed.initialize()`` with
    rank-0's hostname as coordinator.  On single-node MPI or serial runs,
    does nothing.

    Returns True if ``jax.distributed.initialize()`` was called, False
    otherwise.  Safe to call multiple times — skips if JAX already reports
    more than one process (i.e. a previous call succeeded).
    """
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
    if jax.process_count() > 1:
        # Already initialized by a previous call.
        return False

    coordinator = f"{hosts[0]}:{coordinator_port}"
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=size,
        process_id=rank,
    )
    return True
