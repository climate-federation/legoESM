"""Distributed reduction operations for multi-node MPI.

Provides MPI-aware global sums and maxima that combine local
partial results from each MPI rank.

These functions are only called when the halo backend is set to
``"mpi"``; see :func:`legoesm.grids.halo.set_halo_backend`.

``mpi4jax`` is an optional dependency imported lazily.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def global_sum_mpi(local_value: jax.Array) -> jax.Array:
    """Compute a global sum across all MPI ranks.

    Parameters
    ----------
    local_value : jax.Array
        Scalar (or array) local partial sum.

    Returns
    -------
    jax.Array
        The global sum across all processes.
    """
    import mpi4jax
    from mpi4py import MPI

    global_val, _ = mpi4jax.allreduce(
        local_value, op=MPI.SUM, comm=MPI.COMM_WORLD
    )
    return global_val


def global_max_mpi(local_value: jax.Array) -> jax.Array:
    """Compute a global maximum across all MPI ranks.

    Parameters
    ----------
    local_value : jax.Array
        Scalar (or array) local partial maximum.

    Returns
    -------
    jax.Array
        The global maximum across all processes.
    """
    import mpi4jax
    from mpi4py import MPI

    global_val, _ = mpi4jax.allreduce(
        local_value, op=MPI.MAX, comm=MPI.COMM_WORLD
    )
    return global_val
