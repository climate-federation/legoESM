"""Multi-node MPI initialization and state management.

Provides utilities for initializing the JAX distributed runtime,
partitioning state across MPI processes, and gathering results.

Usage
-----
::

    from legoesm.parallel.distributed import initialize_distributed

    config = initialize_distributed()
    # All operators now automatically use MPI halo exchange.
    # State arrays keep shape (6, n, n) with zeros for non-local faces.

Design
------
We keep all arrays at their full ``(6, n, n)`` shape, with zeros for
faces not owned by the current rank.  This is simpler than sub-setting
and lets ``jnp.sum(field * area)`` naturally produce the local
contribution without shape changes.  A final ``allreduce(SUM)`` gives
the global result.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.parallel.comm import CommTopology, build_comm_topology
from legoesm.parallel.mesh import DeviceConfig, create_device_mesh


def initialize_distributed(
    backend: str = "mpi",
) -> DeviceConfig:
    """Initialize JAX distributed runtime and set up MPI halo exchange.

    Must be called before any JAX computation.

    Parameters
    ----------
    backend : str
        Distributed backend (default ``"mpi"``).  Passed to
        ``jax.distributed.initialize()``.

    Returns
    -------
    DeviceConfig
        Device configuration with ``is_distributed=True``.
    """
    # Initialize JAX distributed runtime.
    jax.distributed.initialize()

    rank = jax.process_index()
    n_processes = jax.process_count()

    # Build communication topology for this rank.
    topology = build_comm_topology(rank, n_processes)

    # Set up MPI halo exchange.
    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("mpi", topology)

    # Create device mesh with local devices.
    local_config = create_device_mesh(
        n_devices=len(jax.local_devices()),
    )

    # Return a new config marking distributed mode.
    return DeviceConfig(
        mesh=local_config.mesh,
        face_sharding=local_config.face_sharding,
        replicated_sharding=local_config.replicated_sharding,
        n_devices=local_config.n_devices,
        backend=local_config.backend,
        is_distributed=True,
    )


def partition_state(state, topology: CommTopology):
    """Zero out non-local faces in a state pytree.

    Keeps the full ``(6, n, n)`` shape but sets non-local face data to
    zero.  This ensures that ``jnp.sum(field * area)`` gives the local
    contribution, and ``allreduce(SUM)`` gives the global total.

    Parameters
    ----------
    state
        Any JAX pytree (e.g., ``ShallowWaterState``).
    topology : CommTopology
        Communication topology for this rank.

    Returns
    -------
    Masked pytree with same structure.
    """
    local_faces = set(topology.local_face_ids)

    def _mask_leaf(leaf):
        if not isinstance(leaf, jnp.ndarray):
            return leaf
        if leaf.ndim < 1 or leaf.shape[0] != 6:
            return leaf
        # Create a mask that is 1 for local faces, 0 for remote.
        mask_1d = jnp.array(
            [1.0 if f in local_faces else 0.0 for f in range(6)],
            dtype=leaf.dtype,
        )
        # Broadcast mask to match array shape.
        shape = (6,) + (1,) * (leaf.ndim - 1)
        mask = mask_1d.reshape(shape)
        return leaf * mask

    return jax.tree.map(_mask_leaf, state)


def gather_state(local_state, topology: CommTopology):
    """Gather a partitioned state from all MPI ranks.

    Since non-local faces are zero, ``allreduce(SUM)`` produces the
    complete global state.

    Parameters
    ----------
    local_state
        Partitioned JAX pytree (non-local faces are zero).
    topology : CommTopology
        Communication topology.

    Returns
    -------
    Global state pytree with all faces populated.
    """
    import mpi4jax
    from mpi4py import MPI

    def _allreduce_leaf(leaf):
        if not isinstance(leaf, jnp.ndarray):
            return leaf
        result, _ = mpi4jax.allreduce(
            leaf, op=MPI.SUM, comm=MPI.COMM_WORLD
        )
        return result

    return jax.tree.map(_allreduce_leaf, local_state)
