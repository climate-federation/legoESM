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

_active_topology: CommTopology | None = None


def initialize_distributed(
    *,
    return_topology: bool = False,
) -> DeviceConfig | tuple[DeviceConfig, CommTopology]:
    """Initialize JAX distributed runtime and set up MPI halo exchange.

    Must be called before any JAX computation.  Uses MPI for inter-process
    communication and halo exchange.

    Parameters
    ----------
    return_topology : bool
        If ``True``, also return the resolved :class:`CommTopology`.

    Returns
    -------
    DeviceConfig or (DeviceConfig, CommTopology)
        Device configuration with ``is_distributed=True``. If
        ``return_topology=True``, returns ``(config, topology)``.
    """
    # Initialize JAX distributed runtime.
    jax.distributed.initialize()

    # Use MPI as the authoritative source for rank/size,
    # since halo_exchange.py uses MPI.COMM_WORLD for all communication.
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_processes = comm.Get_size()

    # Validate JAX agrees with MPI.
    jax_rank = jax.process_index()
    jax_size = jax.process_count()
    if jax_rank != rank or jax_size != n_processes:
        import warnings
        warnings.warn(
            f"JAX process_index/count ({jax_rank}/{jax_size}) differs from "
            f"MPI rank/size ({rank}/{n_processes}). Using MPI values.",
            RuntimeWarning,
            stacklevel=2,
        )

    global _active_topology

    # Build communication topology for this rank.
    topology = build_comm_topology(rank, n_processes)
    _active_topology = topology

    # Set up MPI halo exchange.
    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("mpi", topology)

    # Create device mesh with local devices.
    local_config = create_device_mesh(
        n_devices=len(jax.local_devices()),
    )

    # Return a new config marking distributed mode.
    config = DeviceConfig(
        mesh=local_config.mesh,
        face_sharding=local_config.face_sharding,
        replicated_sharding=local_config.replicated_sharding,
        n_devices=local_config.n_devices,
        backend=local_config.backend,
        is_distributed=True,
    )
    if return_topology:
        return config, topology
    return config


def get_active_topology() -> CommTopology | None:
    """Return the active MPI communication topology, if initialized."""
    return _active_topology


def partition_state(state, topology: CommTopology | None = None):
    """Zero out non-local faces in a state pytree.

    Keeps the full ``(6, n, n)`` shape but sets non-local face data to
    zero.  This ensures that ``jnp.sum(field * area)`` gives the local
    contribution, and ``allreduce(SUM)`` gives the global total.

    Parameters
    ----------
    state
        Any JAX pytree (e.g., ``ShallowWaterState``).
    topology : CommTopology, optional
        Communication topology for this rank. If omitted, uses the
        active topology set by :func:`initialize_distributed`.

    Returns
    -------
    Masked pytree with same structure.
    """
    topology = topology or _active_topology
    if topology is None:
        raise ValueError(
            "No CommTopology provided and no active topology is set. "
            "Call initialize_distributed() first or pass topology explicitly."
        )

    local_faces = set(topology.local_face_ids)

    def _mask_leaf(leaf):
        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
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


def gather_state(local_state, topology: CommTopology | None = None):
    """Gather a partitioned state from all MPI ranks.

    Since non-local faces are zero, ``allreduce(SUM)`` produces the
    complete global state.

    Parameters
    ----------
    local_state
        Partitioned JAX pytree (non-local faces are zero).
    topology : CommTopology, optional
        Communication topology. If omitted, uses the active topology
        set by :func:`initialize_distributed`.

    Returns
    -------
    Global state pytree with all faces populated.
    """
    topology = topology or _active_topology
    if topology is None:
        raise ValueError(
            "No CommTopology provided and no active topology is set. "
            "Call initialize_distributed() first or pass topology explicitly."
        )

    import mpi4jax
    from mpi4py import MPI

    def _allreduce_leaf(leaf):
        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
            return leaf
        result, _ = mpi4jax.allreduce(
            leaf, op=MPI.SUM, comm=MPI.COMM_WORLD
        )
        return result

    return jax.tree.map(_allreduce_leaf, local_state)
