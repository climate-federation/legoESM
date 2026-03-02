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
from legoesm.parallel.mesh import (
    DeviceConfig,
    create_device_mesh,
    set_active_config,
)
from legoesm.parallel.reductions import _require_mpi_stack, _mpi4jax_array_result

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
    global _active_topology

    if _active_topology is not None:
        import warnings
        warnings.warn(
            "initialize_distributed() called more than once. "
            "Returning existing configuration.",
            RuntimeWarning,
            stacklevel=2,
        )
        from legoesm.parallel.mesh import get_active_config
        config = get_active_config()
        if return_topology:
            return config, _active_topology
        return config

    # Validate MPI dependencies before touching JAX distributed runtime.
    _mpi4jax, MPI = _require_mpi_stack()

    # Initialize JAX distributed runtime.
    jax.distributed.initialize()

    # Use MPI as the authoritative source for rank/size,
    # since halo_exchange.py uses MPI.COMM_WORLD for all communication.
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

    # Build communication topology for this rank.
    topology = build_comm_topology(rank, n_processes)
    _active_topology = topology

    # Set up MPI halo exchange.
    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("mpi", topology)

    # Create device mesh with local devices.
    local_devices = list(jax.local_devices())
    local_device_count = len(local_devices)
    if local_device_count < 1:
        raise RuntimeError("No local JAX devices found for this MPI rank.")
    # Face-only sharding requires a divisor of 6. When local device count
    # is not a divisor (e.g., 4), pick the largest supported local subset.
    mesh_devices = next(
        (d for d in (6, 3, 2, 1) if d <= local_device_count),
        1,
    )
    if mesh_devices != local_device_count:
        import warnings
        warnings.warn(
            "Local device count does not evenly divide 6 cubed-sphere faces. "
            f"Using {mesh_devices} of {local_device_count} local devices.",
            RuntimeWarning,
            stacklevel=2,
        )
    local_config = create_device_mesh(
        n_devices=mesh_devices,
        devices=local_devices,
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
    set_active_config(config)
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

    mpi4jax, MPI = _require_mpi_stack()

    def _allreduce_leaf(leaf):
        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
            return leaf
        if leaf.ndim < 1 or leaf.shape[0] != 6:
            # Non-face-leading leaves are replicated metadata/constants.
            return leaf
        result = _mpi4jax_array_result(
            mpi4jax.allreduce(leaf, op=MPI.SUM, comm=MPI.COMM_WORLD),
        )
        return result

    return jax.tree.map(_allreduce_leaf, local_state)
