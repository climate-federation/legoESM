"""Multi-node MPI initialization and state management.

Supports two decomposition modes:

1. **Face-only** (1–6 MPI ranks): each rank owns full faces.
2. **Sub-face tiling** (N ranks where N is a multiple of 6, N > 6):
   each face is split into tiles, each rank owns one tile.

Usage
-----
::

    from legoesm.parallel.distributed import initialize_distributed

    config = initialize_distributed()
    # All operators now automatically use MPI halo exchange.

Design
------
Face-only mode keeps all arrays at ``(6, n, n)`` shape, with zeros
for non-local faces.  Sub-face tiling mode keeps arrays at
``(6, n_tile, n_tile)`` shape with only the local face populated.
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

    Automatically detects the number of MPI ranks and selects face-only
    or sub-face tiling decomposition:

    - 1, 2, 3, or 6 ranks → face-only.
    - Multiple of 6 and >6 → sub-face tiling.

    Parameters
    ----------
    return_topology : bool
        If ``True``, also return the resolved :class:`CommTopology`.

    Returns
    -------
    DeviceConfig or (DeviceConfig, CommTopology)
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

    # Use MPI as the authoritative source for rank/size.
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

    # Each MPI rank uses its local devices.
    # For face-only: pick largest divisor of 6 ≤ local_device_count.
    # For sub-face: each rank owns one tile, so use 1 device per rank
    # (multiple local GPUs within a rank can be used for intra-rank parallelism).
    if n_processes <= 6:
        mesh_devices = next(
            (d for d in (6, 3, 2, 1) if d <= local_device_count),
            1,
        )
    else:
        # Sub-face tiling: each rank owns one tile, use all local devices
        mesh_devices = local_device_count

    if mesh_devices != local_device_count and n_processes <= 6:
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
    tiling = topology.tiling
    config = DeviceConfig(
        mesh=local_config.mesh,
        face_sharding=local_config.face_sharding,
        replicated_sharding=local_config.replicated_sharding,
        n_devices=local_config.n_devices,
        backend=local_config.backend,
        is_distributed=True,
        tiling=tiling,
        grid_type="cubed_sphere",
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
        Communication topology for this rank.

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
        mask_1d = jnp.array(
            [1.0 if f in local_faces else 0.0 for f in range(6)],
            dtype=leaf.dtype,
        )
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
        Communication topology.

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
            return leaf
        result = _mpi4jax_array_result(
            mpi4jax.allreduce(leaf, op=MPI.SUM, comm=MPI.COMM_WORLD),
        )
        return result

    return jax.tree.map(_allreduce_leaf, local_state)
