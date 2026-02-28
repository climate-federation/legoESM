"""Parallelism utilities for legoESM.

Provides device mesh setup, array sharding, and distributed
communication for multi-GPU, multi-CPU, and multi-node execution.

Single-node multi-device
    Use :func:`create_device_mesh` and :func:`shard_pytree` to shard
    the cubed-sphere state across GPUs.  JAX's XLA compiler handles
    inter-device communication transparently.

Multi-node MPI
    See :mod:`legoesm.parallel.distributed` for ``mpi4jax``-based
    distributed halo exchange and global reductions.

Mac Metal
    See :mod:`legoesm.parallel.metal` for hybrid Metal GPU / CPU
    routing (finite-volume on Metal, spectral on CPU).
"""

from legoesm.parallel.mesh import (
    DeviceConfig,
    create_device_mesh,
    get_active_config,
    replicate_pytree,
    shard_pytree,
)

__all__ = [
    "DeviceConfig",
    "create_device_mesh",
    "get_active_config",
    "replicate_pytree",
    "shard_pytree",
]
