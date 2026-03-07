"""Parallelism utilities for legoESM.

Provides device mesh setup, array sharding, and distributed
communication for multi-GPU, multi-CPU, and multi-node execution.

Parallelism strategies
----------------------
1. **Cubed-sphere face sharding** (1–6 devices):
   ``create_device_mesh()`` shards the 6-face dimension.

2. **Sub-face tiling** (>6 devices, multiples of 6):
   ``create_device_mesh()`` splits each face into tiles,
   enabling up to 6 × tx × ty devices.

3. **Lat-lon domain decomposition**:
   ``create_latlon_mesh()`` shards by latitude.

4. **Level-parallel spectral**:
   ``create_level_mesh()`` distributes SH transforms across devices.

5. **Multi-node MPI**:
   ``initialize_distributed()`` for mpi4jax-based halo exchange
   and reductions, with face-only or sub-face tiling decomposition.

6. **Metal (Apple Silicon)**:
   ``get_metal_config()`` for hybrid Metal/CPU routing.
"""

from legoesm.parallel.mesh import (
    DeviceConfig,
    create_device_mesh,
    create_latlon_mesh,
    create_level_mesh,
    get_active_config,
    replicate_pytree,
    shard_pytree,
    shard_latlon,
    shard_levels,
)

__all__ = [
    "DeviceConfig",
    "create_device_mesh",
    "create_latlon_mesh",
    "create_level_mesh",
    "get_active_config",
    "replicate_pytree",
    "shard_pytree",
    "shard_latlon",
    "shard_levels",
]
