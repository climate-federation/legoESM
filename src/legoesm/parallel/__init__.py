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

7. **Voronoi mesh decomposition**:
   ``partition_voronoi_mesh()`` partitions unstructured MPAS/Voronoi
   meshes via geometric bisection or METIS, with halo exchange via
   ``VoronoiHaloExchange``.

8. **Ensemble parallelism**:
   ``make_ensemble_step()`` vmaps the model step over an ensemble
   dimension; ``ensemble_integrate()`` combines scan + vmap for
   efficient time-integration of many members.  Multi-device
   sharding via ``shard_ensemble()``.
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

from legoesm.parallel.voronoi_partition import (
    HaloCommSchedule,
    VoronoiPartition,
    partition_cells_geometric,
    partition_voronoi_mesh,
    build_local_mesh,
    scatter_to_local,
)

from legoesm.parallel.halo_exchange_voronoi import (
    VoronoiHaloExchange,
    exchange_local_simulated,
)

from legoesm.parallel.ensemble import (
    stack_states,
    unstack_states,
    perturb_initial_conditions,
    perturb_parameters,
    make_ensemble_step,
    make_ensemble_step_jit,
    ensemble_integrate,
    ensemble_integrate_with_forcing,
    ensemble_mean,
    ensemble_std,
    ensemble_percentile,
    ensemble_spread,
    create_ensemble_mesh,
    shard_ensemble,
    gather_ensemble,
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
    # Voronoi mesh decomposition
    "HaloCommSchedule",
    "VoronoiPartition",
    "partition_cells_geometric",
    "partition_voronoi_mesh",
    "build_local_mesh",
    "scatter_to_local",
    "VoronoiHaloExchange",
    "exchange_local_simulated",
    # Ensemble parallelism
    "stack_states",
    "unstack_states",
    "perturb_initial_conditions",
    "perturb_parameters",
    "make_ensemble_step",
    "make_ensemble_step_jit",
    "ensemble_integrate",
    "ensemble_integrate_with_forcing",
    "ensemble_mean",
    "ensemble_std",
    "ensemble_percentile",
    "ensemble_spread",
    "create_ensemble_mesh",
    "shard_ensemble",
    "gather_ensemble",
]
