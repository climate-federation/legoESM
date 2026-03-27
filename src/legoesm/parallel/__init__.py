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

9. **Sharded dynamics** (shard_map-based):
   ``make_sharded_step()`` wraps a dynamics model for SPMD execution
   across devices with explicit sharding constraints and optional
   halo exchange.  ``shard_state()`` / ``gather_state()`` move data
   between single-device and multi-device layouts.

10. **Async halo exchange (compute/communication overlap)**:
   ``overlapped_halo_compute()`` splits stencil evaluation into
   interior (halo-independent) and boundary (halo-dependent) parts,
   enabling the XLA compiler to overlap interior computation with
   halo communication scheduling.

11. **Device configuration and hardware-aware optimization**:
    ``detect_devices()`` auto-detects hardware (CPU/GPU/TPU/Metal),
    ``configure_jax_for_device()`` applies backend-specific XLA flags,
    and ``mixed_precision_policy()`` returns optimal dtype policies.
"""

from legoesm.parallel.runtime import (
    ParallelRuntime,
    HaloBackend,
    ReductionBackend,
    validate_device_count,
)

from legoesm.parallel.mesh import (
    DeviceConfig,
    create_device_mesh,
    create_latlon_mesh,
    create_level_mesh,
    create_voronoi_device_mesh,
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
    reorder_voronoi_for_sharding,
)

from legoesm.parallel.halo_exchange_voronoi import (
    VoronoiHaloExchange,
    exchange_local_simulated,
)

from legoesm.parallel.async_halo import (
    InteriorBoundaryMasks,
    OverlapContext,
    create_interior_boundary_masks,
    split_interior_boundary,
    merge_interior_boundary,
    boundary_slices,
    interior_slice,
    extract_interior_padded,
    overlapped_halo_compute,
    overlapped_halo_compute_vector,
    create_overlap_context,
    overlapped_compute_with_context,
    start_halo_exchange,
    finish_halo_exchange,
    async_halo_step,
    async_halo_step_multi,
    async_halo_step_vector,
)

from legoesm.parallel.device_config import (
    HardwareConfig,
    detect_devices as detect_hardware,
    configure_jax_for_device,
    get_optimal_dtype,
    get_optimal_mesh,
    MixedPrecisionPolicy,
    mixed_precision_policy,
    cast_for_device,
)

from legoesm.parallel.sharded_dynamics import (
    StepCacheKey,
    CompiledShardedStep,
    make_sharded_step,
    make_voronoi_sharded_step,
    shard_state as shard_state_to_devices,
    gather_state as gather_state_from_devices,
    create_output_shardings,
    sharded_step_with_halo,
    make_face_halo_exchange,
    sharded_integrate,
    sharded_integrate_scan,
    check_sharding,
)

from legoesm.parallel.reductions import (
    batch_allreduce_mpi,
)

from legoesm.parallel.layout import (
    DistributedLayout,
    SingleRankLayout,
    FaceOwnership,
    make_layout,
    scatter as layout_scatter,
    scatter_pytree as layout_scatter_pytree,
    gather as layout_gather,
    gather_pytree as layout_gather_pytree,
    local_sum as layout_local_sum,
    local_max as layout_local_max,
    local_min as layout_local_min,
    global_reduce as layout_global_reduce,
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
    # Canonical runtime (preferred entry point)
    "ParallelRuntime",
    "HaloBackend",
    "ReductionBackend",
    "validate_device_count",
    # Device mesh
    "DeviceConfig",
    "create_device_mesh",
    "create_latlon_mesh",
    "create_level_mesh",
    "create_voronoi_device_mesh",
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
    "reorder_voronoi_for_sharding",
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
    # Async halo exchange (compute/communication overlap)
    "InteriorBoundaryMasks",
    "OverlapContext",
    "create_interior_boundary_masks",
    "split_interior_boundary",
    "merge_interior_boundary",
    "boundary_slices",
    "interior_slice",
    "extract_interior_padded",
    "overlapped_halo_compute",
    "overlapped_halo_compute_vector",
    "create_overlap_context",
    "overlapped_compute_with_context",
    "start_halo_exchange",
    "finish_halo_exchange",
    "async_halo_step",
    "async_halo_step_multi",
    "async_halo_step_vector",
    # Sharded dynamics (shard_map-based SPMD)
    "StepCacheKey",
    "CompiledShardedStep",
    "make_sharded_step",
    "shard_state_to_devices",
    "gather_state_from_devices",
    "create_output_shardings",
    "sharded_step_with_halo",
    "make_face_halo_exchange",
    "sharded_integrate",
    "sharded_integrate_scan",
    "check_sharding",
    # Batch MPI reductions
    "batch_allreduce_mpi",
    # Device configuration and hardware-aware optimization
    "HardwareConfig",
    "detect_hardware",
    "configure_jax_for_device",
    "get_optimal_dtype",
    "get_optimal_mesh",
    "MixedPrecisionPolicy",
    "mixed_precision_policy",
    "cast_for_device",
]
